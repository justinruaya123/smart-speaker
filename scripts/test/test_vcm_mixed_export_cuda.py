"""Binding/array/device fixtures only: no ONNX export or neural forward."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from scripts.test.test_vcm_mixed_finalize import (
    ROOT, calibration_fixture, dump, file_sha, mocked_gpu_pipeline, prepared,
)
from scripts.test.vcm_mixed_finalize import load_selected_cuda as selected_cuda_loader


UUID = "GPU-12345678-1234-5678-abcd-123456789abc"


def exporter():
    name = "scripts.test.vcm_mixed_export_cuda"
    assert importlib.util.find_spec(name) is not None, "Missing CUDA-only export helper"
    return importlib.import_module(name)


@pytest.fixture
def postfit(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    directory = prepared["tmp_path"] / "postfit"
    receipt = module.run(prepared["training"], prepared["inputs_path"], directory)
    return {**prepared, "postfit": directory, "receipt": receipt}


def rewrite_policy(fixture, change):
    from scripts.runtime.vcm_intent_slots_contract import digest
    directory = fixture["postfit"]
    path = directory / "calibration/calibration.json"
    policy = json.loads(path.read_text())
    change(policy)
    policy["calibration_id"] = digest({key: value for key, value in policy.items() if key != "calibration_id"})
    dump(path, policy)
    fixture["receipt"]["files"]["calibration"]["sha256"] = file_sha(path)
    dump(directory / "postfit-receipt.json", fixture["receipt"])


def test_postfit_verification_binds_the_completed_selected_candidate(postfit):
    context = exporter().verify_postfit(postfit["postfit"])
    assert context["training_context"]["selected"] == postfit["grid"][-1]
    assert context["normalization_sha256"] == file_sha(postfit["postfit"] / "normalization.json")
    assert context["checkpoint_sha256"] == postfit["grid"][-1]["checkpoint_sha256"]


@pytest.mark.parametrize("damage", ["checkpoint", "threshold", "manifest", "memory",
                                     "grid", "gpu", "coverage", "source", "normalization"])
def test_stale_or_partial_postfit_evidence_cannot_be_exported(postfit, damage):
    receipt = postfit["receipt"]
    if damage == "checkpoint":
        Path(postfit["grid"][-1]["checkpoint"]).write_bytes(b"changed selected weights")
    elif damage == "threshold":
        rewrite_policy(postfit, lambda policy: policy["thresholds"].update(intent=.94))
    elif damage == "manifest":
        rewrite_policy(postfit, lambda policy: policy["binding"].update(manifest="0" * 64))
    elif damage == "memory":
        rewrite_policy(postfit, lambda policy: policy["binding"].update(memory="0" * 64))
    elif damage == "normalization":
        path = postfit["postfit"] / "normalization.json"
        path.write_text(json.dumps(postfit["norm"]), encoding="utf-8")
        receipt["files"]["normalization"]["sha256"] = file_sha(path)
        dump(postfit["postfit"] / "postfit-receipt.json", receipt)
    else:
        if damage == "grid":
            receipt["completed_grid_candidates"] = 5
        elif damage == "gpu":
            receipt["gpu"]["device"] = "cpu"
        elif damage == "coverage":
            receipt["missing_calibration_intents"] = ["OOV"]
        else:
            receipt["source_bindings"]["vcm/frontend.py"]["sha256"] = "0" * 64
        dump(postfit["postfit"] / "postfit-receipt.json", receipt)
    with pytest.raises(ValueError):
        exporter().verify_postfit(postfit["postfit"])


@pytest.mark.parametrize("damage", ["partial_sources", "rebound_source", "missing_run", "rebound_run",
                                     "missing_history", "wrong_run_name", "partial_evidence"])
def test_complete_source_and_grid_receipts_cannot_be_omitted_or_rebound(postfit, damage):
    receipt = postfit["receipt"]
    if damage == "partial_sources":
        receipt["source_bindings"] = {"vcm/frontend.py": receipt["source_bindings"]["vcm/frontend.py"]}
    elif damage in ("rebound_source", "rebound_run"):
        item = (receipt["source_bindings"]["scripts/runtime/vcm_intent_slots_decode.py"]
                if damage == "rebound_source" else receipt["training_runs"]["uniform-53"]["identity"])
        copy = postfit["tmp_path"] / "same-bytes-different-source.py"
        copy.write_bytes(Path(item["path"]).read_bytes())
        item["path"] = str(copy)
    elif damage == "missing_run":
        receipt["training_runs"].pop("uniform-53")
    elif damage == "missing_history":
        receipt["training_runs"]["uniform-53"].pop("history")
    elif damage == "wrong_run_name":
        receipt["training_runs"]["other-run"] = receipt["training_runs"].pop("uniform-53")
    else:
        receipt["training_evidence_bindings"].pop(next(iter(receipt["training_evidence_bindings"])))
    dump(postfit["postfit"] / "postfit-receipt.json", receipt)
    with pytest.raises(ValueError):
        exporter().verify_postfit(postfit["postfit"])


@pytest.mark.parametrize("visible", [None, "", "0", "GPU-123", UUID + "," + UUID,
                                     UUID + ",0", " " + UUID])
def test_export_requires_one_explicit_gpu_uuid(visible):
    with pytest.raises(ValueError, match="UUID"):
        exporter().explicit_gpu_uuid(visible)


def test_export_accepts_one_explicit_gpu_uuid():
    assert exporter().explicit_gpu_uuid(UUID) == UUID


def heads(contract):
    sizes = [19, len(contract["roles"]) * 4, 4,
             max(value[1] for value in contract["value_offsets"].values()), 320]
    return [np.zeros((1, size), dtype=np.float32) for size in sizes]


@pytest.mark.parametrize("damage", ["missing", "extra", "swapped", "nonfinite", "wrong_shape", "drift"])
def test_five_head_order_shape_finiteness_and_parity_are_enforced(prepared, damage):
    reference, actual = heads(prepared["contract"]), heads(prepared["contract"])
    if damage == "missing":
        actual.pop()
    elif damage == "extra":
        actual.append(np.zeros((1, 1), dtype=np.float32))
    elif damage == "swapped":
        actual[0], actual[2] = actual[2], actual[0]
    elif damage == "nonfinite":
        actual[0][0, 0] = np.nan
    elif damage == "wrong_shape":
        actual[4] = actual[4].reshape(5, 64)
    else:
        actual[0][0, 0] = 3e-5
    with pytest.raises(ValueError):
        exporter().compare_heads(reference, actual, prepared["contract"])


def test_head_parity_keeps_each_named_error(prepared):
    reference, actual = heads(prepared["contract"]), heads(prepared["contract"])
    actual[2][0, 0] = 1e-6
    result = exporter().compare_heads(reference, actual, prepared["contract"])
    assert result["max_error"] == pytest.approx(1e-6)
    assert result["head_errors"]["validity"] == pytest.approx(1e-6)
    assert set(result["head_errors"]) == {"intent", "role_states", "validity", "values", "entities"}


def test_small_numeric_error_cannot_hide_a_final_threshold_decision_change(prepared):
    contract = prepared["contract"]
    reference = heads(contract)
    reference[0][0, contract["intent_order"].index("TIME")] = np.log(18 * .95 / .05) + 2e-6
    state = reference[1].reshape(-1, 4)
    state[:] = -20
    state[:, 0] = 20
    reference[2][0] = [20, -20, -20, -20]
    actual = deepcopy(reference)
    actual[0][0, contract["intent_order"].index("TIME")] -= 4e-6
    context = {"contract": contract, "memory": None,
               "catalog": {"entries": contract["catalog_entries"], "custom": contract["custom"]},
               "policy": {"require_calibration": False, "entities": {}}}
    with pytest.raises(ValueError, match="decision"):
        exporter().compare_probe(reference, actual, context)


@pytest.mark.parametrize("kind,index", [("cpu", None), ("cuda", 1)])
def test_device_guard_blocks_a_cpu_or_second_device_before_forward(kind, index):
    tensor = SimpleNamespace(device=SimpleNamespace(type=kind, index=index))
    with pytest.raises(ValueError, match="CUDA"):
        exporter().require_cuda_tensor(tensor)


def test_model_guard_checks_buffers_as_well_as_parameters():
    device = lambda kind: SimpleNamespace(device=SimpleNamespace(type=kind, index=0))
    model = SimpleNamespace(parameters=lambda: iter([device("cuda")]),
                            buffers=lambda: iter([device("cpu")]))
    with pytest.raises(ValueError, match="CUDA"):
        exporter().require_cuda_module(model)


def fake_ort(monkeypatch, contract, *, available=True, providers=None, wrong_heads=False,
             fallback_during_run=False, output_device="cuda", mutate_original=False):
    import torch
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    arrays = heads(contract)

    class Binding:
        def bind_input(self, **kwargs):
            assert kwargs == {"name": "features", "device_type": "cuda", "device_id": 0,
                              "element_type": np.float32, "shape": (1, 40, 17), "buffer_ptr": 123}

        def bind_output(self, name, device_type, device_id):
            assert name in ("intent", "role_states", "validity", "values", "entities")
            assert device_type == "cuda" and device_id == 0

        def get_outputs(self):
            return [SimpleNamespace(device_name=lambda: output_device) for _ in arrays]

        def copy_outputs_to_cpu(self):
            return deepcopy(arrays)

    class Session:
        def __init__(self):
            self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.fallback_disabled = False

        def disable_fallback(self):
            self.fallback_disabled = True

        def get_providers(self):
            return self.providers

        def get_provider_options(self):
            return {"CUDAExecutionProvider": {"device_id": "0", "use_tf32": "0"},
                    "CPUExecutionProvider": {}}

        def get_inputs(self):
            return [SimpleNamespace(name="features", type="tensor(float)", shape=[1, 40, "frames"])]

        def get_outputs(self):
            names = ["intent", "role_states", "validity", "values", "entities"]
            if wrong_heads:
                names[1], names[3] = names[3], names[1]
            return [SimpleNamespace(name=name, type="tensor(float)") for name in names]

        def io_binding(self):
            return Binding()

        def run_with_iobinding(self, binding):
            if not self.fallback_disabled:
                raise AssertionError("Whole-session CPU fallback remained enabled")
            if fallback_during_run:
                self.providers = ["CPUExecutionProvider"]

        def run(self, *args, **kwargs):
            raise AssertionError("CUDA features were replaced by a host forward")

    session = Session()

    def create(path, options, *, providers, provider_options):
        if providers != ["CUDAExecutionProvider", "CPUExecutionProvider"]:
            raise AssertionError("CUDA must be the requested first execution provider")
        if provider_options != [{"device_id": "0", "use_tf32": "0"}, {}]:
            raise AssertionError("Single-device FP32 parity requires explicit device 0 and TF32 off")
        assert options.enable_profiling is True
        assert options.graph_optimization_level == 0
        Path(options.optimized_model_filepath).write_bytes(Path(path).read_bytes())
        if mutate_original:
            Path(path).write_bytes(b"changed graph during mocked session construction")
        return session

    ort = SimpleNamespace(__version__="fixture-only", SessionOptions=SimpleNamespace,
                          GraphOptimizationLevel=SimpleNamespace(ORT_DISABLE_ALL=0),
                          preload_dlls=lambda: None,
                          get_available_providers=lambda: (["CUDAExecutionProvider", "CPUExecutionProvider"]
                                                          if available else ["CPUExecutionProvider"]),
                          InferenceSession=create)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    return session


def gpu_features():
    import torch
    return SimpleNamespace(device=SimpleNamespace(type="cuda", index=0), dtype=torch.float32,
                           shape=(1, 40, 17), is_contiguous=lambda: True, data_ptr=lambda: 123)


def fake_graph(prepared):
    path = prepared["tmp_path"] / "fixture.onnx"
    path.write_bytes(b"fixture input graph; never executed")
    return path


@pytest.mark.parametrize("damage", ["available", "actual", "priority", "heads"])
def test_ort_session_rejects_missing_or_misordered_cuda_provider_and_heads(prepared, monkeypatch, damage):
    fake_ort(monkeypatch, prepared["contract"], available=damage != "available",
             providers=(["CPUExecutionProvider"] if damage == "actual" else
                        ["CPUExecutionProvider", "CUDAExecutionProvider"] if damage == "priority" else None),
             wrong_heads=damage == "heads")
    with pytest.raises(ValueError):
        exporter().create_cuda_session(fake_graph(prepared))


def test_ort_io_binding_keeps_inputs_and_outputs_on_cuda(prepared, monkeypatch):
    fake_ort(monkeypatch, prepared["contract"])
    session, receipt = exporter().create_cuda_session(fake_graph(prepared))
    arrays = exporter().run_ort_cuda(session, gpu_features(), prepared["contract"])
    assert receipt["actual_providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert receipt["whole_session_fallback_disabled"] is True
    assert len(arrays) == 5 and all(np.count_nonzero(value) == 0 for value in arrays)


@pytest.mark.parametrize("damage", ["session_fallback", "cpu_output"])
def test_ort_cannot_finish_a_probe_with_cpu_fallback_or_cpu_output(prepared, monkeypatch, damage):
    fake_ort(monkeypatch, prepared["contract"], fallback_during_run=damage == "session_fallback",
             output_device="cpu" if damage == "cpu_output" else "cuda")
    session, _ = exporter().create_cuda_session(fake_graph(prepared))
    with pytest.raises(ValueError, match="CUDA"):
        exporter().run_ort_cuda(session, gpu_features(), prepared["contract"])


def mock_numerical_export(monkeypatch):
    module = exporter()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)

    def numerical(context, output):
        graph, profile, _ = placement_fixture(output)
        (output / "model.onnx").write_bytes(graph.read_bytes())
        placement = module.finish_cuda_profile(SimpleNamespace(end_profiling=lambda: str(profile)), output / "model.onnx",
            profile_info(graph, output / "model.onnx", provider_options={"CUDAExecutionProvider": {"use_tf32": "0"}}))
        arrays = heads(context["contract"])
        parity = module.compare_probe(arrays, arrays, context)
        return {"gpu": {"device": "cuda", "visible_device_count": 1,
                        "torch": "fixture-only-no-neural-execution", "CUDA_VISIBLE_DEVICES": UUID},
                "ort": {"actual_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                        "whole_session_fallback_disabled": True, "input_device": "cuda", "output_device": "cuda"},
                "parameters": {"total": 123}, "checkpoint_status": "mixed_silver_unqualified_candidate",
                "placement": placement,
                "rewrite_parity": [{"frames": frames, "max_error": 0.0,
                                    "head_errors": {name: 0.0 for name in ("intent", "role_states", "validity", "values", "entities")}}
                                   for frames in (1, 3, 17, 32, 101, 231)],
                "onnx_parity": [{"frames": frames, **deepcopy(parity)} for frames in (1, 3, 17, 32, 101, 231)]}

    monkeypatch.setattr(module, "export_and_parity", numerical)
    return module


def test_bundle_preserves_original_interfaces_and_exact_bound_bytes(postfit, monkeypatch):
    module = mock_numerical_export(monkeypatch)
    output = postfit["tmp_path"] / "cuda-bundle"
    receipt = module.run(postfit["postfit"], output)
    recipe = json.loads((output / "recipe.json").read_text())
    assert recipe["format"] == "vcm-intent-slots-v1"
    assert recipe["head_order"] == ["intent", "role_states", "validity", "values", "entities"]
    assert recipe["engine_status"]["onnxruntime"] == "exported"
    assert recipe["engine_status"]["ncnn"].startswith("blocked:")
    assert recipe["acoustic_qualified"] is False
    for output_name, input_name in (("reference.pt", "checkpoint"), ("contract.json", "contract"),
                                  ("normalization.json", "normalization"), ("policy.json", "calibration"),
                                  ("entities.json", "entities")):
        assert file_sha(output / output_name) == postfit["receipt"]["files"][input_name]["sha256"]
        assert recipe["files"][output_name] == file_sha(output / output_name)
    assert json.loads((output / "catalog.json").read_text()) == {
        "entries": postfit["contract"]["catalog_entries"], "custom": postfit["contract"]["custom"]}
    assert "scripts/test/vcm_mixed_export_cuda.py" not in recipe["source_bindings"]
    assert recipe["checkpoint_source_sha256"] == postfit["grid"][-1]["checkpoint_sha256"]
    assert receipt["qualification"] is False and receipt["CPU_neural_inference_performed"] is False
    assert receipt["ncnn_invoked"] is False
    assert not (output / "network.pt").exists() and not (output / "network.ncnn.param").exists()
    assert json.loads((output / "cuda-export-receipt.json").read_text()) == receipt


def test_existing_export_output_is_preserved_before_any_neural_work(postfit, monkeypatch):
    module = mock_numerical_export(monkeypatch)
    output = postfit["tmp_path"] / "existing"
    output.mkdir()
    (output / "preserve.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.run(postfit["postfit"], output)
    assert (output / "preserve.txt").read_text() == "preserve"


def test_bound_inputs_cannot_change_while_export_is_running(postfit, monkeypatch):
    module = mock_numerical_export(monkeypatch)
    original = module.export_and_parity

    def change(context, output):
        result = original(context, output)
        path = postfit["postfit"] / "observations.json"
        path.write_text("changed calibration observations", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "export_and_parity", change)
    output = postfit["tmp_path"] / "cuda-bundle"
    with pytest.raises(ValueError, match="binding"):
        module.run(postfit["postfit"], output)
    assert (output / "model.onnx").is_file()
    assert not (output / "recipe.json").exists()
    assert not (output / "cuda-export-receipt.json").exists()


def test_a_cpu_model_cannot_reach_onnx_tracing_or_forward(postfit, monkeypatch):
    module = exporter()
    context = module.verify_postfit(postfit["postfit"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)

    class CPUModel:
        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type="cpu", index=None))

        def buffers(self):
            return iter(())

        def __call__(self, *args):
            raise AssertionError("CPU neural forward executed")

    from scripts.test import vcm_mixed_finalize
    monkeypatch.setattr(vcm_mixed_finalize, "load_selected_cuda", lambda context: (
        CPUModel(), {}, {"device": "cuda", "visible_device_count": 1}))
    with pytest.raises(ValueError, match="CUDA"):
        module.export_and_parity(context, postfit["tmp_path"] / "unused")


def test_bundle_copies_cannot_change_before_receipt_emission(postfit, monkeypatch):
    module = mock_numerical_export(monkeypatch)
    original = module.export_and_parity

    def change(context, output):
        result = original(context, output)
        (output / "normalization.json").write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "export_and_parity", change)
    output = postfit["tmp_path"] / "cuda-bundle"
    with pytest.raises(ValueError, match="copy"):
        module.run(postfit["postfit"], output)
    assert (output / "model.onnx").is_file()
    assert not (output / "recipe.json").exists()


@pytest.mark.parametrize("damage", ["partial", "nonfinite", "cpu", "missing_head"])
def test_incomplete_or_invalid_parity_cannot_emit_an_exported_recipe(postfit, monkeypatch, damage):
    module = mock_numerical_export(monkeypatch)
    original = module.export_and_parity

    def change(context, output):
        result = original(context, output)
        if damage == "partial":
            result["onnx_parity"].pop()
        elif damage == "nonfinite":
            result["rewrite_parity"][0]["max_error"] = float("nan")
        elif damage == "cpu":
            result["ort"]["actual_providers"] = ["CPUExecutionProvider"]
        else:
            result["onnx_parity"][0]["head_errors"].pop("entities")
        return result

    monkeypatch.setattr(module, "export_and_parity", change)
    output = postfit["tmp_path"] / "cuda-bundle"
    with pytest.raises(ValueError, match="parity"):
        module.run(postfit["postfit"], output)
    assert not (output / "recipe.json").exists()


def test_postfit_loader_disables_and_records_both_tf32_paths_before_loading(postfit, monkeypatch):
    context = exporter().verify_postfit(postfit["postfit"])["training_context"]
    backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                               cudnn=SimpleNamespace(allow_tf32=True))
    deterministic = {"enabled": False}
    payload = {"contract": context["contract"], "normalization": context["normalization"],
               "training": context["protocol"]["training"], "inputs": context["inputs"],
               "arm": "balanced", "seed": 89, "epoch": 30,
               "status": "mixed_silver_unqualified_candidate", "weights": {"fixture": np.ones(1, np.float32)}}

    def load(path, *, map_location, weights_only):
        assert map_location == "cuda" and weights_only is False
        assert backends.cuda.matmul.allow_tf32 is False
        assert backends.cudnn.allow_tf32 is False
        return deepcopy(payload)

    class Model:
        def __init__(self, contract):
            self.contract = contract

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

        def load_state_dict(self, weights, *, strict):
            assert strict is True

        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type="cuda", index=0))

        def __call__(self, *args):
            raise AssertionError("No neural forward is permitted in this settings test")

    fake_torch = SimpleNamespace(
        __version__="fake-only-no-execution", version=SimpleNamespace(cuda="fake"),
        backends=backends, load=load, isfinite=np.isfinite,
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                             get_device_name=lambda index: "fake CUDA device"),
        use_deterministic_algorithms=lambda value: deterministic.update(enabled=value),
        are_deterministic_algorithms_enabled=lambda: deterministic["enabled"])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "scripts.models.vcm_intent_slots", SimpleNamespace(IntentSlotsVCM=Model))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    _, _, gpu = selected_cuda_loader(context)
    assert gpu["cuda_matmul_allow_tf32"] is False and gpu["cudnn_allow_tf32"] is False
    assert gpu["deterministic_algorithms"] is True
    assert backends.cuda.matmul.allow_tf32 is False and backends.cudnn.allow_tf32 is False


def placement_fixture(directory, *, cpu_data=False, cast_data=False):
    """Static ONNX plus invented timing records; never constructs an ORT session."""
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper

    directory.mkdir(parents=True, exist_ok=True)
    nodes = [h.make_node("Shape", ["features"], ["shape"], name="shape"),
             h.make_node("Gather", ["shape", "index"], ["dim"], name="gather"),
             h.make_node("Cast", ["dim"], ["float_dim"], name="cast_float", to=T.FLOAT),
             h.make_node("Div", ["float_dim", "one"], ["scaled_dim"], name="divide"),
             h.make_node("Cast", ["scaled_dim"], ["frames"], name="cast_back", to=T.INT64),
             h.make_node("Identity", ["features"], ["result"], name="neural_data")]
    if cast_data:
        nodes[2].input[0] = "features"
    graph = h.make_graph(nodes, "static-placement-fixture",
        [h.make_tensor_value_info("features", T.FLOAT, [1, 40, "frames"])],
        [h.make_tensor_value_info("result", T.FLOAT, [1, 40, "frames"]),
         h.make_tensor_value_info("frames", T.INT64, None)],
        initializer=[numpy_helper.from_array(np.array(2, np.int64), "index"),
                     numpy_helper.from_array(np.array(1, np.float32), "one")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    path = directory / "onnx-cuda-optimized.onnx"
    onnx.save(model, path)
    events = []
    for run, frame_count in enumerate((1, 3, 17, 32, 101, 231)):
        start = run * 1000
        for n, node in enumerate(nodes):
            provider = "CPUExecutionProvider" if n < 5 or cpu_data else "CUDAExecutionProvider"
            events.append({"cat": "Node", "ph": "X", "name": node.name + "_kernel_time",
                           "ts": start + 10 + n * 10, "dur": 1,
                           "args": {"provider": provider, "op_name": node.op_type, "node_index": str(n)}})
        events.append({"cat": "Session", "ph": "X", "name": "model_run", "ts": start, "dur": 100,
                       "args": {}})
    profile = dump(directory / "actual-profile.json", events)
    return path, profile, events


def profile_info(graph, original, **fields):
    return {"optimized_graph_path": str(graph), "original_graph": {"path": str(original), "sha256": file_sha(original)},
            "optimized_graph_binding": {"path": str(graph), "sha256": file_sha(graph)}, **fields}


def test_profile_proves_cpu_shape_arithmetic_and_cuda_neural_placement(tmp_path):
    graph, profile, _ = placement_fixture(tmp_path)
    result = exporter().validate_placement(graph, profile)
    assert result["status"] == "verified_cuda_neural_placement"
    assert result["CPU_neural_inference_performed"] is False
    assert result["profile"]["sha256"] == file_sha(profile)
    assert result["optimized_graph"]["sha256"] == file_sha(graph)
    assert [probe["frames"] for probe in result["probes"]] == [1, 3, 17, 32, 101, 231]
    assert {node["name"] for node in result["cpu_metadata_nodes"]} == {
        "shape", "gather", "cast_float", "divide", "cast_back"}
    assert result["cuda_nodes"] == ["neural_data"]


@pytest.mark.parametrize("damage", ["cpu_float", "float_data_cast", "unknown_provider", "missing_provider",
                                     "unknown_node", "changed_operator", "missing_node", "partial_runs",
                                     "outside_run", "extra_run", "missing_call", "unknown_dtype"])
def test_unknown_partial_or_cpu_neural_placement_is_blocked(tmp_path, damage):
    graph, profile, events = placement_fixture(tmp_path, cpu_data=damage == "cpu_float",
                                               cast_data=damage == "float_data_cast")
    kernels = [event for event in events if event["cat"] == "Node"]
    if damage == "unknown_provider":
        kernels[0]["args"]["provider"] = "UnknownExecutionProvider"
    elif damage == "missing_provider":
        kernels[0]["args"].pop("provider")
    elif damage == "unknown_node":
        kernels[0]["name"] = "not_in_optimized_graph_kernel_time"
    elif damage == "changed_operator":
        kernels[0]["args"]["op_name"] = "MatMul"
    elif damage == "missing_node":
        events = [event for event in events if event["name"] != "neural_data_kernel_time"]
    elif damage == "partial_runs":
        events = events[:-7]
    elif damage == "outside_run":
        kernels[0]["ts"] = -10
    elif damage == "extra_run":
        events.append({"cat": "Session", "ph": "X", "name": "model_run", "ts": 10000, "dur": 100, "args": {}})
    elif damage == "missing_call":
        events.remove(kernels[0])
    elif damage == "unknown_dtype":
        import onnx
        model = onnx.load(graph)
        model.graph.node[2].op_type = "UnspecifiedCast"
        model.graph.node[2].domain = "unreviewed.domain"
        onnx.save(model, graph)
        for event in kernels:
            if event["name"] == "cast_float_kernel_time":
                event["args"]["op_name"] = "UnspecifiedCast"
    dump(profile, events)
    with pytest.raises(ValueError, match="placement"):
        exporter().validate_placement(graph, profile)


def test_integer_neural_operator_is_not_mislabeled_as_shape_metadata(tmp_path):
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper

    graph, profile, events = placement_fixture(tmp_path)
    model = onnx.load(graph)
    model.graph.node.append(h.make_node("MatMulInteger", ["integer_a", "integer_b"], ["integer_nn"], name="integer_nn"))
    model.graph.initializer.extend([numpy_helper.from_array(np.ones((2, 2), np.uint8), "integer_a"),
                                   numpy_helper.from_array(np.ones((2, 2), np.uint8), "integer_b")])
    model.graph.output.append(h.make_tensor_value_info("integer_nn", T.INT32, [2, 2]))
    onnx.save(model, graph)
    for run in range(6):
        events.append({"cat": "Node", "ph": "X", "name": "integer_nn_kernel_time", "ts": run * 1000 + 75,
                       "dur": 1, "args": {"provider": "CPUExecutionProvider", "op_name": "MatMulInteger"}})
    dump(profile, events)
    with pytest.raises(ValueError, match="placement"):
        exporter().validate_placement(graph, profile)


@pytest.mark.parametrize("cpu_neural", [False, True])
def test_profile_finish_preserves_files_and_truthful_pass_or_block_report(tmp_path, cpu_neural):
    graph, profile, _ = placement_fixture(tmp_path, cpu_data=cpu_neural)
    original = tmp_path / "model.onnx"
    original.write_bytes(graph.read_bytes())
    session = SimpleNamespace(end_profiling=lambda: str(profile))
    info = profile_info(graph, original)
    if cpu_neural:
        with pytest.raises(ValueError, match="placement"):
            exporter().finish_cuda_profile(session, original, info)
        result = json.loads((tmp_path / "ort-node-placement.json").read_text())
        assert result["status"] == "blocked_node_placement"
        assert result["CPU_neural_inference_performed"] is True
    else:
        result = exporter().finish_cuda_profile(session, original, info)
        assert result["status"] == "verified_cuda_neural_placement"
        assert result["report"]["sha256"] == file_sha(tmp_path / "ort-node-placement.json")
        assert result["CPU_neural_inference_performed"] is False
    assert result["profile"]["sha256"] == file_sha(profile)
    assert result["optimized_graph"]["sha256"] == file_sha(graph)
    assert result["original_graph"]["sha256"] == file_sha(original)


def test_missing_placement_proof_cannot_emit_an_exported_recipe(postfit, monkeypatch):
    module = mock_numerical_export(monkeypatch)
    original = module.export_and_parity

    def change(context, output):
        result = original(context, output)
        result.pop("placement", None)
        return result

    monkeypatch.setattr(module, "export_and_parity", change)
    output = postfit["tmp_path"] / "cuda-bundle"
    with pytest.raises(ValueError, match="placement"):
        module.run(postfit["postfit"], output)
    assert not (output / "recipe.json").exists()


def test_placement_verifier_recomputes_freshly_rebound_cpu_data_profile(tmp_path):
    graph, profile, events = placement_fixture(tmp_path)
    original = tmp_path / "model.onnx"
    original.write_bytes(graph.read_bytes())
    placement = exporter().finish_cuda_profile(SimpleNamespace(end_profiling=lambda: str(profile)), original,
                                              profile_info(graph, original))
    for event in events:
        if event["name"] == "neural_data_kernel_time":
            event["args"]["provider"] = "CPUExecutionProvider"
    dump(profile, events)
    placement["profile"]["sha256"] = file_sha(profile)
    saved = json.loads(Path(placement["report"]["path"]).read_text())
    saved["profile"] = placement["profile"]
    dump(Path(placement["report"]["path"]), saved)
    placement["report"]["sha256"] = file_sha(Path(placement["report"]["path"]))
    with pytest.raises(ValueError, match="placement"):
        exporter().verify_placement_evidence(placement, graph=original)


def test_profile_finish_records_unknown_provider_as_unknown_cpu_execution(tmp_path):
    graph, profile, events = placement_fixture(tmp_path)
    original = tmp_path / "model.onnx"
    original.write_bytes(graph.read_bytes())
    events[0]["args"].pop("provider")
    dump(profile, events)
    with pytest.raises(ValueError, match="placement"):
        exporter().finish_cuda_profile(SimpleNamespace(end_profiling=lambda: str(profile)), original,
                                       profile_info(graph, original))
    assert json.loads((tmp_path / "ort-node-placement.json").read_text())["CPU_neural_inference_performed"] is None


def test_original_graph_cannot_change_during_mocked_session_construction(prepared, monkeypatch):
    fake_ort(monkeypatch, prepared["contract"], mutate_original=True)
    with pytest.raises(ValueError, match="binding"):
        exporter().create_cuda_session(fake_graph(prepared))


@pytest.mark.parametrize("which", ["original", "optimized"])
def test_cached_session_cannot_publish_changed_original_or_optimized_graph(tmp_path, which):
    graph, profile, _ = placement_fixture(tmp_path)
    original = tmp_path / "model.onnx"
    original.write_bytes(graph.read_bytes())
    info = {"optimized_graph_path": str(graph), "original_graph": {
        "path": str(original), "sha256": file_sha(original)}, "optimized_graph_binding": {
        "path": str(graph), "sha256": file_sha(graph)}}
    changed = original if which == "original" else graph
    if which == "original":
        changed.write_bytes(b"replacement graph bytes")
    else:
        import onnx
        model = onnx.load(changed)
        model.producer_name = "changed after cached session construction"
        onnx.save(model, changed)
    with pytest.raises(ValueError, match="binding"):
        exporter().finish_cuda_profile(SimpleNamespace(end_profiling=lambda: str(profile)), original, info)
