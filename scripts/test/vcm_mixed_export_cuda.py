"""Separate CUDA-only ONNX export of the completed, calibrated mixed candidate.

Preserves the existing v1 bundle and OrderedHeads interfaces. CPU-only inference,
ncnn conversion, waveform evaluation, installation and Pi qualification remain
deferred; no neural execution occurs at import time.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

from scripts.runtime.pcv_schema import canonical
from scripts.train.vcm_intent_slots_data import sha
from scripts.train.vcm_intent_slots_data import write_new
from scripts.test import vcm_mixed_finalize as finalize

ROOT = Path(__file__).resolve().parents[2]
HEAD_NAMES = ("intent", "role_states", "validity", "values", "entities")
PROBE_FRAMES = (1, 3, 17, 32, 101, 231)
TOLERANCE = 2e-5
METADATA_OPERATIONS = {"Identity", "Cast", "Gather", "GatherElements", "Slice", "Concat", "Unsqueeze",
                       "Squeeze", "Reshape", "Transpose", "Expand", "Add", "Sub", "Mul", "Div", "Mod",
                       "Neg", "Abs", "Max", "Min", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual",
                       "And", "Or", "Not", "Where", "Range", "ReduceProd", "ReduceSum", "ConstantOfShape"}
METADATA_ARGUMENTS = {"Reshape": {1}, "Expand": {1}, "Slice": {1, 2, 3, 4}, "Gather": {1},
                      "GatherElements": {1}, "Unsqueeze": {1}, "Squeeze": {1}, "Pad": {1, 3},
                      "Tile": {1}, "Resize": {2, 3}, "Split": {1}, "TopK": {1}, "ConstantOfShape": {0}}
INTEGER_TYPES = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64", "BOOL"}
NUMERIC_TYPES = INTEGER_TYPES | {"FLOAT", "DOUBLE", "FLOAT16", "BFLOAT16"}
RUNTIME_SOURCES = ("vcm/frontend.py", "scripts/runtime/pcv_schema.py",
                   "scripts/models/vcm_intent_slots.py", "scripts/runtime/vcm_intent_slots_bindings.py",
                   "scripts/runtime/vcm_intent_slots_contract.py", "scripts/runtime/vcm_intent_slots_entities.py",
                   "scripts/runtime/vcm_intent_slots_decode.py", "scripts/runtime/vcm_intent_slots_runtime.py",
                   "scripts/runtime/pcv_benchmark_adapter.py")


def explicit_gpu_uuid(visible):
    if not isinstance(visible, str) or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", visible):
        raise ValueError("Exactly one explicit GPU UUID in CUDA_VISIBLE_DEVICES is required")
    return visible


def check_binding(item):
    if (not isinstance(item, dict) or not isinstance(item.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or sha(item["path"]) != item["sha256"]):
        raise ValueError("Changed or missing frozen export input binding")


def verify_postfit(postfit):
    from scripts.runtime.vcm_intent_slots_bindings import verify_evidence_bindings

    postfit = Path(postfit).resolve()
    receipt_path = postfit / "postfit-receipt.json"
    receipt = finalize.read_json(receipt_path)
    if (receipt.get("status") != "unqualified_postfit_candidate" or receipt.get("qualification") is not False
            or receipt.get("completed_grid_candidates") != 6 or receipt.get("grid_updates") != 17280
            or receipt.get("gpu", {}).get("device") != "cuda"
            or receipt.get("gpu", {}).get("visible_device_count") != 1
            or receipt.get("missing_calibration_intents") != []
            or receipt.get("thresholds") != {"intent": .95, "slot": .90}
            or any(receipt.get(key) is not False for key in (
                "weights_fitted", "test_accessed", "final_tts_accessed", "evaluation_views_read",
                "CPU_neural_inference_performed"))):
        raise ValueError("Incomplete, contaminated or non-CUDA post-fit evidence")
    files = receipt["files"]
    required = {"grid", "selection", "inputs", "protocol", "contract", "checkpoint", "normalization",
                "frontend", "entities", "calibration", "raw_entity_policy", "observations"}
    sources = receipt.get("source_bindings", {})
    if not required <= set(files) or not set(finalize.SOURCE_FILES) <= set(sources):
        raise ValueError("Missing complete post-fit file/source bindings")
    for name, item in sources.items():
        expected_path = (ROOT / name).resolve()
        if not expected_path.is_relative_to(ROOT) or Path(item["path"]).resolve() != expected_path:
            raise ValueError("Post-fit source binding is not the exact current implementation: " + name)
    for item in [*files.values(), *sources.values()]:
        check_binding(item)
    training = finalize.verify_training(Path(files["grid"]["path"]).parent, files["inputs"]["path"])
    expected_paths = {"selection": training["training"] / "selection-before-calibration.json",
                      "grid": training["training"] / "grid.json", "inputs": training["inputs_path"],
                      "protocol": training["protocol_path"], "contract": training["inputs"]["contract"]["path"],
                      "checkpoint": training["selected"]["checkpoint"], "frontend": ROOT / "vcm/frontend.py",
                      "normalization": postfit / "normalization.json", "entities": postfit / "entities.json",
                      "raw_entity_policy": postfit / "raw-entity-policy.json",
                      "observations": postfit / "observations.json", "calibration": postfit / "calibration/calibration.json"}
    expected_runs = {f"{arm}-{seed}": {
        name: finalize.binding(training["training"] / f"{arm}-{seed}" / filename)
        for name, filename in (("identity", "run-identity.json"), ("history", "history.json"),
                               ("selection", "selection-before-calibration.json"))}
        for arm, seed in finalize.GRID}
    if (any(Path(files[name]["path"]).resolve() != Path(path).resolve() for name, path in expected_paths.items())
            or receipt.get("training_runs") != expected_runs):
        raise ValueError("Post-fit file/grid receipts omit or rebind the actual completed candidate")
    if (training["selected"] != receipt["selected"]
            or receipt.get("training_evidence_bindings") != training["evidence_bindings"]
            or Path(files["checkpoint"]["path"]).resolve() != Path(training["selected"]["checkpoint"])
            or files["checkpoint"]["sha256"] != training["selected"]["checkpoint_sha256"]
            or files["normalization"]["sha256"] != sha(training["normalization_path"])):
        raise ValueError("Post-fit candidate or exact normalization differs from verified complete training")
    contract = finalize.read_json(files["contract"]["path"])
    policy = finalize.read_json(files["calibration"]["path"])
    memory = finalize.read_json(files["entities"]["path"])
    raw_policy = finalize.read_json(files["raw_entity_policy"]["path"])
    expected = {"protocol": training["inputs"]["protocol"]["sha256"],
                "manifest": training["inputs"]["views"]["calibration"]["sha256"],
                "contract_hash": contract["contract_hash"], "checkpoint": files["checkpoint"]["sha256"],
                "normalization": files["normalization"]["sha256"], "frontend": sha(ROOT / "vcm/frontend.py"),
                "memory": files["entities"]["sha256"], "raw_entity_policy": files["raw_entity_policy"]["sha256"],
                "entities": raw_policy["entities"]}
    if (contract != training["contract"] or policy.get("require_calibration") is not True
            or policy.get("calibration_id") != receipt.get("calibration_id")
            or policy.get("qualification") is not False
            or policy.get("entities") != raw_policy["entities"]
            or any(policy.get("binding", {}).get(key) != value for key, value in expected.items())):
        raise ValueError("Calibration differs from exact post-fit candidate/source/entity bindings")
    verify_evidence_bindings(contract, policy, memory, encoder=expected["checkpoint"],
                             frontend=expected["frontend"], normalization=expected["normalization"])
    verify_evidence_bindings(contract, raw_policy, memory, encoder=expected["checkpoint"],
                             frontend=expected["frontend"], normalization=expected["normalization"])
    return {"postfit": postfit, "receipt": receipt, "receipt_path": receipt_path,
            "training_context": training, "contract": contract, "policy": policy, "memory": memory,
            "catalog": {"entries": contract["catalog_entries"], "custom": contract["custom"]},
            "normalization_sha256": expected["normalization"], "checkpoint_sha256": expected["checkpoint"],
            "frozen_bindings": [finalize.binding(receipt_path), *files.values(),
                                *receipt["source_bindings"].values(), *training["evidence_bindings"].values()]}


def require_cuda_tensor(tensor):
    if tensor.device.type != "cuda" or tensor.device.index not in (None, 0):
        raise ValueError("Neural tensors must stay on the single visible CUDA device")


def require_cuda_module(module):
    parameters = list(module.parameters())
    if not parameters:
        raise ValueError("Missing CUDA model parameters")
    for tensor in [*parameters, *module.buffers()]:
        require_cuda_tensor(tensor)


def head_arrays(outputs, contract):
    sizes = [19, len(contract["roles"]) * 4, 4,
             max(value[1] for value in contract["value_offsets"].values()), 320]
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Exactly five ordered graph heads required")
    arrays = [np.asarray(value) for value in outputs]
    if any(value.shape != (1, size) or value.dtype != np.float32 or not np.isfinite(value).all()
           for value, size in zip(arrays, sizes)):
        raise ValueError("Wrong/nonfinite FP32 graph head shape/order")
    return arrays


def compare_heads(reference, actual, contract):
    reference, actual = head_arrays(reference, contract), head_arrays(actual, contract)
    errors = {name: float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
              for name, a, b in zip(HEAD_NAMES, reference, actual)}
    if max(errors.values()) > TOLERANCE:
        raise ValueError("CUDA graph heads exceed fixed 2e-5 rewrite/export parity tolerance")
    return {"max_error": max(errors.values()), "head_errors": errors}


def tensors_from_heads(outputs, contract):
    arrays = head_arrays(outputs, contract)
    return {"intent_logits": arrays[0][0], "role_state_logits": arrays[1].reshape(-1, 4),
            "request_validity_logits": arrays[2][0],
            "value_logits": {role: arrays[3][0, a:b] for role, (a, b) in contract["value_offsets"].items()},
            "entity_embeddings": arrays[4].reshape(5, 64)}


def compare_probe(reference, actual, context):
    from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors

    result = compare_heads(reference, actual, context["contract"])
    decisions = [decide(prediction_from_tensors(tensors_from_heads(outputs, context["contract"]),
                         context["contract"], context["memory"], context["policy"]),
                         context["contract"], context["catalog"], context["policy"])
                 for outputs in (reference, actual)]
    signatures = [{key: decision[key] for key in ("accepted", "command", "reason")} for decision in decisions]
    if canonical(signatures[0]) != canonical(signatures[1]):
        raise ValueError("CUDA ONNX final decision differs from PyTorch CUDA despite numeric parity")
    return {**result, "decision_equal": True, "pytorch_decision": decisions[0], "onnx_decision": decisions[1]}


class PlacementError(ValueError):
    def __init__(self, reason, cpu_neural=None):
        super().__init__("ORT node placement: " + reason)
        self.cpu_neural = cpu_neural


def describe_graph(path):
    """Static types plus value provenance and consumer roles; no graph execution."""
    import onnx

    try:
        model = onnx.load(path, load_external_data=False)
        if any(tensor.external_data for tensor in model.graph.initializer):
            raise PlacementError("unbound external graph initializer")
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception as error:
        raise PlacementError("optimized graph type/shape proof unavailable: " + str(error)) from error
    graph, types, consumers = model.graph, {}, {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type:
            types[value.name] = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
    for value in graph.initializer:
        types[value.name] = onnx.TensorProto.DataType.Name(value.data_type)
    nodes = {}
    for node in graph.node:
        if not node.name or node.name in nodes or any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                                                      for attr in node.attribute):
            raise PlacementError("ambiguous node name or unreviewed control-flow subgraph")
        item = {"name": node.name, "operator": node.op_type, "domain": node.domain,
                "inputs": list(node.input), "outputs": list(node.output),
                "output_dtypes": [types.get(name) for name in node.output],
                "static_initializer": node.op_type == "Constant" and bool(node.attribute)}
        nodes[node.name] = item
        for index, name in enumerate(node.input):
            if name:
                consumers.setdefault(name, []).append((node.name, index))
    potential = {value.name for value in graph.initializer if types.get(value.name) in NUMERIC_TYPES}
    candidates = set()
    for _ in range(len(nodes) + 1):
        previous = len(potential)
        for name, node in nodes.items():
            if node["domain"] not in ("", "ai.onnx") or not node["outputs"]:
                continue
            output_types = node["output_dtypes"]
            shape = node["operator"] in ("Shape", "Size") and all(value == "INT64" for value in output_types)
            numeric = all(value in NUMERIC_TYPES for value in output_types)
            derived = (node["operator"] in METADATA_OPERATIONS
                       and all(value in potential for value in node["inputs"] if value))
            if shape or numeric and (node["static_initializer"] or derived):
                potential.update(node["outputs"])
                candidates.add(name)
        if len(potential) == previous:
            break
    output_names = {value.name for value in graph.output}

    sink_cache = {}

    def metadata_sink(tensor, trail):
        if tensor in sink_cache:
            return sink_cache[tensor]
        if tensor in trail or types.get(tensor) not in NUMERIC_TYPES:
            return False
        if tensor in output_names and types[tensor] not in INTEGER_TYPES:
            return False
        users = consumers.get(tensor, [])
        if not users and tensor not in output_names:
            return False
        for name, index in users:
            user = nodes[name]
            if user["domain"] in ("", "ai.onnx") and (
                    user["operator"] in ("Shape", "Size") and index == 0
                    or index in METADATA_ARGUMENTS.get(user["operator"], set())):
                continue
            if name not in candidates or not all(metadata_sink(value, trail | {tensor}) for value in user["outputs"]):
                sink_cache[tensor] = False
                return False
        sink_cache[tensor] = True
        return True

    for name, node in nodes.items():
        node["cpu_metadata_proven"] = name in candidates and all(metadata_sink(value, set()) for value in node["outputs"])
        node["metadata_reason"] = ("typed Shape/Size inspection or shape/constant arithmetic whose consumers use only metadata"
                                   if node["cpu_metadata_proven"] else "neural values or unproven metadata type/dataflow")
    return {"onnx": onnx.__version__, "nodes": nodes, "tensor_dtypes": types}


def validate_placement(graph, profile, *, frames=PROBE_FRAMES):
    description = describe_graph(graph)
    events, nodes = finalize.read_json(profile), description["nodes"]
    if not isinstance(events, list):
        raise PlacementError("missing profiling events")
    runs = sorted([event for event in events if event.get("cat") == "Session" and event.get("name") == "model_run"],
                  key=lambda event: event.get("ts", -1))
    if (not frames or any(type(frame) is not int or frame <= 0 for frame in frames) or len(runs) != len(frames)):
        raise PlacementError("incomplete expected-probe model_run coverage")
    for n, run in enumerate(runs):
        if (run.get("ph") != "X" or not finalize._finite(run.get("ts")) or run["ts"] < 0
                or not finalize._finite(run.get("dur"), positive=True)
                or n and runs[n-1]["ts"] + runs[n-1]["dur"] > run["ts"]):
            raise PlacementError("invalid/overlapping probe run intervals")
    per_run, providers, starts = [dict() for _ in runs], {}, [run["ts"] for run in runs]
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = event.get("name", "")
        if name.endswith(("_fence_before", "_fence_after")):
            continue
        if not name.endswith("_kernel_time"):
            raise PlacementError("unknown node execution event")
        name, args = name.removesuffix("_kernel_time"), event.get("args", {})
        if name not in nodes or args.get("op_name") != nodes[name]["operator"]:
            raise PlacementError("profile node/operator absent from the optimized graph")
        provider = args.get("provider")
        if provider not in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            raise PlacementError("missing or unknown node execution provider")
        if provider == "CPUExecutionProvider" and not nodes[name]["cpu_metadata_proven"]:
            outputs = nodes[name]["output_dtypes"]
            raise PlacementError("CPU neural/unproven-metadata node: " + name,
                                 True if any(value in NUMERIC_TYPES - INTEGER_TYPES for value in outputs) else None)
        if (event.get("ph") != "X" or not finalize._finite(event.get("ts"))
                or not finalize._finite(event.get("dur")) or event["dur"] < 0):
            raise PlacementError("invalid kernel timing")
        index = bisect_right(starts, event["ts"]) - 1
        if (index < 0 or event["ts"] + event["dur"] > runs[index]["ts"] + runs[index]["dur"]
                or name in per_run[index]):
            raise PlacementError("kernel outside a unique run or duplicate node execution")
        per_run[index][name] = provider
        providers.setdefault(name, set()).add(provider)
    required = {name for name, node in nodes.items() if not node["static_initializer"]}
    if not required or any(not required <= set(run) for run in per_run) or any(len(value) != 1 for value in providers.values()):
        raise PlacementError("missing or inconsistent per-probe node placement")
    return {"status": "verified_cuda_neural_placement", "CPU_neural_inference_performed": False,
            "optimized_graph": finalize.binding(graph), "profile": finalize.binding(profile),
            "graph_description": description,
            "probes": [{"frames": frame, "node_providers": run} for frame, run in zip(frames, per_run)],
            "cpu_metadata_nodes": [nodes[name] for name, provider in providers.items() if provider == {"CPUExecutionProvider"}],
            "cuda_nodes": sorted(name for name, provider in providers.items() if provider == {"CUDAExecutionProvider"})}


def require_cuda_provider(session):
    providers = session.get_providers()
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise ValueError("ORT lost its requested CUDA provider; CPU whole-session fallback is forbidden")
    return providers


def create_cuda_session(graph, *, evidence_dir=None):
    import onnxruntime as ort

    explicit_gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES"))
    evidence_dir = Path(evidence_dir or Path(graph).parent).resolve()
    optimized = evidence_dir / "onnx-cuda-optimized.onnx"
    if optimized.exists():
        raise FileExistsError("Preserve existing optimized graph evidence: " + str(optimized))
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("CUDA ONNX Runtime execution provider is unavailable")
    original_binding = finalize.binding(graph)
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
    options.enable_profiling = True
    options.profile_file_prefix = str(evidence_dir / "onnx-cuda-profile")
    options.optimized_model_filepath = str(optimized)
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(graph), options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                provider_options=[{"device_id": "0", "use_tf32": "0"}, {}])
    check_binding(original_binding)
    optimized_binding = finalize.binding(optimized)
    require_cuda_provider(session)
    session.disable_fallback()
    provider_options = session.get_provider_options()
    cuda_options = provider_options.get("CUDAExecutionProvider", {})
    if str(cuda_options.get("device_id")) != "0" or str(cuda_options.get("use_tf32")) != "0":
        raise ValueError("ORT CUDA must use visible device zero and FP32 precision with TF32 disabled")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != "features" or inputs[0].type != "tensor(float)"
            or inputs[0].shape != [1, 40, "frames"]
            or [value.name for value in outputs] != list(HEAD_NAMES)
            or any(value.type != "tensor(float)" for value in outputs)):
        raise ValueError("Changed ONNX input or five-head ordering")
    return session, {"ort": ort.__version__, "actual_providers": require_cuda_provider(session),
                     "provider_options": provider_options, "whole_session_fallback_disabled": True,
                     "input_device": "cuda", "output_device": "cuda",
                     "profiling_enabled": True, "optimized_graph_path": str(optimized),
                     "original_graph": original_binding, "optimized_graph_binding": optimized_binding,
                     "graph_optimization_level": "ORT_DISABLE_ALL",
                     "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]}


def finish_cuda_profile(session, graph, ort_info, *, frames=PROBE_FRAMES):
    """Preserve and audit profile evidence, including failures and unknown placement."""
    optimized = Path(ort_info["optimized_graph_path"]).resolve()
    report_path = optimized.parent / "ort-node-placement.json"
    profile = Path(session.end_profiling()).resolve()
    artifacts = {"original_graph": ort_info.get("original_graph"), "optimized_graph": ort_info.get("optimized_graph_binding"),
                 "profile": finalize.binding(profile)}
    try:
        if (artifacts["original_graph"] is None or artifacts["optimized_graph"] is None
                or Path(artifacts["original_graph"]["path"]).resolve() != Path(graph).resolve()
                or Path(artifacts["optimized_graph"]["path"]).resolve() != optimized):
            raise PlacementError("missing or rebound graph binding snapshot")
        check_binding(artifacts["original_graph"])
        check_binding(artifacts["optimized_graph"])
        if profile.parent != optimized.parent:
            raise PlacementError("profile outside the new evidence directory")
        result = {**validate_placement(optimized, profile, frames=frames), **artifacts,
                  "provider_options": ort_info.get("provider_options", {})}
        for artifact in artifacts.values():
            check_binding(artifact)
    except Exception as error:
        write_new(report_path, {"status": "blocked_node_placement", "qualification": False,
                  "CPU_neural_inference_performed": getattr(error, "cpu_neural", None),
                  "reason": str(error), **artifacts, "provider_options": ort_info.get("provider_options", {})})
        raise
    write_new(report_path, result)
    return {**result, "report": finalize.binding(report_path)}


def verify_placement_evidence(placement, *, graph=None, frames=PROBE_FRAMES, provider_options=None):
    """Read-only verification for an assembler or a complete-waveform receipt."""
    if (placement.get("status") != "verified_cuda_neural_placement"
            or placement.get("CPU_neural_inference_performed") is not False
            or not all(key in placement for key in ("original_graph", "optimized_graph", "profile", "report"))):
        raise PlacementError("missing complete placement proof")
    for key in ("original_graph", "optimized_graph", "profile", "report"):
        check_binding(placement[key])
    if graph is not None and placement["original_graph"] != finalize.binding(graph):
        raise PlacementError("profile does not bind the supplied original graph")
    actual = validate_placement(placement["optimized_graph"]["path"], placement["profile"]["path"], frames=frames)
    saved = finalize.read_json(placement["report"]["path"])
    if (any(saved.get(key) != value for key, value in actual.items())
            or any(placement.get(key) != saved.get(key) for key in saved)
            or provider_options is not None and saved.get("provider_options") != provider_options):
        raise PlacementError("placement report differs from actual graph/profile/provider proof")
    return actual


def run_ort_cuda(session, features, contract):
    import torch

    require_cuda_provider(session)
    require_cuda_tensor(features)
    if (features.dtype != torch.float32 or len(features.shape) != 3
            or tuple(features.shape[:2]) != (1, 40) or features.shape[2] < 1
            or not features.is_contiguous()):
        raise ValueError("CUDA I/O binding requires contiguous batch-one FP32 features")
    torch.cuda.synchronize(features.device)
    io = session.io_binding()
    io.bind_input(name="features", device_type="cuda", device_id=0, element_type=np.float32,
                  shape=tuple(features.shape), buffer_ptr=features.data_ptr())
    for name in HEAD_NAMES:
        io.bind_output(name, "cuda", 0)
    session.run_with_iobinding(io)
    require_cuda_provider(session)
    outputs = io.get_outputs()
    if len(outputs) != 5 or any(value.device_name().lower() != "cuda" for value in outputs):
        raise ValueError("ORT must produce all five heads on CUDA")
    # This is an output transfer for numeric decoding, never a CPU neural forward.
    arrays = io.copy_outputs_to_cpu()
    require_cuda_provider(session)
    return head_arrays(arrays, contract)


def cuda_head_arrays(outputs, contract):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 5:
        raise ValueError("Exactly five CUDA graph heads required")
    for value in outputs:
        require_cuda_tensor(value)
    return head_arrays([value.detach().cpu().numpy().copy() for value in outputs], contract)


def export_and_parity(context, output):
    """Only this explicit execution boundary loads/forwards/exports neural graphs."""
    import torch
    from scripts.train.vcm_intent_slots_export import OrderedHeads

    explicit_gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES"))
    model, payload, gpu = finalize.load_selected_cuda(context["training_context"])
    require_cuda_module(model)
    if gpu.get("device") != "cuda" or gpu.get("visible_device_count") != 1:
        raise ValueError("Exactly one actual CUDA model/device required")
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in payload["weights"].values()):
        raise ValueError("Selected graph weights must retain FP32 precision")
    wrapper = OrderedHeads(model).eval()
    require_cuda_module(wrapper)
    probes, rewrite = [], []
    with torch.no_grad():
        for frames in PROBE_FRAMES:
            features = torch.linspace(-1, 1, 40 * frames, dtype=torch.float32,
                                      device="cuda").reshape(1, 40, frames)
            lengths = torch.tensor([frames], dtype=torch.long, device="cuda")
            require_cuda_tensor(features)
            require_cuda_tensor(lengths)
            require_cuda_module(model)
            masked = model(features, lengths)
            reference = cuda_head_arrays((masked["intent_logits"], masked["role_state_logits"].flatten(1),
                        masked["request_validity_logits"], torch.cat(list(masked["value_logits"].values()), dim=1),
                        masked["entity_embeddings"].flatten(1)), context["contract"])
            require_cuda_module(wrapper)
            rewritten = cuda_head_arrays(wrapper(features), context["contract"])
            rewrite.append({"frames": frames, **compare_heads(reference, rewritten, context["contract"])})
            probes.append((frames, features, reference))
        sample = torch.zeros(1, 40, 128, dtype=torch.float32, device="cuda")
        require_cuda_tensor(sample)
        require_cuda_module(wrapper)
        torch.onnx.export(wrapper, sample, str(output / "model.onnx"), input_names=["features"],
                          output_names=list(HEAD_NAMES), dynamic_axes={"features": {2: "frames"}},
                          opset_version=17, dynamo=False)
    require_cuda_module(model)
    require_cuda_module(wrapper)
    session, ort = create_cuda_session(output / "model.onnx")
    parity, parity_error = [], None
    try:
        parity = [{"frames": frames, **compare_probe(reference, run_ort_cuda(session, features, context["contract"]), context)}
                  for frames, features, reference in probes]
    except Exception as error:
        parity_error = error
    placement = finish_cuda_profile(session, output / "model.onnx", ort, frames=PROBE_FRAMES)
    if parity_error is not None:
        raise parity_error
    return {"gpu": gpu, "ort": ort, "rewrite_parity": rewrite, "onnx_parity": parity,
            "placement": placement,
            "parameters": model.parameter_counts(), "checkpoint_status": payload["status"]}


def verify_parity_report(report, gpu_uuid):
    gpu, ort = report.get("gpu", {}), report.get("ort", {})
    providers = ort.get("actual_providers", [])
    if (gpu.get("device") != "cuda" or gpu.get("visible_device_count") != 1
            or gpu.get("CUDA_VISIBLE_DEVICES") != gpu_uuid
            or not providers or providers[0] != "CUDAExecutionProvider"
            or ort.get("whole_session_fallback_disabled") is not True
            or ort.get("input_device") != "cuda" or ort.get("output_device") != "cuda"):
        raise ValueError("Incomplete actual CUDA parity metadata")
    verify_placement_evidence(report.get("placement", {}), provider_options=ort.get("provider_options"))
    for key in ("rewrite_parity", "onnx_parity"):
        records = report.get(key)
        if (not isinstance(records, list) or [item.get("frames") for item in records] != list(PROBE_FRAMES)):
            raise ValueError("Missing complete fixed CUDA parity probe coverage")
        for item in records:
            errors = item.get("head_errors", {})
            if (set(errors) != set(HEAD_NAMES)
                    or not all(finalize._finite(value) and 0 <= value <= TOLERANCE for value in errors.values())
                    or not finalize._finite(item.get("max_error"))
                    or item["max_error"] != max(errors.values())):
                raise ValueError("Incomplete/nonfinite CUDA head parity evidence")
            if key == "onnx_parity" and (item.get("decision_equal") is not True or canonical(
                    {field: item["pytorch_decision"][field] for field in ("accepted", "command", "reason")}) != canonical(
                    {field: item["onnx_decision"][field] for field in ("accepted", "command", "reason")})):
                raise ValueError("CUDA final decision parity evidence differs")


def run(postfit, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing export evidence must be preserved: " + str(output))
    gpu_uuid = explicit_gpu_uuid(os.environ.get("CUDA_VISIBLE_DEVICES"))
    context = verify_postfit(postfit)
    sources = dict.fromkeys((*RUNTIME_SOURCES, *finalize.SOURCE_FILES,
                            "scripts/train/vcm_intent_slots_export.py", "scripts/test/vcm_mixed_export_cuda.py"))
    source_bindings = {name: finalize.binding(ROOT / name) for name in sources}
    output.mkdir(parents=True, exist_ok=False)
    copies = {"reference.pt": "checkpoint", "contract.json": "contract", "normalization.json": "normalization",
              "policy.json": "calibration", "entities.json": "entities"}
    for name, source in copies.items():
        item = context["receipt"]["files"][source]
        check_binding(item)
        shutil.copyfile(item["path"], output / name)
        if sha(output / name) != item["sha256"]:
            raise ValueError("Export copy changed exact frozen input bytes")
    write_new(output / "catalog.json", context["catalog"])
    catalog_sha256 = sha(output / "catalog.json")
    numerical = export_and_parity(context, output)
    verify_parity_report(numerical, gpu_uuid)
    if numerical["placement"]["original_graph"] != finalize.binding(output / "model.onnx"):
        raise PlacementError("profile does not bind this exported graph")
    if (any(sha(output / name) != context["receipt"]["files"][source]["sha256"] for name, source in copies.items())
            or sha(output / "catalog.json") != catalog_sha256):
        raise ValueError("Export copy changed after exact frozen bundle inputs were written")
    for item in [*context["frozen_bindings"], *source_bindings.values()]:
        check_binding(item)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise ValueError("Explicit CUDA GPU UUID changed during export")
    contract = context["contract"]
    files = [*copies, "catalog.json", "model.onnx"]
    recipe = {"format": "vcm-intent-slots-v1", "contract_hash": contract["contract_hash"],
              "head_order": list(HEAD_NAMES), "value_offsets": contract["value_offsets"],
              "files": {name: sha(output / name) for name in files},
              "source_bindings": {name: source_bindings[name]["sha256"] for name in RUNTIME_SOURCES},
              "engine_status": {"onnxruntime": "exported", "ncnn": "blocked: CPU-inference hold; pnnx not invoked"},
              "rewrite_parity": numerical["rewrite_parity"], "checkpoint_source_sha256": context["checkpoint_sha256"],
              "encoder_id": context["checkpoint_sha256"], "checkpoint_status": numerical["checkpoint_status"],
              "parameters": numerical["parameters"], "graph_bytes": (output / "model.onnx").stat().st_size,
              "installation_allowed": False, "provider_actions_allowed": False, "acoustic_qualified": False}
    write_new(output / "recipe.json", recipe)
    receipt = {"status": "unqualified_cuda_onnx_export", "qualification": False,
               "postfit_receipt": finalize.binding(context["receipt_path"]),
               "selected": context["training_context"]["selected"], "calibration_id": context["policy"]["calibration_id"],
               "pid": os.getpid(), "python": sys.version, "CUDA_VISIBLE_DEVICES": gpu_uuid,
               "numerical": numerical, "parity_tolerance": TOLERANCE, "opset": 17, "dynamic_frames": True,
               "placement_evidence": {key: numerical["placement"][key]
                                      for key in ("original_graph", "optimized_graph", "profile", "report")},
               "files": {name: finalize.binding(output / name) for name in [*files, "recipe.json"]},
               "source_bindings": source_bindings, "postfit_bindings": context["frozen_bindings"],
               "CPU_neural_inference_performed": False, "ncnn_invoked": False,
               "ncnn_status": "blocked_cpu_hold", "CPU_resident_parity_status": "deferred_cpu_hold",
               "waveform_parity_status": "deferred", "physical_Pi_status": "unmeasured_deferred",
               "parity_inputs": "fixed_numeric_features_only", "reference_labels_used": False,
               "test_accessed": False, "final_tts_accessed": False,
               "installation_allowed": False, "provider_actions_allowed": False}
    write_new(output / "cuda-export-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.postfit, args.output), allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked_unqualified_cuda_export", "qualification": False,
                          "error": type(error).__name__, "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
