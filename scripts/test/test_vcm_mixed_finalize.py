"""Post-fit accounting/binding tests; no neural CPU or GPU forward is executed."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def finalize():
    name = "scripts.test.vcm_mixed_finalize"
    assert importlib.util.find_spec(name) is not None, "Missing bounded post-fit finalizer"
    return importlib.import_module(name)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha(path)}


@pytest.fixture
def prepared(tmp_path):
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
    from scripts.runtime.vcm_intent_slots_contract import build_slot_contract

    cases = baseline_cases() + extended_cases()
    contract = build_slot_contract(cases, cases, DEFAULT_CUSTOM)
    protocol = json.loads((ROOT / "experiments/vcm-mixed-final-v2/protocol.json").read_text())
    protocol.update(contract_hash=contract["contract_hash"], catalog_hash=contract["catalog_hash"])
    protocol_path = dump(tmp_path / "protocol.json", protocol)
    contract_path = dump(tmp_path / "contract.json", contract)
    views = {}
    for split in ("train", "dev", "calibration", "enrollment"):
        path = tmp_path / (split + ".jsonl")
        path.write_text("", encoding="utf-8")
        views[split] = file_binding(path)
    # These must remain unopened by this post-fit calibration stage.
    views["test"] = {"path": str(tmp_path / "never-read-test.jsonl"), "sha256": "a" * 64}
    views["final_tts"] = {"path": str(tmp_path / "never-read-final-tts.jsonl"), "sha256": "b" * 64}
    inputs = {"protocol": file_binding(protocol_path), "contract": file_binding(contract_path),
              "views": views, "audio_root": str(tmp_path)}
    inputs_path = dump(tmp_path / "inputs.json", inputs)
    training = tmp_path / "training"
    norm = {"mean": [0.0] * 40, "std": [1.0] * 40, "fit_split": "train", "frames": 123}
    dump(training / "features" / "normalization.json", norm)
    grid = []
    for index, (arm, seed) in enumerate((a, s) for a in ("uniform", "balanced") for s in (53, 71, 89)):
        run = training / f"{arm}-{seed}"
        run.mkdir(parents=True)
        checkpoint = run / "checkpoint-30.pt"
        checkpoint.write_bytes(f"fixture weights {arm}/{seed}".encode())
        history = [{"epoch": epoch, "optimizer_updates": 96, "cumulative_updates": epoch * 96,
                    "loss": 1.0 / epoch, "gradient_norm_min": 0.1, "gradient_norm_max": 0.2,
                    "finite_losses_and_gradients": True, "actual_parameter_delta_l2": 0.01,
                    "selection_key": [0.4 + index * 0.01 + epoch * 0.001, 0, 0.4],
                    "synthetic_development_used_for_selection": False}
                   for epoch in range(1, 31)]
        dump(run / "history.json", history)
        for row in history:
            dump(run / f"epoch-{row['epoch']:02}.json", row)
        result = {"status": "mixed_silver_unqualified_candidate", "arm": arm, "seed": seed,
                  "epoch": 30, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint),
                  "development_key": history[-1]["selection_key"], "optimizer_updates": 2880,
                  "test_accessed": False, "final_tts_accessed": False,
                  "initial_state_sha256": "1" * 64, "final_state_sha256": "2" * 64}
        dump(run / "selection-before-calibration.json", result)
        dump(run / "run-identity.json", {"arm": arm, "seed": seed, "initialization": "random",
             "visible_device_count": 1, "protocol": protocol, "inputs": inputs,
             "model_source_sha256": file_sha(ROOT / "scripts/models/vcm_intent_slots.py"),
             "final_tts_accessed": False, "selection": "original human development only"})
        grid.append(result)
    dump(training / "grid.json", grid)
    selection = {"selected": grid[-1], "candidate_count": 6, "selection": "human development only",
                 "protocol_sha256": file_sha(protocol_path), "inputs_sha256": file_sha(inputs_path),
                 "calibration_accessed": False, "test_accessed": False, "final_tts_accessed": False}
    dump(training / "selection-before-calibration.json", selection)
    return {"training": training, "inputs_path": inputs_path, "inputs": inputs,
            "protocol": protocol, "contract": contract, "grid": grid, "norm": norm,
            "selection": selection, "tmp_path": tmp_path}


def test_grid_verification_accepts_all_six_completed_runs_without_evaluation_reads(prepared):
    context = finalize().verify_training(prepared["training"], prepared["inputs_path"])
    assert context["selected"]["arm"] == "balanced" and context["selected"]["seed"] == 89
    assert context["selected"]["checkpoint_sha256"] == prepared["grid"][-1]["checkpoint_sha256"]
    assert context["grid_updates"] == 17280


@pytest.mark.parametrize("damage", ["missing_arm", "duplicate_arm", "incomplete_updates", "nan_loss",
                                     "zero_delta", "changed_checkpoint", "wrong_winner", "stale_inputs",
                                     "synthetic_selector", "partial_curve"])
def test_incomplete_or_changed_training_cannot_reach_calibration(prepared, damage):
    run = prepared["training"]
    grid = deepcopy(prepared["grid"])
    if damage == "missing_arm":
        dump(run / "grid.json", grid[:-1])
    elif damage == "duplicate_arm":
        dump(run / "grid.json", grid[:-1] + [grid[0]])
    elif damage == "incomplete_updates":
        grid[0]["optimizer_updates"] = 96
        dump(run / "grid.json", grid)
    elif damage in ("nan_loss", "zero_delta", "synthetic_selector", "partial_curve"):
        path = run / "uniform-53" / "history.json"
        history = json.loads(path.read_text())
        if damage == "partial_curve":
            history.pop()
        elif damage == "synthetic_selector":
            history[0]["synthetic_development_used_for_selection"] = True
        else:
            history[0]["loss" if damage == "nan_loss" else "actual_parameter_delta_l2"] = (
                float("nan") if damage == "nan_loss" else 0)
        path.write_text(json.dumps(history), encoding="utf-8")
    elif damage == "changed_checkpoint":
        Path(grid[-1]["checkpoint"]).write_bytes(b"changed")
    elif damage == "wrong_winner":
        dump(run / "selection-before-calibration.json", {**prepared["selection"], "selected": grid[0]})
    else:
        dump(prepared["inputs_path"], {**prepared["inputs"], "audio_root": "changed"})
    with pytest.raises(ValueError):
        finalize().verify_training(run, prepared["inputs_path"])


@pytest.mark.parametrize("expected,candidate,want", [
    (None, {}, False),
    ({"intent": "TIME", "slots": {}}, {"intent": "TIME", "slots": {}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": None}}, {"intent": "BRIGHTNESS", "slots": {}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": 20}}, {"intent": "BRIGHTNESS", "slots": {}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": 20}}, {"intent": "TIME", "slots": {"percent": 20}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": 20}}, {"intent": "BRIGHTNESS", "slots": {"percent": 20.0}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": 1}}, {"intent": "BRIGHTNESS", "slots": {"percent": True}}, False),
    ({"intent": "BRIGHTNESS", "slots": {"percent": 20}}, {"intent": "BRIGHTNESS", "slots": {"percent": 20}}, True),
])
def test_slot_truth_requires_present_typed_reference_and_matching_candidate_intent(expected, candidate, want):
    assert finalize().slot_truth(expected, candidate, "percent") is want


def test_partial_negative_proposal_cannot_gain_none_equals_none_calibration_truth(prepared):
    row = {"id": "negative", "audio": "negative.wav", "split": "calibration", "kind": "synthetic",
           "speaker_id": "Kokoro-82M/am_echo", "sha256": "a" * 64, "pcm_sha256": "b" * 64,
           "command": None, "transcript": "tell me a joke", "supervision_status": "reference_derived_silver"}
    proposal = {"candidate": None, "slot_evidence": {"percent": {
        "constituent_scores": [0.8, 0.9], "provenance": "finite_acoustic_value"}}}
    observation = finalize().make_observation(row, {"intent_logits": np.zeros(19)}, proposal,
                                               prepared["contract"])
    assert observation["slots"]["percent"]["correct"] == 0
    assert observation["intent_label"] == 0
    assert observation["source_row"] == row and observation["source_row"] is not row
    assert all(observation[key] == row[key] for key in (
        "id", "audio", "split", "kind", "speaker_id", "sha256", "pcm_sha256"))


def test_checkpoint_metadata_and_normalization_cannot_be_changed_after_selection(prepared):
    context = finalize().verify_training(prepared["training"], prepared["inputs_path"])
    payload = {"contract": prepared["contract"], "normalization": prepared["norm"],
               "training": prepared["protocol"]["training"], "inputs": prepared["inputs"],
               "arm": "balanced", "seed": 89, "epoch": 30, "status": "mixed_silver_unqualified_candidate"}
    finalize().verify_checkpoint_metadata(payload, context)
    payload["normalization"]["mean"][0] = 1.0
    with pytest.raises(ValueError, match="normalization"):
        finalize().verify_checkpoint_metadata(payload, context)


def calibration_fixture(prepared, *, missing_intent=False):
    data = []
    intents = prepared["contract"]["intent_order"]
    for index, intent in enumerate(intents[:-1] if missing_intent else intents):
        reference = next(row for row in prepared["contract"]["cases"]
                         if (row.get("command") or {}).get("intent", "OOV") == intent)
        path = prepared["tmp_path"] / f"calibration-audio-{index}.wav"
        path.write_bytes(f"audio file fixture {index}".encode())
        data.append({"id": f"calibration/{index}", "audio": str(path), "split": "calibration",
                     "kind": "synthetic", "speaker_id": "Kokoro-82M/am_echo", "voice_id": "am_echo",
                     "sha256": file_sha(path), "pcm_sha256": hashlib.sha256(f"pcm/{index}".encode()).hexdigest(),
                     "source": "hexgrad/Kokoro-82M", "source_revision": prepared["protocol"]["source"]["revision"],
                     "license": "Apache-2.0", "pair_id": f"pair/{index}", "recording_group": f"group/{index}",
                     "transcript": reference.get("spoken_text", reference["text"]),
                     "reference_id": reference["id"], "command": reference["command"],
                     "reference_status": "supported" if reference["command"] is not None else "unsupported",
                     "role": "command", "metric_eligible": True, "qa_eligible": False,
                     "supervision_status": "reference_derived_silver", "acoustic_listening_verified": False})
    path = Path(prepared["inputs"]["views"]["calibration"]["path"])
    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
    prepared["inputs"]["views"]["calibration"] = file_binding(path)
    dump(prepared["inputs_path"], prepared["inputs"])
    for arm, seed in ((a, s) for a in ("uniform", "balanced") for s in (53, 71, 89)):
        path = prepared["training"] / f"{arm}-{seed}" / "run-identity.json"
        dump(path, {**json.loads(path.read_text()), "inputs": prepared["inputs"]})
    prepared["selection"]["inputs_sha256"] = file_sha(prepared["inputs_path"])
    dump(prepared["training"] / "selection-before-calibration.json", prepared["selection"])
    return data


def mocked_gpu_pipeline(prepared, monkeypatch, *, wrong_export_norm=False, stale_memory=False):
    module = finalize()
    from scripts.runtime.vcm_intent_slots_contract import digest

    payload = {"contract": prepared["contract"], "normalization": prepared["norm"],
               "training": prepared["protocol"]["training"], "inputs": prepared["inputs"],
               "arm": "balanced", "seed": 89, "epoch": 30, "status": "mixed_silver_unqualified_candidate"}
    monkeypatch.setattr(module, "load_selected_cuda", lambda context: (
        object(), deepcopy(payload), {"device": "cuda", "visible_device_count": 1, "name": "fixture GPU"}))

    def rebuild(model, context, norm, encoder_binding):
        if stale_memory:
            encoder_binding = {**encoder_binding, "encoder": "a" * 64}
        memory = {"version": 1, "encoder_binding": encoder_binding,
                  "catalog_hash": context["contract"]["catalog_hash"], "examples": []}
        memory["memory_id"] = digest(memory)
        return memory, {"entities": {}, "require_calibration": False,
                        "contract_hash": context["contract"]["contract_hash"]}

    monkeypatch.setattr(module, "rebuild_entities", rebuild)

    def collect(model, data, context, norm, memory, raw_policy):
        return [module.make_observation(row, {"intent_logits": np.zeros(19)},
                 {"candidate": None, "slot_evidence": {}}, context["contract"]) for row in data]

    monkeypatch.setattr(module, "collect_observations", collect)

    def fit(observations, cal_binding, output, *, protocol, contract, manifest):
        policy = {"binding": cal_binding, "contract_hash": contract["contract_hash"],
                  "entities": cal_binding["entities"], "require_calibration": True,
                  "thresholds": {"intent": .95, "slot": .90}, "missing_slot_calibration": [],
                  "calibration_counts": {"human": 0, "synthetic": len(observations),
                                         "reference_derived_silver": len(observations)},
                  "qualification": False, "status": "fixture_frozen_mixed_unqualified"}
        policy["calibration_id"] = digest(policy)
        dump(Path(output) / "calibration.json", policy)
        return policy

    monkeypatch.setattr(module, "fit_mixed_score_calibration", fit)

    def export(checkpoint, contract, output_dir, *, calibration, memory):
        norm = deepcopy(prepared["norm"])
        if wrong_export_norm:
            norm["mean"][0] = 1
        dump(Path(output_dir) / "normalization.json", norm)
        dump(Path(output_dir) / "recipe.json", {"acoustic_qualified": False})
        return {"acoustic_qualified": False}

    monkeypatch.setattr(module, "export_candidate", export)
    return module


def test_postfit_binds_real_checkpoint_and_exact_norm_and_stays_unqualified(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    output = prepared["tmp_path"] / "postfit"
    receipt = module.run(prepared["training"], prepared["inputs_path"], output)
    calibration = json.loads((output / "calibration" / "calibration.json").read_text())
    memory = json.loads((output / "entities.json").read_text())
    assert calibration["binding"]["checkpoint"] == prepared["grid"][-1]["checkpoint_sha256"]
    assert calibration["binding"]["normalization"] == file_sha(output / "normalization.json")
    assert calibration["binding"]["normalization"] == file_sha(prepared["training"] / "features/normalization.json")
    assert memory["encoder_binding"] == {"encoder": prepared["grid"][-1]["checkpoint_sha256"],
        "frontend": file_sha(ROOT / "vcm/frontend.py"), "normalization": file_sha(output / "normalization.json")}
    assert calibration["binding"]["memory"] == file_sha(output / "entities.json")
    assert receipt["status"] == "unqualified_postfit_candidate" and receipt["qualification"] is False
    assert receipt["calibration_rows"] == 19
    assert receipt["export_performed"] is False and not (output / "bundle").exists()
    assert receipt["evaluation_views_read"] is False and receipt["final_tts_accessed"] is False
    assert file_sha(output / "entities.json") == receipt["files"]["entities"]["sha256"]
    assert json.loads((output / "postfit-receipt.json").read_text()) == receipt


def test_missing_calibration_classes_preserve_scores_and_block_export(prepared, monkeypatch):
    calibration_fixture(prepared, missing_intent=True)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    output = prepared["tmp_path"] / "postfit"
    receipt = module.run(prepared["training"], prepared["inputs_path"], output, export=True)
    assert receipt["status"] == "unqualified_postfit_candidate"
    assert receipt["missing_calibration_intents"] == [prepared["contract"]["intent_order"][-1]]
    assert receipt["export_performed"] is False
    assert (output / "observations.json").is_file()
    assert not (output / "bundle").exists()


def test_stale_entity_encoder_binding_blocks_calibration(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch, stale_memory=True)
    with pytest.raises(ValueError, match="memory|binding"):
        module.run(prepared["training"], prepared["inputs_path"], prepared["tmp_path"] / "postfit")


@pytest.mark.parametrize("wrong_norm", [False, True])
def test_optional_export_preserves_exact_serialized_normalization(prepared, monkeypatch, wrong_norm):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch, wrong_export_norm=wrong_norm)
    output = prepared["tmp_path"] / "postfit"
    if wrong_norm:
        with pytest.raises(ValueError, match="normalization"):
            module.run(prepared["training"], prepared["inputs_path"], output, export=True)
    else:
        receipt = module.run(prepared["training"], prepared["inputs_path"], output, export=True)
        assert receipt["export_performed"] is True
        assert receipt["qualification"] is False
        assert file_sha(output / "bundle/normalization.json") == file_sha(output / "normalization.json")


def test_existing_postfit_outputs_are_preserved(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    output = prepared["tmp_path"] / "postfit"
    output.mkdir()
    (output / "existing.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module.run(prepared["training"], prepared["inputs_path"], output)
    assert (output / "existing.json").read_text() == "preserve"


def test_source_changes_during_calibration_cannot_receive_a_complete_receipt(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    temporary_source = prepared["tmp_path"] / "source.py"
    temporary_source.write_text("frozen source", encoding="utf-8")
    original_binding, original_collect = module.binding, module.collect_observations

    def bind(path):
        if Path(path) == ROOT / "scripts/test/vcm_mixed_finalize.py":
            path = temporary_source
        return original_binding(path)

    def collect(*args):
        temporary_source.write_text("changed source", encoding="utf-8")
        return original_collect(*args)

    monkeypatch.setattr(module, "binding", bind)
    monkeypatch.setattr(module, "collect_observations", collect)
    output = prepared["tmp_path"] / "postfit"
    with pytest.raises(ValueError, match="source implementation changed"):
        module.run(prepared["training"], prepared["inputs_path"], output)
    assert (output / "observations.json").is_file()
    assert not (output / "postfit-receipt.json").exists()


def test_training_receipts_cannot_change_during_postfit_work(prepared, monkeypatch):
    calibration_fixture(prepared)
    module = mocked_gpu_pipeline(prepared, monkeypatch)
    original_collect = module.collect_observations

    def collect(*args):
        path = prepared["training"] / "uniform-53/epoch-01.json"
        record = json.loads(path.read_text())
        record["loss"] = .123
        dump(path, record)
        return original_collect(*args)

    monkeypatch.setattr(module, "collect_observations", collect)
    output = prepared["tmp_path"] / "postfit"
    with pytest.raises(ValueError, match="Verified training evidence changed"):
        module.run(prepared["training"], prepared["inputs_path"], output)
    assert (output / "observations.json").is_file()
    assert not (output / "postfit-receipt.json").exists()


def test_collection_uses_features_only_and_retains_the_full_calibration_row(prepared, monkeypatch):
    module = finalize()
    data = calibration_fixture(prepared)
    row = next(row for row in data if (row["command"] or {}).get("intent") == "BRIGHTNESS")
    context = module.verify_training(prepared["training"], prepared["inputs_path"])
    from scripts.test import speaker_demo_suite
    monkeypatch.setattr(speaker_demo_suite, "read_audio", lambda row, root: np.sin(np.arange(3200) * .1) * .2)

    def infer(model, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[0] != 40:
            raise AssertionError("Reference labels reached the neural boundary")
        contract = prepared["contract"]
        intent = np.full(19, -20.0)
        intent[contract["intent_order"].index("BRIGHTNESS")] = 20
        states = np.full((len(contract["roles"]), 4), -20.0)
        states[:, 0] = 20
        states[contract["roles"].index("brightness_percent"), 0] = -20
        states[contract["roles"].index("brightness_percent"), 1] = 20
        values = {role: np.full(len(options), -20.0) for role, options in contract["finite_values"].items()}
        for scores in values.values():
            scores[0] = 20
        values["brightness_percent"][:] = -20
        values["brightness_percent"][20] = 20
        return {"intent_logits": intent, "role_state_logits": states,
                "request_validity_logits": np.array([20, -20, -20, -20]),
                "value_logits": values, "entity_embeddings": np.ones((5, 64))}

    monkeypatch.setattr(module, "infer_features", infer)
    observations = module.collect_observations(object(), [row], context, prepared["norm"], None, {"entities": {}})
    assert len(observations) == 1
    assert observations[0]["source_row"] == row
    assert observations[0]["slots"]["percent"]["correct"] == 1
    assert observations[0]["intent_label"] == prepared["contract"]["intent_order"].index("BRIGHTNESS")


def test_cpu_neural_boundary_is_blocked_before_any_forward():
    from types import SimpleNamespace

    class Model:
        def parameters(self):
            yield SimpleNamespace(device=SimpleNamespace(type="cpu"))

        def __call__(self, *args):
            raise AssertionError("A neural forward was executed")

    with pytest.raises(ValueError, match="CPU|CUDA"):
        finalize().infer_features(Model(), np.ones((40, 2), dtype=np.float32))


@pytest.mark.parametrize("available,count", [(False, 1), (True, 0), (True, 2)])
def test_actual_cuda_availability_and_single_device_gate_precede_loading(
        prepared, monkeypatch, available, count):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(ValueError, match="CUDA"):
        finalize().load_selected_cuda(finalize().verify_training(prepared["training"], prepared["inputs_path"]))


def test_protocol_recipe_cannot_drift_between_fitting_and_finalization(prepared):
    prepared["protocol"]["training"]["learning_rate"] = .1
    path = Path(prepared["inputs"]["protocol"]["path"])
    dump(path, prepared["protocol"])
    prepared["inputs"]["protocol"] = file_binding(path)
    dump(prepared["inputs_path"], prepared["inputs"])
    for arm, seed in ((a, s) for a in ("uniform", "balanced") for s in (53, 71, 89)):
        path = prepared["training"] / f"{arm}-{seed}" / "run-identity.json"
        dump(path, {**json.loads(path.read_text()), "protocol": prepared["protocol"], "inputs": prepared["inputs"]})
    prepared["selection"].update(protocol_sha256=file_sha(prepared["inputs"]["protocol"]["path"]),
                                   inputs_sha256=file_sha(prepared["inputs_path"]))
    dump(prepared["training"] / "selection-before-calibration.json", prepared["selection"])
    with pytest.raises(ValueError, match="recipe|training"):
        finalize().verify_training(prepared["training"], prepared["inputs_path"])
