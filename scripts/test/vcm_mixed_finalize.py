"""Bounded post-fit CUDA calibration; export requires explicit --export.

Never fits weights, changes thresholds, reads final/evaluation TTS, installs a
candidate or claims qualification. The default path performs no CPU neural
forward; the optional exporter is for use after the caller's CPU hold is lifted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from scripts.runtime.pcv_schema import canonical
from scripts.train.vcm_intent_slots_data import rows, sha, write_new

ROOT = Path(__file__).resolve().parents[2]
GRID = [(arm, seed) for arm in ("uniform", "balanced") for seed in (53, 71, 89)]
FIT_VIEWS = ("train", "dev", "calibration", "enrollment")
IDENTITY_FIELDS = ("id", "audio", "split", "kind", "speaker_id", "sha256", "pcm_sha256")
SOURCE_FILES = ("scripts/test/vcm_mixed_finalize.py", "scripts/train/vcm_intent_slots_mixed_train.py",
                "scripts/train/vcm_intent_slots_mixed_data.py", "scripts/train/vcm_intent_slots_mixed_calibration.py",
                "scripts/train/vcm_intent_slots_calibration.py", "scripts/train/vcm_intent_slots_train.py",
                "scripts/train/vcm_intent_slots_data.py", "scripts/train/vcm_intent_slots_export.py",
                "scripts/models/vcm_intent_slots.py", "scripts/runtime/pcv_schema.py",
                "scripts/runtime/vcm_intent_slots_contract.py", "scripts/runtime/vcm_intent_slots_entities.py",
                "scripts/runtime/vcm_intent_slots_decode.py", "scripts/runtime/vcm_intent_slots_bindings.py",
                "scripts/test/speaker_demo_suite.py", "vcm/catalog.py", "vcm/frontend.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def bound_json(item):
    if sha(item["path"]) != item["sha256"]:
        raise ValueError("Changed frozen input binding: " + item["path"])
    return read_json(item["path"])


def _finite(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else True))


def _selection_key(value):
    if (not isinstance(value, list) or len(value) != 3
            or not all(_finite(number) for number in value)):
        raise ValueError("Missing finite human-development selection curve")
    return tuple(value)


def verify_training(training, inputs_path):
    """Verify every completed arm and its human-only selection, without test reads."""
    from scripts.train.vcm_intent_slots_mixed_data import validate_protocol
    from scripts.train.vcm_intent_slots_mixed_train import validate_training

    training, inputs_path = Path(training).resolve(), Path(inputs_path).resolve()
    inputs = read_json(inputs_path)
    protocol = bound_json(inputs["protocol"])
    contract = bound_json(inputs["contract"])
    validate_protocol(protocol, contract)
    for arm, seed in GRID:
        validate_training(protocol, arm, seed, "cuda")
    for split in FIT_VIEWS:
        item = inputs["views"].get(split)
        if not item or sha(item["path"]) != item["sha256"]:
            raise ValueError("Missing/changed frozen fitting view: " + split)
    norm_path = training / "features" / "normalization.json"
    normalization = read_json(norm_path)
    grid = read_json(training / "grid.json")
    selection = read_json(training / "selection-before-calibration.json")
    if (not isinstance(grid, list) or [(row.get("arm"), row.get("seed")) for row in grid] != GRID
            or selection.get("candidate_count") != 6
            or selection.get("selection") != "human development only"
            or selection.get("protocol_sha256") != inputs["protocol"]["sha256"]
            or selection.get("inputs_sha256") != sha(inputs_path)
            or any(selection.get(key) is not False for key in (
                "calibration_accessed", "test_accessed", "final_tts_accessed"))):
        raise ValueError("Incomplete, changed or contaminated six-candidate selection")
    for result, (arm, seed) in zip(grid, GRID):
        run = training / f"{arm}-{seed}"
        identity = read_json(run / "run-identity.json")
        history = read_json(run / "history.json")
        if read_json(run / "selection-before-calibration.json") != result:
            raise ValueError("Grid differs from completed arm selection")
        if (result.get("status") != "mixed_silver_unqualified_candidate"
                or result.get("optimizer_updates") != 2880
                or any(result.get(key) is not False for key in ("test_accessed", "final_tts_accessed"))
                or result.get("initial_state_sha256") == result.get("final_state_sha256")
                or identity.get("arm") != arm or identity.get("seed") != seed
                or identity.get("initialization") != "random" or identity.get("visible_device_count") != 1
                or identity.get("selection") != "original human development only"
                or identity.get("final_tts_accessed") is not False
                or identity.get("protocol") != protocol or identity.get("inputs") != inputs
                or identity.get("model_source_sha256") != sha(ROOT / "scripts/models/vcm_intent_slots.py")
                or not isinstance(history, list) or len(history) != 30):
            raise ValueError("Missing actual completed updates or stale arm evidence")
        for epoch, curve in enumerate(history, 1):
            if (curve.get("epoch") != epoch or curve.get("optimizer_updates") != 96
                    or curve.get("cumulative_updates") != epoch * 96
                    or curve.get("finite_losses_and_gradients") is not True
                    or curve.get("synthetic_development_used_for_selection") is not False
                    or not _finite(curve.get("loss")) or curve["loss"] < 0
                    or not _finite(curve.get("gradient_norm_min")) or curve["gradient_norm_min"] < 0
                    or not _finite(curve.get("gradient_norm_max"))
                    or curve["gradient_norm_max"] < curve["gradient_norm_min"]
                    or not _finite(curve.get("actual_parameter_delta_l2"), positive=True)):
                raise ValueError("Incomplete/nonfinite actual optimization curve")
            _selection_key(curve.get("selection_key"))
            if read_json(run / f"epoch-{epoch:02}.json") != curve:
                raise ValueError("Changed individual epoch receipt")
        selected_epoch = max(range(30), key=lambda index: _selection_key(history[index]["selection_key"])) + 1
        checkpoint = run / f"checkpoint-{selected_epoch:02}.pt"
        if (result.get("epoch") != selected_epoch
                or _selection_key(result.get("development_key")) != _selection_key(history[selected_epoch-1]["selection_key"])
                or Path(result["checkpoint"]).resolve() != checkpoint
                or sha(checkpoint) != result.get("checkpoint_sha256")):
            raise ValueError("Selected checkpoint/epoch/hash differs from the complete development curve")
    winner = max(grid, key=lambda row: (*_selection_key(row["development_key"]), -row["epoch"]))
    if selection.get("selected") != winner:
        raise ValueError("Selected candidate differs from the completed human-development grid")
    evidence_files = [inputs_path, inputs["protocol"]["path"], inputs["contract"]["path"], norm_path,
                      training / "grid.json", training / "selection-before-calibration.json"]
    for result, (arm, seed) in zip(grid, GRID):
        run = training / f"{arm}-{seed}"
        evidence_files.extend([result["checkpoint"], run / "run-identity.json", run / "history.json",
                               run / "selection-before-calibration.json"])
        evidence_files.extend(run / f"epoch-{epoch:02}.json" for epoch in range(1, 31))
    return {"training": training, "inputs_path": inputs_path, "inputs": inputs,
            "protocol_path": Path(inputs["protocol"]["path"]), "protocol": protocol,
            "contract": contract, "normalization": normalization, "normalization_path": norm_path,
            "selected": winner, "grid_updates": 17280, "grid": grid,
            "evidence_bindings": {str(Path(path).resolve()): binding(path) for path in evidence_files}}


def verify_checkpoint_metadata(payload, context):
    selected = context["selected"]
    if (payload.get("contract") != context["contract"]
            or payload.get("training") != context["protocol"]["training"]
            or payload.get("inputs") != context["inputs"]
            or any(payload.get(key) != selected[key] for key in ("arm", "seed", "epoch"))
            or payload.get("status") != "mixed_silver_unqualified_candidate"):
        raise ValueError("Selected checkpoint metadata differs from frozen training")
    norm = payload.get("normalization", {})
    if norm != context["normalization"]:
        raise ValueError("Selected checkpoint normalization differs from training-only normalization")
    if (norm.get("fit_split") != "train" or type(norm.get("frames")) is not int or norm["frames"] <= 0
            or np.asarray(norm.get("mean")).shape != (40,) or np.asarray(norm.get("std")).shape != (40,)
            or not np.isfinite(np.asarray(norm["mean"], float)).all()
            or not np.isfinite(np.asarray(norm["std"], float)).all() or min(norm["std"]) <= 0):
        raise ValueError("Invalid frozen training normalization")


def slot_truth(expected, candidate, field):
    if not isinstance(expected, dict) or not isinstance(candidate, dict):
        return False
    wanted, actual = expected.get("slots"), candidate.get("slots")
    return bool(isinstance(wanted, dict) and isinstance(actual, dict)
                and field in wanted and field in actual
                and candidate.get("intent") == expected.get("intent")
                and canonical(actual[field]) == canonical(wanted[field]))


def make_observation(row, tensors, proposal, contract):
    logits = np.asarray(tensors["intent_logits"], dtype=float)
    if logits.shape != (19,) or not np.isfinite(logits).all():
        raise ValueError("Calibration requires finite actual 19-class logits")
    candidate = proposal.get("candidate") or {}
    return {**{key: row[key] for key in IDENTITY_FIELDS}, "source_row": deepcopy(row),
            "intent_logits": logits.tolist(), "intent_label": contract["intent_order"].index(
                (row.get("command") or {}).get("intent", "OOV")),
            "slots": {field: {"scores": deepcopy(evidence["constituent_scores"]),
                              "provenance": evidence["provenance"],
                              "correct": int(slot_truth(row.get("command"), candidate, field))}
                      for field, evidence in proposal["slot_evidence"].items()}}


def load_selected_cuda(context):
    import torch
    from scripts.models.vcm_intent_slots import IntentSlotsVCM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible CUDA GPU is required for post-fit inference")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(context["selected"]["checkpoint"])
    if sha(checkpoint) != context["selected"]["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed before CUDA load")
    payload = torch.load(checkpoint, map_location="cuda", weights_only=False)
    if sha(checkpoint) != context["selected"]["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed during CUDA load")
    verify_checkpoint_metadata(payload, context)
    if any(not torch.isfinite(value).all() for value in payload["weights"].values()):
        raise ValueError("Nonfinite selected CUDA checkpoint weights")
    model = IntentSlotsVCM(context["contract"]).to("cuda").eval()
    model.load_state_dict(payload["weights"], strict=True)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise ValueError("Selected model must remain entirely on CUDA")
    return model, payload, {"device": "cuda", "visible_device_count": 1,
                           "name": torch.cuda.get_device_name(0), "torch": torch.__version__,
                           "cuda_runtime": torch.version.cuda,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def rebuild_entities(model, context, norm, encoder_binding):
    from scripts.train.vcm_intent_slots_mixed_train import memory_for_model, require_enrollment

    data = {split: rows(context["inputs"]["views"][split]["path"]) for split in FIT_VIEWS}
    require_enrollment(data, context["protocol"], context["contract"])
    return memory_for_model(model, data["enrollment"], context["protocol"], context["contract"],
                            norm, context["inputs"]["audio_root"], binding=encoder_binding)


def infer_features(model, x):
    """The neural boundary accepts normalized features and length only."""
    import torch
    from scripts.train.vcm_intent_slots_train import tensor_prediction

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != 40 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected finite 40-bin calibration features")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("CPU neural inference is held; calibration requires CUDA")
    model.eval()
    with torch.no_grad():
        result = model(torch.tensor(x[None], device=device),
                       torch.tensor([x.shape[1]], device=device))
    return tensor_prediction(result)


def collect_observations(model, data, context, norm, memory, raw_policy):
    from scripts.runtime.vcm_intent_slots_decode import prediction_from_tensors, propose
    from scripts.test.speaker_demo_suite import read_audio
    from scripts.train.vcm_intent_slots_mixed_data import eligible
    from vcm.frontend import features

    observations, seen = [], set()
    for row in data:
        if row["split"] != "calibration" or row["id"] in seen:
            raise ValueError("Only unique full frozen calibration rows may be observed")
        eligible(row, context["protocol"], context["contract"])
        seen.add(row["id"])
        pcm = read_audio(row, Path(context["inputs"]["audio_root"]))
        x = features(pcm, norm["mean"], norm["std"])
        tensors = infer_features(model, x)
        prediction = prediction_from_tensors(tensors, context["contract"], memory, raw_policy)
        proposal = propose(prediction, context["contract"], context["contract"])
        # Reference labels enter only after the model has returned its tensors.
        observations.append(make_observation(row, tensors, proposal, context["contract"]))
    return observations


def fit_mixed_score_calibration(*args, **kwargs):
    from scripts.train.vcm_intent_slots_mixed_calibration import fit_mixed_score_calibration as fit
    return fit(*args, **kwargs)


def export_candidate(*args, **kwargs):
    from scripts.train.vcm_intent_slots_export import export_candidate as export
    return export(*args, **kwargs)


def run(training, inputs_path, output, *, export=False):
    from scripts.runtime.vcm_intent_slots_bindings import verify_evidence_bindings
    from scripts.train.vcm_intent_slots_mixed_data import eligible

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing post-fit evidence must be preserved: " + str(output))
    context = verify_training(training, inputs_path)
    source_bindings = {path: binding(ROOT / path) for path in SOURCE_FILES}
    contract, selected = context["contract"], context["selected"]
    manifest = context["inputs"]["views"]["calibration"]
    data = rows(manifest["path"])
    if not data or len({row["id"] for row in data}) != len(data):
        raise ValueError("Empty or duplicate frozen calibration manifest")
    for row in data:
        if row["split"] != "calibration":
            raise ValueError("Separate calibration view required")
        eligible(row, context["protocol"], contract)
    model, payload, gpu = load_selected_cuda(context)
    verify_checkpoint_metadata(payload, context)
    if gpu.get("device") != "cuda" or gpu.get("visible_device_count") != 1:
        raise ValueError("Post-fit model must use exactly one actual CUDA device")
    output.mkdir(parents=True, exist_ok=False)
    norm = payload["normalization"]
    write_new(output / "normalization.json", norm)
    if sha(output / "normalization.json") != sha(context["normalization_path"]):
        raise ValueError("Freshly serialized normalization differs from its frozen training bytes")
    encoder_binding = {"encoder": selected["checkpoint_sha256"],
                       "frontend": sha(ROOT / "vcm/frontend.py"),
                       "normalization": sha(output / "normalization.json")}
    memory, raw_policy = rebuild_entities(model, context, norm, encoder_binding)
    verify_evidence_bindings(contract, raw_policy, memory, **encoder_binding)
    write_new(output / "entities.json", memory)
    write_new(output / "raw-entity-policy.json", raw_policy)
    observations = collect_observations(model, data, context, norm, memory, raw_policy)
    if (len(observations) != len(data)
            or {row["id"] for row in observations} != {row["id"] for row in data}):
        raise ValueError("Missing full calibration observation coverage")
    write_new(output / "observations.json", observations)
    cal_binding = {"protocol": context["inputs"]["protocol"]["sha256"],
                   "manifest": manifest["sha256"], "contract_hash": contract["contract_hash"],
                   "checkpoint": encoder_binding["encoder"], "normalization": encoder_binding["normalization"],
                   "frontend": encoder_binding["frontend"], "memory": sha(output / "entities.json"),
                   "raw_entity_policy": sha(output / "raw-entity-policy.json"),
                   "entities": deepcopy(raw_policy["entities"])}
    observed_classes = {observation["intent_label"] for observation in observations}
    missing = [intent for index, intent in enumerate(contract["intent_order"]) if index not in observed_classes]
    policy, exported = None, False
    if not missing:
        policy = fit_mixed_score_calibration(observations, cal_binding, output / "calibration",
                    protocol=context["protocol_path"], contract=contract, manifest=manifest["path"])
        verify_evidence_bindings(contract, policy, memory, **encoder_binding)
        if export:
            export_candidate(selected["checkpoint"], contract, output / "bundle",
                             calibration=output / "calibration/calibration.json", memory=output / "entities.json")
            if sha(output / "bundle/normalization.json") != encoder_binding["normalization"]:
                raise ValueError("Export changed the exact serialized normalization hash")
            exported = True
    if sha(selected["checkpoint"]) != encoder_binding["encoder"]:
        raise ValueError("Selected checkpoint changed during post-fit calibration")
    for split in FIT_VIEWS:
        item = context["inputs"]["views"][split]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Frozen fitting/calibration inputs changed during post-fit work")
    for item in source_bindings.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Post-fit source implementation changed during calibration")
    for item in context["evidence_bindings"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Verified training evidence changed during post-fit work")
    files = {"selection": context["training"] / "selection-before-calibration.json",
             "grid": context["training"] / "grid.json", "inputs": context["inputs_path"],
             "protocol": context["protocol_path"], "contract": context["inputs"]["contract"]["path"],
             "checkpoint": selected["checkpoint"], "normalization": output / "normalization.json",
             "frontend": ROOT / "vcm/frontend.py", "entities": output / "entities.json",
             "raw_entity_policy": output / "raw-entity-policy.json", "observations": output / "observations.json"}
    if policy is not None:
        files["calibration"] = output / "calibration/calibration.json"
    if exported:
        files["export_recipe"] = output / "bundle/recipe.json"
    silver = sum(row.get("supervision_status") == "reference_derived_silver" for row in data)
    receipt = {"status": "unqualified_postfit_candidate", "qualification": False,
               "pid": os.getpid(), "python": sys.version,
               "selected": selected, "completed_grid_candidates": 6, "grid_updates": context["grid_updates"],
               "gpu": gpu, "calibration_rows": len(data), "reference_derived_silver_rows": silver,
               "missing_calibration_intents": missing,
               "missing_slot_calibration": policy.get("missing_slot_calibration", []) if policy else [],
               "calibration_id": policy.get("calibration_id") if policy else None,
               "export_requested": bool(export), "export_performed": exported,
               "files": {key: binding(path) for key, path in files.items()},
               "source_bindings": source_bindings,
               "training_evidence_bindings": context["evidence_bindings"],
               "training_runs": {f"{arm}-{seed}": {
                   name: binding(context["training"] / f"{arm}-{seed}" / filename)
                   for name, filename in (("identity", "run-identity.json"), ("history", "history.json"),
                                          ("selection", "selection-before-calibration.json"))}
                   for arm, seed in GRID},
               "thresholds": {"intent": .95, "slot": .90}, "evaluation_views_read": False,
               "final_tts_accessed": False, "test_accessed": False,
               "weights_fitted": False, "CPU_neural_inference_performed": exported,
               "installation_allowed": False, "provider_actions_allowed": False,
               "qualification_limits": ["Reference-derived silver and independent acoustic review remain distinct",
                                        "Selected-candidate held-out, ownership and physical Pi qualification remain pending"]}
    write_new(output / "postfit-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", action="store_true",
                        help="Explicit CPU export/verification after the caller's CPU hold is lifted")
    args = parser.parse_args()
    try:
        result = run(args.training, args.inputs, args.output, export=args.export)
        print(json.dumps(result, allow_nan=False))
        return 0 if result["calibration_id"] else 2
    except Exception as error:
        print(json.dumps({"status": "blocked_unqualified", "error": type(error).__name__,
                          "reason": str(error), "qualification": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
