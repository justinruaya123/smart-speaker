"""Separate complete-waveform CUDA parity after immutable first-pass evaluation.

Every frozen cohort row is retained. This never recomputes recognition accuracy,
fits/selects/calibrates/enrolls, changes thresholds, reads final TTS, or starts a
CPU/ncnn/resident/Pi backend. Only explicit run() executes neural models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final import assemble_card as accounting
from scripts.test import vcm_mixed_benchmark as benchmark
from scripts.test import vcm_mixed_export_cuda as gpu
from scripts.test import vcm_mixed_finalize as finalize
from scripts.test.speaker_demo_suite import read_audio as read_frozen_audio
from scripts.train.vcm_intent_slots_data import write_new

TOLERANCE = 2e-5
THRESHOLDS = {"intent": .95, "slot": .90}
SUCCESS_STATUSES = {"passed", "matching_capture_fault"}
TRIAL_STATUSES = SUCCESS_STATUSES | {"capture_fault", "execution_fault", "first_pass_missing",
                                     "first_pass_fault", "numeric_mismatch", "decision_mismatch",
                                     "first_pass_mismatch"}
_UNSUPPLIED = object()


def read_json(path):
    return accounting.strict_json(Path(path).read_bytes())


def read_jsonl(path):
    return [accounting.strict_json(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def signature(decision):
    if not isinstance(decision, dict):
        raise ValueError("Missing actual final application decision")
    if (type(decision.get("accepted")) is not bool or "command" not in decision
            or "reason" not in decision or decision["reason"] is not None
            and not isinstance(decision["reason"], str)):
        raise ValueError("Incomplete actual final application decision")
    return {key: deepcopy(decision[key]) for key in ("accepted", "command", "reason")}


def _same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(",", ":"))


def replay_provenance(rule):
    """A replay is regression even when its preserved first pass was fresh."""
    return {"fresh": False, "regression": True, "original_evaluation_fresh": not rule["regression"],
            "original_evaluation_regression": rule["regression"]}


def decision_from_heads(heads, context):
    from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors

    prediction = prediction_from_tensors(gpu.tensors_from_heads(heads, context["contract"]),
                                         context["contract"], context["memory"], context["policy"])
    return decide(prediction, context["contract"], context["catalog"], context["policy"])


def _head_hashes(heads):
    return {name: hashlib.sha256(np.asarray(array, dtype="<f4").tobytes()).hexdigest()
            for name, array in zip(gpu.HEAD_NAMES, heads)}


def compare_wave_heads(reference, actual, context, first_pass):
    result = {"numeric_pass": False, "decision_equal": False, "first_pass_equal": False,
              "head_errors": None, "max_error": None, "final_decision_agreement": False}
    try:
        reference, actual = gpu.head_arrays(reference, context["contract"]), gpu.head_arrays(actual, context["contract"])
        errors = {name: float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                  for name, a, b in zip(gpu.HEAD_NAMES, reference, actual)}
        result.update(head_errors=errors, max_error=max(errors.values()), numeric_pass=max(errors.values()) <= TOLERANCE,
                      pytorch_head_sha256=_head_hashes(reference), onnx_head_sha256=_head_hashes(actual))
        left = decision_from_heads(reference, context)
        result["pytorch_decision"] = left
        if isinstance(first_pass, dict) and not first_pass.get("fault"):
            result["first_pass_equal"] = _same_json(signature(left), signature(first_pass.get("final_application")))
        right = decision_from_heads(actual, context)
        result["onnx_decision"] = right
        equal = _same_json(signature(left), signature(right))
        result.update(decision_equal=equal, final_decision_agreement=equal)
        if first_pass is None:
            result["status"] = "first_pass_missing"
        elif first_pass.get("fault"):
            result.update(status="first_pass_fault", preserved_first_pass_fault=first_pass["fault"])
        else:
            if not result["numeric_pass"]:
                result["status"] = "numeric_mismatch"
            elif not result["decision_equal"]:
                result["status"] = "decision_mismatch"
            elif not result["first_pass_equal"]:
                result["status"] = "first_pass_mismatch"
            else:
                # Reuse the exporter’s exact head/layout/decoder comparator.
                gpu.compare_probe(reference, actual, context)
                result["status"] = "passed"
    except Exception as error:
        result.update(status="execution_fault", fault=type(error).__name__ + ": " + str(error))
    return result


def run_trial(row, first_pass, audio_root, normalization, context, infer_heads, *, read_audio=read_frozen_audio):
    record = {"id": row["id"], "source_row": deepcopy(row), "first_pass_prediction": deepcopy(first_pass),
              "numeric_pass": False, "decision_equal": False, "first_pass_equal": False,
              "final_decision_agreement": False, "matching_capture_fault": False,
              "valid_capture": False, "ort_called": False, "head_errors": None, "max_error": None, "frames": None}
    try:
        pcm = read_audio(row, Path(audio_root))
    except Exception as error:
        record.update(status="capture_fault", capture_stage="file_or_pcm_integrity",
                      fault=type(error).__name__ + ": " + str(error))
        return record
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm
    try:
        pcm = validate_pcm(pcm, 16000)
    except Exception as error:
        fault = type(error).__name__ + ": " + str(error)
        matching = isinstance(first_pass, dict) and first_pass.get("fault") == fault
        record.update(status="matching_capture_fault" if matching else "capture_fault",
                      capture_stage="shared_unchanged_runtime_pcm_boundary", fault=fault,
                      matching_capture_fault=matching, pytorch_capture_fault=fault, onnx_capture_fault=fault)
        return record
    record["valid_capture"] = True
    try:
        from vcm.frontend import features
        x = np.ascontiguousarray(features(pcm, normalization["mean"], normalization["std"]), dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != 40 or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError("Expected finite frozen normalized 40-bin features")
        record["frames"] = x.shape[1]
        record["features_sha256"] = hashlib.sha256(
            f"features:f32le:40:{x.shape[1]}\0".encode() + np.asarray(x, dtype="<f4").tobytes()).hexdigest()
        # This boundary receives normalized acoustic features only, never labels.
        reference, actual = infer_heads(x)
        record["ort_called"] = bool(getattr(infer_heads, "last_ort_called", True))
        record.update(compare_wave_heads(reference, actual, context, first_pass))
    except Exception as error:
        record.update(status="execution_fault", fault=type(error).__name__ + ": " + str(error),
                      ort_called=bool(getattr(infer_heads, "last_ort_called", False)))
    return record


def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_trial(row, *, source_row=_UNSUPPLIED, first_pass_prediction=_UNSUPPLIED):
    """Pure consistency and exact-original checks; a status string proves nothing."""
    if (not isinstance(row, dict) or row.get("status") not in TRIAL_STATUSES
            or not isinstance(row.get("source_row"), dict)):
        raise ValueError("Invalid waveform trial identity or status evidence")
    if (source_row is not _UNSUPPLIED and not _same_json(row["source_row"], source_row)
            or first_pass_prediction is not _UNSUPPLIED and not _same_json(row.get("first_pass_prediction"), first_pass_prediction)):
        raise ValueError("Waveform trial changed the original source reference or first-pass prediction")
    if row.get("id") != row["source_row"].get("id"):
        raise ValueError("Invalid waveform trial identity evidence")
    flags = ("numeric_pass", "decision_equal", "first_pass_equal", "final_decision_agreement",
             "matching_capture_fault", "valid_capture", "ort_called")
    if any(type(row.get(key)) is not bool for key in flags):
        raise ValueError("Missing actual Boolean waveform trial evidence")
    status, errors, original = row["status"], row.get("head_errors"), row.get("first_pass_prediction")
    if original is not None and (not isinstance(original, dict) or original.get("id", row["id"]) != row["id"]):
        raise ValueError("Invalid original first-pass prediction identity")
    if row["matching_capture_fault"] is not (status == "matching_capture_fault"):
        raise ValueError("Contradictory waveform capture-fault evidence")
    if status in ("capture_fault", "matching_capture_fault"):
        if (row["valid_capture"] or row["ort_called"] or row.get("frames") is not None or errors is not None
                or row.get("max_error") is not None or any(row[key] for key in flags[:4])
                or not isinstance(row.get("fault"), str) or not row["fault"]
                or row.get("capture_stage") not in ("file_or_pcm_integrity", "shared_unchanged_runtime_pcm_boundary")
                or any(key in row for key in ("pytorch_decision", "onnx_decision", "features_sha256",
                                              "pytorch_head_sha256", "onnx_head_sha256"))):
            raise ValueError("Invalid waveform capture-fault evidence")
        if status == "matching_capture_fault" and (
                row["capture_stage"] != "shared_unchanged_runtime_pcm_boundary" or not isinstance(original, dict)
                or original.get("fault") != row["fault"] or row.get("pytorch_capture_fault") != row["fault"]
                or row.get("onnx_capture_fault") != row["fault"]):
            raise ValueError("Matching waveform fault lacks exact original/shared-boundary evidence")
        return row
    if not row["valid_capture"]:
        raise ValueError("Compared/executed waveform lacks actual valid capture evidence")
    if row.get("frames") is not None:
        if type(row["frames"]) is not int or row["frames"] <= 0 or not _sha256(row.get("features_sha256")):
            raise ValueError("Invalid actual waveform frames/feature hash evidence")
    elif row["ort_called"] or errors is not None:
        raise ValueError("Actual waveform neural evidence requires finite feature frames")
    if errors is not None:
        if (not row["ort_called"] or not isinstance(errors, dict) or set(errors) != set(gpu.HEAD_NAMES)
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in errors.values())
                or type(row.get("max_error")) not in (int, float) or row["max_error"] != max(errors.values())):
            raise ValueError("Invalid actual per-head waveform error/max-error evidence")
        for key in ("pytorch_head_sha256", "onnx_head_sha256"):
            hashes = row.get(key)
            if not isinstance(hashes, dict) or set(hashes) != set(gpu.HEAD_NAMES) or not all(_sha256(value) for value in hashes.values()):
                raise ValueError("Missing actual waveform head hash evidence")
        if row["numeric_pass"] is not (max(errors.values()) <= TOLERANCE):
            raise ValueError("Waveform numeric flag contradicts actual head errors")
    elif row["numeric_pass"] or row.get("max_error") is not None:
        raise ValueError("Waveform numeric claim lacks actual per-head evidence")
    left = signature(row["pytorch_decision"]) if "pytorch_decision" in row else None
    right = signature(row["onnx_decision"]) if "onnx_decision" in row else None
    equal = left is not None and right is not None and _same_json(left, right)
    if row["decision_equal"] is not equal or row["final_decision_agreement"] is not equal:
        raise ValueError("Waveform final-decision flags contradict actual backend decisions")
    original_equal = (left is not None and isinstance(original, dict) and not original.get("fault")
                      and _same_json(left, signature(original.get("final_application"))))
    if row["first_pass_equal"] is not original_equal:
        raise ValueError("Waveform first-pass flag contradicts actual original decision evidence")
    if status == "execution_fault":
        if not isinstance(row.get("fault"), str) or not row["fault"]:
            raise ValueError("Waveform execution fault lacks actual fault evidence")
        return row
    if errors is None or left is None or right is None or row.get("fault"):
        raise ValueError("Waveform comparison status lacks complete actual head/decision evidence")
    if original is None:
        expected = "first_pass_missing"
    elif original.get("fault"):
        expected = "first_pass_fault"
        if row.get("preserved_first_pass_fault") != original["fault"]:
            raise ValueError("Waveform first-pass fault evidence was rewritten")
    elif not row["numeric_pass"]:
        expected = "numeric_mismatch"
    elif not equal:
        expected = "decision_mismatch"
    elif not original_equal:
        expected = "first_pass_mismatch"
    else:
        expected = "passed"
    if status != expected:
        raise ValueError("Waveform status contradicts actual numeric/decision/original evidence")
    return row


def summarize_records(records, expected_ids, *, identity_field="id"):
    ids = [row.get(identity_field) for row in records]
    if len(set(expected_ids)) != len(expected_ids) or len(set(ids)) != len(ids) or not set(ids) <= set(expected_ids):
        raise ValueError("Duplicate or foreign waveform parity trial identity")
    missing = len(set(expected_ids) - set(ids))
    counts = Counter(row.get("status") for row in records)
    head_max = {name: None for name in gpu.HEAD_NAMES}
    for row in records:
        validate_trial(row)
        errors = row.get("head_errors")
        if errors is not None:
            if (not isinstance(errors, dict) or set(errors) != set(gpu.HEAD_NAMES)
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in errors.values())):
                raise ValueError("Invalid actual per-head waveform parity errors")
            for name, value in errors.items():
                head_max[name] = value if head_max[name] is None else max(head_max[name], value)
    passed = counts["passed"]
    failures = sum(count for status, count in counts.items() if status not in SUCCESS_STATUSES)
    return {"trials": len(expected_ids), "written_rows": len(records), "missing": missing,
            "valid_captures": sum(row.get("valid_capture") is True for row in records),
            "ort_calls": sum(row.get("ort_called") is True for row in records),
            "compared_valid": sum(row.get("head_errors") is not None for row in records),
            "numeric_passes": sum(row.get("numeric_pass") is True for row in records),
            "decision_equal": sum(row.get("decision_equal") is True for row in records),
            "first_pass_equal": sum(row.get("first_pass_equal") is True for row in records),
            "passed_valid": passed, "final_decision_agreements": sum(row["final_decision_agreement"] for row in records),
            "matching_capture_faults": counts["matching_capture_fault"],
            "capture_faults": counts["matching_capture_fault"] + counts["capture_fault"],
            "execution_faults": counts["execution_fault"],
            "faults": counts["matching_capture_fault"] + counts["capture_fault"] + counts["execution_fault"],
            "preserved_first_pass_faults": counts["first_pass_fault"],
            "preserved_first_pass_missing": counts["first_pass_missing"],
            "parity_failures": failures, "statuses": dict(sorted(counts.items())),
            "head_max_errors": head_max, "parity_passed": bool(expected_ids) and missing == 0 and failures == 0,
            "scope": "CUDA numeric/final-decision agreement; no label accuracy is measured; matching capture faults are separate from compared decisions"}


def cpu_neural_claim(placement):
    claim = placement.get("CPU_neural_inference_performed")
    if claim is False and placement.get("status") != "verified_cuda_neural_placement":
        raise ValueError("False CPU-neural claim requires actual verified operator placement")
    if claim is not None and type(claim) is not bool:
        raise ValueError("Invalid CPU-neural placement evidence")
    return claim


def freeze_parity_lock(lock, output):
    write_new(Path(output), lock)
    item = benchmark.binding(output)
    verify_parity_lock(item)
    return item


def verify_parity_lock(binding):
    lock = read_json(benchmark.verify_file(binding, "waveform parity lock"))
    if (lock.get("format") != "vcm-mixed-cuda-wave-parity-lock-v1" or lock.get("neural_calls") != 0
            or lock.get("thresholds") != THRESHOLDS
            or any(lock.get(key) is not False for key in (
                "first_pass_accuracy_recomputed", "weights_fitted", "calibration_fitted", "enrollment_rebuilt", "final_tts_accessed"))):
        raise ValueError("Incomplete or contaminated pre-call waveform parity lock")
    for group in ("files", "audio"):
        for name, item in lock.get(group, {}).items():
            benchmark.verify_file(item, "waveform lock/" + group + "/" + name)
    return lock


def _register(files, label, item):
    if not isinstance(item, dict):
        item = benchmark.binding(item)
    benchmark.verify_file(item, label)
    if label in files and files[label] != item:
        raise ValueError("Changed immutable artifact label: " + label)
    files[label] = deepcopy(item)
    return item


def _prepare(training, inputs, postfit, evaluation, bundle):
    context = finalize.verify_training(training, inputs)
    fitted = benchmark.verify_postfit(context, postfit)
    evidence = accounting.Evidence()
    exported = accounting.verify_bundle(bundle, context, fitted, evidence)
    first_path = Path(evaluation).resolve()
    first_path = first_path / "benchmark-receipt.json" if first_path.is_dir() else first_path
    first = read_json(first_path)
    if (first.get("format") != "vcm-mixed-offline-benchmark-receipt-v2"
            or first.get("status") != "unqualified_offline_replay_complete"
            or first.get("selected") != context["selected"] or first.get("thresholds") != THRESHOLDS
            or first.get("evaluation_order") != list(benchmark.COHORTS)
            or first.get("completed_trial_rows") != 4922 or set(first.get("cohorts", {})) != set(benchmark.COHORTS)
            or first.get("postfit_receipt") != benchmark.binding(fitted["receipt_path"])
            or first.get("CPU_neural_inference_performed") is not False
            or any(first.get(key) is not False for key in ("weights_fitted", "calibration_fitted", "enrollment_rebuilt", "final_tts_accessed"))):
        raise ValueError("Complete unchanged first-pass CUDA evaluation is required before waveform parity")
    accounting.verify_evaluation_files(first, first_path.parent, evidence)
    selected_lock = benchmark.verify_selection(first["selection_lock"]["path"], expected_sha256=first["selection_lock"]["sha256"])
    if (selected_lock.get("selected") != context["selected"]
            or selected_lock.get("calibration_id") != fitted["policy"]["calibration_id"]
            or selected_lock.get("memory_id") != fitted["memory"]["memory_id"]):
        raise ValueError("First-pass selected-candidate lock changed")
    files, data, predictions = {}, {}, {}
    _register(files, "first_pass_receipt", first_path)
    _register(files, "first_pass_selection_lock", first["selection_lock"])
    for name, item in context["evidence_bindings"].items():
        _register(files, "training/" + name, item)
    for name, item in fitted["files"].items():
        _register(files, "postfit/" + name, item)
    _register(files, "postfit/receipt", fitted["receipt_path"])
    for source, item in evidence.sources.items():
        _register(files, "verified_artifact/" + source, {"path": source, "sha256": item["sha256"]})
    _register(files, "bundle/recipe", Path(bundle) / "recipe.json")
    _register(files, "bundle/export_receipt", Path(bundle) / "cuda-export-receipt.json")
    for name, digest in exported["recipe"]["files"].items():
        _register(files, "bundle/" + name, {"path": str((Path(bundle) / name).resolve()), "sha256": digest})
    for name, item in first["files"].items():
        _register(files, "first_pass/" + name, item)
    for name, item in selected_lock["sources"].items():
        _register(files, "first_pass_source/" + name, item)
    for name, item in selected_lock["files"].items():
        _register(files, "first_pass_input/" + name, item)
    for name, rule in benchmark.COHORTS.items():
        cohort = selected_lock["cohorts"][name]
        _register(files, "view/" + name, cohort["view"])
        data[name] = read_jsonl(cohort["view"]["path"])
        records = read_jsonl(first_path.parent / name / "trial-results.jsonl")
        metrics = read_json(first["cohorts"][name]["metrics"]["path"])
        accounting.validate_metrics(metrics, records, expected_n=rule["n"])
        if ([row["source_row"] for row in records] != data[name]
                or cohort.get("rows") != rule["n"] or cohort.get("regression") is not rule["regression"]
                or first["cohorts"][name].get("regression") is not rule["regression"]
                or first["cohorts"][name].get("fresh") is not (not rule["regression"])
                or metrics.get("selected_checkpoint_sha256") != context["selected"]["checkpoint_sha256"]
                or metrics.get("calibration_id") != fitted["policy"]["calibration_id"]):
            raise ValueError("First-pass rows, bindings or fresh/regression accounting changed")
        if "input_view" in rule and cohort["view"] != context["inputs"]["views"][rule["input_view"]]:
            raise ValueError("First-pass cohort differs from original frozen inputs")
        if rule.get("sha256") and cohort["view"]["sha256"] != rule["sha256"]:
            raise ValueError("Original benchmark references changed")
        predictions[name] = {row["id"]: row["prediction"] for row in records}
    for name in ("scripts/test/vcm_mixed_wave_parity_cuda.py", "scripts/test/vcm_mixed_export_cuda.py",
                 "final/assemble_card.py", "vcm/frontend.py", "scripts/runtime/vcm_intent_slots_runtime.py"):
        _register(files, "parity_source/" + name, ROOT / name)
    bindings = {"checkpoint_sha256": context["selected"]["checkpoint_sha256"],
                "recipe_sha256": exported["recipe_sha256"], "contract_hash": context["contract"]["contract_hash"],
                "normalization_sha256": fitted["files"]["normalization"]["sha256"]}
    lock = {"format": "vcm-mixed-cuda-wave-parity-lock-v1", "files": files,
            "audio": deepcopy(selected_lock["audio"]), "cohorts": deepcopy(selected_lock["cohorts"]),
            "bindings": bindings, "selected": deepcopy(context["selected"]), "thresholds": THRESHOLDS,
            "neural_calls": 0, "first_pass_accuracy_recomputed": False, "weights_fitted": False,
            "calibration_fitted": False, "enrollment_rebuilt": False, "final_tts_accessed": False,
            "first_pass_receipt": benchmark.binding(first_path), "first_pass_selection_lock": first["selection_lock"],
            "bundle": str(Path(bundle).resolve()), "mode": "Complete PCM-to-head/final-decision CUDA parity only"}
    return context, fitted, exported, selected_lock, data, predictions, lock


class CUDAReplay:
    """The only neural boundary; one actual CUDA model and profiled ORT session."""
    def __init__(self, context, bundle, output):
        import torch
        self.model, payload, self.gpu = finalize.load_selected_cuda(context)
        gpu.require_cuda_module(self.model)
        if self.gpu.get("device") != "cuda" or self.gpu.get("visible_device_count") != 1:
            raise ValueError("Exactly one actual visible CUDA device is required")
        self.contract = context["contract"]
        self.graph = Path(bundle).resolve() / "model.onnx"
        Path(output).mkdir(parents=True, exist_ok=False)
        self.session, self.ort = gpu.create_cuda_session(self.graph, evidence_dir=Path(output))
        self.frames, self.last_ort_called = [], False
        if any(value.is_floating_point() and value.dtype != torch.float32 for value in payload["weights"].values()):
            raise ValueError("All selected neural weights must remain FP32")

    def __call__(self, x):
        import torch
        self.last_ort_called = False
        features = torch.tensor(np.ascontiguousarray(x[None]), dtype=torch.float32, device="cuda")
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device="cuda")
        gpu.require_cuda_tensor(features)
        gpu.require_cuda_tensor(lengths)
        gpu.require_cuda_module(self.model)
        with torch.no_grad():
            masked = self.model(features, lengths)
            reference = gpu.cuda_head_arrays((masked["intent_logits"], masked["role_state_logits"].flatten(1),
                        masked["request_validity_logits"], torch.cat(list(masked["value_logits"].values()), dim=1),
                        masked["entity_embeddings"].flatten(1)), self.contract)
        gpu.require_cuda_module(self.model)
        self.frames.append(x.shape[1])
        self.last_ort_called = True
        actual = gpu.run_ort_cuda(self.session, features, self.contract)
        return reference, actual

    def finish(self):
        placement = gpu.finish_cuda_profile(self.session, self.graph, self.ort, frames=self.frames)
        if placement.get("status") == "verified_cuda_neural_placement":
            gpu.verify_placement_evidence(placement, graph=self.graph, frames=self.frames,
                                          provider_options=self.ort["provider_options"])
        return placement


def _safe_profile_failure(error):
    return {"status": "blocked_node_placement", "qualification": False,
            "CPU_neural_inference_performed": getattr(error, "cpu_neural", None),
            "reason": type(error).__name__ + ": " + str(error)}


def run(training, inputs, postfit, evaluation, bundle, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve prior waveform parity evidence; choose a new directory")
    context, fitted, exported, old_lock, data, first_pass, lock = _prepare(training, inputs, postfit, evaluation, bundle)
    output.mkdir(parents=True, exist_ok=False)
    lock_binding = freeze_parity_lock(lock, output / "parity-lock.json")
    decoder = {"contract": context["contract"], "memory": fitted["memory"], "policy": fitted["policy"],
               "catalog": {"entries": context["contract"]["catalog_entries"], "custom": context["contract"]["custom"]}}
    backend, setup_error = None, None
    try:
        backend = CUDAReplay(context, bundle, output / "ort")
    except Exception as error:
        setup_error = error
    def unavailable(x):
        raise RuntimeError("CUDA parity setup failed: " + type(setup_error).__name__ + ": " + str(setup_error))
    reports, records, frames = {}, {}, []
    for name, rule in benchmark.COHORTS.items():
        verify_parity_lock(lock_binding)
        destination = output / name
        destination.mkdir(exist_ok=False)
        results = []
        with (destination / "trial-results.jsonl").open("x", encoding="utf-8") as stream:
            for row in data[name]:
                result = run_trial(row, first_pass[name][row["id"]], old_lock["cohorts"][name]["audio_root"],
                                   context["normalization"], decoder, backend or unavailable)
                result.update(cohort=name, **replay_provenance(rule))
                results.append(result)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                if result["ort_called"]:
                    frames.append(result["frames"])
        summary = summarize_records(results, [row["id"] for row in data[name]])
        summary.update(**replay_provenance(rule))
        write_new(destination / "summary.json", summary)
        reports[name], records[name] = summary, results
        print(json.dumps({"completed_parity_cohort": name, **summary}, allow_nan=False), flush=True)
    try:
        placement = backend.finish() if backend is not None else _safe_profile_failure(setup_error)
    except Exception as error:
        placement = _safe_profile_failure(error)
    if backend is not None and list(backend.frames) != frames:
        placement = _safe_profile_failure(ValueError("ORT call/frame sequence differs from every preserved trial"))
    cpu_claim = cpu_neural_claim(placement)
    all_results = [row for name in benchmark.COHORTS for row in records[name]]
    # Original views can share names across cohorts; use qualified identities.
    total = summarize_records([{**row, "trial_identity": row["cohort"] + "/" + row["id"]} for row in all_results],
                              [name + "/" + row["id"] for name in benchmark.COHORTS for row in data[name]],
                              identity_field="trial_identity")
    verify_parity_lock(lock_binding)
    success = (total["parity_passed"] and total["compared_valid"] > 0 and cpu_claim is False
               and placement.get("status") == "verified_cuda_neural_placement")
    artifacts = deepcopy(lock["files"])
    _register(artifacts, "parity_lock", lock_binding)
    for path in sorted(output.rglob("*")):
        if path.is_file():
            _register(artifacts, "parity_output/" + str(path.relative_to(output)), path)
    receipt = {"format": "vcm-mixed-cuda-waveform-parity-v1", "status": "passed_cuda_waveform_parity" if success else "failed_cuda_waveform_parity",
               "parity_passed": success, "qualification": False, "selected": deepcopy(context["selected"]),
               "bindings": lock["bindings"], "thresholds": THRESHOLDS, "tolerance": TOLERANCE,
               "first_pass_receipt": lock["first_pass_receipt"], "first_pass_selection_lock": lock["first_pass_selection_lock"],
               "parity_lock": lock_binding, "cohorts": reports, "total": total, "neural_frame_sequence": frames,
               "gpu": deepcopy(backend.gpu) if backend else None, "ort": deepcopy(backend.ort) if backend else None,
               "placement": placement, "CPU_neural_inference_performed": cpu_claim,
               "files": artifacts, "first_pass_accuracy_recomputed": False, "first_pass_results_preserved": True,
               "weights_fitted": False, "calibration_fitted": False, "enrollment_rebuilt": False,
               "reference_labels_used_for_inference": False, "final_tts_accessed": False,
               "ncnn_evidence": False, "resident_service_evidence": False, "physical_pi_evidence": False,
               "scope": "Complete PCM-to-head/final-decision PyTorch CUDA vs ORT CUDA agreement only; capture faults stay accuracy failures"}
    write_new(output / "wave-parity-receipt.json", receipt)
    return receipt


def verify_waveform_receipt(binding, *, selected, recipe_sha256, contract_hash, first_pass_receipt, expected_ids):
    """Pure/hash consumer for optional card evidence; no model replay occurs."""
    path = benchmark.verify_file(binding, "actual waveform parity receipt")
    receipt = read_json(path)
    if (receipt.get("format") != "vcm-mixed-cuda-waveform-parity-v1" or receipt.get("qualification") is not False
            or receipt.get("selected") != selected or receipt.get("first_pass_receipt") != first_pass_receipt
            or receipt.get("thresholds") != THRESHOLDS or receipt.get("tolerance") != TOLERANCE
            or receipt.get("bindings", {}).get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or receipt.get("bindings", {}).get("recipe_sha256") != recipe_sha256
            or receipt.get("bindings", {}).get("contract_hash") != contract_hash
            or set(expected_ids) != set(benchmark.COHORTS)
            or any(len(expected_ids[name]) != rule["n"] for name, rule in benchmark.COHORTS.items())
            or any(receipt.get(key) is not False for key in (
                "first_pass_accuracy_recomputed", "weights_fitted", "calibration_fitted", "enrollment_rebuilt",
                "reference_labels_used_for_inference", "final_tts_accessed", "ncnn_evidence", "resident_service_evidence", "physical_pi_evidence"))
            or receipt.get("first_pass_results_preserved") is not True):
        raise ValueError("Incomplete, contaminated or differently selected waveform parity receipt")
    for name, item in receipt.get("files", {}).items():
        benchmark.verify_file(item, "waveform receipt artifact/" + name)
    lock = verify_parity_lock(receipt["parity_lock"])
    if lock.get("bindings") != receipt["bindings"] or lock.get("first_pass_receipt") != first_pass_receipt:
        raise ValueError("Waveform parity lock/candidate differs from the actual first pass")
    # Reconstruct the exact pre-call source map from independently bound first-pass
    # evidence, not whichever candidate paths a newly rebound parity lock supplies.
    original_path = benchmark.verify_file(first_pass_receipt, "original first-pass receipt")
    original_receipt = read_json(original_path)
    original_selection_binding = original_receipt.get("selection_lock")
    if (receipt.get("first_pass_selection_lock") != original_selection_binding
            or lock.get("first_pass_selection_lock") != original_selection_binding):
        raise ValueError("Waveform source lock changed the original first-pass selection")
    original_selection = read_json(benchmark.verify_file(original_selection_binding, "original first-pass selection"))
    original_files = original_selection.get("files", {})
    if not {"selection", "inputs"} <= set(original_files):
        raise ValueError("Missing original selected training/input source bindings")
    _, _, _, _, original_data, original_predictions, expected_lock = _prepare(
        Path(original_files["selection"]["path"]).resolve().parent, original_files["inputs"]["path"],
        original_receipt["postfit_receipt"]["path"], original_path, lock["bundle"])
    if not _same_json(lock, expected_lock):
        raise ValueError("Waveform pre-call lock differs from the complete original selected source/graph map")
    required_artifacts = {**expected_lock["files"], "parity_lock": receipt["parity_lock"],
                          "parity_output/parity-lock.json": receipt["parity_lock"]}
    if any(receipt.get("files", {}).get(name) != item for name, item in required_artifacts.items()):
        raise ValueError("Waveform receipt omitted or substituted immutable artifact/source bindings")
    if Path(receipt["parity_lock"]["path"]).resolve() != path.parent / "parity-lock.json":
        raise ValueError("Waveform parity lock points outside its actual preserved output directory")
    all_records, frames = [], []
    for name, rule in benchmark.COHORTS.items():
        file_key = "parity_output/" + name + "/trial-results.jsonl"
        item = receipt.get("files", {}).get(file_key)
        if item is None or Path(item["path"]).resolve() != path.parent / name / "trial-results.jsonl":
            raise ValueError("Missing complete bound waveform parity trial artifact")
        records = read_jsonl(item["path"])
        if ([row["id"] for row in records] != expected_ids[name]
                or [row["id"] for row in original_data[name]] != expected_ids[name]):
            raise ValueError("Waveform parity dropped, reordered or substituted original trials")
        for row, source in zip(records, original_data[name]):
            validate_trial(row, source_row=source, first_pass_prediction=original_predictions[name][source["id"]])
            if row.get("cohort") != name or any(row.get(key) is not value for key, value in replay_provenance(rule).items()):
                raise ValueError("Waveform replay changed regression/original-evaluation provenance accounting")
        summary = summarize_records(records, expected_ids[name])
        summary.update(**replay_provenance(rule))
        if summary != receipt.get("cohorts", {}).get(name):
            raise ValueError("Waveform parity receipt disagrees with preserved trial accounting")
        summary_item = receipt.get("files", {}).get("parity_output/" + name + "/summary.json")
        if (summary_item is None or Path(summary_item["path"]).resolve() != path.parent / name / "summary.json"
                or read_json(summary_item["path"]) != summary):
            raise ValueError("Missing or inconsistent complete waveform cohort summary artifact")
        frames.extend(row["frames"] for row in records if row.get("ort_called") is True)
        all_records.extend({**row, "trial_identity": name + "/" + row["id"]} for row in records)
    total = summarize_records(all_records, [name + "/" + identifier for name in benchmark.COHORTS for identifier in expected_ids[name]],
                              identity_field="trial_identity")
    if total != receipt.get("total") or frames != receipt.get("neural_frame_sequence"):
        raise ValueError("Changed full waveform parity/fault/frame accounting")
    placement = receipt.get("placement", {})
    claim = cpu_neural_claim(placement)
    if claim != receipt.get("CPU_neural_inference_performed"):
        raise ValueError("CPU-neural claim differs from actual placement proof")
    success = False
    if placement.get("status") == "verified_cuda_neural_placement":
        graph = Path(lock["bundle"]) / "model.onnx"
        gpu.verify_placement_evidence(placement, graph=graph, frames=frames,
                                      provider_options=receipt["ort"]["provider_options"])
        portable = {(Path(item["path"]).resolve(), item["sha256"]) for item in receipt["files"].values()}
        if any((Path(placement[key]["path"]).resolve(), placement[key]["sha256"]) not in portable
               for key in ("original_graph", "optimized_graph", "profile", "report")):
            raise ValueError("Waveform receipt omitted immutable actual operator-placement artifacts")
        ort, actual_gpu = receipt.get("ort") or {}, receipt.get("gpu") or {}
        if (actual_gpu.get("device") != "cuda" or actual_gpu.get("visible_device_count") != 1
                or not ort.get("actual_providers") or ort["actual_providers"][0] != "CUDAExecutionProvider"
                or ort.get("whole_session_fallback_disabled") is not True
                or not str(ort.get("input_device", "")).startswith("cuda")
                or not str(ort.get("output_device", "")).startswith("cuda")
                or ort.get("original_graph") != benchmark.binding(graph)
                or ort.get("optimized_graph_binding") != placement["optimized_graph"]):
            raise ValueError("Waveform parity lacks actual frozen CUDA session/input/output binding evidence")
        success = total["parity_passed"] and total["compared_valid"] > 0 and claim is False
    if (receipt.get("parity_passed") is not success
            or receipt.get("status") != ("passed_cuda_waveform_parity" if success else "failed_cuda_waveform_parity")):
        raise ValueError("Waveform parity gate claim lacks actual complete CUDA evidence")
    return receipt


def compact_summary(receipt):
    """Summarize a verified receipt; raw graph/profile/trials remain file artifacts."""
    placement, ort = receipt.get("placement") or {}, receipt.get("ort") or {}
    result = {key: deepcopy(receipt.get(key)) for key in (
        "format", "status", "parity_passed", "qualification", "selected", "bindings", "thresholds", "tolerance",
        "first_pass_receipt", "first_pass_selection_lock", "parity_lock", "cohorts", "total", "gpu",
        "CPU_neural_inference_performed", "first_pass_accuracy_recomputed", "first_pass_results_preserved",
        "weights_fitted", "calibration_fitted", "enrollment_rebuilt", "reference_labels_used_for_inference",
        "final_tts_accessed", "ncnn_evidence", "resident_service_evidence", "physical_pi_evidence")}
    result["ort"] = {key: deepcopy(ort.get(key)) for key in (
        "actual_providers", "provider_options", "whole_session_fallback_disabled", "input_device", "output_device",
        "CUDA_VISIBLE_DEVICES", "original_graph", "optimized_graph_binding")}
    result["placement"] = {"status": placement.get("status"), "reason": placement.get("reason"),
        "CPU_neural_inference_performed": placement.get("CPU_neural_inference_performed"),
        "profiled_calls": len(placement.get("probes", [])), "cuda_node_count": len(placement.get("cuda_nodes", [])),
        "cpu_metadata_node_count": len(placement.get("cpu_metadata_nodes", []))}
    result["placement_evidence"] = {key: deepcopy(placement[key]) for key in (
        "original_graph", "optimized_graph", "profile", "report") if key in placement}
    result["scope"] = "Complete PCM-to-head/final-decision CUDA backend agreement; every replay is regression; no label accuracy is measured"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "inputs", "postfit", "evaluation", "bundle", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = run(args.training, args.inputs, args.postfit, args.evaluation, args.bundle, args.output)
    print(json.dumps({"status": result["status"], "total": result["total"],
                      "receipt": str(args.output / "wave-parity-receipt.json")}, allow_nan=False))
    return 0 if result["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
