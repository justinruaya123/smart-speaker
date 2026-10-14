"""Assemble a semi-final card from bound, completed selected-candidate evidence.

No training, inference, export, calibration, enrollment, source mutation, fixture
promotion or physical measurements occur here. Outputs require a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from final import render_card
from scripts.test import vcm_mixed_benchmark as benchmark
from scripts.test import vcm_mixed_export_cuda as export_cuda
from scripts.test.vcm_mixed_finalize import GRID, SOURCE_FILES, verify_training
from scripts.test.vcm_mixed_benchmark import verify_postfit
from scripts.test.speaker_demo_suite import file_digest

FINAL_TTS_VOICES = ("af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck")
REQUIRED_DATA = {"preparation_audit", "fresh_human_audit", "default_benchmark_audit",
                 "default_benchmark_source_lock", "final_tts_generation_audit",
                 "final_tts_reference_accounting", "data_duration_receipt", "normalization_recheck",
                 "training_launch", "training_process"}
RUNTIME_SOURCES = ("vcm/frontend.py", "scripts/runtime/pcv_schema.py",
                   "scripts/models/vcm_intent_slots.py", "scripts/runtime/vcm_intent_slots_bindings.py",
                   "scripts/runtime/vcm_intent_slots_contract.py", "scripts/runtime/vcm_intent_slots_entities.py",
                   "scripts/runtime/vcm_intent_slots_decode.py", "scripts/runtime/vcm_intent_slots_runtime.py",
                   "scripts/runtime/pcv_benchmark_adapter.py")
HEADS = ("intent", "role_states", "validity", "values", "entities")
PROBE_FRAMES = (1, 3, 17, 32, 101, 231)
EXPORT_SOURCES = set(RUNTIME_SOURCES) | set(SOURCE_FILES) | {
    "scripts/train/vcm_intent_slots_export.py", "scripts/test/vcm_mixed_export_cuda.py"}


def strict_json(payload):
    def reject(value):
        raise ValueError("Nonfinite JSON evidence: " + value)

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON evidence field: " + key)
            result[key] = value
        return result

    return json.loads(payload, parse_constant=reject, parse_float=finite, object_pairs_hook=unique)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class Evidence:
    """Snapshot exact bytes; all source bindings are rechecked before publishing."""
    def __init__(self):
        self.sources, self.payloads = {}, {}

    def capture(self, value):
        item = value if isinstance(value, dict) else {"path": str(value)}
        path = Path(item["path"]).resolve()
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if "sha256" in item and (not _sha(item["sha256"]) or digest != item["sha256"]):
            raise ValueError("Missing or changed evidence hash: " + str(path))
        if str(path) in self.sources and self.sources[str(path)]["sha256"] != digest:
            raise ValueError("Evidence changed during assembly: " + str(path))
        target = f"evidence/{digest}/{path.name}"
        result = {"path": target, "sha256": digest, "bytes": len(payload), "source_path": str(path)}
        self.sources[str(path)] = result
        self.payloads[target] = payload
        return deepcopy(result)

    def json(self, value):
        ref = self.capture(value)
        return strict_json(self.payloads[ref["path"]])

    def jsonl(self, value):
        ref = self.capture(value)
        rows = [strict_json(line) for line in self.payloads[ref["path"]].splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Expected JSONL evidence objects")
        return rows

    def verify_unchanged(self):
        for source, item in self.sources.items():
            if file_digest(Path(source)) != item["sha256"]:
                raise ValueError("Evidence changed before final output: " + source)

    def write(self, output):
        output = Path(output)
        for name, payload in sorted(self.payloads.items()):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            if file_digest(path) != hashlib.sha256(payload).hexdigest():
                raise ValueError("Preserved evidence copy hash mismatch")


def validate_metrics(metrics, records, *, expected_n):
    """Reject altered outcomes/denominators using saved predictions only."""
    if len(records) != expected_n or len({row.get("id") for row in records}) != expected_n:
        raise ValueError("Incomplete or duplicate trial records")
    trials, predictions = [], []
    for record in records:
        row, prediction = record.get("source_row"), record.get("prediction")
        if not isinstance(row, dict) or row.get("id") != record.get("id"):
            raise ValueError("Changed trial source identity")
        actual = benchmark.record_result(row, prediction)
        if actual["outcome"] != record.get("outcome"):
            raise ValueError("Changed trial outcome")
        if actual["verbatim"] != record.get("verbatim"):
            raise ValueError("Changed trial verbatim")
        if prediction is not None and prediction.get("id") != row["id"]:
            raise ValueError("Changed trial prediction identity")
        trials.append(row)
        if prediction is not None:
            predictions.append(prediction)
    computed = benchmark.summarize(trials, predictions)
    for key, actual in computed.items():
        if key not in metrics or _canonical(metrics[key]) != _canonical(actual):
            raise ValueError("Saved " + key + " metrics disagree with exact trial accounting")


def group_misses(cohorts):
    groups = {}
    for cohort, records in cohorts.items():
        for row in records:
            if row["outcome"] in ("correct", "correct_rejection", "ambiguous_reference"):
                continue
            expected = row["source_row"].get("command")
            key = (_canonical(row["verbatim"]), _canonical(expected), row["outcome"])
            item = groups.setdefault(key, {"verbatim": row["verbatim"], "expected": deepcopy(expected),
                                          "outcome": row["outcome"], "count": 0,
                                          "cohorts": {}, "trial_ids": []})
            item["count"] += 1
            item["cohorts"][cohort] = item["cohorts"].get(cohort, 0) + 1
            item["trial_ids"].append({"cohort": cohort, "id": row["id"]})
    return [groups[key] for key in sorted(groups)]


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid nonnegative evidence count: " + name)
    return value


def tts_summary(audit, accounting):
    for key in ("plan_sha256", "manifest_sha256"):
        if not _sha(audit.get(key)) or audit[key] != accounting.get(key):
            raise ValueError("Final TTS plan/manifest binding mismatch")
    if (audit.get("fit_selection_calibration_enrollment_use") is not False
            or audit.get("VCM_results_accessed") is not False
            or accounting.get("VCM_results_accessed") is not False
            or audit.get("TTS_accuracy") is not None
            or audit.get("every_slot_correctness") is not None
            or accounting.get("inference_metrics") is not None
            or audit.get("takes_regenerated") is not False):
        raise ValueError("Final TTS cannot supply unsealed scores or fitting evidence")
    attempted, generated, faults = [_count(audit.get(key), "TTS/" + key)
                                    for key in ("attempted", "generated", "generation_faults")]
    voices, refs = audit.get("per_voice", {}), accounting.get("per_voice", {})
    if (set(voices) != set(FINAL_TTS_VOICES) or set(refs) != set(FINAL_TTS_VOICES)
            or attempted <= 0 or generated + faults != attempted
            or accounting.get("unique_audio") != attempted):
        raise ValueError("Incomplete final TTS take accounting")
    per_voice = {}
    for voice in FINAL_TTS_VOICES:
        take, ref = voices[voice], refs[voice]
        ta, tg, tf = [_count(take.get(key), "TTS/" + key)
                      for key in ("attempted", "generated", "generation_fault")]
        rn, rp, rneg, rf = [_count(ref.get(key, 0), "TTS reference/" + key)
                            for key in ("references", "supported", "negative", "generation_fault_references")]
        if tg + tf != ta or rp + rneg != rn or rf > rn:
            raise ValueError("Inconsistent final TTS voice/reference accounting")
        per_voice[voice] = {"attempted": ta, "generated": tg, "generation_faults": tf,
                            "case_references": rn, "supported_reference_n": rp,
                            "negative_reference_n": rneg, "generation_fault_references": rf,
                            "complete_command_accuracy": None, "strict_accuracy": None,
                            "every_slot_accuracy": None, "false_accept_rate": None,
                            "negative_n": None, "false_accepts": None, "faults": None, "missing": None}
    if (sum(row["attempted"] for row in per_voice.values()) != attempted
            or sum(row["generated"] for row in per_voice.values()) != generated
            or sum(row["generation_faults"] for row in per_voice.values()) != faults
            or sum(row["case_references"] for row in per_voice.values()) != accounting.get("case_references")):
        raise ValueError("Final TTS total/reference accounting mismatch")
    return {"status": audit.get("gate_status", "unqualified_generation_only"),
            "attempted": attempted, "generated": generated, "generation_faults": faults,
            "case_references": accounting["case_references"],
            "generation_fault_references": sum(row["generation_fault_references"] for row in per_voice.values()),
            "per_voice": per_voice, "per_speed": deepcopy(accounting.get("per_speed")),
            "per_track": deepcopy(accounting.get("per_track")), "speeds": [.9, 1.0, 1.1],
            "complete_command_accuracy": None, "every_slot_correctness": None,
            "false_accept_rate": None, "recognition_faults": None, "recognition_missing": None,
            "independent_listening_performed": audit.get("independent_listening_performed") is True,
            "supplemental_qualified_lock": deepcopy(audit.get("supplemental_qualified_lock")),
            "VCM_results_accessed": False,
            "final_tts_used_for_fit_selection_calibration_enrollment": False,
            "generation_fault_counts_are_not_recognition_errors": True,
            "generation_audit": deepcopy(audit), "reference_accounting": deepcopy(accounting)}


def verify_duration(receipt, inputs, views, *, inputs_sha256):
    if receipt.get("inputs_sha256") != inputs_sha256:
        raise ValueError("Changed duration receipt inputs binding")
    if set(receipt.get("splits", {})) != set(views):
        raise ValueError("Incomplete duration receipt split accounting")
    for split, rows in views.items():
        observed = receipt["splits"][split]
        if (observed.get("view_sha256") != inputs["views"][split]["sha256"]
                or observed.get("utterances") != len(rows)):
            raise ValueError("Changed duration receipt view/count")
        by_kind = observed.get("by_kind", {})
        if set(by_kind) != {row["kind"] for row in rows}:
            raise ValueError("Changed duration receipt human/synthetic accounting")
        for kind, supplied in by_kind.items():
            data = [row for row in rows if row["kind"] == kind]
            samples = sum(_count(row.get("samples"), "samples") for row in data)
            if (any(row.get("sample_rate") != 16000 for row in data)
                    or supplied.get("utterances") != len(data) or supplied.get("samples") != samples
                    or supplied.get("declared_speaker_ids") != len({row["speaker_id"] for row in data})
                    or type(supplied.get("hours")) not in (int, float)
                    or not math.isclose(supplied["hours"], samples / 16000 / 3600, abs_tol=1e-12)):
                raise ValueError("Changed duration receipt samples/hours/declared identity accounting")


def verify_calibration_provenance(views, policy, receipt, observations):
    for split in ("train", "dev", "calibration", "enrollment"):
        if any(row.get("voice_id") in FINAL_TTS_VOICES
               or row.get("speaker_id") in {"Kokoro-82M/" + voice for voice in FINAL_TTS_VOICES}
               for row in views[split]):
            raise ValueError("Sealed final TTS voice appeared in fitting/calibration/enrollment data")
    rows = views["calibration"]
    counts = {"human": sum(row["kind"] == "human" for row in rows),
              "synthetic": sum(row["kind"] == "synthetic" for row in rows),
              "reference_derived_silver": sum(row.get("supervision_status") == "reference_derived_silver" for row in rows)}
    if (not rows or policy.get("calibration_counts") != counts
            or policy.get("calibration_manifest_rows") != len(rows)
            or receipt.get("calibration_rows") != len(rows)
            or receipt.get("reference_derived_silver_rows") != counts["reference_derived_silver"]
            or policy.get("calibration_people") != sorted({row["speaker_id"] for row in rows if row["kind"] == "human"})
            or policy.get("calibration_voices") != sorted({row["voice_id"] for row in rows if row["kind"] == "synthetic"})
            or policy.get("qualification") is not False
            or policy.get("thresholds") != {"intent": .95, "slot": .90}):
        raise ValueError("Changed human/synthetic calibration counts or thresholds")
    expected = {row["id"]: row for row in rows}
    if (len(expected) != len(rows) or len(observations) != len(rows)
            or len({row.get("id") for row in observations}) != len(rows)
            or any(row.get("source_row") != expected.get(row.get("id")) for row in observations)):
        raise ValueError("Changed full calibration source-row provenance")


def architecture_summary(contract, parameters, frontend):
    if (len(contract["roles"]) != 15 or len(contract["entity_roles"]) != 5
            or len(contract["intent_order"]) != 19 or parameters.get("total") != 280818
            or len(contract["role_states"]) != 4 or len(contract["request_validity_states"]) != 4):
        raise ValueError("Settled semantic head architecture changed")
    return {"family": "IntentSlotsVCM complete-utterance temporal CNN", "parameters": deepcopy(parameters),
            "frontend": deepcopy(frontend), "stem_channels": 128, "stem_kernel": 5, "stem_stride": 2,
            "residual_blocks": 8, "depthwise_kernel": 5, "dilations": [1, 2, 4, 8, 1, 2, 4, 8],
            "semantic_queries": 16, "intents": 19, "roles": deepcopy(contract["roles"]),
            "role_states": deepcopy(contract["role_states"]), "validity_states": deepcopy(contract["request_validity_states"]),
            "finite_value_heads": {role: len(values) for role, values in contract["finite_values"].items()},
            "entity_roles": deepcopy(contract["entity_roles"]), "entity_embedding_dimensions": 64,
            "causal_model": False, "ASR": False, "CTC": False, "KV_cache": False,
            "typed_outputs": True, "new_hardware_limit": False}


def default_audit_summary(raw, artifact):
    """Represent overlap dimensions as data, keeping the raw audit unchanged."""
    result = {key: deepcopy(value) for key, value in raw.items() if key not in (
        "default_vs_fit", "fresh_vs_fit", "current_default_vs_original196_regression",
        "current_default_vs_original_test_regression")}
    comparisons = []
    for group in ("default_vs_fit", "fresh_vs_fit"):
        for split, dimensions in raw.get(group, {}).items():
            comparisons.extend({"comparison": group + "/" + split, "dimension": name,
                                **deepcopy(value)} for name, value in dimensions.items())
    for group in ("current_default_vs_original196_regression", "current_default_vs_original_test_regression"):
        comparisons.extend({"comparison": group, "dimension": name, **deepcopy(value)}
                           for name, value in raw.get(group, {}).items())
    result.update(comparisons=comparisons, original_receipt=deepcopy(artifact))
    return result


def _safe_name(name):
    path = Path(name)
    if not isinstance(name, str) or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe bundle file/source name")
    return path


def verify_bundle(bundle, context, postfit, evidence, *, root=ROOT):
    bundle = Path(bundle).resolve()
    recipe_ref = evidence.capture(bundle / "recipe.json")
    recipe = evidence.json(bundle / "recipe.json")
    selected, contract = context["selected"], context["contract"]
    if (recipe.get("format") != "vcm-intent-slots-v1"
            or recipe.get("checkpoint_status") != "mixed_silver_unqualified_candidate"
            or recipe.get("checkpoint_source_sha256") != selected["checkpoint_sha256"]
            or recipe.get("encoder_id") != selected["checkpoint_sha256"]
            or recipe.get("contract_hash") != contract["contract_hash"]
            or recipe.get("head_order") != list(HEADS)
            or recipe.get("value_offsets") != contract["value_offsets"]
            or recipe.get("parameters", {}).get("total") != 280818
            or recipe.get("acoustic_qualified") is not False):
        raise ValueError("Export recipe does not bind the actual selected architecture/candidate")
    required = {"reference.pt", "contract.json", "catalog.json", "normalization.json",
                "entities.json", "policy.json", "model.onnx"}
    if not required <= set(recipe.get("files", {})) or not set(RUNTIME_SOURCES) <= set(recipe.get("source_bindings", {})):
        raise ValueError("Incomplete selected-candidate bundle")
    for name, digest in recipe["files"].items():
        evidence.capture({"path": str(bundle / _safe_name(name)), "sha256": digest})
    for name, digest in recipe["source_bindings"].items():
        evidence.capture({"path": str(Path(root) / _safe_name(name)), "sha256": digest})
    expected_files = {"reference.pt": selected["checkpoint_sha256"],
                      "normalization.json": postfit["files"]["normalization"]["sha256"],
                      "entities.json": postfit["files"]["entities"]["sha256"],
                      "policy.json": postfit["files"]["calibration"]["sha256"]}
    if any(recipe["files"][key] != digest for key, digest in expected_files.items()):
        raise ValueError("Bundle changed exact checkpoint/normalization/enrollment/calibration file identity")
    if evidence.json(bundle / "contract.json") != contract:
        raise ValueError("Bundle contract differs from the selected training contract")
    if evidence.json(bundle / "catalog.json") != {"entries": contract["catalog_entries"], "custom": contract["custom"]}:
        raise ValueError("Bundle catalog differs from the unchanged selected contract catalog")
    if recipe.get("graph_bytes") != (bundle / "model.onnx").stat().st_size:
        raise ValueError("Bundle graph byte count differs from the actual exported file")
    exported = evidence.json(bundle / "cuda-export-receipt.json")
    if (exported.get("status") != "unqualified_cuda_onnx_export"
            or exported.get("qualification") is not False or exported.get("selected") != selected
            or exported.get("calibration_id") != postfit["policy"]["calibration_id"]
            or exported.get("postfit_receipt") != benchmark.binding(postfit["receipt_path"])
            or exported.get("parity_tolerance") != 2e-5
            or exported.get("parity_inputs") != "fixed_numeric_features_only"
            or exported.get("dynamic_frames") is not True or exported.get("opset") != 17
            or any(exported.get(key) is not False for key in (
                "CPU_neural_inference_performed", "ncnn_invoked", "reference_labels_used", "test_accessed", "final_tts_accessed"))):
        raise ValueError("Missing actual separately bound CUDA export/parity receipt")
    if set(exported.get("files", {})) != set(recipe["files"]) | {"recipe.json"}:
        raise ValueError("Export receipt omits selected bundle artifacts")
    for name, binding in exported["files"].items():
        if Path(binding["path"]).resolve() != bundle / _safe_name(name):
            raise ValueError("Export artifact binding points outside the selected bundle")
        evidence.capture(binding)
    if not EXPORT_SOURCES <= set(exported.get("source_bindings", {})):
        raise ValueError("Incomplete exact export implementation source bindings")
    for name, binding in exported["source_bindings"].items():
        expected = Path(root) / _safe_name(name)
        if not isinstance(binding, dict) or Path(binding["path"]).resolve() != expected.resolve():
            raise ValueError("Export implementation source binding points outside the actual source path")
        evidence.capture(binding)
    bindings = exported.get("postfit_bindings", [])
    for binding in bindings.values() if isinstance(bindings, dict) else bindings:
        evidence.capture(binding)
    numerical = exported.get("numerical", {})
    gpu, ort = numerical.get("gpu", {}), numerical.get("ort", {})
    placement = numerical.get("placement", {})
    export_cuda.verify_placement_evidence(placement, graph=bundle / "model.onnx",
                                          frames=PROBE_FRAMES, provider_options=ort.get("provider_options"))
    if exported.get("placement_evidence") != {key: placement[key] for key in (
            "original_graph", "optimized_graph", "profile", "report")}:
        raise ValueError("Export receipt differs from the complete actual CUDA node placement evidence")
    for item in exported["placement_evidence"].values():
        evidence.capture(item)
    if (gpu.get("device") != "cuda" or gpu.get("visible_device_count") != 1
            or not ort.get("actual_providers") or ort["actual_providers"][0] != "CUDAExecutionProvider"
            or ort.get("whole_session_fallback_disabled") is not True
            or not str(ort.get("input_device", "")).startswith("cuda")
            or not str(ort.get("output_device", "")).startswith("cuda")
            or numerical.get("parameters") != recipe["parameters"]
            or numerical.get("checkpoint_status") != recipe["checkpoint_status"]):
        raise ValueError("Parity receipt does not prove actual CUDA tensor execution")
    for group in ("rewrite_parity", "onnx_parity"):
        probes = numerical.get(group, [])
        if [row.get("frames") for row in probes] != list(PROBE_FRAMES):
            raise ValueError("Incomplete fixed-frame CUDA parity probes")
        for probe in probes:
            errors = probe.get("head_errors", {})
            if (set(errors) != set(HEADS)
                    or any(type(error) not in (int, float) or not math.isfinite(error)
                           or not 0 <= error <= 2e-5 for error in errors.values())
                    or probe.get("max_error") != max(errors.values())):
                raise ValueError("Nonfinite or failed CUDA graph-head parity")
            if group == "onnx_parity":
                signatures = [{key: probe.get(side, {}).get(key) for key in ("accepted", "command", "reason")}
                              for side in ("pytorch_decision", "onnx_decision")]
                if probe.get("decision_equal") is not True or _canonical(signatures[0]) != _canonical(signatures[1]):
                    raise ValueError("CUDA final decision parity failed")
    return {"recipe": recipe, "recipe_sha256": recipe_ref["sha256"],
            "cuda_receipt": exported, "cuda_parity_passed": True,
            "ncnn_status": exported.get("ncnn_status"),
            "CPU_resident_parity_status": exported.get("CPU_resident_parity_status"),
            "waveform_parity_status": exported.get("waveform_parity_status")}


def verify_normalization_recheck(context, receipt, training_rows):
    expected = {"status": "exact_train_only_normalization_verified", "fit_split": "train",
                "inputs_sha256": file_digest(context["inputs_path"]),
                "training_manifest_sha256": context["inputs"]["views"]["train"]["sha256"],
                "features_receipt_sha256": file_digest(context["training"] / "features/features.json"),
                "normalization_sha256": file_digest(context["normalization_path"]),
                "verified_training_feature_arrays": len(training_rows),
                "training_frames": context["normalization"]["frames"]}
    if (any(receipt.get(key) != value for key, value in expected.items())
            or any(type(receipt.get(key)) not in (int, float) or receipt[key] != 0
                   for key in ("mean_max_error", "std_max_error"))
            or any(receipt.get(key) is not False for key in (
                "development_calibration_enrollment_evaluation_final_TTS_frames_used",
                "actual_neural_inference_performed"))):
        raise ValueError("Incomplete, changed or contaminated train-only normalization recheck")


def _training_section(context, evidence, launch, process):
    runs = {}
    for result, (arm, seed) in zip(context["grid"], GRID):
        folder = context["training"] / f"{arm}-{seed}"
        identity = evidence.json(folder / "run-identity.json")
        history = evidence.json(folder / "history.json")
        if identity.get("parameters", {}).get("total") != 280818:
            raise ValueError("Actual run parameter count differs from settled architecture")
        if type(result.get("seconds")) not in (int, float) or not math.isfinite(result["seconds"]) or result["seconds"] <= 0:
            raise ValueError("Missing actual training wall time")
        denominators = result.get("supervision_denominators")
        if not isinstance(denominators, dict) or not denominators:
            raise ValueError("Missing actual training supervision denominators")
        for key, count in denominators.items():
            _count(count, "supervision/" + key)
        curves = []
        for curve in history:
            observed = {key: deepcopy(value) for key, value in curve.items()
                        if key not in ("human_development", "synthetic_development")}
            for key in ("human_development", "synthetic_development"):
                observed[key] = {field: deepcopy(curve[key].get(field)) for field in (
                    "all", "macro_person_recall", "worst_person_recall", "macro_person_population")}
            curves.append(observed)
        runs[f"{arm}-{seed}"] = {"identity": identity, "selected_epoch_result": deepcopy(result),
                                  "learning_curves": curves,
                                  "history_artifact": evidence.capture(folder / "history.json")}
    actual = process.get("training_process", {})
    if (launch.get("inputs") != benchmark.binding(context["inputs_path"])
            or type(launch.get("GPU_count")) is not int or launch["GPU_count"] != 1
            or not isinstance(launch.get("hostname"), str) or not launch["hostname"]
            or not isinstance(launch.get("GPU_UUID"), str) or not launch["GPU_UUID"].startswith("GPU-")
            or actual.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != launch.get("GPU_UUID")
            or actual.get("environment", {}).get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or type(actual.get("pid")) is not int or actual["pid"] <= 0):
        raise ValueError("Incomplete or changed actual training process/launch evidence")
    for name, digest in launch.get("source_files", {}).items():
        evidence.capture({"path": str(ROOT / _safe_name(name)), "sha256": digest})
    evidence.capture(launch["source_archive"])
    value_counts = {role: sum(row["supervision_denominators"].get(role, 0) for row in context["grid"])
                    for role in context["contract"]["finite_values"]}
    return {"selected": deepcopy(context["selected"]), "completed_candidates": 6,
            "updates_per_candidate": 2880, "grid_optimizer_updates": context["grid_updates"],
            "selection": "human development only", "normalization": deepcopy(context["normalization"]),
            "normalization_file": evidence.capture(context["normalization_path"]),
            "protocol": deepcopy(context["protocol"]["training"]), "runs": runs,
            "wall_seconds_all_completed_runs": sum(row["seconds"] for row in context["grid"]),
            "node": launch["hostname"], "GPU_UUID": launch["GPU_UUID"],
            "process_at_launch": deepcopy(process), "launch": deepcopy(launch),
            "node_status": "Actual bound process observation; completed updates verified separately",
            "finite_value_supervision_counts": value_counts,
            "zero_supervision_value_heads": [role for role, count in value_counts.items() if count == 0],
            "final_tts_used": False, "independent_acoustic_supervision": False}


def _enrollment_section(context, postfit, views, evidence):
    source_rows, examples = views["enrollment"], postfit["memory"]["examples"]
    source = {row["id"]: row for row in source_rows}
    if len(source_rows) != 33 or len(source) != 33 or len(examples) != 33:
        raise ValueError("Selected enrollment must retain all 33 original packs/examples")
    seen, groups = set(), defaultdict(list)
    fields = ("id", "audio", "sha256", "pcm_sha256", "speaker_id", "voice_id", "kind",
              "source", "source_revision", "split", "entity_id", "entity_role", "pair_id", "recording_group")
    for row in examples:
        original = source.get(row.get("id"))
        if (original is None or row["id"] in seen
                or any(row.get(key) != original.get(key) for key in fields)
                or row.get("kind") != "synthetic" or row.get("role") != row.get("entity_role")):
            raise ValueError("Selected enrollment differs from exact source file/PCM/person/group provenance")
        seen.add(row["id"])
        groups[row["entity_role"] + "/" + row["entity_id"]].append({key: deepcopy(row.get(key)) for key in fields})
    if len(groups) != 11 or any(len(rows) != 3 for rows in groups.values()):
        raise ValueError("Incomplete 11-name three-example enrollment")
    return {"kind": "synthetic", "named_entities": 11, "examples": 33,
            "voice_packs": sorted({row["voice_id"] for row in source_rows}),
            "entity_examples": dict(sorted(groups.items())),
            "encoder_binding": deepcopy(postfit["memory"]["encoder_binding"]),
            "memory_id": postfit["memory"]["memory_id"],
            "memory_file": evidence.capture(postfit["files"]["entities"]),
            "independent_listening_verified": False}


def required_evaluation_files():
    names = {"run-identity.json", "selection-before-test.json"}
    for cohort in benchmark.COHORTS:
        names.update(cohort + "/" + filename for filename in (
            "metrics.json", "trial-results.jsonl", "misses.jsonl", "predictions.jsonl",
            "negative-results.jsonl", "ambiguous-results.jsonl"))
    names.update("mandatory_defaults202/" + filename for filename in (
        "official-metrics.json", "official-records.jsonl", "benchmark-log.jsonl"))
    return sorted(names)


def verify_evaluation_files(receipt, root, evidence):
    files = receipt.get("files", {})
    if not set(required_evaluation_files()) <= set(files):
        raise ValueError("Missing original receipt binding for a complete evaluation artifact")
    root = Path(root).resolve()
    for name, item in files.items():
        if Path(item["path"]).resolve() != root / _safe_name(name):
            raise ValueError("Evaluation artifact points outside actual receipt directory")
        evidence.capture(item)


def verify_official_outputs(metrics, records, *, files, official_spec, evidence):
    expected = files["mandatory_defaults202/official-metrics.json"]
    if metrics.get("official_metrics") != expected:
        raise ValueError("Official report differs from its frozen evaluation receipt binding")
    official = evidence.json(expected)
    scorer = benchmark.load_official(official_spec)
    source_rows = [record["source_row"] for record in records]
    predictions = [record["prediction"] for record in records if record["prediction"] is not None]
    actual, actual_records = benchmark.official_scores(source_rows, predictions, scorer)
    if _canonical(official) != _canonical(actual):
        raise ValueError("Official scoring disagrees with saved predictions and unchanged publisher references")
    observed_records = evidence.jsonl(files["mandatory_defaults202/official-records.jsonl"])
    if _canonical(observed_records) != _canonical(actual_records):
        raise ValueError("Official parsed records disagree with actual adapter outputs and original references")
    actual_logs = [record["log_record"] for record in actual_records if record["log_record"] is not None]
    observed_logs = evidence.jsonl(files["mandatory_defaults202/benchmark-log.jsonl"])
    if _canonical(observed_logs) != _canonical(actual_logs):
        raise ValueError("Official benchmark logs differ from saved actual final-application predictions")
    evidence.capture(official_spec["snapshot"])
    for item in scorer["files"].values():
        evidence.capture(item)
    return official


def _evaluation_section(path, context, postfit, data, evidence):
    root = Path(path).resolve()
    receipt_path = root / "benchmark-receipt.json" if root.is_dir() else root
    root = receipt_path.parent
    receipt = evidence.json(receipt_path)
    if (receipt.get("format") != "vcm-mixed-offline-benchmark-receipt-v2"
            or receipt.get("status") != "unqualified_offline_replay_complete"
            or receipt.get("selected") != context["selected"] or receipt.get("qualification") is not False
            or receipt.get("thresholds") != {"intent": .95, "slot": .90}
            or receipt.get("evaluation_order") != list(benchmark.COHORTS)
            or set(receipt.get("cohorts", {})) != set(benchmark.COHORTS)
            or receipt.get("completed_trial_rows") != sum(rule["n"] for rule in benchmark.COHORTS.values())
            or receipt.get("postfit_receipt") != benchmark.binding(postfit["receipt_path"])
            or receipt.get("mode") != "CUDA offline command-audio replay"
            or receipt.get("gpu", {}).get("device") != "cuda"
            or receipt.get("gpu", {}).get("visible_device_count") != 1
            or any(receipt.get(key) is not False for key in (
                "CPU_neural_inference_performed", "weights_fitted", "calibration_fitted", "enrollment_rebuilt",
                "final_tts_accessed", "resident_service_evidence", "physical_pi_evidence"))):
        raise ValueError("Incomplete or contaminated selected-candidate evaluation receipt")
    verify_evaluation_files(receipt, root, evidence)
    lock_ref = receipt["selection_lock"]
    lock = benchmark.verify_selection(lock_ref["path"], expected_sha256=lock_ref["sha256"])
    evidence.capture(lock_ref)
    spec = evidence.json(lock["files"]["evaluation_spec"])
    if (lock.get("selected") != context["selected"]
            or lock.get("contract_hash") != context["contract"]["contract_hash"]
            or lock.get("calibration_id") != postfit["policy"]["calibration_id"]
            or lock.get("memory_id") != postfit["memory"]["memory_id"]):
        raise ValueError("Changed selected-candidate pre-test lock")
    results, all_records, successful = {}, {}, 0
    for name, rule in benchmark.COHORTS.items():
        item = receipt["cohorts"][name]
        if any(item.get(key) != receipt["files"][name + "/" + filename]
               for key, filename in (("metrics", "metrics.json"), ("misses", "misses.jsonl"),
                                     ("predictions", "predictions.jsonl"))):
            raise ValueError("Cohort metric/trial artifacts differ from their original receipt bindings")
        metrics = evidence.json(item["metrics"])
        records = evidence.jsonl(root / name / "trial-results.jsonl")
        validate_metrics(metrics, records, expected_n=rule["n"])
        source_view = lock["cohorts"][name]["view"]
        frozen_rows = evidence.jsonl(source_view)
        if [row["source_row"] for row in records] != frozen_rows:
            raise ValueError("Trial source rows differ from unchanged frozen references")
        if "input_view" in rule and source_view != context["inputs"]["views"][rule["input_view"]]:
            raise ValueError("Evaluation cohort changed its pre-fit input view")
        if rule.get("official") and source_view["sha256"] != data["default_benchmark_source_lock"]["current202"]["sha256"]:
            raise ValueError("Mandatory default202 changed its frozen source/reference lock")
        if (item.get("fresh") is not (not rule["regression"])
                or item.get("regression") is not rule["regression"]
                or metrics.get("fresh") is not (not rule["regression"])
                or metrics.get("regression") is not rule["regression"]
                or metrics.get("selected_checkpoint_sha256") != context["selected"]["checkpoint_sha256"]
                or metrics.get("calibration_id") != postfit["policy"]["calibration_id"]
                or metrics.get("source_view") != source_view
                or item.get("all") != metrics["all"] or item.get("kinds") != metrics["kinds"]):
            raise ValueError("Changed fresh/regression or selected metrics accounting")
        wanted_misses = [record for record in records if record["outcome"] not in (
            "correct", "correct_rejection", "ambiguous_reference")]
        if evidence.jsonl(item["misses"]) != wanted_misses:
            raise ValueError("Dropped or changed missed trial verbatims")
        actual_predictions = [record["prediction"] for record in records if record["prediction"] is not None]
        if evidence.jsonl(item["predictions"]) != actual_predictions:
            raise ValueError("Saved predictions differ from exact trial records")
        successful += sum(record["prediction"] is not None and not record["prediction"].get("fault") for record in records)
        result = deepcopy(metrics)
        result["artifacts"] = {key: evidence.capture(root / name / filename) for key, filename in (
            ("trial_results", "trial-results.jsonl"), ("misses", "misses.jsonl"),
            ("negative_results", "negative-results.jsonl"), ("ambiguous_results", "ambiguous-results.jsonl"),
            ("predictions", "predictions.jsonl"), ("metrics", "metrics.json"))}
        if rule.get("official"):
            official = verify_official_outputs(metrics, records, files=receipt["files"],
                                               official_spec=spec["official"], evidence=evidence)
            result["official_metrics"] = official
            result["artifacts"]["official_metrics"] = evidence.capture(metrics["official_metrics"])
            result["artifacts"]["official_records"] = evidence.capture(root / name / "official-records.jsonl")
            result["artifacts"]["benchmark_log"] = evidence.capture(root / name / "benchmark-log.jsonl")
        results[name], all_records[name] = result, records
    if receipt.get("successful_neural_replays") != successful:
        raise ValueError("Actual neural replay/fault accounting mismatch")
    return results, all_records, receipt


def qualification_gates(*, preparation, calibration, tts, export, training=None):
    def gate(status, reason, **actual):
        return {"status": status, "reason": reason, **actual}

    spans = preparation.get("reviewed_acoustic_spans", 0)
    listened = preparation.get("acoustic_listening_verified", 0)
    missing_intents = preparation.get("missing_intents", {}).get("fresh_human", [])
    missing_slots = calibration.get("missing_slot_calibration", [])
    zero_heads = training.get("zero_supervision_value_heads", []) if training else None
    gates = {
        "completed_grid_and_development_selection": gate("passed", "All six completed candidates and human-development selection verified"),
        "train_only_normalization": gate("passed", "Exact serialized training-only normalization is shared by checkpoint, calibration and bundle"),
        "selected_enrollment_bindings": gate("passed", "All 11 names and 33 synthetic enrollment examples bind the selected checkpoint file identity"),
        "separate_calibration": gate("passed", "Separate bound calibration; final TTS excluded; thresholds remain 0.95 intent / 0.90 supplied slot"),
        "slot_calibration_coverage": gate("failed" if missing_slots else "passed", "Missing cells retain rejection behavior", missing_groups=deepcopy(missing_slots)),
        "reviewed_acoustic_supervision": gate("failed" if not spans or not listened else "passed", "Independent listening and reviewed acoustic spans are required", reviewed_spans=spans, listened_rows=listened),
        "value_head_supervision": gate("blocked" if zero_heads is None else "failed" if zero_heads else "passed", "Actual finite-value loss denominators must be nonzero", zero_supervision_heads=deepcopy(zero_heads)),
        "physical_person_alias_verification": gate("blocked", "Declared-source identity, recording-group and file/PCM separation is checked; physical cross-source alias equivalence remains unverified"),
        "fresh_human_scope": gate("failed" if missing_intents else "passed", "Fresh human recordings must cover the complete intent contract", missing_intents=deepcopy(missing_intents)),
        "held_offline_replay": gate("passed", "Complete CUDA command-audio replay with original regression and fresh cohorts separately accounted"),
        "cuda_pytorch_onnx_numeric_parity": gate("passed" if export.get("cuda_parity_passed") else "blocked", "Actual fixed-feature CUDA numeric and final-decision parity; waveform parity is separate"),
        "waveform_cross_engine_parity": gate("blocked", "CUDA exporter records fixed numeric features; complete PCM-to-command cross-engine parity is unmeasured"),
        "ncnn_parity": gate("blocked", "CPU-only inference hold remains unanswered; no selected ncnn replay receipt", export_status=export.get("ncnn_status")),
        "resident_service": gate("blocked", "CPU-only inference hold remains unanswered; actual resident service is unmeasured"),
        "owner_paired_recordings": gate("blocked", "No completed selected-candidate paired owner recording receipt"),
        "final_tts_generation_qa": gate("failed" if tts["generation_faults"] else "passed", "Generation failures retained separately from recognition errors", generation_faults=tts["generation_faults"], generation_fault_references=tts["generation_fault_references"]),
        "final_tts_listening_and_lock": gate("passed" if tts["independent_listening_performed"] and tts["supplemental_qualified_lock"] else "blocked", "Independent pre-result listening and supplemental lock are incomplete"),
        "final_tts_replay": gate("blocked", "Final six-voice ncnn replay has not run; all recognition metrics remain unmeasured"),
    }
    gates["accuracy_ready"] = all(value["status"] == "passed" for value in gates.values())
    return gates


def _cell(value):
    text = value if isinstance(value, str) else _canonical(value)
    return text.replace("|", "\\|").replace("\r", "&#13;").replace("\n", "<br>")


def _table(columns, rows):
    return ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows], ""]


def experiment_report(card):
    lines = ["# VCM mixed final v2 comprehensive experiment report", "", "**Status: " + card["status"] + "**", "",
             "This report describes the actual selected candidate. Human results, historical regression, synthetic generation accounting and physical Pi evidence remain separate.", "",
             "## Selected architecture and bindings", "",
             "The selected IntentSlotsVCM has 280,818 parameters. Mono 16 kHz command audio becomes 40 log-mel channels using a 400-sample window, 160-sample hop and 512-point FFT. Training-only normalization precedes a width-128, kernel-5, stride-2 convolution and eight depthwise residual blocks with dilations 1, 2, 4, 8 repeated twice. Symmetric temporal convolutions and 16 learned semantic queries pool the complete utterance.", "",
             "The pooled heads predict 19 intent logits, four request-validity states, four states for each of 15 roles, the finite typed values, and five 64-dimensional entity embeddings. Separately fitted calibration and enrollment use the exact selected checkpoint and serialized normalization. The settled decoder applies 0.95 intent and 0.90 supplied-slot gates to the preserved grammar and catalog. Wake, owner and physical microphone qualification remain separate measured gates.", "",
             "```mermaid", "flowchart LR", '  A["Mono 16 kHz WAV"] --> B["40 log-mel channels"]',
             '  B --> C["Train-only normalization"]', '  C --> D["128-channel temporal CNN"]',
             '  D --> E["16 whole-utterance semantic queries"]', '  E --> F["Intent, role states, validity, values, entities"]',
             '  F --> G["Bound calibration and entity memory"]', '  G --> H["Preserved typed decoder: 0.95 / 0.90"]',
             '  H --> I["Accepted exact command or rejection"]', "```", "",
             "The full source-bound architecture is recorded below.", "", _canonical(card["architecture"]), "",
             "```json", json.dumps(card["bindings"], indent=2, sort_keys=True), "```", "",
             "## Complete training grid", ""]
    runs = card["training"]["runs"]
    lines += _table(["Run", "Selected epoch", "Updates", "Wall seconds", "Actual GPU", "Visible GPU", "Human development key", "Supervision denominators"], [
        [name, run["selected_epoch_result"]["epoch"], run["selected_epoch_result"]["optimizer_updates"],
         run["selected_epoch_result"]["seconds"], run["identity"].get("device"),
         run["identity"].get("CUDA_VISIBLE_DEVICES"), run["selected_epoch_result"]["development_key"],
         run["selected_epoch_result"]["supervision_denominators"]] for name, run in runs.items()])
    for name, run in runs.items():
        lines += ["### " + name + " learning curve", ""]
        lines += _table(["Epoch", "Loss", "Gradient min / max", "Actual parameter delta L2", "Cumulative updates", "Human selection key", "Seconds"], [
            [row["epoch"], row["loss"], [row["gradient_norm_min"], row["gradient_norm_max"]],
             row["actual_parameter_delta_l2"], row["cumulative_updates"], row["selection_key"], row["seconds"]]
            for row in run["learning_curves"]])
    lines += ["## Exact data duration and declared identities", ""]
    lines += _table(["Split", "Kind", "Utterances", "Samples", "Hours", "Declared source IDs / packs"], [
        [split, kind, item["utterances"], item["samples"], item["hours"], item["declared_speaker_ids"]]
        for split, result in card["data"]["duration_receipt"]["splits"].items()
        for kind, item in result["by_kind"].items()])
    fresh, defaults = card["data"]["fresh_human_audit"], card["data"]["default_benchmark_audit"]
    lines += ["Declared source identities are not verified physical-person counts. Train-only normalization, missing human scope, review limits, and exact source bindings are retained in card.json and preserved receipts.", "",
              f"The fresh human suite has {fresh['exact_reference_recordings']} exact-reference recordings from {fresh['fresh_declared_people']} declared source identities, covering {len(fresh['intents'])} intent labels including OOV. The {fresh['unmatched_reference_recordings']} available recordings without a matching preserved reference remain untested.", "",
              f"Current mandatory defaults contain {defaults['current_default_rows']} clips ({defaults['human_rows']} human / {defaults['synthetic_rows']} synthetic). Their {defaults['original_overlap_typed_reference_matches']} PCM/reference overlaps with original-196 are retained regression trials; none are counted as fresh human recordings.", "",
              "## Separate calibration and selected enrollment", "", "```json",
              json.dumps(card["calibration"], indent=2, sort_keys=True), "```", ""]
    lines += _table(["Named entity", "Examples", "Synthetic packs", "Exact original WAV identities"], [
        [name, len(rows), sorted({row["voice_id"] for row in rows}), [row["sha256"] for row in rows]]
        for name, rows in card["enrollment"]["entity_examples"].items()])
    lines += ["## Command-audio results", "",
              "Complete-command accuracy is accepted exact positives divided by supported references. Strict accuracy also includes correct negative rejections; faults and missing outputs fail. Ambiguous references retain their separate denominator. Every supplied slot requires the correct intent, typed equality and exactly the expected fields.", ""]
    rows = []
    for name, result in card["benchmarks"].items():
        if name == "final_tts":
            continue
        for kind, metric in [("all", result["all"]), *result["kinds"].items()]:
            rows.append([name, kind, result["fresh"], result["regression"], metric["trials"],
                         metric["supported_n"], metric["negative_n"], metric["ambiguous_n"],
                         metric["complete_command_accuracy"], metric["strict_accuracy"],
                         metric["joint_supplied_slots"]["accuracy"],
                         metric["joint_supplied_slots"]["with_supplied_slots_n"],
                         metric["joint_supplied_slots"]["with_supplied_slots_accuracy"], metric["false_accepts"],
                         metric["faults"], metric["missing"]])
    lines += _table(["Cohort", "Kind", "Fresh", "Regression", "Trials", "Supported", "Negative", "Ambiguous", "Exact positive accuracy", "Strict accuracy", "Every-slot joint accuracy", "Slot-bearing commands", "Slot-bearing every-slot accuracy", "False accepts", "Faults", "Missing"], rows)
    for name, result in card["benchmarks"].items():
        if name == "final_tts":
            continue
        lines += ["### " + name + " strict slot accounting", ""]
        lines += _table(["Supplied field", "Correct", "Denominator", "Typed accuracy"], [
            [field, item["correct"], item["n"], item["accuracy"]]
            for field, item in result["all"]["every_slot"].items()])
        if "official_metrics" in result:
            lines += ["### Unmodified pinned official scoring", "",
                      "Pinned publisher classification and conditional-slot metrics are preserved below. Its no-response mapping is reported separately from strict fault/missing accounting.", "",
                      "```json", json.dumps(result["official_metrics"], indent=2, sort_keys=True), "```", ""]
    lines += ["## Final synthetic evaluation set", "",
              "No final-set recognition accuracy is measured. Generation faults are technical failures, preserved before VCM results; case-reference multiplicity is separate from unique audio takes.", ""]
    lines += _table(["Voice", "Attempted takes", "Valid takes", "Generation faults", "Case references", "Fault references", "Exact accuracy", "Every-slot accuracy", "False accept rate"], [
        [voice, item["attempted"], item["generated"], item["generation_faults"], item["case_references"],
         item["generation_fault_references"], item["complete_command_accuracy"], item["every_slot_accuracy"], item["false_accept_rate"]]
        for voice, item in card["benchmarks"]["final_tts"]["per_voice"].items()])
    lines += ["## Failed and blocked gates", ""]
    lines += _table(["Gate", "Status", "Actual evidence / missing requirement"], [
        [name, value["status"], {key: child for key, child in value.items() if key != "status"}]
        for name, value in card["gates"].items() if isinstance(value, dict)])
    lines += ["## Submission metadata", "", "```json",
              json.dumps(card.get("submission", {}), indent=2, sort_keys=True), "```", "",
              "Source dataset revisions, licenses and access receipts remain bound in the data evidence. No dataset DOI, public checkpoint release or matched trained-baseline gain is claimed by this study.", ""]
    lines += ["Physical Raspberry Pi hardware, latency, RTF, memory, CPU, wake, thermal and power fields are deferred. The frozen semi-final baseline and deterministic Pi updater use the selected checkpoint/recipe/contract bindings.", "",
              "## Every missed verbatim", "",
              "This compact table groups identical verbatim, expected command and outcome. Every trial ID, complete source row, prediction and missed-record JSONL remains preserved byte-for-byte in the evidence manifest. Missed trials are never removed from denominators.", ""]
    lines += _table(["Verbatim (unchanged)", "Expected typed command", "Outcome", "Count", "Cohorts"], [
        [row["verbatim"], row["expected"], row["outcome"], row["count"], row["cohorts"]]
        for row in card["missed_verbatims"]])
    lines += ["Full grouped table: [missed-verbatims.csv](missed-verbatims.csv). Generation-fault verbatims remain a separate bound artifact.", "",
              "## Preserved evidence", "", "All receipt, curve, JSONL, source and bundle files in evidence-manifest.json are copies of their original bytes, with SHA-256 bindings. No model or source file was changed by card assembly.", ""]
    return "\n".join(lines)


def _write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def assemble(training, inputs, postfit, evaluation, bundle, data_receipt, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve prior cards/reports; choose a new output directory")
    evidence = Evidence()
    context = verify_training(training, inputs)
    fitted = verify_postfit(context, postfit)
    for item in context["evidence_bindings"].values():
        evidence.capture(item)
    evidence.capture(fitted["receipt_path"])
    for group in (fitted["files"], fitted["sources"]):
        for item in group.values():
            evidence.capture(item)
    wrapper = evidence.json(data_receipt)
    if wrapper.get("format") != "vcm-final-data-receipt-v1" or not REQUIRED_DATA <= set(wrapper.get("files", {})):
        raise ValueError("Missing explicitly bound final data receipt wrapper")
    data = {name: evidence.json(item) for name, item in wrapper["files"].items()}
    for item in wrapper.get("artifacts", []):
        evidence.capture(item)
    views = {name: evidence.jsonl(item) for name, item in context["inputs"]["views"].items()}
    preparation = data["preparation_audit"]
    if (preparation.get("bindings") != context["inputs"]["views"] or preparation.get("fit_ready") is not True
            or preparation.get("experiment_id") != "vcm-mixed-final-v2"):
        raise ValueError("Preparation audit differs from frozen final fit views")
    if (data["fresh_human_audit"].get("manifest") != context["inputs"]["views"]["fresh_human"]
            or data["fresh_human_audit"].get("contract_hash") != context["contract"]["contract_hash"]):
        raise ValueError("Fresh human source audit changed its exact bindings")
    defaults = data["default_benchmark_audit"]
    default_lock = data["default_benchmark_source_lock"]
    if (defaults.get("source_lock_sha256") != wrapper["files"]["default_benchmark_source_lock"]["sha256"]
            or defaults.get("fit_separation_ready") is not True
            or defaults.get("reference_preservation_ready") is not True
            or default_lock.get("inputs", {}).get("sha256") != file_digest(Path(inputs))):
        raise ValueError("Default202 source/reference/separation receipt is incomplete or changed")
    verify_duration(data["data_duration_receipt"], context["inputs"], views,
                    inputs_sha256=file_digest(Path(inputs)))
    verify_normalization_recheck(context, data["normalization_recheck"], views["train"])
    evidence.capture(context["training"] / "features/features.json")
    verify_calibration_provenance(views, fitted["policy"], fitted["receipt"],
                                 evidence.json(fitted["files"]["observations"]))
    exported = verify_bundle(bundle, context, fitted, evidence)
    tts = tts_summary(data["final_tts_generation_audit"], data["final_tts_reference_accounting"])
    fault_binding = data["final_tts_generation_audit"].get("fault_verbatims")
    if not isinstance(fault_binding, dict):
        raise ValueError("Final TTS generation faults require a preserved verbatim artifact")
    tts["generation_fault_verbatims"] = evidence.capture(fault_binding)
    benchmarks, records, replay_receipt = _evaluation_section(evaluation, context, fitted, data, evidence)
    benchmarks["final_tts"] = tts
    normalization = context["normalization"]
    if (normalization.get("fit_split") != "train" or normalization.get("frames", 0) <= 0
            or len(normalization.get("mean", [])) != 40 or len(normalization.get("std", [])) != 40
            or any(not math.isfinite(value) for value in normalization["mean"])
            or any(not math.isfinite(value) or value <= 0 for value in normalization["std"])):
        raise ValueError("Invalid actual train-only normalization")
    contract = context["contract"]
    from vcm.frontend import SPEC
    architecture = architecture_summary(contract, exported["recipe"]["parameters"], SPEC)
    calibration = {"counts": deepcopy(fitted["policy"]["calibration_counts"]),
                   "manifest_rows": fitted["policy"]["calibration_manifest_rows"],
                   "people": deepcopy(fitted["policy"]["calibration_people"]),
                   "voices": deepcopy(fitted["policy"]["calibration_voices"]),
                   "calibration_id": fitted["policy"]["calibration_id"],
                   "intent_temperature": fitted["policy"]["intent_temperature"],
                   "slot_calibrators": deepcopy(fitted["policy"]["slot_calibrators"]),
                   "missing_slot_calibration": deepcopy(fitted["policy"]["missing_slot_calibration"]),
                   "supervision_status": fitted["policy"]["supervision_status"], "qualification": False,
                   "gpu": deepcopy(fitted["receipt"]["gpu"]),
                   "thresholds": {"intent": .95, "slot": .90}, "binding": deepcopy(fitted["policy"]["binding"]),
                   "policy_file": evidence.capture(fitted["files"]["calibration"])}
    training_summary = _training_section(context, evidence, data["training_launch"], data["training_process"])
    gates = qualification_gates(preparation=preparation, calibration=calibration, tts=tts, export=exported,
                                training=training_summary)
    card = {"format": "vcm-model-card-v2", "title": "VCM mixed final v2 semi-final selected candidate",
            "status": "blocked / Pi-deferred", "qualification": False,
            "bindings": {"checkpoint_sha256": context["selected"]["checkpoint_sha256"],
                         "recipe_sha256": exported["recipe_sha256"], "contract_hash": contract["contract_hash"]},
            "selected_candidate": deepcopy(context["selected"]), "architecture": architecture,
            "training": training_summary,
            "data": {"duration_receipt": deepcopy(data["data_duration_receipt"]),
                     "normalization_recheck": deepcopy(data["normalization_recheck"]),
                     "preparation_audit": preparation, "fresh_human_audit": data["fresh_human_audit"],
                     "default_benchmark_audit": default_audit_summary(defaults, evidence.capture(wrapper["files"]["default_benchmark_audit"])),
                     "default_benchmark_source_lock": default_lock,
                     "source_identity_limits": context["protocol"]["source_identity_limits"]},
            "calibration": calibration, "enrollment": _enrollment_section(context, fitted, views, evidence),
            "benchmarks": benchmarks, "export": exported,
            "gates": gates, "failed_gates": [name + ": " + value["status"] + " — " + value["reason"]
                                              for name, value in gates.items() if isinstance(value, dict)
                                              and value["status"] in ("failed", "blocked")],
            "pi": {"status": "deferred", "results": None, "hardware": None, "latency": None,
                   "RTF": None, "RSS": None, "CPU": None, "wake": None, "thermal": None, "power": None},
            "missed_verbatims": group_misses(records),
            "submission": {"authenticated_repository_observation": deepcopy(data.get("repository_identity")),
                           "dataset_DOI": None, "dataset_DOI_status": "No DOI registered by this study",
                           "public_checkpoint_release": None,
                           "public_checkpoint_release_status": "Selected checkpoint is supplied in the Pi handoff; no public release created",
                           "matched_fully_trained_baseline_comparison": None,
                           "matched_baseline_status": "No matched trained-baseline comparison performed"},
            "receipts": {"postfit": deepcopy(fitted["receipt"]), "offline_evaluation": replay_receipt,
                         "data_wrapper": wrapper}, "installation_allowed": False, "provider_actions_allowed": False}
    card["assembly_sources"] = {"assembler": evidence.capture(Path(__file__)),
                                "renderer": evidence.capture(ROOT / "final/render_card.py")}
    render_card.validate_card(card)
    evidence.verify_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    evidence.write(output)
    manifest = {"format": "vcm-final-card-evidence-v1", "files": dict(sorted(evidence.sources.items())),
                "source_mutation": False, "inference_performed": False}
    _write_json_new(output / "evidence-manifest.json", manifest)
    card["evidence_manifest"] = {"path": "evidence-manifest.json", "sha256": file_digest(output / "evidence-manifest.json")}
    _write_json_new(output / "card.json", card)
    card_sha = file_digest(output / "card.json")
    with (output / "MODEL_CARD.md").open("x", encoding="utf-8") as stream:
        stream.write(render_card.render_markdown(card, card_sha))
    with (output / "EXPERIMENT_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(experiment_report(card))
    with (output / "missed-verbatims.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["verbatim", "expected", "outcome", "count", "cohorts", "trial_ids"])
        for row in card["missed_verbatims"]:
            writer.writerow([row["verbatim"], _canonical(row["expected"]), row["outcome"], row["count"],
                             _canonical(row["cohorts"]), _canonical(row["trial_ids"])])
    outputs = {path.name: file_digest(path) for path in sorted(output.iterdir()) if path.is_file()}
    _write_json_new(output / "assembly-receipt.json", {"format": "vcm-final-card-assembly-v1", "status": card["status"],
                    "bindings": card["bindings"], "files": outputs, "source_mutation": False, "inference_performed": False})
    return {"card": str(output / "card.json"), "card_sha256": card_sha,
            "report": str(output / "EXPERIMENT_REPORT.md"), "status": card["status"],
            "failed_gates": card["failed_gates"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("training", "inputs", "postfit", "evaluation", "bundle", "data-receipt", "output"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    result = assemble(args.training, args.inputs, args.postfit, args.evaluation, args.bundle, args.data_receipt, args.output)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
