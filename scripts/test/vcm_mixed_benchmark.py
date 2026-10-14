"""Frozen selected-candidate CUDA command-audio replay, with separate cohorts.

This evaluates weights already selected on human development and scores already
calibrated on a separate partition. It never fits, selects, calibrates, enrolls,
changes thresholds, installs a candidate or operates a resident/Pi service.
Final supplemental TTS is intentionally outside this evaluator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from scripts.runtime.pcv_schema import canonical, validate_command
from scripts.test.vcm_intent_slots_eval import _pairs, cluster_interval, summarize_trials, trial_tier
from scripts.train.vcm_intent_slots_data import rows, sha, write_new

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = {"intent": .95, "slot": .90}
COHORTS = {
    "fresh_human106": {"n": 106, "split": "fresh_human", "regression": False, "input_view": "fresh_human"},
    "original_test4418": {"n": 4418, "split": "test", "regression": True, "input_view": "test",
                          "sha256": "a4fdcedfc86efebb36414114887abdc4ee51eba102b0247740aef18d1ad15abc"},
    "original_holdout196": {"n": 196, "split": "holdout", "regression": True, "input_view": "holdout",
                           "sha256": "a51e720b97f7b31e3e2f025c21bf679006ad444fd2d363916d9a11aa0f488b5b"},
    "mandatory_defaults202": {"n": 202, "split": "holdout", "regression": True, "official": True},
}
OFFICIAL_REVISION = "ab39857cf73dd067d068fed388c321ab160dc191"
OFFICIAL_HASHES = {
    "vcmbench/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "vcmbench/pi.py": "772a73dffa5810645182d2a6a9c16a828bb25d4a84f2d9cb3bbf60993d76545f",
    "vcmbench/schema.py": "cb7cefcd893a2443b9a9b9e9b2756efa27d273f4f0086b94491ca3b304e7a006",
    "vcmbench/slots.py": "7fc483ad851699d09e948ed942a36141012f474333e2ce521e705acce7fa30ef",
    "vcmbench/metrics.py": "518ef4bc0646f93ad936d84ff2982bdd56691783e6892799e0728eff4e9410f6",
    "vcmbench/dataset.py": "17d74702d4b9230965c96fbf7dd9a5b717fea47cc3374adfc48e9c2225dd6ba1",
    "vcmbench/variations.csv": "6004b3b2b58137058c705b5769b0d094e05a5cff6ddcbed6de548b4a0105fcda",
}
REQUIRED_LOCK_FILES = {"selection", "grid", "inputs", "protocol", "contract", "checkpoint",
                       "normalization", "frontend", "calibration", "entities", "raw_entity_policy",
                       "postfit_receipt", "evaluation_spec"}
IDENTITY_FIELDS = ("sha256", "pcm_sha256", "speaker_id", "pair_id", "recording_group")
EVALUATOR_SOURCES = ("scripts/test/vcm_mixed_benchmark.py", "scripts/test/vcm_intent_slots_eval.py",
                     "scripts/test/vcm_intent_slots_diversity.py", "scripts/test/vcm_default_benchmark_data.py",
                     "scripts/runtime/pcv_benchmark_adapter.py", "scripts/runtime/vcm_intent_slots_runtime.py")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def verify_file(item, name="input"):
    if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
            or sha(item["path"]) != item["sha256"]):
        raise ValueError("Changed or incomplete frozen binding: " + name)
    return Path(item["path"]).resolve()


def json_finite(value):
    """Represent undefined official metrics as null, without changing scoring."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return value


def _accepted(prediction):
    if prediction is None or prediction.get("fault"):
        return None
    decision = prediction.get("final_application", {})
    return decision.get("command") if decision.get("accepted") is True else None


def _strict_slots(pairs):
    supplied, fields = {"n": 0, "correct": 0, "with_supplied_slots_n": 0,
                        "with_supplied_slots_correct": 0, "extra_slot_commands": 0}, {}
    for trial, prediction in pairs:
        expected = trial.get("command")
        if expected is None or trial.get("reference_status") == "ambiguous":
            continue
        actual = _accepted(prediction)
        wanted = expected["slots"]
        matching_intent = isinstance(actual, dict) and actual.get("intent") == expected["intent"]
        actual_slots = actual.get("slots", {}) if isinstance(actual, dict) else {}
        valid_slots = isinstance(actual_slots, dict)
        joint = bool(matching_intent and valid_slots and canonical(actual_slots) == canonical(wanted))
        supplied["n"] += 1
        supplied["correct"] += int(joint)
        supplied["with_supplied_slots_n"] += int(bool(wanted))
        supplied["with_supplied_slots_correct"] += int(bool(wanted) and joint)
        supplied["extra_slot_commands"] += int(actual is not None and valid_slots
                                                and bool(set(actual_slots) - set(wanted)))
        for field, value in wanted.items():
            item = fields.setdefault(field, {"n": 0, "correct": 0})
            item["n"] += 1
            item["correct"] += int(matching_intent and valid_slots and field in actual_slots
                                   and canonical(actual_slots[field]) == canonical(value))
    supplied["accuracy"] = supplied["correct"] / supplied["n"] if supplied["n"] else None
    supplied["with_supplied_slots_accuracy"] = (supplied["with_supplied_slots_correct"] /
        supplied["with_supplied_slots_n"] if supplied["with_supplied_slots_n"] else None)
    for item in fields.values():
        item["accuracy"] = item["correct"] / item["n"]
    return supplied, fields


def _enrich(metrics, pairs):
    metrics["complete_command_correct"] = metrics["correct"]
    metrics["complete_command_accuracy"] = metrics["recall"]
    scored = metrics["supported_n"] + metrics["negative_n"]
    metrics["strict_accuracy"] = ((metrics["correct"] + metrics["correct_rejections"]) / scored
                                  if scored else None)
    metrics["joint_supplied_slots"], metrics["every_slot"] = _strict_slots(pairs)
    metrics["slot_counts_scope"] = "Historical field accounting; every_slot requires typed equality and correct intent"


def summarize(trials, predictions):
    """Reuse frozen accounting, adding strict joint and typed per-field scores."""
    result = summarize_trials(trials, predictions)
    pairs = _pairs(trials, predictions)
    _enrich(result["all"], pairs)
    groups = {key: defaultdict(list) for key in ("tracks", "kinds", "intents", "people")}
    for pair in pairs:
        row = pair[0]
        groups["tracks"][row["kind"] + "/" + trial_tier(row)].append(pair)
        groups["kinds"][row["kind"]].append(pair)
        groups["intents"][(row.get("command") or {}).get("intent", "OOV")].append(pair)
        groups["people"][row["kind"] + "/" + row.get("speaker_id", "")].append(pair)
    for group, values in groups.items():
        for key, subset in values.items():
            if key in result[group]:
                _enrich(result[group][key], subset)
    result.update(people_cluster_recall_interval=cluster_interval(trials, predictions),
                  qualification=False, physical_pi_evidence=False, resident_service_evidence=False,
                  evidence="CUDA offline command-audio replay only when executed by this bound evaluator",
                  strict_accuracy_definition="Exact accepted positives plus successful negative rejections, divided by all unambiguous references; faults and missing outputs fail",
                  joint_slots_definition="Every supported reference, including empty-slot commands; correct intent, typed slot values and exactly the expected fields are required")
    return result


def outcome(trial, prediction):
    if prediction is None:
        return "missing"
    if prediction.get("fault"):
        return "fault"
    if trial.get("reference_status") == "ambiguous":
        return "ambiguous_reference"
    actual, expected = _accepted(prediction), trial.get("command")
    if expected is None:
        return "false_accept" if actual is not None else "correct_rejection"
    if actual is None:
        return "false_reject"
    return "correct" if canonical(actual) == canonical(expected) else "wrong_complete_command"


def record_result(trial, prediction):
    verbatim = trial.get("transcript")
    if verbatim is None:
        verbatim = trial.get("publisher_transcript", trial.get("publisher_metadata", {}).get("transcript"))
    return {"id": trial["id"], "verbatim": verbatim, "outcome": outcome(trial, prediction),
            "source_row": deepcopy(trial), "prediction": deepcopy(prediction)}


def misses(trials, predictions):
    return [record_result(row, prediction) for row, prediction in _pairs(trials, predictions)
            if outcome(row, prediction) not in ("correct", "correct_rejection", "ambiguous_reference")]


def audit_audio(row, audio_root):
    from scripts.test.speaker_demo_suite import read_audio

    pcm = read_audio(row, Path(audio_root))
    for field, actual in (("sample_rate", 16000), ("samples", len(pcm)), ("channels", 1)):
        if field in row and (type(row[field]) is not int or row[field] != actual):
            raise ValueError("Misdeclared audio " + field + ": " + row["id"])
    path = Path(row["audio"])
    if not path.is_absolute():
        path = Path(audio_root) / path
    return {"path": str(path.resolve()), "sha256": row["sha256"], "pcm_sha256": row["pcm_sha256"],
            "samples": len(pcm), "sample_rate": 16000, "channels": 1}


def check_separation(data, reserved, *, fresh):
    from scripts.test.vcm_default_benchmark_data import overlap

    checks = overlap(a_rows=data, b_rows=reserved)
    overlaps = [{"field": field, "ids": item["ids"]} for field, item in checks.items() if item["rows"]]
    groups = {row[key] for row in reserved for key in ("pair_id", "recording_group") if row.get(key)}
    group_ids = [row["id"] for row in data
                 if any(row.get(key) in groups for key in ("pair_id", "recording_group"))]
    if group_ids:
        overlaps.append({"field": "pair_or_recording_group", "ids": group_ids})
    if overlaps:
        raise ValueError(("Fresh/exposed" if fresh else "Evaluation/fitting") + " identity overlap: "
                         + json.dumps(overlaps[:10], ensure_ascii=False))
    return {"checked_rows": len(data), "overlaps": []}


def verify_postfit(context, postfit):
    from scripts.runtime.vcm_intent_slots_bindings import verify_evidence_bindings
    from scripts.test.vcm_mixed_finalize import SOURCE_FILES

    root = Path(postfit).resolve()
    receipt_path = root / "postfit-receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("selected") != context["selected"] or receipt.get("completed_grid_candidates") != 6
            or receipt.get("grid_updates") != 17280 or receipt.get("thresholds") != THRESHOLDS
            or receipt.get("weights_fitted") is not False or receipt.get("test_accessed") is not False
            or receipt.get("evaluation_views_read") is not False or receipt.get("final_tts_accessed") is not False
            or receipt.get("qualification") is not False or receipt.get("missing_calibration_intents")):
        raise ValueError("Incomplete, changed or contaminated selected-candidate postfit receipt")
    required = {"selection", "grid", "inputs", "protocol", "contract", "checkpoint", "normalization",
                "frontend", "entities", "raw_entity_policy", "observations", "calibration"}
    files = receipt.get("files", {})
    if not required <= set(files):
        raise ValueError("Missing frozen selected-candidate postfit files")
    for name, item in files.items():
        verify_file(item, "postfit/" + name)
    expected_paths = {"selection": context["training"] / "selection-before-calibration.json",
                      "grid": context["training"] / "grid.json", "inputs": context["inputs_path"],
                      "protocol": context["protocol_path"], "contract": context["inputs"]["contract"]["path"],
                      "checkpoint": context["selected"]["checkpoint"], "frontend": ROOT / "vcm/frontend.py",
                      "normalization": root / "normalization.json", "entities": root / "entities.json",
                      "raw_entity_policy": root / "raw-entity-policy.json",
                      "observations": root / "observations.json", "calibration": root / "calibration/calibration.json"}
    if any(Path(files[key]["path"]).resolve() != Path(path).resolve() for key, path in expected_paths.items()):
        raise ValueError("Postfit receipt does not bind the actual selected-candidate files")
    if files["normalization"]["sha256"] != sha(context["normalization_path"]):
        raise ValueError("Postfit serialized normalization differs from training")
    if read_json(files["normalization"]["path"]) != context["normalization"]:
        raise ValueError("Postfit normalization values differ from training")
    source_bindings = receipt.get("source_bindings", {})
    if not set(SOURCE_FILES) <= set(source_bindings):
        raise ValueError("Missing actual postfit inference/decoder source bindings")
    for name, item in source_bindings.items():
        if (Path(item["path"]).resolve() != (ROOT / name).resolve()
                or verify_file(item, "postfit source/" + name) != (ROOT / name).resolve()):
            raise ValueError("Changed actual postfit source binding: " + name)
    for run in receipt.get("training_runs", {}).values():
        for name, item in run.items():
            verify_file(item, "training run/" + name)
    if len(receipt.get("training_runs", {})) != 6:
        raise ValueError("Missing completed grid receipt bindings")
    policy = read_json(files["calibration"]["path"])
    memory = read_json(files["entities"]["path"])
    raw_policy = read_json(files["raw_entity_policy"]["path"])
    encoder_binding = {"encoder": context["selected"]["checkpoint_sha256"],
                       "frontend": files["frontend"]["sha256"], "normalization": files["normalization"]["sha256"]}
    if (policy.get("require_calibration") is not True or not policy.get("calibration_id")
            or receipt.get("calibration_id") != policy["calibration_id"]):
        raise ValueError("Actual separately frozen calibration is required")
    verify_evidence_bindings(context["contract"], policy, memory, **encoder_binding)
    verify_evidence_bindings(context["contract"], raw_policy, memory, **encoder_binding)
    cal_binding = policy.get("binding", {})
    expected_hashes = {"protocol": context["inputs"]["protocol"]["sha256"],
                       "manifest": context["inputs"]["views"]["calibration"]["sha256"],
                       "memory": files["entities"]["sha256"],
                       "raw_entity_policy": files["raw_entity_policy"]["sha256"]}
    if (any(cal_binding.get(key) != value for key, value in expected_hashes.items())
            or policy.get("entities") != raw_policy.get("entities")
            or cal_binding.get("entities") != raw_policy.get("entities")):
        raise ValueError("Calibration/entity memory provenance binding mismatch")
    return {"root": root, "receipt": receipt, "receipt_path": receipt_path, "files": files,
            "policy": policy, "memory": memory, "raw_policy": raw_policy, "sources": source_bindings}


def verify_official(spec):
    if (not isinstance(spec, dict) or not isinstance(spec.get("root"), str) or not spec["root"].strip()
            or not isinstance(spec.get("snapshot"), dict)):
        raise ValueError("Invalid pinned official benchmark source spec")
    root = Path(spec["root"]).resolve()
    if spec.get("revision") != OFFICIAL_REVISION:
        raise ValueError("Wrong pinned official benchmark revision")
    snapshot = read_json(verify_file(spec.get("snapshot"), "official snapshot"))
    if (snapshot.get("repository") != "airimonda/vcm-benchmark" or snapshot.get("revision") != OFFICIAL_REVISION
            or snapshot.get("defaults_unchanged") is not True):
        raise ValueError("Invalid official benchmark snapshot provenance")
    recorded = {row["path"]: row["sha256"] for row in snapshot.get("files", [])}
    for name, expected in OFFICIAL_HASHES.items():
        if recorded.get(name) != expected or sha(root / name) != expected:
            raise ValueError("Changed pinned official benchmark source: " + name)
    return {name: binding(root / name) for name in OFFICIAL_HASHES}


def load_official(spec):
    """Import unchanged pinned code under its own package, avoiding stale imports."""
    files = verify_official(spec)
    root = Path(spec["root"]).resolve()
    suffix = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    package = "_vcm_mixed_official_" + OFFICIAL_REVISION + "_" + suffix
    if package not in sys.modules:
        loader = importlib.util.spec_from_file_location(package, root / "vcmbench/__init__.py",
                                                       submodule_search_locations=[str(root / "vcmbench")])
        module = importlib.util.module_from_spec(loader)
        sys.modules[package] = module
        loader.loader.exec_module(module)
    modules = {name: importlib.import_module(package + "." + name) for name in ("pi", "schema", "slots", "metrics", "dataset")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != root / "vcmbench" / (name + ".py"):
            raise ValueError("Official scorer import resolved to a different source")
    verify_official(spec)
    return {**modules, "files": files, "revision": OFFICIAL_REVISION}


def publisher_reference(row):
    keys = ("publisher_intent", "publisher_slot", "publisher_variation", "publisher_transcript")
    if not all(key in row for key in keys):
        raise ValueError("Missing unchanged raw publisher references: " + row["id"])
    if not all(isinstance(row[key], str) for key in keys):
        raise ValueError("Non-string publisher reference: " + row["id"])
    return {key: row[key] for key in keys}


def official_reference(row, official):
    """Validate loader labels against actual pinned code; retain raw labels too."""
    import numpy as np

    raw = publisher_reference(row)
    metadata = row.get("publisher_metadata")
    if metadata is None:
        # Permits small serializer/scorer software fixtures. Real evaluation
        # preflight requires the complete immutable publisher metadata object.
        metadata = {"command": raw["publisher_intent"], "slot_value": raw["publisher_slot"],
                    "variation": raw["publisher_variation"], "transcript": raw["publisher_transcript"]}
    if not isinstance(metadata, dict):
        raise ValueError("Missing publisher metadata for pinned official loader")
    clip = official["dataset"]._row_to_clip(0, deepcopy(metadata), np.empty(0, dtype=np.float32))
    reference = {"intent": clip.intent, "slot": clip.slot_value,
                 "variation": clip.variation, "transcript": clip.transcript}
    for key, value in reference.items():
        if "official_" + key in row and row["official_" + key] != value:
            raise ValueError("Changed pinned official loader reference: " + row["id"])
    for raw_key, source_key in (("publisher_intent", "command"), ("publisher_slot", "slot_value"),
                                ("publisher_variation", "variation"), ("publisher_transcript", "transcript")):
        if source_key in metadata and raw[raw_key] != metadata[source_key]:
            raise ValueError("Publisher reference differs from immutable source metadata: " + row["id"])
    return reference


def official_scores(trials, predictions, official):
    from dataclasses import asdict
    from scripts.runtime.pcv_benchmark_adapter import log_record

    parser = official["pi"].LineParser()
    schema, slots, metrics = official["schema"], official["slots"], official["metrics"]
    aliases = schema.build_alias_table()
    truth_intents, pred_intents, truth_commands, pred_commands, distances, records = [], [], [], [], [], []
    faults = missing = parsed_commands = typed_rejections = 0
    for index, (row, prediction) in enumerate(_pairs(trials, predictions)):
        reference = publisher_reference(row)
        loader_reference = official_reference(row, official)
        state = outcome(row, prediction)
        missing += int(prediction is None)
        faults += int(prediction is not None and bool(prediction.get("fault")))
        record = None
        if prediction is not None and not prediction.get("fault"):
            record = log_record(prediction, decision="final_application")
        elif prediction is not None:
            record = log_record({"final_application": {"reason": "capture_fault"}}, decision="final_application")
        event = parser.parse(json.dumps(record, allow_nan=False), float(index)) if record is not None else None
        if record is not None and not prediction.get("fault") and (event is None or event.kind != "command"):
            raise ValueError("Pinned LineParser failed to parse actual canonical adapter output: " + row["id"])
        if prediction is not None and prediction.get("fault") and event is not None:
            raise ValueError("Capture fault unexpectedly became an official command")
        if event is None:
            intent, slot, known = schema.NONE, "", True
        else:
            parsed_commands += 1
            intent, slot, known = schema.normalize_prediction(event.intent, event.slot, aliases)
        if not known or intent not in schema.INTENTS and intent not in (schema.OOS, schema.NONE, ""):
            intent = "OTHER:" + intent
        target = loader_reference["intent"]
        distance = slots.slot_distance(target, loader_reference["slot"], slot) if target in schema.SLOTTED else None
        exact = intent == target and (distance is None or distance["exact"])
        truth_intents.append(metrics.intent_label(target))
        pred_intents.append(metrics.intent_label(intent))
        truth_commands.append(schema.REJECT if target == schema.OOS else loader_reference["variation"])
        pred_commands.append(metrics.command_label(intent, slot,
                             loader_reference["variation"] if exact else None, exact,
                             event.variation if event is not None else ""))
        if target == intent and distance is not None:
            distances.append({"intent": target, "slot": distance})
        typed_rejections += int(state == "correct_rejection")
        records.append({"id": row["id"], **reference, "official_reference": loader_reference,
                        "outcome": state, "log_record": record,
                        "parsed": asdict(event) if event is not None else None,
                        "normalized_prediction": {"intent": intent, "slot": slot, "known": known},
                        "slot_distance": distance, "publisher_exact": bool(exact)})
    return json_finite({"revision": official["revision"], "source_bindings": official["files"],
        "intent": metrics.classification_report(truth_intents, pred_intents),
        "command": metrics.classification_report(truth_commands, pred_commands),
        "conditional_slots": metrics.slot_report(distances), "parsed_commands": parsed_commands,
        "faults": faults, "missing": missing, "successful_typed_rejections": typed_rejections,
        "reference": "Unchanged publisher metadata, transformed by the pinned official loader and scored by pinned official code; raw publisher fields remain preserved",
        "missing_fault_accounting": "Official classification maps no response to reject; strict typed scoring separately counts faults/missing as failures",
        "physical_pi_evidence": False, "resident_service_evidence": False}), records


def prepare_evaluation(context, spec):
    """Audit full views and PCM before creating the selection-before-test lock."""
    if (not isinstance(spec, dict) or spec.get("format") != "vcm-mixed-offline-evaluation-v2"
            or not isinstance(spec.get("cohorts"), dict) or set(spec["cohorts"]) != set(COHORTS)):
        raise ValueError("All four separately bound complete evaluation cohorts are required")
    official_files = verify_official(spec.get("official"))
    fitting = []
    for split in ("train", "dev", "calibration", "enrollment"):
        fitting.extend(rows(verify_file(context["inputs"]["views"][split], "fitting view/" + split)))
    data, cohort_bindings, audio, sources = {}, {}, {}, {"official/" + k: v for k, v in official_files.items()}
    sources["official snapshot"] = spec["official"]["snapshot"]
    for name, rule in COHORTS.items():
        entry = spec["cohorts"][name]
        if not isinstance(entry, dict):
            raise ValueError("Invalid evaluation cohort spec: " + name)
        if not isinstance(entry.get("audio_root"), str) or not entry["audio_root"].strip():
            raise ValueError("Missing or invalid cohort audio_root: " + name)
        receipts = entry.get("source_receipts")
        if not isinstance(receipts, list) or not receipts:
            raise ValueError("Missing frozen source metadata receipts: " + name)
        for index, item in enumerate(receipts):
            verify_file(item, "source receipt/" + name)
            sources[name + "/" + str(index)] = item
        path = verify_file(entry.get("view"), "evaluation view/" + name)
        if rule.get("sha256") and entry["view"]["sha256"] != rule["sha256"]:
            raise ValueError("Original regression reference view changed: " + name)
        if "input_view" in rule and entry["view"] != context["inputs"]["views"][rule["input_view"]]:
            raise ValueError("Evaluation view differs from frozen pre-fit inputs: " + name)
        data[name] = rows(path)
        _pairs(data[name], [])
        if len(data[name]) != rule["n"]:
            raise ValueError("Incomplete evaluation cohort: " + name)
        for row in data[name]:
            if (row.get("split") != rule["split"] or row.get("role") != "command"
                    or row.get("kind") not in ("human", "synthetic")
                    or row.get("reference_status") not in ("supported", "unsupported", "ambiguous")
                    or row.get("metric_eligible") is not (row.get("reference_status") != "ambiguous")):
                raise ValueError("Invalid evaluation row/partition: " + row["id"])
            if row["reference_status"] == "supported":
                validate_command(row["command"])
            elif row.get("command") is not None:
                raise ValueError("Negative/ambiguous reference has a supported command")
            if not isinstance(row.get("transcript", row.get("publisher_metadata", {}).get("transcript")), str):
                raise ValueError("Missing exact source transcript: " + row["id"])
            if name == "fresh_human106" and (row["kind"] != "human"
                    or row.get("fresh_person_against_all_original_views") is not True
                    or row.get("source") != "FluentSpeechCommands"
                    or row.get("source_revision") != context["protocol"]["evaluation"]["fresh_human_source"]["revision"]):
                raise ValueError("Fresh human provenance cannot be replaced or synthesized")
            if rule.get("official"):
                publisher_reference(row)
                if not isinstance(row.get("publisher_metadata"), dict):
                    raise ValueError("Missing full immutable publisher metadata: " + row["id"])
                if row.get("fresh") is not False or row.get("regression") is not True:
                    raise ValueError("Mandatory defaults are exposed regression, never fresh humans")
            audio[name + "/" + row["id"]] = audit_audio(row, entry["audio_root"])
        check_separation(data[name], fitting, fresh=False)
        cohort_bindings[name] = {"view": entry["view"], "audio_root": str(Path(entry["audio_root"]).resolve()),
                                 "rows": len(data[name]), "regression": rule["regression"],
                                 "source_receipts": receipts}
    regression = data["original_test4418"] + data["original_holdout196"] + data["mandatory_defaults202"]
    check_separation(data["fresh_human106"], fitting + regression, fresh=True)
    if len({row["speaker_id"] for row in data["fresh_human106"]}) != 10:
        raise ValueError("Fresh complete cohort must retain all ten declared source people")
    return {"data": data, "cohorts": cohort_bindings, "audio": audio, "sources": sources}


def freeze_selection(evidence, output):
    if not REQUIRED_LOCK_FILES <= set(evidence.get("files", {})):
        raise ValueError("Incomplete selection-before-test file bindings")
    lock = {"format": "vcm-mixed-selection-before-test-v2", "complete": True,
            **deepcopy(evidence), "thresholds": deepcopy(THRESHOLDS),
            "created_utc": datetime.now(timezone.utc).isoformat(), "evaluation_neural_calls": 0,
            "evaluation_order": list(COHORTS), "weights_fitted": False, "calibration_fitted": False,
            "enrollment_rebuilt": False, "final_tts_accessed": False, "qualification": False,
            "evidence_scope": "Selected-candidate CUDA offline command-audio replay; physical Pi and resident service remain separate"}
    write_new(output, lock)
    verify_selection(output)
    return lock


def verify_selection(path, *, expected_sha256=None):
    if expected_sha256 is not None and sha(path) != expected_sha256:
        raise ValueError("The original selection-before-test lock changed")
    lock = read_json(path)
    if (lock.get("format") != "vcm-mixed-selection-before-test-v2" or lock.get("complete") is not True
            or not REQUIRED_LOCK_FILES <= set(lock.get("files", {}))
            or lock.get("thresholds") != THRESHOLDS or lock.get("evaluation_neural_calls") != 0
            or lock.get("evaluation_order") != list(COHORTS)):
        raise ValueError("Incomplete or changed selection-before-test lock")
    for group in ("files", "sources", "audio"):
        for name, item in lock.get(group, {}).items():
            verify_file(item, group + "/" + name)
    for name, item in lock.get("cohorts", {}).items():
        verify_file(item["view"], "cohort/" + name)
        for source in item["source_receipts"]:
            verify_file(source, "cohort source/" + name)
    return lock


def predict_audio(model, pcm, context, postfit):
    """No reference/text metadata enters this PCM-to-neural boundary."""
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm

    pcm = validate_pcm(pcm, 16000)
    import torch
    from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors, propose
    from scripts.test.vcm_mixed_finalize import infer_features
    from vcm.frontend import features

    torch.cuda.synchronize()
    started = time.perf_counter()
    norm = context["normalization"]
    x = features(pcm, norm["mean"], norm["std"])
    frontend_done = time.perf_counter()
    tensors = infer_features(model, x)
    torch.cuda.synchronize()
    neural_done = time.perf_counter()
    decoded = prediction_from_tensors(tensors, context["contract"], postfit["memory"], postfit["policy"])
    final = decide(decoded, context["contract"], context["contract"], postfit["policy"])
    proposal = propose(decoded, context["contract"], context["contract"])
    ended = time.perf_counter()
    return {"final_application": final, "diagnostic_proposal": proposal,
            "model_prediction": decoded, "timing": {"total_ms": (ended-started)*1000,
                "frontend_ms": (frontend_done-started)*1000, "neural_ms": (neural_done-frontend_done)*1000,
                "decoder_ms": (ended-neural_done)*1000}, "audio_seconds": len(pcm)/16000,
            "timing_scope": "Synchronized CUDA replay including frontend and decoder; excludes file IO, speech duration, endpoint and resident service"}


def _write_jsonl(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in data:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def run(training, inputs_path, postfit_path, evaluation_spec, output):
    from scripts.test.speaker_demo_suite import read_audio
    from scripts.test.vcm_mixed_finalize import load_selected_cuda, verify_training

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Existing evaluation evidence must be preserved: " + str(output))
    context = verify_training(training, inputs_path)
    postfit = verify_postfit(context, postfit_path)
    spec = read_json(evaluation_spec)
    evaluation = prepare_evaluation(context, spec)
    files = {**postfit["files"], "postfit_receipt": binding(postfit["receipt_path"]),
             "evaluation_spec": binding(evaluation_spec)}
    sources = {**postfit["sources"], **evaluation["sources"],
               **{name: binding(ROOT / name) for name in EVALUATOR_SOURCES}}
    for run_name, receipts in postfit["receipt"]["training_runs"].items():
        for name, item in receipts.items():
            sources["training/" + run_name + "/" + name] = item
    output.mkdir(parents=True, exist_ok=False)
    lock_path = output / "selection-before-test.json"
    freeze_selection({"files": files, "cohorts": evaluation["cohorts"], "audio": evaluation["audio"],
                      "sources": sources, "selected": context["selected"],
                      "contract_hash": context["contract"]["contract_hash"],
                      "catalog_hash": context["contract"]["catalog_hash"],
                      "calibration_id": postfit["policy"]["calibration_id"],
                      "memory_id": postfit["memory"]["memory_id"]}, lock_path)
    lock_binding = binding(lock_path)
    verify_selection(lock_path, expected_sha256=lock_binding["sha256"])
    # The immutable full-evaluation lock exists before even loading CUDA weights.
    model, payload, gpu = load_selected_cuda(context)
    if gpu.get("device") != "cuda" or gpu.get("visible_device_count") != 1:
        raise ValueError("Offline replay requires one actual CUDA device")
    if payload["normalization"] != context["normalization"]:
        raise ValueError("Loaded selected normalization differs from the evaluation lock")
    official = load_official(spec["official"])
    write_new(output / "run-identity.json", {"pid": os.getpid(), "gpu": gpu, "python": sys.version,
        "selection_lock": lock_binding, "selected": context["selected"], "thresholds": THRESHOLDS,
        "mode": "CUDA offline command-audio replay", "physical_pi_evidence": False,
        "resident_service_evidence": False, "CPU_neural_inference_performed": False})
    reports, neural_calls = {}, 0
    for name, rule in COHORTS.items():
        verify_selection(lock_path, expected_sha256=lock_binding["sha256"])
        destination = output / name
        destination.mkdir(exist_ok=False)
        predictions = []
        with (destination / "predictions.jsonl").open("x", encoding="utf-8") as stream:
            for row in evaluation["data"][name]:
                try:
                    pcm = read_audio(row, Path(spec["cohorts"][name]["audio_root"]))
                    result = predict_audio(model, pcm, context, postfit)
                    neural_calls += 1
                    prediction = {"id": row["id"], **result}
                except Exception as error:
                    prediction = {"id": row["id"], "fault": type(error).__name__ + ": " + str(error)}
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
        data = evaluation["data"][name]
        all_records = [record_result(row, prediction) for row, prediction in _pairs(data, predictions)]
        _write_jsonl(destination / "trial-results.jsonl", all_records)
        _write_jsonl(destination / "misses.jsonl", misses(data, predictions))
        _write_jsonl(destination / "negative-results.jsonl", [record for record in all_records
            if record["source_row"].get("command") is None and record["source_row"]["reference_status"] != "ambiguous"])
        _write_jsonl(destination / "ambiguous-results.jsonl", [record for record in all_records
            if record["source_row"]["reference_status"] == "ambiguous"])
        metrics = summarize(data, predictions)
        metrics.update(cohort=name, evaluation_role="historical_regression" if rule["regression"] else "fresh_unseen_declared_humans",
                       fresh=not rule["regression"], regression=rule["regression"],
                       selected_checkpoint_sha256=context["selected"]["checkpoint_sha256"],
                       calibration_id=postfit["policy"]["calibration_id"], gpu=gpu,
                       selection_lock=lock_binding, source_view=evaluation["cohorts"][name]["view"])
        if rule.get("official"):
            official_report, official_records = official_scores(data, predictions, official)
            write_new(destination / "official-metrics.json", official_report)
            _write_jsonl(destination / "official-records.jsonl", official_records)
            _write_jsonl(destination / "benchmark-log.jsonl", [record["log_record"] for record in official_records
                                                               if record["log_record"] is not None])
            metrics["official_metrics"] = binding(destination / "official-metrics.json")
        write_new(destination / "metrics.json", metrics)
        reports[name] = {"metrics": binding(destination / "metrics.json"), "all": metrics["all"],
                         "kinds": metrics["kinds"], "fresh": not rule["regression"],
                         "regression": rule["regression"], "misses": binding(destination / "misses.jsonl"),
                         "predictions": binding(destination / "predictions.jsonl")}
        print(json.dumps({"completed_cohort": name, "trials": len(data), "all": metrics["all"],
                          "qualification": False}, allow_nan=False), flush=True)
    verify_selection(lock_path, expected_sha256=lock_binding["sha256"])
    receipt = {"format": "vcm-mixed-offline-benchmark-receipt-v2", "status": "unqualified_offline_replay_complete",
        "qualification": False, "mode": "CUDA offline command-audio replay", "gpu": gpu,
        "selected": context["selected"], "selection_lock": lock_binding,
        "postfit_receipt": binding(postfit["receipt_path"]), "thresholds": THRESHOLDS,
        "evaluation_order": list(COHORTS), "cohorts": reports, "completed_trial_rows": sum(r["n"] for r in COHORTS.values()),
        "successful_neural_replays": neural_calls, "CPU_neural_inference_performed": False,
        "weights_fitted": False, "calibration_fitted": False, "enrollment_rebuilt": False,
        "final_tts_accessed": False, "resident_service_evidence": False, "physical_pi_evidence": False,
        "qualification_limits": ["Reference-derived silver is not independent acoustic supervision/review",
            "Mixed calibration missing slot cells remain visible: " + json.dumps(postfit["policy"].get("missing_slot_calibration", [])),
            "CUDA offline replay does not prove ONNX/ncnn parity, resident ownership or physical Pi results",
            "Original test, holdout and current mandatory-default202 are regression, never new human people"],
        "files": {str(path.relative_to(output)): binding(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    write_new(output / "benchmark-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--postfit", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.training, args.inputs, args.postfit, args.evaluation_spec, args.output)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked_unqualified", "qualification": False,
                          "error": type(error).__name__, "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
