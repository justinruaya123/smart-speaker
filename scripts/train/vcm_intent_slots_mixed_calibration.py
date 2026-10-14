"""Separate v2 calibration with immutable provenance; no final-test fitting."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from scripts.runtime.vcm_intent_slots_contract import digest
from scripts.train.vcm_intent_slots_calibration import _fit_numeric_scores
from scripts.train.vcm_intent_slots_data import rows, sha, write_new
from scripts.train.vcm_intent_slots_mixed_data import SOURCE, eligible, validate_protocol


IDENTITY_FIELDS = ("id", "audio", "split", "kind", "speaker_id", "sha256", "pcm_sha256")
BINDING_FIELDS = ("protocol", "manifest", "contract_hash", "checkpoint", "normalization", "frontend")


def _validate_scores(observation, contract, source_row):
    label = observation.get("intent_label")
    expected = (source_row.get("command") or {}).get("intent", "OOV")
    if (type(label) is not int or label != contract["intent_order"].index(expected)):
        raise ValueError("Calibration intent label differs from exact source reference")
    logits = np.asarray(observation.get("intent_logits"), dtype=float)
    if logits.shape != (19,) or not np.isfinite(logits).all():
        raise ValueError("Calibration requires finite 19-class logits")
    slots = observation.get("slots", {})
    if not isinstance(slots, dict):
        raise ValueError("Invalid calibration slot evidence")
    for field, item in slots.items():
        if not isinstance(field, str) or not field or not isinstance(item, dict):
            raise ValueError("Invalid calibration slot evidence")
        scores = np.asarray(item.get("scores"), dtype=float)
        if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)
                or not isinstance(item.get("provenance"), str) or not item["provenance"]
                or type(item.get("correct")) not in (int, float, bool) or item["correct"] not in (0, 1)):
            raise ValueError("Invalid/nonfinite calibration slot scores/outcomes")
        if item["correct"] == 1 and field not in (source_row.get("command") or {}).get("slots", {}):
            raise ValueError("Positive slot calibration truth has no exact source reference field")


def fit_mixed_score_calibration(observations, binding, output, *, protocol, contract, manifest):
    """Fit declared v2 observations, never reclassifying synthetic data as human."""
    protocol, manifest = Path(protocol), Path(manifest)
    for key in BINDING_FIELDS:
        value = binding.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Missing or invalid calibration binding: " + key)
    if binding["protocol"] != sha(protocol) or binding["manifest"] != sha(manifest):
        raise ValueError("Stale frozen calibration protocol/manifest hash binding")
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol_data, contract)
    if binding["contract_hash"] != contract["contract_hash"]:
        raise ValueError("Calibration contract hash binding mismatch")
    data = rows(manifest)
    frozen = {}
    for row in data:
        if row.get("id") in frozen or row.get("split") != "calibration":
            raise ValueError("Duplicate or non-calibration manifest row")
        eligible(row, protocol_data, contract)
        if row["kind"] == "human" and (row.get("voice_id") or row.get("source") == SOURCE
                or row.get("speaker_id", "").startswith("Kokoro-82M/")):
            raise ValueError("Synthetic lineage cannot be relabeled as human calibration")
        frozen[row["id"]] = row
    ids = [observation.get("id") for observation in observations]
    if not observations or len(set(ids)) != len(ids) or set(ids) != set(frozen):
        raise ValueError("Duplicate or incomplete calibration manifest observation coverage")
    for observation in observations:
        source_row = observation.get("source_row")
        if source_row != frozen[observation["id"]]:
            raise ValueError("Calibration source_row differs from frozen manifest reference")
        if any(key not in observation or observation[key] != source_row.get(key) for key in IDENTITY_FIELDS):
            raise ValueError("Calibration observation identity differs from source row")
        _validate_scores(observation, contract, source_row)
    if set(o["intent_label"] for o in observations) != set(range(19)):
        raise ValueError("Missing mixed calibration intent classes; all 19 required")
    fitted = _fit_numeric_scores(observations)
    counts = Counter(o["kind"] for o in observations)
    silver = sum(o["source_row"].get("supervision_status") == "reference_derived_silver" for o in observations)
    policy = {
        "contract_hash": binding["contract_hash"], "binding": deepcopy(binding),
        "intent_temperature": fitted["intent_temperature"], "slot_calibrators": fitted["slot_calibrators"],
        "entities": deepcopy(binding.get("entities", {})),
        "missing_slot_calibration": fitted["missing_slot_calibration"], "require_calibration": True,
        "thresholds": {"intent": .95, "slot": .90},
        "calibration_people": sorted({o["speaker_id"] for o in observations if o["kind"] == "human"}),
        "calibration_voices": sorted({o["source_row"]["voice_id"] for o in observations if o["kind"] == "synthetic"}),
        "calibration_counts": {"human": counts["human"], "synthetic": counts["synthetic"],
                               "reference_derived_silver": silver},
        "calibration_manifest_rows": len(data), "experiment_id": "vcm-mixed-final-v2",
        "supervision_status": "contains_reference_derived_silver" if silver else "source_declared_human_supervision",
        "status": "frozen_mixed_partial_calibration" if fitted["missing_slot_calibration"] else "frozen_mixed_scores_unqualified",
        "qualification": False,
    }
    policy["calibration_id"] = digest(policy)
    write_new(Path(output) / "calibration.json", policy)
    return policy
