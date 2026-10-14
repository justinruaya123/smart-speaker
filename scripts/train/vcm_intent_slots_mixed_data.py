"""Separate v2 silver-label inputs; v1 human-only gates remain unchanged.

Reference text establishes declared supervision, never independent listening or
acoustic spans. The final six Kokoro packs cannot enter any fitting partition.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np

from scripts.runtime.pcv_schema import canonical, normalize_text
from scripts.runtime.vcm_intent_slots_contract import verify_contract
from scripts.test.speaker_demo_suite import read_audio
from scripts.train.vcm_intent_slots_data import eligible as human_eligible, rows, sha, write_new
from vcm.catalog import number_words
from vcm.frontend import features

STUDY = "vcm-mixed-final-v2"
SOURCE = "hexgrad/Kokoro-82M"
VOICE_SPLITS = {
    "train": ["af_sarah", "am_adam"], "dev": ["af_sky"],
    "calibration": ["am_echo"], "enrollment": ["bf_alice", "bf_lily", "bm_fable"],
}
FINAL_VOICES = {"af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck"}


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value)


def validate_protocol(protocol, contract):
    verify_contract(contract)
    source = protocol.get("source", {})
    if (protocol.get("experiment_id") != STUDY or source.get("repository") != SOURCE
            or not _hex(source.get("revision"), 40)
            or protocol.get("contract_hash") != contract["contract_hash"]
            or protocol.get("catalog_hash") != contract["catalog_hash"]
            or protocol.get("thresholds") != {"intent": .95, "slot": .90}
            or protocol.get("final_tts_fit_allowed") is not False
            or protocol.get("voices") != VOICE_SPLITS):
        raise ValueError("Invalid v2 source, preserved contract, thresholds or voice partition")


def _text(row):
    return row.get("spoken_text", row.get("transcript", row.get("text",
                   row.get("publisher_metadata", {}).get("transcript", ""))))


def _reference(row, contract):
    text = normalize_text(_text(row))
    requested = row.get("reference_id", row.get("case_id"))
    matches = [r for r in contract["cases"] + contract["catalog_entries"]
               if normalize_text(r.get("spoken_text", r.get("text", ""))) == text
               and (requested is None or r.get("id") == requested)]
    if not text or not matches:
        return None
    labels = {canonical(r.get("command")) for r in matches}
    if len(labels) != 1:
        raise ValueError("Conflicting exact preserved reference")
    if canonical(row.get("command")) not in labels:
        raise ValueError("Row command differs from exact preserved reference")
    return matches[0]


def eligible(row, protocol, contract):
    """Synthetic silver fitting is explicit; existing human eligibility delegates to v1."""
    validate_protocol(protocol, contract)
    if row.get("kind") == "human":
        human_eligible(row)
        return
    split, voice = row.get("split"), row.get("voice_id")
    if (row.get("kind") != "synthetic" or split not in VOICE_SPLITS
            or voice in FINAL_VOICES or voice not in VOICE_SPLITS[split]
            or row.get("speaker_id") != "Kokoro-82M/" + str(voice)
            or row.get("source") != SOURCE
            or row.get("source_revision") != protocol["source"]["revision"]
            or not row.get("license") or not row.get("pair_id")
            or not row.get("recording_group", row.get("recording_group_id"))
            or row.get("published_split") == "numerals"
            or row.get("role") != "command"
            or row.get("metric_eligible") is not True or row.get("audio_flags")
            or row.get("reviewed_spans")
            or row.get("supervision_status") != "reference_derived_silver"
            or type(row.get("acoustic_listening_verified")) is not bool
            or row.get("reference_status") not in ("supported", "unsupported")
            or any(not _hex(row.get(k), 64)
                   for k in ("sha256", "pcm_sha256"))):
        raise ValueError("Ineligible or unbound v2 synthetic row: " + str(row.get("id")))
    reference = _reference(row, contract)
    if reference is None or row["reference_status"] != (
            "supported" if reference.get("command") is not None else "unsupported"):
        raise ValueError("Missing exact preserved synthetic reference")
    if split == "enrollment":
        role, entity = row.get("entity_role"), row.get("entity_id")
        if (role not in contract["entity_roles"] or entity == "light.all"
                or (row.get("command") or {}).get("slots", {}).get(str(role) + "_id") != entity):
            raise ValueError("Enrollment reference has no matching named entity")


def _words(n):
    if n < 0:
        return "minus " + _words(-n)
    if n <= 100:
        return number_words(n)
    for scale, name in ((1000, "thousand"), (100, "hundred")):
        if n >= scale:
            return _words(n // scale) + " " + name + (" " + _words(n % scale) if n % scale else "")


def _number_forms(n):
    words = _words(n)
    return (str(n), words, words.replace("hundred ", "hundred and "))


def annotate_reference(row, contract):
    """Produce metadata-derived role/value labels; discard unreviewed acoustic spans."""
    result = deepcopy(row)
    result.update(role_labels={}, acoustic_values={}, reviewed_spans=[],
                  supervision_status="reference_derived_silver",
                  supervision_limits=["no_independently_reviewed_acoustic_spans"])
    # Preserve any actual review flag; reference matching never turns it true.
    result.setdefault("acoustic_listening_verified", False)
    reference = _reference(row, contract)
    if reference is None:
        result["supervision_limits"].append("unknown_exact_reference")
        result.pop("request_validity", None)
        return result
    cmd = row.get("command")
    if cmd is None:
        result["request_validity"] = "unsupported"
        return result
    states = {role: "absent" for role in contract["roles"]}
    values, text = {}, normalize_text(_text(row))
    intent, slots = cmd["intent"], cmd["slots"]
    result["request_validity"] = "valid"
    mappings = {
        "BRIGHTNESS": {"brightness_percent": "percent"},
        "VOLUME": {"volume_operation": "operation", "volume_percent": "percent"},
        "COLOR": {"color": "color"}, "TEMPERATURE": {"temperature_degrees": "degrees"},
    }
    for role, field in mappings.get(intent, {}).items():
        if field in slots:
            states[role], values[role] = "present", slots[field]
    for role in contract["entity_roles"]:
        entity = slots.get(role + "_id")
        if entity and entity != "light.all":
            # contact.home is also the bare-call default; a supplied home name is distinct.
            if role != "contact" or entity != "contact.home" or "home" in text.split():
                states[role] = "present"
    if intent == "TIMER":
        units = re.findall(r"\b(seconds?|minutes?|hours?)\b", text)
        if len(units) > 1:
            states.pop("timer_amount")
            states.pop("timer_unit")
            result["supervision_limits"].append("composite_timer_not_representable_as_spoken_single_pair")
        else:
            unit = (units[0].rstrip("s") + "s") if units else None
            multiplier = contract["grammar"]["timer_unit_seconds"].get(unit)
            duration = slots["duration_seconds"]
            amount = duration // multiplier if multiplier and duration % multiplier == 0 else None
            if (amount in contract["finite_values"]["timer_amount"] and any(
                    re.search(r"\b" + re.escape(form) + r" " + units[0] + r"\b", text)
                    for form in _number_forms(amount))):
                states["timer_amount"] = states["timer_unit"] = "present"
                values.update(timer_amount=amount, timer_unit=unit)
            else:
                states.pop("timer_amount")
                states.pop("timer_unit")
                result["supervision_limits"].append("unverified_spoken_timer_pair")
    if intent == "ALARM":
        hour24, minute = (int(v) for v in slots["time"].split(":"))
        hour, meridiem = hour24 % 12 or 12, "PM" if hour24 >= 12 else "AM"
        suffix = "p m" if meridiem == "PM" else "a m"
        h = "(?:" + "|".join(re.escape(s) for s in _number_forms(hour)) + ")"
        m = "(?:" + "|".join(re.escape(s) for s in _number_forms(minute)) + ")"
        if re.search(r"\b" + h + " " + (r"(?:oh )?" if 0 < minute < 10 else "") + m + " " + suffix + r"\b", text):
            states.update(alarm_hour="present", alarm_minute="present", alarm_meridiem="present")
            values.update(alarm_hour=hour, alarm_minute=minute, alarm_meridiem=meridiem)
        elif minute == 0 and re.search(r"\b" + h + " " + suffix + r"\b", text):
            states.update(alarm_hour="present", alarm_meridiem="present")
            values.update(alarm_hour=hour, alarm_meridiem=meridiem)
        else:
            for role in ("alarm_hour", "alarm_minute", "alarm_meridiem"):
                states.pop(role)
            result["supervision_limits"].append("unverified_spoken_clock_fields")
    result.update(role_labels=states, acoustic_values=values)
    return result


def audit_views(view_paths, source_receipts, output_dir, *, protocol, contract, audio_root=None):
    """Audit all supplied partitions, including reserved tests and enrollment."""
    validate_protocol(protocol, contract)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    seen_ids, partitions = set(), {}
    groups = {k: {} for k in ("sha256", "pcm_sha256", "speaker_id", "pair_id", "recording_group")}
    report = {"experiment_id": STUDY, "splits": {}, "bindings": {}, "missing_intents": {},
              "reviewed_acoustic_spans": 0, "acoustic_listening_verified": 0,
              "supervision_status": {}, "accuracy_qualified": False,
              "identity_limits": "Declared human IDs and preset pack IDs only; physical-person aliases unverified"}
    status = Counter()
    for split, path in view_paths.items():
        path = Path(path)
        binding = source_receipts.get("views", {}).get(split)
        if not binding or binding.get("sha256") != sha(path):
            raise ValueError("Missing or stale v2 view hash binding: " + split)
        data = rows(path)
        partitions[split] = data
        intents, kinds, states, values, enrollment = (Counter() for _ in range(5))
        for row in data:
            if row.get("id") in seen_ids or row.get("split") != split:
                raise ValueError("Duplicate row or partition mismatch")
            seen_ids.add(row["id"])
            if split in VOICE_SPLITS:
                eligible(row, protocol, contract)
            for key, index in groups.items():
                value = row.get(key, row.get("recording_group_id") if key == "recording_group" else None)
                if value and value in index and index[value] != split:
                    raise ValueError("Cross-partition leakage: " + key)
                if value:
                    index[value] = split
            if audio_root is not None:
                read_audio(row, Path(audio_root))
            kinds[row.get("kind", "unknown")] += 1
            if row.get("reference_status") != "ambiguous":
                intents[(row.get("command") or {}).get("intent", "OOV")] += 1
            states.update(role + "/" + state for role, state in row.get("role_labels", {}).items())
            values.update(role + "=" + canonical(value) for role, value in row.get("acoustic_values", {}).items())
            report["reviewed_acoustic_spans"] += len(row.get("reviewed_spans", []))
            report["acoustic_listening_verified"] += row.get("acoustic_listening_verified") is True
            status[row.get("supervision_status", "original_command_metadata")] += 1
            if split == "enrollment":
                enrollment[row["entity_role"] + "/" + row["entity_id"]] += 1
        report["bindings"][split] = {"path": str(path.resolve()), "sha256": sha(path)}
        report["splits"][split] = {
            "rows": len(data), "human": kinds["human"], "synthetic": kinds["synthetic"],
            "intents": dict(intents), "role_states": dict(states), "acoustic_values": dict(values),
            "enrollment": dict(enrollment),
        }
        report["missing_intents"][split] = [i for i in contract["intent_order"] if not intents[i]]
    supervised = {role for row in partitions.get("train", []) for role in row.get("role_labels", {})}
    report["missing_role_state_roles"] = sorted(set(contract["roles"]) - supervised)
    report["fit_ready"] = (all(s in partitions and not report["missing_intents"][s] for s in ("train", "dev"))
                           and not report["missing_role_state_roles"])
    report["supervision_status"] = dict(status)
    report["qualification_limits"] = ["Independent acoustic review and reviewed acoustic spans remain separate gates"]
    write_new(output_dir / "audit.json", report)
    return report


def prepare_features(subset, audio_root, output_dir, *, protocol, contract):
    """Verified 40-bin features; only training frames determine normalization."""
    validate_protocol(protocol, contract)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    total, squares, frames, receipt = np.zeros(40), np.zeros(40), 0, {}
    for row in subset:
        if row["split"] not in ("train", "dev"):
            raise ValueError("Calibration/enrollment/test access forbidden during fitting")
        eligible(row, protocol, contract)
        x = features(read_audio(row, Path(audio_root)))
        if not x.shape[1] or not np.isfinite(x).all() or row["id"] in receipt:
            raise ValueError("Missing finite frames or duplicate feature row")
        path = output_dir / (hashlib.sha256(row["id"].encode()).hexdigest() + ".npy")
        np.save(path, x)
        receipt[row["id"]] = {"path": str(path), "sha256": sha(path), "frames": x.shape[1],
                              "audio_sha256": row["sha256"], "pcm_sha256": row["pcm_sha256"]}
        if row["split"] == "train":
            total += x.sum(axis=1, dtype=np.float64)
            squares += np.square(x.astype(np.float64)).sum(axis=1)
            frames += x.shape[1]
    if not frames:
        raise ValueError("No eligible v2 training frames")
    mean = total / frames
    norm = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(squares / frames - mean**2, 1e-6)).tolist(),
            "fit_split": "train", "frames": frames}
    write_new(output_dir / "features.json", receipt)
    write_new(output_dir / "normalization.json", norm)
    return receipt, norm
