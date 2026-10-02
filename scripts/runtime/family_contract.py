"""Versioned family/finite-head research contract; transcripts are labels only."""
from __future__ import annotations

import re
from typing import Any

from .levels import level_parameter, percentage
from .routing import match_target, parse_parameters, without_target, validate_arguments
from .slot_contract import supervision, vocabulary, assemble, catalog_signature, apply_value_policy

LABELS = ["UNKNOWN", "POWER_ON", "POWER_OFF", "BRIGHTNESS", "COLOR_SET",
          "MEDIA_PLAY", "MEDIA_RESUME", "TIME_QUERY", "DATE_QUERY", "VOLUME"]


def contract(catalog: dict[str, Any]) -> dict[str, Any]:
    heads = vocabulary(catalog)
    heads["operation"] = ["INCREASE", "DECREASE", "SET", "MUTE", "MIN", "MAX"]
    heads["amount"] = ["NONE", "OTHER", *map(str, range(101))]
    heads["media_mode"] = ["GENERIC", "CUSTOM"]
    return {"schema_version": "family-finite-v1", "intent_labels": LABELS,
            "vocabulary": heads, "streaming_supported": False,
            "annotation_states": {"null": "unannotated; zero loss", "NONE": "explicitly absent",
                                  "OTHER": "unsupported or unresolved value"},
            "defaults": {"relative_step": {"value": 10, "unit": "percentage_points",
                         "requires": ["accepted INCREASE/DECREASE", "accepted amount NONE"]}},
            "operations": {"INCREASE": "positive relative delta_percentage_points",
                           "DECREASE": "negative relative delta_percentage_points", "SET": "absolute percent",
                           "MUTE": "distinct operation; execution disabled",
                           "MIN": "distinct operation; execution disabled", "MAX": "distinct operation; execution disabled"},
            "custom_media": {"schema_version": 1, "open_text_fields": ["title", "artist"],
                             "required_finite_heads": []},
            "unsupported": ["unmute", "arbitrary numbers", "unseen acoustic device names", "parameter-only audio timing"],
            "activation": "Every finite head disabled until calibrated support/precision and joint-command checks pass",
            "color_gate": {"minimum_accepts": 20, "observed_precision": .99},
            "reference_quality": "programmatic silver; no human adjudication"}


def annotate(row: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    """Retain independent known fields and distinguish unknown from absence."""
    family = row["command"]
    if family not in LABELS:
        raise ValueError(f"Unsupported family: {family}")
    text = row.get("transcript")
    labels = {name: None for name in contract(catalog)["vocabulary"]}
    if text is None:
        return {"labels": labels, "reference": None, "quality": "unannotated"}
    reference = parse_parameters(family, text, catalog)
    old, _ = supervision(family, text, catalog)
    labels["guard"] = old["guard"]
    if family in {"POWER_ON", "POWER_OFF", "COLOR_SET"}:
        labels["target"] = old["target"]
    if family == "COLOR_SET":
        labels["color"] = old["color"]
    if family == "MEDIA_PLAY":
        labels["media"] = old["media"]
        if reference["status"] == "accepted":
            labels["media_mode"] = "CUSTOM" if reference["command"]["arguments"] else "GENERIC"
    if family in {"BRIGHTNESS", "VOLUME"}:
        remainder = text
        if family == "BRIGHTNESS":
            try:
                target = match_target(text, catalog)
                labels["target"] = target.get("acoustic_key", target["id"])
                remainder = without_target(text, target, preserve_numbers=True)
            except ValueError:
                labels["target"] = "OTHER"
        try:
            level = level_parameter(remainder)
            if "percent" in level:
                labels["operation"], labels["amount"] = "SET", str(level["percent"])
            else:
                labels["operation"] = "INCREASE" if level["delta_percentage_points"] > 0 else "DECREASE"
                explicit = percentage(remainder)
                labels["amount"] = "NONE" if explicit is None else str(explicit)
        except ValueError:
            # No guessed direction or numeric default. These endpoint labels are
            # research supervision only and cannot emit an executable command.
            if re.fullmatch(r"(?:volume )?mute(?: (?:the )?(?:volume|sound))?", remainder.strip().lower()):
                labels["operation"] = "MUTE"
            elif re.search(r"\b(max|maximum|full)\b", remainder.lower()):
                labels["operation"] = "MAX"
            elif re.search(r"\b(minimum)\b", remainder.lower()):
                labels["operation"] = "MIN"
            labels["amount"] = "OTHER"
        labels["guard"] = "READY" if reference["status"] == "accepted" else "UNRESOLVED"
    return {"labels": labels, "reference": reference, "quality": "programmatic_silver"}


def legacy_action(family: str, operation: str) -> str:
    """Only exact operation mappings are legal; absolute is never relative."""
    mapping = {("BRIGHTNESS", "INCREASE"): "BRIGHTNESS_UP",
               ("BRIGHTNESS", "DECREASE"): "BRIGHTNESS_DOWN",
               ("BRIGHTNESS", "SET"): "BRIGHTNESS_SET", ("VOLUME", "SET"): "VOLUME_SET"}
    if (family, operation) not in mapping:
        raise ValueError("No exact legacy action mapping")
    return mapping[family, operation]


def required_heads(family: str) -> list[str]:
    heads = ["guard"]
    if family in {"POWER_ON", "POWER_OFF", "BRIGHTNESS", "COLOR_SET"}:
        heads.append("target")
    if family in {"BRIGHTNESS", "VOLUME"}:
        heads += ["operation", "amount"]
    if family == "COLOR_SET":
        heads.append("color")
    if family == "MEDIA_PLAY":
        heads += ["media", "media_mode"]
    return heads


def assemble_family(family: str, values: dict, catalog: dict) -> dict:
    if family not in {"BRIGHTNESS", "VOLUME"}:
        if family == "MEDIA_PLAY" and values.get("media_mode") != "GENERIC":
            return {"status": "needs_parameters", "reason": "custom_media_name_required"}
        return assemble(family, values, catalog)
    operation, amount = values.get("operation"), values.get("amount")
    if operation not in {"INCREASE", "DECREASE", "SET"}:
        return {"status": "needs_parameters", "reason": "unsupported_or_missing_operation"}
    args = {}
    if family == "BRIGHTNESS":
        matches = [e for e in catalog.get("entities", []) if e.get("acoustic_key", e["id"]) == values.get("target")]
        if len(matches) != 1:
            return {"status": "needs_parameters", "reason": "unresolved_target"}
        args["target"] = {"id": matches[0]["id"]}
    if amount == "NONE" and operation in {"INCREASE", "DECREASE"}:
        amount = "10"
    if amount not in {str(n) for n in range(101)}:
        return {"status": "needs_parameters", "reason": "unresolved_amount"}
    if operation == "SET":
        args["percent"] = int(amount)
    else:
        args["delta_percentage_points"] = int(amount) * (1 if operation == "INCREASE" else -1)
    action = ("SET_" if operation == "SET" else "ADJUST_") + family
    return validate_arguments(family, args, catalog, action=action)


def family_candidate(family: str, probabilities: dict, config: dict, catalog: dict) -> dict:
    values, confidence = {}, {}
    for name in required_heads(family):
        p = probabilities[name]
        index = max(range(len(p)), key=p.__getitem__)
        values[name] = config["vocabulary"][name][index]
        confidence[name] = float(p[index])
    result = assemble_family(family, values, catalog) if values["guard"] == "READY" else None
    score = min(confidence.values()) if result and result["status"] == "accepted" else 0.
    return {"values": values, "confidence": confidence, "score": score, "result": result}


def route_family(family: str, probabilities: dict, config: dict, catalog: dict, variant: str) -> dict:
    if catalog_signature(catalog) != config.get("catalog_signature"):
        return {"decision": "fallback", "reason": "acoustic_catalog_changed"}
    policy = config.get("variants", {}).get(variant, {})
    candidate = apply_value_policy(family_candidate(family, probabilities, config, catalog), policy)
    if candidate["score"] >= policy.get("acceptance_thresholds", {}).get(family, 1.1):
        return {"decision": "accepted", "candidate": candidate, **candidate["result"]}
    guard = candidate["values"]["guard"]
    if guard in {"MISSING", "CONFLICT"} and candidate["confidence"]["guard"] >= policy.get("clarification_thresholds", {}).get(guard, 1.1):
        return {"decision": "clarify", "status": "needs_clarification",
                "reason": "acoustic_parameter_"+guard.lower(), "candidate": candidate}
    return {"decision": "fallback", "reason": "unresolved_acoustic_parameters", "candidate": candidate}
