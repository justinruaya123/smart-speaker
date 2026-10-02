"""Finite acoustic slots and typed validation. Never selects or changes an intent."""
from __future__ import annotations

import hashlib
import json
from .routing import COLORS, color_parameter, match_target, parse_parameters, validate_arguments

LABELS = ["UNKNOWN", "POWER_ON", "POWER_OFF", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN",
          "COLOR_SET", "MEDIA_PLAY", "MEDIA_RESUME", "TIME_QUERY", "DATE_QUERY"]
TARGET_COMMANDS = {"POWER_ON", "POWER_OFF", "BRIGHTNESS_UP", "BRIGHTNESS_DOWN", "COLOR_SET"}
RELATIVE_COMMANDS = {"BRIGHTNESS_UP", "BRIGHTNESS_DOWN"}
PARAMETERLESS = {"MEDIA_RESUME", "TIME_QUERY", "DATE_QUERY"}


def catalog_signature(catalog):
    """IDs may be rebound; changing the spoken vocabulary invalidates acoustic slots."""
    vocabulary = sorted((e.get("acoustic_key", e["id"]), e["name"].lower(), e["kind"],
                         sorted(a.lower() for a in e.get("aliases", [])), bool(e.get("generic")))
                        for e in catalog.get("entities", []))
    return hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode()).hexdigest()


def vocabulary(catalog):
    return {
        "guard": ["READY", "MISSING", "UNRESOLVED", "CONFLICT"],
        "target": ["NONE", "OTHER", *[e.get("acoustic_key", e["id"]) for e in catalog["entities"]]],
        "color": ["NONE", "OTHER", *sorted(COLORS)],
        "amount": ["NONE", "OTHER", *[str(n) for n in range(1, 101)]],
        "media": ["NONE", "OTHER", "GENERIC"],
    }


def supervision(command, text, catalog):
    """Training/evaluation only: conservative silver labels from human transcripts."""
    slots = {"guard": "CONFLICT", "target": "NONE", "color": "NONE", "amount": "NONE", "media": "NONE"}
    parsed = parse_parameters(command, text, catalog)
    if command == "UNKNOWN":
        return slots, parsed
    # Parameter evidence is independent: a missing target must not teach the
    # color head that an audible color is absent, or vice versa. The complete
    # parser result still governs readiness and the executable reference.
    target = None
    if command in TARGET_COMMANDS:
        try:
            target = match_target(text, catalog)
            slots["target"] = target.get("acoustic_key", target["id"])
        except ValueError:
            slots["target"] = "OTHER"
    if command == "COLOR_SET":
        try:
            slots["color"] = color_parameter(text, target)
        except ValueError:
            slots["color"] = "OTHER"
    if command in RELATIVE_COMMANDS:
        from .routing import without_target
        from .levels import percentage
        try:
            # Unknown target names may contain numbers; do not infer absence.
            value = percentage(without_target(text, target, preserve_numbers=True)) if target else None
            slots["amount"] = "OTHER" if target is None else "NONE" if value is None else str(value)
        except ValueError:
            slots["amount"] = "OTHER"
    if parsed["status"] != "accepted":
        reason = parsed.get("reason", "")
        if "Unknown or missing target" in reason:
            slots["guard"] = "UNRESOLVED"
        elif "Missing or ambiguous color" in reason:
            slots.update(guard="UNRESOLVED" if not COLORS.intersection(text.lower().split()) else "CONFLICT", color="OTHER")
        elif "Missing media" in reason:
            slots.update(guard="MISSING", media="NONE")
        return slots, parsed
    arguments = parsed["command"]["arguments"]
    slots["guard"] = "READY"
    if command in TARGET_COMMANDS:
        target = next(e for e in catalog["entities"] if e["id"] == arguments["target"]["id"])
        slots["target"] = target.get("acoustic_key", target["id"])
    if command == "COLOR_SET":
        slots["color"] = arguments["color"]
    if command in RELATIVE_COMMANDS:
        # Explicit ten and an omitted amount must remain different labels.
        from .routing import without_target
        from .levels import percentage
        value = percentage(without_target(text, target, preserve_numbers=True))
        slots["amount"] = "NONE" if value is None else str(value)
    if command == "MEDIA_PLAY":
        slots["media"] = "OTHER" if arguments else "GENERIC"
        if arguments:
            slots["guard"] = "UNRESOLVED"
    return slots, parsed


def assemble(command, values, catalog):
    """Convert confidently decoded slots to validated typed arguments, without text."""
    args = {}
    if command in TARGET_COMMANDS:
        matches = [e for e in catalog.get("entities", []) if e.get("acoustic_key", e["id"]) == values.get("target")]
        if len(matches) != 1:
            return {"status": "needs_clarification", "vcm_command": command, "reason": "Unresolved acoustic target"}
        target = matches[0]
        args["target"] = {"id": target["id"], "name": target["name"], "device_kind": target["kind"]}
    if command.startswith("POWER_"):
        args["on"] = command == "POWER_ON"
    if command in RELATIVE_COMMANDS:
        value = values.get("amount")
        if value == "NONE":
            value = 10
        elif value not in {str(n) for n in range(1, 101)}:
            return {"status": "needs_clarification", "vcm_command": command, "reason": "Unresolved spoken amount"}
        args["delta_percentage_points"] = int(value) * (1 if command.endswith("_UP") else -1)
    if command == "COLOR_SET":
        args["color"] = values.get("color")
    if command == "MEDIA_PLAY" and values.get("media") != "GENERIC":
        return {"status": "needs_clarification", "vcm_command": command, "reason": "Unresolved media selection"}
    return validate_arguments(command, args, catalog)


def required_heads(command):
    heads = ["guard"]
    if command in TARGET_COMMANDS:
        heads.append("target")
    if command in RELATIVE_COMMANDS:
        heads.append("amount")
    if command == "COLOR_SET":
        heads.append("color")
    if command == "MEDIA_PLAY":
        heads.append("media")
    return heads


def acoustic_candidate(command, probabilities, vocab, catalog):
    values, confidence = {}, {}
    for name in required_heads(command):
        p = probabilities[name]
        index = max(range(len(p)), key=p.__getitem__)
        values[name], confidence[name] = vocab[name][index], float(p[index])
    result = assemble(command, values, catalog) if values["guard"] == "READY" else None
    score = min(confidence.values()) if result and result["status"] == "accepted" else 0.
    return {"values": values, "confidence": confidence, "score": score, "result": result}


def apply_value_policy(candidate, policy):
    """An unvalidated vocabulary entry cannot inherit a common value's threshold."""
    thresholds = policy.get("value_thresholds")
    if thresholds is None:
        return candidate
    unsupported = {name: value for name, value in candidate["values"].items() if name != "guard"
                   and candidate["confidence"][name] < thresholds.get(name, {}).get(value, 1.1)}
    return {**candidate, "score": 0. if unsupported else candidate["score"], "unsupported_values": unsupported}


def route_acoustic(command, probabilities, config, catalog, variant="fp32", *, intent_confidence=1.):
    if config.get("schema_version") == "family-finite-v1":
        from .family_contract import route_family
        return route_family(command, probabilities, config, catalog, variant)
    policy = config.get("variants", {}).get(variant, {})
    if catalog_signature(catalog) != config["catalog_signature"]:
        return {"decision": "fallback", "reason": "acoustic_catalog_changed"}
    candidate = apply_value_policy(acoustic_candidate(command, probabilities, config["vocabulary"], catalog), policy)
    if command == "MEDIA_PLAY" and config.get("custom_media") and candidate["values"].get("media") == "GENERIC":
        mode = probabilities.get("media_mode", [])
        names = config["vocabulary"].get("media_mode", [])
        if len(mode) != len(names) or not len(mode):
            candidate = {**candidate, "score": 0.}
        else:
            index = max(range(len(mode)), key=mode.__getitem__)
            if names[index] != "GENERIC" or mode[index] < policy.get("generic_media_threshold", 1.1):
                candidate = {**candidate, "score": 0.}
    if candidate["score"] >= policy.get("acceptance_thresholds", {}).get(command, 1.1) and intent_confidence >= policy.get("joint_intent_thresholds", {}).get(command, 0.):
        return {"decision": "accepted", "candidate": candidate, **candidate["result"]}
    guard = candidate["values"]["guard"]
    if guard in {"MISSING", "CONFLICT"} and candidate["confidence"]["guard"] >= policy.get("clarification_thresholds", {}).get(guard, 1.1):
        return {"decision": "clarify", "status": "needs_clarification", "vcm_command": command,
                "reason": "acoustic_parameter_" + guard.lower(), "candidate": candidate}
    return {"decision": "fallback", "reason": "unresolved_acoustic_parameters", "candidate": candidate}


def custom_media_eligibility(command, probabilities, config, catalog, variant):
    """Only a separately calibrated CUSTOM acoustic decision can authorize a future custom reader.

    Legacy media=OTHER also represents unresolved/unsupported requests, so it
    cannot authorize custom requests. The current media schema has no finite
    target/qualifier fields; new finite fields require a new routing schema.
    """
    if command != "MEDIA_PLAY":
        return False, "finite_parameters_require_clarification"
    custom = (config or {}).get("custom_media", {})
    if (not probabilities or custom.get("schema_version") != 1
            or custom.get("open_text_fields") != ["title", "artist"]
            or custom.get("required_finite_heads") != []
            or catalog_signature(catalog) != config.get("catalog_signature")):
        return False, "custom_media_mode_unavailable"
    vocab = config.get("vocabulary", {})
    policy = config.get("variants", {}).get(variant, {})
    media = probabilities.get("media", [])
    media_names = vocab.get("media", [])
    if len(media) and len(media) == len(media_names):
        index = max(range(len(media)), key=media.__getitem__)
        generic_floor = policy.get("value_thresholds", {}).get("media", {}).get(
            "GENERIC", policy.get("field_thresholds", {}).get("media", 1.1))
        if media_names[index] == "GENERIC" and media[index] >= generic_floor:
            return False, "acoustic_media_mode_conflict"
    for name, allowed, threshold_name in (
        ("media_mode", {"CUSTOM"}, "custom_media_threshold"),
        ("guard", {"READY", "UNRESOLVED"}, "custom_guard_threshold"),
    ):
        values, scores = vocab.get(name, []), probabilities.get(name, [])
        if not values or len(values) != len(scores):
            return False, "custom_media_mode_unavailable"
        index = max(range(len(scores)), key=scores.__getitem__)
        threshold = policy.get(threshold_name, 1.1)
        if values[index] not in allowed or not 0 <= threshold <= 1 or not scores[index] >= threshold:
            return False, "custom_media_mode_unconfirmed"
    return True, "accepted_custom_media"


def agrees_with_acoustic(command, candidate, arguments, config, catalog, variant):
    if not candidate:
        return True
    policy = config.get("variants", {}).get(variant, {})
    thresholds = policy.get("field_thresholds", {})
    for name, value in candidate["values"].items():
        if name == "guard" or value in {"NONE", "OTHER"} or candidate["confidence"][name] < thresholds.get(name, 1.1):
            continue
        if "value_thresholds" in policy and candidate["confidence"][name] < policy["value_thresholds"].get(name, {}).get(value, 1.1):
            continue
        if name == "target":
            matches = [e for e in catalog.get("entities", []) if e.get("acoustic_key", e["id"]) == value]
            if len(matches) != 1 or arguments.get("target", {}).get("id") != matches[0]["id"]:
                return False
        elif name == "color" and arguments.get("color") != value:
            return False
        elif name == "amount" and abs(arguments.get("delta_percentage_points", 0)) != int(value):
            return False
        elif name == "media" and value == "GENERIC" and arguments:
            return False
    return True
