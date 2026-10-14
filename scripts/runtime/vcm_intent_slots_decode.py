"""One authoritative intent, typed slots, and explicit confidence provenance."""
from __future__ import annotations

import math
import numpy as np

from scripts.runtime.pcv_schema import validate_command
from scripts.runtime.vcm_intent_slots_contract import catalog_digest, verify_contract
from scripts.runtime.vcm_intent_slots_entities import catalog_ids, match_entity


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def calibrated_slot(constituents, name, policy, provenance=""):
    """Frozen logistic score, or an explicitly uncalibrated conservative score."""
    if not constituents or not all(probability(float(v)) for v in constituents):
        raise ValueError("Invalid constituent score")
    key = name + "/" + provenance + "/" + str(len(constituents))
    calibration = policy.get("slot_calibrators", {}).get(key)
    if calibration:
        weights = calibration["weights"]
        if len(weights) != len(constituents):
            raise ValueError("Calibration feature layout mismatch")
        x = np.asarray(constituents)
        z = float(np.asarray(weights) @ x + calibration["bias"])
        if not math.isfinite(z):
            raise ValueError("Invalid calibrated score")
        return float(1 / (1 + np.exp(-np.clip(z, -700, 700))))
    if policy.get("require_calibration"):
        raise ValueError("Missing frozen slot calibration: " + name)
    return float(min(constituents))


def _decision(prediction, contract, catalog, policy, *, diagnostic=False):
    verify_contract(contract)
    result = {"mode": "experimental", "accepted": False, "command": None,
              "reason": None, "slot_evidence": {}, "intent_threshold": .95,
              "slot_threshold": .90, "intent_confidence": prediction.get("intent_confidence"),
              "score_status": ("frozen_calibration" if policy.get("require_calibration")
                               else "uncalibrated_selection_score")}

    def reject(reason):
        return {**result, "reason": reason}

    if catalog_digest(catalog.get("catalog_entries", catalog.get("entries", []))) != contract["catalog_hash"]:
        return reject("stale_catalog")
    if policy.get("require_calibration") and (
            policy.get("contract_hash") != contract["contract_hash"]
            or not policy.get("calibration_id")):
        return reject("missing_or_stale_calibration")
    intent = prediction.get("intent")
    confidence = prediction.get("intent_confidence")
    if intent == "OOV":
        return reject("OOV")
    if intent not in contract["intent_order"][1:] or not probability(confidence) or (confidence < .95 and not diagnostic):
        return reject("intent_confidence")
    validity = prediction.get("validity", {})
    if validity.get("state") != "valid" or not probability(validity.get("confidence")) or (validity["confidence"] < .90 and not diagnostic):
        return reject("request_validity")
    roles = prediction.get("roles", {})
    if set(roles) != set(contract["roles"]):
        return reject("missing_or_extra_role_evidence")
    applicable = set(contract["applicable_roles"][intent])
    for role, record in roles.items():
        if record.get("state") not in ("absent", "present") or not probability(record.get("confidence")):
            return reject("conflicting_or_unsupported_role")
        if (record["confidence"] < .90 and not diagnostic) or (role not in applicable and record["state"] != "absent"):
            return reject("extra_or_uncertain_role")
    slots = {}

    def state(role):
        return roles[role]["state"]

    def finite(role, optional=False):
        if state(role) == "absent":
            if optional:
                return None, [roles[role]["confidence"], validity["confidence"]]
            raise ValueError("Missing " + role)
        record = prediction.get("values", {}).get(role, {})
        value, score = record.get("value"), record.get("confidence")
        inventory = contract["finite_values"][role]
        if type(value) is not type(inventory[0]) or value not in inventory or not probability(score):
            raise ValueError("Invalid finite value")
        return value, [roles[role]["confidence"], score, validity["confidence"]]

    def put(field, value, scores, provenance):
        score = calibrated_slot(scores, field, policy, provenance)
        result["slot_evidence"][field] = {
            "confidence": score, "constituent_scores": scores,
            "provenance": provenance, "score_status": result["score_status"]}
        if score < .90 and not diagnostic:
            raise ValueError("Slot confidence")
        slots[field] = value

    def entity(role, optional=False, default=False):
        if state(role) == "absent":
            if default:
                rule = contract["defaults"][role]
                put(role+"_id", rule["value"], [roles[role]["confidence"], validity["confidence"]],
                    rule["provenance"])
            elif not optional:
                raise ValueError("Missing entity")
            return
        match = prediction.get("entities", {}).get(role, {})
        if (match.get("accepted") is not True or not probability(match.get("confidence"))
                or match.get("entity_id") not in catalog_ids(catalog)[role]):
            raise ValueError("Unresolved named entity")
        put(role+"_id", match["entity_id"], [roles[role]["confidence"], match["confidence"],
                                           validity["confidence"]], "acoustic_enrollment")

    try:
        if intent in ("LIGHT_ON", "LIGHT_OFF", "BRIGHTNESS", "COLOR"):
            entity("light", default=True)
        if intent == "CALL":
            entity("contact", default=True)
        if intent == "PLAY_MUSIC":
            entity("song", optional=True)
            entity("artist", optional=True)
        if intent in ("BRIGHTNESS", "COLOR", "TEMPERATURE"):
            role, field = {"BRIGHTNESS": ("brightness_percent", "percent"),
                           "COLOR": ("color", "color"),
                           "TEMPERATURE": ("temperature_degrees", "degrees")}[intent]
            value, scores = finite(role)
            put(field, value, scores, "acoustic_finite_role")
            if intent == "TEMPERATURE":
                put("unit", "C", [validity["confidence"], roles[role]["confidence"]], "schema_constant")
        if intent == "VOLUME":
            operation, evidence = finite("volume_operation")
            put("operation", operation, evidence, "acoustic_finite_role")
            if operation == "set":
                value, scores = finite("volume_percent")
                put("percent", value, scores, "acoustic_finite_role")
            elif state("volume_percent") != "absent":
                raise ValueError("Percent inconsistent with volume operation")
        if intent == "TIMER":
            amount, a = finite("timer_amount")
            unit, b = finite("timer_unit")
            put("duration_seconds", amount*contract["grammar"]["timer_unit_seconds"][unit],
                a+b, "recognized_amount_unit_pair")
        if intent == "ALARM":
            hour, h = finite("alarm_hour")
            minute, m = finite("alarm_minute", optional=True)
            meridiem, a = finite("alarm_meridiem")
            minute = 0 if minute is None else minute
            hour = hour % 12 + (12 if meridiem == "PM" else 0)
            put("time", f"{hour:02}:{minute:02}", h+m+a, "recognized_clock_with_whole_hour_rule")
            put("timezone", "Asia/Manila", [validity["confidence"]], "schema_constant")
        if intent == "CREATE_REMINDER":
            entity("task")
        command = {"intent": intent, "slots": slots}
        validate_command(command)
    except (ValueError, TypeError, KeyError, OverflowError):
        return reject("invalid_argument_or_slot_confidence")
    if diagnostic:
        return {**result, "accepted": False, "command": None,
                "candidate": command, "reason": "diagnostic_only"}
    return {**result, "accepted": True, "command": command}


def decide(prediction, contract, catalog, policy):
    return _decision(prediction, contract, catalog, policy)


def propose(prediction, contract, catalog):
    """Calibration diagnostics can never serialize as an accepted command."""
    return _decision(prediction, contract, catalog, {}, diagnostic=True)


def prediction_from_tensors(tensors, contract, memory=None, policy=None):
    """Only model outputs enter decoding; expected labels never enter this API."""
    policy = policy or {}

    def softmax(x, temperature=1):
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all() or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Nonfinite graph output/calibration")
        y = np.exp(x/temperature - np.max(x/temperature, axis=-1, keepdims=True))
        return y / y.sum(axis=-1, keepdims=True)

    intent = softmax(tensors["intent_logits"], policy.get("intent_temperature", 1))
    roles = softmax(tensors["role_state_logits"])
    validity = softmax(tensors["request_validity_logits"])
    result = {"intent": contract["intent_order"][int(intent.argmax())],
              "intent_confidence": float(intent.max()),
              "validity": {"state": contract["request_validity_states"][int(validity.argmax())],
                           "confidence": float(validity.max())},
              "roles": {}, "values": {}, "entities": {}}
    for n, role in enumerate(contract["roles"]):
        result["roles"][role] = {
            "state": contract["role_states"][int(roles[n].argmax())],
            "confidence": float(roles[n].max())}
    for role, logits in tensors["value_logits"].items():
        scores = softmax(logits)
        result["values"][role] = {
            "value": contract["finite_values"][role][int(scores.argmax())],
            "confidence": float(scores.max())}
    for n, role in enumerate(contract["entity_roles"]):
        result["entities"][role] = match_entity(
            tensors["entity_embeddings"][n], role, memory, policy.get("entities", {}).get(role))
    return result
