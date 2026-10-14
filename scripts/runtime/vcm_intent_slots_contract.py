"""Versioned direct acoustic heads; schema validation never parses transcripts."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command
from vcm.frontend import SPEC

VERSION = "vcm-intent-slots-v1"
FINITE = {
    "brightness_percent": list(range(101)),
    "volume_operation": ["up", "down", "mute", "set"],
    "volume_percent": list(range(101)),
    "timer_unit": ["seconds", "minutes", "hours"],
    "alarm_hour": list(range(1, 13)),
    "alarm_minute": list(range(60)),
    "alarm_meridiem": ["AM", "PM"],
    "temperature_degrees": list(range(-50, 61)),
    "color": ["red", "blue", "green", "yellow", "white", "orange", "purple",
              "pink", "cyan", "magenta", "warm white", "cool white"],
}
ENTITY_ROLES = ["light", "song", "artist", "contact", "task"]
APPLICABLE = {
    "LIGHT_ON": ["light"], "LIGHT_OFF": ["light"], "CALL": ["contact"],
    "PLAY_MUSIC": ["song", "artist"],
    "VOLUME": ["volume_operation", "volume_percent"],
    "TIMER": ["timer_amount", "timer_unit"],
    "ALARM": ["alarm_hour", "alarm_minute", "alarm_meridiem"],
    "TEMPERATURE": ["temperature_degrees"],
    "BRIGHTNESS": ["light", "brightness_percent"],
    "COLOR": ["light", "color"], "CREATE_REMINDER": ["task"],
}


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def catalog_digest(catalog):
    return digest(catalog.get("entries", catalog) if isinstance(catalog, dict) else catalog)


def build_slot_contract(cases, catalog_entries, custom):
    """Bind the complete supplied catalog, including additional historical entries.

    Timer amounts are the union of exact integer representations of declared
    durations in seconds/minutes/hours. This finite inventory is reported,
    not claimed to cover every value up to the schema's 86400-second bound.
    """
    entries = deepcopy(catalog_entries)
    for row in [*cases, *entries]:
        if row.get("command") is not None:
            validate_command(row["command"])
    finite = deepcopy(FINITE)
    durations = {r["command"]["slots"]["duration_seconds"]
                 for r in [*cases, *entries]
                 if r.get("command") and r["command"]["intent"] == "TIMER"}
    finite["timer_amount"] = sorted({s // unit for s in durations
                                    for unit in (1, 60, 3600) if s % unit == 0})
    if not finite["timer_amount"]:
        raise ValueError("Missing declared timer inventory")
    roles = list(finite) + ENTITY_ROLES
    result = {
        "version": VERSION, "frontend": dict(SPEC),
        "intent_order": ["OOV", *INTENTS], "roles": roles,
        "entity_roles": ENTITY_ROLES, "finite_values": finite,
        "role_states": ["absent", "present", "conflicting", "unsupported"],
        "request_validity_states": ["valid", "missing", "conflicting", "unsupported"],
        "applicable_roles": {i: APPLICABLE.get(i, []) for i in INTENTS},
        "cases": deepcopy(cases), "catalog_entries": entries,
        "catalog_hash": catalog_digest(entries), "custom": deepcopy(custom),
        "custom_hash": digest(custom),
        "defaults": {
            "light": {"value": "light.all", "provenance": "qualified_generic_request"},
            "contact": {"value": "contact.home", "provenance": "qualified_bare_call"},
            "timezone": {"value": "Asia/Manila", "provenance": "schema_constant"},
            "unit": {"value": "C", "provenance": "schema_constant"},
            "alarm_minute": {"value": 0, "provenance": "qualified_whole_hour_rule"},
        },
        "grammar": {"version": "typed-v1", "timer_unit_seconds": {
            "seconds": 1, "minutes": 60, "hours": 3600},
            "alarm": "1..12 hour plus AM/PM; absent minute requires whole-hour rule",
            "timer_amount_coverage": "declared catalog values only; unqualified until heard"},
        "thresholds": {"intent": 0.95, "slot": 0.90},
    }
    offset = 0
    result["value_offsets"] = {}
    for role, values in finite.items():
        result["value_offsets"][role] = [offset, offset + len(values)]
        offset += len(values)
    result["contract_hash"] = digest(result)
    return result


def verify_contract(contract):
    payload = {k: v for k, v in contract.items() if k != "contract_hash"}
    if (contract.get("version") != VERSION or contract.get("frontend") != SPEC
            or contract.get("intent_order") != ["OOV", *INTENTS]
            or contract.get("contract_hash") != digest(payload)):
        raise ValueError("Invalid contract/frontend/intent binding")
    if contract["catalog_hash"] != catalog_digest(contract["catalog_entries"]):
        raise ValueError("Invalid catalog binding")
