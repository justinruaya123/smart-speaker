import copy
import numpy as np
import pytest

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
from scripts.test.test_vcm_intent_slots_contract import contract
from scripts.runtime.vcm_intent_slots_entities import enroll_examples, match_entity
from scripts.runtime.vcm_intent_slots_decode import decide


def prediction(intent="STOP"):
    c = contract()
    return {"intent": intent, "intent_confidence": .99,
            "validity": {"state": "valid", "confidence": .99},
            "roles": {r: {"state": "absent", "confidence": .99} for r in c["roles"]},
            "values": {}, "entities": {}}


def present(p, role, value, confidence=.99):
    p["roles"][role] = {"state": "present", "confidence": .99}
    p["values"][role] = {"value": value, "confidence": confidence}
    return p


def entity_example(role="song", entity="song.dynamite", name="a", vector=None):
    v = vector if vector is not None else np.eye(1, 64, 0)[0].tolist()
    return {"role": role, "entity_id": entity, "embedding": v,
            "sha256": name*64, "pcm_sha256": name*64,
            "speaker_id": "human/"+name, "source": "original enrollment",
            "kind": "human"}


def test_enrollment_singleton_absolute_gate_duplicate_and_catalog_binding():
    c = contract()
    ex = entity_example()
    mem = enroll_examples([ex, ex], {"encoder": "a"*64, "frontend": "b"*64, "normalization": "c"*64}, c)
    assert len(mem["examples"]) == 1
    policy = {"absolute": .8, "margin": .1, "encoder_binding": mem["encoder_binding"],
              "catalog_hash": c["catalog_hash"], "catalog_sizes": [1]}
    assert match_entity(np.eye(1, 64, 0)[0], "song", mem, policy)["entity_id"] == "song.dynamite"
    assert not match_entity(np.eye(1, 64, 1)[0], "song", mem, policy)["accepted"]
    stale = {**policy, "catalog_hash": "0"*64}
    assert not match_entity(np.eye(1, 64, 0)[0], "song", mem, stale)["accepted"]


def test_ties_empty_stale_invalid_and_role_mismatch_reject():
    c = contract()
    binding = {"encoder": "a"*64, "frontend": "b"*64, "normalization": "c"*64}
    with pytest.raises(ValueError):
        enroll_examples([entity_example(entity="light.kitchen")], binding, c)
    with pytest.raises(ValueError):
        enroll_examples([entity_example(vector=[0]*64)], binding, c)
    mem = enroll_examples([entity_example(), entity_example(entity="song.butter", name="b")], binding, c)
    policy = {"absolute": .8, "margin": .1, "encoder_binding": binding,
              "catalog_hash": c["catalog_hash"], "catalog_sizes": [0, 2]}
    for v, role in [(np.eye(1, 64, 0)[0], "song"), (np.ones(64), "contact"),
                    (np.full(64, np.nan), "song")]:
        assert not match_entity(v, role, mem, policy)["accepted"]


def test_default_provenance_and_explicit_unknown_never_defaults():
    c = contract()
    for intent, field, value in [("LIGHT_ON", "light_id", "light.all"),
                                  ("CALL", "contact_id", "contact.home")]:
        p = prediction(intent)
        out = decide(p, c, c, {})
        assert out["command"]["slots"][field] == value
        assert "qualified" in out["slot_evidence"][field]["provenance"]
        role = "light" if field == "light_id" else "contact"
        p["roles"][role] = {"state": "present", "confidence": .99}
        assert not decide(p, c, c, {})["accepted"]


def test_music_artist_cannot_manufacture_song():
    c = contract()
    p = prediction("PLAY_MUSIC")
    assert decide(p, c, c, {})["accepted"]
    p["roles"]["artist"] = {"state": "present", "confidence": .99}
    p["entities"]["artist"] = {"accepted": True, "entity_id": "artist.bts", "confidence": .99}
    assert not decide(p, c, c, {})["accepted"]
    p["roles"]["song"] = {"state": "present", "confidence": .99}
    p["entities"]["song"] = {"accepted": False, "entity_id": None}
    assert not decide(p, c, c, {})["accepted"]


@pytest.mark.parametrize("confidence,accepted", [(.949999, False), (.95, True), (float("nan"), False), (None, False)])
def test_intent_gate_cannot_be_rescued(confidence, accepted):
    c = contract()
    p = prediction()
    p["intent_confidence"] = confidence
    assert decide(p, c, c, {})["accepted"] is accepted


@pytest.mark.parametrize("confidence,accepted", [(.899999, False), (.9, True), (float("inf"), False), (None, False)])
def test_every_supplied_slot_boundary(confidence, accepted):
    c = contract()
    p = present(prediction("VOLUME"), "volume_operation", "up", confidence)
    assert decide(p, c, c, {})["accepted"] is accepted


@pytest.mark.parametrize("fault", ["conflicting", "unsupported", "missing"])
def test_complete_request_and_extra_roles_reject(fault):
    c = contract()
    p = prediction()
    p["validity"]["state"] = fault
    assert not decide(p, c, c, {})["accepted"]
    p = present(prediction(), "brightness_percent", 20)
    assert not decide(p, c, c, {})["accepted"]


def test_numeric_rules_and_defaults():
    c = contract()
    p = present(prediction("BRIGHTNESS"), "brightness_percent", 60)
    assert decide(p, c, c, {})["command"]["slots"] == {"percent": 60, "light_id": "light.all"}
    p["values"]["brightness_percent"]["value"] = 101
    assert not decide(p, c, c, {})["accepted"]
    p = present(present(prediction("TIMER"), "timer_amount", 1), "timer_unit", "minutes")
    assert decide(p, c, c, {})["command"]["slots"]["duration_seconds"] == 60
    p = present(present(prediction("ALARM"), "alarm_hour", 9), "alarm_meridiem", "PM")
    result = decide(p, c, c, {})
    assert result["command"]["slots"] == {"time": "21:00", "timezone": "Asia/Manila"}
    assert result["slot_evidence"]["timezone"]["provenance"] == "schema_constant"
    assert result["score_status"] == "uncalibrated_selection_score"
    assert not decide(p, c, c, {"require_calibration": True})["accepted"]
