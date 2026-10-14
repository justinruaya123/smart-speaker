"""Protocol fixtures for the real resident-worker smoke runner."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def load_smoke():
    path = Path(__file__).with_name("vcm_resident_smoke.py")
    assert path.is_file(), "Resident smoke runner has not been implemented"
    spec = importlib.util.spec_from_file_location("resident_smoke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def traffic():
    identity = {
        "loaded": True, "engine": "ncnn", "model_sha256": "a" * 64,
        "contract_hash": "b" * 64, "qualification_verified": False,
        "command_only": True, "owner_enforced": False,
        "application_config": {
            "intent_threshold": 0.95, "slot_threshold": 0.90,
        },
    }
    reject = {"accepted": False, "command": None, "reason": "fixture"}
    rows = [
        {"event": "ready", **identity},
        {
            "event": "command", "request_id": "wav-0", "source": "wav",
            "final_application": reject, "application": reject,
        },
        {"event": "reply", "id": "wav-0", "ok": True},
        {"event": "reply", "id": "status", "ok": True, **identity},
    ]
    return [{
        **row, "sequence": index + 1, "session_id": "session",
        "actions_enabled": False,
    } for index, row in enumerate(rows)]


def test_valid_traffic_only_proves_protocol(traffic):
    smoke = load_smoke()
    result = smoke.verify_traffic(
        traffic, ["wav-0"], "ncnn", "a" * 64, "b" * 64,
    )
    assert result["status"] == "passed_service_smoke"
    assert result["commands"] == 1
    assert result["accepted"] == 0
    assert result["rejected"] == 1
    assert result["acoustic_qualification"] is False
    assert result["physical_pi_qualification"] is False


@pytest.mark.parametrize("index, key, value", [
    (0, "engine", "onnxruntime"),
    (0, "model_sha256", "c" * 64),
    (0, "contract_hash", "c" * 64),
    (0, "owner_enforced", True),
    (0, "qualification_verified", True),
    (1, "actions_enabled", True),
    (1, "session_id", "other"),
    (1, "sequence", 5),
    (1, "request_id", "wrong"),
    (1, "source", "microphone"),
    (2, "ok", False),
    (3, "engine", "onnxruntime"),
])
def test_bad_or_misattributed_traffic_fails(traffic, index, key, value):
    smoke = load_smoke()
    traffic[index][key] = value
    with pytest.raises(ValueError):
        smoke.verify_traffic(traffic, ["wav-0"], "ncnn", "a" * 64, "b" * 64)


def test_diagnostic_accept_cannot_replace_final(traffic):
    smoke = load_smoke()
    traffic[1]["application"] = {
        "accepted": True, "command": {"intent": "TIME", "slots": {}},
    }
    with pytest.raises(ValueError):
        smoke.verify_traffic(traffic, ["wav-0"], "ncnn", "a" * 64, "b" * 64)


def test_missing_and_duplicate_commands_fail(traffic):
    smoke = load_smoke()
    for trial in (traffic[:1] + traffic[2:], traffic + [deepcopy(traffic[1])]):
        with pytest.raises(ValueError):
            smoke.verify_traffic(trial, ["wav-0"], "ncnn", "a" * 64, "b" * 64)


def test_fixed_thresholds_are_verified(traffic):
    smoke = load_smoke()
    traffic[0]["application_config"]["slot_threshold"] = 0.2
    with pytest.raises(ValueError):
        smoke.verify_traffic(traffic, ["wav-0"], "ncnn", "a" * 64, "b" * 64)
