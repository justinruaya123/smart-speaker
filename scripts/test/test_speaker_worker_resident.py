"""Real WAV/IPC tests; recognizer doubles do not establish acoustic quality."""
from __future__ import annotations

import base64
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf


WORKER = Path(__file__).resolve().parents[1] / "runtime/speaker_worker.py"


def make_wav(samples: int = 1600, channels: int = 1, rate: int = 16000):
    """Return original PCM-16 WAV bytes, not a mocked decoder result."""
    shape = samples if channels == 1 else (samples, channels)
    stream = io.BytesIO()
    sf.write(stream, np.full(shape, 0.1), rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


@pytest.fixture
def worker(monkeypatch):
    """Isolate routing from unmounted shipping weights and audio hardware."""
    legacy = ModuleType("scripts.runtime.pcv_final_runtime")
    legacy.Runtime = None

    def forbidden_gate(*args, **kwargs):
        pytest.fail("Experimental command-only service invoked a wake backend")

    legacy.make_gate = forbidden_gate
    monkeypatch.setitem(sys.modules, legacy.__name__, legacy)
    spec = importlib.util.spec_from_file_location(
        "resident_worker_under_test", WORKER,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def settings():
    return {
        "inference": {
            "mode": "demo", "engine": "ncnn", "intent_threshold": 0.1,
            "slot_threshold": 0.2, "threads": 1,
        },
        "actions": {"enabled": True},
        "ui": {"debug": True, "readout": True},
    }


class Candidate:
    action_free = True

    def __init__(self, bundle=Path("candidate"), engine="ncnn", threads=1):
        self.root = Path(bundle)
        self.engine = engine
        self.contract = {
            "contract_hash": "contract-fixture", "custom": {},
            "thresholds": {"intent": 0.95, "slot": 0.90},
        }
        self.recipe = {
            "checkpoint_source_sha256": "checkpoint-fixture",
            "checkpoint_status": "software_fixture",
            "contract_hash": "contract-fixture",
        }
        self.policy = {"require_calibration": False}
        self.calls = []
        self.accepted = False

    def recognize(self, pcm):
        self.calls.append(np.array(pcm, copy=True))
        command = {"intent": "TIME", "slots": {}} if self.accepted else None
        application = {
            "schema_version": 3, "accepted": self.accepted,
            "command": command, "reason": "fixture_only",
        }
        return {
            "event": "command", "output": application,
            "application": application, "final_application": application,
            "diagnostic_candidate": {"intent": "TIME", "slots": {}},
            "actions_enabled": False,
        }


def run_resident(worker, monkeypatch, capsys, settings, requests, argv=None):
    instances = []
    runtime_module = ModuleType("scripts.runtime.vcm_intent_slots_runtime")

    def build(*args):
        instance = Candidate(*args)
        instances.append(instance)
        return instance

    runtime_module.IntentSlotsRuntime = build
    config = ModuleType("scripts.runtime.speaker_config")
    config.resolve_config = lambda *args, **kwargs: (deepcopy(settings), {})
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    monkeypatch.setitem(sys.modules, config.__name__, config)
    monkeypatch.delenv("PCV_BENCHMARK_LOG", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "speaker_worker", "--bundle", "candidate",
        "--experimental-intent-slots",
        *(argv or []),
    ])
    lines = [
        json.dumps(item) if isinstance(item, dict) else item
        for item in requests
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    worker.main()
    events = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    return instances, events


def wav_request(identifier="wav-1", **extra):
    return {
        "op": "wav", "id": identifier,
        "data": base64.b64encode(make_wav()).decode("ascii"), **extra,
    }


def test_resident_loads_once_and_keeps_authoritative_rejections(
    worker, monkeypatch, capsys, settings,
):
    instances, events = run_resident(worker, monkeypatch, capsys, settings, [
        wav_request("first"), wav_request("second"), {"op": "shutdown"},
    ])
    assert len(instances) == 1
    assert len(instances[0].calls) == 2
    commands = [event for event in events if event.get("event") == "command"]
    assert len(commands) == 2
    assert [event["request_id"] for event in commands] == ["first", "second"]
    assert all(
        event["final_application"]["accepted"] is False for event in commands
    )
    assert all(event["actions_enabled"] is False for event in events)
    assert all(
        event["application"] == event["final_application"]
        for event in commands
    )
    assert [event["sequence"] for event in events] == list(
        range(1, len(events) + 1)
    )
    assert len({event["session_id"] for event in events}) == 1


def test_ready_and_status_report_candidate_not_legacy_configuration(
    worker, monkeypatch, capsys, settings,
):
    _, events = run_resident(worker, monkeypatch, capsys, settings, [
        {"op": "status", "id": "status-1"}, {"op": "shutdown"},
    ])
    ready = next(event for event in events if event.get("event") == "ready")
    status = next(event for event in events if event.get("id") == "status-1")
    for event in (ready, status):
        assert event["loaded"] is True
        assert event["model_sha256"] == "checkpoint-fixture"
        assert event["contract_hash"] == "contract-fixture"
        assert event["application_config"]["intent_threshold"] == 0.95
        assert event["application_config"]["slot_threshold"] == 0.90
        assert event["owner_enforced"] is False
        assert event["qualification_verified"] is False
        assert event["command_only"] is True
        assert event["session_load_ms"] >= 0
    assert status["ok"] is True


@pytest.mark.parametrize("control", [
    {"op": "listen", "device": 0, "id": "blocked"},
    wav_request("blocked", wake=True),
    wav_request("blocked", expected_command={"intent": "TIME", "slots": {}}),
    wav_request("blocked", expected_intent="TIME"),
    {"op": "unknown", "id": "blocked"},
])
def test_denied_controls_recover_without_inference_or_hardware(
    worker, monkeypatch, capsys, settings, control,
):
    instances, events = run_resident(worker, monkeypatch, capsys, settings, [
        control, wav_request("valid"), {"op": "shutdown"},
    ])
    replies = {
        event["id"]: event for event in events if event.get("event") == "reply"
    }
    assert replies["blocked"]["ok"] is False
    assert replies["valid"]["ok"] is True
    assert len(instances[0].calls) == 1
    assert all(event["actions_enabled"] is False for event in events)


def test_bad_base64_is_a_fault_and_next_request_still_runs(
    worker, monkeypatch, capsys, settings,
):
    instances, events = run_resident(worker, monkeypatch, capsys, settings, [
        {"op": "wav", "id": "bad", "data": "!!not base64!!"},
        wav_request("valid"), {"op": "shutdown"},
    ])
    replies = {
        event["id"]: event for event in events if event.get("event") == "reply"
    }
    assert replies["bad"]["ok"] is False
    assert len(instances[0].calls) == 1
    commands = [event for event in events if event.get("event") == "command"]
    assert len(commands) == 1


@pytest.mark.parametrize("samples", [1, 399, 128001, 240000])
def test_candidate_audio_contract_enforced_before_recognizer(
    worker, settings, samples,
):
    candidate = Candidate()
    service = worker.AudioService(candidate, lambda event: None, settings)
    with pytest.raises(ValueError):
        service.wav(make_wav(samples))
    assert candidate.calls == []


@pytest.mark.parametrize("samples", [400, 401, 128000])
def test_candidate_valid_audio_edges_preserve_samples(
    worker, settings, samples,
):
    candidate = Candidate()
    events = []
    service = worker.AudioService(candidate, events.append, settings)
    service.wav(make_wav(samples))
    assert len(candidate.calls[0]) == samples
    assert service.state == "awaiting_microphone"
    assert all(event["actions_enabled"] is False for event in events)


@pytest.mark.parametrize("operation", ["listen", "new_gate", "wake_wav"])
def test_direct_service_forbids_unqualified_live_path(
    worker, settings, operation,
):
    service = worker.AudioService(Candidate(), lambda event: None, settings)
    with pytest.raises(ValueError, match="command-only"):
        if operation == "listen":
            service.listen(0)
        elif operation == "new_gate":
            service.new_gate()
        else:
            service.wav(make_wav(), wake=True)


@pytest.mark.parametrize("channels, rate, samples", [
    (2, 16000, 400), (1, 8000, 400), (1, 16000, 256000),
])
def test_existing_wav_boundary_is_preserved(worker, channels, rate, samples):
    with pytest.raises(ValueError):
        worker.decode_wav(make_wav(samples, channels, rate))


def test_accepted_fixture_still_cannot_enable_provider_actions(
    worker, settings,
):
    candidate = Candidate()
    candidate.accepted = True
    events = []
    service = worker.AudioService(candidate, events.append, settings)
    service.wav(make_wav())
    command = next(
        event for event in events if event.get("event") == "command"
    )
    assert command["final_application"]["accepted"] is True
    assert command["actions_enabled"] is False


def test_legacy_wav_does_not_gain_candidate_duration_restriction(worker):
    candidate = Candidate()
    candidate.action_free = False
    service = worker.AudioService(candidate, lambda event: None)
    assert service.wav(make_wav(160000))["commands"] == 1


def test_malformed_json_recovers_then_eof_shuts_down(
    worker, monkeypatch, capsys, settings,
):
    instances, events = run_resident(worker, monkeypatch, capsys, settings, [
        "{broken", wav_request(),
    ])
    assert len(instances[0].calls) == 1
    assert any(event.get("event") == "error" for event in events)


def test_many_requests_use_one_model(worker, monkeypatch, capsys, settings):
    instances, events = run_resident(worker, monkeypatch, capsys, settings, [
        *[wav_request(str(index)) for index in range(32)], {"op": "shutdown"},
    ])
    assert len(instances) == 1
    assert len(instances[0].calls) == 32
    assert sum(event.get("event") == "command" for event in events) == 32


def test_legacy_ready_fields_remain_compatible(worker, settings):
    candidate = Candidate()
    candidate.action_free = False
    candidate.recipe = {"binding": {"model_sha256": "legacy-model"}}
    candidate.policy = {"enabled": True}
    candidate.load_ms = 12.5
    status = worker.runtime_status(candidate, settings, 1, 99)
    assert status["model_sha256"] == "legacy-model"
    assert status["policy_enabled"] is True
    assert status["session_load_ms"] == 12.5
    assert "qualification_verified" not in status


@pytest.mark.parametrize("arguments", [
    ["--threads", "0"], ["--threads", "9"], ["--wake"], ["--list-mics"],
])
def test_invalid_startup_controls_fail_before_loading(
    worker, monkeypatch, capsys, settings, arguments,
):
    with pytest.raises(SystemExit) as error:
        run_resident(worker, monkeypatch, capsys, settings, [], arguments)
    assert error.value.code == 2


def test_existing_one_shot_candidate_wav_remains_available(
    worker, monkeypatch, capsys, settings, tmp_path,
):
    audio = tmp_path / "command.wav"
    audio.write_bytes(make_wav())
    instances, events = run_resident(
        worker, monkeypatch, capsys, settings, [], ["--wav", str(audio)],
    )
    assert len(instances[0].calls) == 1
    assert sum(event.get("event") == "command" for event in events) == 1
    assert not any(event.get("event") == "ready" for event in events)


def test_smoke_verifier_accepts_real_worker_protocol_events(
    worker, monkeypatch, capsys, settings,
):
    _, events = run_resident(worker, monkeypatch, capsys, settings, [
        wav_request("wav-0"), wav_request("wav-1"),
        {"op": "status", "id": "status"}, {"op": "shutdown"},
    ])
    path = Path(__file__).with_name("vcm_resident_smoke.py")
    spec = importlib.util.spec_from_file_location("smoke_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify_traffic(
        events, ["wav-0", "wav-1"], "ncnn", "checkpoint-fixture",
        "contract-fixture",
    )
    assert result["commands"] == 2
    assert result["acoustic_qualification"] is False


def test_reader_fault_does_not_inherit_active_wav_request(
    worker, monkeypatch, capsys, settings,
):
    import time

    original = Candidate.recognize

    def delayed(self, pcm):
        time.sleep(0.02)
        return original(self, pcm)

    monkeypatch.setattr(Candidate, "recognize", delayed)
    _, events = run_resident(worker, monkeypatch, capsys, settings, [
        *[wav_request(str(index)) for index in range(9)], "{broken",
        {"op": "shutdown"},
    ])
    errors = [event for event in events if event.get("event") == "error"]
    assert len(errors) == 1
    assert "request_id" not in errors[0]


def test_slow_inference_preserves_all_queued_wav_requests(
    worker, monkeypatch, capsys, settings,
):
    import time

    original = Candidate.recognize

    def delayed_first(self, pcm):
        if not self.calls:
            time.sleep(1.4)
        return original(self, pcm)

    monkeypatch.setattr(Candidate, "recognize", delayed_first)
    identifiers = [str(index) for index in range(10)]
    instances, events = run_resident(
        worker, monkeypatch, capsys, settings,
        [*[wav_request(identifier) for identifier in identifiers],
         {"op": "status", "id": "status"}, {"op": "shutdown"}],
    )
    assert len(instances[0].calls) == len(identifiers)
    commands = [event for event in events if event.get("event") == "command"]
    assert [event["request_id"] for event in commands] == identifiers
    replies = [event for event in events if event.get("event") == "reply"]
    assert [event["id"] for event in replies] == [*identifiers, "status"]
    assert all(event["ok"] is True for event in replies)
    assert not any(event.get("event") == "error" for event in events)


@pytest.mark.parametrize("identifier", ["1e400", "NaN", '["nested",NaN]'])
def test_invalid_request_id_rejects_and_next_wav_still_runs(
    worker, monkeypatch, capsys, settings, identifier,
):
    requests = [
        '{"op":"status","id":' + identifier + '}',
        wav_request("next"), {"op": "shutdown"},
    ]
    try:
        instances, events = run_resident(
            worker, monkeypatch, capsys, settings, requests,
        )
    except ValueError as error:
        pytest.fail("Invalid request ID terminated the service: " + str(error))
    assert len(instances[0].calls) == 1
    replies = [event for event in events if event.get("event") == "reply"]
    assert len(replies) == 2
    assert replies[0]["ok"] is False
    assert replies[0]["id"] is None
    assert replies[1]["ok"] is True
    assert replies[1]["id"] == "next"
    commands = [event for event in events if event.get("event") == "command"]
    assert [event["request_id"] for event in commands] == ["next"]
