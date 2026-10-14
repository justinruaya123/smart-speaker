import pytest

from scripts.runtime.pcv_benchmark_adapter import log_record, class_prediction
from scripts.runtime.pcv_schema import baseline_cases, extended_cases


def event(application):
    return {"event": "command", "output": {"accepted": False, "command": None},
            "application": application, "final_application": application,
            "diagnostic_candidate": {"candidate": {"intent": "TIME", "slots": {}}},
            "audio_seconds": 1., "timing": {"total_ms": 6.}}


def test_named_decision_selection_preserves_rejected_diagnostics():
    rejected = {"accepted": False, "command": None, "reason": "intent_confidence"}
    accepted = {"accepted": True, "command": {"intent": "STOP", "slots": {}}}
    e = event(accepted)
    assert log_record(e)["intent"] == "OUT_OF_SCOPE"
    assert log_record(e, decision="final_application")["intent"] == "STOP"
    assert log_record(event(rejected), decision="final_application")["intent"] == "OUT_OF_SCOPE"
    with pytest.raises(ValueError):
        log_record({"output": accepted, "audio_seconds": 1., "timing": {"total_ms": 6.}}, decision="final_application")


def test_capture_fault_and_dynamic_task_catalog_are_preserved():
    e = event({"accepted": False, "command": None, "reason": "capture_fault"})
    assert log_record(e, decision="final_application")["event"] == "capture_fault"
    cmd = {"intent": "CREATE_REMINDER", "slots": {"task_id": "task.new"}}
    assert class_prediction(cmd, custom={"tasks": [{"id": "task.new", "name": "Take medicine"}]}) == ("CREATE_REMINDER", "Take medicine")


def test_all_contract_definitions_use_same_projection():
    for case in baseline_cases()+extended_cases():
        cmd = case["command"]
        out = log_record(event({"accepted": cmd is not None, "command": cmd}), decision="final_application")
        assert out["intent"] == class_prediction(cmd)[0]
