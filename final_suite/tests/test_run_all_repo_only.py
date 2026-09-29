"""Repository-only finalization must fail closed, not fail to start."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_all", ROOT / "final_suite/run_all.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_historical_payload_is_optional(tmp_path):
    checks = MODULE.plan_checks(root=tmp_path, include_browser=False)
    assert any(check["name"] == "source_tests" and check["required"] for check in checks)
    historical = [check for check in checks if check["name"] == "historical_evidence"]
    assert historical and historical[0]["required"] is False
    assert historical[0]["ready"] is False


def test_qualification_is_evaluated_without_summary(tmp_path):
    snapshot = MODULE.qualification_snapshot(tmp_path)
    assert snapshot["evaluated"] is True
    assert snapshot["release"]["eligible"] is False
    assert "human_test_passed" in snapshot["release"]["failed_requirements"]
    assert snapshot["evidence_source"] == "fail_closed_repository_state"


def test_optional_checks_do_not_make_software_failure():
    evidence = [
        {"name": "source_tests", "status": "passed", "returncode": 0},
        {"name": "historical_evidence", "status": "skipped", "returncode": None},
    ]
    assert MODULE.software_checks_passed(evidence) is True
