"""Pi evidence updates preserve the frozen candidate and prior failures."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "pi_finalize.py"


def module():
    assert SCRIPT.is_file(), "Pi card updater has not been implemented"
    spec = importlib.util.spec_from_file_location("pi_finalize_under_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    bindings = {"checkpoint_sha256": "a" * 64, "recipe_sha256": "b" * 64,
                "contract_hash": "c" * 64}
    card = {"format": "vcm-model-card-v2", "bindings": bindings,
            "architecture": {"thresholds": {"intent": 0.95, "slot": 0.90}},
            "training": {"updates": 2880}, "calibration": {"separate": True},
            "benchmarks": {"human": {"correct": 93, "missed_verbatim": ["call home"]}},
            "gates": {"accuracy_ready": True, "tts_reviewed": True},
            "pi": {"status": "deferred", "results": None},
            "status": "accuracy-ready / Pi-deferred", "unknown_extension": {"keep": [1, None]}}
    receipt = {"kind": "physical_pi_benchmark", "bindings": bindings,
               "hardware": {"model": "Raspberry Pi 5", "memory_bytes": 8589934592},
               "environment": {"os": "Pi OS 64-bit", "python": "3.13.9", "engine": "ncnn"},
               "metrics": {"trials": 12, "supported_n": 10, "negative_n": 2,
                           "correct": 9, "false_accepts": 0, "faults": 0, "missing": 0,
                           "all_slots_n": 8, "all_slots_correct": 7, "p95_ms": 80.5},
               "gates": {"benchmark_complete": True, "accuracy": True}}
    return write(tmp_path / "baseline.json", card), write(tmp_path / "pi-receipt.json", receipt), card, receipt


def test_updates_actual_pi_fields_and_preserves_every_non_pi_field(inputs, tmp_path):
    baseline, receipt_path, card, receipt = inputs
    original = baseline.read_bytes()
    result = module().finalize(baseline, receipt_path, tmp_path / "results")
    expected = deepcopy(card)
    expected.pop("pi")
    expected["status"] = "accuracy-ready / Pi-validated"
    assert {k: v for k, v in result.items() if k != "pi"} == expected
    assert result["pi"]["status"] == "passed"
    assert result["pi"]["results"]["metrics"] == receipt["metrics"]
    assert result["pi"]["results"]["hardware"] == receipt["hardware"]
    assert result["pi"]["receipt"]["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert result["pi"]["baseline"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert baseline.read_bytes() == original
    assert json.loads((tmp_path / "results/card.json").read_text()) == result
    assert "call home" in (tmp_path / "results/MODEL_CARD.md").read_text()


def test_repeat_update_is_deterministic(inputs, tmp_path):
    baseline, receipt_path, _, _ = inputs
    updater = module()
    updater.finalize(baseline, receipt_path, tmp_path / "results")
    before = [(tmp_path / "results" / name).read_bytes() for name in ("card.json", "MODEL_CARD.md")]
    updater.finalize(baseline, receipt_path, tmp_path / "results")
    assert [(tmp_path / "results" / name).read_bytes() for name in ("card.json", "MODEL_CARD.md")] == before


@pytest.mark.parametrize("change", ["gate", "fault", "missing"])
def test_failure_cannot_be_promoted_to_a_pass(inputs, tmp_path, change):
    baseline, receipt_path, _, receipt = inputs
    if change == "gate":
        receipt["gates"]["accuracy"] = False
    else:
        receipt["metrics"][change + ("s" if change == "fault" else "")] = 1
    write(receipt_path, receipt)
    result = module().finalize(baseline, receipt_path, tmp_path / "results")
    assert result["pi"]["status"] == "failed"
    assert result["status"] == "accuracy-ready / Pi-failed"
    assert result["pi"]["results"]["metrics"] == receipt["metrics"]


def test_pi_pass_preserves_blocked_non_pi_status(inputs, tmp_path):
    baseline, receipt_path, card, _ = inputs
    card["gates"]["accuracy_ready"] = False
    card["status"] = "failed: human complete-command gate"
    write(baseline, card)
    result = module().finalize(baseline, receipt_path, tmp_path / "results")
    assert result["status"] == "failed: human complete-command gate"
    assert result["gates"] == card["gates"]
    assert result["pi"]["status"] == "passed"


@pytest.mark.parametrize("change", ["passed", "failed_gate", "fault", "missing"])
def test_blocked_headline_reflects_pi_receipt_with_immutable_non_pi_findings(inputs, tmp_path, change):
    baseline, receipt_path, card, receipt = inputs
    card["gates"]["accuracy_ready"] = False
    card["gates"]["acoustic_review"] = {"status": "blocked", "reviewed_spans": 0}
    card["failed_gates"] = ["acoustic review: blocked", "fresh human intent scope: failed"]
    card["qualification"] = False
    card["status"] = "blocked / Pi-deferred"
    if change == "failed_gate":
        receipt["gates"]["accuracy"] = False
    elif change == "fault":
        receipt["metrics"]["faults"] = 1
    elif change == "missing":
        receipt["metrics"]["missing"] = 1
    write(baseline, card)
    write(receipt_path, receipt)
    baseline_bytes, receipt_bytes = baseline.read_bytes(), receipt_path.read_bytes()
    updater = module()
    result = updater.finalize(baseline, receipt_path, tmp_path / "results")
    expected = "blocked / Pi-passed" if change == "passed" else "blocked / Pi-failed"
    assert result["status"] == expected
    assert result["pi"]["status"] == ("passed" if change == "passed" else "failed")
    non_pi = lambda value: {key: child for key, child in value.items() if key not in ("pi", "status")}
    assert json.dumps(non_pi(result), sort_keys=True) == json.dumps(non_pi(card), sort_keys=True)
    assert result["gates"]["accuracy_ready"] is False and result["qualification"] is False
    assert result["failed_gates"] == card["failed_gates"]
    assert result["benchmarks"]["human"]["missed_verbatim"] == ["call home"]
    assert result["pi"]["results"] == receipt
    assert result["pi"]["baseline"]["sha256"] == hashlib.sha256(baseline_bytes).hexdigest()
    assert result["pi"]["receipt"]["sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
    assert baseline.read_bytes() == baseline_bytes and receipt_path.read_bytes() == receipt_bytes
    output_paths = [tmp_path / "results" / name for name in ("card.json", "MODEL_CARD.md")]
    derived_bytes = [path.read_bytes() for path in output_paths]
    assert "Status: " + expected in derived_bytes[1].decode()
    assert "call home" in derived_bytes[1].decode()
    updater.finalize(baseline, receipt_path, tmp_path / "results")
    assert [path.read_bytes() for path in output_paths] == derived_bytes
    assert baseline.read_bytes() == baseline_bytes and receipt_path.read_bytes() == receipt_bytes


@pytest.mark.parametrize("field", ["checkpoint_sha256", "recipe_sha256", "contract_hash"])
def test_mixed_candidate_receipt_is_rejected(inputs, tmp_path, field):
    baseline, receipt_path, _, receipt = inputs
    receipt["bindings"][field] = "d" * 64
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="binding"):
        module().finalize(baseline, receipt_path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("field,value", [
    ("supported_n", 0), ("negative_n", 0), ("faults", None), ("missing", -1),
    ("correct", 11), ("false_accepts", 3), ("trials", 11), ("faults", True),
])
def test_missing_invalid_or_inconsistent_denominators_rejected(inputs, tmp_path, field, value):
    baseline, receipt_path, _, receipt = inputs
    receipt["metrics"][field] = value
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="metric|denominator"):
        module().finalize(baseline, receipt_path, tmp_path / "results")


@pytest.mark.parametrize("field,value", [
    ("kind", "host_replay"), ("hardware", {}), ("environment", {}),
    ("gates", {}), ("gates", {"complete": "true"}),
])
def test_unmeasured_or_untyped_evidence_is_rejected(inputs, tmp_path, field, value):
    baseline, receipt_path, _, receipt = inputs
    receipt[field] = value
    write(receipt_path, receipt)
    with pytest.raises(ValueError):
        module().finalize(baseline, receipt_path, tmp_path / "results")


def test_nonfinite_receipt_and_baseline_overwrite_are_rejected(inputs):
    baseline, receipt_path, _, receipt = inputs
    receipt["metrics"]["p95_ms"] = float("nan")
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match="finite"):
        module().finalize(baseline, receipt_path, baseline.parent)
    receipt["metrics"]["p95_ms"] = 1.0
    write(receipt_path, receipt)
    renamed = baseline.with_name("card.json")
    renamed.write_bytes(baseline.read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        module().finalize(renamed, receipt_path, renamed.parent)


@pytest.mark.parametrize("which", ["baseline", "receipt"])
def test_duplicate_json_evidence_fields_are_not_silently_resolved(inputs, tmp_path, which):
    baseline, receipt_path, _, _ = inputs
    path = baseline if which == "baseline" else receipt_path
    text = path.read_text()
    key = '"format": "vcm-model-card-v2"' if which == "baseline" else '"kind": "physical_pi_benchmark"'
    path.write_text(text.replace(key, key + ", " + key))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        module().finalize(baseline, receipt_path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("field,value", [
    ("bindings", None), ("bindings", []),
    ("hardware", {"model": None}), ("environment", {"engine": ""}),
])
def test_placeholder_or_untyped_receipt_fields_are_rejected(inputs, tmp_path, field, value):
    baseline, receipt_path, _, receipt = inputs
    receipt[field] = value
    write(receipt_path, receipt)
    with pytest.raises(ValueError):
        module().finalize(baseline, receipt_path, tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_card_update_cli_needs_no_inference_dependencies(inputs, tmp_path):
    baseline, receipt_path, _, _ = inputs
    result = subprocess.run([sys.executable, "-S", "-B", str(SCRIPT), "--baseline", str(baseline),
                             "--receipt", str(receipt_path), "--output", str(tmp_path / "results")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "accuracy-ready / Pi-validated"
    assert json.loads((tmp_path / "results/card.json").read_text())["pi"]["results"]["metrics"]["correct"] == 9
