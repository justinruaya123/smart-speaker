"""Contract fixtures are software evidence, never acoustic qualification."""
import hashlib
import json

import pytest

from scripts.runtime.pcv_schema import (
    DEFAULT_CUSTOM, baseline_cases, canonical, extended_cases, validate_command,
)
from scripts.runtime.vcm_intent_slots_contract import build_slot_contract
from scripts.train.vcm_intent_slots_data import audit_views


def contract():
    cases = baseline_cases() + extended_cases()
    return build_slot_contract(cases, cases, DEFAULT_CUSTOM)


def test_optionb_and_extensions_preserved():
    result = contract()
    assert result["intent_order"][0] == "OOV"
    assert len(result["intent_order"]) == 19
    assert result["cases"] == baseline_cases() + extended_cases()
    assert len(baseline_cases()) == 93
    assert len({canonical(r["command"]) for r in baseline_cases()}) == 31
    assert result["catalog_entries"] == baseline_cases() + extended_cases()
    assert result["finite_values"]["brightness_percent"] == list(range(101))
    assert result["finite_values"]["temperature_degrees"] == list(range(-50, 61))
    assert "warm white" in result["finite_values"]["color"]
    assert "playlist" not in result["entity_roles"]
    assert result["defaults"]["light"]["value"] == "light.all"
    assert result["defaults"]["contact"]["value"] == "contact.home"
    assert result["defaults"]["timezone"]["provenance"] == "schema_constant"


def test_selected_larger_catalog_is_preserved():
    cases = baseline_cases() + extended_cases()
    larger = cases + [{"id": "extra", "command": {"intent": "TIMER", "slots": {"duration_seconds": 120}}}]
    assert build_slot_contract(cases, larger, DEFAULT_CUSTOM)["catalog_entries"] == larger


@pytest.mark.parametrize("cmd", [
    {"intent": "VOLUME", "slots": {"operation": "up", "percent": 20}},
    {"intent": "VOLUME", "slots": {"operation": "set"}},
    {"intent": "PLAY_MUSIC", "slots": {"artist_id": "artist.bts"}},
    {"intent": "PLAY_MUSIC", "slots": {"playlist_id": "playlist.work"}},
    {"intent": "BRIGHTNESS", "slots": {"light_id": "light.all", "percent": 101}},
    {"intent": "ALARM", "slots": {"time": "06:00", "timezone": "UTC"}},
    {"intent": "TEMPERATURE", "slots": {"degrees": 22, "unit": "F"}},
])
def test_schema_rejects_invalid_combinations(cmd):
    with pytest.raises(ValueError):
        build_slot_contract([{"command": cmd}], [], DEFAULT_CUSTOM)


def inputs(tmp_path, rows):
    view = tmp_path / "train.jsonl"
    view.write_text("".join(json.dumps(r) + "\n" for r in rows))
    digest = hashlib.sha256(view.read_bytes()).hexdigest()
    receipt = {"views": {"train": {"path": str(view), "sha256": digest}}}
    return {"train": view}, receipt


def row(**kw):
    return dict(id="one", audio="one.wav", sha256="a"*64, pcm_sha256="b"*64,
                speaker_id="human/one", pair_id="pair/one", source="source",
                license="research", source_revision="c"*40, kind="human",
                split="train", published_split="train", role="command",
                reference_status="supported", qa_eligible=True, metric_eligible=True,
                command={"intent": "STOP", "slots": {}}, **kw)


def test_audit_reports_missing_coverage_without_fabricating(tmp_path):
    views, receipts = inputs(tmp_path, [row()])
    result = audit_views(views, receipts, tmp_path / "audit")
    assert "MESSAGE" in result["missing_intents"]["train"]
    assert result["fit_ready"] is False
    assert result["splits"]["train"]["rows"] == 1


@pytest.mark.parametrize("field,value", [
    ("kind", "synthetic"), ("qa_eligible", False), ("reference_status", "auxiliary"),
    ("published_split", "numerals"), ("metric_eligible", False),
])
def test_fitting_contamination_rejected(tmp_path, field, value):
    r = row()
    r[field] = value
    views, receipts = inputs(tmp_path, [r])
    with pytest.raises(ValueError, match="eligible|human|numeral"):
        audit_views(views, receipts, tmp_path / "audit")


def test_missing_receipt_and_stale_view_rejected(tmp_path):
    views, receipts = inputs(tmp_path, [row()])
    with pytest.raises(ValueError, match="binding|receipt"):
        audit_views(views, {}, tmp_path / "audit")
    receipts["views"]["train"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="hash|binding"):
        audit_views(views, receipts, tmp_path / "audit")


@pytest.mark.parametrize("key", ["pcm_sha256", "speaker_id", "pair_id"])
def test_cross_partition_leakage_rejected(tmp_path, key):
    a = row()
    b = row()
    b.update(id="two", split="dev", sha256="d"*64, pcm_sha256="e"*64,
             speaker_id="human/two", pair_id="pair/two")
    b[key] = a[key]
    views, receipts = inputs(tmp_path, [a])
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps(b) + "\n")
    views["dev"] = dev
    receipts["views"]["dev"] = {"path": str(dev), "sha256": hashlib.sha256(dev.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="leak"):
        audit_views(views, receipts, tmp_path / "audit")
