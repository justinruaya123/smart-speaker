"""Software-fixture document tests; no measured model or benchmark results."""
from copy import deepcopy
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "render_card.py"


def module():
    assert SCRIPT.is_file(), "Missing JSON-only model-card renderer"
    spec = importlib.util.spec_from_file_location("render_card_under_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def software_card():
    return {
        "format": "vcm-model-card-v2", "title": "SOFTWARE FIXTURE / unmeasured card",
        "status": "BLOCKED / software fixture", "failed_gates": ["independent audio review"],
        "bindings": {"checkpoint_sha256": "a" * 64, "recipe_sha256": "b" * 64,
                     "contract_hash": "c" * 64, "selected_receipt_sha256": "d" * 64},
        "architecture": {"name": "software fixture", "parameters": None},
        "training": {"updates": None}, "data": {"utterances": None},
        "calibration": {"status": "not measured"},
        "benchmarks": {
            "human_regression": {"trials": 0, "complete_command_accuracy": None,
                                 "missed_verbatims": ["<fixture> & unmeasured command"]},
            "fresh_human": {"trials": None, "every_slot_accuracy": None},
            "tts": {"status": "not measured", "per_voice": {
                "af_heart": {"complete_command_accuracy": None, "faults": None},
                "am_puck": {"complete_command_accuracy": None, "faults": None}}},
        },
        "pi": {"status": "deferred", "results": None},
        "unrecognized_extension": {"preserve_verbatim": "fixture detail", "values": [None, 0, False]},
    }


def write_card(path, card):
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    return path


def test_unmeasured_values_stay_explicit_and_all_fields_survive(software_card):
    renderer = module()
    original = deepcopy(software_card)
    renderer.validate_card(software_card)
    markdown = renderer.render_markdown(software_card)
    html = renderer.render_html(software_card)
    for value in ("BLOCKED / software fixture", "independent audio review", "not measured",
                  "human_regression", "fresh_human", "af_heart", "am_puck", "fixture detail", "d" * 64):
        assert value in markdown and value in html
    assert "&lt;fixture&gt; &amp; unmeasured command" in html
    assert "contenteditable=\"true\"" in html
    assert json.dumps(software_card, ensure_ascii=False, indent=2, sort_keys=True) in markdown
    assert software_card == original


@pytest.mark.parametrize("field", ["checkpoint_sha256", "recipe_sha256", "contract_hash", "selected_receipt_sha256"])
def test_invalid_candidate_or_receipt_hashes_are_rejected(software_card, field):
    software_card["bindings"][field] = "not a SHA-256"
    with pytest.raises(ValueError, match="hash|binding|SHA"):
        module().validate_card(software_card)


def test_format_nonfinite_numbers_and_contradictory_pi_fields_are_rejected(software_card):
    renderer = module()
    with pytest.raises(ValueError, match="format|vcm-model-card"):
        renderer.validate_card({**software_card, "format": "invented"})
    wrong = deepcopy(software_card)
    wrong["benchmarks"]["fresh_human"]["trials"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        renderer.validate_card(wrong)
    wrong = deepcopy(software_card)
    wrong["pi"]["results"] = {"latency_ms": 0}
    with pytest.raises(ValueError, match="Pi|deferred"):
        renderer.validate_card(wrong)


@pytest.mark.parametrize("text", [
    '{"format":"vcm-model-card-v2","format":"duplicate"}',
    '{"value":NaN}', '{"value":1e400}',
])
def test_json_reader_rejects_duplicate_and_nonfinite_evidence(tmp_path, text):
    source = tmp_path / "bad.json"
    source.write_text(text)
    with pytest.raises(ValueError, match="uplicate|finite"):
        module().read_card(source)


def test_help_requires_no_document_or_model_packages():
    assert SCRIPT.is_file(), "Missing model-card renderer"
    result = subprocess.run([sys.executable, "-S", "-B", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--card" in result.stdout and "--output" in result.stdout


def test_rendering_is_deterministic_and_preserves_source_bytes(tmp_path, software_card):
    pytest.importorskip("reportlab")
    pptx = pytest.importorskip("pptx")
    renderer = module()
    source = write_card(tmp_path / "input.json", software_card)
    original = source.read_bytes()
    result = renderer.render(source, tmp_path / "cards")
    names = ("MODEL_CARD.md", "MODEL_CARD.html", "MODEL_CARD.pptx", "MODEL_CARD.pdf", "card.source.json")
    before = {name: (tmp_path / "cards" / name).read_bytes() for name in names}
    assert before["card.source.json"] == original and source.read_bytes() == original
    assert result["source_card_sha256"] == hashlib.sha256(original).hexdigest()
    assert all(result["artifacts"][name] == hashlib.sha256(data).hexdigest() for name, data in before.items())
    assert before["MODEL_CARD.pdf"].startswith(b"%PDF-")
    slides = pptx.Presentation(tmp_path / "cards/MODEL_CARD.pptx")
    assert len(slides.slides) == 3
    texts = " ".join(shape.text for slide in slides.slides for shape in slide.shapes if shape.has_text_frame)
    texts += " ".join(cell.text for slide in slides.slides for shape in slide.shapes if shape.has_table
                      for row in shape.table.rows for cell in row.cells)
    assert "BLOCKED / software fixture" in texts and "af_heart" in texts and "deferred" in texts
    with zipfile.ZipFile(tmp_path / "cards/MODEL_CARD.pptx") as archive:
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in archive.infolist())
    renderer.render(source, tmp_path / "cards")
    assert all((tmp_path / "cards" / name).read_bytes() == data for name, data in before.items())


def test_source_card_cannot_be_overwritten(tmp_path, software_card):
    source = write_card(tmp_path / "card.source.json", software_card)
    original = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite|source"):
        module().render(source, tmp_path)
    assert source.read_bytes() == original


def test_six_voice_card_and_many_failed_gates_stay_inside_editable_slides(software_card):
    pptx = pytest.importorskip("pptx")
    software_card["gates"] = {"fixture_gate_" + str(n): False for n in range(12)}
    software_card["pi"].update({"unmeasured_field_" + str(n): None for n in range(10)})
    software_card["benchmarks"]["tts"]["details"] = {"fixture_field_" + str(n): None for n in range(10)}
    software_card["benchmarks"]["tts"]["per_voice"] = {voice: {"complete_command_accuracy": None,
                "every_slot_accuracy": None, "negative_n": None, "false_accepts": None,
                "faults": None, "missing": None} for voice in (
                "af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck")}
    slides = pptx.Presentation(BytesIO(module().render_pptx(software_card, "e" * 64)))
    assert all(0 <= shape.top and shape.top + shape.height <= slides.slide_height
               for slide in slides.slides for shape in slide.shapes)
    text = " ".join(shape.text for slide in slides.slides for shape in slide.shapes if shape.has_text_frame)
    tables = " ".join(cell.text for slide in slides.slides for shape in slide.shapes if shape.has_table
                      for row in shape.table.rows for cell in row.cells)
    for voice in software_card["benchmarks"]["tts"]["per_voice"]:
        assert voice in text + tables


def test_native_tts_accounting_fields_are_rendered_without_recomputing_scores(software_card):
    software_card["benchmarks"]["tts"]["per_voice"] = {"fixture_voice": {
        "complete_command_recall": 0.25, "strict_accuracy": 0.125,
        "joint_supplied_slots": {"n": 4, "correct_accepted": 0, "accuracy": 0.0},
        "negative": 3, "false_accepts": 2, "faults": 1, "missing": 2}}
    tables = module()._pages(software_card)[2]["voice_tables"]
    assert tables[0][2][0] == ["fixture_voice", "0.25", "0.125", "0.0", "3", "2", "1 / 2"]


def test_regression_and_supplemental_voice_tables_do_not_overlap_footer(software_card):
    pptx = pytest.importorskip("pptx")
    software_card["benchmarks"]["synthetic_regression"] = deepcopy(software_card["benchmarks"]["tts"])
    slides = pptx.Presentation(BytesIO(module().render_pptx(software_card, "e" * 64)))
    assert all(shape.top + shape.height <= slides.slide_height
               for slide in slides.slides for shape in slide.shapes)
