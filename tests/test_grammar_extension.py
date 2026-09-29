"""Check coverage, immutability and the no-extra-network contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/no_asr_review"))
from expand_grammar import configure, expanded_graph

PAYLOAD = Path(os.environ.get("VCM_RESEARCH_PAYLOAD", ROOT))
AVAILABLE = (
    PAYLOAD / "experiments/vcm_streaming_v3/src/phonetic_catalog.py"
).is_file()
pytestmark = pytest.mark.skipif(
    not AVAILABLE, reason="Restore the VCM research payload"
)


@pytest.fixture
def inputs():
    configure(PAYLOAD)
    path = PAYLOAD / "experiments/vcm_services_v6/data/services_compiled.json"
    graph = json.loads(path.read_text())
    training = [
        {"intent": "BRIGHTNESS", "split": "train", "transcript": text}
        for text in (
            "Brightness 20 percent", "Adjust brightness to 20 percent",
            "Brightness level 20 percent",
        )
    ]
    return graph, training


def test_each_template_supports_all_percentages(inputs):
    from semantics import number_words

    graph = expanded_graph(*inputs)
    by_text = {form["text"]: form["command"] for form in graph["forms"]}
    templates = (
        "brightness {n} percent", "adjust brightness to {n} percent",
        "brightness level {n} percent",
    )
    for value in range(101):
        for template in templates:
            assert by_text[template.format(n=number_words(value))] == {
                "intent": "SET_BRIGHTNESS", "entity_id": None,
                "value": value, "unit": "percent",
            }


def test_original_forms_and_ids_are_not_mutated(inputs):
    original, rows = inputs
    frozen = json.dumps(original, sort_keys=True)
    result = expanded_graph(original, rows)
    assert json.dumps(original, sort_keys=True) == frozen
    assert result["forms"][:len(original["forms"])] == original["forms"]
    assert result["graph_hash"] != original["graph_hash"]


def test_graph_hash_binds_new_rules(inputs):
    result = expanded_graph(*inputs)
    digest = result.pop("graph_hash")
    encoded = json.dumps(result, sort_keys=True).encode()
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert result["extension_provenance"]["automatic_acceptance"] is False


def test_templates_cannot_be_introduced_using_test_only(inputs):
    original, rows = inputs
    for item in rows:
        item["split"] = "test"
    with pytest.raises(ValueError, match="training"):
        expanded_graph(original, rows)


def test_duplicate_forms_not_created(inputs):
    result = expanded_graph(*inputs)
    keys = [
        (tuple(form["target"]), json.dumps(form["command"], sort_keys=True))
        for form in result["forms"]
    ]
    assert len(keys) == len(set(keys))


def test_no_value_outside_zero_to_one_hundred(inputs):
    graph = expanded_graph(*inputs)
    added = graph["forms"][len(inputs[0]["forms"]):]
    assert all(0 <= form["command"]["value"] <= 100 for form in added)
