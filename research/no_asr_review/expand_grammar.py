"""Compile complete training-derived forms for every 0..100 value.

Only the catalog changes. Old forms/IDs are retained, and a new graph identity
must not inherit the old catalog's acceptance thresholds.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys


def configure(payload: Path) -> Path:
    v3 = payload / "experiments/vcm_streaming_v3"
    sys.path.insert(0, str(v3 / "src"))
    sys.path.insert(0, str(v3 / "runtime"))
    return v3


def expanded_graph(original: dict, training_rows: list[dict]) -> dict:
    from phonetic_catalog import pronounce
    from semantics import number_words

    observed = {
        row["transcript"].lower() for row in training_rows
        if row["split"] == "train" and row["intent"] == "BRIGHTNESS"
    }
    templates = [
        "brightness {n} percent", "adjust brightness to {n} percent",
        "brightness level {n} percent",
    ]
    for template in templates:
        if not any(template.format(n=value) in observed
                   for value in ("20", "60", "100")):
            raise ValueError("Each new template needs a training example")
    graph = copy.deepcopy(original)
    seen = {
        (tuple(form["target"]), json.dumps(form["command"], sort_keys=True))
        for form in graph["forms"]
    }
    for value in range(101):
        command = {
            "intent": "SET_BRIGHTNESS", "value": value,
            "unit": "percent", "entity_id": None,
        }
        for template in templates:
            text = template.format(n=number_words(value))
            target = list(pronounce(text))
            key = (tuple(target), json.dumps(command, sort_keys=True))
            if key not in seen:
                graph["forms"].append({
                    "target": target, "text": text, "command": command
                })
                seen.add(key)
    graph["grammar_version"] = "complete-command-v6-brightness-templates-1"
    graph["extension_provenance"] = {
        "templates": templates, "source_partition": "train",
        "scope": "all integer brightness values 0 through 100",
        "automatic_acceptance": False,
    }
    graph.pop("graph_hash", None)
    graph["graph_hash"] = hashlib.sha256(
        json.dumps(graph, sort_keys=True).encode()
    ).hexdigest()
    return graph
