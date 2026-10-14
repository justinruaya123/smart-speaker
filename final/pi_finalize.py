"""Apply actual physical Pi evidence to a derived card; keep the release baseline."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


BINDINGS = ("checkpoint_sha256", "recipe_sha256", "contract_hash")


def read_json(path):
    data = Path(path).read_bytes()

    def nonfinite(value):
        raise ValueError("Nonfinite JSON evidence: " + value)

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            nonfinite(value)
        return result

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON evidence field: " + key)
            result[key] = value
        return result

    return json.loads(data, parse_constant=nonfinite, parse_float=number,
                      object_pairs_hook=unique), data


def validate_receipt(card, receipt):
    if not isinstance(card, dict) or card.get("format") != "vcm-model-card-v2":
        raise ValueError("Expected vcm-model-card-v2 baseline")
    if not isinstance(receipt, dict) or receipt.get("kind") != "physical_pi_benchmark":
        raise ValueError("Requires an actual physical_pi_benchmark receipt")
    if any(not isinstance(value.get("bindings"), dict) for value in (card, receipt)):
        raise ValueError("Missing typed candidate bindings")
    for key in BINDINGS:
        expected = card.get("bindings", {}).get(key)
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or receipt.get("bindings", {}).get(key) != expected):
            raise ValueError("Pi receipt candidate binding mismatch: " + key)
    for key in ("hardware", "environment"):
        actual = receipt.get(key)
        if (not isinstance(actual, dict) or not any(
                isinstance(v, str) and v.strip()
                or type(v) in (int, float) and v > 0
                or isinstance(v, dict) and bool(v) for v in actual.values())):
            raise ValueError("Missing actual Pi " + key + " evidence")
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Missing Pi metrics and denominators")
    for key in ("trials", "supported_n", "negative_n", "correct", "false_accepts", "faults", "missing"):
        value = metrics.get(key)
        if type(value) is not int or value < 0:
            raise ValueError("Missing or invalid metric denominator/count: " + key)
    if any(metrics[key] <= 0 for key in ("trials", "supported_n", "negative_n")):
        raise ValueError("Pi positive and negative denominators must be positive")
    ambiguous = metrics.get("ambiguous_n", 0)
    if (type(ambiguous) is not int or ambiguous < 0
            or metrics["trials"] != metrics["supported_n"] + metrics["negative_n"] + ambiguous
            or metrics["correct"] > metrics["supported_n"]
            or metrics["false_accepts"] > metrics["negative_n"]
            or sum(metrics[k] for k in ("correct", "false_accepts", "faults", "missing")) > metrics["trials"]):
        raise ValueError("Inconsistent Pi metric denominators/counts")
    gates = receipt.get("gates")
    if not isinstance(gates, dict) or not gates or any(type(v) is not bool for v in gates.values()):
        raise ValueError("Pi receipt requires explicit boolean measured gates")
    return all(gates.values()) and metrics["faults"] == 0 and metrics["missing"] == 0


def render_markdown(card):
    """Keep every card field visible, including unchanged failures and verbatims."""
    return ("# VCM model card\n\n" + "Status: " + str(card.get("status", "unknown"))
            + "\n\nThis derived card records the supplied physical Pi receipt. The packaged semi-final baseline remains unchanged.\n\n"
            + "```json\n" + json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n")


def atomic_write(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize(baseline_path, receipt_path, output_dir=Path("results")):
    baseline_path, receipt_path = Path(baseline_path).resolve(), Path(receipt_path).resolve()
    card, baseline_bytes = read_json(baseline_path)
    receipt, receipt_bytes = read_json(receipt_path)
    passed = validate_receipt(card, receipt)
    output_dir = Path(output_dir).resolve()
    paths = [output_dir / "card.json", output_dir / "MODEL_CARD.md"]
    if any(p.resolve() in (baseline_path, receipt_path) for p in paths):
        raise ValueError("Output would overwrite the frozen baseline or actual receipt")
    result = deepcopy(card)
    result["pi"] = {
        "status": "passed" if passed else "failed",
        "baseline": {"path": str(baseline_path), "sha256": hashlib.sha256(baseline_bytes).hexdigest()},
        "receipt": {"path": str(receipt_path), "sha256": hashlib.sha256(receipt_bytes).hexdigest()},
        "results": deepcopy(receipt),
    }
    if isinstance(card.get("gates"), dict) and card["gates"].get("accuracy_ready") is True:
        result["status"] = "accuracy-ready / Pi-validated" if passed else "accuracy-ready / Pi-failed"
    json_bytes = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    markdown_bytes = render_markdown(result).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, data in zip(paths, (json_bytes, markdown_bytes)):
        atomic_write(path, data)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("final/card.json"))
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    result = finalize(args.baseline, args.receipt, args.output)
    print(json.dumps({"status": result.get("status"), "pi": result["pi"]["status"],
                      "card": str((args.output / "card.json").resolve())}, allow_nan=False))


if __name__ == "__main__":
    main()
