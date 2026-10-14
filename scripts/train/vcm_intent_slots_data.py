"""Read-only audited human views and immutable feature preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.runtime.pcv_schema import INTENTS, baseline_cases, canonical, validate_command
from scripts.test.speaker_demo_suite import read_audio
from vcm.frontend import features


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def eligible(row):
    if (row.get("kind") != "human" or row.get("role") != "command"
            or row.get("published_split") == "numerals"
            or row.get("reference_status") not in ("supported", "unsupported")
            or row.get("qa_eligible") is not True
            or row.get("metric_eligible") is not True):
        raise ValueError("Ineligible human fitting row or numeral contamination: " + row["id"])
    for key in ("source", "source_revision", "license", "speaker_id",
                "sha256", "pcm_sha256"):
        if not row.get(key):
            raise ValueError("Missing provenance: " + key)
    if row.get("command") is not None:
        validate_command(row["command"])
    elif row["reference_status"] != "unsupported":
        raise ValueError("Missing positive label")


def audit_views(view_paths, source_receipts, output_dir, *, audio_root=None):
    """Fail on contamination; retain evaluation ambiguity and all denominators."""
    if not source_receipts.get("views"):
        raise ValueError("Missing source receipt/view binding")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    partitions = {}
    seen_ids = set()
    groups = {key: {} for key in ("sha256", "pcm_sha256", "speaker_id", "pair_id")}
    baseline = {canonical(r["command"]): r["command"] for r in baseline_cases()}
    report = {"splits": {}, "missing_intents": {}, "missing_optionb_commands": {},
              "bindings": {}, "fit_ready": False}
    for split, path in view_paths.items():
        path = Path(path)
        binding = source_receipts["views"].get(split)
        if not binding or binding["sha256"] != sha(path):
            raise ValueError("Stale or missing view hash binding: " + split)
        subset = rows(path)
        partitions[split] = subset
        report["bindings"][split] = {"path": str(path.resolve()), "sha256": sha(path)}
        commands, intents, people, roles = Counter(), Counter(), Counter(), Counter()
        ambiguous = 0
        for row in subset:
            if row["split"] != split or row["id"] in seen_ids:
                raise ValueError("Partition or duplicate row identity")
            seen_ids.add(row["id"])
            if split in ("train", "dev", "calibration"):
                eligible(row)
            ambiguous += row.get("reference_status") == "ambiguous"
            for key, index in groups.items():
                value = row.get(key)
                if value and value in index and index[value] != split:
                    raise ValueError("Cross-partition leakage: " + key)
                if value:
                    index[value] = split
            if audio_root is not None:
                read_audio(row, Path(audio_root))
            if row.get("reference_status") == "ambiguous":
                continue
            cmd = row.get("command")
            commands[canonical(cmd)] += 1
            intents[cmd["intent"] if cmd else "OOV"] += 1
            people[row.get("speaker_id", "unknown")] += 1
            if cmd:
                for field, value in cmd["slots"].items():
                    roles[field + "=" + canonical(value)] += 1
        report["splits"][split] = {
            "rows": len(subset), "ambiguous": ambiguous, "intents": dict(intents),
            "commands": dict(commands), "people": dict(people), "slot_values": dict(roles),
            "human": sum(r["kind"] == "human" for r in subset),
            "synthetic": sum(r["kind"] == "synthetic" for r in subset),
        }
        report["missing_intents"][split] = [i for i in ["OOV", *INTENTS] if not intents[i]]
        report["missing_optionb_commands"][split] = [
            cmd for key, cmd in baseline.items() if not commands[key]]
    report["fit_ready"] = all(
        s in report["splits"] and not report["missing_intents"][s]
        for s in ("train", "dev"))
    report["qualification_ready"] = report["fit_ready"] and (
        "calibration" in report["splits"] and not report["missing_intents"]["calibration"])
    report["missing_supervision"] = {
        "reviewed_role_spans": sum(not r.get("reviewed_spans")
                                  for rs in partitions.values() for r in rs),
        "role_states": sum(not r.get("role_labels")
                           for rs in partitions.values() for r in rs),
        "ownership_pairs": "not supplied by command-only views",
    }
    report["identity_limits"] = "Declared IDs only; voice aliases, near duplicates and independent listening unverified"
    write_new(output_dir / "audit.json", report)
    return report


def prepare_features(subset, audio_root, output_dir):
    """Only human train/dev; streaming statistics consume train frames only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    total, squares, frames = np.zeros(40), np.zeros(40), 0
    receipt = {}
    for row in subset:
        if row["split"] not in ("train", "dev"):
            raise ValueError("Calibration/test access forbidden during fitting")
        eligible(row)
        pcm = read_audio(row, Path(audio_root))
        x = features(pcm)
        if x.shape[1] == 0:
            raise ValueError("No valid acoustic frames: " + row["id"])
        name = hashlib.sha256(row["id"].encode()).hexdigest() + ".npy"
        path = output_dir / name
        np.save(path, x)
        receipt[row["id"]] = {"path": str(path), "sha256": sha(path), "frames": x.shape[1],
                              "audio_sha256": row["sha256"], "pcm_sha256": row["pcm_sha256"]}
        if row["split"] == "train":
            total += x.sum(axis=1, dtype=np.float64)
            squares += np.square(x.astype(np.float64)).sum(axis=1)
            frames += x.shape[1]
    if not frames:
        raise ValueError("No eligible human training frames")
    mean = total / frames
    std = np.sqrt(np.maximum(squares / frames - mean**2, 1e-6))
    normalization = {"mean": mean.tolist(), "std": std.tolist(), "fit_split": "train", "frames": frames}
    write_new(output_dir / "features.json", receipt)
    write_new(output_dir / "normalization.json", normalization)
    return receipt, normalization
