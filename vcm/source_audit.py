"""Audit pinned metadata and WAVs without inferring recording provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def metadata_audit(rows: list[dict], tree: dict) -> dict:
    """Use actual manifest/tree contents, not the README's larger totals."""
    if tree.get("truncated"):
        raise ValueError("A complete source tree is required")
    paths = [row["path"] for row in rows]
    wavs = {
        item["path"]: item for item in tree["tree"]
        if item["type"] == "blob" and item["path"].lower().endswith(".wav")
    }
    speakers = defaultdict(set)
    pairs = defaultdict(set)
    hashes = defaultdict(set)
    for row in rows:
        split = row["split"]
        speakers[row["speaker"]].add(split)
        pairs[(row["speaker"], row["label"], row["phrase_id"])].add(split)
        if row["path"] in wavs:
            hashes[wavs[row["path"]]["sha"]].add(split)
    by_split = {}
    for split in sorted({row["split"] for row in rows}):
        group = [row for row in rows if row["split"] == split]
        by_split[split] = {
            "rows": len(group),
            "speakers": len({row["speaker"] for row in group}),
            "conditions": dict(Counter(r["variant_id"] for r in group)),
            "phrases": len({row["transcript"] for row in group}),
        }
    extensions = Counter(
        Path(item["path"]).suffix.lower()
        for item in tree["tree"] if item["type"] == "blob"
    )
    return {
        "manifest_rows": len(rows),
        "tree_wavs": len(wavs),
        "tree_wav_bytes": sum(item.get("size", 0) for item in wavs.values()),
        "speakers": len(speakers),
        "phrases": len({row["transcript"] for row in rows}),
        "intent_counts": dict(sorted(Counter(
            row["intent"] for row in rows).items())),
        "slot_label_counts": dict(sorted(Counter(
            row["label"] for row in rows).items())),
        "conditions": dict(Counter(row["variant_id"] for row in rows)),
        "splits": by_split,
        "duplicate_manifest_paths": [
            path for path, count in Counter(paths).items() if count > 1
        ],
        "manifest_paths_absent_from_tree": sorted(set(paths) - set(wavs)),
        "tree_wavs_absent_from_manifest": sorted(set(wavs) - set(paths)),
        "cross_split_speakers": sorted(
            key for key, splits in speakers.items() if len(splits) > 1
        ),
        "cross_split_clean_noisy_groups": [
            list(key) for key, splits in pairs.items() if len(splits) > 1
        ],
        "cross_split_identical_blobs": sorted(
            key for key, splits in hashes.items() if len(splits) > 1
        ),
        "file_extensions": dict(extensions),
        "training_source_files": [
            item["path"] for item in tree["tree"]
            if Path(item["path"]).suffix in {".py", ".ipynb"}
        ],
        "limits": [
            "Source IDs do not establish command-recording provenance.",
            "Blob identity does not exclude near-duplicates or cloning.",
            "Shared texts across splits are not unseen-phrase validation.",
        ],
    }


def wav_audit(directory: Path, tree: dict) -> dict:
    """Decode and hash records; do not judge transcript correctness."""
    import numpy as np
    import soundfile as sf

    records = json.loads((directory / "manifest.json").read_text())["records"]
    wavs = {
        item["path"]: item for item in tree["tree"] if item["type"] == "blob"
    }
    samples, problems = [], []
    rates, channels, pcm_ids = Counter(), Counter(), defaultdict(list)
    for row in records:
        path = directory / row["path"]
        raw = path.read_bytes()
        header = b"blob " + str(len(raw)).encode() + b"\0"
        blob = hashlib.sha1(header + raw).hexdigest()
        digest = hashlib.sha256(raw).hexdigest()
        if (digest != row["file_sha256"]
                or blob != wavs[row["path"]]["sha"]):
            raise ValueError(f"Reference file integrity failed: {path}")
        pcm, rate = sf.read(path, dtype="float32", always_2d=True)
        if not np.isfinite(pcm).all() or not len(pcm):
            problems.append({"path": row["path"], "problem": "invalid_pcm"})
        duration = len(pcm) / rate
        if abs(duration - float(row["duration_sec"])) > .0011:
            problems.append({
                "path": row["path"], "problem": "duration_metadata"
            })
        rates[rate] += 1
        channels[pcm.shape[1]] += 1
        pcm_digest = hashlib.sha256(
            str(rate).encode() + pcm.astype("<f4").tobytes()
        ).hexdigest()
        pcm_ids[pcm_digest].append(row["path"])
        samples.append({
            "path": row["path"], "pcm_sha256": pcm_digest,
            "duration_seconds": duration,
            "peak": float(np.max(np.abs(pcm))),
            "clipped_fraction": float(np.mean(np.abs(pcm) >= .999)),
            "rms": float(np.sqrt(np.mean(pcm ** 2))),
        })
    return {
        "recordings_decoded": len(records), "sample_rates": dict(rates),
        "channels": dict(channels), "metadata_or_pcm_problems": problems,
        "identical_pcm_groups": [
            paths for paths in pcm_ids.values() if len(paths) > 1
        ],
        "duration_seconds": {
            "min": min(row["duration_seconds"] for row in samples),
            "max": max(row["duration_seconds"] for row in samples),
        },
        "recordings_with_clipped_samples": sum(
            row["clipped_fraction"] > 0 for row in samples
        ),
        "rows": samples,
        "limits": "PCM audit, not listening or transcript adjudication.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = json.loads((args.reference / "tree.json").read_text())
    with (args.reference / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    report = {
        "metadata": metadata_audit(rows, tree),
        "provenance": json.loads(
            (args.reference / "provenance.json").read_text()
        ),
    }
    if args.audio:
        report["waveforms"] = wav_audit(args.audio, tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report["metadata"], indent=2))


if __name__ == "__main__":
    main()
