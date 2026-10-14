"""Preserve cached official defaults and audit held evaluation/fit separation.

This performs data inspection only. It neither downloads nor runs a recognizer.
Original WAV bytes and publisher references remain unchanged. Canonical typed
references use the same frozen metadata annotation as the original human views.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.test.speaker_demo_suite import (
    canonical, file_digest, pcm_digest, read_audio, read_json, read_jsonl, write_new,
)
from scripts.train import pcv_release_prepare as reviewed
from scripts.train.pcv_updated_prepare import pair_identity

BENCHMARK_REVISION = "ab39857cf73dd067d068fed388c321ab160dc191"
BENCHMARK_DOWNLOAD_SHA = "71c01b0ef4e4979ba9d77d4e4377ece271950d23bb31db70bbfdb834791c3c36"
LOADER_SHA = "17d74702d4b9230965c96fbf7dd9a5b717fea47cc3374adfc48e9c2225dd6ba1"
PARSER_SHA = "772a73dffa5810645182d2a6a9c16a828bb25d4a84f2d9cb3bbf60993d76545f"
DATASET_REVISION = "bfca0bbebfe17e00d6a685d1da573615d29ecdce"
PARQUET_SHA = "c36227c81eee4eac63de62cee94631ee9e73952ccf8e16c27912eb049508eb6e"
FIT_SPLITS = ("train", "dev", "calibration", "enrollment")
QUALIFICATION_LIMITS = [
    "Declared source identities and explicit aliases only; cross-source physical speaker aliases are not acoustically adjudicated.",
    "No independent listening or near-duplicate acoustic comparison performed.",
    "Canonical supervision is frozen metadata annotation, not independently reviewed acoustic ground truth.",
    "Current defaults and original holdout are regression evaluations; evaluation/evaluation overlap is retained and is never fresh.",
]


def load_reviewed_policy(path=None):
    """Load the existing review without editing its LF/CRLF representation."""
    path = Path(path or reviewed.PROPOSAL)
    disk = path.read_bytes()
    verified = disk
    representation = "original hash verified"
    if hashlib.sha256(verified).hexdigest() != reviewed.PROPOSAL_SHA256:
        verified = disk.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        representation = "verified CRLF recovery in memory"
        if hashlib.sha256(verified).hexdigest() != reviewed.PROPOSAL_SHA256:
            raise ValueError("Frozen annotation review hash mismatch")
    proposal = json.loads(verified)
    if proposal["provenance"]["source_sha256"] != reviewed.SOURCE_SHA256:
        raise ValueError("Frozen annotation review belongs to another source")
    supported, decisions = {}, []
    # Identical policy construction to pcv_release_prepare.annotation_policy;
    # only its verified byte loading accepts the original CRLF representation.
    for evidence in proposal["supported_correction_evidence"]:
        text, expected = evidence["text"], evidence["expected"]
        reviewed.validate_command(expected)
        approved = text not in reviewed.REJECTED_PROPOSALS
        if approved:
            supported[text] = expected
        decisions.append({
            "text": text, "approved": approved, "expected": expected,
            "reason": reviewed.REJECTED_PROPOSALS.get(text, evidence["rationale"]),
            "review_source_ids": evidence["source_ids"],
        })
    policy = {
        "version": "pcv-release-train-v1-annotation-1", "supported": supported,
        "ambiguous": proposal["text_to_ambiguous_corrections"],
        "decisions": decisions, "proposal_sha256": reviewed.PROPOSAL_SHA256,
        "source_sha256": reviewed.SOURCE_SHA256,
        "rules": "Exact normalized-text corrections across every split; publisher out_of_scope and unintelligible flags take precedence; no generic rules added.",
        "evidence": "Supplied transcript metadata review, not independent listening; no teacher or ASR generated ground truth.",
    }
    return policy, {
        "path": str(path.resolve()), "disk_sha256": hashlib.sha256(disk).hexdigest(),
        "verified_original_sha256": hashlib.sha256(verified).hexdigest(),
        "representation": representation,
        "policy_sha256": hashlib.sha256(canonical(policy).encode()).hexdigest(),
    }


def _raw_metadata(raw):
    result = {key: value for key, value in raw.items() if key != "audio"}
    # Enforce finite JSON without changing publisher strings or reference values.
    canonical(result)
    return result


def derive_rows(raw_rows, clips, output, policy):
    """Copy every native WAV once while retaining every original default row."""
    if len(raw_rows) != len(clips):
        raise ValueError("Raw/default loader row count mismatch")
    output = Path(output).resolve()
    audio_dir = output / "audio"
    result = []
    for idx, (raw, clip) in enumerate(zip(raw_rows, clips)):
        blob = raw["audio"]["bytes"]
        info = sf.info(io.BytesIO(blob))
        if (info.samplerate, info.channels, info.format, info.subtype) != (
            16000, 1, "WAV", "PCM_16"
        ):
            raise ValueError(f"Expected native mono 16 kHz PCM_16 WAV at row {idx}")
        pcm, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
        pcm_sha = pcm_digest(pcm, rate)
        metadata = _raw_metadata(raw)
        normalized = {key: "" if value is None else value
                      for key, value in metadata.items()}
        out_of_scope = (int(normalized.get("out_of_scope") or 0) == 1
                        or normalized.get("command") == "OUT_OF_SCOPE")
        expected_official = {
            "idx": idx, "file": str(normalized.get("file", "")),
            "transcript": str(normalized.get("transcript", "")),
            "intent": "OUT_OF_SCOPE" if out_of_scope else str(normalized["command"]),
            "variation": "" if out_of_scope else str(normalized.get("variation") or ""),
            "slot_value": "" if out_of_scope else str(normalized.get("slot_value") or ""),
            "speaker_id": str(normalized.get("speaker_id", "")),
            "is_synthetic": int(normalized.get("is_synthetic") or 0) == 1,
            "accent_group": str(normalized.get("accent_group", "")),
        }
        if any(getattr(clip, key) != value for key, value in expected_official.items()):
            raise ValueError(f"Unchanged default loader metadata mismatch at row {idx}")
        if not np.array_equal(np.asarray(clip.audio, dtype=np.float32), pcm):
            raise ValueError(f"Unchanged default loader PCM mismatch at row {idx}")
        normalized["published_split"] = "holdout"
        reference = reviewed.annotate(normalized, policy)
        file_sha = hashlib.sha256(blob).hexdigest()
        audio_path = audio_dir / f"{file_sha}.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        if audio_path.exists():
            if audio_path.read_bytes() != blob:
                raise ValueError("Existing derived audio bytes changed")
        else:
            with audio_path.open("xb") as stream:
                stream.write(blob)
        group = pair_identity(normalized)
        result.append({
            "id": f"airimonda/ai231-me2-voice-commands/primary/default-holdout-{DATASET_REVISION}/{idx:06d}",
            "audio": str(audio_path), "sha256": file_sha, "pcm_sha256": pcm_sha,
            "sample_rate": rate, "samples": len(pcm), "channels": 1,
            "subtype": info.subtype, "decoded_duration_seconds": len(pcm) / rate,
            "kind": "synthetic" if clip.is_synthetic else "human",
            "source": "airimonda/ai231-me2-voice-commands",
            "source_revision": DATASET_REVISION, "config": "primary",
            "published_split": "holdout", "split": "holdout", "parquet_row": idx,
            "speaker_id": str(normalized["source"]) + "/" + str(clip.speaker_id),
            "raw_speaker_id": clip.speaker_id, "speaker_aliases_resolved": False,
            "pair_id": group, "recording_group": group,
            "publisher_metadata": metadata,
            "publisher_intent": metadata["command"],
            "publisher_slot": metadata.get("slot_value", ""),
            "publisher_variation": metadata.get("variation", ""),
            "publisher_transcript": metadata.get("transcript", ""),
            "official_intent": clip.intent, "official_slot": clip.slot_value,
            "official_variation": clip.variation, "official_transcript": clip.transcript,
            "transcript": clip.transcript, "command": reference["expected"],
            "reference_status": reference["reference_status"],
            "reference_reason": reference["reason"],
            "slot_metadata_conflict": reference["slot_metadata_conflict"],
            "role": "command", "metric_eligible": True, "qa_eligible": False,
            "regression": True, "fresh": False,
            "evaluation_track": "official_current_default_regression",
            "acoustic_listening_verified": False,
            "license": "Source-dependent research/education terms; pinned primary source card retained",
        })
    return result


def _identities(row):
    result = {str(row.get("speaker_id", ""))}
    raw = row.get("raw_speaker_id") or row.get("publisher_metadata", {}).get("speaker_id")
    if raw:
        result.add(str(raw))
    for key in ("speaker_aliases", "resolved_speaker_aliases", "person_aliases"):
        values = row.get(key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError(f"Malformed explicit speaker aliases: {row.get('id')}")
        result.update(values)
    return result - {""}


def overlap(*, a_rows, b_rows):
    result = {}
    for key in ("sha256", "pcm_sha256", "speaker_id", "pair_id", "recording_group"):
        right = {str(row[key]) for row in b_rows if row.get(key)}
        ids = [row["id"] for row in a_rows if row.get(key) and str(row[key]) in right]
        result[key] = {"rows": len(ids), "ids": ids}
    right = set().union(*(_identities(row) for row in b_rows)) if b_rows else set()
    ids = [row["id"] for row in a_rows if _identities(row) & right]
    result["declared_identity_or_alias"] = {"rows": len(ids), "ids": ids}
    return result


def separation(default_rows, views):
    checks = {name: overlap(a_rows=default_rows, b_rows=views[name])
              for name in FIT_SPLITS}
    checks["fresh_human"] = overlap(a_rows=default_rows, b_rows=views["fresh_human"])
    fresh = {name: overlap(a_rows=views["fresh_human"], b_rows=views[name])
             for name in FIT_SPLITS}
    failures = [{"comparison": name, "dimension": key, **entry}
                for name, values in checks.items() for key, entry in values.items()
                if entry["rows"]]
    failures += [{"comparison": "fresh_vs_" + name, "dimension": key, **entry}
                 for name, values in fresh.items() for key, entry in values.items()
                 if entry["rows"]]
    return {
        "default_vs_fit": checks, "fresh_vs_fit": fresh,
        "current_default_vs_original196_regression": overlap(
            a_rows=default_rows, b_rows=views["holdout"]),
        "current_default_vs_original_test_regression": overlap(
            a_rows=default_rows, b_rows=views["test"]),
        "fit_separation_ready": not failures, "failures": failures,
        "regression_overlap_allowed": True,
    }


def _verify_snapshot(snapshot):
    receipt_path = snapshot / "snapshot-receipt.json"
    receipt = read_json(receipt_path)
    if (receipt.get("repository") != "airimonda/vcm-benchmark"
            or receipt.get("revision") != BENCHMARK_REVISION
            or receipt.get("download_sha256") != BENCHMARK_DOWNLOAD_SHA):
        raise ValueError("Pinned benchmark source receipt mismatch")
    files = {}
    for item in receipt["files"]:
        name = Path(item["path"])
        if name.is_absolute() or ".." in name.parts or str(name) in files:
            raise ValueError("Unsafe or duplicate pinned source path")
        path = snapshot / name
        if file_digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Pinned benchmark source bytes changed: {name}")
        files[str(name)] = item["sha256"]
    if files.get("vcmbench/dataset.py") != LOADER_SHA or files.get("vcmbench/pi.py") != PARSER_SHA:
        raise ValueError("Pinned default loader/parser changed")
    return {"path": str(snapshot), "revision": BENCHMARK_REVISION,
            "receipt_sha256": file_digest(receipt_path), "files": files}


def _load_default_loader(snapshot):
    name = "vcm_default202_pinned_" + BENCHMARK_REVISION
    package = ModuleType(name)
    package.__path__ = [str(snapshot / "vcmbench")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".dataset", snapshot / "vcmbench/dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_views(inputs_path):
    inputs = read_json(inputs_path)
    views, bindings = {}, {}
    for name, binding in inputs["views"].items():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = ROOT / path
        if file_digest(path) != binding["sha256"]:
            raise ValueError(f"Prepared view changed: {name}")
        views[name] = read_jsonl(path)
        bindings[name] = {"path": str(path), "sha256": binding["sha256"],
                          "rows": len(views[name])}
    return inputs, views, bindings


def _verify_fit_and_fresh_audio(views, audio_root):
    all_rows = [row for split in (*FIT_SPLITS, "fresh_human") for row in views[split]]
    unique = {(row["audio"], row["sha256"], row["pcm_sha256"]): row for row in all_rows}

    def verify(row):
        pcm = read_audio(row, audio_root)
        if len(pcm) != row["samples"] or row.get("sample_rate", 16000) != 16000:
            raise ValueError(f"Prepared sample metadata changed: {row['id']}")
        return row["id"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, unique.values()))
    return {"fit_rows": sum(len(views[name]) for name in FIT_SPLITS),
            "fresh_human_rows": len(views["fresh_human"]),
            "unique_actual_files_and_decoded_pcm_verified": len(verified)}


def derive_default_benchmark(snapshot, cache, inputs_path, output):
    """Freeze 202 untouched current defaults plus separate original 196 receipt."""
    import pandas as pd

    snapshot, cache, inputs_path, output = map(
        lambda p: Path(p).resolve(), (snapshot, cache, inputs_path, output))
    if output.exists():
        raise FileExistsError(f"Preserve prior evidence; choose a new output: {output}")
    source_binding = _verify_snapshot(snapshot)
    parquet = cache / "holdout.parquet"
    # Verify before invoking the unchanged None/default path: no missing-cache
    # download is allowed, and the source argument remains exactly None.
    if not parquet.is_file() or file_digest(parquet) != PARQUET_SHA:
        raise ValueError("Required cached default parquet is missing or changed")
    revision_receipt = read_json(cache / "source-revision-receipt.json")
    if (revision_receipt.get("repository") != "airimonda/ai231-me2-voice-commands"
            or revision_receipt.get("revision") != DATASET_REVISION
            or revision_receipt.get("file_sha256") != PARQUET_SHA
            or revision_receipt.get("upstream_lfs_sha256") != PARQUET_SHA
            or revision_receipt.get("default_URL_retained") is not True):
        raise ValueError("Cached default dataset revision receipt changed")
    inputs, views, view_bindings = _load_views(inputs_path)
    policy, policy_binding = load_reviewed_policy()
    loader = _load_default_loader(snapshot)
    clips = loader.load_holdout(None, cache)
    raw_rows = pd.read_parquet(parquet).to_dict("records")
    if len(clips) != 202 or len(raw_rows) != 202 or len(views["holdout"]) != 196:
        raise ValueError("Required current202/original196 count changed")
    rows = derive_rows(raw_rows, clips, output, policy)
    if len(rows) != 202 or len({row["id"] for row in rows}) != 202:
        raise ValueError("Incomplete current-default derivation")
    report = separation(rows, views)
    actual_audio = _verify_fit_and_fresh_audio(views, Path(inputs["audio_root"]))
    old_by_pcm = {}
    for old in views["holdout"]:
        old_by_pcm.setdefault(old["pcm_sha256"], []).append(old)
    mismatches, matches = [], 0
    for row in rows:
        old = old_by_pcm.get(row["pcm_sha256"], [])
        if not old:
            continue
        if any(item["command"] == row["command"]
               and item["reference_status"] == row["reference_status"] for item in old):
            matches += 1
        else:
            mismatches.append({"current_id": row["id"], "current_command": row["command"],
                               "original_ids": [item["id"] for item in old],
                               "original_commands": [item["command"] for item in old]})
    references = [{key: row[key] for key in (
        "id", "parquet_row", "publisher_intent", "publisher_slot",
        "publisher_variation", "publisher_transcript", "official_intent",
        "official_slot", "official_variation", "official_transcript", "command",
        "reference_status", "reference_reason", "pcm_sha256")}
        for row in rows]
    view_path, reference_path = output / "default202.jsonl", output / "reference-lock.jsonl"
    write_new(view_path, rows, jsonl=True)
    write_new(reference_path, references, jsonl=True)
    original_path = output / "original196-regression.jsonl"
    original_bytes = Path(view_bindings["holdout"]["path"]).read_bytes()
    with original_path.open("xb") as stream:
        stream.write(original_bytes)
    if file_digest(original_path) != view_bindings["holdout"]["sha256"]:
        raise ValueError("Original196 frozen view changed during preservation")
    source_lock = {
        "format": "vcm-default-benchmark-data-lock-v1", "benchmark": source_binding,
        "dataset": {"repository": revision_receipt["repository"], "revision": DATASET_REVISION,
                    "parquet": str(parquet), "parquet_sha256": PARQUET_SHA,
                    "source_receipt_sha256": file_digest(cache / "source-revision-receipt.json"),
                    "default_source_argument": None, "default_url": loader.HF_URL},
        "annotation_policy": policy_binding,
        "annotation_code": {str(path.relative_to(ROOT)): file_digest(path) for path in (
            ROOT / "scripts/train/pcv_updated_prepare.py",
            ROOT / "scripts/train/pcv_release_prepare.py", Path(__file__))},
        "inputs": {"path": str(inputs_path), "sha256": file_digest(inputs_path)},
        "views": view_bindings,
        "current202": {"path": str(view_path), "sha256": file_digest(view_path),
                       "audio_root": str(ROOT), "rows": 202, "fresh": False,
                       "references": str(reference_path), "references_sha256": file_digest(reference_path)},
        "original196": {"path": str(original_path), "sha256": file_digest(original_path),
                        "audio_root": inputs["audio_root"], "rows": 196, "fresh": False},
        "native_audio_bytes_copied_unchanged": True, "references_changed": False,
        "neural_inference_performed": False, "physical_pi_run": False,
    }
    write_new(output / "source-lock.json", source_lock)
    report.update({
        "format": "vcm-default-benchmark-data-audit-v1", "current_default_rows": len(rows),
        "source_lock_sha256": file_digest(output / "source-lock.json"),
        "human_rows": sum(row["kind"] == "human" for row in rows),
        "synthetic_rows": sum(row["kind"] == "synthetic" for row in rows),
        "publisher_intents": dict(sorted(Counter(row["official_intent"] for row in rows).items())),
        "canonical_reference_status": dict(sorted(Counter(row["reference_status"] for row in rows).items())),
        "declared_human_people": sorted({row["speaker_id"] for row in rows if row["kind"] == "human"}),
        "declared_synthetic_sources": sorted({row["speaker_id"] for row in rows if row["kind"] == "synthetic"}),
        "duplicate_pcm_groups_retained": [
            {"pcm_sha256": digest, "ids": [row["id"] for row in rows if row["pcm_sha256"] == digest]}
            for digest, count in sorted(Counter(row["pcm_sha256"] for row in rows).items()) if count > 1],
        "actual_audio": actual_audio,
        "original_overlap_typed_reference_matches": matches,
        "original_overlap_typed_reference_mismatches": mismatches,
        "reference_preservation_ready": not mismatches,
        "qualification_limits": QUALIFICATION_LIMITS,
        "neural_inference_performed": False, "physical_pi_run": False,
        "source_audio_or_reference_modified": False,
    })
    write_new(output / "audit.json", report)
    return {"view": str(view_path), "view_sha256": file_digest(view_path),
            "source_lock": str(output / "source-lock.json"),
            "source_lock_sha256": file_digest(output / "source-lock.json"),
            "audit": str(output / "audit.json"), "audit_sha256": file_digest(output / "audit.json"),
            "fit_separation_ready": report["fit_separation_ready"],
            "reference_preservation_ready": report["reference_preservation_ready"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = derive_default_benchmark(args.snapshot, args.cache, args.inputs, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["fit_separation_ready"] and result["reference_preservation_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
