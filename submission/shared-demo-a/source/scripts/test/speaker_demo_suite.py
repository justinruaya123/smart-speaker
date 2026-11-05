"""Action-free, immutable audit/plan/lock/replay/score for speaker-demo-v1.

Paired recordings test ownership on annotated crops. They DO NOT test the wake
word detector, endpoint capture, natural overlap, microphone or physical Pi.
Use original-audio and live-session tracks separately for those claims.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import random
import re
import time
from typing import Mapping, Sequence

import numpy as np

SPLITS = {"train", "dev", "calibration", "test", "holdout"}
BASELINE = "7d8296c96e19e1e65fc3f0cd1b82d2391df33dbc"


def canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def file_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _invalid_json(value: str) -> None:
    raise ValueError(f"Nonfinite JSON: {value}")


def read_json(path: Path) -> dict:
    return json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=_invalid_json
    )


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line, parse_constant=_invalid_json)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            result.append(value)
    return result


def write_new(path: Path, value: object, *, jsonl: bool = False) -> None:
    """Exclusive creation preserves earlier evidence, including failed runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        if jsonl:
            for row in value:
                stream.write(canonical(row) + "\n")
        else:
            stream.write(
                json.dumps(
                    value, indent=2, sort_keys=True, allow_nan=False
                ) + "\n"
            )


def pcm_digest(pcm: np.ndarray, sample_rate: int = 16000) -> str:
    values = np.asarray(pcm, dtype="<f4")
    if sample_rate != 16000 or values.ndim != 1 or not values.size:
        raise ValueError("Expected nonempty mono 16 kHz PCM")
    if not np.isfinite(values).all() or float(np.max(np.abs(values))) > 1:
        raise ValueError("Invalid PCM values")
    header = f"pcm:f32le:{sample_rate}:{len(values)}\0".encode()
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def read_audio(row: Mapping, root: Path = Path(".")) -> np.ndarray:
    import soundfile as sf
    path = Path(row["audio"])
    if not path.is_absolute():
        path = root / path
    if file_digest(path) != row["sha256"]:
        raise ValueError(f"Audio file hash mismatch: {row['id']}")
    pcm, rate = sf.read(path, dtype="float32", always_2d=False)
    if pcm_digest(pcm, rate) != row["pcm_sha256"]:
        raise ValueError(f"Decoded PCM hash mismatch: {row['id']}")
    return pcm


def _command_valid(value: object) -> bool:
    return value is None or (
        isinstance(value, dict) and set(value) == {"intent", "slots"}
        and isinstance(value["intent"], str) and bool(value["intent"])
        and isinstance(value["slots"], dict)
    )


def audit(
    rows: Sequence[Mapping], *, check_audio: bool = False,
    root: Path = Path("."),
) -> dict:
    """Audit declared global speaker identities, bytes and decoded PCM.

    Does not infer identity from voice or resolve cross-dataset aliasing. The
    manifest owner must resolve aliases and verify speech labels independently.
    """
    if not rows:
        raise ValueError("Empty manifest")
    ids, file_hashes, pcm_hashes = set(), set(), set()
    speaker_splits = {}
    counts = Counter()
    for row in rows:
        for key in ("id", "audio", "speaker_id", "source", "license"):
            value = row.get(key)
            if (
                not isinstance(value, str) or not value.strip()
                or value.lower() in {"unknown", "none", "unverified"}
            ):
                raise ValueError(f"Missing or unresolved {key}")
        if row["id"] in ids:
            raise ValueError("Duplicate recording id")
        ids.add(row["id"])
        for key in ("sha256", "pcm_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(
                r"[a-f0-9]{64}", row[key]
            ):
                raise ValueError(f"Invalid {key}")
        if row["sha256"] in file_hashes:
            raise ValueError("Duplicate audio file hash")
        if row["pcm_sha256"] in pcm_hashes:
            raise ValueError(
                "Duplicate PCM identity (including cross-split leakage)"
            )
        file_hashes.add(row["sha256"])
        pcm_hashes.add(row["pcm_sha256"])
        split = row.get("split")
        if split not in SPLITS or row.get("kind") not in {
            "human", "synthetic"
        }:
            raise ValueError("Invalid split or human/synthetic provenance")
        if row.get("role") not in {"wake", "command"}:
            raise ValueError("role must be wake or command")
        if "command" not in row or not _command_valid(row["command"]):
            raise ValueError("Expected typed command or explicit null")
        if row["role"] == "wake" and row["command"] is not None:
            raise ValueError("Wake reference must not carry a command label")
        canonical(row["command"])
        speaker = row["speaker_id"]
        if speaker in speaker_splits and speaker_splits[speaker] != split:
            raise ValueError(f"Cross-split speaker leakage: {speaker}")
        speaker_splits[speaker] = split
        counts[f"{split}/{row['kind']}/{row['role']}"] += 1
        if check_audio:
            read_audio(row, root)
    return {
        "recordings": len(rows), "speakers": len(speaker_splits),
        "strata": dict(sorted(counts.items())), "audio_verified": check_audio,
        "identity_scope": (
            'declared globally resolved IDs; voice identity not audited'
        ),
        "near_duplicates": "not audited", "label_listening": "not performed",
    }


def plan(
    rows: Sequence[Mapping], *, split: str, seed: int,
    impostors: int = 3, command_only: bool = False,
) -> list[dict]:
    """Deterministic same/different-speaker pairs; never silently drop
    commands."""
    audit(rows)
    if split not in SPLITS or type(impostors) is not int or impostors < 1:
        raise ValueError("Invalid split or impostor count")
    pool = sorted(
        (r for r in rows if r["split"] == split), key=lambda r: r["id"]
    )
    commands = [r for r in pool if r["role"] == "command"]
    if not commands:
        raise ValueError("No commands in requested split")
    wakes = defaultdict(list)
    for row in pool:
        if row["role"] == "wake":
            wakes[row["kind"], row["speaker_id"]].append(row)
    rng = random.Random(seed)
    result = []

    def append(command, wake, case):
        keys = {"case": case, "command_id": command["id"],
                "wake_id": wake["id"] if wake else None, "split": split}
        result.append({
            **keys, "trial_id": hashlib.sha256(
                canonical(keys).encode()
            ).hexdigest(),
            "speaker_id": command["speaker_id"],
            "owner_speaker_id": wake["speaker_id"] if wake else None,
            "kind": command["kind"], "source": command["source"],
            "expected": None
            if case == "wrong_speaker" else command["command"],
            "evidence_tier": "original_command_crop"
            if wake is None else "paired_crops_ownership_only",
            "gap_seconds": 0.5 if wake else 0.0,
        })

    for command in commands:
        append(command, None, "command_only")
        if command_only:
            continue
        own_key = (command["kind"], command["speaker_id"])
        if not wakes[own_key]:
            raise ValueError(
                f"Missing wake recording for {command['speaker_id']}"
            )
        append(command, rng.choice(wakes[own_key]), "same_speaker")
        other_keys = sorted(
            k for k in wakes
            if k[0] == command["kind"] and k != own_key and wakes[k]
        )
        if not other_keys:
            raise ValueError("Need a different speaker's wake recording")
        for key in rng.sample(other_keys, min(impostors, len(other_keys))):
            append(command, rng.choice(wakes[key]), "wrong_speaker")
    return result


def validate_trials(
    rows: Sequence[Mapping], trials: Sequence[Mapping]
) -> None:
    """Bind expectations, identities, cases and complete command coverage."""
    if not trials:
        raise ValueError("Empty trial plan")
    lookup = {row["id"]: row for row in rows}
    splits = {row["split"] for row in trials}
    if len(splits) != 1:
        raise ValueError("Do not mix evaluation splits")
    seen = set()
    cases = defaultdict(Counter)
    for trial in trials:
        case = trial["case"]
        if case not in {"command_only", "same_speaker", "wrong_speaker"}:
            raise ValueError("Unknown trial case")
        command = lookup.get(trial["command_id"])
        if command is None or command["role"] != "command":
            raise ValueError("Missing command recording")
        for key in ("split", "speaker_id", "kind", "source"):
            if trial.get(key) != command[key]:
                raise ValueError("Trial metadata does not match manifest")
        expected = None if case == "wrong_speaker" else command["command"]
        if trial.get("expected") != expected:
            raise ValueError("Trial expected command was changed")
        wake = lookup.get(trial.get("wake_id"))
        if case == "command_only":
            if (
                trial.get("wake_id") is not None
                or trial.get("owner_speaker_id") is not None
            ):
                raise ValueError("Command-only trial has an owner")
        else:
            if wake is None or wake["role"] != "wake":
                raise ValueError("Missing wake recording")
            if (
                wake["split"] != command["split"]
                or wake["kind"] != command["kind"]
            ):
                raise ValueError("Wake provenance/split mismatch")
            if trial.get("owner_speaker_id") != wake["speaker_id"]:
                raise ValueError("Owner metadata mismatch")
            same = wake["speaker_id"] == command["speaker_id"]
            if same != (case == "same_speaker"):
                raise ValueError("Trial speaker relationship is incorrect")
        keys = {
            key: trial[key] for key in (
                "case", "command_id", "wake_id", "split"
            )
        }
        identity = hashlib.sha256(canonical(keys).encode()).hexdigest()
        if trial.get("trial_id") != identity or identity in seen:
            raise ValueError("Invalid or duplicate trial identity")
        seen.add(identity)
        gap = trial.get("gap_seconds")
        if (
            type(gap) not in (int, float)
            or not math.isfinite(gap) or not 0 <= gap <= 15
        ):
            raise ValueError("Invalid trial gap")
        cases[command["id"]][case] += 1
    required = {
        row["id"]
        for row in rows if row["split"] in splits and row["role"] == "command"
    }
    if set(cases) != required:
        raise ValueError("Incomplete command coverage")
    paired = any(row["wake_id"] is not None for row in trials)
    for counts in cases.values():
        if counts["command_only"] != 1:
            raise ValueError("Incomplete command-only coverage")
        if (
            paired and (
                counts["same_speaker"] != 1 or counts["wrong_speaker"] < 1
            )
        ):
            raise ValueError("Incomplete paired coverage")


def _metrics(pairs: Sequence[tuple[Mapping, Mapping | None]]) -> dict:
    positive_n = sum(row["expected"] is not None for row, _ in pairs)
    negative_n = len(pairs) - positive_n
    correct = emitted = false_accepts = missing = errors = correct_rejects = 0
    baseline_correct = added_rejections = 0
    latencies = []
    for row, prediction in pairs:
        if prediction is None:
            missing += 1
            continue
        application = prediction.get("application", {})
        malformed = not isinstance(application, dict)
        if malformed:
            application = {}
        accepted = application.get("accepted") is True
        value = application.get("command")
        invalid = (
            malformed or type(application.get("accepted")) is not bool
            or not _command_valid(value)
            or (
                accepted and value is None
            ) or (not accepted and value is not None)
        )
        failed = bool(prediction.get("error")) or invalid
        errors += int(failed)
        emitted += int(accepted)
        if accepted and row["expected"] is None:
            false_accepts += 1
        exact = not failed and accepted and value == row["expected"]
        correct += int(exact)
        correct_rejects += int(
            not failed and not accepted and row["expected"] is None
        )
        before = prediction.get("application_before_speaker", {})
        if not isinstance(before, dict):
            before = {}
        before_correct = (
            row["expected"] is not None
            and before.get("accepted") is True
            and before.get("command") == row["expected"]
        )
        baseline_correct += int(before_correct)
        added_rejections += int(before_correct and not accepted and not failed)
        latency = prediction.get("total_ms")
        if (
            isinstance(
                latency, (int, float)
            ) and math.isfinite(latency) and latency >= 0
        ):
            latencies.append(latency)

    def rate(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "n": len(pairs), "positive_n": positive_n, "negative_n": negative_n,
        "correct_accepted": correct,
        "emitted": emitted, "false_accepts": false_accepts,
        "correct_rejections": correct_rejects, "missing_predictions": missing,
        "errors": errors, "complete": not (missing or errors),
        "recall": rate(correct, positive_n),
        "precision": rate(correct, emitted),
        "false_accept_rate": rate(false_accepts, negative_n),
        "baseline_correct_before_speaker": baseline_correct,
        "owner_added_rejections": added_rejections,
        "total_ms_p50": float(
            np.percentile(latencies, 50)
        ) if latencies else None,
        "total_ms_p95": float(
            np.percentile(latencies, 95)
        ) if latencies else None,
        "timing_n": len(latencies),
    }


def summarize(
    trials: Sequence[Mapping], predictions: Sequence[Mapping]
) -> dict:
    if not trials:
        raise ValueError("Empty trial plan")
    ids = {r["trial_id"] for r in trials}
    if len(ids) != len(trials):
        raise ValueError("duplicate trial id")
    seen = {}
    for row in predictions:
        identity = row["trial_id"]
        if identity not in ids:
            raise ValueError("unknown prediction trial id")
        if identity in seen:
            raise ValueError("duplicate prediction")
        seen[identity] = row
    pairs = [(row, seen.get(row["trial_id"])) for row in trials]
    result = {"overall": _metrics(pairs)}
    for column, name in (
        ("case", "by_case"), ("kind", "by_kind"),
        ("speaker_id", "by_speaker"), ("source", "by_source"),
    ):
        groups = defaultdict(list)
        for pair in pairs:
            groups[pair[0][column]].append(pair)
        result[name] = {
            key: _metrics(values) for key, values in sorted(groups.items())
        }
    groups = defaultdict(list)
    for pair in pairs:
        groups[canonical(pair[0]["expected"])].append(pair)
    result["by_command"] = {
        key: _metrics(values) for key, values in sorted(groups.items())
    }
    recalls = [
        m["recall"]
        for m in result["by_speaker"].values() if m["recall"] is not None
    ]
    result.update({
        "macro_speaker_recall": float(np.mean(recalls)) if recalls else None,
        "worst_speaker_recall": min(recalls) if recalls else None,
        "unique_command_recordings": len({r["command_id"] for r in trials}),
        "unique_wake_recordings": len(
            {r["wake_id"] for r in trials if r["wake_id"]}
        ),
        "unique_command_speakers": len({r["speaker_id"] for r in trials}),
        "qualification": "not_evaluated",
        "independence": (
            'Pairs reusing recordings/speakers are dependent; no binomial '
            'confidence claim.'
        ),
        "scope": (
            'command crops / annotated wake-command crops, not wake '
            'detection or live capture'
        ),
    })
    return result


def _bindings(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.resolve()): file_digest(path) for path in sorted(set(paths))
    }


def create_lock(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.manifest)
    audited = audit(rows, check_audio=True)
    trials = read_jsonl(args.trials)
    validate_trials(rows, trials)
    protocol = read_json(args.protocol)
    application = protocol["application"]
    if (
        application != {
            "mode": "demo", "intent_threshold": 0.95, "slot_threshold": 0.90
        }
    ):
        raise ValueError(
            "This experiment preserves the 0.95/0.90 demo contract"
        )
    paired = any(row["wake_id"] for row in trials)
    if (
        paired and not (
            args.speaker_backend and args.speaker_policy and args.backend_files
        )
    ):
        raise ValueError(
            (
                'Paired replay requires a validated backend, policy and '
                'bound backend files'
            )
        )
    files = [args.manifest, args.trials, args.protocol]
    bundle_files = [p for p in args.bundle.rglob("*") if p.is_file()]
    if not bundle_files:
        raise ValueError(
            "No model bundle files; do not substitute random weights"
        )
    files.extend(bundle_files)
    source_files = [Path(__file__), Path("scripts/runtime/speaker_session.py")]
    for directory in (
        "scripts/runtime", "vcm", "experiments/phrase-contract-v1/pcvcm"
    ):
        source_files.extend(Path(directory).glob("*.py"))
    backend_files = [Path(p) for p in (args.backend_files or [])]
    backend_binding = _bindings(backend_files)
    if args.speaker_policy:
        from scripts.runtime.speaker_session import SpeakerPolicy
        SpeakerPolicy(**read_json(args.speaker_policy))
        files.append(args.speaker_policy)
    if args.speaker_backend:
        module, _, factory = args.speaker_backend.partition(":")
        if not factory:
            raise ValueError("Use module:factory for the speaker backend")
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError("Speaker backend module not installed")
        source_files.append(Path(spec.origin))
    return {
        "schema_version": 1, "baseline": BASELINE, "created_unix": time.time(),
        "root": str(
            Path.cwd().resolve()
        ), "manifest": str(args.manifest.resolve()),
        "trials": str(
            args.trials.resolve()
        ), "bundle": str(args.bundle.resolve()),
        "engine": "ncnn", "application": application, "audit": audited,
        "file_bindings": _bindings(files + source_files + backend_files),
        "speaker_backend": args.speaker_backend,
        "speaker_policy": read_json(
            args.speaker_policy
        ) if args.speaker_policy else None,
        "backend_files": backend_binding,
        "backend_id": hashlib.sha256(
            canonical(backend_binding).encode()
        ).hexdigest() if paired else None,
        "selection_scope": (
            'configuration frozen before replay; no claim of never-viewed '
            'historical test data'
        ),
    }


def replay(lock_path: Path, output: Path) -> dict:
    """Direct runtime use: no gateway, AudioService, action imports or
    dispatch."""
    lock = read_json(lock_path)
    for path, expected in lock["file_bindings"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Locked input changed: {path}")
    if output.exists():
        raise FileExistsError(
            "Use a new output directory; partial runs remain evidence"
        )
    from scripts.runtime.pcv_final_runtime import Runtime
    from scripts.runtime.speaker_confidence import application_decision
    from scripts.runtime.speaker_session import (
        SpeakerPolicy, SpeakerSession, gate_application
    )

    records = read_jsonl(Path(lock["manifest"]))
    trials = read_jsonl(Path(lock["trials"]))
    audit(records, check_audio=True, root=Path(lock["root"]))
    validate_trials(records, trials)
    if Path.cwd().resolve() != Path(lock["root"]):
        raise ValueError("Run replay from the locked repository root")
    lookup = {r["id"]: r for r in records}
    runtime = Runtime(
        Path(lock["bundle"]), diagnostic=True, threads=1, engine="ncnn"
    )
    backend = None
    if lock["speaker_backend"]:
        module, factory = lock["speaker_backend"].split(":", 1)
        backend = getattr(importlib.import_module(module), factory)({
            "model_id": lock["backend_id"], "files": lock["backend_files"],
        })
    output.mkdir(parents=True)
    predictions = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for trial in trials:
            started = time.perf_counter()
            result = {"trial_id": trial["trial_id"]}
            try:
                pcm = read_audio(
                    lookup[trial["command_id"]], Path(lock["root"])
                )
                event = runtime.recognize(pcm)
                application = application_decision(event, lock["application"])
                result["application_before_speaker"] = application
                if trial["wake_id"]:
                    if backend is None:
                        raise ValueError("Missing speaker backend")
                    owner = SpeakerSession(
                        SpeakerPolicy(**lock["speaker_policy"])
                    )
                    wake = read_audio(
                        lookup[trial["wake_id"]], Path(lock["root"])
                    )
                    reference = backend.extract(wake, 16000)
                    if reference.model_id != lock["backend_id"]:
                        raise ValueError(
                            (
                                'Speaker evidence is not bound to the locked '
                                'backend'
                            )
                        )
                    armed = owner.arm(trial["trial_id"], reference, now=0.0)
                    if armed.accepted:
                        command = backend.extract(pcm, 16000)
                        windows = backend.windows(pcm, 16000)
                        verdict = owner.finish(
                            trial["trial_id"], command,
                            now=trial["gap_seconds"] + len(pcm) / 16000,
                            windows=windows,
                        )
                    else:
                        verdict = armed
                    result["speaker_verification"] = asdict(verdict)
                    application = gate_application(application, verdict)
                result["application"] = application
                result["runtime_timing"] = event.get("timing", {})
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["application"] = {"accepted": False, "command": None,
                                         "reason": "evaluation_error"}
            result["total_ms"] = (time.perf_counter() - started) * 1000
            stream.write(canonical(result) + "\n")
            stream.flush()
            predictions.append(result)
    metrics = summarize(trials, predictions)
    metrics["execution"] = {
        "lock_sha256": file_digest(lock_path),
        "python": platform.python_version(),
        "platform": platform.platform(), "engine": "ncnn", "threads": 1,
        "actions_dispatched": 0, "physical_pi_confirmed": False,
        "timing_scope": (
            'file decoding + command runtime + optional speaker backend; '
            'first call not removed'
        ),
    }
    write_new(output / "metrics.json", metrics)
    write_new(
        output / "complete.json", {
            "completed": True,
            "all_trials_valid": metrics["overall"]["complete"]
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--manifest", type=Path, required=True)
    audit_parser.add_argument("--check-audio", action="store_true")
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    plan_parser.add_argument("--seed", type=int, default=53)
    plan_parser.add_argument("--impostors", type=int, default=3)
    plan_parser.add_argument("--command-only", action="store_true")
    lock_parser = commands.add_parser("lock")
    for name in ("manifest", "trials", "bundle", "protocol"):
        lock_parser.add_argument(f"--{name}", type=Path, required=True)
    lock_parser.add_argument("--speaker-backend")
    lock_parser.add_argument("--speaker-policy", type=Path)
    lock_parser.add_argument("--backend-files", type=Path, nargs="+")
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--lock", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--trials", type=Path, required=True)
    score_parser.add_argument("--predictions", type=Path, required=True)
    for subparser in (
        audit_parser, plan_parser, lock_parser, replay_parser, score_parser
    ):
        subparser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.output.resolve().is_relative_to(
            (Path.cwd() / ".cache/speaker-demo-v1").resolve()
        )
    ):
        parser.error(
            (
                'Write outputs under .cache/speaker-demo-v1; use a fresh '
                'path per run'
            )
        )
    if args.command == "audit":
        write_new(
            args.output, audit(
                read_jsonl(args.manifest), check_audio=args.check_audio
            )
        )
    elif args.command == "plan":
        write_new(
            args.output,
            plan(
                read_jsonl(args.manifest), split=args.split,
                seed=args.seed, impostors=args.impostors,
                command_only=args.command_only,
            ),
            jsonl=True,
        )
    elif args.command == "lock":
        write_new(args.output, create_lock(args))
    elif args.command == "replay":
        result = replay(args.lock, args.output)
        if not result["overall"]["complete"]:
            raise SystemExit(2)
    else:
        write_new(
            args.output, summarize(
                read_jsonl(args.trials), read_jsonl(args.predictions)
            )
        )


if __name__ == "__main__":
    main()
