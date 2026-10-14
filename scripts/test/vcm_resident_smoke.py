"""Exercise the existing candidate worker without labels or provider actions.

This is a service-protocol smoke, not an acoustic scorer or Pi qualification.
Run with the repository's configured interpreter and an explicit candidate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
MAX_INPUT_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_traffic(events: list[dict], request_ids: list[str], engine: str,
                   checkpoint_hash: str, contract_hash: str) -> dict:
    """Reject incomplete, misbound or actionable service traces."""
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("Missing or invalid worker events")
    if any(event.get("actions_enabled") is not False for event in events):
        raise ValueError("Every event must explicitly disable actions")
    if len({event.get("session_id") for event in events}) != 1:
        raise ValueError("Expected exactly one resident model session")
    if not events[0].get("session_id"):
        raise ValueError("Missing resident session identity")
    if [event.get("sequence") for event in events] != list(
        range(1, len(events) + 1)
    ):
        raise ValueError("Missing, duplicated or reordered service events")
    ready = [event for event in events if event.get("event") == "ready"]
    if len(ready) != 1:
        raise ValueError("Expected exactly one successful model load")
    replies = [event for event in events if event.get("event") == "reply"]
    if any(event.get("ok") is not True for event in replies):
        raise ValueError("Worker reported a request fault")
    reply_ids = [event.get("id") for event in replies]
    if sorted(reply_ids) != sorted([*request_ids, "status"]):
        raise ValueError("Reply coverage mismatch")
    status = next(event for event in replies if event["id"] == "status")
    for event in (ready[0], status):
        if (event.get("loaded") is not True or event.get("engine") != engine
                or event.get("model_sha256") != checkpoint_hash
                or event.get("contract_hash") != contract_hash
                or event.get("command_only") is not True
                or event.get("owner_enforced") is not False
                or event.get("qualification_verified") is not False):
            raise ValueError("Incorrect candidate identity or evidence claim")
        config = event.get("application_config", {})
        if (config.get("intent_threshold") != 0.95
                or config.get("slot_threshold") != 0.90):
            raise ValueError("Fixed application thresholds changed")
    commands = [event for event in events if event.get("event") == "command"]
    if [event.get("request_id") for event in commands] != request_ids:
        raise ValueError("Command count or request attribution mismatch")
    accepted = 0
    for event in commands:
        app = event.get("final_application")
        if (not isinstance(app, dict) or app != event.get("application")
                or event.get("source") != "wav"
                or type(app.get("accepted")) is not bool):
            raise ValueError("Invalid or inconsistent authoritative verdict")
        if app["accepted"] != (app.get("command") is not None):
            raise ValueError("Command presence disagrees with acceptance")
        accepted += app["accepted"]
    if any(event.get("event") == "error" for event in events):
        raise ValueError("Worker emitted a protocol fault")
    return {
        "status": "passed_service_smoke", "commands": len(commands),
        "accepted": accepted, "rejected": len(commands) - accepted,
        "session_loads": len(ready), "engine": engine,
        "checkpoint_sha256": checkpoint_hash, "contract_hash": contract_hash,
        "acoustic_qualification": False, "physical_pi_qualification": False,
        "owner_qualification": False, "actions_enabled": False,
    }


def run(args: argparse.Namespace, output: Path) -> dict:
    recipe_path = args.bundle / "recipe.json"
    contract_path = args.bundle / "contract.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if recipe.get("format") != "vcm-intent-slots-v1":
        raise ValueError("Explicit intent-slot candidate required")
    request_ids, controls, recordings = [], [], []
    total = 0
    for index, path in enumerate(args.wav):
        if not path.is_file() or not 44 <= path.stat().st_size <= 4 * 1024**2:
            raise ValueError("Smoke WAV must be a file of 44 bytes..4 MiB")
        data = path.read_bytes()
        recordings.append({
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
        encoded = base64.b64encode(data).decode("ascii")
        for repeat in range(args.repeat):
            identifier = f"wav-{index}-{repeat}"
            request_ids.append(identifier)
            control = json.dumps({
                "op": "wav", "id": identifier, "data": encoded,
            })
            total += len(control) + 1
            if total > MAX_INPUT_BYTES:
                raise ValueError("Smoke payload exceeds 64 MiB; split the run")
            controls.append(control)
    controls.extend([
        json.dumps({"op": "status", "id": "status"}),
        json.dumps({"op": "shutdown"}),
    ])
    command = [
        sys.executable, "-m", "scripts.runtime.speaker_worker",
        "--experimental-intent-slots", "--bundle", str(args.bundle.resolve()),
        "--engine", args.engine, "--threads", str(args.threads),
    ]
    identity = {
        "command": command, "recipe_sha256": sha256(recipe_path),
        "contract_sha256": sha256(contract_path), "recordings": recordings,
        "requested_trials": len(request_ids), "python": sys.version,
        "platform": platform.platform(), "actions_enabled": False,
    }
    write_new(output / "run-identity.json", identity)
    environment = dict(os.environ)
    environment.pop("PCV_BENCHMARK_LOG", None)
    environment.update({"ACTIONS_ENABLED": "false", "PYTHONUNBUFFERED": "1"})
    started = time.perf_counter()
    with (output / "events.jsonl").open("x", encoding="utf-8") as stdout:
        with (output / "stderr.txt").open("x", encoding="utf-8") as stderr:
            result = subprocess.run(
                command, input="\n".join(controls) + "\n", text=True,
                stdout=stdout, stderr=stderr, cwd=ROOT, env=environment,
                timeout=args.timeout, check=False,
            )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise ValueError(f"Worker exited {result.returncode}; see stderr.txt")
    # Detect changes to the inputs whose bytes were actually replayed.
    if (sha256(recipe_path) != identity["recipe_sha256"]
            or sha256(contract_path) != identity["contract_sha256"]
            or any(sha256(Path(row["path"])) != row["sha256"]
                   for row in recordings)):
        raise ValueError("Candidate or WAV input changed during replay")
    text = (output / "events.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in text.splitlines()]
    report = verify_traffic(
        events, request_ids, args.engine, recipe["checkpoint_source_sha256"],
        contract["contract_hash"],
    )
    report.update({
        "session_wall_seconds": elapsed,
        "timing_scope": (
            "Host process startup, WAV IPC, inference and shutdown"
        ),
        "events_sha256": sha256(output / "events.jsonl"),
        "accuracy_measured": False,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wav", type=Path, nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--engine", choices=["ncnn", "onnxruntime"], default="ncnn",
    )
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if (not args.run_id or args.run_id in (".", "..")
            or "/" in args.run_id or "\\" in args.run_id):
        parser.error("Use a simple, unique run ID")
    if not 1 <= args.repeat <= 1000 or not 0 < args.timeout <= 3600:
        parser.error("Repeat must be 1..1000 and timeout 0..3600 seconds")
    output = ROOT / ".cache/vcm-final-readiness-v1" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(args, output)
    except (ValueError, OSError, KeyError, TypeError,
            subprocess.SubprocessError) as error:
        report = {
            "status": "failed_service_smoke", "reason": str(error),
            "acoustic_qualification": False,
            "physical_pi_qualification": False,
            "actions_enabled": False,
        }
        write_new(output / "result.json", report)
        print(json.dumps(report, allow_nan=False))
        raise SystemExit(2) from error
    write_new(output / "result.json", report)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
