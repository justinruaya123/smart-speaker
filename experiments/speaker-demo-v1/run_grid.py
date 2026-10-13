"""Thin matched-ablation driver. Prints a plan unless --execute is supplied."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def commands_for(args: argparse.Namespace) -> list[list[str]]:
    arms = (("uniform-ce", "uniform", "0"),
            ("balanced-ce", "balanced", "0"),
            ("balanced-supcon", "balanced", "0.1"))
    commands = []
    for name, sampling, weight in arms:
        for seed in args.seeds:
            commands.append([
                sys.executable, "-m", "scripts.train.speaker_demo_train",
                "--manifest", str(args.manifest),
                "--catalog", str(args.catalog),
                "--output", str(args.output / f"{name}-s{seed}"),
                "--seed", str(seed), "--sampling", sampling,
                "--contrastive-weight", weight, "--device", args.device,
                "--epochs", str(args.epochs),
                "--steps-per-epoch", str(args.steps_per_epoch),
                "--batch-size", str(args.batch_size),
            ])
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[53, 71, 89])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cache = (Path.cwd() / ".cache/speaker-demo-v1").resolve()
    if not args.output.resolve().is_relative_to(cache):
        parser.error("Output must be under .cache/speaker-demo-v1")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    commands = commands_for(args)
    if not args.execute:
        print(json.dumps(commands, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    receipts = []
    failed = False
    for index, command in enumerate(commands):
        started = time.time()
        log = args.output / f"stage-{index:02d}.log"
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        receipts.append({
            "command": command, "returncode": completed.returncode,
            "seconds": time.time() - started, "log": str(log),
        })
        if completed.returncode:
            failed = True
            break
    receipt = {"stages": receipts, "planned": len(commands),
               "complete": not failed and len(receipts) == len(commands)}
    with (args.output / "grid.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
