"""Rebuild the runtime bundle from pinned public sources in the root environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MODEL = Path(__file__).resolve().parents[1] / "model"
STEPS = [
    ["scripts/acquire.py", "mirror"],
    ["scripts/acquire.py", "libri"],
    ["scripts/prepare.py", "vcm"],
    ["scripts/prepare.py", "asr"],
    ["-m", "tinyvcm.train", "vcm", "--run", "vcm_seed17", "--seed", "17", "--epochs", "50", "--batch-size", "48"],
    ["-m", "tinyvcm.train", "vcm", "--run", "vcm_context_seed17", "--seed", "17", "--epochs", "60", "--batch-size", "48",
     "--width", "192", "--pooling", "mean_max", "--kernels", "7", "11", "15", "19", "23"],
    ["-m", "tinyvcm.export", "vcm"],
    ["-m", "tinyvcm.evaluate", "vcm"],
    ["scripts/evaluate_open_set.py"],
    ["-m", "tinyvcm.train", "asr", "--stage", "acoustic", "--run", "asr_acoustic_seed17", "--seed", "17",
     "--epochs", "40", "--batch-size", "48"],
    ["-m", "tinyvcm.train", "asr", "--stage", "commands", "--run", "asr_commands_seed17", "--seed", "17",
     "--epochs", "30", "--batch-size", "48", "--lr", "0.0005", "--initialize", "runs/asr_acoustic_seed17/best.pt"],
    ["-m", "tinyvcm.export", "asr"],
    ["-m", "tinyvcm.evaluate", "asr"],
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the recipe without downloading or training")
    args = parser.parse_args(argv)
    if not args.dry_run:
        import torch

        if not torch.cuda.is_available():
            parser.error("The recorded training recipe requires CUDA. Use --dry-run to inspect it.")
        existing = [name for name in ("vcm_seed17", "vcm_context_seed17", "asr_acoustic_seed17", "asr_commands_seed17")
                    if (MODEL / "runs" / name / "last.pt").exists()]
        if existing:
            parser.error(f"Training runs already exist: {', '.join(existing)}. Rebuild in a fresh checkout.")
    for index, step in enumerate(STEPS, 1):
        command = [sys.executable, "-B", *step]
        print(f"[{index}/{len(STEPS)}] (cwd: model) {subprocess.list2cmdline(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=MODEL, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
