"""Train ASR independently; the VCM checkpoint is never read by this process."""
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TORCH_HOME"] = str(root / ".cache/torch")
os.environ["CUDA_CACHE_PATH"] = str(root / ".cache/cuda")
deadline = time.monotonic() + 7200
required = [root / "data/manifests/librispeech.source.json", root / "reports/split_audit.json"]
while not all(path.exists() for path in required):
    if time.monotonic() > deadline:
        raise RuntimeError("ASR source acquisition/split audit did not finish")
    time.sleep(5)
subprocess.run([sys.executable, "-B", "-u", "scripts/prepare.py", "asr"], check=True)
subprocess.run([sys.executable, "-B", "-u", "-m", "tinyvcm.train", "asr", "--stage", "acoustic", "--run", "asr_acoustic_seed17", "--epochs", "40", "--batch-size", "48"], check=True)
subprocess.run([sys.executable, "-B", "-u", "-m", "tinyvcm.train", "asr", "--stage", "commands", "--run", "asr_commands_seed17", "--epochs", "30", "--batch-size", "48", "--lr", "0.0005", "--initialize", "runs/asr_acoustic_seed17/best.pt"], check=True)
