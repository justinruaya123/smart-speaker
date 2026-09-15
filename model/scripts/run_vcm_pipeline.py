"""Wait for complete public audio acquisition, then prepare and train one fresh run."""
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
while not (root / "data/manifests/slurp_audio_mirror.source.json").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("SLURP acquisition did not finish; refusing incomplete-corpus training")
    time.sleep(5)
subprocess.run([sys.executable, "-B", "-u", "scripts/prepare.py", "vcm"], check=True)
subprocess.run([sys.executable, "-B", "-u", "-m", "tinyvcm.train", "vcm", "--run", "vcm_seed17", "--epochs", "50", "--batch-size", "48"], check=True)
