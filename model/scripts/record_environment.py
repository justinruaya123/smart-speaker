import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
packages = {}
for name in ["torch", "numpy", "scipy", "einops", "soundfile", "sounddevice", "onnx", "onnxruntime", "nbformat", "nbclient", "matplotlib", "pytest", "ipykernel"]:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
import torch
report = {"recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version, "executable": sys.executable, "system": platform.system(), "release": platform.release(), "processor": platform.processor(), "machine": platform.machine(), "cpu_count": os.cpu_count(), "packages": packages, "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None, "runtime_lock_note": "requirements-runtime.lock is resolved separately; the fresh runtime-only smoke environment is tested and recorded in runtime_smoke.json", "local_optional_dependencies": {"pyarrow": "25.0.1 in model/.deps; acquisition only"}}
(ROOT / "reports/environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
