"""Sequential finalization after the independent ASR training process completes."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
runtime = ROOT / (".runtime-venv/Scripts/python.exe" if os.name == "nt" else ".runtime-venv/bin/python")
deadline = time.monotonic() + 7200
while not (ROOT / "runs/asr_commands_seed17/complete.json").exists():
    if time.monotonic() > deadline:
        raise RuntimeError("ASR training did not complete before finalization deadline")
    time.sleep(5)

def run(python, args):
    print("FINALIZE " + " ".join(args), flush=True)
    subprocess.run([str(python), "-B", "-u", *args], check=True)

run(sys.executable, ["-m", "tinyvcm.export", "asr"])
run(sys.executable, ["-m", "tinyvcm.evaluate", "asr"])
run(runtime, ["scripts/evaluate_accent.py"])
run(runtime, ["scripts/runtime_smoke.py"])
for variant in ("fp32", "int8"):
    run(runtime, ["scripts/benchmark.py", "--variant", variant, "--threads", "2", "--output", f"reports/benchmark_windows_{variant}.json"])
run(sys.executable, ["scripts/build_report.py"])
run(runtime, ["scripts/package_pi4.py"])
print("ALL FINALIZATION STAGES COMPLETE", flush=True)
