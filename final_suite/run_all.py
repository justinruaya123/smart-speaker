"""Run software/evidence checks separately from the deliberately failing release gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "final_suite/reports"


def run(include_browser: bool = False, qualify: bool = False) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    required = [ROOT / "experiments/vcm_services_v6/reports/probe_predictions.json",
                ROOT / "visualizer/assets/weights.bin"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("Restore the full payload before running the complete suite:", missing)
        return 2
    commands = [
        [sys.executable, "experiments/vcm_streaming_v3/src/build_native.py"],
        [sys.executable, "final_suite/review.py"],
        [sys.executable, "final_suite/dataset_audit.py"],
        [sys.executable, "-m", "pytest", "-q"],
        ["node", "--test", "visualizer/tests/core.test.mjs"],
        ["node", "visualizer/tests/run_parity.mjs"],
        [sys.executable, "final_suite/verify_math.py"],
    ]
    if include_browser:
        commands.append([sys.executable, "final_suite/browser_check.py"])
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    evidence = []
    for number, command in enumerate(commands):
        started = time.monotonic()
        log = REPORTS / f"verification-{number:02d}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=300)
        evidence.append({"command": command, "returncode": result.returncode,
                         "seconds": time.monotonic() - started,
                         "log": str(log.relative_to(ROOT))})
        print(result.returncode, " ".join(command), flush=True)
        if result.returncode:
            print(log.read_text()[-4000:])
            break
    summary = json.loads((ROOT / "final_suite/data/summary.json").read_text())
    output = {
        "software_checks_passed": all(r["returncode"] == 0 for r in evidence)
        and len(evidence) == len(commands),
        "checks": evidence,
        "release": summary["release"],
        "browser_requested": include_browser,
        "physical_pi4_tested": False,
        "physical_microphone_tested": False,
        "hardware_gpu_tested": False,
        "onnx_runtime_web_tested": False,
        "qualification_note": "Software success is not acoustic release approval.",
    }
    (REPORTS / "final_verification.json").write_text(json.dumps(output, indent=2))
    if not output["software_checks_passed"]:
        return 1
    if qualify and not summary["release"]["eligible"]:
        print("NO RELEASE WINNER:", summary["release"]["failed_requirements"])
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.browser, args.qualify))
