"""Run source checks, optional historical checks, then evaluate qualification.

A fresh clone is a supported starting point. Historical payloads are optional
regression evidence; their absence must not prevent fail-closed qualification.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "final_suite/reports"


def _review_module(root: Path):
    path = root / "final_suite/review.py"
    spec = importlib.util.spec_from_file_location("final_suite_review", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load release gate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_files(root: Path, relatives: list[str]) -> bool:
    return all((root / relative).is_file() for relative in relatives)


def plan_checks(root: Path = ROOT, include_browser: bool = False) -> list[dict]:
    """Return required and optional checks without touching the filesystem."""
    python = sys.executable
    checks = [
        {
            "name": "source_tests",
            "required": True,
            "ready": True,
            "command": [python, "-m", "pytest", "-q"],
            "reason": "Repository source and contract tests.",
        },
        {
            "name": "historical_evidence",
            "required": False,
            "ready": _all_files(root, [
                "experiments/vcm_services_v6/reports/probe_predictions.json",
                "experiments/vcm_services_v6/reports/training.json",
                "experiments/vcm_services_v6/reports/selection.json",
                "experiments/vcm_streaming_v3/data/phone_manifest.jsonl",
                "experiments/vcm_services_v6/data/probe_manifest.jsonl",
            ]),
            "command": [python, "final_suite/review.py"],
            "reason": "Optional historical prediction consolidation.",
        },
        {
            "name": "historical_dataset_audit",
            "required": False,
            "ready": _all_files(root, [
                "final_suite/archive/v1/manifest.jsonl",
                "final_suite/archive/v2/manifest.jsonl",
                "final_suite/archive/v2/expanded_manifest.jsonl",
                "experiments/vcm_streaming_v3/data/phone_manifest.jsonl",
                "experiments/vcm_streaming_v3/data/oov_manifest.jsonl",
                "experiments/vcm_streaming_v3/reports/phone_s53_test_predictions.json",
                "experiments/vcm_services_v6/data/probe_manifest.jsonl",
            ]),
            "command": [python, "final_suite/dataset_audit.py"],
            "reason": "Optional historical manifest audit.",
        },
        {
            "name": "native_decoder",
            "required": False,
            "ready": (root / "experiments/vcm_streaming_v3/src/build_native.py").is_file(),
            "command": [python, "experiments/vcm_streaming_v3/src/build_native.py"],
            "reason": "Optional restored native decoder build.",
        },
    ]

    visualizer_ready = (
        shutil.which("node") is not None
        and _all_files(root, [
            "visualizer/tests/core.test.mjs",
            "visualizer/tests/run_parity.mjs",
            "visualizer/assets/weights.bin",
            "visualizer/assets/recipe.json",
            "visualizer/assets/catalog.json",
        ])
    )
    checks.extend([
        {
            "name": "visualizer_core",
            "required": False,
            "ready": visualizer_ready,
            "command": ["node", "--test", "visualizer/tests/core.test.mjs"],
            "reason": "Optional browser/WASM source parity.",
        },
        {
            "name": "visualizer_parity",
            "required": False,
            "ready": visualizer_ready,
            "command": ["node", "visualizer/tests/run_parity.mjs"],
            "reason": "Optional restored tensor parity.",
        },
        {
            "name": "historical_math",
            "required": False,
            "ready": visualizer_ready and _all_files(root, [
                "experiments/vcm_streaming_v3/models/phone_s53.pt",
                "experiments/vcm_streaming_v3/models/phone_s53_40ms.onnx",
                "experiments/vcm_streaming_v3/runtime/normalization.npz",
            ]),
            "command": [python, "final_suite/verify_math.py"],
            "reason": "Optional restored PyTorch/ONNX/WASM numerical comparison.",
        },
    ])

    if include_browser:
        checks.append({
            "name": "browser_scenarios",
            "required": False,
            "ready": visualizer_ready and (root / "final_suite/browser_check.py").is_file(),
            "command": [python, "final_suite/browser_check.py"],
            "reason": "Optional browser interaction scenarios.",
        })
    return checks


def software_checks_passed(evidence: list[dict]) -> bool:
    """Skipped optional checks do not turn a source-only run into a failure."""
    for item in evidence:
        if item["status"] == "failed":
            return False
        if item.get("required") and item["status"] != "passed":
            return False
    return True


def qualification_snapshot(
    root: Path = ROOT,
    *,
    use_generated_summary: bool = False,
) -> dict:
    """Evaluate release eligibility even when no historical payload exists."""
    review = _review_module(root)
    summary_path = root / "final_suite/data/summary.json"
    if use_generated_summary and summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        release = payload.get("release")
        if not isinstance(release, dict) or "eligible" not in release:
            raise ValueError("Generated summary does not contain a release decision")
        return {
            "evaluated": True,
            "release": release,
            "evidence_source": "generated_historical_summary",
        }
    return {
        "evaluated": True,
        "release": review.release_decision({}),
        "evidence_source": "fail_closed_repository_state",
    }


def run(include_browser: bool = False, qualify: bool = False) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    evidence: list[dict] = []
    historical_passed = False

    for number, check in enumerate(plan_checks(ROOT, include_browser)):
        if not check["ready"]:
            evidence.append({
                "name": check["name"],
                "required": check["required"],
                "status": "missing" if check["required"] else "skipped",
                "returncode": None,
                "seconds": 0.0,
                "log": None,
                "reason": check["reason"],
            })
            continue

        started = time.monotonic()
        log = REPORTS / f"verification-{number:02d}-{check['name']}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                check["command"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        record = {
            "name": check["name"],
            "required": check["required"],
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "seconds": time.monotonic() - started,
            "log": str(log.relative_to(ROOT)),
            "reason": check["reason"],
        }
        evidence.append(record)
        print(result.returncode, " ".join(check["command"]), flush=True)
        if check["name"] == "historical_evidence" and result.returncode == 0:
            historical_passed = True
        if result.returncode:
            print(log.read_text(encoding="utf-8")[-4000:])
            if check["required"]:
                break

    qualification = qualification_snapshot(
        ROOT, use_generated_summary=historical_passed
    )
    output = {
        "software_checks_passed": software_checks_passed(evidence),
        "checks": evidence,
        "qualification_evaluated": qualification["evaluated"],
        "release": qualification["release"],
        "qualification_evidence_source": qualification["evidence_source"],
        "browser_requested": include_browser,
        "physical_pi4_tested": False,
        "physical_microphone_tested": False,
        "hardware_gpu_tested": False,
        "onnx_runtime_web_tested": False,
        "qualification_note": (
            "Missing historical payload is not a bootstrap error. "
            "Release remains fail-closed until required evidence is produced."
        ),
    }
    (REPORTS / "final_verification.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    if not output["software_checks_passed"]:
        return 1
    if qualify and not output["release"]["eligible"]:
        print("NO RELEASE WINNER:", output["release"]["failed_requirements"])
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.browser, args.qualify))
