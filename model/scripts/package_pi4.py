"""Bundle CPU inference and the fixed public voice suite for transfer to Pi 4."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
rows = [json.loads(line) for line in (ROOT / "data/manifests/voice_test_suite.jsonl").read_text().splitlines()]
files = [ROOT / name for name in ("README.md", "IMPLEMENTATION_NOTES.md", "TinyVCM_report.ipynb", "pyproject.toml", "requirements-runtime.lock", "scripts/benchmark.py", "data/manifests/voice_test_suite.jsonl", "reports/voice_test_results.jsonl", "reports/vcm_metrics.json", "reports/asr_metrics.json", "reports/voice_suite_metrics.json", "reports/joint_metrics.json", "reports/coverage.json", "reports/runtime_smoke.json", "reports/benchmark_windows_fp32.json", "reports/benchmark_windows_int8.json")]
files += list((ROOT / "tinyvcm").glob("*.py"))
files += list((ROOT / "configs").glob("*"))
files += list((ROOT / "artifacts/tinyvcm").glob("*"))
files += [ROOT / row["audio"] for row in rows]
target = ROOT / "artifacts/tinyvcm_pi4.tar.gz"
target.parent.mkdir(exist_ok=True)
with tarfile.open(target, "w:gz") as archive:
    for path in sorted(set(files)):
        if path.is_file():
            archive.add(path, arcname="model/" + path.relative_to(ROOT).as_posix(), recursive=False)
facts = {"path": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "voice_recordings": len(rows), "purpose": "Runtime source, pinned dependencies, models, configuration, benchmark and fixed public voice-test clips. Full training data and report execution dependencies are not included. Read model/README.md for source audio attribution and licenses."}
(ROOT / "reports/pi4_package.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
print(json.dumps(facts, indent=2))
