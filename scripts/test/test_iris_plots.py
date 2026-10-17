"""Plot regeneration must work from committed receipts, with honest populations."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path):
    paths = [
        "scripts/plot_benchmarks.py",
        "docs/assets/iris-training-histories.json",
        *[f"experiments/iris-release-v1/{name}.json" for name in (
            "recognition-results", "release-choice", "training-execution",
            "development-selection", "pipeline-resources",
        )],
    ]
    for name in paths:
        source = ROOT / name
        assert source.is_file(), f"Missing committed plot input: {name}"
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path


def render(checkout):
    return subprocess.run(
        [sys.executable, str(checkout / "scripts/plot_benchmarks.py")],
        cwd=checkout, text=True, capture_output=True, timeout=90,
    )


def test_fresh_checkout_regenerates_from_receipts_without_cache(checkout):
    result = render(checkout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (checkout / ".cache").exists()
    assets = checkout / "docs/assets"
    assert len(list(assets.glob("*.png"))) == len(list(assets.glob("*.svg"))) == 4
    data = json.loads((assets / "iris-benchmark-data.json").read_text())
    official, personal, human, fresh = data["command_benchmarks"]
    assert official["denominator"] == 196
    assert official["baseline"]["policy_exact"] == 138
    assert official["baseline"]["policy_correct_rejects"] == 9
    assert personal["denominator"] == 36
    assert personal["original_human_positive_n"] == 0
    assert human["baseline"]["unknown_accepts"] == 12
    assert human["candidate"]["unknown_accepts"] == 14
    assert fresh["denominator"] == 444
    assert data["promotion"]["promoted"] is False
    assert data["desktop_resources"]["physical_pi_measured"] is False


def test_changed_official_population_rejects_mislabelled_plot(checkout):
    path = checkout / "experiments/iris-release-v1/recognition-results.json"
    value = json.loads(path.read_text())
    value["release_official"]["accepted"]["command_level"]["n"] = 195
    path.write_text(json.dumps(value))
    result = render(checkout)
    assert result.returncode != 0
    assert "196-row" in result.stderr


def test_history_selection_must_match_locked_receipt(checkout):
    path = checkout / "docs/assets/iris-training-histories.json"
    value = json.loads(path.read_text())
    value["runs"][0]["history"][0]["accepted_selection"][0] += 0.1
    path.write_text(json.dumps(value))
    result = render(checkout)
    assert result.returncode != 0
    assert "history" in result.stderr.lower()
