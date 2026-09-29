"""Zero-state bootstrap contracts: deterministic, hash-checked, non-destructive."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/bootstrap_semifinal.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_semifinal", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_source(root: Path, name: str, payload: dict) -> None:
    folder = root / "data/sources"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.source.json").write_text(json.dumps(payload))


def test_source_plan_normalizes_repo_paths_without_downloading(tmp_path):
    payload = b"abc"
    write_source(tmp_path, "sample", {
        "url": "https://example.invalid/sample.bin",
        "path": "data\\raw\\sample\\sample.bin",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "license": "test",
    })
    plan = MODULE.build_download_plan(tmp_path, ["sample"])
    assert len(plan) == 1
    assert plan[0]["relative_path"] == "data/raw/sample/sample.bin"
    assert not (tmp_path / "data/raw").exists()


@pytest.mark.parametrize("bad", ["../outside.bin", "/absolute.bin", "data/../../bad"])
def test_source_path_escape_is_rejected(tmp_path, bad):
    write_source(tmp_path, "bad", {
        "url": "https://example.invalid/bad",
        "path": bad,
        "sha256": "0" * 64,
    })
    with pytest.raises(ValueError):
        MODULE.build_download_plan(tmp_path, ["bad"])


def test_matching_existing_file_is_reused_without_fetch(tmp_path):
    payload = b"already here"
    digest = hashlib.sha256(payload).hexdigest()
    entry = {
        "source": "sample",
        "url": "https://example.invalid/sample",
        "relative_path": "data/raw/sample.bin",
        "sha256": digest,
        "bytes": len(payload),
    }
    destination = tmp_path / entry["relative_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    called = False

    def fetch(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("matching file must not be fetched")

    result = MODULE.procure(entry, tmp_path, fetcher=fetch)
    assert result["status"] == "reused"
    assert called is False


def test_mismatched_existing_file_is_never_overwritten(tmp_path):
    entry = {
        "source": "sample",
        "url": "https://example.invalid/sample",
        "relative_path": "data/raw/sample.bin",
        "sha256": hashlib.sha256(b"expected").hexdigest(),
        "bytes": len(b"expected"),
    }
    destination = tmp_path / entry["relative_path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"user work")

    with pytest.raises(FileExistsError):
        MODULE.procure(entry, tmp_path, fetcher=lambda *_a, **_k: b"expected")
    assert destination.read_bytes() == b"user work"


def test_untracked_paths_are_allowed_but_tracked_edits_are_not():
    state = MODULE.classify_status([
        "?? datasets/",
        "?? notes.md",
        " M tracked.py",
        "A  staged.py",
    ])
    assert state["untracked"] == ["datasets/", "notes.md"]
    assert state["tracked_or_staged"] == [" M tracked.py", "A  staged.py"]


def test_pipeline_stage_order_is_fixed():
    names = [item["name"] for item in MODULE.pipeline_stages("run-001")]
    assert names == ["train", "validate", "publish_diagnostic"]
