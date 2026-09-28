"""Restoration must not rewrite production history or accept path traversal."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest

PATH = Path(__file__).resolve().parents[1] / "restore_payload.py"
SPEC = importlib.util.spec_from_file_location("restore_payload", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def package(tmp_path, files):
    path = tmp_path / "payload.zip"
    with zipfile.ZipFile(path, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    return path


def test_dry_run_makes_no_changes(tmp_path):
    archive = package(tmp_path, {"payload/visualizer/assets/x.bin": b"abc"})
    repo = tmp_path / "repo"
    result = MODULE.restore(archive, repo)
    assert result["new_files"] == 1
    assert not repo.exists()


def test_production_readme_and_git_are_not_overwritten(tmp_path):
    archive = package(tmp_path, {
        "payload/README.md": "new", "payload/.git/config": "bad",
        "payload/final_suite/data/x.json": "{}"})
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("original")
    MODULE.restore(archive, repo, True)
    assert (repo / "README.md").read_text() == "original"
    assert not (repo / ".git").exists()
    assert (repo / "final_suite/data/x.json").is_file()


@pytest.mark.parametrize("member", ["payload/../bad", "payload/visualizer/../../bad"])
def test_traversal_is_rejected(tmp_path, member):
    archive = package(tmp_path, {member: "bad"})
    with pytest.raises(ValueError):
        MODULE.restore(archive, tmp_path / "repo", True)


def test_checksum_mismatch_rejected(tmp_path):
    archive = package(tmp_path, {"payload/visualizer/assets/x": "abc"})
    with pytest.raises(ValueError):
        MODULE.restore(archive, tmp_path / "repo", True, "0" * 64)


def test_conflicting_models_not_overwritten(tmp_path):
    archive = package(tmp_path, {"payload/visualizer/assets/x.bin": "new"})
    repo = tmp_path / "repo"
    destination = repo / "visualizer/assets/x.bin"
    destination.parent.mkdir(parents=True)
    destination.write_text("old")
    with pytest.raises(FileExistsError):
        MODULE.restore(archive, repo, True)
    assert destination.read_text() == "old"


def test_identical_restore_is_idempotent(tmp_path):
    archive = package(tmp_path, {"payload/visualizer/assets/x.bin": "same"})
    repo = tmp_path / "repo"
    MODULE.restore(archive, repo, True)
    result = MODULE.restore(archive, repo, True)
    assert result["new_files"] == 0
    assert result["unchanged"] == 1
