"""Exercise metadata provenance and leakage checks without downloading audio."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcm.source_audit import metadata_audit


def row(path="a.wav", speaker="s1", split="train"):
    return {
        "path": path, "speaker": speaker, "split": split,
        "label": "BRIGHTNESS_20", "phrase_id": "v1",
        "transcript": "Brightness 20 percent", "variant_id": "clean",
        "intent": "BRIGHTNESS",
    }


def tree(paths=("a.wav",)):
    return {
        "truncated": False,
        "tree": [
            {"type": "blob", "path": p, "sha": "same", "size": 20}
            for p in paths
        ],
    }


def test_audit_uses_files_not_claimed_readme_totals():
    result = metadata_audit([row()], tree())
    assert result["manifest_rows"] == result["tree_wavs"] == 1
    assert result["training_source_files"] == []


def test_speaker_and_pair_and_identical_blob_leakage_detected():
    rows = [row(), row("b.wav", split="test")]
    result = metadata_audit(rows, tree(("a.wav", "b.wav")))
    assert result["cross_split_speakers"] == ["s1"]
    assert result["cross_split_clean_noisy_groups"]
    assert result["cross_split_identical_blobs"] == ["same"]


def test_missing_and_duplicate_paths_are_not_ignored():
    result = metadata_audit([row("x.wav"), row("x.wav")], tree())
    assert result["duplicate_manifest_paths"] == ["x.wav"]
    assert result["manifest_paths_absent_from_tree"] == ["x.wav"]
    assert result["tree_wavs_absent_from_manifest"] == ["a.wav"]


def test_truncated_tree_cannot_establish_coverage():
    with pytest.raises(ValueError, match="complete"):
        metadata_audit([], {"truncated": True})
