"""Data-only preservation and separation checks; no model inference."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from scripts.test import vcm_default_benchmark_data as data
from scripts.test.speaker_demo_suite import pcm_digest


def native_wav():
    stream = io.BytesIO()
    sf.write(stream, np.array([0.0, 0.25, -0.25], dtype="float32"), 16000,
             format="WAV", subtype="PCM_16")
    return stream.getvalue()


def source_row(*, negative=False):
    return {
        "audio": {"bytes": native_wav(), "path": "unused-source-name.wav"},
        "file": "audio/holdout_000000_real_voice_play.wav",
        "transcript": "Play music" if not negative else "The beach is lovely.",
        "command": "PLAY_MUSIC" if not negative else "OUT_OF_SCOPE",
        "variation": "Play music" if not negative else "ignored raw variation",
        "slot_value": "" if not negative else "ignored raw slot",
        "speaker_id": "speaker-original", "source": "real_voice",
        "is_synthetic": 0, "out_of_scope": int(negative),
        "accent_group": "original accent", "whisper_check": "",
    }


def clip(row, idx):
    pcm, _ = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    negative = bool(row["out_of_scope"])
    return SimpleNamespace(
        idx=idx, file=row["file"], transcript=row["transcript"],
        intent="OUT_OF_SCOPE" if negative else row["command"],
        slot_value="" if negative else row["slot_value"],
        variation="" if negative else row["variation"],
        speaker_id=row["speaker_id"], is_synthetic=bool(row["is_synthetic"]),
        accent_group=row["accent_group"], audio=pcm,
    )


def test_policy_recovers_only_hash_verified_original_line_endings(tmp_path):
    from scripts.train.pcv_release_prepare import PROPOSAL, PROPOSAL_SHA256
    path = tmp_path / "review.json"
    original = PROPOSAL.read_bytes().replace(b"\r\n", b"\n")
    path.write_bytes(original)
    policy, receipt = data.load_reviewed_policy(path)
    assert policy["proposal_sha256"] == PROPOSAL_SHA256
    assert receipt["verified_original_sha256"] == PROPOSAL_SHA256
    assert receipt["disk_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    path.write_bytes(original.replace(b"PLAY_MUSIC", b"PLAY_OTHER", 1))
    with pytest.raises(ValueError, match="review.*hash"):
        data.load_reviewed_policy(path)


def test_native_audio_and_both_reference_layers_are_preserved(tmp_path):
    rows = [source_row(), source_row(negative=True)]
    policy, _ = data.load_reviewed_policy()
    derived = data.derive_rows(rows, [clip(r, i) for i, r in enumerate(rows)],
                               tmp_path, policy)
    assert len(derived) == 2
    for i, result in enumerate(derived):
        raw = rows[i]
        assert Path(result["audio"]).read_bytes() == raw["audio"]["bytes"]
        assert result["publisher_intent"] == raw["command"]
        assert result["publisher_variation"] == raw["variation"]
        assert result["publisher_slot"] == raw["slot_value"]
        assert result["publisher_transcript"] == raw["transcript"]
        assert result["publisher_metadata"] == {
            k: v for k, v in raw.items() if k != "audio"
        }
        assert result["transcript"] == raw["transcript"]
        assert result["fresh"] is False and result["regression"] is True
    assert derived[0]["command"] == {"intent": "PLAY_MUSIC", "slots": {}}
    assert derived[1]["command"] is None
    assert derived[1]["reference_status"] == "unsupported"
    assert derived[1]["official_variation"] == ""
    assert derived[1]["official_slot"] == ""


def test_duplicate_pcm_does_not_drop_either_default_row(tmp_path):
    raw = source_row(negative=True)
    policy, _ = data.load_reviewed_policy()
    rows = data.derive_rows([raw, raw], [clip(raw, 0), clip(raw, 1)],
                            tmp_path, policy)
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]
    assert rows[0]["pcm_sha256"] == rows[1]["pcm_sha256"]
    assert rows[0]["audio"] == rows[1]["audio"]


@pytest.mark.parametrize("change", ["samples", "transcript", "intent", "idx"])
def test_disagreement_with_unchanged_default_loader_fails(tmp_path, change):
    raw = source_row()
    original = clip(raw, 0)
    if change == "samples":
        original.audio = original.audio * 0.5
    elif change == "idx":
        original.idx = 3
    else:
        setattr(original, change, "changed")
    policy, _ = data.load_reviewed_policy()
    with pytest.raises(ValueError, match="default loader"):
        data.derive_rows([raw], [original], tmp_path, policy)


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 2)])
def test_derivation_never_silently_resamples_or_downmixes(tmp_path, rate, channels):
    raw = source_row()
    stream = io.BytesIO()
    sf.write(stream, np.zeros((3,) if channels == 1 else (3, channels)), rate,
             format="WAV", subtype="PCM_16")
    raw["audio"]["bytes"] = stream.getvalue()
    policy, _ = data.load_reviewed_policy()
    with pytest.raises(ValueError, match="native mono 16 kHz"):
        data.derive_rows([raw], [clip(raw, 0)], tmp_path, policy)


def test_complete_count_is_required(tmp_path):
    policy, _ = data.load_reviewed_policy()
    with pytest.raises(ValueError, match="count"):
        data.derive_rows([source_row()], [], tmp_path, policy)


def identity_row(name, **extra):
    return {"id": name, "speaker_id": name + "-speaker",
            "sha256": name + "-file", "pcm_sha256": name + "-pcm",
            "pair_id": name + "-pair", "recording_group": name + "-group",
            **extra}


@pytest.mark.parametrize("key", ["sha256", "pcm_sha256", "speaker_id",
                                 "pair_id", "recording_group"])
def test_each_overlap_dimension_retains_specific_row_ids(key):
    a, b = identity_row("held"), identity_row("fit")
    b[key] = a[key]
    result = data.overlap(a_rows=[a], b_rows=[b])
    assert result[key] == {"rows": 1, "ids": ["held"]}


def test_explicit_aliases_and_raw_publisher_identities_are_checked():
    held = identity_row("held", publisher_metadata={"speaker_id": "raw-person"})
    fit = identity_row("fit", speaker_aliases=["raw-person"])
    result = data.overlap(a_rows=[held], b_rows=[fit])
    assert result["declared_identity_or_alias"] == {"rows": 1, "ids": ["held"]}


def test_unresolved_alias_flag_is_a_limit_and_does_not_invent_aliases():
    a = identity_row("held", speaker_aliases_resolved=False)
    b = identity_row("fit", speaker_aliases_resolved=False)
    result = data.overlap(a_rows=[a], b_rows=[b])
    assert not any(value["rows"] for value in result.values())


def test_regression_overlap_does_not_become_fitting_contamination():
    held = identity_row("held")
    fit = identity_row("fit")
    views = {"train": [fit], "dev": [], "calibration": [], "enrollment": [],
             "fresh_human": [], "holdout": [held], "test": [held]}
    report = data.separation([held], views)
    assert report["fit_separation_ready"] is True
    assert report["current_default_vs_original196_regression"]["pcm_sha256"]["rows"] == 1
    assert report["regression_overlap_allowed"] is True


def test_fit_overlap_fails_separation_without_removing_rows():
    held = identity_row("held")
    views = {"train": [held], "dev": [], "calibration": [], "enrollment": [],
             "fresh_human": [], "holdout": [], "test": []}
    report = data.separation([held], views)
    assert report["fit_separation_ready"] is False
    assert report["default_vs_fit"]["train"]["pcm_sha256"]["ids"] == ["held"]


def test_pcm_hash_is_existing_canonical_hash(tmp_path):
    raw = source_row()
    reference = clip(raw, 0)
    policy, _ = data.load_reviewed_policy()
    result = data.derive_rows([raw], [reference], tmp_path, policy)[0]
    assert result["pcm_sha256"] == pcm_digest(reference.audio)
