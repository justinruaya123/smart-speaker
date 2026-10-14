"""Numerical/accounting tests; no synthesized or human speech claims."""
import copy

import numpy as np
import pytest

from final.tts_sanity import (
    asset_paths,
    merge_cases,
    review_complete,
    score_cases,
    to_pcm16,
    trial_id,
)


def contract():
    command = {"intent": "BRIGHTNESS", "slots": {"percent": 20}}
    return {
        "cases": [{"id": "B1-001", "text": "Brightness 20 percent",
                   "spoken_text": "Brightness twenty percent",
                   "command": command, "tier": "baseline"}],
        "catalog_entries": [
            {"id": "B1-001", "text": "brightness twenty percent",
             "spoken_text": "Brightness twenty percent",
             "command": command, "tier": "baseline"},
            {"id": "N1", "text": "turn on the mystery light",
             "command": None, "reason": "invalid_argument"},
        ],
    }


def test_merge_preserves_exact_text_and_every_source_case():
    cases = merge_cases(contract())
    assert len(cases) == 3
    assert cases[0]["spoken_text"] == "Brightness twenty percent"
    assert len({row["case_id"] for row in cases}) == 3
    assert len({row["spoken_text"] for row in cases}) == 2


def test_conflicting_labels_for_same_audio_fail():
    data = contract()
    data["catalog_entries"][0]["command"] = {"intent": "TIME", "slots": {}}
    with pytest.raises(ValueError, match="Conflicting"):
        merge_cases(data)


def test_duplicate_source_case_id_fails():
    data = contract()
    data["cases"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        merge_cases(data)


def test_missing_catalog_is_not_full_coverage():
    data = contract()
    data["catalog_entries"] = []
    with pytest.raises(ValueError):
        merge_cases(data)


def trials():
    return [{**case, "voice": "af_heart",
             "audio_id": trial_id("af_heart", case["spoken_text"])}
            for case in merge_cases(contract())]


def positive_result(percent=20):
    return {"audio_id": trials()[0]["audio_id"], "final_application": {
        "accepted": True, "command": {
            "intent": "BRIGHTNESS", "slots": {"percent": percent}},
        "reason": None}}


def negative_result(reason="invalid_argument"):
    return {"audio_id": trials()[-1]["audio_id"], "final_application": {
        "accepted": False, "command": None, "reason": reason}}


def test_exact_all_slots_and_duplicate_audio_denominators():
    result = score_cases(trials(), [positive_result(), negative_result()])
    assert result["unique_audio"]["trials"] == 2
    assert result["case_references"] == 3
    assert result["pass"] is True
    bad = score_cases(trials(), [positive_result(60), negative_result()])
    assert bad["unique_audio"]["recall"] == 0
    assert bad["pass"] is False


@pytest.mark.parametrize("negative", [None, {"fault": "read_error"}])
def test_fault_or_missing_is_not_a_correct_negative(negative):
    results = [positive_result()]
    if negative is not None:
        results.append({"audio_id": trials()[-1]["audio_id"], **negative})
    metrics = score_cases(trials(), results)
    assert metrics["unique_audio"]["correct_rejections"] == 0
    assert metrics["pass"] is False


def test_wrong_rejection_reason_fails_strict_sanity():
    outputs = [positive_result(), negative_result("OOV")]
    metrics = score_cases(trials(), outputs)
    assert metrics["unique_audio"]["false_accepts"] == 0
    assert metrics["pass"] is False


def test_rejected_candidate_cannot_be_promoted():
    value = positive_result()
    value["diagnostic_candidate"] = copy.deepcopy(value["final_application"])
    value["final_application"] = {"accepted": False, "command": None,
                                 "reason": "OOV"}
    result = score_cases(trials(), [value, negative_result()])
    assert result["unique_audio"]["recall"] == 0
    assert result["unique_audio"]["precision"] is None


def test_duplicate_or_unexpected_outputs_fail():
    with pytest.raises(ValueError):
        score_cases(trials(), [positive_result(), positive_result()])
    with pytest.raises(ValueError):
        score_cases(trials(), [{"audio_id": "unexpected", "fault": "x"}])


def test_cross_voice_scores_are_not_pooled_human_evidence():
    ts = trials()
    extra = [{**r, "voice": "am_fenrir", "audio_id": trial_id(
        "am_fenrir", r["spoken_text"])} for r in ts]
    result = score_cases(ts + extra, [positive_result(), negative_result()])
    assert result["per_voice"]["af_heart"]["recall"] == 1
    assert result["per_voice"]["am_fenrir"]["recall"] == 0
    assert result["human_generalization_established"] is False
    assert result["pass"] is False


def reviewed():
    generated = [{"audio_id": "a", "sha256": "abc", "status": "generated"}]
    review = {"blind_to_vcm": True, "items": [{
        "audio_id": "a", "sha256": "abc", "heard_exact": True,
        "quality": 4, "reviewer": "reviewer-01", "reviewed_at": "2026-10-15"}]}
    return generated, review


def test_quality_review_is_required_and_hash_bound():
    generated, review = reviewed()
    review_complete(generated, review)
    review["items"][0]["sha256"] = "changed"
    with pytest.raises(ValueError):
        review_complete(generated, review)


@pytest.mark.parametrize("field,value", [
    ("heard_exact", False), ("quality", 3), ("reviewer", ""),
    ("reviewed_at", ""), ("quality", True),
])
def test_inadequate_qa_blocks_qualification(field, value):
    generated, review = reviewed()
    review["items"][0][field] = value
    with pytest.raises(ValueError):
        review_complete(generated, review)


def test_missing_duplicate_or_unblinded_reviews_fail():
    generated, review = reviewed()
    for changed in [{"blind_to_vcm": False, "items": review["items"]},
                    {"blind_to_vcm": True, "items": []},
                    {"blind_to_vcm": True, "items": review["items"] * 2}]:
        with pytest.raises(ValueError):
            review_complete(generated, changed)


def test_resampling_is_real_not_a_rate_header_change():
    wave = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000) * 0.2
    result = to_pcm16(wave, 24000)
    assert result.shape == (16000,)
    assert np.isfinite(result).all()
    peak = np.argmax(abs(np.fft.rfft(result)))
    assert peak == 440


@pytest.mark.parametrize("wave", [np.zeros(24000), np.array([np.nan]),
                                  np.ones((1000, 2)), np.ones(24000) * 1.1])
def test_invalid_audio_is_not_silently_repaired(wave):
    with pytest.raises(ValueError):
        to_pcm16(wave, 24000)


def test_wrong_native_rate_is_not_silently_assumed():
    with pytest.raises(ValueError):
        to_pcm16(np.ones(1000) * 0.1, 16000)


def test_asset_hashes_are_checked(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"wrong")
    spec = {"files": {"model.onnx": "0" * 64}}
    with pytest.raises(ValueError, match="hash"):
        asset_paths(tmp_path, spec)


def test_per_intent_and_per_case_denominators_are_reported():
    result = score_cases(trials(), [positive_result(), negative_result()])
    assert result["per_intent"]["BRIGHTNESS"]["supported"] == 1
    assert result["per_case"]["cases/B1-001"]["supported"] == 1


def test_malformed_accepted_slots_count_as_fault():
    result = positive_result()
    result["final_application"]["command"]["slots"] = []
    metrics = score_cases(trials(), [result, negative_result()])
    assert metrics["unique_audio"]["faults"] == 1
    assert metrics["pass"] is False
