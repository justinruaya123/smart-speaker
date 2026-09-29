"""Development selection must protect command semantics, not only phonemes."""
from copy import deepcopy

import pytest

from research.no_asr_followup.selection import assess_candidate


def row(key, category, correct=True):
    target = {"intent": category, "value": 73}
    return {
        "id": key, "pcm_sha256": key.zfill(64), "split": "dev",
        "category": category, "target": target,
        "candidate": target if correct else {"intent": category, "value": 3},
        "ties": [],
    }


def test_improvement_is_eligible_only_on_matched_development():
    replay = [row("1", "number"), row("2", "song")]
    before, after = [row("3", "external", False)], [row("3", "external")]
    assert assess_candidate(replay, replay, before, after)["eligible"]


def test_low_per_cannot_hide_a_lost_music_command():
    replay = [row("1", "number"), row("2", "song")]
    candidate = [row("1", "number"), row("2", "song", False)]
    report = assess_candidate(replay, candidate,
                              [row("3", "external", False)],
                              [row("3", "external")])
    assert not report["eligible"]
    assert report["replay"]["lost_ids"] == ["2"]


def test_net_gain_does_not_hide_individual_regression():
    replay = [row("1", "number"), row("2", "number", False)]
    candidate = [row("1", "number", False), row("2", "number")]
    report = assess_candidate(replay, candidate,
                              [row("3", "external", False)],
                              [row("3", "external")])
    assert not report["eligible"]


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("split", "calibration"),
    ("pcm_sha256", "different"), ("target", {"intent": "other"}),
    ("category", "other"),
])
def test_incompatible_or_test_rows_are_rejected(field, value):
    replay = [row("1", "number")]
    changed = deepcopy(replay)
    changed[0][field] = value
    with pytest.raises(ValueError):
        assess_candidate(replay, changed, [row("2", "external")],
                         [row("2", "external")])


def test_empty_or_missing_rows_do_not_pass():
    with pytest.raises(ValueError):
        assess_candidate([], [], [], [])
    with pytest.raises(ValueError):
        assess_candidate([row("1", "number")], [],
                         [row("2", "external")], [row("2", "external")])


def test_duplicate_ids_do_not_inflate_support():
    replay = [row("1", "number"), row("1", "number")]
    with pytest.raises(ValueError):
        assess_candidate(replay, replay, [row("2", "external")],
                         [row("2", "external")])


def test_raw_correct_flag_is_recomputed():
    replay = [row("1", "number")]
    changed = [row("1", "number", False)]
    changed[0]["raw_correct"] = True
    assert not assess_candidate(replay, changed,
                                [row("2", "external", False)],
                                [row("2", "external")])["eligible"]


def test_ties_are_not_counted_as_correct():
    replay = [row("1", "number")]
    changed = deepcopy(replay)
    changed[0]["ties"] = [changed[0]["target"], {"value": 3}]
    assert not assess_candidate(replay, changed,
                                [row("2", "external", False)],
                                [row("2", "external")])["eligible"]


def test_no_external_gain_keeps_the_parent():
    replay = [row("1", "number")]
    external = [row("2", "external")]
    assert not assess_candidate(replay, replay, external, external)["eligible"]


def test_boolean_is_not_an_integer_parameter():
    baseline = [row("1", "number")]
    baseline[0]["target"]["value"] = 1
    proposed = deepcopy(baseline)
    proposed[0]["candidate"] = {"intent": "number", "value": True}
    assert not assess_candidate(
        baseline, proposed, [row("2", "external", False)],
        [row("2", "external")])["eligible"]


def test_invalid_command_types_are_rejected():
    baseline = [row("1", "number")]
    proposed = deepcopy(baseline)
    proposed[0]["candidate"] = "73"
    with pytest.raises(ValueError):
        assess_candidate(baseline, proposed, [row("2", "external")],
                         [row("2", "external")])


def test_reference_type_cannot_change_between_runs():
    baseline = [row("1", "number")]
    baseline[0]["target"]["value"] = 1
    changed = deepcopy(baseline)
    changed[0]["target"]["value"] = True
    with pytest.raises(ValueError):
        assess_candidate(baseline, changed, [row("2", "external", False)],
                         [row("2", "external")])
