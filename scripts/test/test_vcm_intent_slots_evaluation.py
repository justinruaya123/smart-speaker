"""Reporting must preserve people, populations, rejects and fault denominators."""
import copy

import numpy as np
import pytest

from scripts.test.vcm_intent_slots_eval import cluster_interval, summarize_trials


def trial(identifier, person, *, kind="human", command=None, **extra):
    return {"id": identifier, "speaker_id": person, "kind": kind,
            "command": command or {"intent": "STOP", "slots": {}}, **extra}


def prediction(identifier, accepted=True, command=None):
    return {"id": identifier, "final_application": {
        "accepted": accepted, "command": (command or {"intent": "STOP", "slots": {}})
        if accepted else None}}


def test_synthetic_voice_success_cannot_improve_human_person_results():
    trials = [trial("h", "shared"), trial("s", "shared", kind="synthetic")]
    result = summarize_trials(trials, [prediction("h", False), prediction("s")])
    assert result["macro_person_recall"] == 0
    assert result["worst_person_recall"] == 0
    assert result["kinds"]["human"]["recall"] == 0
    assert result["kinds"]["synthetic"]["recall"] == 1
    assert result["person_summary"]["synthetic"]["declared_ids"] == 1
    assert cluster_interval(trials, [prediction("h", False), prediction("s")]) == [0, 0]


def test_typed_output_tiers_and_negative_fault_are_separate():
    extension = {"intent": "PLAY_MUSIC", "slots": {"song_id": "song.dynamite"}}
    trials = [trial("b", "a"), trial("e", "a", command=extension),
              dict(trial("n", "a"), command=None)]
    result = summarize_trials(trials, [{"id": "n", "fault": "capture_fault"}])
    assert result["tracks"]["human/optionb"]["supported_n"] == 1
    assert result["tracks"]["human/extension"]["supported_n"] == 1
    assert result["tracks"]["human/negative"]["faults"] == 1
    assert result["all"]["correct_rejections"] == 0
    assert result["all"]["missing"] == 2


def test_duplicate_trial_ids_cannot_inflate_denominators():
    with pytest.raises(ValueError, match="Duplicate trial"):
        summarize_trials([trial("x", "a"), trial("x", "b")], [])
    with pytest.raises(ValueError, match="Duplicate trial"):
        cluster_interval([trial("x", "a"), trial("x", "b")], [])


def test_cluster_interval_matches_explicit_human_cluster_resampling():
    trials = [trial("a0", "a"), trial("a1", "a"), trial("b0", "b")]
    predictions = [prediction("a0"), prediction("a1", False), prediction("b0")]
    rng = np.random.default_rng(71)
    scores = []
    for _ in range(100):
        drawn = rng.choice(["a", "b"], 2)
        correct = sum(1 for _ in drawn)
        supported = sum(2 if person == "a" else 1 for person in drawn)
        scores.append(correct / supported)
    assert cluster_interval(trials, predictions, seed=71, repetitions=100) == pytest.approx(
        np.quantile(scores, [.025, .975]).tolist())
    assert cluster_interval([trial("s", "voice", kind="synthetic")], []) is None


def test_unknown_people_are_not_counted_as_one_measured_person():
    result = summarize_trials([trial("x", "unknown"), trial("y", "")], [])
    assert result["all"]["supported_n"] == 2
    assert result["person_summary"]["human"]["declared_ids"] == 0
    assert result["person_summary"]["human"]["unidentified_trials"] == 2
    assert result["macro_person_recall"] is None
    assert cluster_interval([trial("x", "unknown")], []) is None


def test_reporting_does_not_mutate_labels_or_predictions():
    trials = [trial("x", "a")]
    predictions = [prediction("x")]
    before = copy.deepcopy((trials, predictions))
    summarize_trials(trials, predictions)
    cluster_interval(trials, predictions)
    assert (trials, predictions) == before


def test_host_timing_excludes_faults_and_invalid_measurements():
    trials = [trial(str(i), "a", samples=16000, sample_rate=16000) for i in range(4)]
    predictions = [dict(prediction("0"), timing={"total_ms": 10}),
                   dict(prediction("1", False), timing={"total_ms": 30}),
                   {"id": "2", "fault": "capture_fault", "timing": {"total_ms": 1000}},
                   dict(prediction("3"), timing={"total_ms": float("nan")})]
    timing = summarize_trials(trials, predictions)["host_timing"]
    assert timing["measured_n"] == 2
    assert timing["p50_ms"] == 20
    assert timing["p95_ms"] == pytest.approx(29)
    assert timing["p50_rtf"] == pytest.approx(.02)
    assert timing["physical_pi_evidence"] is False
