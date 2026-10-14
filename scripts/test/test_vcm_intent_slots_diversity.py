"""Metadata-only speaker diversity audits never infer identity or demographics."""
import copy

import pytest

from scripts.test.vcm_intent_slots_diversity import audit_diversity


def row(identifier, person, split="train", kind="human", metadata=None, **extra):
    return {"id": identifier, "speaker_id": person, "split": split, "kind": kind,
            "pcm_sha256": identifier, "pair_id": identifier,
            "command": {"intent": "STOP", "slots": {}},
            "publisher_metadata": {"source": "corpus", **(metadata or {})}, **extra}


def test_human_people_and_synthetic_voices_are_counted_separately():
    data = [row("a", "p"), row("b", "p"), row("s", "voice", kind="synthetic")]
    result = audit_diversity({"train": data})
    assert result["splits"]["train"]["human"]["recordings"] == 2
    assert result["splits"]["train"]["human"]["declared_speaker_ids"] == 1
    assert result["splits"]["train"]["synthetic"]["declared_voice_ids"] == 1
    assert result["total_human"]["demographics"]["gender"]["people"] == {"unknown": 1}


def test_only_supplied_gender_counts_with_person_and_recording_denominators():
    rows = [row("a", "a", metadata={"gender": "female"}),
            row("b", "a", metadata={"gender": "female"}),
            row("c", "b", metadata={"gender": "nonbinary"}), row("d", "c")]
    gender = audit_diversity({"train": rows})["total_human"]["demographics"]["gender"]
    assert gender["people"] == {"female": 1, "nonbinary": 1, "unknown": 1}
    assert gender["recordings"] == {"female": 2, "nonbinary": 1, "unknown": 1}
    assert gender["known_people"] == 2


def test_sex_accent_voice_name_and_person_id_do_not_impute_gender():
    r = row("x", "female_sounding_name", metadata={"sex": "female", "accent_group": "Filipino"})
    result = audit_diversity({"train": [r]})["total_human"]["demographics"]
    assert result["gender"]["people"] == {"unknown": 1}
    assert result["sex"]["people"] == {"female": 1}
    assert result["accent_group"]["people"] == {"Filipino": 1}


def test_conflicting_person_metadata_is_reported_without_majority_guess():
    rows = [row("a", "p", metadata={"gender": "male"}),
            row("b", "p", metadata={"gender": "female"})]
    gender = audit_diversity({"train": rows})["total_human"]["demographics"]["gender"]
    assert gender["people"] == {"conflicting": 1}
    assert gender["conflicting_people"] == 1
    assert gender["known_people"] == 0


def test_cross_split_identity_pairs_and_pcm_are_audited():
    a = row("a", "p", pair_id="session", pcm_sha256="same")
    b = row("b", "p", "test", pair_id="session", pcm_sha256="same")
    result = audit_diversity({"train": [a], "test": [b]})
    assert result["cross_split_groups"] == {"human_speaker_id": 1, "synthetic_voice_id": 0,
                                             "pair_id": 1, "pcm_sha256": 1}
    assert result["declared_split_disjoint"] is False


def test_source_qualified_ids_are_not_claimed_as_verified_distinct_people():
    rows = [row("a", "source1/7"), row("b", "source2/7")]
    result = audit_diversity({"train": rows})
    assert result["total_human"]["declared_speaker_ids"] == 2
    assert result["identity_scope"] == "declared source-qualified IDs; cross-source aliases unresolved"
    assert result["independent_human_identities_verified"] is False


def test_missing_identity_and_attributes_are_explicit():
    result = audit_diversity({"train": [row("a", "")]})["total_human"]
    assert result["declared_speaker_ids"] == 0
    assert result["unidentified_recordings"] == 1
    assert result["demographics"]["gender"]["people"] == {}
    assert result["demographics"]["gender"]["recordings"] == {"unknown": 1}


def test_partition_mismatch_duplicate_ids_and_unknown_kind_fail():
    for rows in [[row("a", "p", "test")], [row("a", "p"), row("a", "q")],
                 [row("a", "p", kind="unknown")]]:
        with pytest.raises(ValueError):
            audit_diversity({"train": rows})


def test_metadata_audit_is_pure_and_never_reads_audio():
    views = {"train": [row("a", "p", audio="/missing/never-open.wav")]}
    before = copy.deepcopy(views)
    audit_diversity(views)
    assert views == before
