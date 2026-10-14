"""V2 provenance/annotation checks; no acoustic-model inference or QA claim."""
from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest
import soundfile as sf

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
from scripts.runtime.vcm_intent_slots_contract import build_slot_contract
from scripts.runtime.vcm_intent_slots_entities import enroll_examples
from scripts.test.speaker_demo_suite import pcm_digest
from scripts.test.test_vcm_intent_slots_contract import row as human_row
from scripts.train.vcm_intent_slots_data import eligible as human_eligible, sha
from vcm.frontend import features


def mixed():
    name = "scripts.train.vcm_intent_slots_mixed_data"
    assert importlib.util.find_spec(name) is not None, "Missing versioned mixed-data support"
    return importlib.import_module(name)


@pytest.fixture
def contract():
    cases = baseline_cases() + extended_cases() + [
        {"id": "composite", "text": "set the timer for one hour one minute",
         "command": {"intent": "TIMER", "slots": {"duration_seconds": 3660}}},
        {"id": "minute", "text": "alarm six thirty a m",
         "command": {"intent": "ALARM", "slots": {"time": "06:30", "timezone": "Asia/Manila"}}},
    ]
    return build_slot_contract(cases, cases, DEFAULT_CUSTOM)


@pytest.fixture
def protocol(contract):
    return {
        "experiment_id": "vcm-mixed-final-v2",
        "source": {"repository": "hexgrad/Kokoro-82M", "revision": "a" * 40},
        "contract_hash": contract["contract_hash"], "catalog_hash": contract["catalog_hash"],
        "voices": {"train": ["af_sarah", "am_adam"], "dev": ["af_sky"],
                   "calibration": ["am_echo"], "enrollment": ["bf_alice", "bf_lily", "bm_fable"]},
        "thresholds": {"intent": .95, "slot": .90}, "final_tts_fit_allowed": False,
    }


def synthetic(contract, *, text="Stop", split="train", voice="af_sarah", identity="synth"):
    reference = next(r for r in contract["catalog_entries"]
                     if r.get("spoken_text", r["text"]).lower() == text.lower())
    return dict(id=identity, audio=identity + ".wav", kind="synthetic", role="command",
                source="hexgrad/Kokoro-82M", source_revision="a" * 40, license="Apache-2.0",
                voice_id=voice, speaker_id="Kokoro-82M/" + voice, split=split, published_split=split,
                pair_id="group/" + identity, recording_group="group/" + identity,
                sha256="b" * 64, pcm_sha256="c" * 64, text=text,
                reference_id=reference["id"], command=deepcopy(reference["command"]),
                reference_status="supported" if reference["command"] is not None else "unsupported",
                qa_eligible=False, metric_eligible=True,
                supervision_status="reference_derived_silver", acoustic_listening_verified=False)


def test_protocol_binds_preserved_contract_and_exact_partition_voices(protocol, contract):
    mixed().validate_protocol(protocol, contract)
    for path, value in (("contract_hash", "0" * 64), ("catalog_hash", "0" * 64),
                        ("thresholds", {"intent": .94, "slot": .90}),
                        ("final_tts_fit_allowed", True)):
        changed = {**protocol, path: value}
        with pytest.raises(ValueError):
            mixed().validate_protocol(changed, contract)
    changed = deepcopy(protocol)
    changed["voices"]["train"][0] = "af_heart"
    with pytest.raises(ValueError):
        mixed().validate_protocol(changed, contract)


def test_v1_still_rejects_synthetic_and_v2_preserves_silver_status(protocol, contract):
    r = synthetic(contract)
    with pytest.raises(ValueError, match="human"):
        human_eligible(r)
    mixed().eligible(r, protocol, contract)
    mixed().eligible(human_row(), protocol, contract)
    assert r["kind"] == "synthetic" and r["acoustic_listening_verified"] is False


@pytest.mark.parametrize("field,value", [
    ("voice_id", "af_heart"), ("voice_id", "af_sky"),
    ("speaker_id", "Kokoro-82M/af_heart"), ("source_revision", "d" * 40),
    ("sha256", "wrong"), ("pcm_sha256", ""), ("sha256", None),
    ("supervision_status", "human_reviewed"), ("audio_flags", ["nonfinite"]),
    ("command", {"intent": "NEXT", "slots": {}}),
    ("reviewed_spans", [{"role": "song", "encoded_start": 0, "encoded_end": 10}]),
])
def test_unbound_or_reserved_synthetic_fitting_is_rejected(protocol, contract, field, value):
    r = synthetic(contract)
    r[field] = value
    with pytest.raises(ValueError):
        mixed().eligible(r, protocol, contract)


@pytest.mark.parametrize("text,present,values", [
    ("Lights on", set(), {}),
    ("Call", set(), {}),
    ("call mom", {"contact"}, {}),
    ("play Dynamite by BTS", {"song", "artist"}, {}),
    ("set volume to twenty percent", {"volume_operation", "volume_percent"},
     {"volume_operation": "set", "volume_percent": 20}),
    ("Adjust brightness to sixty percent", {"brightness_percent"}, {"brightness_percent": 60}),
    ("Timer one minute", {"timer_amount", "timer_unit"}, {"timer_amount": 1, "timer_unit": "minutes"}),
    ("Alarm six a m", {"alarm_hour", "alarm_meridiem"}, {"alarm_hour": 6, "alarm_meridiem": "AM"}),
    ("alarm six thirty a m", {"alarm_hour", "alarm_minute", "alarm_meridiem"},
     {"alarm_hour": 6, "alarm_minute": 30, "alarm_meridiem": "AM"}),
])
def test_exact_reference_annotations_preserve_defaults_and_spoken_grammar(contract, text, present, values):
    r = synthetic(contract, text=text)
    original = deepcopy(r)
    annotated = mixed().annotate_reference(r, contract)
    assert {role for role, state in annotated["role_labels"].items() if state == "present"} == present
    assert set(annotated["role_labels"]) == set(contract["roles"])
    assert annotated["acoustic_values"] == values
    assert annotated["reviewed_spans"] == []
    assert annotated["request_validity"] == "valid"
    assert annotated["supervision_status"] == "reference_derived_silver"
    assert annotated["acoustic_listening_verified"] is False
    assert annotated["command"] == original["command"] and r == original


def test_composite_timer_masks_acoustic_values_and_retains_reference(contract):
    r = synthetic(contract, text="set the timer for one hour one minute")
    annotated = mixed().annotate_reference(r, contract)
    assert annotated["command"] == r["command"]
    assert "timer_amount" not in annotated["acoustic_values"]
    assert "timer_unit" not in annotated["acoustic_values"]
    assert "composite_timer_not_representable_as_spoken_single_pair" in annotated["supervision_limits"]
    assert "timer_amount" not in annotated["role_labels"]


def test_unknown_text_and_unreviewed_spans_remain_masked(contract):
    r = human_row()
    r.update(text="some unfamiliar paraphrase", reviewed_spans=[
        {"role": "song", "encoded_start": 0, "encoded_end": 10}])
    annotated = mixed().annotate_reference(r, contract)
    assert annotated["command"] == r["command"]
    assert annotated["role_labels"] == {} and annotated["acoustic_values"] == {}
    assert annotated["reviewed_spans"] == []
    assert "unknown_exact_reference" in annotated["supervision_limits"]


def test_exact_negative_stays_oov_without_fabricated_role_labels(contract):
    r = synthetic(contract, text="tell me a joke")
    annotated = mixed().annotate_reference(r, contract)
    assert annotated["command"] is None and annotated["request_validity"] == "unsupported"
    assert annotated["role_labels"] == {} and annotated["reviewed_spans"] == []


def test_missing_protocol_revision_rejects_as_a_gate(protocol, contract):
    protocol["source"]["revision"] = None
    with pytest.raises(ValueError, match="protocol|source|v2"):
        mixed().validate_protocol(protocol, contract)


def test_enrollment_eligibility_uses_exact_named_reference(protocol, contract):
    good = synthetic(contract, text="play Dynamite", split="enrollment", voice="bf_alice")
    good.update(entity_role="song", entity_id="song.dynamite")
    mixed().eligible(good, protocol, contract)
    with pytest.raises(ValueError, match="entity|Enrollment"):
        mixed().eligible({**good, "entity_id": "song.butter"}, protocol, contract)


def write_views(tmp_path, partitions):
    views, receipts = {}, {"views": {}}
    for split, data in partitions.items():
        p = tmp_path / (split + ".jsonl")
        p.write_text("".join(json.dumps(r) + "\n" for r in data))
        views[split] = p
        receipts["views"][split] = {"path": str(p), "sha256": sha(p)}
    return views, receipts


@pytest.mark.parametrize("field", ["pcm_sha256", "sha256", "speaker_id", "pair_id", "recording_group"])
def test_audit_rejects_reserved_overlap_before_any_acoustic_use(tmp_path, protocol, contract, field):
    train = synthetic(contract)
    test = human_row()
    test.update(id="reserved", split="test", sha256="d" * 64, pcm_sha256="e" * 64,
                speaker_id="human/reserved", pair_id="pair/reserved", recording_group="group/reserved")
    test[field] = train[field]
    views, receipts = write_views(tmp_path, {"train": [train], "test": [test]})
    with pytest.raises(ValueError, match="leak|partition"):
        mixed().audit_views(views, receipts, tmp_path / "audit", protocol=protocol, contract=contract)


def test_audit_reports_silver_and_missing_acoustic_review(tmp_path, protocol, contract):
    r = mixed().annotate_reference(synthetic(contract), contract)
    views, receipts = write_views(tmp_path, {"train": [r]})
    report = mixed().audit_views(views, receipts, tmp_path / "audit", protocol=protocol, contract=contract)
    assert report["splits"]["train"]["synthetic"] == 1
    assert report["splits"]["train"]["human"] == 0
    assert report["reviewed_acoustic_spans"] == 0
    assert report["acoustic_listening_verified"] == 0
    assert report["accuracy_qualified"] is False and report["fit_ready"] is False


def audio_row(tmp_path, contract, *, split, voice, identity, frequency):
    r = synthetic(contract, split=split, voice=voice, identity=identity)
    path = tmp_path / r["audio"]
    sf.write(path, .2 * np.sin(np.arange(3200) * frequency), 16000, subtype="PCM_16")
    pcm, _ = sf.read(path, dtype="float32")
    r.update(sha256=sha(path), pcm_sha256=pcm_digest(pcm))
    return r, pcm


def test_features_verify_pcm_and_fit_normalization_on_train_only(tmp_path, protocol, contract):
    train, pcm = audio_row(tmp_path, contract, split="train", voice="af_sarah", identity="train", frequency=.1)
    dev, _ = audio_row(tmp_path, contract, split="dev", voice="af_sky", identity="dev", frequency=.3)
    receipt, norm = mixed().prepare_features([train, dev], tmp_path, tmp_path / "features",
                                            protocol=protocol, contract=contract)
    x = features(pcm)
    np.testing.assert_allclose(norm["mean"], x.mean(axis=1, dtype=np.float64), atol=1e-10)
    assert norm["frames"] == x.shape[1] and norm["fit_split"] == "train"
    assert set(receipt) == {"train", "dev"}
    wrong = {**train, "pcm_sha256": "0" * 64}
    with pytest.raises(ValueError, match="PCM"):
        mixed().prepare_features([wrong], tmp_path, tmp_path / "wrong",
                                 protocol=protocol, contract=contract)
    cal, _ = audio_row(tmp_path, contract, split="calibration", voice="am_echo", identity="cal", frequency=.5)
    with pytest.raises(ValueError, match="Calibration|fitting"):
        mixed().prepare_features([cal], tmp_path, tmp_path / "cal-features",
                                 protocol=protocol, contract=contract)


def test_synthetic_enrollment_needs_explicit_v2_revision_voice_policy(contract):
    ex = synthetic(contract, text="play Dynamite", split="enrollment", voice="bf_alice")
    ex.update(role="song", entity_role="song", entity_id="song.dynamite", embedding=np.ones(64).tolist())
    binding = {"encoder": "1" * 64, "frontend": "2" * 64, "normalization": "3" * 64}
    policy = {"study": "vcm-mixed-final-v2", "source": "hexgrad/Kokoro-82M",
              "source_revision": "a" * 40, "voice_ids": ["bf_alice", "bf_lily", "bm_fable"]}
    with pytest.raises(ValueError, match="human"):
        enroll_examples([ex], binding, contract)
    memory = enroll_examples([ex], binding, contract, provenance_policy=policy)
    assert memory["examples"][0]["kind"] == "synthetic"
    assert memory["provenance_policy"] == policy
    for field, value in (("voice_id", "af_heart"), ("source_revision", "d" * 40), ("split", "test")):
        bad = {**ex, field: value}
        with pytest.raises(ValueError):
            enroll_examples([bad], binding, contract, provenance_policy=policy)
