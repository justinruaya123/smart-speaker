"""Numeric/provenance calibration fixtures; never model or audio inference."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
from scripts.runtime.vcm_intent_slots_contract import build_slot_contract, digest
from scripts.runtime.vcm_intent_slots_decode import calibrated_slot
from scripts.train.vcm_intent_slots_calibration import fit_score_calibration
from scripts.train.vcm_intent_slots_data import sha


def mixed():
    name = "scripts.train.vcm_intent_slots_mixed_calibration"
    assert importlib.util.find_spec(name) is not None, "Missing versioned mixed calibration"
    return importlib.import_module(name)


def write_manifest(path, observations):
    path.write_text("".join(json.dumps(o["source_row"]) + "\n" for o in observations))


@pytest.fixture
def inputs(tmp_path):
    cases = baseline_cases() + extended_cases()
    contract = build_slot_contract(cases, cases, DEFAULT_CUSTOM)
    protocol = {"experiment_id": "vcm-mixed-final-v2",
                "source": {"repository": "hexgrad/Kokoro-82M", "revision": "a" * 40},
                "contract_hash": contract["contract_hash"], "catalog_hash": contract["catalog_hash"],
                "voices": {"train": ["af_sarah", "am_adam"], "dev": ["af_sky"],
                           "calibration": ["am_echo"], "enrollment": ["bf_alice", "bf_lily", "bm_fable"]},
                "thresholds": {"intent": .95, "slot": .90}, "final_tts_fit_allowed": False}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    observations = []
    for label, intent in enumerate(contract["intent_order"]):
        reference = next(r for r in cases if ((r["command"] or {}).get("intent", "OOV")) == intent)
        for kind in ("human", "synthetic"):
            identifier = "software-calibration-fixture/" + kind + "/" + str(label)
            row = {"id": identifier, "audio": identifier + ".wav", "kind": kind,
                   "role": "command", "split": "calibration", "published_split": "calibration",
                   "source": "software-human-fixture" if kind == "human" else "hexgrad/Kokoro-82M",
                   "source_revision": "b" * 40 if kind == "human" else "a" * 40, "license": "research",
                   "speaker_id": "software-human/" + str(label) if kind == "human" else "Kokoro-82M/am_echo",
                   "pair_id": identifier, "recording_group": identifier,
                   "sha256": hashlib.sha256((identifier + "/file").encode()).hexdigest(),
                   "pcm_sha256": hashlib.sha256((identifier + "/pcm").encode()).hexdigest(),
                   "text": reference.get("spoken_text", reference["text"]), "reference_id": reference["id"],
                   "command": deepcopy(reference["command"]),
                   "reference_status": "supported" if reference["command"] else "unsupported",
                   "qa_eligible": kind == "human", "metric_eligible": True,
                   "supervision_status": "reference_derived_silver", "acoustic_listening_verified": False}
            if kind == "synthetic":
                row["voice_id"] = "am_echo"
            observation = {key: row[key] for key in ("id", "audio", "split", "kind", "speaker_id", "sha256", "pcm_sha256")}
            observation.update(source_row=deepcopy(row), intent_logits=[0.] * 19, intent_label=label, slots={})
            if intent == "BRIGHTNESS":
                observation["slots"] = {"percent": {"scores": [0. if kind == "human" else 1.],
                    "provenance": "acoustic", "correct": int(kind == "synthetic")}}
            if intent == "LIGHT_ON":
                observation["slots"] = {"light_id": {"scores": [.9], "provenance": "qualified_default", "correct": 1}}
            observations.append(observation)
    manifest = tmp_path / "calibration.jsonl"
    write_manifest(manifest, observations)
    binding = {"protocol": sha(protocol_path), "manifest": sha(manifest),
               "contract_hash": contract["contract_hash"], "checkpoint": "c" * 64,
               "normalization": "d" * 64, "frontend": "e" * 64}
    return observations, binding, protocol_path, contract, manifest


def fit(inputs, output):
    observations, binding, protocol, contract, manifest = inputs
    return mixed().fit_mixed_score_calibration(observations, binding, output,
                                             protocol=protocol, contract=contract, manifest=manifest)


def test_mixed_counts_do_not_relabel_voices_as_human_people(inputs, tmp_path):
    before = deepcopy(inputs[:2])
    policy = fit(inputs, tmp_path / "scores")
    assert policy["calibration_counts"] == {"human": 19, "synthetic": 19, "reference_derived_silver": 38}
    assert policy["calibration_voices"] == ["am_echo"]
    assert policy["calibration_people"] == sorted("software-human/" + str(i) for i in range(19))
    assert policy["thresholds"] == {"intent": .95, "slot": .90}
    assert policy["status"] == "frozen_mixed_partial_calibration"
    assert policy["qualification"] is False
    assert policy["calibration_id"] == digest({k: v for k, v in policy.items() if k != "calibration_id"})
    assert json.loads((tmp_path / "scores/calibration.json").read_text()) == policy
    assert inputs[:2] == before
    with pytest.raises(FileExistsError):
        fit(inputs, tmp_path / "scores")


def test_numeric_fit_keeps_original_algorithms_and_missing_outcome_groups(inputs, tmp_path):
    policy = fit(inputs, tmp_path / "scores")
    assert policy["intent_temperature"] == pytest.approx(1.0)
    calibrator = policy["slot_calibrators"]["percent/acoustic/1"]
    assert calibrator["weights"] == pytest.approx([7.072912847382023], abs=1e-6)
    assert calibrator["bias"] == pytest.approx(-3.5363177484934227, abs=1e-6)
    assert calibrator["brier"] == pytest.approx(.0008004681636770833, abs=1e-9)
    assert calibrator["n"] == 2 and calibrator["positive_n"] == 1
    assert policy["missing_slot_calibration"] == ["light_id/qualified_default/1"]
    assert "light_id/qualified_default/1" not in policy["slot_calibrators"]
    with pytest.raises(ValueError, match="Missing frozen slot"):
        calibrated_slot([.99], "light_id", policy, "qualified_default")


def test_v1_numeric_reference_and_human_only_guard_remain_unchanged(tmp_path):
    observations = [{"split": "calibration", "kind": "human", "speaker_id": "fixture/" + str(i),
                     "intent_label": i, "intent_logits": [0.] * 19, "slots": {}} for i in range(19)]
    observations[0]["slots"] = {"percent": {"scores": [0.], "provenance": "acoustic", "correct": 0}}
    observations[16]["slots"] = {"percent": {"scores": [1.], "provenance": "acoustic", "correct": 1}}
    binding = dict(contract_hash="c" * 64, checkpoint="a" * 64, normalization="b" * 64,
                   manifest="d" * 64, frontend="e" * 64)
    policy = fit_score_calibration(observations, binding, tmp_path / "v1")
    assert policy["intent_temperature"] == pytest.approx(1.0)
    assert policy["slot_calibrators"]["percent/acoustic/1"]["weights"] == pytest.approx([7.072912847382023], abs=1e-6)
    assert policy["status"] == "frozen_scores_unqualified"
    assert "calibration_voices" not in policy
    with pytest.raises(ValueError, match="human"):
        fit_score_calibration([{**observations[0], "kind": "synthetic"}], binding, tmp_path / "v1-bad")
    with pytest.raises(ValueError, match="classes"):
        fit_score_calibration(observations[1:], binding, tmp_path / "v1-missing")


@pytest.mark.parametrize("field,value", [
    ("id", "another"), ("kind", "synthetic"), ("split", "dev"), ("speaker_id", "other-person"),
    ("sha256", "f" * 64), ("pcm_sha256", "f" * 64), ("audio", "other.wav"),
])
def test_observation_identity_cannot_differ_from_source_row(inputs, tmp_path, field, value):
    inputs[0][0][field] = value
    with pytest.raises(ValueError, match="identity|source|split|partition|coverage"):
        fit(inputs, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_source_row_must_equal_frozen_manifest_reference(inputs, tmp_path):
    inputs[0][0]["source_row"]["command"] = {"intent": "STOP", "slots": {}}
    with pytest.raises(ValueError, match="manifest|reference"):
        fit(inputs, tmp_path / "bad")


def test_observed_intent_truth_cannot_rewrite_source_reference(inputs, tmp_path):
    inputs[0][0]["intent_label"] = 1
    with pytest.raises(ValueError, match="intent|reference"):
        fit(inputs, tmp_path / "bad")


@pytest.mark.parametrize("field", ["protocol", "manifest", "contract_hash", "checkpoint", "normalization", "frontend"])
def test_calibration_requires_every_binding(inputs, tmp_path, field):
    inputs[1][field] = None
    with pytest.raises(ValueError, match="binding|hash|protocol|manifest"):
        fit(inputs, tmp_path / "bad")


@pytest.mark.parametrize("field", ["protocol", "manifest"])
def test_stale_frozen_file_hash_is_rejected(inputs, tmp_path, field):
    inputs[1][field] = "f" * 64
    with pytest.raises(ValueError, match="hash|binding"):
        fit(inputs, tmp_path / "bad")


def test_final_evaluation_voice_never_becomes_eligible_calibration(inputs, tmp_path):
    observation = inputs[0][1]
    observation["source_row"].update(voice_id="af_heart", speaker_id="Kokoro-82M/af_heart")
    observation["speaker_id"] = "Kokoro-82M/af_heart"
    write_manifest(inputs[4], inputs[0])
    inputs[1]["manifest"] = sha(inputs[4])
    with pytest.raises(ValueError, match="eligible|synthetic|voice"):
        fit(inputs, tmp_path / "bad")


def test_synthetic_lineage_cannot_be_redeclared_human(inputs, tmp_path):
    observation = inputs[0][1]
    observation["source_row"].update(kind="human", qa_eligible=True)
    observation["kind"] = "human"
    write_manifest(inputs[4], inputs[0])
    inputs[1]["manifest"] = sha(inputs[4])
    with pytest.raises(ValueError, match="lineage|synthetic|human"):
        fit(inputs, tmp_path / "bad")


@pytest.mark.parametrize("damage", ["missing", "duplicate"])
def test_calibration_cannot_cherry_pick_or_duplicate_manifest_rows(inputs, tmp_path, damage):
    if damage == "missing":
        inputs[0].pop()
    else:
        inputs[0].append(deepcopy(inputs[0][0]))
    with pytest.raises(ValueError, match="coverage|duplicate|manifest"):
        fit(inputs, tmp_path / "bad")


def test_missing_intent_class_cannot_be_qualified(inputs, tmp_path):
    inputs[0][:] = [o for o in inputs[0] if o["intent_label"] != 18]
    write_manifest(inputs[4], inputs[0])
    inputs[1]["manifest"] = sha(inputs[4])
    with pytest.raises(ValueError, match="classes|19|intent"):
        fit(inputs, tmp_path / "bad")


@pytest.mark.parametrize("value", [[0.] * 18, [float("nan")] * 19, [float("inf")] * 19])
def test_corrupt_logits_never_fit_a_temperature(inputs, tmp_path, value):
    inputs[0][0]["intent_logits"] = value
    with pytest.raises(ValueError, match="logit|finite|shape"):
        fit(inputs, tmp_path / "bad")


@pytest.mark.parametrize("value", [[], [float("nan")], [1.1]])
def test_corrupt_slot_scores_never_fit_a_probability(inputs, tmp_path, value):
    observation = next(o for o in inputs[0] if "percent" in o["slots"])
    observation["slots"]["percent"]["scores"] = value
    with pytest.raises(ValueError, match="slot|score|finite"):
        fit(inputs, tmp_path / "bad")


def test_positive_slot_truth_cannot_be_invented_for_an_absent_reference_field(inputs, tmp_path):
    inputs[0][0]["slots"] = {"percent": {"scores": [.99], "provenance": "acoustic", "correct": 1}}
    with pytest.raises(ValueError, match="slot|reference"):
        fit(inputs, tmp_path / "bad")


def test_preserved_protocol_cannot_lower_thresholds_even_with_a_new_file_hash(inputs, tmp_path):
    data = json.loads(inputs[2].read_text())
    data["thresholds"]["slot"] = .89
    inputs[2].write_text(json.dumps(data) + "\n")
    inputs[1]["protocol"] = sha(inputs[2])
    with pytest.raises(ValueError, match="threshold|protocol|v2"):
        fit(inputs, tmp_path / "bad")
