import json
from pathlib import Path
import numpy as np
import pytest
import torch

from scripts.test.test_vcm_intent_slots_contract import contract, row
from scripts.models.vcm_intent_slots import IntentSlotsVCM, joint_loss
from scripts.train.vcm_intent_slots_train import sample_indices, make_targets, validate_training_protocol
from scripts.test.vcm_intent_slots_eval import summarize_trials, freeze_selection, verify_selection, verify_evaluation_context
from scripts.train.vcm_intent_slots_calibration import fit_score_calibration


def test_balanced_complete_command_then_person_and_fixed_budget():
    data = []
    for name, intent, count in [("a", "STOP", 100), ("b", "STOP", 2), ("c", "NEXT", 2), ("d", None, 2)]:
        for n in range(count):
            r = row()
            r.update(id=f"{name}/{n}", speaker_id=name,
                     command={"intent": intent, "slots": {}} if intent else None)
            data.append(r)
    one = sample_indices(data, "balanced", 53, 19200)
    assert np.array_equal(one, sample_indices(data, "balanced", 53, 19200))
    masses = {name:sum(data[i]["speaker_id"] == name for i in one)/len(one) for name in "abcd"}
    assert .14 < masses["a"] < .19 and .14 < masses["b"] < .19
    assert .3 < masses["c"] < .36 and .3 < masses["d"] < .36
    assert len(sample_indices(data, "uniform", 53, 19200)) == len(one)


def test_missing_role_supervision_masked_and_numeric_fixture_optimization():
    torch.manual_seed(53)
    c = contract()
    r = row()
    targets = make_targets([r, r], c)
    assert torch.all(targets["role_states"] == -100)
    assert targets["spans"] == []
    model = IntentSlotsVCM(c)
    x = torch.randn(2, 40, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        loss, counts = joint_loss(model(x, torch.tensor([7, 7])), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert counts["intents"] == 2


def test_training_protocol_no_unapproved_changes():
    protocol = json.loads(Path("experiments/vcm-intent-slots-v1/protocol.json").read_text())
    validate_training_protocol(protocol, "uniform", 53)
    for field, value in [("update_artifacts", True), ("human_only", False), ("epochs", 1)]:
        other = json.loads(json.dumps(protocol))
        other["training"][field] = value
        with pytest.raises(ValueError):
            validate_training_protocol(other, "uniform", 53)


def test_trial_denominators_faults_and_track_separation():
    good = {"intent": "STOP", "slots": {}}
    trials = [{"id": "p", "kind": "human", "tier": "baseline", "speaker_id": "a", "command": good},
              {"id": "n", "kind": "human", "tier": "extension", "speaker_id": "a", "command": None},
              {"id": "s", "kind": "synthetic", "tier": "baseline", "speaker_id": "clone", "command": good}]
    predictions = [{"id": "p", "application": {"accepted": False, "command": None}},
                   {"id": "n", "fault": "capture_fault"}]
    m = summarize_trials(trials, predictions)
    assert m["all"]["supported_n"] == 2 and m["all"]["correct"] == 0
    assert m["all"]["faults"] == 1 and m["all"]["missing"] == 1
    assert m["all"]["correct_rejections"] == 0
    assert m["tracks"]["human/baseline"]["supported_n"] == 1
    assert m["tracks"]["synthetic/baseline"]["supported_n"] == 1


def test_selection_lock_complete_before_fresh_test_and_stale_files(tmp_path):
    p = tmp_path/"source"
    p.write_text("binding")
    with pytest.raises(ValueError):
        freeze_selection({"checkpoint": p}, tmp_path/"incomplete.json")
    names = ("protocol", "manifest", "checkpoint", "contract", "catalog", "normalization",
             "calibration", "frontend", "runtime")
    lock = tmp_path/"selection-before-test.json"
    freeze_selection({k:p for k in names}, lock)
    verify_selection(lock)
    p.write_text("changed")
    with pytest.raises(ValueError):
        verify_selection(lock)


def test_calibration_is_partition_separate_and_requires_both_outcomes(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        fit_score_calibration([{"split": "dev", "kind": "human"}], {}, tmp_path/"bad")
    with pytest.raises(ValueError):
        fit_score_calibration([{"split": "calibration", "kind": "synthetic"}], {}, tmp_path/"synthetic")


def test_fault_and_missing_keep_per_slot_denominators():
    cmd = {"intent": "VOLUME", "slots": {"operation": "set", "percent": 20}}
    trials = [{"id": str(n), "kind": "human", "speaker_id": "a", "command": cmd} for n in range(2)]
    m = summarize_trials(trials, [{"id": "0", "fault": "capture_fault"}])
    assert m["all"]["slot_counts"]["percent"]["n"] == 2
    assert m["all"]["slot_counts"]["percent"]["correct_accepted"] == 0


def test_unrelated_lock_cannot_authorize_fresh_runtime(tmp_path):
    class Runtime:
        root = tmp_path
        policy = {"require_calibration": False}
    with pytest.raises(ValueError, match="calibration|runtime"):
        verify_evaluation_context({"files": {}}, Runtime(), tmp_path/"manifest")
