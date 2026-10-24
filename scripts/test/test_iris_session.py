"""Software policy fixtures are not human acoustic evidence."""
import importlib
import importlib.util
import numpy as np
import pytest

def load(name):
    assert (
        importlib.util.find_spec(name) is not None
    ), f"Missing feature: {name}"
    return importlib.import_module(name)



@pytest.fixture
def owner():
    return load("scripts.train.iris_session")



@pytest.fixture
def learning():
    return load("scripts.models.speaker_demo")



def evidence(module, vector=(1.0, 0.0), **changes):
    fields = dict(
        embedding=vector, model_id="a" * 64, voiced_seconds=1.2,
        overlap=False, clipped=False,
    )
    fields.update(changes)
    return module.SpeakerEvidence(**fields)



def session(module, **changes):
    fields = dict(threshold=0.8, require_windows=False)
    fields.update(changes)
    return module.SpeakerSession(module.SpeakerPolicy(**fields))



def test_same_speaker_consumes_session(owner):
    lock = session(owner)
    assert lock.arm("s1", evidence(owner), now=0).accepted
    assert lock.finish("s1", evidence(owner), now=1).accepted
    assert lock.finish("s1", evidence(owner), now=2).reason == "no_owner"



def test_wrong_speaker_cannot_issue_command(owner):
    lock = session(owner)
    lock.arm("s1", evidence(owner), now=0)
    result = lock.finish("s1", evidence(owner, (0.0, 1.0)), now=1)
    assert not result.accepted
    assert result.reason == "speaker_mismatch"



def test_cannot_replace_active_owner(owner):
    lock = session(owner)
    lock.arm("s1", evidence(owner), now=0)
    assert (
        lock.arm(
            "s2", evidence(owner, (0.0, 1.0)), now=1
        ).reason == "owner_busy"
    )
    assert lock.finish("s1", evidence(owner), now=2).accepted



@pytest.mark.parametrize("change,reason", [
    ({"embedding": (float("nan"), 0.0)}, "invalid_embedding"),
    ({"embedding": (0.0, 0.0)}, "invalid_embedding"),
    ({"overlap": True}, "overlap"),
    ({"overlap": None}, "overlap_unchecked"),
    ({"clipped": True}, "clipped_audio"),
    ({"voiced_seconds": 0.1}, "insufficient_speech"),
    ({"model_id": "unknown"}, "invalid_model_identity"),
])
def test_bad_reference_rejected(owner, change, reason):
    assert (
        session(owner).arm(
            "s1", evidence(owner, **change), now=0
        ).reason == reason
    )



@pytest.mark.parametrize("case,reason", [
    ("expired", "owner_expired"),
    ("foreign_session", "session_mismatch"),
    ("clock_reversal", "invalid_clock"),
    ("model_change", "model_mismatch"),
    ("shape_change", "embedding_shape"),
])
def test_boundaries_fail_closed(owner, case, reason):
    lock = session(owner)
    lock.arm("s1", evidence(owner), now=2)
    now = {"expired": 20, "clock_reversal": 1}.get(case, 3)
    token = "s2" if case == "foreign_session" else "s1"
    changes = {}
    if case == "model_change":
        changes["model_id"] = "b" * 64
    if case == "shape_change":
        changes["embedding"] = (1.0, 0.0, 0.0)
    assert (
        lock.finish(
            token, evidence(owner, **changes), now=now
        ).reason == reason
    )
    assert lock.finish("s1", evidence(owner), now=4).reason == "no_owner"



def test_windows_required_and_handover_rejected(owner):
    lock = session(owner, require_windows=True)
    lock.arm("s1", evidence(owner), now=0)
    assert (
        lock.finish(
            "s1", evidence(owner), now=1
        ).reason == "continuity_unchecked"
    )
    lock.arm("s2", evidence(owner), now=2)
    result = lock.finish(
        "s2", evidence(owner), now=3,
        windows=[evidence(owner), evidence(owner, (0.0, 1.0))],
    )
    assert result.reason == "speaker_mismatch"



def test_reference_copied_and_never_adapted(owner):
    vector = np.array([1.0, 0.0])
    lock = session(owner)
    lock.arm("s1", evidence(owner, vector), now=0)
    vector[:] = [0.0, 1.0]
    assert lock.finish("s1", evidence(owner), now=1).accepted



def test_speaker_gate_never_rescues_vcm(owner):
    decision = {
        "accepted": False, "command": None, "reason": "intent_confidence"
    }
    verdict = owner.SpeakerVerdict(True, None, "s1", 1.0)
    assert owner.gate_application(decision, verdict) == decision
    command = {"intent": "STOP", "slots": {}}
    accepted = {"accepted": True, "command": command, "reason": None}
    rejected = owner.gate_application(
        accepted, owner.SpeakerVerdict(False, "speaker_mismatch", "s1", 0.0)
    )
    assert rejected["command"] is None and not rejected["accepted"]
    assert accepted["command"] == command



def test_class_speaker_weights_balance(learning):
    rows = [
        {"label": 1, "speaker_id": "a"},
        {"label": 1, "speaker_id": "a"},
        {"label": 1, "speaker_id": "b"},
        {"label": 2, "speaker_id": "c"},
    ]
    weights = learning.class_speaker_weights(rows)
    assert np.isclose(weights[:3].sum(), weights[3])
    assert np.isclose(weights[:2].sum(), weights[2])



def test_contrastive_loss_zero_without_cross_speaker_pairs(learning):
    import torch
    values = torch.randn(4, 8, requires_grad=True)
    loss = learning.cross_speaker_contrastive(
        values, torch.tensor([1, 1, 0, 0]), torch.tensor([2, 2, 3, 4])
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(values.grad).all()



def test_contrastive_loss_gradients_and_positive_closeness(learning):
    import torch
    good = torch.tensor(
        [[1., 0.], [1., .1], [0., 1.], [.1, 1.]], requires_grad=True
    )
    labels, speakers = torch.tensor([1, 1, 2, 2]), torch.tensor([1, 2, 1, 2])
    good_loss = learning.cross_speaker_contrastive(good, labels, speakers)
    bad_loss = learning.cross_speaker_contrastive(
        good[[0, 2, 1, 3]], labels, speakers
    )
    assert good_loss < bad_loss
    good_loss.backward()
    assert torch.isfinite(good.grad).all()
