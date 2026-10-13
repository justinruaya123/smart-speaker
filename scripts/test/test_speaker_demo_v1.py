"""Numerical/software tests; no fixture here is human speaker evidence."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name):
    assert (
        importlib.util.find_spec(name) is not None
    ), f"Missing feature: {name}"
    return importlib.import_module(name)


@pytest.fixture
def owner():
    return load("scripts.runtime.speaker_session")


@pytest.fixture
def suite():
    return load("scripts.test.speaker_demo_suite")


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


def test_model_padding_invariance_and_gradients(learning):
    import torch
    torch.manual_seed(11)
    model = learning.DemoKeywordNet(classes=5, width=24).eval()
    features = torch.randn(1, 40, 37)
    original, _ = model(features, torch.tensor([37]))
    padded = torch.nn.functional.pad(features, (0, 23), value=17)
    result, embedding = model(padded, torch.tensor([37]))
    torch.testing.assert_close(original, result, atol=2e-6, rtol=2e-5)
    assert embedding.shape == (1, 24)
    result.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def rows():
    result = []
    for split in ("train", "dev", "calibration", "test"):
        for speaker in ("a", "b"):
            for role in ("wake", "command"):
                identity = f"{split}-{speaker}-{role}"
                result.append({
                    "id": identity, "audio": identity + ".wav",
                    "sha256": hashlib.sha256(identity.encode()).hexdigest(),
                    "pcm_sha256": hashlib.sha256(
                        (identity + "pcm").encode()
                    ).hexdigest(),
                    "speaker_id": f"{split}-{speaker}", "source": "fixture",
                    "license": "test-only", "kind": "human", "split": split,
                    "role": role,
                    "command": {
                        "intent": "STOP", "slots": {}
                    } if role == "command" else None,
                })
    return result


def test_manifest_audit_and_cross_split_leakage(suite):
    assert suite.audit(rows())["recordings"] == 16
    changed = rows()
    changed[-1]["speaker_id"] = changed[0]["speaker_id"]
    with pytest.raises(ValueError, match="speaker"):
        suite.audit(changed)
    changed = rows()
    changed[-1]["pcm_sha256"] = changed[0]["pcm_sha256"]
    with pytest.raises(ValueError, match="PCM"):
        suite.audit(changed)


def test_duplicate_identity_and_pcm_conflicts(suite):
    with pytest.raises(ValueError, match="id"):
        suite.audit(rows() + [rows()[0]])
    changed = rows()
    changed[1]["pcm_sha256"] = changed[0]["pcm_sha256"]
    with pytest.raises(ValueError, match="PCM"):
        suite.audit(changed)


def test_pair_plan_is_deterministic_and_split_local(suite):
    first = suite.plan(rows(), split="test", seed=31, impostors=5)
    assert first == suite.plan(rows(), split="test", seed=31, impostors=5)
    lookup = {r["id"]: r for r in rows()}
    assert len(first) == 6
    for trial in first:
        assert lookup[trial["command_id"]]["split"] == "test"
        if trial["case"] == "wrong_speaker":
            assert (
                lookup[trial["wake_id"]]["speaker_id"] != trial["speaker_id"]
            )


def test_missing_wakes_are_not_silently_skipped(suite):
    data = [r for r in rows() if r["role"] == "command"]
    with pytest.raises(ValueError, match="wake"):
        suite.plan(data, split="test", seed=1)
    assert len(suite.plan(data, split="test", seed=1, command_only=True)) == 2


def test_missing_prediction_stays_in_denominator(suite):
    trials = suite.plan(rows(), split="test", seed=1, command_only=True)
    result = suite.summarize(trials, [])
    assert result["overall"]["positive_n"] == 2
    assert result["overall"]["correct_accepted"] == 0
    assert result["overall"]["missing_predictions"] == 2
    assert result["overall"]["recall"] == 0


def test_wrong_speaker_high_confidence_is_error(suite):
    trials = suite.plan(rows(), split="test", seed=1)
    predictions = [
        {"trial_id": row["trial_id"], "application": {
            "accepted": True, "command": {"intent": "STOP", "slots": {}},
            "reason": None,
        }} for row in trials
    ]
    metrics = suite.summarize(trials, predictions)
    assert metrics["by_case"]["wrong_speaker"]["false_accept_rate"] == 1
    assert metrics["by_case"]["same_speaker"]["recall"] == 1


def test_duplicate_or_foreign_predictions_are_errors(suite):
    trials = suite.plan(rows(), split="test", seed=1, command_only=True)
    prediction = {"trial_id": trials[0]["trial_id"], "application": {
        "accepted": False, "command": None, "reason": "OOV"}}
    with pytest.raises(ValueError, match="duplicate"):
        suite.summarize(trials, [prediction, prediction])
    with pytest.raises(ValueError, match="unknown"):
        suite.summarize(trials, [{**prediction, "trial_id": "foreign"}])


def test_nonfinite_json_rejected(suite, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"value": NaN}\n')
    with pytest.raises(ValueError):
        suite.read_jsonl(path)


def test_audio_hash_audit(suite, tmp_path):
    import hashlib
    import soundfile as sf
    data = rows()[:1]
    path = tmp_path / "tone.wav"
    sf.write(
        path, np.sin(np.arange(1600) * 0.13) * 0.2, 16000, subtype="PCM_16"
    )
    pcm, rate = sf.read(path, dtype="float32")
    data[0].update(
        audio=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        pcm_sha256=suite.pcm_digest(pcm, rate),
    )
    assert suite.audit(data, check_audio=True)["audio_verified"]
    data[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        suite.audit(data, check_audio=True)


def test_feature_modules_exist():
    for name in ("scripts.runtime.speaker_session",
    "scripts.models.speaker_demo",
                 "scripts.test.speaker_demo_suite"):
        load(name)


def test_tampered_expectation_or_owner_is_rejected(suite):
    trial_rows = suite.plan(rows(), split="test", seed=31)
    changed = [dict(row) for row in trial_rows]
    wrong = next(row for row in changed if row["case"] == "wrong_speaker")
    wrong["expected"] = {"intent": "STOP", "slots": {}}
    with pytest.raises(ValueError, match="expected"):
        suite.validate_trials(rows(), changed)
    changed = [dict(row) for row in trial_rows]
    changed[0]["speaker_id"] = "foreign"
    with pytest.raises(ValueError, match="metadata"):
        suite.validate_trials(rows(), changed)


def test_deleted_command_coverage_is_rejected(suite):
    trials = suite.plan(rows(), split="test", seed=31, command_only=True)
    with pytest.raises(ValueError, match="coverage"):
        suite.validate_trials(rows(), trials[:1])


@pytest.mark.parametrize("application", [None, [], "accepted"])
def test_malformed_application_counted_not_crashed(suite, application):
    trials = suite.plan(rows(), split="test", seed=31, command_only=True)
    result = suite.summarize(trials, [{
        "trial_id": trials[0]["trial_id"], "application": application,
    }])
    assert result["overall"]["errors"] == 1
    assert result["overall"]["missing_predictions"] == 1
    assert result["overall"]["recall"] == 0


def test_training_step_updates_weights_and_reports_pairs(learning):
    import torch
    trainer = load("scripts.train.speaker_demo_train")
    torch.manual_seed(17)
    model = learning.DemoKeywordNet(classes=3, width=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.classifier.weight.detach().clone()
    batch = (torch.randn(4, 40, 20), torch.tensor([20, 17, 18, 20]),
             torch.tensor([1, 1, 2, 2]), torch.tensor([0, 1, 0, 1]))
    result = trainer.optimization_step(model, batch, optimizer, weight=0.1)
    assert np.isfinite(result["loss"])
    assert result["paired_anchor_fraction"] == 1.0
    assert not torch.equal(before, model.classifier.weight)


def test_training_negative_weight_rejected(learning):
    import torch
    trainer = load("scripts.train.speaker_demo_train")
    model = learning.DemoKeywordNet(classes=3, width=16)
    batch = (torch.randn(2, 40, 20), torch.tensor([20, 20]),
             torch.tensor([1, 1]), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        trainer.optimization_step(
            model, batch, torch.optim.AdamW(model.parameters()), weight=-1.0)


def test_training_grid_has_nine_distinct_runs():
    import argparse
    import runpy
    path = Path("experiments/speaker-demo-v1/run_grid.py")
    assert path.exists(), "Missing grid driver"
    module = runpy.run_path(str(path))
    args = argparse.Namespace(
        manifest=Path("manifest.jsonl"), catalog=Path("catalog.json"),
        output=Path(".cache/speaker-demo-v1/grid"), seeds=[53, 71, 89],
        device="cpu", epochs=30, steps_per_epoch=96, batch_size=64,
    )
    commands = module["commands_for"](args)
    assert len(commands) == 9
    outputs = [c[c.index("--output") + 1] for c in commands]
    assert len(set(outputs)) == 9


def test_registry_patch_preserves_existing_content(tmp_path):
    import subprocess

    original = (
        "# Experiment directory and findings\n\n"
        "Evidence snapshot: preserve this existing content.\n"
    )
    target = tmp_path / "EXPERIMENTS.md"
    target.write_text(original, encoding="utf-8")
    patch = Path("experiments/speaker-demo-v1/register.patch").resolve()
    result = subprocess.run(
        ["git", "apply", "--unidiff-zero", str(patch)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    updated = target.read_text(encoding="utf-8")
    assert updated.count("## Speaker demo v1") == 1
    assert "Evidence snapshot: preserve this existing content." in updated
