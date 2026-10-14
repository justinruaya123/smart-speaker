import numpy as np
import pytest
from dataclasses import asdict
from scripts.runtime.vcm_intent_slots_contract import digest

from scripts.runtime.speaker_session import SpeakerEvidence, SpeakerPolicy
from scripts.runtime.speaker_evidence import BoundObservation, CoverageWindow, OwnerApplication
from scripts.test.speaker_demo_suite import pcm_digest

MODEL = "a"*64
RECEIPT = {"model_id": MODEL, "policy": asdict(SpeakerPolicy(.8)),
           "status": "software_fixture"}
CALIBRATION = digest(RECEIPT)


class NumericalBackend:
    """Software fixture, not measured acoustic identity or overlap."""
    def __init__(self):
        self.voice = [1., 0.]
        self.overlap = False
        self.covered = True
        self.clipped = False
    def observe(self, pcm, sample_rate, *, session_id, sequence, start_sample):
        evidence = SpeakerEvidence(self.voice, MODEL, len(pcm)/sample_rate,
                                   self.overlap, self.clipped)
        windows = [CoverageWindow(start_sample, start_sample+len(pcm), evidence)]
        return BoundObservation(evidence, pcm_digest(pcm), session_id, sequence,
                                start_sample, start_sample+len(pcm),
                                CALIBRATION, self.covered, windows)


def owner(backend=None):
    return OwnerApplication(backend, SpeakerPolicy(.8), MODEL, CALIBRATION,
                            calibration_receipt=RECEIPT)


def app():
    return {"accepted": True, "command": {"intent": "STOP", "slots": {}}, "reason": None}


def test_genuine_one_wake_one_command_and_new_person():
    b = NumericalBackend()
    o = owner(b)
    pcm = np.ones(16000, np.float32)*.1
    assert o.arm(pcm, "A", now=0, sequence=1)["accepted"]
    assert not o.arm(pcm, "B", now=.1, sequence=2)["accepted"]
    assert o.finish(app(), pcm, "A", now=1, sequence=2)["accepted"]
    assert not o.finish(app(), pcm, "A", now=2, sequence=3)["accepted"]
    b.voice = [0., 1.]
    assert o.arm(pcm, "B", now=3, sequence=4)["accepted"]
    assert o.finish(app(), pcm, "B", now=4, sequence=5)["accepted"]


@pytest.mark.parametrize("change", ["impostor", "overlap", "unknown_overlap", "coverage", "clipped", "stale", "overflow", "expired"])
def test_unverifiable_and_mismatched_commands_fail_closed(change):
    b = NumericalBackend()
    o = owner(b)
    pcm = np.ones(16000, np.float32)*.1
    assert o.arm(pcm, "A", now=0, sequence=1)["accepted"]
    kwargs = dict(now=1, sequence=2)
    if change == "impostor": b.voice = [0., 1.]
    if change == "overlap": b.overlap = True
    if change == "unknown_overlap": b.overlap = None
    if change == "coverage": b.covered = False
    if change == "clipped": b.clipped = True
    if change == "stale": kwargs["sequence"] = 1
    if change == "overflow": kwargs["fault"] = "overflow"
    if change == "expired": kwargs["now"] = 9
    result = o.finish(app(), pcm, "A", **kwargs)
    assert not result["accepted"]
    assert "embedding" not in str(result)
    assert not o.finish(app(), pcm, "A", now=10, sequence=3)["accepted"]


def test_missing_backend_short_reference_and_vcm_rejection_not_rescued():
    pcm = np.ones(16000, np.float32)*.1
    assert not owner().arm(pcm, "A", now=0, sequence=1)["accepted"]
    b = NumericalBackend()
    o = owner(b)
    assert not o.arm(pcm[:3000], "A", now=0, sequence=1)["accepted"]
    assert o.arm(pcm, "A", now=1, sequence=2)["accepted"]
    rejected = {"accepted": False, "command": None, "reason": "intent_confidence"}
    assert o.finish(rejected, pcm, "A", now=2, sequence=3) == rejected


def test_content_identity_binding_rejected():
    with pytest.raises(ValueError):
        OwnerApplication(NumericalBackend(), SpeakerPolicy(.8), MODEL, CALIBRATION,
                         command_encoder_id=MODEL)


def test_owner_replay_fault_does_not_drop_remaining_trials(tmp_path, monkeypatch):
    import dataclasses
    import sys
    import types
    from scripts.test import vcm_intent_slots_owner_replay as replay
    from scripts.train.vcm_intent_slots_data import sha
    path = tmp_path/"identity.weights"
    path.write_text("numerical owner fixture")
    files = {str(path): sha(path)}
    model_id = digest(files)
    receipt = {"model_id": model_id, "policy": asdict(SpeakerPolicy(.8)),
               "status": "human_calibration_frozen"}
    calibration_id = digest(receipt)
    class Backend(NumericalBackend):
        def observe(self, *args, **kwargs):
            observation = super().observe(*args, **kwargs)
            evidence = dataclasses.replace(observation.evidence, model_id=model_id)
            windows = [dataclasses.replace(w, evidence=evidence) for w in observation.windows]
            return dataclasses.replace(observation, evidence=evidence, windows=windows,
                                       calibration_id=calibration_id)
    module = types.ModuleType("owner_fixture_backend")
    module.factory = lambda binding: Backend()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    def read(row, root):
        if row["id"] == "missing":
            raise FileNotFoundError("missing original WAV")
        return np.ones(16000, np.float32)*.1
    monkeypatch.setattr(replay, "read_audio", read)
    class Runtime:
        recipe = {"encoder_id": "c"*64}
        def recognize(self, pcm):
            return {"final_application": app()}
    base = {"track": "original_paired_human_owner_sessions",
            "wake": {"id": "wake", "kind": "human"},
            "command": {"id": "ok", "kind": "human", "speaker_id": "person",
                        "command": {"intent": "STOP", "slots": {}}},
            "verified_identity_labels": True, "verified_wake_span": True,
            "wake_span": [0, 16000], "command_elapsed_seconds": 1,
            "command_start_sample": 16000, "same_person": True}
    pairs = {"backend": {"files": files, "factory": "owner_fixture_backend:factory",
                         "model_id": model_id}, "policy": asdict(SpeakerPolicy(.8)),
             "calibration_id": calibration_id, "calibration_receipt": receipt,
             "calibration_people_disjoint": True,
             "trials": [{**base, "id": "fault", "command": {**base["command"], "id": "missing"}},
                        {**base, "id": "good"}]}
    result = replay.replay_owner(pairs, Runtime(), tmp_path, tmp_path/"replay")
    for track in ("command_only", "owner_aware"):
        assert result[track]["all"]["supported_n"] == 2
        assert result[track]["all"]["faults"] == 1
        assert result[track]["all"]["correct"] == 1


def test_owner_policy_cannot_reuse_calibration_id_after_threshold_change():
    o = owner(NumericalBackend())
    with pytest.raises(ValueError, match="calibration|policy"):
        OwnerApplication(NumericalBackend(), SpeakerPolicy(.95), MODEL, CALIBRATION,
                         calibration_receipt=o.calibration_receipt)
