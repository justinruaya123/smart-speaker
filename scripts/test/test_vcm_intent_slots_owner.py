import numpy as np
import pytest

from scripts.runtime.speaker_session import SpeakerEvidence, SpeakerPolicy
from scripts.runtime.speaker_evidence import BoundObservation, CoverageWindow, OwnerApplication
from scripts.test.speaker_demo_suite import pcm_digest

MODEL = "a"*64
CALIBRATION = "b"*64


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
    return OwnerApplication(backend, SpeakerPolicy(.8), MODEL, CALIBRATION)


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
