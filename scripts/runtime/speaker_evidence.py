"""Exact PCM/session evidence binding around the independent owner policy."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re

import numpy as np

from scripts.runtime.speaker_session import (
    SpeakerEvidence, SpeakerSession, SpeakerVerdict, gate_application,
)


def pcm_hash(pcm):
    pcm = np.asarray(pcm, np.float32)
    data = pcm.astype("<f4", copy=False).tobytes()
    prefix = f"pcm:f32le:16000:{len(pcm)}\0".encode()
    return hashlib.sha256(prefix+data).hexdigest()


@dataclass(frozen=True)
class CoverageWindow:
    start: int
    end: int
    evidence: SpeakerEvidence


@dataclass(frozen=True)
class BoundObservation:
    evidence: SpeakerEvidence
    pcm_sha256: str
    session_id: str
    sequence: int
    start_sample: int
    end_sample: int
    calibration_id: str
    coverage_verified: bool
    windows: list[CoverageWindow]


class OwnerApplication:
    """Backend.observe consumes only PCM and exact session/sample-span bindings.

    The backend must independently establish identity, overlap and coverage.
    This wrapper cannot establish those facts from a command embedding.
    """
    def __init__(self, backend, policy, model_id, calibration_id, *, command_encoder_id=None):
        if (model_id == command_encoder_id or not re.fullmatch("[a-f0-9]{64}", model_id)
                or not re.fullmatch("[a-f0-9]{64}", calibration_id)):
            raise ValueError("Separate calibrated owner identity required")
        self.backend, self.model_id, self.calibration_id = backend, model_id, calibration_id
        self.session = SpeakerSession(policy)
        self.active = None

    def _observe(self, pcm, token, sequence, start_sample, command):
        pcm = np.asarray(pcm, dtype=np.float32)
        if (pcm.ndim != 1 or len(pcm) == 0 or len(pcm) > 128000
                or not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1
                or type(sequence) is not int or sequence < 1
                or type(start_sample) is not int or start_sample < 0):
            raise ValueError("invalid_pcm_or_sequence")
        if self.backend is None:
            raise ValueError("speaker_backend_unavailable")
        observation = self.backend.observe(pcm, 16000, session_id=token,
                                           sequence=sequence, start_sample=start_sample)
        if not isinstance(observation, BoundObservation):
            raise ValueError("invalid_bound_observation")
        if (observation.pcm_sha256 != pcm_hash(pcm)
                or observation.session_id != token or observation.sequence != sequence
                or observation.start_sample != start_sample
                or observation.end_sample != start_sample+len(pcm)
                or observation.calibration_id != self.calibration_id
                or observation.evidence.model_id != self.model_id):
            raise ValueError("stale_or_mismatched_evidence")
        if not 0 <= observation.evidence.voiced_seconds <= len(pcm)/16000:
            raise ValueError("invalid_voiced_duration")
        if command:
            if observation.coverage_verified is not True or not observation.windows:
                raise ValueError("continuity_unchecked")
            expected = start_sample
            for window in observation.windows:
                if (type(window.start) is not int or type(window.end) is not int
                        or window.start != expected or window.end <= window.start
                        or window.end > start_sample+len(pcm)
                        or window.evidence.model_id != self.model_id
                        or not 0 <= window.evidence.voiced_seconds <= (window.end-window.start)/16000):
                    raise ValueError("coverage_gap_or_duration")
                expected = window.end
            if expected != start_sample+len(pcm):
                raise ValueError("coverage_gap")
        return observation

    def arm(self, wake_pcm, session_id, *, now, sequence, start_sample=0):
        if self.active is not None:
            started = self.active[2]
            if math.isfinite(now) and 0 <= now-started < self.session.policy.timeout_seconds:
                return {"accepted": False, "reason": "owner_busy", "session_id": session_id}
            self.session.reset()
            self.active = None
        try:
            # Only the actual accepted wake span may be supplied here.
            if len(wake_pcm) > 32000:
                raise ValueError("wake_reference_exceeds_bounded_buffer")
            observation = self._observe(wake_pcm, session_id, sequence, start_sample, False)
            verdict = self.session.arm(session_id, observation.evidence, now=now)
            if verdict.accepted:
                self.active = (session_id, sequence, now)
            return {"accepted": verdict.accepted, "reason": verdict.reason, "session_id": session_id}
        except Exception as error:
            self.session.reset()
            return {"accepted": False, "reason": str(error), "session_id": session_id}

    def finish(self, application, command_pcm, session_id, *, now, sequence,
               start_sample=0, fault=None):
        active = self.active
        self.active = None
        try:
            if fault:
                raise ValueError("capture_fault:"+fault)
            if active is None or session_id != active[0] or sequence != active[1]+1:
                raise ValueError("stale_session_or_sequence_discontinuity")
            observation = self._observe(command_pcm, session_id, sequence, start_sample, True)
            verdict = self.session.finish(session_id, observation.evidence, now=now,
                                          windows=[w.evidence for w in observation.windows])
        except Exception as error:
            self.session.reset()
            verdict = SpeakerVerdict(False, str(error), session_id)
        result = gate_application(application, verdict)
        # No raw identity embeddings leave this interface.
        if result.get("accepted"):
            result = {**result, "owner": {"session_id": session_id,
                      "model_id": self.model_id, "calibration_id": self.calibration_id,
                      "accepted": True, "overlap_policy": "reject"}}
        return result
