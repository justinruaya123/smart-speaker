"""One-wake/one-command speaker ownership; no model or identity database.

The caller supplies evidence from a separately validated speaker/overlap
backend.
This policy cannot turn embeddings or synthetic fixtures into acoustic
evidence.
Do not serialize embeddings into application logs. Shipping defaults are
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SpeakerEvidence:
    """Backend evidence for speech, not silence-padded audio duration."""

    embedding: Sequence[float]
    model_id: str
    voiced_seconds: float
    overlap: bool | None
    clipped: bool = False


@dataclass(frozen=True)
class SpeakerPolicy:
    """Thresholds must be calibrated on wake-to-command human pairs."""

    threshold: float
    min_reference_seconds: float = 0.8
    min_command_seconds: float = 0.4
    min_window_seconds: float = 0.25
    timeout_seconds: float = 8.0
    require_windows: bool = True

    def __post_init__(self) -> None:
        if not _finite(self.threshold) or not -1 <= self.threshold <= 1:
            raise ValueError("Cosine threshold must be finite and in [-1, 1]")
        for name in ("min_reference_seconds", "min_command_seconds",
                     "min_window_seconds", "timeout_seconds"):
            value = getattr(self, name)
            if not _finite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.require_windows) is not bool:
            raise ValueError("require_windows must be a boolean")


@dataclass(frozen=True)
class SpeakerVerdict:
    accepted: bool
    reason: str | None
    session_id: str
    score: float | None = None


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _validate(
    evidence: SpeakerEvidence, minimum: float,
) -> tuple[np.ndarray | None, str | None]:
    if not isinstance(evidence, SpeakerEvidence):
        return None, "invalid_evidence"
    if not isinstance(evidence.model_id, str) or not re.fullmatch(
        r"[a-f0-9]{64}", evidence.model_id
    ):
        return None, "invalid_model_identity"
    if type(evidence.clipped) is not bool:
        return None, "invalid_evidence"
    if evidence.clipped:
        return None, "clipped_audio"
    if evidence.overlap is True:
        return None, "overlap"
    if evidence.overlap is not False:
        return None, "overlap_unchecked"
    if not _finite(
        evidence.voiced_seconds
    ) or evidence.voiced_seconds < minimum:
        return None, "insufficient_speech"
    try:
        vector = np.array(evidence.embedding, dtype=np.float64, copy=True)
    except (TypeError, ValueError):
        return None, "invalid_embedding"
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        return None, "invalid_embedding"
    # Scale first to avoid overflow when normalizing large finite embeddings.
    peak = float(np.max(np.abs(vector)))
    if peak < 1e-12:
        return None, "invalid_embedding"
    vector /= peak
    vector /= np.linalg.norm(vector)
    vector.setflags(write=False)
    return vector, None


class SpeakerSession:
    """Single owner; finish consumes the session even when validation fails.

    Continuity windows must cover all command-bearing speech, as established by
    the backend. This class validates supplied windows, not their segmentation.
    A backend lacking verified coverage must not supply a fabricated
    window list.
    """

    def __init__(self, policy: SpeakerPolicy):
        self.policy = policy
        self.reset()

    def reset(self) -> None:
        self._owner: tuple[str, str, np.ndarray, float] | None = None

    def arm(
        self, session_id: str, reference: SpeakerEvidence, *, now: float,
    ) -> SpeakerVerdict:
        if not isinstance(session_id, str) or not session_id:
            return SpeakerVerdict(False, "invalid_session", "")
        if not _finite(now):
            self.reset()
            return SpeakerVerdict(False, "invalid_clock", session_id)
        if self._owner is not None:
            _, _, _, start = self._owner
            if now < start:
                self.reset()
                return SpeakerVerdict(False, "invalid_clock", session_id)
            if now - start < self.policy.timeout_seconds:
                return SpeakerVerdict(False, "owner_busy", session_id)
            self.reset()
        vector, reason = _validate(
            reference, self.policy.min_reference_seconds
        )
        if reason:
            return SpeakerVerdict(False, reason, session_id)
        self._owner = (session_id, reference.model_id, vector, float(now))
        return SpeakerVerdict(True, None, session_id)

    def finish(
        self, session_id: str, command: SpeakerEvidence, *, now: float,
        windows: Sequence[SpeakerEvidence] | None = None,
    ) -> SpeakerVerdict:
        owner = self._owner
        self.reset()
        if owner is None:
            return SpeakerVerdict(False, "no_owner", session_id)
        token, model_id, reference, started = owner
        if session_id != token:
            return SpeakerVerdict(False, "session_mismatch", session_id)
        if not _finite(now) or now < started:
            return SpeakerVerdict(False, "invalid_clock", session_id)
        if now - started >= self.policy.timeout_seconds:
            return SpeakerVerdict(False, "owner_expired", session_id)
        if self.policy.require_windows and (
            windows is None or len(windows) == 0
        ):
            return SpeakerVerdict(False, "continuity_unchecked", session_id)
        observations = [(command, self.policy.min_command_seconds)]
        if windows is not None:
            observations.extend(
                (item, self.policy.min_window_seconds) for item in windows
            )
        scores = []
        for observation, minimum in observations:
            vector, reason = _validate(observation, minimum)
            if reason:
                return SpeakerVerdict(False, reason, session_id)
            if observation.model_id != model_id:
                return SpeakerVerdict(False, "model_mismatch", session_id)
            if vector.shape != reference.shape:
                return SpeakerVerdict(False, "embedding_shape", session_id)
            score = float(np.clip(np.dot(reference, vector), -1.0, 1.0))
            scores.append(score)
            if score < self.policy.threshold:
                return SpeakerVerdict(
                    False, "speaker_mismatch", session_id, score
                )
        return SpeakerVerdict(True, None, session_id, min(scores))


def gate_application(
    application: Mapping, verdict: SpeakerVerdict,
) -> dict:
    """Intersect trusted server decisions; never rescue an existing
    rejection."""
    result = dict(application)
    if result.get("accepted") is not True:
        return result
    if not isinstance(verdict, SpeakerVerdict) or verdict.accepted is not True:
        reason = verdict.reason if isinstance(
            verdict, SpeakerVerdict
        ) else None
        return {**result, "accepted": False, "command": None,
                "reason": reason or "speaker_unverified"}
    return result
