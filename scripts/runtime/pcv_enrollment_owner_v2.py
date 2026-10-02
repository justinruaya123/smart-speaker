"""Enrollment and speaker-ownership mechanisms, not acoustic recognizers.

Similarities are raw cosine scores, NOT probabilities. Tests use fabricated
vectors. A production caller needs trained embeddings, segment/voicing
validation, reference/session binding and independently calibrated thresholds.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.array(vector, dtype=np.float64, copy=True)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError('Expected a finite embedding vector')
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError('Empty embedding energy')
    return value / norm


class ExemplarBank:
    """Bounded, role-specific acoustic prototypes with max-per-ID scoring."""
    def __init__(self, role: str, entries: list[tuple[str, np.ndarray]],
                 encoder_hash: str, frontend_hash: str) -> None:
        if role not in {'light', 'song', 'artist', 'wake'}:
            raise ValueError('Invalid role')
        if not 1 <= len(entries) <= 96:
            raise ValueError('Exemplar budget must be 1..96')
        if not encoder_hash or not frontend_hash:
            raise ValueError('Missing encoder/frontend binding')
        self.role = role
        self.ids = []
        vectors = []
        for identifier, vector in entries:
            pattern = re.escape(role) + r'\.[a-z0-9_.-]+'
            if not re.fullmatch(pattern, identifier):
                raise ValueError('Wrong exemplar role')
            self.ids.append(identifier)
            vectors.append(unit(vector))
        if len({len(v) for v in vectors}) != 1:
            raise ValueError('Embedding dimensions differ')
        self.vectors = np.stack(vectors)
        self.vectors.flags.writeable = False
        self.ids = tuple(self.ids)
        payload = {'role': role, 'ids': self.ids,
                   'vectors': self.vectors.tolist(),
                   'encoder': encoder_hash, 'frontend': frontend_hash}
        self.binding = hashlib.sha256(json.dumps(
            payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def rank(self, query: np.ndarray) -> dict[str, Any]:
        query = unit(query)
        if query.size != self.vectors.shape[1]:
            raise ValueError('Query dimension mismatch')
        scores = self.vectors @ query
        by_id: dict[str, float] = {}
        for identifier, score in zip(self.ids, scores, strict=True):
            by_id[identifier] = max(by_id.get(identifier, -1.), float(score))
        ordered = sorted(by_id.items(), key=lambda item: (-item[1], item[0]))
        name, score = ordered[0]
        margin = score - ordered[1][1] if len(ordered) > 1 else 0.
        return {'id': name, 'similarity': score, 'margin': margin}

    def accept(self, query: np.ndarray, minimum: float, min_margin: float,
               policy_binding: str, *, acoustic_pass: bool, qualified: bool
               ) -> dict[str, Any]:
        if policy_binding != self.binding:
            raise ValueError('Enrollment policy binding differs')
        if (not np.isfinite([minimum, min_margin]).all()
                or not -1 <= minimum <= 1 or not 0 < min_margin <= 2
                or type(acoustic_pass) is not bool
                or type(qualified) is not bool):
            raise ValueError('Invalid policy')
        candidate = self.rank(query)
        accepted = (qualified and acoustic_pass
                    and candidate['similarity'] >= minimum
                    and candidate['margin'] >= min_margin)
        return {**candidate, 'accepted': accepted,
                'command': None, 'actionable': False,
                'evidence': 'cosine_similarity_not_calibrated_probability'}


class OwnerGuard:
    """Fail-closed ownership; explicitly bypassed for spliced class trials.

    Each observe() call represents a NEW non-overlapping voiced evidence span.
    Upstream must verify sequencing and total coverage; duplicate windows
    cannot count twice. No automatic reference update from command audio.
    This class does not implement speaker recognition or source separation.
    """
    def __init__(self, mode: str, minimum_similarity: float, *,
                 min_reference_ms: int, min_command_ms: int,
                 qualified: bool = False) -> None:
        if mode not in {'same_speaker', 'official_spliced'}:
            raise ValueError('Explicit owner mode required')
        if (not np.isfinite(minimum_similarity)
                or not -1 <= minimum_similarity <= 1
                or type(qualified) is not bool
                or any(type(n) is not int or n <= 0
                       for n in (min_reference_ms, min_command_ms))):
            raise ValueError('Invalid owner calibration')
        self.mode = mode
        self.minimum = minimum_similarity
        self.min_reference_ms = min_reference_ms
        self.min_command_ms = min_command_ms
        self.qualified = qualified
        self.reference = None
        self.reference_ms = 0
        self.voiced_ms = 0
        self.problem = None

    def arm(self, reference: np.ndarray, voiced_ms: int, *,
            overlap: bool = False) -> None:
        if type(voiced_ms) is not int or voiced_ms <= 0:
            raise ValueError('Invalid reference duration')
        if type(overlap) is not bool:
            raise ValueError("Overlap flag must be bool")
        self.reference = unit(reference)
        self.reference.flags.writeable = False
        self.reference_ms = voiced_ms
        self.voiced_ms = 0
        self.problem = "overlapping_wake" if overlap else None

    def observe(self, embedding: np.ndarray | None, milliseconds: int, *,
                overlap: bool = False, voiced: bool = True) -> None:
        if self.reference is None:
            raise ValueError('Turn is not armed')
        if (type(milliseconds) is not int or milliseconds <= 0
                or type(voiced) is not bool or type(overlap) is not bool):
            raise ValueError('Invalid evidence span')
        if overlap:
            self.problem = 'overlap'
        if not voiced:
            return
        self.voiced_ms += milliseconds
        if embedding is None:
            self.problem = self.problem or 'uncertain_speaker'
            return
        value = unit(embedding)
        if value.shape != self.reference.shape:
            raise ValueError('Speaker embedding dimension mismatch')
        if float(self.reference @ value) < self.minimum:
            self.problem = self.problem or 'speaker_changed'

    def finish(self) -> dict[str, Any]:
        if self.reference is None:
            return {'pass': False, 'owner_enforced': True, 'reason': 'unarmed'}
        if self.mode == 'official_spliced':
            return {'pass': True, 'owner_enforced': False,
                    'reason': 'official_spliced_not_owner_test'}
        reason = None
        if not self.qualified:
            reason = 'unqualified_speaker_model'
        elif (self.reference_ms < self.min_reference_ms
              or self.voiced_ms < self.min_command_ms):
            reason = 'insufficient_speaker_evidence'
        elif self.problem:
            reason = self.problem
        return {'pass': reason is None, 'owner_enforced': True,
                'reason': reason}


class AudioSequence:
    """Track contiguous NEW PCM samples; a fault requires explicit reset."""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.expected = 0
        self.faulted = False

    def feed(self, offset: int, count: int) -> None:
        if (self.faulted or type(offset) is not int or type(count) is not int
                or count <= 0 or offset != self.expected):
            self.faulted = True
            raise ValueError("Invalid audio sequence: reset required")
        self.expected += count
