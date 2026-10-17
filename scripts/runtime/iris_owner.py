"""Portable CPU identity/overlap inference; experimental proxy calibration.

Embeddings stay in memory. Energy masks conservatively account for detected
signal duration; they do not turn short commands into human wake qualification.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .audio import DEFAULT, features
from .speaker_session import SpeakerEvidence, SpeakerPolicy

PCM_NORMALIZATION = {'kind': 'utterance_rms', 'target': .035, 'floor': 1e-6}


def _pcm(pcm) -> np.ndarray:
    values = np.asarray(pcm, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Owner input must be finite mono 16 kHz PCM')
    return values


def speech_mask(pcm) -> np.ndarray:
    """Frozen 25-ms energy windows / 10-ms hop; no silence-padding credit."""
    values = _pcm(pcm)
    if len(values) < DEFAULT.window:
        return np.zeros(0, dtype=bool)
    frames = np.lib.stride_tricks.sliding_window_view(values, DEFAULT.window)[::DEFAULT.hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    threshold = max(.001, float(rms.max(initial=0)) * .08)
    return rms > threshold


def voiced_seconds(pcm) -> float:
    mask = speech_mask(pcm)
    if not mask.any():
        return 0.0
    # Union of valid detected frames, including the final window support.
    starts = np.flatnonzero(mask) * DEFAULT.hop
    total, last_end = 0, -1
    for start in starts:
        end = int(start) + DEFAULT.window
        total += end - max(int(start), last_end)
        last_end = end
    return total / DEFAULT.sample_rate


def owner_features(pcm):
    """Shared log-mel frontend after declared level normalization.

    Equalizing single and mixed waveform RMS prevents a trivial overlap head
    from relying on the deliberately normalized construction level alone.
    """
    values = _pcm(pcm)
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2))) if len(values) else 0.
    normalized = values * (PCM_NORMALIZATION['target'] / max(rms, PCM_NORMALIZATION['floor']))
    return features(normalized)


def continuity_spans(pcm, window_seconds: float = 1.2, min_voiced_seconds: float = .25):
    """Return overlapping PCM spans covering *every* detected signal frame.

    No detected frame may be omitted because its evidence is difficult. A tiny
    island is merged into a neighboring span, or coverage fails closed.
    """
    values = _pcm(pcm)
    mask = speech_mask(values)
    if not mask.any() or voiced_seconds(values) < min_voiced_seconds:
        return None
    indices = np.flatnonzero(mask)
    # Cover the entire supplied capture, including quiet edges. Energy masks
    # measure duration, but must not authorize dropping a difficult quiet
    # leading/trailing speaker from continuity evidence.
    first, last = 0, len(values)
    size = max(400, round(window_seconds * DEFAULT.sample_rate))
    step = size // 2
    starts = list(range(first, max(first + 1, last - size + 1), step))
    starts.append(max(first, last - size))
    spans = sorted(set((start, min(last, start + size)) for start in starts))
    # Extend sparse speech spans into their neighbours without dropping speech.
    for i in range(len(spans)):
        a, b = spans[i]
        if voiced_seconds(values[a:b]) < min_voiced_seconds:
            a = spans[max(0, i - 1)][0]
            b = spans[min(len(spans) - 1, i + 1)][1]
            if voiced_seconds(values[a:b]) < min_voiced_seconds:
                return None
            spans[i] = (a, b)
    covered = np.zeros(len(values), dtype=bool)
    for a, b in spans:
        covered[a:b] = True
    if not covered.all():
        return None
    return tuple(sorted(set(spans)))


class OwnerBackend:
    """One hash-bound ONNX backend and its explicitly experimental policy."""
    def __init__(self, bundle: str | Path, *, threads: int = 1):
        import onnxruntime as ort
        bundle = Path(bundle)
        metadata = bundle if bundle.is_file() else bundle / 'owner.json'
        self.metadata = json.loads(metadata.read_text(encoding='utf8'))
        if self.metadata.get('schema_version') != 1 or self.metadata.get('frontend') != asdict(DEFAULT):
            raise ValueError('Unsupported owner metadata/frontend')
        if self.metadata.get('pcm_normalization') != PCM_NORMALIZATION:
            raise ValueError('Unsupported owner PCM normalization')
        if self.metadata.get('calibration_kind') != 'short_utterance_proxy+synthetic_wake_proxy':
            raise ValueError('Missing declared owner calibration provenance')
        graph = metadata.parent / self.metadata['model']['path']
        self.model_id = hashlib.sha256(graph.read_bytes()).hexdigest()
        if self.model_id != self.metadata['model']['sha256']:
            raise ValueError('Owner graph hash mismatch')
        norm = self.metadata['normalization']
        self.mean = np.array(norm['mean'], dtype=np.float32).reshape(40, 1)
        self.std = np.array(norm['std'], dtype=np.float32).reshape(40, 1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid owner normalization')
        declared = dict(self.metadata['policy'])
        self.overlap_threshold = float(declared.pop('overlap_threshold'))
        if not np.isfinite(self.overlap_threshold) or not 0 <= self.overlap_threshold <= 1:
            raise ValueError('Invalid overlap threshold')
        self.policy = SpeakerPolicy(**declared)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])

    def predict(self, pcm) -> tuple[np.ndarray, float]:
        values = _pcm(pcm)
        acoustic = (owner_features(values) - self.mean) / self.std
        result = self.session.run(None, {'features': acoustic[None].astype(np.float32),
                                        'lengths': np.array([acoustic.shape[1]], dtype=np.int64)})
        vector = np.asarray(result[0][0], dtype=np.float32)
        probability = float(np.asarray(result[1]).reshape(-1)[0])
        if vector.shape != (128,) or not np.isfinite(vector).all() or not np.isfinite(probability):
            raise ValueError('Invalid owner model output')
        return vector, probability

    def evidence(self, pcm) -> SpeakerEvidence:
        values = _pcm(pcm)
        duration = voiced_seconds(values)
        if duration == 0:
            return SpeakerEvidence((), self.model_id, 0, None, False)
        embedding, probability = self.predict(values)
        clipped = bool(np.mean(np.abs(values) >= .999) > .001)
        return SpeakerEvidence(embedding, self.model_id, duration,
                               bool(probability >= self.overlap_threshold), clipped)

    def continuity(self, pcm) -> tuple[SpeakerEvidence, ...] | None:
        values = _pcm(pcm)
        spans = continuity_spans(values, min_voiced_seconds=self.policy.min_window_seconds)
        if spans is None:
            return None
        evidence = tuple(self.evidence(values[start:end]) for start, end in spans)
        if any(item.voiced_seconds < self.policy.min_window_seconds for item in evidence):
            return None
        return evidence
