"""Causal energy endpoint with sample-count timing and sticky capture faults."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EndpointConfig:
    sample_rate: int = 16000
    frame_samples: int = 320
    rms_threshold: float = 0.01
    minimum_speech_ms: int = 300
    trailing_silence_ms: int = 700
    maximum_ms: int = 12000

    def __post_init__(self) -> None:
        if self.sample_rate != 16000 or self.frame_samples != 320:
            raise ValueError('PC-v1 endpoint uses 16 kHz and 20 ms frames')
        if not 0 < self.rms_threshold < 1:
            raise ValueError('RMS threshold must be in (0,1)')
        if self.minimum_speech_ms != 300 or self.trailing_silence_ms != 700 or self.maximum_ms != 12000:
            raise ValueError('Endpoint timing changes require a protocol revision')


class Endpoint:
    def __init__(self, config: EndpointConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.pending = np.empty(0, dtype=np.float32)
        self.received = 0
        self.processed = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.end_sample = None
        self.fault = None
        self.last_active_sample = None

    def status(self) -> dict:
        return {'ended': self.end_sample is not None and self.fault is None,
                'end_sample': self.end_sample, 'fault': self.fault,
                'last_active_sample': self.last_active_sample}

    def feed(self, pcm: np.ndarray, sequence: int, *, overflow: bool = False) -> dict:
        if self.fault or self.end_sample is not None:
            return self.status()
        x = np.asarray(pcm, dtype=np.float32)
        if overflow:
            self.fault = 'capture_overflow'
        elif type(sequence) is not int or sequence != self.received:
            self.fault = 'capture_gap'
        elif x.ndim != 1 or len(x) > 16000 * 30 or not np.isfinite(x).all():
            self.fault = 'invalid_audio'
        if self.fault:
            return self.status()
        self.received += len(x)
        self.pending = np.concatenate((self.pending, x))
        size = self.config.frame_samples
        used = 0
        for start in range(0, len(self.pending) - size + 1, size):
            frame = self.pending[start:start + size]
            self.processed += size
            used += size
            active = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) >= self.config.rms_threshold
            if active:
                self.speech_samples += size
                self.silence_samples = 0
                self.last_active_sample = self.processed
            elif self.speech_samples:
                self.silence_samples += size
            if self.processed >= self.config.maximum_ms * 16:
                self.fault = 'utterance_too_long'
                break
            if (self.speech_samples >= self.config.minimum_speech_ms * 16
                    and self.silence_samples >= self.config.trailing_silence_ms * 16):
                self.end_sample = self.processed
                break
        self.pending = self.pending[used:].copy()
        return self.status()
