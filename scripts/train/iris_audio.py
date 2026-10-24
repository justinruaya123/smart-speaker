"""Offline training audio/frontend reference; no microphone capture adapter.

Extracted from scripts/runtime/audio.py at 94c0021d7e22c9aa719246ac97105d7140a9f2a2.
Historical reconstruction continues to use its immutable source snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from math import gcd
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from einops import rearrange, reduce

import soundfile as sf


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 16000
    window: int = 400
    hop: int = 160
    fft: int = 512
    mel_bins: int = 40
    fmin: float = 60.0
    fmax: float = 7600.0
    log_floor: float = 1e-8
    convention: str = "htk_peak_triangles_power_natural_log"


DEFAULT = FrontendConfig()


@lru_cache(maxsize=8)
def mel_filters(config=DEFAULT):
    frequencies = np.linspace(0, config.sample_rate / 2, config.fft // 2 + 1)
    low, high = 2595 * np.log10(1 + np.array([config.fmin, config.fmax]) / 700)
    boundaries = 700 * (10 ** (np.linspace(low, high, config.mel_bins + 2) / 2595) - 1)
    filters = np.zeros((config.mel_bins, len(frequencies)), dtype=np.float32)
    for m in range(config.mel_bins):
        a, b, c = boundaries[m:m + 3]
        filters[m] = np.maximum(0, np.minimum((frequencies - a) / (b - a), (c - frequencies) / (c - b)))
    return filters


def load_audio(path, channel=0):
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    if not 0 <= channel < pcm.shape[1]:
        raise ValueError(f"Channel {channel} unavailable in {pcm.shape[1]}-channel audio")
    return resample_audio(pcm[:, channel], rate), rate


def resample_audio(pcm, rate, target=16000):
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.ndim != 1:
        raise ValueError("Choose a microphone channel before resampling")
    if rate != target:
        from scipy.signal import resample_poly
        common = gcd(int(rate), target)
        pcm = resample_poly(pcm, target // common, int(rate) // common).astype(np.float32)
    return pcm


def features(pcm, config=DEFAULT):
    """No future/center padding and no utterance-dependent normalization."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if len(pcm) < config.window:
        pcm = np.pad(pcm, (0, config.window - len(pcm)))
    windows = np.lib.stride_tricks.sliding_window_view(pcm, config.window)[::config.hop]
    window = np.hanning(config.window + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(windows * window, n=config.fft, axis=-1)
    power = np.abs(spectrum).astype(np.float32) ** 2
    mel = np.einsum("tf,mf->mt", power, mel_filters(config), optimize=True)
    return np.log(np.maximum(mel, config.log_floor)).astype(np.float32)


class StreamingFrontend:
    def __init__(self, config=DEFAULT):
        self.config = config
        self.pending = np.empty(0, dtype=np.float32)

    def push(self, pcm):
        joined = np.empty(len(self.pending) + len(pcm), dtype=np.float32)
        joined[:len(self.pending)] = self.pending
        joined[len(self.pending):] = pcm
        if len(joined) < self.config.window:
            self.pending = joined
            return np.empty((self.config.mel_bins, 0), dtype=np.float32)
        frames = features(joined, self.config)
        self.pending = joined[frames.shape[1] * self.config.hop:].copy()
        return frames




def cache_path(record, root):
    import hashlib
    import json
    recipe = json.dumps(record.get("augmentation"), sort_keys=True) if record.get("augmentation") else ""
    key = hashlib.sha256((record["id"] + str(asdict(DEFAULT)) + recipe).encode()).hexdigest()
    return Path(root) / ".cache/features" / key[:2] / (key + ".npy")


def cached_features(record, root):
    target = cache_path(record, root)
    if target.exists():
        return np.load(target)
    pcm, _ = load_audio(Path(root) / record["audio"])
    array = features(pcm)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Separate writers may prepare the same deterministic feature entry concurrently.
    # Publish a complete file atomically; never share a writable temporary filename.
    with NamedTemporaryFile(dir=target.parent, prefix=target.stem + "-", suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
        np.save(f, array, allow_pickle=False)
    try:
        try:
            temporary.rename(target)
        except FileExistsError:
            pass  # Another complete writer won the publication race on Windows.
    finally:
        temporary.unlink(missing_ok=True)
    return array
