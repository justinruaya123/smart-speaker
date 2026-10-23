"""Experimental, uninstalled Hey Iris openWakeWord CPU companion.

Feature streaming is narrowly adapted from David Scripka's Apache-2.0
openWakeWord AudioFeatures at 368c03716d1e92591906a84949bc477f3a834455.
Copyright 2022 David Scripka. Modified: bounded buffers, deterministic reset,
strict float-PCM boundary and independent two-hit hysteresis. Full license:
experiments/iris-openwakeword-publication-v1/LICENSE-openWakeWord-Apache-2.0.txt.
The classifier is ASHR12/iris full-10k, not Jarvis or a transcription graph.
Only NumPy and existing ONNX Runtime are required. No downloads occur here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

FILES = {
    'hey_iris.onnx': '82f6e6ed82837400e0a12a6e6cf3a49bac7ff8181efbb965827f6e2a29961eda',
    'melspectrogram.onnx': 'ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f',
    'embedding_model.onnx': '70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f',
}


class HeyIrisOpenWakeWord:
    """Consume NEW mono 16kHz float PCM, never overlapping rolling windows.

    push returns None until a complete 1280-sample step, otherwise 0.0/1.0.
    One call is bounded to 16000 samples; pending storage stays below 1280.
    The fixed policy is .5 / two hits / .3 release / 16000-sample cooldown.
    This is deliberately NOT a RollingWake score callback: its features retain
    history. Parent integration must inject this push/reset object directly.
    """
    def __init__(self, folder):
        import onnxruntime as ort
        folder = Path(folder).resolve()
        payloads = {}
        for name, expected in FILES.items():
            path = folder / name
            if (not path.is_file() or not path.resolve().is_relative_to(folder)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError('Hey Iris openWakeWord identity mismatch: ' + name)
            payloads[name] = path.read_bytes()
        cfg = ort.SessionOptions()
        cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
        self.sessions = {name: ort.InferenceSession(data, cfg, providers=['CPUExecutionProvider'])
                         for name, data in payloads.items()}
        expected_inputs = {'hey_iris.onnx': ('embeddings', [16, 96]),
                           'embedding_model.onnx': ('input_1', [76, 32, 1]),
                           'melspectrogram.onnx': ('input', None)}
        for name, (key, tail) in expected_inputs.items():
            inputs = self.sessions[name].get_inputs()
            if (len(inputs) != 1 or inputs[0].name != key
                    or inputs[0].type != 'tensor(float)'
                    or (tail is not None and inputs[0].shape[1:] != tail)):
                raise ValueError('Unexpected Hey Iris ONNX interface')
        self.reset()

    def _mel(self, pcm):
        raw = self.sessions['melspectrogram.onnx'].run(None, {'input': pcm.astype(np.float32)[None]})[0]
        return np.squeeze(raw) / 10 + 2

    def _embed(self, windows):
        data = np.asarray(windows, dtype=np.float32)[..., None]
        return self.sessions['embedding_model.onnx'].run(None, {'input_1': data})[0].reshape(-1, 96)

    def reset(self):
        # Upstream uses random 4-second noise for the feature warm start. Make
        # this local reset deterministic, without mutating NumPy global RNG.
        warm = np.random.RandomState(731).randint(-1000, 1000, 64000).astype(np.int16)
        spec = self._mel(warm)
        windows = [spec[i:i + 76] for i in range(0, len(spec) - 75, 8)]
        self.embeddings = self._embed(windows)[-16:]
        self.mels = np.ones((76, 32))
        self.raw = np.empty(0, dtype=np.int16)
        self.pending = np.empty(0, dtype=np.int16)
        self.samples = self.frames = self.hits = self.cooldown = 0
        self.latched = False
        self.last_scores = []

    def push(self, pcm):
        x = np.asarray(pcm)
        if (x.ndim != 1 or x.dtype.kind != 'f' or len(x) > 16000
                or not np.isfinite(x).all() or (len(x) and np.max(np.abs(x)) > 1)):
            raise ValueError('Invalid wake PCM: bounded finite mono float samples in [-1,1] required')
        x = np.clip(x.astype(np.float64) * 32768, -32768, 32767).astype(np.int16)
        self.pending = np.concatenate((self.pending, x))
        self.last_scores = []
        result = None
        while len(self.pending) >= 1280:
            block, self.pending = self.pending[:1280], self.pending[1280:]
            self.raw = np.concatenate((self.raw, block))
            self.mels = np.concatenate((self.mels, self._mel(self.raw)), axis=0)[-76:]
            self.raw = self.raw[-480:]
            self.embeddings = np.concatenate((self.embeddings, self._embed([self.mels])), axis=0)[-16:]
            score = float(self.sessions['hey_iris.onnx'].run(
                None, {'embeddings': self.embeddings[None].astype(np.float32)})[0].reshape(-1)[0])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid Hey Iris ONNX score')
            if self.frames < 5:
                score = 0.0
            self.frames += 1
            self.samples += 1280
            self.last_scores.append((self.samples, score))
            self.cooldown = max(0, self.cooldown - 1280)
            if score < .3 and not self.cooldown:
                self.latched = False
            self.hits = self.hits + 1 if score >= .5 else 0
            trigger = self.hits >= 2 and not self.latched and not self.cooldown
            if trigger:
                self.latched = True
                self.cooldown = 16000
                self.hits = 0
            result = max(result or 0.0, float(trigger))
        return result
