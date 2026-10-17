"""Local CPU adapter for the pinned Silero v5.1 VAD (Silero Team, MIT).
See the shipped upstream license files and wake-models.json for provenance.
No package download or transcription occurs in this module.
"""
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np
from einops import pack, rearrange
import onnxruntime as ort


def session(folder, name):
    folder = Path(folder)
    expected = json.loads((folder / 'wake-models.json').read_text(encoding='utf-8'))['files'][name]
    path = folder / name
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != expected['bytes'] or actual != expected['sha256']:
        raise ValueError(f'Pretrained model integrity failed: {name}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


class SileroProbability:
    def __init__(self, folder):
        self.model = session(folder, 'silero-vad.onnx')
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)

    def __call__(self, pcm):
        if pcm.shape != (512,):
            raise ValueError('Silero VAD requires exactly 512 samples at 16 kHz')
        audio, _ = pack([self.context, pcm], '*')
        probability, self.state = self.model.run(None, {
            'input': rearrange(audio, 'samples -> 1 samples'), 'state': self.state,
            'sr': np.array(16000, dtype=np.int64)})
        self.context = pcm[-64:].copy()
        return float(probability.item())


class SileroVAD:
    """Neural endpointing; original PCM is retained for the command model."""
    frame_size = 512

    def __init__(self, folder, audio, probability=None):
        self.audio = audio
        self.model = probability if probability is not None else SileroProbability(folder)
        self.silence_limit = round(16 * audio['silence_ms'])
        self.limit = round(16000 * audio['max_utterance_seconds'])
        self.threshold = audio.get('speech_threshold', .5)
        self.reset()

    def reset(self):
        self.model.reset()
        self.active = False
        self.probability = 0.
        self.preroll = deque(maxlen=max(1, int(np.ceil(16 * self.audio['preroll_ms'] / 512))))
        self.buffer = []
        self.elapsed_samples = self.silence_samples = 0

    def diagnostics(self):
        return {'state': 'speech' if self.active else 'listening',
                'speech_probability': self.probability, 'threshold': self.threshold,
                'utterance_seconds': self.elapsed_samples / 16000,
                'silence_ms': self.silence_samples / 16}

    def push(self, pcm):
        self.probability = self.model(pcm)
        speech = self.probability >= self.threshold
        if not self.active:
            self.preroll.append(pcm.copy())
            if not speech:
                return None
            self.active, self.buffer = True, list(self.preroll)
            self.elapsed_samples = sum(map(len, self.buffer))
        else:
            self.buffer.append(pcm.copy())
            self.elapsed_samples += len(pcm)
        # Hysteresis prevents uncertain frames from rapidly splitting speech.
        self.silence_samples = self.silence_samples + len(pcm) if self.probability < self.threshold - .15 else 0
        timed_out = self.elapsed_samples >= self.limit
        if timed_out or self.silence_samples >= self.silence_limit:
            audio, _ = pack(self.buffer, '*')
            self.last_endpoint = self.diagnostics()
            self.reset()
            return audio[:self.limit], timed_out
        return None

    def segment(self, pcm):
        self.reset()
        tail = self.silence_limit + 512
        padded = np.pad(pcm, (0, tail + (-len(pcm) - tail) % 512))
        result = []
        for offset in range(0, len(padded), 512):
            endpoint = self.push(padded[offset:offset + 512])
            if endpoint is not None:
                result.append(endpoint)
        return result
