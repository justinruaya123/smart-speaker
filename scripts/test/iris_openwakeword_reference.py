"""Audit-only reference transcription, not imported downloaded vendor code.

Copyright 2022 David Scripka. Apache-2.0; see experiment's full license.
Methods retain the pinned upstream algorithm's full-size deque/mel/feature
buffers. Modified only to inject existing CPU sessions and deterministic RNG.
No untrusted repository scripts, pickle files, downloads or trainer run.
"""
from collections import deque
import numpy as np


class UpstreamFeatureReference:
    def __init__(self, sessions):
        self.sessions = sessions
        self.raw_data_buffer = deque(maxlen=16000 * 10)
        self.melspectrogram_buffer = np.ones((76, 32))
        self.melspectrogram_max_len = 970
        self.feature_buffer_max_len = 120
        warm = np.random.RandomState(731).randint(-1000, 1000, 64000).astype(np.int16)
        spec = self.mel(warm)
        windows = []
        for i in range(0, spec.shape[0], 8):
            window = spec[i:i + 76]
            if window.shape[0] == 76:
                windows.append(window)
        self.feature_buffer = self.embedding(np.array(windows)[..., None].astype(np.float32))
        self.frames = 0

    def mel(self, data):
        x = np.array(data).astype(np.int16) if isinstance(data, list) else data
        x = x[None] if len(x.shape) < 2 else x
        spec = np.squeeze(self.sessions['melspectrogram.onnx'].run(None, {'input': x.astype(np.float32)})[0])
        return spec / 10 + 2

    def embedding(self, data):
        return self.sessions['embedding_model.onnx'].run(None, {'input_1': data})[0].squeeze()

    def step(self, block):
        assert block.shape == (1280,) and block.dtype == np.int16
        self.raw_data_buffer.extend(block.tolist())
        self.melspectrogram_buffer = np.vstack((
            self.melspectrogram_buffer, self.mel(list(self.raw_data_buffer)[-1280 - 160 * 3:])))
        self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:]
        ndx = len(self.melspectrogram_buffer)
        x = self.melspectrogram_buffer[-76 + ndx:ndx].astype(np.float32)[None, :, :, None]
        self.feature_buffer = np.vstack((self.feature_buffer, self.embedding(x)))
        self.feature_buffer = self.feature_buffer[-self.feature_buffer_max_len:]
        features = self.feature_buffer[-16:][None].astype(np.float32)
        value = float(self.sessions['hey_iris.onnx'].run(None, {'embeddings': features})[0].reshape(-1)[0])
        if self.frames < 5:
            value = 0.0
        self.frames += 1
        return value
