"""Causal 25 ms HTK log-mel frontend with no utterance normalization."""
import numpy as np
from einops import rearrange, pack

SPEC = dict(sample_rate=16000, window=400, hop=160, fft=512, bins=40,
            low_hz=20, high_hz=7600, floor=1e-8, centered=False, version=1)


def mel_filter():
    hz = np.arange(257) * 16000 / 512
    low, high = 2595*np.log10(1+np.array([20., 7600.])/700)
    edges = 700*(10**(np.linspace(low, high, 42)/2595)-1)
    bank = np.zeros((257, 40))
    for k in range(40):
        bank[:, k] = np.maximum(0, np.minimum((hz-edges[k])/(edges[k+1]-edges[k]),
                                             (edges[k+2]-hz)/(edges[k+2]-edges[k+1])))
    return bank.astype('float32')


class StreamingFrontend:
    def __init__(self, mean=None, std=None):
        self.mean = np.asarray(mean if mean is not None else [0]*40, dtype='float32')
        self.std = np.asarray(std if std is not None else [1]*40, dtype='float32')
        if self.mean.shape != (40,) or self.std.shape != (40,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError('Invalid frozen normalization')
        self.window = (0.5-0.5*np.cos(2*np.pi*np.arange(400)/400)).astype('float32')
        self.bank = mel_filter()
        self.reset()

    def reset(self):
        self.pending = np.empty(0, dtype='float32')

    def feed(self, pcm):
        pcm = np.asarray(pcm, dtype='float32')
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Expected finite mono PCM')
        self.pending, _ = pack([self.pending, pcm], '*')
        count = max(0, 1+(len(self.pending)-400)//160)
        if not count:
            return np.empty((40, 0), dtype='float32')
        frames = np.lib.stride_tricks.sliding_window_view(self.pending, 400)[::160][:count]
        power = np.abs(np.fft.rfft(frames*self.window, n=512))**2
        logs = np.log(np.maximum(power @ self.bank, 1e-8)).astype('float32')
        self.pending = self.pending[count*160:].copy()
        return rearrange((logs-self.mean)/self.std, 't c -> c t')


def features(pcm, mean=None, std=None):
    return StreamingFrontend(mean, std).feed(pcm)
