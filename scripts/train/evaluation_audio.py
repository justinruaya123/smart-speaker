"""Original bounded WAV decoding for model evaluation-data export only.

No benchmark runner, application worker, capture, resampling or provider action.
The two functions retain the frozen implementation and constants verbatim.
"""
import hashlib
import io
from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
MAX_SAMPLES = 240000


def _wav_info(path):
    import soundfile as sf
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.format not in {'WAV', 'WAVEX'}:
        raise ValueError('Replay requires original mono 16000 Hz WAV PCM; no mixing or resampling is performed.')
    if not 1 <= info.frames <= MAX_SAMPLES:
        raise ValueError('Command WAV must contain 1..240000 samples (at most 15 seconds).')
    if Path(path).stat().st_size > 32 * 1024**2:
        raise ValueError('Command WAV exceeds the 32 MiB file bound.')
    return info


def read_pcm(path, expected_sha256=None):
    import soundfile as sf
    _wav_info(path)
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Recording identity changed after preflight: ' + str(path))
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        if stream.samplerate != SAMPLE_RATE or stream.channels != 1 or not 1 <= len(stream) <= MAX_SAMPLES:
            raise ValueError('Expected original mono 16000 Hz command PCM')
        pcm = stream.read(dtype='float32', always_2d=False)
    if pcm.ndim != 1 or not np.isfinite(pcm).all() or np.max(np.abs(pcm), initial=0) > 1:
        raise ValueError('Expected finite mono PCM bounded to [-1, 1]')
    return np.ascontiguousarray(pcm)
