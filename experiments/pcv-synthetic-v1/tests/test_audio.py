import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthesis import audit_waveform, persist_audio
from synthesis import dia_text


def test_silent_nonfinite_and_clipped_generation_is_rejected():
    for x in (np.zeros(16000), np.full(16000, np.nan), np.ones(16000),
              2e-4*np.sin(np.arange(16000)*.03)):
        with pytest.raises(ValueError):
            audit_waveform(x, 16000)


def test_native_and_derived_receipts_are_hash_bound_and_immutable(tmp_path):
    x = .1 * np.sin(np.arange(24000) * .03)
    row = dict(recording_id='qwen/train/x', audio='audio/a.wav', native_audio='native/a.wav')
    result = persist_audio(tmp_path, row, x, 24000)
    assert result['sample_rate'] == 16000
    assert result['synthetic'] is True
    assert result['text_verified_by_listener'] is False
    assert result['audio_sha256'] and result['native_sha256'] and result['pcm_sha256']
    with pytest.raises(FileExistsError):
        persist_audio(tmp_path, row, x, 24000)


def test_dia_sentence_closure_preserves_every_spoken_word():
    assert dia_text('play focus playlist')=='play focus playlist.'
    assert dia_text('what time is it?')=='what time is it?'


def test_invalid_resampled_audio_leaves_no_orphan_native_file(tmp_path):
    x=.1*np.sin(2*np.pi*11000*np.arange(24000)/24000)
    row=dict(audio='audio/a.wav',native_audio='native/a.wav')
    with pytest.raises(ValueError):persist_audio(tmp_path,row,x,24000)
    assert not (tmp_path/'native/a.wav').exists()
    assert not (tmp_path/'audio/a.wav').exists()
