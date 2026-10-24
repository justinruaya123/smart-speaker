"""Training data export retains the original bounded WAV/PCM decoder exactly."""
import ast
import hashlib
from pathlib import Path
import wave

import numpy as np
import pytest

# Derived from actual 94c0021d7e22c9aa719246ac97105d7140a9f2a2 source ASTs.
EXPECTED_AST = {'_wav_info': '42fcaf2a5044eeee42c0a2e1c7fae54488f609c066acdfe1efc8b9746a90d722', 'read_pcm': '40b14b61e53589b7b6509ea58832eaab7cd632d2893a0db2cdf49d9c647dc62d'}


def test_model_audio_decoder_is_the_unchanged_frozen_reference():
    from scripts.train import evaluation_audio as module
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf8'))
    actual = {node.name:hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
              for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in EXPECTED_AST}
    assert actual == EXPECTED_AST
    assert module.SAMPLE_RATE == 16000 and module.MAX_SAMPLES == 240000


def test_model_audio_export_keeps_pcm_identity_and_rejects_changed_bytes(tmp_path):
    from scripts.train.evaluation_audio import read_pcm
    path=tmp_path/'original.wav'
    samples=np.asarray([-32768,0,8192,32767],dtype='<i2')
    with wave.open(str(path),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(16000)
        stream.writeframes(samples.tobytes())
    pin=hashlib.sha256(path.read_bytes()).hexdigest()
    actual=read_pcm(path,pin)
    assert actual.dtype==np.dtype('float32') and actual.flags.c_contiguous
    assert np.array_equal(actual,samples.astype(np.float32)/32768)
    with pytest.raises(ValueError,match='identity'):
        read_pcm(path,'0'*64)
