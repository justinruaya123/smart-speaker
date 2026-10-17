import importlib
import numpy as np
import pytest
import torch


def test_wake_plan_separates_voices_and_includes_confusable_negatives():
    m=importlib.import_module('scripts.train.mvp_wake')
    voices=[{'id':'train-a','split':'train'},{'id':'test-a','split':'test'}]
    plan=m.generation_plan(voices,[])
    assert len({r['id'] for r in plan})==len(plan)
    assert all(r['speaker']==r['voice_id'] for r in plan)
    assert any(r['text']=='Hey Iris.' and r['wake_positive'] for r in plan)
    for text in ('Hey Siri.','Hey Irish.','Iris.','Hey.'):
        assert any(r['text']==text and not r['wake_positive'] for r in plan)
    assert {r['split'] for r in plan if r['voice_id']=='test-a'}=={'test'}


def test_wake_window_never_truncates_a_positive_phrase():
    m=importlib.import_module('scripts.train.mvp_wake')
    pcm=np.ones(30000,dtype='float32')*.1
    with pytest.raises(ValueError,match='long'):m.window(pcm,True,np.random.default_rng(53))
    pcm=np.ones(10000,dtype='float32')*.1
    result=m.window(pcm,True,np.random.default_rng(53))
    assert result.shape==(25600,) and np.count_nonzero(result)==10000


def test_wake_network_is_small_and_has_no_imported_weights():
    m=importlib.import_module('scripts.models.mvp_wake')
    torch.manual_seed(61);a=m.WakeNet().eval()
    torch.manual_seed(62);b=m.WakeNet().eval()
    assert 30000<sum(p.numel() for p in a.parameters())<120000
    x=torch.zeros(2,40,158)
    with torch.no_grad():assert a(x).shape==(2,2)
    assert not torch.equal(a.classifier.weight,b.classifier.weight)


def test_streaming_wake_requires_consecutive_scores_and_rearms_after_release():
    m=importlib.import_module('scripts.runtime.mvp_wake')
    scores=iter([.99,.1,.99,.99]+[.99]*30+[.1]*25+[.99,.99])
    stream=m.RollingWake(lambda pcm:next(scores),threshold=.9,release=.3,consecutive=2,cooldown_samples=16000)
    stream.push(np.zeros(24320,dtype='float32'))
    outputs=[stream.push(np.zeros(1280,dtype='float32')) for _ in range(61)]
    assert sum(x==1. for x in outputs)==2
    assert len(stream.audio)<=25600 and len(stream.pending)<1280
    stream.reset()
    assert len(stream.audio)==0


def test_generation_seed_accepts_hex_and_receipt_hashes_saved_pcm(tmp_path):
    import hashlib
    import soundfile as sf
    m=importlib.import_module('scripts.train.mvp_wake')
    assert m.generation_seed('train-en',0)==0x7204eb18
    path=tmp_path/'clip.wav';pcm=np.full(16000,.123456,dtype='float32')
    receipt=m.save_waveform(path,pcm)
    saved,rate=sf.read(path,dtype='float32')
    assert rate==16000
    assert receipt['pcm_sha256']==hashlib.sha256(saved.tobytes()).hexdigest()
    assert receipt['pcm_sha256']!=hashlib.sha256(pcm.tobytes()).hexdigest()


def test_wake_export_preserves_ort_ncnn_logits(tmp_path):
    from scripts.models.mvp_wake import WakeNet
    m=importlib.import_module('scripts.train.mvp_wake')
    receipt=m.export_engines(WakeNet().eval(),tmp_path)
    assert receipt['max_absolute_error']<2e-5
    assert set(receipt['files'])=={'iris.onnx','iris.ncnn.param','iris.ncnn.bin'}
