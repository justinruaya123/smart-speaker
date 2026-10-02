"""Real audio boundary and state contracts; use a recognizer double only for routing."""
import io
import numpy as np
import pytest
import soundfile as sf


def wav(pcm,rate=16000):
    stream=io.BytesIO();sf.write(stream,pcm,rate,format='WAV',subtype='PCM_16');return stream.getvalue()


def test_wav_rejects_stereo_wrong_rate_and_overlong():
    from scripts.runtime.speaker_worker import decode_wav
    for data in (wav(np.zeros((400,2))),wav(np.zeros(400),8000),wav(np.zeros(16*16000))):
        with pytest.raises(ValueError):decode_wav(data,wake=False)


def test_wav_keeps_pcm_and_has_no_microphone_side_effect():
    from scripts.runtime.speaker_worker import AudioService
    class Recognizer:
        recipe={'binding':{'model_sha256':'test'}};policy={'enabled':False};load_ms=0
        def recognize(self,pcm):
            assert len(pcm)==1600
            return {'event':'command','output':{'schema_version':3,'accepted':False,'command':None,'reason':'unqualified_model'},'diagnostic_candidate':{'candidate':{'intent':'TIME','slots':{}}}}
    events=[];service=AudioService(Recognizer(),events.append)
    result=service.wav(wav(np.ones(1600,dtype='float32')*.1))
    assert [e['state'] for e in events if e.get('event')=='state']==['processing','awaiting_microphone']
    command=next(e for e in events if e.get('event')=='command')
    assert command['output']['accepted'] is False
    assert command['diagnostic_candidate']['candidate']['intent']=='TIME'
    assert command['source']=='wav' and command['actions_enabled'] is False
    assert result['commands']==1
