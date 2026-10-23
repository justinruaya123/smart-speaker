"""Closed cue IDs, host ownership and playback lifecycle (no real devices)."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.runtime.speaker_worker import AudioService
from scripts.runtime.speaker_session import SpeakerEvidence, SpeakerPolicy

ROOT=Path(__file__).resolve().parents[2]


def feedback_module():
    from scripts.runtime import iris_feedback
    return iris_feedback


def service_fixture(playback,*,enabled=True):
    module=feedback_module()
    player=module.FeedbackPlayer(ROOT/'submission/feedback',speaker_index=7,enabled=enabled,playback=playback)
    settings={'inference':{'mode':'demo'},'actions':{'enabled':True}}
    evidence=SpeakerEvidence([1.,0.],'a'*64,1.,False)
    owner=SimpleNamespace(policy=SpeakerPolicy(.8),evidence=lambda pcm:evidence,continuity=lambda pcm:[evidence])
    runtime=SimpleNamespace(root=ROOT,profile='owner-protected',recognize=lambda pcm:{
        'event':'command','output':{'schema_version':3,'accepted':True,'command':{'intent':'TIME','slots':{}},'reason':None},
        'application':{'mode':'joint_complete_command'}})
    events=[]
    service=AudioService(runtime,events.append,settings,owner=owner,profile='owner-protected',feedback=player)
    order=[]
    service.stream=SimpleNamespace(abort=lambda:order.append('abort'),start=lambda:order.append('start'),close=lambda:order.append('close'))
    service.gate=SimpleNamespace(state='wake',reset=lambda:order.append('reset'))
    return service,events,order


def test_only_closed_ids_and_verified_predecoded_assets_can_play(tmp_path):
    module=feedback_module();played=[]
    player=module.FeedbackPlayer(ROOT/'submission/feedback',speaker_index=7,enabled=True,
        playback=lambda pcm,rate,speaker:played.append((pcm.copy(),rate,speaker)))
    for invalid in ('../wake_yes.wav','https://example.com/cue','Hello','intent;whoami','',None):
        with pytest.raises(ValueError):player.play(invalid)
    assert played==[]
    result=player.play('intent')
    assert result['played'] is True
    assert len(played)==1 and played[0][1:]==(16000,7)
    assert np.isfinite(played[0][0]).all()
    copied=tmp_path/'feedback';shutil.copytree(ROOT/'submission/feedback',copied)
    (copied/'intent.wav').write_bytes(b'altered')
    with pytest.raises(ValueError,match='identity|hash'):module.FeedbackPlayer(copied,speaker_index=7)


def test_feedback_requires_known_final_live_utterance_intent_then_one_completion():
    played=[];service,events,order=service_fixture(lambda *a:played.append('played'))
    with pytest.raises(ValueError,match='utterance'):service.feedback_control('intent','forged')
    service.utterance='u1'
    service.forward(service.runtime.recognize(np.ones(16000)*.1))
    with pytest.raises(ValueError,match='intent'):service.feedback_control('success','u1')
    assert service.feedback_control('intent','u1')['played'] is True
    assert service.feedback_control('intent','u1')['duplicate'] is True
    assert service.feedback_control('success','u1')['played'] is True
    assert service.feedback_control('success','u1')['duplicate'] is True
    with pytest.raises(ValueError,match='completion'):service.feedback_control('failure','u1')
    assert len(played)==2
    assert order==['abort','start','abort','start']
    with pytest.raises(ValueError,match='session'):service.feedback_control('intent','u1',session_id='other')


def test_wake_ack_rotates_without_resetting_wake_owner_or_sequence():
    played=[]
    service,events,order=service_fixture(lambda pcm,rate,speaker:played.append(len(pcm)))
    service.offset=5120;service.gate.state='release'
    for _ in range(5):
        service.arm_owner(np.ones(16000)*.1)
        assert service.owner_session._owner is not None
        assert service.gate.state=='release' and service.offset==5120
        result=service.recognize(np.ones(16000)*.1)
        assert result['owner_result']['accepted'] is True
        assert result['output']['accepted'] is True
    cues=[e['cue_id'] for e in events if e.get('event')=='feedback']
    assert cues==['wake_yes','wake_hmm','wake_yeah','wake_here','wake_yes']
    assert 'reset' not in order and order==['abort','start']*5


def test_playback_failure_restores_capture_and_preserves_owner():
    def failure(*a):raise RuntimeError('output disconnected')
    service,events,order=service_fixture(failure)
    service.gate.state='release'
    service.arm_owner(np.ones(16000)*.1)
    assert service.owner_session._owner is not None
    assert order==['abort','start'] and service.gate.state=='release'
    assert next(e for e in events if e['event']=='feedback')['played'] is False
    result=service.recognize(np.ones(16000)*.1)
    assert result['owner_result']['accepted'] is True and result['output']['accepted'] is True


def test_wav_disabled_and_busy_capture_never_play():
    played=[];service,events,order=service_fixture(lambda *a:played.append('bad'),enabled=False)
    service.arm_owner(np.ones(16000)*.1)
    assert played==[] and order==[]
    service,events,order=service_fixture(lambda *a:played.append('bad'))
    service.source='wav';service.arm_owner(np.ones(16000)*.1)
    assert played==[] and order==[]
    service.source='microphone';service.utterance='old'
    service.forward(service.runtime.recognize(np.ones(16000)*.1))
    service.gate.state='command'
    with pytest.raises(ValueError,match='capture'):service.feedback_control('intent','old')
    assert played==[]


def test_resume_failure_stops_capture_fail_closed():
    service,events,order=service_fixture(lambda *a:None)
    service.stream.start=lambda:(_ for _ in ()).throw(RuntimeError('cannot restart'))
    service.gate.state='release'
    service.arm_owner(np.ones(16000)*.1)
    assert service.stream is None
    assert service.owner_session._owner is None
    assert service.state=='capture_error'
    assert 'close' in order


def test_stop_resets_owner_and_closes_capture_even_when_abort_raises():
    service,events,order=service_fixture(lambda *a:None,enabled=False)
    service.arm_owner(np.ones(16000)*.1)
    service.stream.abort=lambda:(_ for _ in ()).throw(RuntimeError('device disconnected'))
    with pytest.raises(RuntimeError,match='disconnected'):service.stop(announce=False)
    assert service.stream is None and service.owner_session._owner is None
    assert order==['close','reset']


def test_native_playback_drains_and_closes_its_own_output_before_return(monkeypatch):
    import sys
    module=feedback_module();order=[]
    class Stream:
        def __init__(self,**kwargs):
            assert kwargs['device']==7 and kwargs['channels']==1
        def start(self):order.append('start')
        def write(self,pcm):
            assert pcm.shape==(800,1);order.append('write')
        def stop(self):order.append('drained')
        def abort(self):order.append('abort')
        def close(self):order.append('closed')
    monkeypatch.setitem(sys.modules,'sounddevice',SimpleNamespace(OutputStream=Stream))
    module.SounddevicePlayback()(np.zeros(800,dtype='float32'),16000,7)
    assert order==['start','write','drained','closed']


def test_failed_wake_capture_resume_cannot_claim_listening_again():
    service,events,order=service_fixture(lambda *a:None)
    service.stream.start=lambda:(_ for _ in ()).throw(RuntimeError('cannot restart'))
    service.gate.state='release'
    service.arm_owner(np.ones(16000)*.1)
    service.forward({'stage':'wake','status':'detected','wake_phrase':'hey iris'})
    assert service.stream is None and service.state=='capture_error'
