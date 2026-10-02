import base64
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from scripts.runtime.pcv_deploy.stream import StreamManager


class Gate:
    def __init__(self,label):self.label=label;self.sequence=0
    def reset(self):self.sequence=0
    def feed(self,pcm,sequence,overflow=False):
        if sequence!=self.sequence or overflow:return [dict(accepted=False,reason='capture_fault')]
        self.sequence+=len(pcm);return [dict(stage='command',accepted=True,command={'intent':self.label})]
    def eof(self):return dict(accepted=False,command=None,reason='incomplete')


def snapshot(name):return dict(binding=name,gate=Gate(name),diagnostic=False)
def audio(sequence=0):return dict(type='audio',sequence=sequence,pcm_s16le=base64.b64encode(b'\0\0'*640).decode())


def test_session_pins_bundle_during_validated_reload_and_new_stream_uses_new():
    manager=StreamManager(snapshot('old'))
    first=manager.start('one');assert first['binding']=='old'
    manager.stage(snapshot('new'))
    assert manager.audio('one',audio())['events'][0]['command']['intent']=='old'
    manager.end('one')
    assert manager.start('two')['binding']=='new'
    assert manager.audio('two',audio())['events'][0]['command']['intent']=='new'


def test_single_stream_limit_and_disconnect_releases_slot():
    manager=StreamManager(snapshot('one'));manager.start('a')
    with pytest.raises(ValueError,match='busy'):manager.start('b')
    manager.cancel('a');assert manager.start('b')['binding']=='one'


@pytest.mark.parametrize('message',[dict(type='audio',sequence=True,pcm_s16le='AAAA'),
    dict(type='audio',sequence=0,pcm_s16le='not base64'),
    dict(type='audio',sequence=0,pcm_s16le=base64.b64encode(b'\0'*3202).decode()),
    audio(1)])
def test_invalid_audio_cancels_stream_without_acceptance(message):
    manager=StreamManager(snapshot('one'));manager.start('a')
    output=manager.audio('a',message)
    assert output['events']==[dict(stage='stream',accepted=False,command=None,reason='capture_fault')]
    assert manager.active is None


def test_stale_stream_id_cannot_inject_or_cancel_another_stream():
    manager=StreamManager(snapshot('one'));manager.start('a')
    with pytest.raises(ValueError):manager.audio('wrong',audio())
    manager.cancel('wrong');assert manager.active is not None
