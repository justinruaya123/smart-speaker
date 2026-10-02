import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wake_gate import WakeGate


class Detector:
    def reset(self):pass
    def push(self,pcm):return 1. if np.all(pcm==.25) else 0.


class Voice:
    def reset(self):pass
    def __call__(self,pcm):return 1. if np.max(np.abs(pcm))>.05 else 0.


class Command:
    def __init__(self):self.received=[];self.reset()
    def reset(self):self.finalized=False;self.local_sequence=0;self.blocks=0
    def feed(self,pcm,sequence):
        assert sequence==self.local_sequence
        self.local_sequence+=len(pcm);self.received.append(pcm.copy());self.blocks+=1
        self.finalized=self.blocks==3
        return {'accepted':self.finalized,'command':{'intent':'POWER_ON'} if self.finalized else None}


def push(gate,blocks,sequence=0):
    events=[]
    for value,n in blocks:
        for _ in range(n):
            events.extend(gate.feed(np.full(512,value,dtype=np.float32),sequence));sequence+=512
    return events,sequence


def test_no_command_inference_without_wake_and_one_wake_arms_only_one_command():
    model=Command();gate=WakeGate(model,Detector(),Voice(),release_ms=64)
    events,seq=push(gate,[(.5,10)])
    assert not model.received and not any(e.get('accepted') for e in events)
    events,seq=push(gate,[(.25,1),(0,2),(.5,8)],seq)
    assert sum(e.get('accepted',False) for e in events)==1
    assert not any(np.all(x==.25) for x in model.received)
    before=len(model.received)
    more,seq=push(gate,[(.5,8)],seq)
    assert len(model.received)==before and not any(e.get('accepted') for e in more)
    events,seq=push(gate,[(.25,1),(0,2),(.5,3)],seq)
    assert sum(e.get('accepted',False) for e in events)==1


def test_missing_release_pause_and_command_start_timeout_rearm_without_inference():
    model=Command();gate=WakeGate(model,Detector(),Voice(),release_ms=64,release_timeout_seconds=.2,command_timeout_seconds=.2)
    events,seq=push(gate,[(.25,1),(.5,8)])
    assert any(e.get('reason')=='wake_word_not_separated' for e in events)
    assert not model.received and gate.state=='wake'
    events,seq=push(gate,[(.25,1),(0,12)],seq)
    assert any(e.get('reason')=='command_start_timeout' for e in events)
    assert not model.received and gate.state=='wake'


@pytest.mark.parametrize('fault',['gap','overflow','nan'])
def test_capture_fault_cannot_bypass_wake(fault):
    model=Command();gate=WakeGate(model,Detector(),Voice(),release_ms=64)
    _,seq=push(gate,[(.25,1),(0,2)])
    pcm=np.full(512,np.nan if fault=='nan' else .5,dtype=np.float32)
    events=gate.feed(pcm,seq+1 if fault=='gap' else seq,overflow=fault=='overflow')
    assert events[-1]['reason']=='capture_fault' and events[-1]['accepted'] is False
    assert gate.state=='wake' and not model.received


def test_real_local_wake_models_load_and_silence_never_arms_command():
    from scripts.runtime.wake_models import HeyJarvis,SileroProbability
    root=Path(__file__).resolve().parents[3]
    model=Command();gate=WakeGate(model,HeyJarvis(root/'artifacts'),SileroProbability(root/'artifacts'))
    events,_=push(gate,[(0.,75)])
    assert gate.state=='wake' and not model.received
    assert not any(e.get('accepted') for e in events)


def test_file_replay_and_benchmark_share_warmup_flush_and_wake_gating():
    from wake_gate import replay_pcm
    from einops import pack
    model=Command();gate=WakeGate(model,Detector(),Voice(),release_ms=64)
    assert not any(e.get('accepted') for e in replay_pcm(gate,np.full(4096,.5,dtype=np.float32)))
    assert not model.received
    pcm,_=pack([np.full(1024,.25,dtype=np.float32),np.zeros(2048,dtype=np.float32),
        np.full(4096,.5,dtype=np.float32)],'*')
    assert sum(e.get('accepted',False) for e in replay_pcm(gate,pcm))==1
    assert gate.state=='wake'


@pytest.mark.parametrize('value',[-.1,1.1,float('nan')])
def test_invalid_vad_score_rearms_without_command(value):
    class InvalidVoice:
        def reset(self):pass
        def __call__(self,_pcm):return value
    model=Command();gate=WakeGate(model,Detector(),InvalidVoice())
    events,_=push(gate,[(.25,1)])
    assert events[-1]['reason']=='capture_fault' and gate.state=='wake' and not model.received
