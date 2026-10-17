"""Real preview scheduling and intersection of recognition with ownership."""
import threading
from types import SimpleNamespace

import numpy as np

from scripts.runtime.speaker_worker import AudioService


def test_preview_inflight_is_discarded_after_finalization():
    from scripts.runtime.speaker_worker import LatestPreview
    started=threading.Event();release=threading.Event();finished=threading.Event();emitted=[]
    def infer(pcm):
        started.set();assert release.wait(2);finished.set();return {'candidate':len(pcm)}
    worker=LatestPreview(infer,emitted.append)
    worker.submit(np.ones(8000,dtype='float32'),'old')
    assert started.wait(2)
    worker.clear();release.set();worker.pending.join();assert finished.is_set()
    assert emitted==[]
    worker.close()


def test_preview_keeps_only_latest_pending_audio():
    from scripts.runtime.speaker_worker import LatestPreview
    started=threading.Event();release=threading.Event();seen=[];emitted=[]
    def infer(pcm):
        seen.append(len(pcm))
        if len(seen)==1:started.set();assert release.wait(2)
        return {'candidate':len(pcm)}
    worker=LatestPreview(infer,emitted.append)
    worker.submit(np.ones(8000,dtype='float32'),'u');assert started.wait(2)
    worker.submit(np.ones(16000,dtype='float32'),'u')
    worker.submit(np.ones(24000,dtype='float32'),'u');release.set();worker.pending.join()
    assert seen==[8000,24000]
    assert all(e['event']=='tentative' and e['is_final'] is False for e in emitted)
    worker.close()


def test_final_owner_rejection_cannot_be_rescued_by_atomic_policy():
    from scripts.runtime.speaker_session import SpeakerEvidence,SpeakerPolicy
    command={'intent':'TIME','slots':{}}
    runtime=SimpleNamespace(root=None,profile='benchmark-compatible',decision_authority='atomic_command',
        recognize=lambda pcm:{'event':'command','output':{'accepted':True,'command':command,'reason':None}})
    evidence=lambda vector:SpeakerEvidence(vector,'a'*64,1.,False)
    backend=SimpleNamespace(policy=SpeakerPolicy(.8),evidence=lambda pcm:evidence([0.,1.]),
        continuity=lambda pcm:[evidence([0.,1.])])
    events=[];settings={'inference':{'mode':'demo'},'actions':{'enabled':True}}
    service=AudioService(runtime,events.append,settings,owner=backend,profile='owner-protected')
    service.arm_owner(np.ones(16000,dtype='float32'),reference=evidence([1.,0.]))
    result=service.recognize(np.ones(16000,dtype='float32'));service.forward(result)
    event=next(e for e in events if e.get('event')=='command')
    assert event['application']['accepted'] is False
    assert event['application']['reason']=='speaker_mismatch'
    assert event['owner_gating_enabled'] is True
    assert event['application']['command'] is None


def test_new_entity_maps_and_explicit_light_groups_reach_gateway(tmp_path):
    from scripts.runtime.speaker_config import resolve_config
    import json
    bindings={'lights':{'light.all':['light.a','light.b']},'climate':{'default':'climate.home'},
        'weather':{'default':'weather.home'},'media_player':{'provider':'home_assistant','entity_id':'media_player.home'},
        'contacts':{'contact.mom':'script.call_mom'},'messages':{'default':'script.message'}}
    cfg,env=resolve_config(tmp_path,environ={'SMART_SPEAKER_BINDINGS':json.dumps(bindings),'RECOGNITION_PROFILE':'owner-protected'})
    assert cfg['inference']['profile']=='owner-protected'
    assert json.loads(env['SMART_SPEAKER_BINDINGS'])['lights']['light.all']==['light.a','light.b']


def test_launcher_exposes_explicit_benchmark_profile():
    from scripts.runtime.speaker_start import make_parser
    args=make_parser(True).parse_args(['--profile','benchmark-compatible','--wav','command.wav'])
    assert args.profile=='benchmark-compatible'


def test_large_audio_chunk_arms_owner_before_command_inference():
    from scripts.runtime.wake_gate import WakeGate
    order=[]
    class Command:
        def reset(self):self.finalized=False
        def feed(self,pcm,sequence):
            order.append('recognize');self.finalized=True
            return {'output':{'accepted':False,'command':None,'reason':'OOV'}}
    class Wake:
        audio=np.ones(25600,dtype='float32')
        def reset(self):pass
        def push(self,pcm):return 1.
    class Voice:
        def __init__(self):self.values=iter([1.,0.,1.])
        def reset(self):pass
        def __call__(self,pcm):return next(self.values)
    gate=WakeGate(Command(),Wake(),Voice(),release_ms=32)
    gate.on_wake=lambda pcm:order.append('arm')
    gate.feed(np.ones(1536,dtype='float32')*.1,0)
    assert order==['arm','recognize']


def test_uploaded_wav_is_declared_action_free_command_preview():
    import io
    import soundfile as sf
    from scripts.runtime.speaker_session import SpeakerPolicy
    command={'intent':'TIME','slots':{}}
    runtime=SimpleNamespace(root=None,profile='benchmark-compatible',decision_authority='atomic_command',
        recognize=lambda pcm:{'event':'command','output':{'accepted':True,'command':command,'reason':None}})
    backend=SimpleNamespace(policy=SpeakerPolicy(.8))
    events=[];settings={'inference':{'mode':'demo'},'actions':{'enabled':True}}
    service=AudioService(runtime,events.append,settings,owner=backend,profile='owner-protected')
    stream=io.BytesIO();sf.write(stream,np.ones(16000,dtype='float32')*.1,16000,format='WAV',subtype='PCM_16')
    service.wav(stream.getvalue())
    final=next(e for e in events if e.get('event')=='command')
    assert final['application']['accepted'] is True
    assert final['source']=='wav' and final['profile']=='benchmark-compatible'
    assert final['actions_enabled'] is False and final['owner_gating_enabled'] is False
    assert service.profile=='owner-protected' and service.owner_session is not None


def test_preview_emissions_are_spaced_even_after_slow_inference():
    import time
    from scripts.runtime.speaker_worker import LatestPreview
    started=threading.Event();release=threading.Event();emitted=[]
    def infer(pcm):
        if len(pcm)==1:started.set();assert release.wait(2)
        return {'candidate':len(pcm)}
    worker=LatestPreview(infer,lambda event:emitted.append(time.monotonic()))
    worker.submit(np.ones(1,dtype='float32'),'u');assert started.wait(2)
    worker.submit(np.ones(2,dtype='float32'),'u')
    time.sleep(.55);release.set();worker.pending.join();worker.close()
    assert len(emitted)==2 and emitted[1]-emitted[0]>=.48


def test_active_command_capture_cannot_wait_forever_for_second_vad():
    from scripts.runtime.wake_gate import WakeGate
    class Command:
        def reset(self):self.finalized=False
        def feed(self,pcm,sequence):return {'reason':None}
    class Wake:
        def reset(self):pass
        def push(self,pcm):return 1.
    class Voice:
        def reset(self):self.count=0
        def __call__(self,pcm):self.count+=1;return 0. if self.count==2 else 1.
    gate=WakeGate(Command(),Wake(),Voice(),release_ms=32)
    events=gate.feed(np.ones(16000*16,dtype='float32')*.1,0)
    assert any(e.get('reason')=='command_capture_timeout' for e in events)
