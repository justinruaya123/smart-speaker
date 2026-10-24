"""Resident CPU inference worker. JSONL controls/events; never executes actions."""
import argparse
import base64
from collections import OrderedDict
from copy import deepcopy
import io
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

# Bound numeric-library defaults before the first NumPy/frontend import.
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np
from einops import rearrange
import soundfile as sf

MAX_WAV_BYTES=4*1024*1024


class LatestPreview:
    """One in-flight computation and one replaceable pending prefix."""
    def __init__(self,infer,emit):
        self.infer=infer;self.deliver=emit;self.pending=queue.Queue(maxsize=1)
        self.lock=threading.RLock();self.generation=0;self.closed=False
        self.last_started=self.last_delivered=-float('inf')
        threading.Thread(target=self._run,daemon=True).start()

    def submit(self,pcm,utterance_id):
        with self.lock:
            if self.closed:return
            job=(self.generation,np.asarray(pcm,dtype='float32').copy(),utterance_id)
            try:self.pending.put_nowait(job)
            except queue.Full:
                try:self.pending.get_nowait();self.pending.task_done()
                except queue.Empty:pass
                self.pending.put_nowait(job)

    def clear(self):
        with self.lock:
            self.generation+=1
            try:self.pending.get_nowait();self.pending.task_done()
            except queue.Empty:pass

    def close(self):
        self.clear()
        with self.lock:self.closed=True

    def _run(self):
        while True:
            try:token,pcm,utterance=self.pending.get(timeout=.5)
            except queue.Empty:
                if self.closed:return
                continue
            try:
                delay=.5-(time.monotonic()-self.last_started)
                if delay>0:time.sleep(delay)
                self.last_started=time.monotonic()
                value=self.infer(pcm)
                delay=.5-(time.monotonic()-self.last_delivered)
                if delay>0:time.sleep(delay)
                with self.lock:
                    if token==self.generation and not self.closed:
                        self.deliver({**value,'event':'tentative','utterance_id':utterance,'is_final':False})
                        self.last_delivered=time.monotonic()
            except Exception:
                # A preview never changes the authoritative final path.
                pass
            finally:self.pending.task_done()


def decode_wav(data, *, wake=False):
    if not isinstance(data,bytes) or not 44<=len(data)<=MAX_WAV_BYTES or data[:4]!=b'RIFF' or data[8:12]!=b'WAVE':
        raise ValueError('Use a WAV file, maximum 4 MiB')
    with sf.SoundFile(io.BytesIO(data)) as file:
        if file.samplerate!=16000 or file.channels!=1 or not 1<=len(file)<=16000*(60 if wake else 15):
            raise ValueError('Requires mono 16 kHz WAV; command <=15 seconds, wake recording <=60 seconds')
        pcm=file.read(dtype='float32')
    if not np.isfinite(pcm).all() or np.max(np.abs(pcm))>1:raise ValueError('WAV contains invalid PCM')
    return pcm


class AudioService:
    def __init__(self,runtime,emit,settings=None,*,owner=None,profile=None,feedback=None):
        self.runtime=runtime;self.deliver=emit;self.sequence=0;self.session=uuid.uuid4().hex
        self.state='awaiting_microphone';self.stream=None;self.gate=None;self.offset=0;self.source='microphone'
        self.last_signal=time.monotonic();self.warned_silent=False
        self.settings=settings
        self.feedback=feedback;self.feedback_utterances=OrderedDict();self.wake_feedback_index=0
        self.profile=profile or getattr(runtime,'profile','benchmark-compatible')
        self.owner=owner;self.owner_session=None;self.utterance=uuid.uuid4().hex
        self.inference_lock=threading.Lock();self.emission_lock=threading.RLock();self.previews=None
        if self.profile=='owner-protected':
            if owner is None:raise ValueError('Owner-protected profile requires the speaker backend')
            from scripts.runtime.speaker_session import SpeakerSession
            self.owner_session=SpeakerSession(owner.policy)

    def emit(self,value):
        with self.emission_lock:
            self.sequence+=1
            if value.get('event')=='command' and self.settings:
                mode=value.get('application',{}).get('mode',self.settings['inference']['mode'])
                value={**value,'application':{'mode':mode,**value['output']}}
            self.deliver({**value,'session_id':self.session,'sequence':self.sequence,'source':self.source,
                'profile':self.profile,'owner_gating_enabled':self.owner_session is not None,
                'ownership_experimental':self.owner_session is not None,
                'actions_enabled':bool(self.settings and self.settings['actions']['enabled'] and
                    self.source=='microphone' and self.profile=='owner-protected')})

    def set_state(self,state,**extra):
        self.state=state;self.emit({'event':'state','state':state,**extra})

    def recognize(self,pcm):
        if self.previews:self.previews.clear()
        self.set_state('processing')
        with self.inference_lock:event=self.runtime.recognize(pcm)
        event={**event,'model_output':deepcopy(event.get('model_output',event['output']))}
        if self.owner_session:
            from scripts.runtime.speaker_session import gate_application
            verdict=self.owner_session.finish(self.utterance,self.owner.evidence(pcm),
                now=time.monotonic(),windows=self.owner.continuity(pcm))
            event={**event,'output':gate_application(event['output'],verdict),
                'owner_result':{'accepted':verdict.accepted,'reason':verdict.reason}}
        return event

    def arm_owner(self,pcm,*,reference=None):
        if self.previews:self.previews.clear()
        self.utterance=uuid.uuid4().hex
        if self.owner_session:
            verdict=self.owner_session.arm(self.utterance,reference or self.owner.evidence(pcm),now=time.monotonic())
            self.emit({'event':'ownership','accepted':verdict.accepted,'reason':verdict.reason})
        if self.feedback and self.feedback.enabled and self.stream is not None and self.source=='microphone' and self.profile=='owner-protected':
            from scripts.runtime.iris_feedback import WAKE_CUES
            cue=WAKE_CUES[self.wake_feedback_index%len(WAKE_CUES)];self.wake_feedback_index+=1
            try:self.play_feedback(cue,self.utterance)
            except Exception as error:
                self.emit({'event':'feedback','cue_id':cue,'utterance_id':self.utterance,
                           'played':False,'reason':'playback_error','message':str(error)[:200]})

    def play_feedback(self,cue_id,utterance_id):
        if not self.feedback or not self.feedback.enabled or self.source!='microphone' or self.stream is None or self.profile!='owner-protected':
            return {'cue_id':cue_id,'played':False,'reason':'feedback_disabled'}
        stream=self.stream
        try:stream.abort()  # Suspend/discard input without resetting gate or wake ownership.
        except Exception:
            try:self.stop(announce=False)
            finally:self.set_state('capture_error',message='Unable to suspend capture for feedback')
            raise
        try:
            result=self.feedback.play(cue_id)
        finally:
            # Keep capture off through a short room/driver tail; never feed our cue to Iris.
            time.sleep(.08)
            try:stream.start()
            except Exception:
                try:self.stop(announce=False)
                finally:self.set_state('capture_error',message='Capture could not resume after feedback')
                raise
        self.emit({'event':'feedback','utterance_id':utterance_id,**result})
        return result

    def feedback_control(self,cue_id,utterance_id,*,session_id=None):
        from scripts.runtime.iris_feedback import CONTROL_CUES
        if session_id is not None and session_id!=self.session:raise ValueError('Feedback session mismatch')
        if not isinstance(cue_id,str) or cue_id not in CONTROL_CUES:raise ValueError('Unknown closed feedback cue ID')
        if (not isinstance(utterance_id,str) or not 1<=len(utterance_id)<=128
                or any(ord(c)<32 for c in utterance_id) or utterance_id not in self.feedback_utterances):
            raise ValueError('Feedback requires a known final microphone utterance')
        record=self.feedback_utterances[utterance_id]
        if record['source']!='microphone':raise ValueError('Feedback cannot play for WAV/replay utterances')
        if cue_id=='intent':
            if not record['accepted']:raise ValueError('Intent feedback requires final accepted recognition')
            if record['intent']:return {'cue_id':cue_id,'played':False,'duplicate':True}
            if record['completion'] is not None:raise ValueError('Feedback completion already consumed')
        else:
            if record['completion'] is not None:
                if record['completion']!=cue_id:raise ValueError('Feedback completion already consumed')
                return {'cue_id':cue_id,'played':False,'duplicate':True}
            if record['accepted'] and not record['intent']:raise ValueError('Completion feedback requires intent phase first')
            if cue_id=='success' and not record['accepted']:raise ValueError('Success requires final accepted recognition')
        if self.gate and self.gate.state!='wake':raise ValueError('Feedback deferred: active command capture must not be interrupted')
        # Consume even if output fails: no retry can replay a phase or affect an action.
        if cue_id=='intent':record['intent']=True
        else:record['completion']=cue_id
        return self.play_feedback(cue_id,utterance_id)

    def reset_command(self):
        if self.previews:self.previews.clear()
        if self.owner_session:self.owner_session.reset()

    def preview(self,pcm):
        if self.source!='microphone':return
        if self.previews is None:
            def infer(prefix):
                with self.inference_lock:event=self.runtime.recognize(prefix)
                diagnostic=event.get('diagnostic_candidate') or {}
                return {'candidate':diagnostic.get('candidate'),
                    'score':diagnostic.get('class_probability'),
                    'confidence_interpretation':'model score; tentative, not executed'}
            self.previews=LatestPreview(infer,self.emit)
        self.previews.submit(pcm,self.utterance)

    def new_gate(self):
        # A thin proxy adds actual processing state without changing the frozen recognizer.
        service=self
        class Observed:
            root=service.runtime.root
            def recognize(self,pcm):return service.recognize(pcm)
            def on_wake(self,pcm):service.arm_owner(pcm)
            def on_reset(self):service.reset_command()
            def preview(self,pcm):service.preview(pcm)
        return self.runtime.make_gate(Observed())

    def forward(self,event):
        if self.state=='capture_error' and self.source=='microphone' and self.stream is None and event.get('stage')=='wake':
            self.emit({'event':'capture_warning','reason':'capture_fault','message':'Wake invalidated by capture failure'})
            return
        if event.get('status')=='detected':self.set_state('wake_detected')
        elif event.get('status')=='command_ready':self.set_state('command_ready')
        if 'output' in event:
            if self.previews:self.previews.clear()
            self.feedback_utterances.setdefault(self.utterance,{'source':self.source,
                'accepted':event['output'].get('accepted') is True,'intent':False,'completion':None})
            while len(self.feedback_utterances)>128:self.feedback_utterances.popitem(last=False)
            self.emit({**event,'event':'command','utterance_id':self.utterance})
            self.utterance=uuid.uuid4().hex
            self.set_state('wake_listening' if self.stream else 'awaiting_microphone')
        else:
            self.emit(event)
            if event.get('reason'):
                self.reset_command()
                self.set_state('wake_listening' if self.stream else 'awaiting_microphone',reason=event['reason'])

    def wav(self,data,*,wake=False):
        pcm=decode_wav(data,wake=wake)
        self.stop(announce=False);self.source='wav';self.utterance=uuid.uuid4().hex;commands=0;wake_detected=False
        saved_profile,saved_owner=self.profile,self.owner_session
        if not wake:
            # A file operation is explicitly a command-only preview. Its events
            # declare the benchmark profile and can never authorize actions.
            self.profile='benchmark-compatible';self.owner_session=None
        try:
            if wake:
                from scripts.runtime.wake_gate import iter_replay_pcm
                gate=self.new_gate();self.set_state('wake_listening')
                for event in iter_replay_pcm(gate,pcm):
                    commands+=int('output' in event);wake_detected|=event.get('status')=='detected';self.forward(event)
                if not commands:self.emit({'event':'wake_result','wake_detected':wake_detected,**gate.eof()})
            else:
                self.forward(self.recognize(pcm));commands=1
            return {'commands':commands,'wake_detected':wake_detected,'audio_seconds':len(pcm)/16000}
        finally:
            if self.state!='awaiting_microphone':self.set_state('awaiting_microphone')
            self.profile,self.owner_session=saved_profile,saved_owner

    def devices(self):
        try:
            import sounddevice as sd
            hosts=sd.query_hostapis()
            return [{'index':i,'name':d['name'],'hostapi':hosts[d['hostapi']]['name'],'channels':d['max_input_channels'],
                     'default_sample_rate':d['default_samplerate']} for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]
        except Exception as error:
            raise RuntimeError('Microphone discovery failed. On Pi OS run start.py --system-packages --list-mics; check that the input is connected to this host. '+str(error)[:200]) from error

    def listen(self,index):
        if type(index) is not int or index<0:raise ValueError('Choose a listed microphone index')
        import sounddevice as sd
        devices=self.devices()
        selected=next((d for d in devices if d['index']==index),None)
        if selected is None:raise ValueError('Microphone no longer exists; refresh inputs')
        self.stop(announce=False);sd.check_input_settings(device=index,channels=1,dtype='float32',samplerate=16000)
        self.gate=self.new_gate();self.offset=0;self.source='microphone'
        self.stream=sd.InputStream(device=index,samplerate=16000,channels=1,dtype='float32',blocksize=512)
        try:self.stream.start()
        except Exception:self.stop(announce=False);raise
        self.last_signal=time.monotonic();self.warned_silent=False
        self.set_state('wake_listening',microphone=selected)
        return selected

    def output_devices(self):
        import sounddevice as sd
        hosts=sd.query_hostapis()
        return [{'index':i,'name':d['name'],'hostapi':hosts[d['hostapi']]['name'],
                 'channels':d['max_output_channels'],'default_sample_rate':d['default_samplerate']}
                for i,d in enumerate(sd.query_devices()) if d['max_output_channels']>0]

    def tick(self):
        pcm,overflow=self.stream.read(512);pcm=rearrange(pcm,'t 1 -> t')
        if np.max(np.abs(pcm))>1e-5:self.last_signal=time.monotonic();self.warned_silent=False
        elif not self.warned_silent and time.monotonic()-self.last_signal>5:
            self.emit({'event':'capture_warning','reason':'no_signal','message':'No microphone signal for 5 seconds'});self.warned_silent=True
        for event in self.gate.feed(pcm,self.offset,overflow=overflow):self.forward(event)
        if self.gate.state=='command' and self.gate.started and self.state!='recording':self.set_state('recording')
        self.offset+=len(pcm)

    def stop(self,*,announce=True):
        try:
            if self.stream is not None:
                stream=self.stream;self.stream=None
                try:stream.abort()
                finally:stream.close()
        finally:
            try:
                if self.gate:self.gate.reset()
            finally:
                self.reset_command()
                if announce:self.set_state('muted')


def is_epoch44(bundle):
    recipe=Path(bundle)/'recipe.json'
    return recipe.is_file() and json.loads(recipe.read_bytes()).get('architecture')=='OptionBTinyVCM-1'


def load_runtime(bundle,*,threads=1,engine=None,mode='demo',profile='benchmark-compatible'):
    if is_epoch44(bundle):
        from scripts.runtime.iris_epoch44 import Runtime
        return Runtime(bundle,threads=threads,engine=engine or 'onnxruntime',mode=mode,profile=profile)
    from scripts.runtime.mvp_static import Runtime
    return Runtime(bundle,threads=threads,engine=engine or 'ncnn',mode=mode)


def runtime_identity(runtime,owner=None):
    from scripts.runtime.mvp_static import digest
    result={'model_sha256':getattr(runtime,'model_sha256',runtime.recipe['binding'].get('model_sha256')),
            'model_qualified':False}
    if hasattr(runtime,'parameters'):
        result.update(parameters=runtime.parameters,selected_epoch=runtime.selected_epoch)
    wake=getattr(runtime,'wake_root',runtime.root/'wake')
    vad=getattr(runtime,'vad_root',wake)
    result['companions']={
        'wake':{'model_sha256':digest(wake/'iris.onnx'),'included_in_vcm_parameters':False},
        'vad':{'model_sha256':digest(vad/'silero-vad.onnx'),'included_in_vcm_parameters':False},
        'owner':{'model_sha256':owner.model_id,'included_in_vcm_parameters':False} if owner else None}
    return result


def application_config(runtime,settings):
    if hasattr(runtime,'parameters'):
        return {'mode':'joint_complete_command','engine':runtime.engine,
                'min_probability':runtime.policy['min_probability'],'min_margin':runtime.policy['min_margin']}
    return settings['inference']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--profile',choices=['benchmark-compatible','owner-protected'])
    p.add_argument('--engine',choices=['ncnn','onnxruntime'])
    p.add_argument('--wake-root',type=Path,help='Explicit independent Hey Iris companion override')
    p.add_argument('--no-audio',action='store_true',help='Disable feedback and actions, even with inherited flags')
    p.add_argument('--check',action='store_true',help='Load and verify command, wake, VAD and owner assets; no capture')
    p.add_argument('--wav',type=Path);p.add_argument('--wake',action='store_true');p.add_argument('--list-mics',action='store_true')
    p.add_argument('--list-devices',action='store_true')
    args=p.parse_args()
    benchmark_log = None
    if os.environ.get('PCV_BENCHMARK_LOG'):
        from scripts.runtime.pcv_benchmark_adapter import BenchmarkLog
        benchmark_log = BenchmarkLog(os.environ['PCV_BENCHMARK_LOG'])
    def emit(event):
        if benchmark_log is not None:benchmark_log(event)
        print(json.dumps(event,allow_nan=False),flush=True)
    from scripts.runtime.speaker_config import resolve_config
    settings,_=resolve_config(Path(os.environ.get('SMART_SPEAKER_ROOT',Path.cwd())),overrides={'engine':args.engine})
    profile=args.profile or settings['inference'].get('profile','benchmark-compatible')
    if args.list_mics or args.list_devices:
        discovery=AudioService(None,emit)
        emit({'event':'devices','devices':discovery.devices(),
              **({'outputs':discovery.output_devices()} if args.list_devices else {})});return
    engine=args.engine or ('onnxruntime' if is_epoch44(args.bundle) else settings['inference']['engine'])
    runtime=load_runtime(args.bundle,threads=args.threads,engine=engine,mode=settings['inference']['mode'],profile=profile)
    if args.wake_root is not None:runtime.wake_root=args.wake_root.resolve()
    if profile=='benchmark-compatible' or args.no_audio:settings['actions']['enabled']=False
    owner=None
    if profile=='owner-protected':
        from scripts.runtime.iris_owner import OwnerBackend
        owner=OwnerBackend(getattr(runtime,'owner_root',args.bundle/'owner'),threads=args.threads)
    feedback=None
    if is_epoch44(args.bundle):
        from scripts.runtime.iris_feedback import FeedbackPlayer
        enabled=(os.environ.get('IRIS_FEEDBACK_ENABLED')=='true' and profile=='owner-protected'
                 and not (args.no_audio or args.wav or args.check))
        speaker=int(os.environ.get('IRIS_SPEAKER_INDEX','-1')) if enabled else None
        feedback=FeedbackPlayer(speaker_index=speaker,enabled=enabled)
    service=AudioService(runtime,emit,settings,owner=owner,profile=profile,feedback=feedback)
    if args.check:
        service.new_gate()
        emit({'event':'checked','profile':profile,'owner_gating_enabled':owner is not None,'actions_executed':False,
              **runtime_identity(runtime,owner)})
        return
    if args.list_mics:emit({'event':'devices','devices':service.devices()});return
    if args.wav:emit({'event':'done',**service.wav(args.wav.read_bytes(),wake=args.wake)});return
    service.emit({'event':'ready','engine':runtime.engine,'loaded':True,'nativeSession':True,'state':service.state,
                  'profile':profile,'owner_gating_enabled':owner is not None,
                  'ownership_experimental':owner is not None,
                  'application_config':application_config(runtime,settings),'ui':settings['ui'],
                  **runtime_identity(runtime,owner),
                  'capabilities':{'feedback':feedback is not None},'feedback_enabled':bool(feedback and feedback.enabled),
                  'wake_phrase':'hey iris',
                  'policy_enabled':runtime.policy['enabled'] and getattr(runtime,'mode','demo')=='demo',
                  'threads':args.threads,'session_load_ms':runtime.load_ms})
    incoming=queue.Queue(maxsize=8)
    def reader():
        while True:
            line=sys.stdin.readline(6*1024*1024)
            if not line:incoming.put({'op':'shutdown'});return
            try:
                if not line.endswith('\n'):raise ValueError('Control message too large')
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('Expected control object')
                incoming.put(item,timeout=1)
            except Exception as error:emit({'event':'error','message':str(error)})
    threading.Thread(target=reader,daemon=True).start()
    recent_requests=OrderedDict()
    try:
        while True:
            try:command=incoming.get(timeout=0 if service.stream else .1)
            except queue.Empty:
                if service.stream:
                    try:service.tick()
                    except Exception as error:service.stop(announce=False);service.set_state('capture_error',message=str(error))
                continue
            request=command.get('id')
            cache_key=None;fingerprint=None
            try:
                if request is not None:
                    if type(request) not in (int,str) or isinstance(request,str) and not 1<=len(request)<=128:
                        raise ValueError('Invalid runtime request ID')
                    cache_key=(type(request).__name__,request)
                    fingerprint=hashlib.sha256(json.dumps(command,sort_keys=True,allow_nan=False).encode()).hexdigest()
                    if cache_key in recent_requests:
                        prior_hash,prior_reply=recent_requests[cache_key]
                        if fingerprint!=prior_hash:raise ValueError('Conflicting duplicate runtime request ID')
                        service.emit(prior_reply);continue
                op=command.get('op')
                if op=='shutdown':break
                if op=='devices':result={'devices':service.devices()}
                elif op=='listen':result={'microphone':service.listen(command.get('device'))}
                elif op=='mute':service.stop();result={}
                elif op=='wav':result=service.wav(base64.b64decode(command['data'],validate=True),wake=command.get('wake') is True)
                elif op=='feedback':
                    if set(command)-{'id','op','cue_id','utterance_id','session_id'}:raise ValueError('Unsupported feedback control fields')
                    result=service.feedback_control(command.get('cue_id'),command.get('utterance_id'),session_id=command.get('session_id'))
                else:raise ValueError('Unknown runtime operation')
                reply={'event':'reply','id':request,'ok':True,**result}
            except Exception as error:
                if command.get('op')!='feedback':
                    service.set_state('capture_error' if command.get('op')=='listen' else 'runtime_error',message=str(error))
                reply={'event':'reply','id':request,'ok':False,'message':str(error)}
            if cache_key is not None and cache_key not in recent_requests:
                recent_requests[cache_key]=(fingerprint,reply)
                while len(recent_requests)>128:recent_requests.popitem(last=False)
            service.emit(reply)
    finally:
        service.stop(announce=False)
        if service.previews:service.previews.close()


if __name__=='__main__':main()
