"""Resident CPU inference worker. JSONL controls/events; never executes actions."""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

import numpy as np
from einops import rearrange
import soundfile as sf

from scripts.runtime.pcv_final_runtime import Runtime, make_gate

MAX_WAV_BYTES=4*1024*1024


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
    def __init__(self,runtime,emit,settings=None):
        self.runtime=runtime;self.deliver=emit;self.sequence=0;self.session=uuid.uuid4().hex
        self.state='awaiting_microphone';self.stream=None;self.gate=None;self.offset=0;self.source='microphone'
        self.last_signal=time.monotonic();self.warned_silent=False
        self.settings=settings
        self.request_id = None

    def emit(self,value):
        self.sequence+=1
        if getattr(self.runtime, 'action_free', False):
            value = {**value, 'owner_enforced': False, 'command_only': True}
            if self.request_id is not None:
                value['request_id'] = self.request_id
        if value.get('event')=='command' and self.settings and not getattr(self.runtime, 'action_free', False):
            from scripts.runtime.speaker_confidence import application_decision
            value={**value,'application':application_decision(value,self.settings['inference'])}
        self.deliver({**value,'session_id':self.session,'sequence':self.sequence,'source':self.source,
                      'actions_enabled':bool(not getattr(self.runtime, 'action_free', False) and self.settings and self.settings['actions']['enabled'] and self.source=='microphone')})

    def set_state(self,state,**extra):
        self.state=state;self.emit({'event':'state','state':state,**extra})

    def recognize(self,pcm):
        self.set_state('processing')
        return self.runtime.recognize(pcm)

    def require_live_backend(self):
        """A candidate replay service does not establish wake/owner support."""
        if getattr(self.runtime, 'action_free', False):
            raise ValueError(
                'Experimental service is command-only; wake/owner/live '
                'microphone qualification requires a separate backend'
            )

    def new_gate(self):
        self.require_live_backend()
        # A thin proxy adds actual processing state without changing the frozen recognizer.
        service=self
        class Observed:
            root=service.runtime.root
            def recognize(self,pcm):return service.recognize(pcm)
        return make_gate(Observed())

    def forward(self,event):
        if event.get('status')=='detected':self.set_state('wake_detected')
        elif event.get('status')=='command_ready':self.set_state('command_ready')
        if 'output' in event:
            self.emit({**event,'event':'command','utterance_id':uuid.uuid4().hex})
            self.set_state('wake_listening' if self.stream else 'awaiting_microphone')
        else:
            self.emit(event)
            if event.get('reason'):self.set_state('wake_listening' if self.stream else 'awaiting_microphone',reason=event['reason'])

    def wav(self,data,*,wake=False):
        if wake:
            self.require_live_backend()
        pcm=decode_wav(data,wake=wake)
        if (getattr(self.runtime, 'action_free', False)
                and not 400 <= len(pcm) <= 128000):
            raise ValueError('Candidate command requires 25 ms..8 s of PCM')
        self.stop(announce=False);self.source='wav';commands=0;wake_detected=False
        try:
            if wake:
                gate=self.new_gate();self.set_state('wake_listening')
                for offset in range(0,len(pcm),512):
                    for event in gate.feed(pcm[offset:offset+512],offset):
                        commands+=int('output' in event);wake_detected|=event.get('status')=='detected';self.forward(event)
                if not commands:self.emit({'event':'wake_result','wake_detected':wake_detected,**gate.eof()})
            else:
                self.forward(self.recognize(pcm));commands=1
            return {'commands':commands,'wake_detected':wake_detected,'audio_seconds':len(pcm)/16000}
        finally:
            if self.state!='awaiting_microphone':self.set_state('awaiting_microphone')

    def devices(self):
        try:
            import sounddevice as sd
            hosts=sd.query_hostapis()
            return [{'index':i,'name':d['name'],'hostapi':hosts[d['hostapi']]['name'],'channels':d['max_input_channels'],
                     'default_sample_rate':d['default_samplerate']} for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]
        except Exception as error:
            raise RuntimeError('Microphone discovery failed. On Pi OS run start.py --system-packages --list-mics; check that the input is connected to this host. '+str(error)[:200]) from error

    def listen(self,index):
        self.require_live_backend()
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

    def tick(self):
        pcm,overflow=self.stream.read(512);pcm=rearrange(pcm,'t 1 -> t')
        if np.max(np.abs(pcm))>1e-5:self.last_signal=time.monotonic();self.warned_silent=False
        elif not self.warned_silent and time.monotonic()-self.last_signal>5:
            self.emit({'event':'capture_warning','reason':'no_signal','message':'No microphone signal for 5 seconds'});self.warned_silent=True
        for event in self.gate.feed(pcm,self.offset,overflow=overflow):self.forward(event)
        if self.gate.state=='command' and self.gate.started and self.state!='recording':self.set_state('recording')
        self.offset+=len(pcm)

    def stop(self,*,announce=True):
        if self.stream is not None:
            stream=self.stream;self.stream=None
            try:stream.abort()
            finally:stream.close()
        if self.gate:self.gate.reset()
        if announce:self.set_state('muted')


def runtime_status(runtime, settings: dict, threads: int,
                   measured_load_ms: float) -> dict:
    """Expose actual bundle identity without treating loaded as qualified."""
    status = {
        'engine': runtime.engine, 'loaded': True, 'nativeSession': True,
        'application_config': dict(settings['inference']),
        'ui': dict(settings['ui']), 'threads': threads,
    }
    if not getattr(runtime, 'action_free', False):
        return {
            **status,
            'model_sha256': runtime.recipe['binding']['model_sha256'],
            'policy_enabled': runtime.policy['enabled'],
            'session_load_ms': runtime.load_ms,
        }
    thresholds = runtime.contract['thresholds']
    status['application_config'].update({
        'engine': runtime.engine, 'threads': threads,
        'intent_threshold': thresholds['intent'],
        'slot_threshold': thresholds['slot'],
    })
    return {
        **status,
        'model_sha256': runtime.recipe['checkpoint_source_sha256'],
        'contract_hash': runtime.contract['contract_hash'],
        'checkpoint_status': runtime.recipe.get('checkpoint_status'),
        'calibration_id': runtime.policy.get('calibration_id'),
        'policy_enabled': runtime.policy.get('require_calibration') is True,
        'session_load_ms': measured_load_ms,
        'command_only': True, 'owner_enforced': False,
        'qualification_verified': False,
        'inference_mode': 'complete_utterance',
    }


def validate_candidate_control(command: dict) -> None:
    """Allow only action-free controls; expected labels are never inputs."""
    identifier = command.get('id')
    if identifier is not None and type(identifier) not in (str, int):
        raise ValueError('Experimental request ID must be a string or integer')
    operation = command.get('op')
    fields = {'op', 'id'}
    if operation == 'wav':
        fields.update({'data', 'wake'})
        if command.get('wake', False) is not False:
            raise ValueError('Experimental service is command-only')
    elif operation not in ('status', 'mute', 'shutdown'):
        raise ValueError('Experimental service is command-only')
    if set(command) - fields:
        raise ValueError('Unexpected experimental control fields')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--engine',choices=['ncnn','onnxruntime'])
    p.add_argument(
        '--experimental-intent-slots', action='store_true',
        help='Action-free WAV replay or resident JSONL test service',
    )
    p.add_argument('--wav',type=Path);p.add_argument('--wake',action='store_true');p.add_argument('--list-mics',action='store_true')
    args=p.parse_args()
    if not 1 <= args.threads <= 8:
        p.error('--threads must be 1..8')
    if args.experimental_intent_slots and (args.wake or args.list_mics):
        p.error('Experimental service is command-only; no wake/microphone mode')
    benchmark_log = None
    if os.environ.get('PCV_BENCHMARK_LOG'):
        from scripts.runtime.pcv_benchmark_adapter import BenchmarkLog
        benchmark_log = BenchmarkLog(os.environ['PCV_BENCHMARK_LOG'])
    def emit(event):
        if benchmark_log is not None:benchmark_log(event)
        print(json.dumps(event,allow_nan=False),flush=True)
    from scripts.runtime.speaker_config import resolve_config
    settings,_=resolve_config(Path(os.environ.get('SMART_SPEAKER_ROOT',Path.cwd())),overrides={'engine':args.engine})
    if args.list_mics:
        emit({'event':'devices','devices':AudioService(None,emit).devices()});return
    loading_started = time.perf_counter()
    if args.experimental_intent_slots:
        from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime
        runtime = IntentSlotsRuntime(args.bundle, settings['inference']['engine'], args.threads)
        if benchmark_log is not None:
            benchmark_log = BenchmarkLog(os.environ['PCV_BENCHMARK_LOG'],
                                         decision='final_application', custom=runtime.contract['custom'])
    else:
        runtime=Runtime(args.bundle,diagnostic=args.diagnostic,threads=args.threads,engine=settings['inference']['engine'])
    measured_load_ms = (time.perf_counter() - loading_started) * 1000
    service=AudioService(runtime,emit,settings)
    if args.list_mics:emit({'event':'devices','devices':service.devices()});return
    if args.wav:emit({'event':'done',**service.wav(args.wav.read_bytes(),wake=args.wake)});return
    service.emit({
        'event': 'ready', 'state': service.state,
        **runtime_status(runtime, settings, args.threads, measured_load_ms),
    })
    incoming=queue.Queue(maxsize=8)
    protocol_fault = object()
    def reader():
        while True:
            line=sys.stdin.readline(6*1024*1024)
            if not line:incoming.put({'op':'shutdown'});return
            try:
                if not line.endswith('\n'):raise ValueError('Control message too large')
                item=json.loads(line)
                if not isinstance(item,dict):raise ValueError('Expected control object')
                if args.experimental_intent_slots:
                    incoming.put(item)
                else:
                    incoming.put(item, timeout=1)
            except Exception as error:
                if args.experimental_intent_slots:
                    incoming.put((protocol_fault, str(error)))
                else:
                    emit({'event': 'error', 'message': str(error)})
    threading.Thread(target=reader,daemon=True).start()
    try:
        while True:
            try:command=incoming.get(timeout=0 if service.stream else .1)
            except queue.Empty:
                if service.stream:
                    try:service.tick()
                    except Exception as error:service.stop(announce=False);service.set_state('capture_error',message=str(error))
                continue
            if isinstance(command, tuple) and command[0] is protocol_fault:
                service.emit({'event': 'error', 'message': command[1]})
                continue
            request=command.get('id')
            if args.experimental_intent_slots:
                if request is not None and type(request) not in (str, int):
                    request = None
                service.request_id = request
            try:
                if args.experimental_intent_slots:
                    validate_candidate_control(command)
                op=command.get('op')
                if op=='shutdown':break
                if op == 'status':
                    result = {
                        'state': service.state,
                        **runtime_status(runtime, settings, args.threads,
                                         measured_load_ms),
                    }
                elif op=='devices':result={'devices':service.devices()}
                elif op=='listen':result={'microphone':service.listen(command.get('device'))}
                elif op=='mute':service.stop();result={}
                elif op=='wav':result=service.wav(base64.b64decode(command['data'],validate=True),wake=command.get('wake') is True)
                else:raise ValueError('Unknown runtime operation')
                service.emit({'event':'reply','id':request,'ok':True,**result})
            except Exception as error:
                service.set_state('capture_error' if command.get('op')=='listen' else 'runtime_error',message=str(error))
                service.emit({'event':'reply','id':request,'ok':False,'message':str(error)})
            finally:
                service.request_id = None
    finally:service.stop(announce=False)


if __name__=='__main__':main()
