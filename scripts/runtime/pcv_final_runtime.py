"""CPU-only complete-utterance PCV; finite candidate scores never yield text."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from einops import rearrange, reduce, pack
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
if (ROOT/'experiments/phrase-contract-v1').exists():
    sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from pcvcm.decoder import PrefixCTC, rank_commands
from vcm.frontend import features, SPEC
from vcm.catalog import PHONES
from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command, output

BINDINGS = {'model_sha256':'model.onnx','catalog_sha256':'catalog.json',
            'normalization_sha256':'normalization.json','schema_sha256':'output.schema.json',
            'custom_slots_sha256':'custom-slots.json'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read_json(path):
    def invalid(value): raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=invalid)


def verify_bundle(folder, *, diagnostic=False):
    root=Path(folder)
    recipe,policy=read_json(root/'recipe.json'),read_json(root/'policy.json')
    if recipe.get('inference_mode')!='complete_utterance' or recipe.get('state_shapes')!=[]:
        raise ValueError('Expected complete_utterance model with no recurrent state')
    if recipe.get('heads')!=['ctc','intent','target'] or recipe.get('output_hop_ms')!=20:
        raise ValueError('Unsupported head layout or output stride')
    for name,expected in (('intent_labels',['OOV',*INTENTS]),('acoustic_labels',['<blank>',*PHONES])):
        if name in recipe or recipe.get('architecture_class')=='PooledContextVCM':
            if recipe.get(name)!=expected: raise ValueError('Unsupported or missing head labels: '+name)
    binding={key:sha(root/name) for key,name in BINDINGS.items()}
    if binding!=recipe.get('binding') or binding!=policy.get('binding'):
        raise ValueError('Bundle identity mismatch')
    if type(policy.get('enabled')) is not bool or policy['enabled']!=recipe.get('policy_enabled'):
        raise ValueError('Invalid policy identity')
    for key in ('min_score','min_margin','min_intent_probability'):
        value=policy.get(key)
        if type(value) not in (int,float) or not np.isfinite(value):raise ValueError('Invalid policy threshold: '+key)
    if policy['min_margin']<0 or not 0<=policy['min_intent_probability']<=1:
        raise ValueError('Invalid policy threshold range')
    if not policy['enabled'] and not diagnostic:
        raise ValueError('Unqualified model requires explicit --diagnostic')
    # Whole-directory deployment identity is added by the publisher.
    if (root/'deployment.json').exists():
        manifest=read_json(root/'deployment.json')
        required=set(BINDINGS.values())|{'recipe.json','policy.json'}
        if not required <= set(manifest.get('files',{})):
            raise ValueError('Missing deployment identity')
        for name,digest in manifest['files'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path)!=digest:
                raise ValueError('Deployment identity mismatch: '+name)
    norm=read_json(root/'normalization.json')
    if norm.get('frontend')!=SPEC:
        raise ValueError('Unsupported frontend')
    catalog=read_json(root/'catalog.json')['entries']
    for row in catalog:
        if row['command'] is not None: validate_command(row['command'])
    if not catalog: raise ValueError('Empty finite catalog')
    return recipe,policy,norm,catalog


def decision(record,policy):
    """Same semantic/target checks as frozen public qualification, without trainers."""
    if not policy['enabled']: return output(None,'unqualified_model')
    c=record['candidate']
    valid=c is not None and record['score'] is not None
    if valid:
        valid=(record['intent_index']==INTENTS.index(c['intent'])+1 and
               record['score']>=policy['min_score'] and record['margin']>=policy['min_margin'] and
               record['intent_probability']>=policy['min_intent_probability'])
        if c['intent'] in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'}:
            named=c['slots']['light_id']!='light.all'
            valid=valid and ((named and record['target_named_probability']>=.9) or
                              (not named and record['target_named_probability']<=.1))
    if valid: return output(c)
    intent=INTENTS[record['intent_index']-1] if record['intent_index'] else 'OOV'
    confident=record['top_intent_probability']>=max(.7,policy['min_intent_probability'])
    needs=intent in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR','CALL','CREATE_REMINDER',
                    'TIMER','ALARM','TEMPERATURE','PLAY_MUSIC','VOLUME'}
    return output(None,'invalid_argument' if confident and needs else 'OOV')


def probabilities(logits):
    value=np.exp(np.asarray(logits,dtype='float64')-np.max(logits))
    return value/value.sum()


class Runtime:
    def __init__(self,folder,*,diagnostic=False,threads=1,engine='onnxruntime'):
        start=time.perf_counter()
        if type(threads) is not int or not 1<=threads<=8: raise ValueError('Threads must be 1..8')
        self.root=Path(folder); self.threads=threads; self.diagnostic=diagnostic;self.engine=engine
        if engine not in ('onnxruntime','ncnn'):raise ValueError('Engine must be onnxruntime or ncnn')
        self.recipe,self.policy,self.norm,self.entries=verify_bundle(folder,diagnostic=diagnostic)
        if engine=='ncnn':
            from scripts.runtime.speaker_engine import NCNNSession
            self.session=NCNNSession(self.root,threads)
        else:
            options=ort.SessionOptions()
            options.intra_op_num_threads=threads;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(self.root/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
            inputs=self.session.get_inputs();outputs=self.session.get_outputs()
            if len(inputs)!=1 or inputs[0].name!='features' or inputs[0].type!='tensor(float)' or inputs[0].shape[:2]!=[1,40]:
                raise ValueError('Model input/state interface mismatch')
            if [p.name for p in outputs]!=['ctc','intent','target'] or any(p.type!='tensor(float)' or len(p.shape)!=3 or (isinstance(p.shape[0],int) and p.shape[0]!=1) or p.shape[2]!=n for p,n in zip(outputs,[40,19,2])):
                raise ValueError('Model head interface mismatch')
        self.engine_model_sha256=sha(self.root/'ncnn/model.ncnn.bin') if engine=='ncnn' else self.recipe['binding']['model_sha256']
        self.decoder=PrefixCTC([e['phones'] for e in self.entries])
        self.load_ms=(time.perf_counter()-start)*1000

    def recognize(self,pcm,sample_rate=16000):
        started=time.perf_counter()
        x=np.asarray(pcm,dtype='float32')
        if sample_rate!=16000: raise ValueError('Requires 16000 Hz; resample explicitly before calling')
        if x.ndim!=1 or not np.isfinite(x).all() or (len(x) and np.max(np.abs(x))>1.):
            raise ValueError('Requires finite mono floating PCM in [-1,1]')
        duration=len(x)/16000
        event={'schema_version':1,'event':'command','model_sha256':self.recipe['binding']['model_sha256'],
               'policy_enabled':self.policy['enabled'],'diagnostic_candidate':None,'engine':self.engine,'engine_model_sha256':self.engine_model_sha256,
               'audio_seconds':duration,'endpoint_mode':'complete_utterance'}
        if len(x)<400 or duration>15 or float(np.max(np.abs(x),initial=0))<1e-6:
            event['output']=output(None,'capture_fault')
            event['timing']={'frontend_ms':0.,'model_ms':0.,'finite_decoder_ms':0.,
                             'total_ms':(time.perf_counter()-started)*1000,'rtf':None}
            return event
        f=features(x,self.norm['mean'],self.norm['std']);front=time.perf_counter()
        arrays=self.session.run(None,{'features':rearrange(f,'c t -> 1 c t')});encoded=time.perf_counter()
        n=(f.shape[1]+1)//2
        for arr,width in zip(arrays,[40,19,2]):
            if arr.shape!=(1,n,width) or not np.isfinite(arr).all(): raise ValueError('Malformed model output')
        scores=self.decoder.score(arrays[0][0])
        pred=rank_commands(scores,self.entries)
        ip=probabilities(reduce(arrays[1],'1 t c -> c','mean'))
        tp=probabilities(reduce(arrays[2],'1 t c -> c','mean'))
        winner=int(ip.argmax());c=pred['command'];ci=INTENTS.index(c['intent'])+1 if c else 0
        record={'candidate':c,'score':pred['score'],'margin':pred['margin'],'intent_index':winner,
                'intent_probability':float(ip[ci]),'top_intent_probability':float(ip[winner]),
                'target_named_probability':float(tp[1]),'candidate_phrase_id':pred['phrase_id']}
        event['output']=decision(record,self.policy)
        if self.diagnostic:
            from scripts.runtime.speaker_confidence import slot_confidences
            record['slot_confidence']=slot_confidences(scores,self.entries,c,n)
            record['slot_confidence_method']='best_value_ctc_likelihood_within_intent'
            event['diagnostic_candidate']=record
        finished=time.perf_counter()
        event['timing']={'frontend_ms':1000*(front-started),'model_ms':1000*(encoded-front),
            'finite_decoder_ms':1000*(finished-encoded),'total_ms':1000*(finished-started),'rtf':(finished-started)/duration}
        return event


def read_wav(path):
    import soundfile as sf
    pcm,rate=sf.read(path,dtype='float32',always_2d=False)
    if rate!=16000 or pcm.ndim!=1: raise ValueError('WAV must be mono 16000 Hz')
    return pcm


class EndpointCommand:
    """WakeGate adapter retaining the full utterance until a neural endpoint."""
    def __init__(self,runtime,voice): self.runtime=runtime;self.voice=voice;self.reset()
    def reset(self):
        self.blocks=[];self.samples=0;self.silence=0;self.finalized=False;self.voice.reset()
    def feed(self,pcm,sequence):
        if sequence!=self.samples:
            self.finalized=True;return {'output':output(None,'capture_fault'),'reason':'capture_fault'}
        self.blocks.append(pcm.copy());self.samples+=len(pcm)
        self.silence=self.silence+len(pcm) if self.voice(pcm)<.35 else 0
        if self.silence>=6400 or self.samples>=16000*15:
            self.finalized=True
            if self.samples>=16000*15: return {'output':output(None,'capture_fault'),'reason':'capture_fault'}
            audio,_=pack(self.blocks,'*')
            return self.runtime.recognize(audio)
        return {'reason':None}


def make_gate(runtime):
    start=time.perf_counter()
    from scripts.runtime.wake_models import HeyJarvis,SileroProbability
    try:
        from wake_gate import WakeGate
    except ImportError:
        sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
        from wake_gate import WakeGate
    folder=runtime.root/'wake'
    gate=WakeGate(EndpointCommand(runtime,SileroProbability(folder)),HeyJarvis(folder),SileroProbability(folder))
    gate.load_ms=(time.perf_counter()-start)*1000
    return gate


def capture(runtime,device,*,seconds=30,wake=True):
    import sounddevice as sd
    if not np.isfinite(seconds) or not 0<seconds<=300:raise ValueError('Capture duration must be finite and 0..300 seconds')
    if not wake:
        if seconds>15:raise ValueError('Command-only capture must be at most 15 seconds')
        pcm=sd.rec(round(seconds*16000),samplerate=16000,channels=1,dtype='float32',device=device)
        sd.wait()
        if sd.get_status():
            yield {'schema_version':1,'event':'command','output':output(None,'capture_fault'),
                   'diagnostic_candidate':None,'capture_status':'overflow_or_underflow'};return
        yield runtime.recognize(rearrange(pcm,'t 1 -> t'));return
    gate=make_gate(runtime);sequence=0;deadline=time.monotonic()+seconds
    yield {'stage':'wake','status':'ready','session_load_ms':gate.load_ms,'wake_phrase':'hey jarvis'}
    with sd.InputStream(samplerate=16000,channels=1,dtype='float32',blocksize=512,device=device) as stream:
        while time.monotonic()<deadline:
            pcm,overflow=stream.read(512)
            for event in gate.feed(rearrange(pcm,'t 1 -> t'),sequence,overflow=overflow): yield event
            sequence+=512


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path('models/default'))
    p.add_argument('--diagnostic',action='store_true');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--engine',choices=['ncnn','onnxruntime'])
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--wav',type=Path);source.add_argument('--mic',type=int);source.add_argument('--check',action='store_true')
    p.add_argument('--seconds',type=float);p.add_argument('--no-wake',action='store_true')
    args=p.parse_args()
    from scripts.runtime.speaker_config import resolve_config
    settings,_=resolve_config(Path.cwd(),overrides={'engine':args.engine})
    runtime=Runtime(args.bundle,diagnostic=args.diagnostic,threads=args.threads,engine=settings['inference']['engine'])
    if args.check:
        print(canonical({'event':'ready','inference_mode':'complete_utterance','policy_enabled':runtime.policy['enabled'],
                         'model_sha256':runtime.recipe['binding']['model_sha256'],'load_ms':runtime.load_ms}));return
    seconds=args.seconds if args.seconds is not None else (5 if args.no_wake else 60)
    events=[runtime.recognize(read_wav(args.wav))] if args.wav else capture(runtime,args.mic,seconds=seconds,wake=not args.no_wake)
    for event in events: print(canonical(event),flush=True)


if __name__=='__main__':main()
