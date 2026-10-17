"""Selected Hey Iris streaming adapter plus explicit pretrained Jarvis fallback.

RollingWake/HeyIris reuse scripts/runtime/mvp_wake.py from the frozen
vcm-mvp-hey-iris-v1 worktree; source identity is recorded beside the copied graph.
"""
import json
from pathlib import Path
import time
import numpy as np
from scripts.submission.common import ROOT, sha

MODES = ('iris_fallback', 'iris', 'jarvis')
IRIS_FOLDER = ROOT/'submission/wake/hey-iris'


class RollingWake:
    def __init__(self,score,*,threshold,release=.3,consecutive=2,cooldown_samples=16000):
        if not 0<=release<threshold<=1 or consecutive<1 or cooldown_samples<0:
            raise ValueError('Invalid wake policy')
        self.score=score;self.threshold=threshold;self.release=release
        self.consecutive=consecutive;self.cooldown_samples=cooldown_samples;self.reset()

    def reset(self):
        self.pending=np.empty(0,dtype='float32');self.audio=np.empty(0,dtype='float32')
        self.hits=0;self.latched=False;self.cooldown=0

    def push(self,pcm):
        x=np.asarray(pcm,dtype='float32')
        if x.ndim!=1 or not np.isfinite(x).all():raise ValueError('Invalid wake PCM')
        self.pending=np.concatenate((self.pending,x));result=None
        while len(self.pending)>=1280:
            self.audio=np.concatenate((self.audio,self.pending[:1280]))[-25600:]
            self.pending=self.pending[1280:];self.cooldown=max(0,self.cooldown-1280)
            if len(self.audio)<25600:continue
            value=float(self.score(self.audio))
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid wake score')
            if value<self.release and not self.cooldown:self.latched=False
            self.hits=self.hits+1 if value>=self.threshold else 0
            trigger=self.hits>=self.consecutive and not self.latched and not self.cooldown
            if trigger:self.latched=True;self.cooldown=self.cooldown_samples;self.hits=0
            result=max(result or 0.,float(trigger))
        return result


class HeyIris(RollingWake):
    def __init__(self,folder=IRIS_FOLDER):
        from vcm.frontend import features
        import onnxruntime as ort
        folder=Path(folder);meta=json.loads((folder/'iris.json').read_text())
        if sha(folder/'iris.onnx')!=meta['files']['iris.onnx']:
            raise ValueError('Hey Iris model integrity failed')
        cfg=ort.SessionOptions();cfg.intra_op_num_threads=cfg.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(folder/'iris.onnx'),cfg,providers=['CPUExecutionProvider'])
        norm=meta['normalization'];self.meta=meta
        def score(pcm):
            f=features(pcm,norm['mean'],norm['std'])
            values=self.session.run(None,{'features':f[None]})[0][0].astype('float64')
            p=np.exp(values-values.max());return float(p[1]/p.sum())
        super().__init__(score,threshold=meta['threshold'],release=meta['release'],
                         consecutive=meta['consecutive'],cooldown_samples=meta['cooldown_samples'])


class WakeChoice:
    def __init__(self,iris,jarvis):
        self.iris=iris;self.jarvis=jarvis;self.last_phrase='hey iris' if iris else 'hey jarvis'
        self.warning=''
    def reset(self):
        for model in (self.iris,self.jarvis):
            if model is not None:model.reset()
    def push(self,pcm):
        try:iris=self.iris.push(pcm) if self.iris is not None else None
        except Exception:
            if self.jarvis is None:raise
            self.iris=None;iris=None;self.warning='Hey Iris inference failed. Hey Jarvis is active.'
        jarvis=self.jarvis.push(pcm) if self.jarvis is not None else None
        if iris is not None and iris>=.5:self.last_phrase='hey iris'
        elif jarvis is not None and jarvis>=.5:self.last_phrase='hey jarvis'
        values=[value for value in (iris,jarvis) if value is not None]
        return max(values) if values else None


def make_gate(runtime, observed, mode='iris_fallback'):
    if mode not in MODES:raise ValueError('Unknown wake mode')
    from scripts.runtime.pcv_final_runtime import EndpointCommand
    from scripts.runtime.wake_models import HeyJarvis, SileroProbability
    start=time.perf_counter();iris=None;warning=''
    if mode!='jarvis':
        try:iris=HeyIris()
        except Exception as error:
            if mode=='iris':raise RuntimeError('Hey Iris could not load; choose Hey Jarvis or fallback mode.') from error
            warning='Hey Iris could not load. Hey Jarvis is active.'
    # Reuse the frozen gate implementation, replacing only its detector.
    try:
        from wake_gate import WakeGate
    except ImportError:
        import sys
        sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
        from wake_gate import WakeGate
    detector=WakeChoice(iris,HeyJarvis(runtime.root/'wake') if mode!='iris' else None)
    folder=runtime.root/'wake'
    gate=WakeGate(EndpointCommand(observed,SileroProbability(folder)),detector,SileroProbability(folder))
    original=gate.feed
    def feed(*args,**kwargs):
        events=original(*args,**kwargs)
        if detector.warning and gate.wake_config['active']!='jarvis':
            gate.wake_config.update(active='jarvis',warning=detector.warning)
            events.insert(0,{'event':'wake_config','wake':dict(gate.wake_config)})
        return [{**event,**({'wake_phrase':detector.last_phrase} if 'wake_phrase' in event else {})} for event in events]
    gate.feed=feed
    gate.wake_config={'selected':mode,'active':'iris_fallback' if iris is not None and mode=='iris_fallback' else 'iris' if iris is not None else 'jarvis',
                      'warning':warning,'iris_experimental':True,'available':list(MODES)}
    gate.load_ms=(time.perf_counter()-start)*1000
    return gate
