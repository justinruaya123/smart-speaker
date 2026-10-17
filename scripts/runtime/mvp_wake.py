"""Bounded 80 ms wake inference with two-hit hysteresis and cooldown."""
import json
from pathlib import Path
import numpy as np


class RollingWake:
    def __init__(self,score,*,threshold,release=.3,consecutive=2,cooldown_samples=16000):
        if not 0<=release<threshold<=1 or consecutive<1 or cooldown_samples<0:raise ValueError('Invalid wake policy')
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
    def __init__(self,folder,engine='onnxruntime'):
        from scripts.runtime.mvp_static import digest
        from vcm.frontend import features
        import onnxruntime as ort
        folder=Path(folder);meta=json.loads((folder/'iris.json').read_text())
        for name,expected in meta['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path)!=expected:raise ValueError('Hey Iris identity mismatch')
        if engine=='onnxruntime':
            cfg=ort.SessionOptions();cfg.intra_op_num_threads=cfg.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(folder/'iris.onnx'),cfg,providers=['CPUExecutionProvider'])
        elif engine=='ncnn':
            import ncnn
            self.ncnn=ncnn;self.net=ncnn.Net();self.net.opt.num_threads=1;self.net.opt.use_vulkan_compute=False
            self.net.opt.use_fp16_packed=False;self.net.opt.use_fp16_storage=False;self.net.opt.use_fp16_arithmetic=False
            self.net.opt.use_bf16_packed=False;self.net.opt.use_bf16_storage=False;self.net.opt.use_packing_layout=False
            if self.net.load_param(str(folder/'iris.ncnn.param')) or self.net.load_model(str(folder/'iris.ncnn.bin')):
                raise ValueError('Wake ncnn load failed')
        else:raise ValueError('Unknown wake engine')
        self.engine=engine
        self.norm=meta['normalization'];self.meta=meta
        def score(pcm):
            f=features(pcm,self.norm['mean'],self.norm['std'])
            if self.engine=='onnxruntime':values=self.session.run(None,{'features':f[None]})[0][0].astype('float64')
            else:
                data=np.ascontiguousarray(f);mat=self.ncnn.Mat(data).clone()
                with self.net.create_extractor() as ex:
                    if ex.input('in0',mat):raise ValueError('Wake ncnn input failed')
                    code,value=ex.extract('out0')
                    if code:raise ValueError('Wake ncnn extraction failed')
                    values=np.array(value,copy=True).reshape(-1).astype('float64')
            p=np.exp(values-values.max());return float(p[1]/p.sum())
        super().__init__(score,threshold=meta['threshold'],release=meta['release'],
                         consecutive=meta['consecutive'],cooldown_samples=meta['cooldown_samples'])


def make_gate(runtime,observed=None):
    import sys
    from scripts.runtime.wake_models import SileroProbability,SileroVAD
    from scripts.runtime.pcv_schema import output
    try:from wake_gate import WakeGate
    except ImportError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/pcv-synthetic-v2'))
        from wake_gate import WakeGate
    folder=runtime.root/'wake';recognizer=observed or runtime
    class Endpoint:
        def __init__(self):
            self.vad=SileroVAD(folder,{'silence_ms':400,'max_utterance_seconds':15,'preroll_ms':128,'speech_threshold':.5})
            self.reset()
        def reset(self):self.vad.reset();self.samples=0;self.finalized=False
        def feed(self,pcm,sequence):
            if sequence!=self.samples:
                self.finalized=True;return {'output':output(None,'capture_fault'),'reason':'capture_fault'}
            self.samples+=len(pcm);result=self.vad.push(pcm)
            if result is None:return {'reason':None}
            self.finalized=True;audio,timed_out=result
            return {'output':output(None,'capture_fault'),'reason':'capture_fault'} if timed_out else recognizer.recognize(audio)
    class IrisGate(WakeGate):
        def feed(self,*args,**kwargs):
            return [{**event,**({'wake_phrase':'hey iris'} if 'wake_phrase' in event else {}),
                     'profile':runtime.profile,'owner_gating_enabled':False} for event in super().feed(*args,**kwargs)]
    return IrisGate(Endpoint(),HeyIris(folder,engine=runtime.engine),SileroProbability(folder))
