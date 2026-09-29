"""Single-session portable CPU runtime. No torch, transcript or action API."""
from pathlib import Path
import numpy as np
import time
from einops import rearrange
from .frontend import StreamingFrontend, SPEC
from .decoder import CTCRanker
from .integrity import load, digest, identity
from .acceptance import decide
from .catalog import validate_command, PHONES


class CaptureGuard:
    def __init__(self): self.reset()
    def reset(self): self.next_sample=0; self.fault=None
    def check(self, pcm, sequence, overflow=False):
        if self.fault: return False
        if overflow: self.fault='capture_overflow'
        elif sequence != self.next_sample: self.fault='capture_gap'
        elif pcm.ndim != 1 or not np.isfinite(pcm).all(): self.fault='invalid_audio'
        elif self.next_sample+len(pcm)>16000*30: self.fault='utterance_too_long'
        self.next_sample += len(pcm)
        return self.fault is None


class Runtime:
    def __init__(self,bundle,threads=1):
        import onnxruntime as ort
        self.bundle=Path(bundle)
        self.recipe=load(self.bundle/'recipe.json')
        for name,expected in self.recipe['files'].items():
            path=(self.bundle/name).resolve()
            if not path.is_relative_to(self.bundle.resolve()) or digest(path)!=expected:
                raise ValueError(f'Bundle identity mismatch: {name}')
        if self.recipe['frontend']['spec'] != SPEC: raise ValueError('Frontend contract mismatch')
        self.catalog=load(self.bundle/'catalog.json')
        if self.catalog['phone_inventory']!=['<blank>']+PHONES: raise ValueError('Phone inventory mismatch')
        if any(c['command'] is not None and not validate_command(c['command']) for c in self.catalog['commands']):
            raise ValueError('Invalid typed catalog command')
        self.policy=load(self.bundle/'policy.json')
        self.binding={'model':digest(self.bundle/'model.onnx'), 'catalog':identity(self.catalog),
                      'frontend':identity(self.recipe['frontend'])}
        if self.policy['binding'] != self.binding: raise ValueError('Policy identity mismatch')
        options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.session=ort.InferenceSession(str(self.bundle/'model.onnx'),options,providers=['CPUExecutionProvider'])
        names=[item.name for item in self.session.get_inputs()]
        if names!=['features']+[f'state_{i}' for i in range(6)]: raise ValueError('Model input contract mismatch')
        if self.recipe['chunk_frames'] not in (4,8): raise ValueError('Expected 40 or 80 ms inspection cadence')
        self.frontend=StreamingFrontend(self.recipe['frontend']['mean'],self.recipe['frontend']['std'])
        self.ranker=CTCRanker([c['phones'] for c in self.catalog['commands']])
        self.guard=CaptureGuard(); self.reset()

    def reset(self):
        self.frontend.reset(); self.ranker.reset(); self.guard.reset()
        self.states={f'state_{i}':np.zeros(shape,dtype='float32') for i,shape in enumerate(self.recipe['state_shapes'])}
        self.scores=None; self.finalized=False
        self.timing={'frontend_ms':0.,'encoder_ms':0.,'decoder_ms':0.}

    def feed(self,pcm,sequence,*,endpoint=False,overflow=False):
        pcm=np.asarray(pcm,dtype='float32')
        if self.finalized: raise ValueError('Reset required after endpoint')
        if not self.guard.check(pcm,sequence,overflow):
            return {'accepted':False,'reason':self.guard.fault,'command':None,'candidate':None}
        started=time.perf_counter(); feats=self.frontend.feed(pcm)
        self.timing['frontend_ms']+=(time.perf_counter()-started)*1000
        for start in range(0,feats.shape[1],self.recipe['chunk_frames']):
            inputs={'features':rearrange(feats[:,start:start+self.recipe['chunk_frames']],'c t -> 1 c t'),**self.states}
            started=time.perf_counter(); out=self.session.run(None,inputs)
            self.timing['encoder_ms']+=(time.perf_counter()-started)*1000
            self.states={f'state_{i}':x for i,x in enumerate(out[1:])}
            started=time.perf_counter(); self.scores=self.ranker.feed(out[0][0])
            self.timing['decoder_ms']+=(time.perf_counter()-started)*1000
        self.finalized=endpoint
        if self.scores is None: return {'accepted':False,'reason':'insufficient_audio','command':None,'candidate':None}
        order=np.argsort(self.scores)[::-1]; top=int(order[0]); command=self.catalog['commands'][top]['command']
        if not np.isfinite(self.scores[top]):
            return {'accepted':False,'reason':'no_complete_candidate' if endpoint else 'tentative','command':None,'candidate':None}
        # Competing aliases of the same typed command do not create artificial ambiguity.
        others=[i for i in order[1:] if self.catalog['commands'][int(i)]['command'] != command]
        margin=float(self.scores[top]-self.scores[others[0]]) if others else 0.
        result=decide(command,float(self.scores[top]),margin,self.policy,self.binding,endpoint=endpoint)
        result['candidate_phrase']=self.catalog['commands'][top]['phrase']
        return result

    def predict(self,pcm,chunk_samples=640):
        self.reset(); result=None
        for i in range(0,len(pcm),chunk_samples):
            result=self.feed(pcm[i:i+chunk_samples],i,endpoint=i+chunk_samples>=len(pcm))
        return result or self.feed(np.empty(0,dtype='float32'),0,endpoint=True)
