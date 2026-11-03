"""Torch-free direct shared-VCM inference. Recognition never enables actions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnxruntime as ort
from scripts.runtime.demo_slots_contract import COMMANDS, ACTIONS, decode_scores
from scripts.runtime.pcv_schema import output
from vcm.frontend import features
POLICY=dict(method='argmax',calibrated=False,action_authoritative=True,missing_required_slot='reject')


class Runtime:
    def __init__(self,bundle,threads=1):
        bundle=Path(bundle)
        recipe=json.loads((bundle/'recipe.json').read_text())
        self.mode=recipe.get('mode')
        parameters={'flat':175139,'heads':175653}
        if (recipe.get('architecture')!='SharedDemoSlotVCM-v1' or self.mode not in parameters
                or recipe.get('parameters')!=parameters[self.mode]
                or recipe.get('qualified') is not False or recipe.get('actions_enabled') is not False):
            raise ValueError('Invalid shared-demo research recipe')
        assets={'model.onnx','normalization.json','commands.json','policy.json'}
        if set(recipe.get('binding',{}))!=assets:
            raise ValueError('Incomplete shared-demo asset binding')
        for name,digest in recipe['binding'].items():
            if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Shared-demo asset binding mismatch: '+name)
        if json.loads((bundle/'commands.json').read_text())!=list(COMMANDS):
            raise ValueError('Changed shared-demo complete class order')
        if json.loads((bundle/'policy.json').read_text())!=POLICY:
            raise ValueError('Only the frozen argmax/no-intent-rescue policy is supported')
        self.norm=json.loads((bundle/'normalization.json').read_text())
        mean,std=[np.asarray(self.norm[k]) for k in ('mean','std')]
        if (mean.shape!=(40,) or std.shape!=(40,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std<=0)):
            raise ValueError('Invalid shared-demo normalization')
        if type(threads) is not int or not 1<=threads<=4:
            raise ValueError('Declare1..4 inference threads')
        options=ort.SessionOptions()
        options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(bundle/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.size=len(COMMANDS) if self.mode=='flat' else len(ACTIONS)+8
        if self.session.get_outputs()[0].shape[-1]!=self.size:
            raise ValueError('Graph output does not match the frozen demo head layout')

    def recognize(self,pcm):
        pcm=np.asarray(pcm,dtype=np.float32)
        common=dict(actions_enabled=False,model_qualified=False,qualified_output=output(None,'unqualified_model'))
        if (pcm.ndim!=1 or not 400<=pcm.size<=320000 or not np.isfinite(pcm).all()
                or np.max(np.abs(pcm))<1e-6 or np.max(np.abs(pcm))>1):
            return dict(class_index=0,logits=None,output=output(None,'capture_fault'),**common)
        x=features(pcm,self.norm['mean'],self.norm['std'])
        scores=self.session.run(None,dict(features=x[None],lengths=np.asarray([x.shape[1]],dtype=np.int64)))[0][0]
        decoded=decode_scores(scores,self.mode)
        command=decoded['command']
        reason=None if command is not None else ('invalid_argument' if decoded.get('heads',{}).get('action',0)>0 else 'OOV')
        return dict(class_index=decoded['class_index'],logits=scores.tolist(),head_predictions=decoded.get('heads'),
            output=output(command,reason),**common)


if __name__=='__main__':
    import argparse
    import soundfile as sf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--wav',type=Path,required=True)
    args=parser.parse_args()
    pcm,rate=sf.read(args.wav,dtype='float32')
    if rate!=16000:raise ValueError('Require original mono16kHz PCM; no implicit resampling')
    print(json.dumps(Runtime(args.bundle).recognize(pcm)))
