"""Export the selected endpoint checkpoint to ncnn; no training or installation."""
import argparse
import json
import hashlib
import shutil
import subprocess
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from einops import rearrange, reduce
from scripts.models.pcv_release_pooling import PooledContextVCM


class NCNNEndpoint(nn.Module):
    """Batch-one full utterance: masks are all true; positions supplied explicitly."""
    def __init__(self,model):
        super().__init__();self.model=model
        self.attention=nn.ModuleList()
        for block in model.encoder.blocks:
            width=block.qkv.in_features
            attention=nn.MultiheadAttention(width,block.heads,batch_first=True,dropout=0.)
            attention.in_proj_weight.data.copy_(block.qkv.weight)
            attention.in_proj_bias.data.copy_(block.qkv.bias)
            attention.out_proj.load_state_dict(block.attention_output.state_dict())
            self.attention.append(attention)

    def forward(self,features,position):
        encoder=self.model.encoder
        x=rearrange(encoder.stem(features),'b c t -> b t c')+position
        for block,attention in zip(encoder.blocks,self.attention):
            normalized=block.attention_norm(x)
            value,_=attention(normalized,normalized,normalized,need_weights=False)
            x=x+value
            x=x+rearrange(block.local(rearrange(block.local_norm(x),'b t c -> b c t')),'b c t -> b t c')
            x=x+block.ff(block.ff_norm(x))
        x=encoder.norm(x)
        ctc=encoder.ctc(x)
        weights=self.model.pool(rearrange(ctc,'b t c -> b c t')).softmax(-1)
        intent=reduce(encoder.intent(x)*rearrange(weights[:,0:1,:],'b 1 t -> b t 1'),'b t c -> b c','sum')
        target=reduce(encoder.target(x)*rearrange(weights[:,1:2,:],'b 1 t -> b t 1'),'b t c -> b c','sum')
        return ctc,intent,target


def prepare_ncnn(root,release,run_id):
    """Cache by bound checkpoint and exporter revision, never mutate selected bundle."""
    from scripts.runtime.speaker_engine import digest
    selection=json.loads((root/'experiments/pcv-release-train-v1/runs'/run_id/'selection-before-calibration.json').read_text())
    selected=selection['selected'];checkpoint=root/selected['checkpoint'].replace('\\','/')
    if digest(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Selected checkpoint identity mismatch')
    onnx=root/release['bundle'].replace('\\','/')/'model.onnx'
    key=hashlib.sha256((digest(checkpoint)+digest(onnx)+digest(__file__)+digest(root/'scripts/runtime/speaker_engine.py')).encode()).hexdigest()[:20]
    folder=root/'.cache/speaker-publication-v2/engines'/key/'ncnn'
    record=folder/'engine.json'
    if record.exists() and json.loads(record.read_text()).get('parity',{}).get('passed'):
        return folder
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(checkpoint),'--onnx',str(onnx),'--output',str(folder)],check=True)
    return folder


def main():
    import pnnx
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--onnx',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(20261002)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=PooledContextVCM(**checkpoint['model_config']);model.load_state_dict(checkpoint['weights']);model.eval()
    wrapper=NCNNEndpoint(model).eval()
    def inputs(n):
        features=torch.randn(1,40,n)
        angle=rearrange(torch.arange((n+1)//2,dtype=torch.float32),'t -> t 1')*rearrange(model.encoder.position_frequency,'d -> 1 d')
        position=rearrange([angle.sin(),angle.cos()],'two t d -> 1 t (d two)')
        return features,position
    errors=[]
    for n in (3,37,128,231):
        f,pos=inputs(n)
        with torch.no_grad():actual=wrapper(f,pos);expected=model(f)
        errors.append(max(float((actual[0]-rearrange(expected[0],'b c t -> b t c')).abs().max()),
                          float((actual[1]-expected[1][:,:,0]).abs().max()),float((actual[2]-expected[2][:,:,0]).abs().max())))
    if max(errors)>2e-5:raise ValueError(f'Endpoint rewrite parity failed: {errors}')
    (args.output/'position-frequency.json').write_text(json.dumps(model.encoder.position_frequency.tolist()))
    (args.output/'rewrite-parity.json').write_text(json.dumps({'errors':errors,'max_absolute_error':max(errors)}))
    torch.backends.mha.set_fastpath_enabled(False)
    pnnx.export(wrapper,(args.output/'model.pt').as_posix(),inputs=inputs(128),inputs2=inputs(231),fp16=False)
    from scripts.runtime.speaker_engine import NCNNSession,digest
    receipt={'engine':'ncnn','version':'1.0.20260526','conversion':'selected FP32 checkpoint; mask-free complete utterance; explicit positional encoding',
             'checkpoint_sha256':digest(args.checkpoint),'onnx_sha256':digest(args.onnx),'files':{name:digest(args.output/name) for name in ('model.ncnn.param','model.ncnn.bin','position-frequency.json')},
             'onnx_note':'ONNX reference is the selected INT8 export; ncnn preserves the same checkpoint in FP32.'}
    (args.output/'engine.json').write_text(json.dumps(receipt,indent=2))
    shutil.copyfile(args.onnx,args.output.parent/'model.onnx')
    # The output directory must be named ncnn, matching the packaged model layout.
    session=NCNNSession(args.output.parent,1);ncnn_errors=[]
    for n in (1,3,16,37,128,231,511,1499):
        f,pos=inputs(n)
        with torch.no_grad():expected=model(f)
        actual=session.run(None,{'features':f.numpy()})
        error=max(float(abs(a-rearrange(b,'b c t -> b t c').numpy()).max()) for a,b in zip(actual,expected[:3]))
        ncnn_errors.append({'frames':n,'max_absolute_error':error})
    repeated=session.run(None,{'features':f.numpy()})
    if any(not np.array_equal(a,b) for a,b in zip(actual,repeated)):raise ValueError('ncnn repeated inference is unstable')
    receipt['repeat_stability']=True
    receipt['parity']={'reference':'selected PyTorch FP32','tolerance':2e-5,'observations':ncnn_errors,'passed':max(x['max_absolute_error'] for x in ncnn_errors)<=2e-5}
    (args.output/'engine.json').write_text(json.dumps(receipt,indent=2))
    if not receipt['parity']['passed']:raise ValueError('ncnn numerical parity failed: '+str(ncnn_errors))
    print(json.dumps(receipt['parity']),flush=True)


if __name__=='__main__':main()
