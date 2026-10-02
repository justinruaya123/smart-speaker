"""Small endpoint-only finite VCM with bidirectional acoustic context."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from einops import rearrange
from scripts.runtime.pcv_schema import INTENTS


class ContextBlock(nn.Module):
    def __init__(self,width,heads,feedforward,dropout):
        super().__init__();self.heads=heads;self.scale=(width//heads)**-.5
        self.attention_norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width)
        self.attention_output=nn.Linear(width,width)
        self.local_norm=nn.LayerNorm(width)
        self.local=nn.Sequential(nn.Conv1d(width,width,7,padding=3,groups=width),nn.SiLU(),nn.Conv1d(width,width,1))
        self.ff_norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,feedforward),nn.SiLU(),nn.Dropout(dropout),nn.Linear(feedforward,width))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,valid):
        mask=rearrange(valid,'b t -> b t 1')
        q,k,v=rearrange(self.qkv(self.attention_norm(x)),
                        'b t (three heads dim) -> three b heads t dim',three=3,heads=self.heads).unbind(0)
        logits=torch.matmul(q,rearrange(k,'b h t d -> b h d t'))*self.scale
        logits=logits.masked_fill(~rearrange(valid,'b t -> b 1 1 t'),-10000.)
        attended=torch.matmul(self.dropout(logits.softmax(-1)),v)
        x=(x+self.dropout(self.attention_output(rearrange(attended,'b h t d -> b t (h d)'))))*mask
        local_input=rearrange(self.local_norm(x)*mask,'b t c -> b c t')
        x=(x+self.dropout(rearrange(self.local(local_input),'b c t -> b t c')))*mask
        return (x+self.dropout(self.ff(self.ff_norm(x))))*mask


class ContextualVCM(nn.Module):
    def __init__(self,width=80,layers=6,heads=4,feedforward=160,dropout=.1):
        super().__init__()
        if width%heads or width%2:raise ValueError('Width must be even and divisible by heads')
        self.config={'width':width,'layers':layers,'heads':heads,'feedforward':feedforward,'dropout':dropout}
        self.stem=nn.Conv1d(40,width,5,stride=2,padding=2)
        self.register_buffer('position_frequency',torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width)))
        self.blocks=nn.ModuleList([ContextBlock(width,heads,feedforward,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(width)
        self.ctc=nn.Linear(width,40);self.intent=nn.Linear(width,len(INTENTS)+1);self.target=nn.Linear(width,2)

    def forward(self,features,lengths=None):
        if lengths is None:
            lengths=torch.ones(features.shape[0],dtype=torch.long,device=features.device)*features.shape[2]
        input_valid=rearrange(torch.arange(features.shape[2],device=features.device),'t -> 1 t')<rearrange(lengths,'b -> b 1')
        x=rearrange(self.stem(features*rearrange(input_valid,'b t -> b 1 t')),'b c t -> b t c')
        valid=rearrange(torch.arange(x.shape[1],device=x.device),'t -> 1 t')<rearrange((lengths+1)//2,'b -> b 1')
        positions=rearrange(torch.arange(x.shape[1],device=x.device,dtype=x.dtype),'t -> t 1')*rearrange(self.position_frequency,'d -> 1 d')
        encoding=rearrange([positions.sin(),positions.cos()],'two t d -> 1 t (d two)')
        x=(x+encoding)*rearrange(valid,'b t -> b t 1')
        for block in self.blocks:x=block(x,valid)
        x=self.norm(x)
        return (rearrange(self.ctc(x),'b t c -> b c t'),
                rearrange(self.intent(x),'b t c -> b c t'),
                rearrange(self.target(x),'b t c -> b c t'),[])


class ExportWrapper(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,features):
        return tuple(rearrange(x,'b c t -> b t c') for x in self.model(features)[:3])


def export_onnx(model,path):
    import onnx
    import onnxruntime as ort
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    wrapper=ExportWrapper(model.cpu().eval()).eval();generator=torch.Generator().manual_seed(902)
    sample=torch.randn(1,40,37,generator=generator)
    torch.onnx.export(wrapper,(sample,),str(path),input_names=['features'],output_names=['ctc','intent','target'],
        dynamic_axes={'features':{2:'input_frames'},**{k:{1:'output_frames'} for k in ('ctc','intent','target')}},
        opset_version=17,dynamo=False)
    onnx.checker.check_model(str(path));options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    errors=[]
    for frames in (1,3,16,37,80,121):
        x=torch.randn(1,40,frames,generator=generator)
        with torch.no_grad():expected=wrapper(x)
        actual=session.run(None,{'features':x.numpy()})
        if any(o.shape[1]!=(frames+1)//2 for o in actual):raise ValueError('Contextual output time shape changed')
        errors.append(max(float(np.max(np.abs(a-b.numpy()))) for a,b in zip(actual,expected)))
    return {'pass':max(errors)<2e-5 and path.stat().st_size<2*1024*1024,
        'max_absolute_error':max(errors),'errors':errors,'tolerance':2e-5,
        'parameters':sum(p.numel() for p in model.parameters()),'onnx_bytes':path.stat().st_size,
        'mode':'single complete-utterance inference after VAD endpoint; no streaming caches',
        'input_feature_hop_ms':10,'output_hop_ms':20,'frame_counts':[1,3,16,37,80,121],
        'output_heads':['ctc','intent','target'],'logits':'raw; finite decoder applies log-softmax once'}
