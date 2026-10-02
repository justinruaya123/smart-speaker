"""Finite endpoint VCM with learned utterance pooling for semantic heads.

The CTC acoustic inventory and finite decoder are unchanged. Two attention
distributions summarize intent and explicit-light targeting across the complete
utterance. Repeating the summaries preserves the existing three-head ONNX ABI.
"""
import torch
from torch import nn
from einops import rearrange,reduce,repeat
from scripts.models.pcv_contextual import ContextualVCM


class PooledContextVCM(nn.Module):
    def __init__(self,width=192,layers=6,heads=4,feedforward=384,dropout=.1):
        super().__init__()
        self.encoder=ContextualVCM(width,layers,heads,feedforward,dropout)
        self.config=dict(self.encoder.config)
        self.pool=nn.Conv1d(40,2,1)
        nn.init.zeros_(self.pool.weight);nn.init.zeros_(self.pool.bias)

    def forward(self,features,lengths=None):
        ctc,intent,target,_=self.encoder(features,lengths)
        weights=self.pool(ctc)
        if lengths is not None:
            valid=rearrange(torch.arange(ctc.shape[2],device=ctc.device),'t -> 1 t')<rearrange((lengths+1)//2,'b -> b 1')
            weights=weights.masked_fill(~rearrange(valid,'b t -> b 1 t'),-10000.)
        weights=weights.softmax(-1)
        pooled=[]
        for index,value in enumerate((intent,target)):
            summary=reduce(value*weights[:,index:index+1,:],'b c t -> b c','sum')
            pooled.append(repeat(summary,'b c -> b c t',t=ctc.shape[2]))
        return ctc,pooled[0],pooled[1],[]
