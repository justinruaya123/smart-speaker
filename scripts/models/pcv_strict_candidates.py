"""Scratch-initialized comparison candidates; not a qualified runtime replacement.

Direct strict command head: 31 semantic outcomes + OOS. The 93-phrase + OOS
head and phoneme CTC head are training auxiliaries / separate extension probes.
No pretrained weights, sentence decoder, or network dependencies.
"""
from __future__ import annotations
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self,width:int,dilation:int):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.depthwise=nn.Conv1d(width,width,7,padding=3*dilation,dilation=dilation,groups=width)
        self.pointwise=nn.Conv1d(width,width,1)
        self.drop=nn.Dropout(.1)
    def forward(self,x,mask):
        z=self.norm(x.transpose(1,2)).transpose(1,2)*mask
        z=self.pointwise(torch.nn.functional.silu(self.depthwise(z)))
        return (x+self.drop(z))*mask


class StrictTinyVCM(nn.Module):
    def __init__(self,kind:str='tcn',width:int=96):
        super().__init__()
        if kind not in {'tcn','crnn'}:raise ValueError('kind must be tcn or crnn')
        if width<8 or width%2:raise ValueError('width must be even and >=8')
        self.kind=kind;self.width=width
        self.stem=nn.Conv1d(40,width,5,stride=2,padding=2)
        if kind=='tcn':
            self.blocks=nn.ModuleList([TemporalBlock(width,d) for d in (1,2,4,8,16,1)])
        else:
            # Width per direction is width/2; output width matches the TCN.
            self.gru=nn.GRU(width,width//2,num_layers=2,batch_first=True,bidirectional=True,dropout=.1)
        self.norm=nn.LayerNorm(width)
        self.attention=nn.Conv1d(width,1,1)
        self.command=nn.Linear(width,32)
        self.phrase=nn.Linear(width,94)
        self.ctc=nn.Conv1d(width,40,1)

    def forward(self,features,lengths=None):
        batch,_,frames=features.shape
        supplied=lengths is not None
        if lengths is None:lengths=torch.full((batch,),frames,dtype=torch.long,device=features.device)
        imask=(torch.arange(frames,device=features.device)[None,:]<lengths[:,None])[:,None,:]
        x=self.stem(features*imask)
        out_lengths=(lengths+1)//2
        mask=(torch.arange(x.shape[-1],device=x.device)[None,:]<out_lengths[:,None])[:,None,:]
        x=x*mask
        if self.kind=='tcn':
            for block in self.blocks:x=block(x,mask)
        else:
            x=x.transpose(1,2)
            if supplied:
                packed=nn.utils.rnn.pack_padded_sequence(x,out_lengths.cpu(),batch_first=True,enforce_sorted=False)
                packed,_=self.gru(packed)
                x,_=nn.utils.rnn.pad_packed_sequence(packed,batch_first=True,total_length=mask.shape[-1])
            else:x,_=self.gru(x)
            x=x.transpose(1,2)*mask
        x=self.norm(x.transpose(1,2)).transpose(1,2)*mask
        attention=self.attention(x).masked_fill(~mask,-10000.).softmax(-1)
        pooled=(x*attention).sum(-1)
        return {'command':self.command(pooled),'phrase':self.phrase(pooled),'ctc':self.ctc(x)}


class StrictExport(nn.Module):
    """Batch-one variable-duration ABI. Explicitly exports only selected heads."""
    def __init__(self,model:StrictTinyVCM,extensions:bool=False):
        super().__init__();self.model=model;self.extensions=extensions
    def forward(self,features):
        out=self.model(features)
        return (out['command'],out['ctc'].transpose(1,2)) if self.extensions else out['command']
