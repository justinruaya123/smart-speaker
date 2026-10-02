"""Small Conv2D VCM baseline informed by Speech-to-Intent-Micro; original implementation."""
import torch
from torch import nn
from einops import rearrange,reduce
from .vcm import valid_frames
from .keyword_data import HEADS,VOCAB


class MicroIntent(nn.Module):
    def __init__(self):
        super().__init__()
        channels=[1,16,16,24,48,128]
        self.convs=nn.ModuleList([nn.Conv2d(channels[i],channels[i+1],3,padding=1,bias=False) for i in range(5)])
        self.norms=nn.ModuleList([nn.BatchNorm2d(c) for c in channels[1:]])
        self.pool=nn.MaxPool2d(2,ceil_mode=True)
        self.dense=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2))
        self.intent=nn.Linear(64,len(HEADS['intent']));self.ready=nn.Linear(64,2);self.words=nn.Linear(64,len(VOCAB))

    def forward(self,x,lengths):
        valid=valid_frames(x,lengths);denom=rearrange(lengths,'b -> b 1 1')
        mean=reduce(x*valid,'b c t -> b c 1','sum')/denom
        var=reduce((x-mean).square()*valid,'b c t -> b c 1','sum')/denom
        x=rearrange((x-mean)/var.clamp_min(.01).sqrt()*valid,'b f t -> b 1 f t')
        for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
            x=torch.relu(norm(conv(x)))
            mask=rearrange(torch.arange(x.shape[-1],device=x.device),'t -> 1 1 1 t')<rearrange(lengths,'b -> b 1 1 1')
            x=x*mask
            if i in (1,2,3):x=self.pool(x);lengths=(lengths+1)//2
        mask=rearrange(torch.arange(x.shape[-1],device=x.device),'t -> 1 1 1 t')<rearrange(lengths,'b -> b 1 1 1')
        pooled=self.dense(reduce(x.masked_fill(~mask,-1e4),'b c f t -> b c','max'))
        return self.intent(pooled),self.ready(pooled),self.words(pooled)
