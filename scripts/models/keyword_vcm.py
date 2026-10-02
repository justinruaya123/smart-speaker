"""Fixed contextual keyword VCM; ordered acoustic cues, no open-vocabulary ASR."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange, reduce, pack
from .vcm import FrameNorm, valid_frames
from .keyword_data import VOCAB, HEADS


class KeywordVCM(nn.Module):
    def __init__(self,width=128):
        super().__init__(); self.width=width
        self.conv1=nn.Conv2d(1,24,3,stride=2,padding=1);self.norm1=nn.LayerNorm(24)
        self.conv2=nn.Conv2d(24,32,3,stride=2,padding=1);self.norm2=nn.LayerNorm(32)
        self.project=nn.Linear(32*20,width)
        self.rnn=nn.LSTM(width,width,num_layers=2,bidirectional=True,batch_first=True,dropout=.15)
        self.ctc=nn.Conv1d(width*2,len(VOCAB),1)
        self.attention=nn.Conv1d(width*2,1,1)
        self.shared=nn.Sequential(nn.LayerNorm(width*4),nn.Linear(width*4,width),nn.SiLU(),nn.Dropout(.2))
        self.intent=nn.Linear(width,len(HEADS['intent']));self.ready=nn.Linear(width,2)
        with torch.no_grad():self.ctc.bias[0]=-1.

    def forward(self,x,lengths):
        valid=valid_frames(x,lengths)
        divisor=rearrange(lengths,'b -> b 1 1')
        mean=reduce(x*valid,'b c t -> b c 1','sum')/divisor
        var=reduce((x-mean).square()*valid,'b c t -> b c 1','sum')/divisor
        x=(x-mean)/var.clamp_min(.01).sqrt()*valid
        x=self.conv1(rearrange(x,'b f t -> b 1 f t'))
        x=rearrange(nn.functional.silu(self.norm1(rearrange(x,'b c f t -> b f t c'))),'b f t c -> b c f t')
        middle_lengths=(lengths+1)//2
        mask=rearrange(torch.arange(x.shape[-1],device=x.device),'t -> 1 1 1 t')<rearrange(middle_lengths,'b -> b 1 1 1')
        x=self.conv2(x*mask)
        x=nn.functional.silu(self.norm2(rearrange(x,'b c f t -> b f t c')))
        x=nn.functional.silu(self.project(rearrange(x,'b f t c -> b t (c f)')))
        out_lengths=(lengths+3)//4
        sorted_lengths,order=torch.sort(out_lengths,descending=True)
        packed=pack_padded_sequence(x.index_select(0,order),sorted_lengths.cpu(),batch_first=True,enforce_sorted=True)
        packed,_=self.rnn(packed)
        x,_=pad_packed_sequence(packed,batch_first=True,total_length=x.shape[1])
        x=rearrange(x.index_select(0,torch.argsort(order)),'b t c -> b c t')
        valid=valid_frames(x,out_lengths);x=x*valid
        weight=self.attention(x).masked_fill(~valid,-1e4).softmax(-1)
        attended=reduce(x*weight,'b c t -> b c','sum')
        maximum=reduce(x.masked_fill(~valid,-1e4),'b c t -> b c','max')
        pooled,_=pack([attended,maximum],'b *')
        pooled=self.shared(pooled)
        return self.intent(pooled),self.ready(pooled),rearrange(self.ctc(x),'b c t -> b t c'),out_lengths
