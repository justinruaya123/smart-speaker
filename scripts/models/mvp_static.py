"""Our contextual encoder with one atomic, finite command output."""
import torch
from torch import nn
import math
from einops import rearrange


class ContextBlock(nn.Module):
    """The same six-block encoder, retained without retired acoustic heads."""
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


class StaticCommandNet(nn.Module):
    def __init__(self, classes, width=128):
        super().__init__()
        if width%4 or width%2:raise ValueError('Width must be divisible by four')
        self.stem=nn.Conv1d(40,width,5,stride=2,padding=2)
        self.blocks=nn.ModuleList([ContextBlock(width,4,2*width,.1) for _ in range(6)])
        self.norm=nn.LayerNorm(width)
        self.register_buffer('position_frequency',torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width)))
        self.classifier = nn.Linear(width, classes)
        self.config = {'classes': classes, 'width': width}

    def forward_with_embedding(self, features, lengths):
        """Training-only command invariance; this is not a speaker embedding."""
        steps = torch.arange(features.shape[2], device=features.device)[None, :]
        valid_input = steps < lengths[:, None]
        x = self.stem(features * valid_input[:, None, :]).transpose(1, 2)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < ((lengths+1)//2)[:, None]
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None] * self.position_frequency[None, :]
        encoding = torch.stack((positions.sin(), positions.cos()), dim=-1).flatten(-2)[None]
        x = (x + encoding) * valid[:, :, None]
        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x) * valid[:, :, None]
        pooled = x.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return self.classifier(pooled), pooled

    def forward(self, features, lengths):
        return self.forward_with_embedding(features, lengths)[0]

    def initialize_our_encoder(self, checkpoint):
        """Declared continuation from our own training; classifier stays random."""
        weights = checkpoint['weights']
        own = self.state_dict()
        transferred = {key: weights['encoder.'+key] for key in own
                       if not key.startswith('classifier.') and 'encoder.'+key in weights}
        if len(transferred) != len(own)-2:
            raise ValueError('Incomplete encoder transfer')
        self.load_state_dict({**own, **transferred})
        return len(transferred)


class ExportStatic(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, features):
        lengths = torch.ones(features.shape[0], dtype=torch.long, device=features.device) * features.shape[2]
        return self.model(features, lengths)
