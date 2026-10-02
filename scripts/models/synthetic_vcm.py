"""Fixed acoustic VCM with finite attended slots and auxiliary keyword spotting."""
import torch
from torch import nn
from einops import rearrange, reduce, pack
from .vcm import VCM, valid_frames
from .synthetic_data import HEADS, WORDS


class NumberHead(nn.Module):
    """Compositional finite 0..100 classifier; no text decoder or new classes."""
    def __init__(self, width):
        super().__init__()
        self.tens = nn.Linear(width, 11)
        self.units = nn.Linear(width, 10)
        self.none = nn.Linear(width, 1)
        self.residual = nn.Linear(width, 101)
        self.register_buffer("ten_index", torch.tensor([n//10 for n in range(101)]))
        self.register_buffer("unit_index", torch.tensor([n%10 for n in range(101)]))

    def forward(self, x):
        numeric = self.tens(x).index_select(-1, self.ten_index) + self.units(x).index_select(-1, self.unit_index) + self.residual(x)
        combined, _ = pack([self.none(x), numeric], "b *")
        return combined


class FixedVCM(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.encoder = VCM(len(HEADS["intent"]), width=width, kernels=(5, 5, 7, 7, 9),
                           dilations=(1, 2, 4, 8, 16), pooling="mean_max", dropout=.15)
        self.names = list(HEADS)
        self.attention = nn.Conv1d(width, len(self.names), 1)
        self.project = nn.ModuleDict({name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Dropout(.15)) for name in self.names})
        self.heads = nn.ModuleDict({name: NumberHead(width) if name == "amount" else nn.Linear(width, len(values)) for name, values in HEADS.items()})
        self.keywords = nn.Linear(width*2, len(WORDS))

    def representation(self, x, lengths):
        valid = valid_frames(x, lengths)
        denom = rearrange(lengths, "b -> b 1 1")
        mean = reduce(x * valid, "b c t -> b c 1", "sum") / denom
        variance = reduce((x-mean).square() * valid, "b c t -> b c 1", "sum") / denom
        x = (x-mean) / variance.clamp_min(.01).sqrt() * valid
        hidden = self.encoder.encode(x)
        weights = self.attention(hidden).masked_fill(~valid, -1e4).softmax(-1)
        contexts = torch.einsum("bht,bct->bhc", weights, hidden)
        pooled = self.encoder.pool_embedding(hidden, lengths)
        return contexts, pooled

    def forward(self, x, lengths):
        contexts, pooled = self.representation(x, lengths)
        outputs = [self.heads[name](self.project[name](contexts[:, i])) for i, name in enumerate(self.names)]
        return (*outputs, self.keywords(pooled))
