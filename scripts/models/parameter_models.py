"""Small randomly initialized parameter models for the human-speech study."""
import torch
from torch import nn
from einops import rearrange, reduce


class KeywordEncoder(nn.Module):
    """Temporal depthwise residual encoder; inspired by DS-CNN/TC-ResNet.

    This is a study implementation, not a reproduction of either paper.
    Masking keeps padded samples out of convolutions and summary statistics.
    """
    def __init__(self, classes, width=96, embedding=96):
        super().__init__()
        self.config = dict(classes=classes, width=width, embedding=embedding)
        self.input = nn.Conv1d(80, width, 7, stride=2, padding=3)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            nn.ModuleList([nn.Conv1d(width, width, 5, padding=2*d, dilation=d, groups=width),
                           nn.Conv1d(width, width, 1), nn.LayerNorm(width)])
            for d in (1, 2, 4, 8, 1, 2)])
        self.projection = nn.Linear(width*2, embedding)
        self.head = nn.Linear(embedding, classes)

    def forward(self, features, lengths):
        lengths = (lengths+1)//2
        x = rearrange(self.input(rearrange(features, 'b t f -> b f t')), 'b c t -> b t c')
        valid = rearrange(torch.arange(x.shape[1], device=x.device), 't -> 1 t') < rearrange(lengths, 'b -> b 1')
        mask = rearrange(valid, 'b t -> b t 1')
        x = nn.functional.gelu(self.norm(x)).masked_fill(~mask, 0.)
        for depthwise, pointwise, norm in self.blocks:
            y = pointwise(depthwise(rearrange(x, 'b t c -> b c t')))
            x = (x + nn.functional.gelu(norm(rearrange(y, 'b c t -> b t c')))).masked_fill(~mask, 0.)
        mean = reduce(x, 'b t c -> b c', 'sum') / rearrange(lengths, 'b -> b 1')
        maximum = reduce(x.masked_fill(~mask, -1e4), 'b t c -> b c', 'max')
        from einops import pack
        summary, _ = pack([mean, maximum], 'b *')
        embedding = nn.functional.normalize(self.projection(summary), dim=-1)
        return self.head(embedding*10), embedding
