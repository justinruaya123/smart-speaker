"""Random-initialization audio/pronunciation embeddings, without a transcript decoder.

The ten acoustic windows pool intermediate frames from an intact utterance.
They are weak localization hypotheses, not verified name boundaries or new audio.
"""
import torch
from torch import nn
from einops import rearrange, reduce


WINDOWS = [(0., 1.), (0., .75), (.25, 1.), (0., .5), (.25, .75),
           (.5, 1.), (0., .25), (.25, .5), (.5, .75), (.75, 1.)]


class TemporalEncoder(nn.Module):
    def __init__(self, inputs, width, embedding, stride, dilations):
        super().__init__()
        self.stride = stride
        self.input = nn.Conv1d(inputs, width, 5, stride=stride, padding=2)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            nn.ModuleList([nn.Conv1d(width, width, 5, padding=2*d, dilation=d, groups=width),
                           nn.Conv1d(width, width, 1), nn.LayerNorm(width)])
            for d in dilations])
        self.projection = nn.Linear(width * 4, embedding)

    def forward(self, features, lengths, windows):
        # Mask input too: padding content must never influence valid features.
        time = rearrange(torch.arange(features.shape[1], device=features.device), 't -> 1 t')
        valid = time < rearrange(lengths, 'b -> b 1')
        features = features.masked_fill(~rearrange(valid, 'b t -> b t 1'), 0.)
        x = rearrange(self.input(rearrange(features, 'b t f -> b f t')), 'b c t -> b t c')
        lengths = (lengths + self.stride - 1) // self.stride
        time = rearrange(torch.arange(x.shape[1], device=x.device), 't -> 1 t')
        valid = time < rearrange(lengths, 'b -> b 1')
        mask = rearrange(valid, 'b t -> b t 1')
        x = nn.functional.gelu(self.norm(x)).masked_fill(~mask, 0.)
        for depthwise, pointwise, norm in self.blocks:
            y = pointwise(depthwise(rearrange(x, 'b t c -> b c t')))
            x = (x + nn.functional.gelu(norm(rearrange(y, 'b c t -> b t c')))).masked_fill(~mask, 0.)
        position = (time + .5) / rearrange(lengths, 'b -> b 1')
        # These are constant pooling boundaries, independent of any query label.
        left = torch.tensor([a+(z-a)*k/4 for a,z in windows for k in range(4)], device=x.device)
        right = torch.tensor([a+(z-a)*(k+1)/4 for a,z in windows for k in range(4)], device=x.device)
        position = rearrange(position, 'b t -> b 1 t')
        use = ((position >= rearrange(left, 'p -> 1 p 1')) &
               (position < rearrange(right, 'p -> 1 p 1')) & rearrange(valid, 'b t -> b 1 t'))
        weights = use.to(x.dtype)
        totals = torch.einsum('bpt,btc->bpc', weights, x)
        counts = reduce(weights, 'b p t -> b p 1', 'sum').clamp(min=1.)
        summary = rearrange(totals/counts, 'b (w p) c -> b w (p c)', w=len(windows), p=4)
        return nn.functional.normalize(self.projection(summary), dim=-1)


class AudioEncoder(nn.Module):
    def __init__(self, classes, width=96, embedding=96):
        super().__init__()
        self.encoder = TemporalEncoder(80, width, embedding, 2, (1, 2, 4, 8, 1, 2))
        self.head = nn.Linear(embedding, classes)

    def forward(self, features, lengths):
        embeddings = self.encoder(features, lengths, WINDOWS)
        return embeddings, self.head(embeddings[:, 0] * 10)


class PronunciationEncoder(nn.Module):
    def __init__(self, vocabulary, width=64, embedding=96):
        super().__init__()
        self.lookup = nn.Embedding(vocabulary, 48, padding_idx=0)
        self.encoder = TemporalEncoder(48, width, embedding, 1, (1, 2, 4))

    def forward(self, tokens, lengths):
        return self.encoder(self.lookup(tokens), lengths, [(0., 1.)])[:, 0]


class SpeechEmbedding(nn.Module):
    def __init__(self, classes, vocabulary, width=96, embedding=96):
        super().__init__()
        self.config = dict(classes=classes, vocabulary=vocabulary, width=width, embedding=embedding)
        self.audio = AudioEncoder(classes, width, embedding)
        self.text = PronunciationEncoder(vocabulary, 64, embedding)


def contrastive(scores, positives):
    """Multi-positive InfoNCE: duplicate phrases must not be false negatives."""
    logp = scores.log_softmax(-1)
    positive = positives.to(logp.dtype)
    return reduce(-logp * positive, 'b q -> b', 'sum') / reduce(positive, 'b q -> b', 'sum').clamp(min=1)
