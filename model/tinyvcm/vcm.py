"""Small causal audio classifier. No ASR model, token decoder, or transcript input."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce, pack


class FrameNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return rearrange(self.norm(rearrange(x, "b c t -> b t c")), "b t c -> b c t")


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, groups=1):
        super().__init__()
        self.left = kernel - 1
        self.inputs = inputs
        self.conv = nn.Conv1d(inputs, outputs, kernel, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))

    def step(self, x, cache):
        joined, _ = pack([cache, x], "b c *")
        return self.conv(joined), joined[:, :, -self.left:]


class ResidualBlock(nn.Module):
    def __init__(self, width, kernel, dropout):
        super().__init__()
        self.depthwise = nn.ModuleList([CausalConv(width, width, kernel, groups=width) for _ in range(2)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1, bias=False) for _ in range(2)])
        self.norms = nn.ModuleList([FrameNorm(width) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        for depth, point, norm in zip(self.depthwise, self.pointwise, self.norms):
            x = self.dropout(F.silu(norm(point(depth(x)))))
        return (x + residual) * (2 ** -0.5)

    def step(self, x, caches):
        residual = x
        next_caches = []
        for depth, point, norm, cache in zip(self.depthwise, self.pointwise, self.norms, caches):
            x, next_cache = depth.step(x, cache)
            next_caches.append(next_cache)
            x = F.silu(norm(point(x)))
        return (x + residual) * (2 ** -0.5), next_caches


class VCM(nn.Module):
    def __init__(self, classes, width=128, kernels=(5, 7, 9, 11, 13), dropout=0.1, pooling="mean"):
        super().__init__()
        self.classes, self.width = classes, width
        self.pooling = pooling
        self.stem = CausalConv(40, width, 5)
        self.stem_norm = FrameNorm(width)
        self.blocks = nn.ModuleList([ResidualBlock(width, k, dropout) for k in kernels])
        self.head = nn.Linear(width * (2 if pooling == "mean_max" else 1), classes)
        self.config = {"classes": classes, "width": width, "kernels": list(kernels), "dropout": dropout, "pooling": pooling}

    def encode(self, x):
        x = F.silu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x, lengths):
        hidden = self.encode(x)
        valid = rearrange(torch.arange(x.shape[-1], device=x.device), "t -> 1 1 t") < rearrange(lengths, "b -> b 1 1")
        pooled = reduce(hidden * valid, "b c t -> b c", "sum") / rearrange(lengths, "b -> b 1")
        if self.pooling == "mean_max":
            maximum = reduce(hidden.masked_fill(~valid, -1e4), "b c t -> b c", "max")
            pooled, _ = pack([pooled, maximum], "b *")
        return self.head(pooled)

    def prefix_logits(self, x):
        hidden = self.encode(x)
        pooled = hidden.cumsum(-1) / rearrange(torch.arange(1, x.shape[-1] + 1, device=x.device), "t -> 1 1 t")
        if self.pooling == "mean_max":
            pooled, _ = pack([pooled, hidden.cummax(-1).values], "b * t")
        return self.head(rearrange(pooled, "b c t -> b t c"))

    def initial_state(self, device="cpu"):
        states = [torch.zeros(1, self.width, device=device), torch.zeros(1, 1, device=device)]
        if self.pooling == "mean_max":
            states.append(torch.full((1, self.width), -1e4, device=device))
        states.append(torch.zeros(1, 40, self.stem.left, device=device))
        for block in self.blocks:
            states.extend(torch.zeros(1, self.width, d.left, device=device) for d in block.depthwise)
        return states

    def step(self, x, states):
        total, count = states[:2]
        offset = 3 if self.pooling == "mean_max" else 2
        x, stem_cache = self.stem.step(x, states[offset])
        x = F.silu(self.stem_norm(x))
        new_caches = [stem_cache]
        cursor = offset + 1
        for block in self.blocks:
            x, caches = block.step(x, states[cursor:cursor + 2])
            cursor += 2
            new_caches.extend(caches)
        total = total + reduce(x, "b c t -> b c", "sum")
        count = count + x.shape[-1]
        pooled = total / count
        next_states = [total, count]
        if self.pooling == "mean_max":
            maximum = torch.maximum(states[2], reduce(x, "b c t -> b c", "max"))
            pooled, _ = pack([pooled, maximum], "b *")
            next_states.append(maximum)
        return self.head(pooled), [*next_states, *new_caches]


class StreamingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *states):
        logits, states = self.model.step(features, states)
        return (logits, *states)
