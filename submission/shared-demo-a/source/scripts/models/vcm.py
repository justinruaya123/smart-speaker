"""Small causal audio classifier. No ASR model, token decoder, or transcript input."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce, pack


def valid_frames(x, lengths):
    """Validate eager inputs; exported callers must supply 1 <= length <= frames."""
    if not torch.jit.is_tracing():
        if (x.ndim != 3 or lengths.ndim != 1 or lengths.shape[0] != x.shape[0]
                or lengths.dtype not in {torch.int32, torch.int64}
                or torch.any(lengths < 1) or torch.any(lengths > x.shape[-1])):
            raise ValueError("Each utterance requires 1 <= valid length <= frame count")
    return rearrange(torch.arange(x.shape[-1], device=x.device), "t -> 1 1 t") < rearrange(lengths, "b -> b 1 1")


class FrameNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return rearrange(self.norm(rearrange(x, "b c t -> b t c")), "b t c -> b c t")


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel, groups=1, dilation=1):
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.inputs = inputs
        self.conv = nn.Conv1d(inputs, outputs, kernel, groups=groups, dilation=dilation, bias=False)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))

    def step(self, x, cache):
        joined, _ = pack([cache, x], "b c *")
        return self.conv(joined), joined[:, :, -self.left:]


class ResidualBlock(nn.Module):
    def __init__(self, width, kernel, dropout, dilation=1):
        super().__init__()
        self.depthwise = nn.ModuleList([CausalConv(width, width, kernel, groups=width, dilation=dilation) for _ in range(2)])
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
    def __init__(self, classes, width=128, kernels=(5, 7, 9, 11, 13), dropout=0.1, pooling="mean", dilations=None, keyword_classes=0, evidence_gate=None):
        super().__init__()
        self.classes, self.width = classes, width
        self.pooling = pooling
        if pooling not in {"mean", "mean_max", "attention"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        dilations = list(dilations or [1] * len(kernels))
        if len(dilations) != len(kernels) or any(d < 1 for d in dilations):
            raise ValueError("Provide one positive dilation per convolution block")
        self.stem = CausalConv(40, width, 5)
        self.stem_norm = FrameNorm(width)
        self.blocks = nn.ModuleList([ResidualBlock(width, k, dropout, d) for k, d in zip(kernels, dilations)])
        if pooling == "attention":
            self.attention = nn.Conv1d(width, 1, 1)
        self.head = nn.Linear(width * (2 if pooling == "mean_max" else 1), classes)
        self.config = {"classes": classes, "width": width, "kernels": list(kernels), "dropout": dropout, "pooling": pooling, "dilations": dilations}
        if keyword_classes:
            self.keyword_head = nn.Linear(self.head.in_features, keyword_classes)
            self.config["keyword_classes"] = keyword_classes
        self.evidence_gate = evidence_gate
        if evidence_gate:
            if len(evidence_gate["words"]) != keyword_classes or len(evidence_gate["intents"]) != classes:
                raise ValueError("Evidence vocabulary must match both trained heads")
            self.config["evidence_gate"] = evidence_gate

    def attention_weights(self, hidden):
        # Bounded positive weights keep full, prefix, and cached streaming sums identical.
        return self.attention(hidden).clamp(-10, 10).exp()

    def encode(self, x):
        x = F.silu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def embedding(self, x, lengths):
        return self.pool_embedding(self.encode(x), lengths)

    def pool_embedding(self, hidden, lengths):
        """Reuse one encoded sequence for family pooling and optional finite heads."""
        valid = valid_frames(hidden, lengths)
        pooled = reduce(hidden * valid, "b c t -> b c", "sum") / rearrange(lengths, "b -> b 1")
        if self.pooling == "attention":
            weights = self.attention_weights(hidden) * valid
            pooled = reduce(hidden * weights, "b c t -> b c", "sum") / reduce(weights, "b c t -> b c", "sum").clamp_min(1e-8)
        if self.pooling == "mean_max":
            maximum = reduce(hidden.masked_fill(~valid, -1e4), "b c t -> b c", "max")
            pooled, _ = pack([pooled, maximum], "b *")
        return pooled

    def forward(self, x, lengths):
        return self.classify_embedding(self.embedding(x, lengths))

    def classify_embedding(self, pooled):
        logits = self.head(pooled)
        if not self.evidence_gate:
            return logits
        word_probabilities = self.keyword_head(pooled).sigmoid()
        words = self.evidence_gate["words"]
        def any_word(names):
            indices = [words.index(w) for w in names.split() if w in words]
            return reduce(word_probabilities[..., indices], "... k -> ...", "max") if indices else torch.zeros_like(word_probabilities[..., 0])
        light = any_word("light lights lamp lamps brightness")
        volume_absent = 1 - any_word("volume")
        up = torch.maximum(any_word("brighten brighter"), any_word("up increase raise more higher") * light)
        down = torch.maximum(any_word("dim dimmer dimming darker darken"), any_word("down decrease lower less reduce") * torch.maximum(light, any_word("bright brighter")))
        up = up * (1 - any_word("less down decrease lower dim")) * volume_absent
        down = down * volume_absent
        color = any_word("color colors colour colours red orange yellow green blue purple pink cyan white warm cool")
        color = color * (1 - any_word("play music song songs what whether confirm"))
        evidence = {"BRIGHTNESS_UP": up, "BRIGHTNESS_DOWN": down, "COLOR_SET": color}
        adjustments = []
        for label in self.evidence_gate["intents"]:
            if label in evidence:
                p = evidence[label].clamp(.01, .99)
                adjustments.append(p.log() - (1-p).log())
            else:
                adjustments.append(torch.zeros_like(word_probabilities[..., 0]))
        return logits + self.evidence_gate["strength"] * rearrange(adjustments, "c ... -> ... c")

    def forward_with_keywords(self, x, lengths):
        """Training-only word supervision. The runtime still takes audio alone."""
        pooled = self.embedding(x, lengths)
        return self.classify_embedding(pooled), self.keyword_head(pooled)

    def prefix_logits(self, x):
        hidden = self.encode(x)
        pooled = hidden.cumsum(-1) / rearrange(torch.arange(1, x.shape[-1] + 1, device=x.device), "t -> 1 1 t")
        if self.pooling == "attention":
            weights = self.attention_weights(hidden)
            pooled = (hidden * weights).cumsum(-1) / weights.cumsum(-1).clamp_min(1e-8)
        if self.pooling == "mean_max":
            pooled, _ = pack([pooled, hidden.cummax(-1).values], "b * t")
        return self.classify_embedding(rearrange(pooled, "b c t -> b t c"))

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
        if self.pooling == "attention":
            weights = self.attention_weights(x)
            total = total + reduce(x * weights, "b c t -> b c", "sum")
            count = count + reduce(weights, "b c t -> b c", "sum")
        else:
            total = total + reduce(x, "b c t -> b c", "sum")
            count = count + x.shape[-1]
        pooled = total / count
        next_states = [total, count]
        if self.pooling == "mean_max":
            maximum = torch.maximum(states[2], reduce(x, "b c t -> b c", "max"))
            pooled, _ = pack([pooled, maximum], "b *")
            next_states.append(maximum)
        return self.classify_embedding(pooled), [*next_states, *new_caches]


class StreamingExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *states):
        logits, states = self.model.step(features, states)
        return (logits, *states)


class EnsembleVCM(nn.Module):
    """Average independent acoustic classifiers with the same input normalization."""
    def __init__(self, ensemble):
        super().__init__()
        self.members = nn.ModuleList([VCM(**config) for config in ensemble])
        if len(self.members) < 2 or len({m.classes for m in self.members}) != 1:
            raise ValueError("An ensemble requires at least two classifiers with the same vocabulary size")
        self.config = {"ensemble": ensemble}
        self.state_counts = [len(m.initial_state()) for m in self.members]

    def forward(self, x, lengths):
        logits, _ = pack([m(x, lengths) for m in self.members], "b c *")
        return reduce(logits, "b c n -> b c", "mean")

    def prefix_logits(self, x):
        logits, _ = pack([m.prefix_logits(x) for m in self.members], "b t c *")
        return reduce(logits, "b t c n -> b t c", "mean")

    def initial_state(self, device="cpu"):
        return [state for member in self.members for state in member.initial_state(device)]

    def step(self, x, states):
        outputs, updated = [], []
        cursor = 0
        for member, count in zip(self.members, self.state_counts):
            logits, next_states = member.step(x, states[cursor:cursor + count])
            outputs.append(logits)
            updated.extend(next_states)
            cursor += count
        logits, _ = pack(outputs, "b c *")
        return reduce(logits, "b c n -> b c", "mean"), updated


def build_vcm(config):
    return EnsembleVCM(**config) if "ensemble" in config else VCM(**config)
