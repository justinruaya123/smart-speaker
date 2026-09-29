"""One acoustic backbone pass for the original intents and finite parameter heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import pack, reduce

from .vcm import build_vcm


def masked_head_losses(outputs, targets, masks):
    """Normalize each task by its own labels, including a differentiable empty loss."""
    losses = {}
    for name, logits in outputs.items():
        mask = masks[name]
        if mask.any():
            losses[name] = F.cross_entropy(logits[mask], targets[name][mask])
        else:
            losses[name] = reduce(logits, "b c ->", "sum") * 0.
    return losses


class SlotHeads(nn.Module):
    def __init__(self, features, sizes, classes, hidden=192, adapt_intent=False):
        super().__init__()
        self.names = list(sizes)
        self.project = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, hidden), nn.SiLU(), nn.Dropout(.15))
        self.slots = nn.ModuleDict({name: nn.Linear(hidden, size) for name, size in sizes.items()})
        self.adapt_intent = adapt_intent
        self.intent = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, hidden), nn.SiLU(), nn.Dropout(.15), nn.Linear(hidden, classes))
        nn.init.zeros_(self.intent[-1].weight)
        nn.init.zeros_(self.intent[-1].bias)
        if not adapt_intent:
            for parameter in self.intent.parameters():
                parameter.requires_grad_(False)

    def forward(self, features, original_logits):
        hidden = self.project(features)
        intent = original_logits + self.intent(features) if self.adapt_intent else original_logits
        return (intent, *[self.slots[name](hidden) for name in self.names])


class SlotVCM(nn.Module):
    def __init__(self, backbone_config, vocabulary, hidden=192, adapt_intent=False):
        super().__init__()
        self.backbone = build_vcm(backbone_config)
        members = list(self.backbone.members) if hasattr(self.backbone, "members") else [self.backbone]
        width = sum(member.head.in_features for member in members)
        self.heads = SlotHeads(width, {k: len(v) for k, v in vocabulary.items()}, members[0].classes, hidden, adapt_intent)
        self.config = {"backbone_config": backbone_config, "vocabulary": vocabulary, "hidden": hidden, "adapt_intent": adapt_intent}

    def encode(self, x, lengths):
        members = list(self.backbone.members) if hasattr(self.backbone, "members") else [self.backbone]
        embeddings = [member.embedding(x, lengths) for member in members]
        logits = [member.classify_embedding(value) for member, value in zip(members, embeddings)]
        joined, _ = pack(embeddings, "b *")
        combined, _ = pack(logits, "b c *")
        return joined, reduce(combined, "b c m -> b c", "mean")

    def forward(self, x, lengths):
        return self.heads(*self.encode(x, lengths))
