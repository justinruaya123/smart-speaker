"""One acoustic backbone pass for the original intents and finite parameter heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import pack, reduce

from .vcm import build_vcm, valid_frames


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
    def __init__(self, features, sizes, classes, hidden=192, adapt_intent=False, frame_features=0):
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
        self.attended_names = [name for name in ("target", "operation", "amount") if name in sizes] if frame_features else []
        if self.attended_names:
            self.attention = nn.Conv1d(frame_features, len(self.attended_names), 1)
            self.temporal = nn.ModuleDict({name: nn.Sequential(
                nn.LayerNorm(frame_features), nn.Linear(frame_features, hidden), nn.SiLU(),
                nn.Dropout(.15), nn.Linear(hidden, sizes[name])) for name in self.attended_names})
            # Start as the original checkpoint; learn acoustic residuals without
            # changing family logits, vocabulary, routing or default amounts.
            for head in self.temporal.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, features, original_logits, frames=None, lengths=None):
        hidden = self.project(features)
        intent = original_logits + self.intent(features) if self.adapt_intent else original_logits
        slots = {name: self.slots[name](hidden) for name in self.names}
        if self.attended_names:
            weights = self.attention(frames).masked_fill(~valid_frames(frames, lengths), -1e4).softmax(-1)
            contexts = torch.einsum("bkt,bct->bkc", weights, frames)
            for index, name in enumerate(self.attended_names):
                slots[name] = slots[name] + self.temporal[name](contexts[:, index])
        return (intent, *[slots[name] for name in self.names])


class SlotVCM(nn.Module):
    def __init__(self, backbone_config, vocabulary, hidden=192, adapt_intent=False, slot_attention=False):
        super().__init__()
        self.backbone = build_vcm(backbone_config)
        members = list(self.backbone.members) if hasattr(self.backbone, "members") else [self.backbone]
        width = sum(member.head.in_features for member in members)
        self.slot_attention = slot_attention
        frame_features = sum(member.width for member in members) if slot_attention else 0
        self.heads = SlotHeads(width, {k: len(v) for k, v in vocabulary.items()}, members[0].classes, hidden, adapt_intent, frame_features)
        self.config = {"backbone_config": backbone_config, "vocabulary": vocabulary, "hidden": hidden, "adapt_intent": adapt_intent}
        if slot_attention:
            self.config["slot_attention"] = True

    def encode(self, x, lengths):
        members = list(self.backbone.members) if hasattr(self.backbone, "members") else [self.backbone]
        if self.slot_attention:
            frames = [member.encode(x) for member in members]
            embeddings = [member.pool_embedding(value, lengths) for member, value in zip(members, frames)]
        else:
            embeddings = [member.embedding(x, lengths) for member in members]
        logits = [member.classify_embedding(value) for member, value in zip(members, embeddings)]
        joined, _ = pack(embeddings, "b *")
        combined, _ = pack(logits, "b c *")
        encoded = joined, reduce(combined, "b c m -> b c", "mean")
        if self.slot_attention:
            frame_features, _ = pack(frames, "b * t")
            return (*encoded, frame_features, lengths)
        return encoded

    def forward(self, x, lengths):
        return self.heads(*self.encode(x, lengths))
