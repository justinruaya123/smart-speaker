"""Independent random-initialized identity/overlap network for Iris.

This network consumes acoustic features and speaker labels. It shares neither
weights nor embeddings with the command model and never predicts text.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange


class _OwnerBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 5, padding=2 * dilation,
                                   dilation=dilation, groups=width)
        self.norm = nn.LayerNorm(width)
        self.expand = nn.Conv1d(width, width * 2, 1)
        self.project = nn.Conv1d(width * 2, width, 1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        hidden = self.depthwise(x * mask)
        hidden = rearrange(self.norm(rearrange(hidden, 'b c t -> b t c')), 'b t c -> b c t')
        hidden = self.project(F.silu(self.expand(F.silu(hidden))))
        return F.silu(x + hidden) * mask


class OwnerEncoder(nn.Module):
    """Temporal depthwise encoder, masked mean/std, 128-D speaker embedding.

    The training-only cosine identity classifier is omitted from the runtime
    graph. The independent overlap head consumes pooled acoustic statistics.
    """
    def __init__(self, speakers: int, width: int = 64):
        super().__init__()
        if speakers < 2 or width < 8:
            raise ValueError('Need at least two training speakers and width >= 8')
        self.width, self.speakers = width, speakers
        self.stem = nn.Conv1d(40, width, 1)
        self.blocks = nn.ModuleList(_OwnerBlock(width, d) for d in (1, 2, 4, 8, 1, 2))
        self.embedding = nn.Linear(width * 2, 128)
        self.identity = nn.Linear(128, speakers, bias=False)
        self.overlap = nn.Sequential(nn.Linear(width * 2, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask = (torch.arange(features.shape[2], device=features.device)[None, :] < lengths[:, None])
        mask = mask[:, None, :].to(features.dtype)
        x = F.silu(self.stem(features * mask)) * mask
        for block in self.blocks:
            x = block(x, mask)
        count = mask.sum(dim=2).clamp_min(1)
        mean = (x * mask).sum(dim=2) / count
        variance = ((x - mean[:, :, None]).square() * mask).sum(dim=2) / count
        pooled = torch.cat((mean, variance.clamp_min(1e-5).sqrt()), dim=1)
        embedding = F.normalize(self.embedding(pooled), dim=1, eps=1e-8)
        logits = F.linear(embedding, F.normalize(self.identity.weight, dim=1)) * 16
        return logits, embedding, self.overlap(pooled).squeeze(1)


def speaker_metric_loss(embeddings: Tensor, speakers: Tensor, temperature: float = .12) -> Tensor:
    """Supervised contrastive metric term; positives are independent identity rows."""
    normalized = F.normalize(embeddings, dim=1)
    scores = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(speakers), device=speakers.device, dtype=torch.bool)
    positives = (speakers[:, None] == speakers[None, :]) & ~diagonal
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return embeddings.sum() * 0
    log_probability = scores - torch.logsumexp(scores.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    return -(torch.where(positives, log_probability, 0).sum(dim=1)[valid]
             / positives.sum(dim=1)[valid]).mean()


class OwnerExport(nn.Module):
    def __init__(self, model: OwnerEncoder):
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        _, embedding, overlap = self.model(features, lengths)
        return embedding, torch.sigmoid(overlap)
