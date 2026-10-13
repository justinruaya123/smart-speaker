"""Random-init direct keyword candidate and training-only speaker invariance.

Class zero is UNKNOWN. Positive labels identify complete fixed demo
outputs, not
broad intents with unresolved slots. This is not the deployed model or a
speaker
identity encoder. Frontend and normalization must reuse the pinned VCM
frontend.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional


def class_speaker_weights(rows: Sequence[Mapping]) -> np.ndarray:
    """Equal mass per class, then per speaker within each class."""
    if not rows:
        raise ValueError("Cannot balance an empty training set")
    counts = Counter()
    speakers = defaultdict(set)
    for row in rows:
        label, speaker = row["label"], row["speaker_id"]
        if not isinstance(speaker, str) or not speaker:
            raise ValueError("Every training row needs a resolved speaker_id")
        counts[label, speaker] += 1
        speakers[label].add(speaker)
    weights = np.array([
        1.0 / (
            len(speakers[row["label"]])
            * counts[row["label"], row["speaker_id"]]
        )
        for row in rows
    ], dtype=np.float64)
    return weights / weights.sum()


def cross_speaker_contrastive(
    embeddings: Tensor, labels: Tensor, speakers: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """SupCon adaptation: same non-UNKNOWN label, different speaker positives.

    Same-label/same-speaker examples are excluded from each anchor's
    denominator,
    rather than incorrectly treated as negatives. UNKNOWN can be a negative but
    never an anchor or a positive. A batch without eligible pairs returns zero.
    """
    if (
        embeddings.ndim != 2
        or labels.shape != embeddings.shape[:1]
        or speakers.shape != labels.shape
    ):
        raise ValueError(
            "Expected [batch, embedding] and aligned label/speaker vectors"
        )
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if labels.dtype not in (
        torch.int32, torch.int64
    ) or speakers.dtype not in (torch.int32, torch.int64):
        raise ValueError("Labels and speaker indices must be integral")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    same_label = labels[:, None] == labels[None, :]
    different_speaker = speakers[:, None] != speakers[None, :]
    positives = same_label & different_speaker & (labels[:, None] > 0)
    eligible = positives.sum(dim=1) > 0
    if not bool(eligible.any()):
        return embeddings.sum() * 0.0
    normalized = functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T / temperature
    denominator = (~same_label) | positives
    selected = similarities[eligible]
    allowed = denominator[eligible]
    positive = positives[eligible]
    log_probabilities = selected - torch.logsumexp(
        selected.masked_fill(~allowed, -torch.inf), dim=1, keepdim=True
    )
    positive_sum = torch.where(positive, log_probabilities, 0.0).sum(dim=1)
    return -(positive_sum / positive.sum(dim=1)).mean()


class _TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            width, width, 5, padding=2 * dilation,
            dilation=dilation, groups=width,
        )
        self.norm = nn.LayerNorm(width)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        residual = values
        values = self.depthwise(values * mask)
        values = functional.silu(
            self.norm(values.transpose(1, 2))
        ).transpose(1, 2)
        values = self.pointwise(values * mask)
        return functional.silu(values + residual) * mask


class DemoKeywordNet(nn.Module):
    """Small bidirectional endpoint model; no streaming or accuracy claim.

    Inputs: frozen 40-bin features [batch, 40, frames], valid lengths [batch].
    Outputs: complete-demo-command logits [batch, classes], keyword embeddings.
    All temporal layers mask padding; per-frame LayerNorm avoids padded batch
    statistics. Embeddings here must never be reused as speaker
    identity evidence.
    """

    def __init__(self, classes: int, width: int = 64):
        super().__init__()
        if (
            type(classes) is not int
            or classes < 2 or type(width) is not int or width < 8
        ):
            raise ValueError("Need UNKNOWN plus commands and width >= 8")
        self.stem = nn.Conv1d(40, width, 1)
        self.blocks = nn.ModuleList(
            _TemporalBlock(width, d) for d in (1, 2, 4, 8)
        )
        self.classifier = nn.Linear(width, classes)

    def forward(
        self, features: Tensor, lengths: Tensor
    ) -> tuple[Tensor, Tensor]:
        if (
            features.ndim != 3
            or features.shape[1] != 40 or features.shape[2] < 1
        ):
            raise ValueError("Expected [batch, 40, nonempty frames]")
        if lengths.shape != features.shape[:1] or lengths.dtype not in (
            torch.int32, torch.int64
        ):
            raise ValueError("Expected one integral length per sample")
        if bool(((lengths < 1) | (lengths > features.shape[2])).any()):
            raise ValueError("Lengths outside feature extent")
        index = torch.arange(features.shape[2], device=features.device)
        mask = (
            index[None, :] < lengths[:, None]
        ).unsqueeze(1).to(features.dtype)
        values = functional.silu(self.stem(features * mask)) * mask
        for block in self.blocks:
            values = block(values, mask)
        pooled = values.sum(dim=2) / lengths[:, None].to(values.dtype)
        return self.classifier(pooled), functional.normalize(pooled, dim=1)
