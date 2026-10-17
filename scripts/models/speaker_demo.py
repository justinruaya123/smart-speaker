"""Training-only command invariance and speaker-balanced weighting helpers."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
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
