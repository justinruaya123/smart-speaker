"""Scratch-only causal Slot-TCN-96 and independent speaker tower.

No acoustic labels, phone inventory, transcript decoder, or pretrained weights.
Outputs are finite command/value logits and learned acoustic embeddings. Random
initialization is for testing only; it must never enter a deployment package.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

WIDTH = 96
DILATIONS = (1, 2, 4, 8, 16, 32)
POOL_QUERIES = 5
HEADS = ('semantic', 'percent', 'intent', 'operation', 'target', 'light',
         'song', 'artist', 'entity_queries', 'acoustic', 'speaker_frames')
CACHE_SHAPES = ((40, 4), *((WIDTH, 4 * d) for d in DILATIONS))
SPEAKER_SHAPES = ((40, 4), (32, 4), (32, 8), (32, 16))
STATE_COUNT = 14


def initial_pool(batch: int, width: int, device: torch.device,
                 dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    """Five independent attention summaries; empty history is finite."""
    return (
        torch.full((batch, POOL_QUERIES), -1e30, device=device, dtype=dtype),
        torch.zeros(batch, POOL_QUERIES, device=device, dtype=dtype),
        torch.zeros(batch, POOL_QUERIES, width, device=device, dtype=dtype),
    )


def merge_pool(hidden: Tensor, scores: Tensor, state: Sequence[Tensor]
               ) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Stable exact incremental softmax pooling, up to float rounding."""
    maximum, denominator, numerator = state
    next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
    rescale = torch.exp(maximum - next_maximum)
    weights = torch.exp(scores - next_maximum.unsqueeze(-1))
    next_denominator = denominator * rescale + weights.sum(dim=-1)
    next_numerator = (numerator * rescale.unsqueeze(-1)
                      + torch.matmul(weights, hidden.transpose(1, 2)))
    pooled = next_numerator / next_denominator.clamp_min(1e-20).unsqueeze(-1)
    return pooled, (next_maximum, next_denominator, next_numerator)


class CausalBlock(nn.Module):
    def __init__(self, dilation: int, width: int = WIDTH) -> None:
        super().__init__()
        self.history = 4 * dilation
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, dilation=dilation,
                                   groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm(x.transpose(1, 2)).transpose(1, 2)
        context = torch.cat((cache, normalized), dim=-1)
        residual = self.pointwise(F.silu(self.depthwise(context)))
        return x + self.dropout(residual), context[..., -self.history:]


class SpeakerTower(nn.Module):
    """Separate weights prevent speaker and content objectives conflicting.

    Frame embeddings require a separately calibrated voiced-window aggregator.
    They are NOT probabilities, target masks, overlap labels, or voiceprints
    qualified from a single wake word merely by constructing this module.
    """
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(40, 32, 5)
        self.blocks = nn.ModuleList(CausalBlock(d, 32) for d in (1, 2, 4))
        self.projection = nn.Conv1d(32, 64, 1)

    def forward(self, features: Tensor, states: Sequence[Tensor]
                ) -> tuple[Tensor, tuple[Tensor, ...]]:
        context = torch.cat((states[0], features), dim=-1)
        x = F.silu(self.stem(context))
        next_states = [context[..., -4:]]
        for block, cache in zip(self.blocks, states[1:], strict=True):
            x, next_cache = block(x, cache)
            next_states.append(next_cache)
        return self.projection(x), tuple(next_states)


class StreamingVCM(nn.Module):
    """One architecture. Counts describe a frozen vocabulary, not search knobs.

    Default fixtures: light UNKNOWN/all/kitchen/bedroom/bathroom; song
    UNKNOWN/Dynamite/Butter; artist UNKNOWN/BTS. Rebinding real inventories
    requires corresponding labeled data, refitting and new calibration.
    """
    def __init__(self, light_count: int = 5, song_count: int = 3,
                 artist_count: int = 2) -> None:
        super().__init__()
        if any(type(n) is not int or not 2 <= n <= 129
               for n in (light_count, song_count, artist_count)):
            raise ValueError('Invalid frozen vocabulary size')
        self.stem = nn.Conv1d(40, WIDTH, 5)
        self.blocks = nn.ModuleList(CausalBlock(d) for d in DILATIONS)
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = nn.Conv1d(WIDTH, POOL_QUERIES, 1)
        self.semantic = nn.Linear(WIDTH, 33)
        self.percent = nn.Linear(WIDTH, 102)
        self.intent = nn.Linear(WIDTH, 19)
        self.operation = nn.Linear(WIDTH, 5)
        self.target = nn.Linear(WIDTH, 3)
        self.light = nn.Linear(WIDTH, light_count)
        self.song = nn.Linear(WIDTH, song_count)
        self.artist = nn.Linear(WIDTH, artist_count)
        self.embedding = nn.Conv1d(WIDTH, 64, 1)
        self.speaker = SpeakerTower()

    def initial_state(self, batch: int = 1) -> tuple[Tensor, ...]:
        reference = self.stem.weight
        caches = tuple(reference.new_zeros(batch, channels, history)
                       for channels, history in CACHE_SHAPES)
        speaker = tuple(reference.new_zeros(batch, channels, history)
                        for channels, history in SPEAKER_SHAPES)
        return caches + initial_pool(batch, WIDTH, reference.device,
                                     reference.dtype) + speaker

    def encode(self, features: Tensor, caches: Sequence[Tensor]
               ) -> tuple[Tensor, tuple[Tensor, ...]]:
        context = torch.cat((caches[0], features), dim=-1)
        x = F.silu(self.stem(context))
        next_caches = [context[..., -4:]]
        for block, cache in zip(self.blocks, caches[1:], strict=True):
            x, next_cache = block(x, cache)
            next_caches.append(next_cache)
        return (self.norm(x.transpose(1, 2)).transpose(1, 2),
                tuple(next_caches))

    def heads(self, hidden: Tensor, pooled: Tensor) -> dict[str, Tensor]:
        queries = self.embedding(pooled[:, 2:].transpose(1, 2))
        return {
            'semantic': self.semantic(pooled[:, 0]),
            'percent': self.percent(pooled[:, 1]),
            'intent': self.intent(pooled[:, 0]),
            'operation': self.operation(pooled[:, 0]),
            'target': self.target(pooled[:, 2]),
            'light': self.light(pooled[:, 2]),
            'song': self.song(pooled[:, 3]),
            'artist': self.artist(pooled[:, 4]),
            'entity_queries': F.normalize(queries, dim=1).transpose(1, 2),
            'acoustic': self.embedding(hidden),
        }

    def step(self, features: Tensor, state: Sequence[Tensor]
             ) -> tuple[dict[str, Tensor], tuple[Tensor, ...]]:
        if not torch.jit.is_tracing():
            if features.shape[-1] == 0:
                raise ValueError('Expected nonempty feature chunk')
            if len(state) != STATE_COUNT:
                raise ValueError('Wrong state count')
        hidden, caches = self.encode(features, state[:7])
        pooled, statistics = merge_pool(hidden, self.attention(hidden),
                                         state[7:10])
        outputs = self.heads(hidden, pooled)
        outputs['speaker_frames'], speaker = self.speaker(features, state[10:])
        return outputs, caches + statistics + speaker

    def full(self, features: Tensor, lengths: Tensor | None = None
             ) -> dict[str, Tensor]:
        states = self.initial_state(features.shape[0])
        hidden, _ = self.encode(features, states[:7])
        scores = self.attention(hidden)
        if lengths is not None:
            if not torch.jit.is_tracing() and torch.any(
                    (lengths < 1) | (lengths > features.shape[-1])):
                raise ValueError('Invalid feature lengths')
            valid = (torch.arange(features.shape[-1], device=features.device)
                     .unsqueeze(0) < lengths.unsqueeze(1))
            scores = scores.masked_fill(~valid.unsqueeze(1), -1e30)
        pooled = torch.matmul(scores.softmax(-1), hidden.transpose(1, 2))
        result = self.heads(hidden, pooled)
        result['speaker_frames'], _ = self.speaker(features, states[10:])
        return result


class StreamingExport(nn.Module):
    """Flat batch-one candidate ABI. Neither export qualifies random weights.

    Both graphs retain the same state ABI; export dead-code elimination should
    remove unused demo/speaker computations from the benchmark graph. Verify
    that on the actual ONNX graph, not by counting the training module alone.
    """
    def __init__(self, model: StreamingVCM, profile: str) -> None:
        super().__init__()
        if profile not in {'benchmark', 'demo'}:
            raise ValueError('Unknown export profile')
        self.model = model
        self.profile = profile
        self.head_names = HEADS if profile == 'demo' else HEADS[:2]

    def forward(self, features: Tensor, *state: Tensor) -> tuple[Tensor, ...]:
        outputs, next_state = self.model.step(features, state)
        heads = tuple(outputs[key].transpose(1, 2)
                      if key in {'acoustic', 'speaker_frames'}
                      else outputs[key] for key in self.head_names)
        # Pass-through preserves ABI but prunes the unused speaker tower.
        if self.profile == 'benchmark':
            next_state = next_state[:10] + tuple(state[10:])
        return heads + next_state
