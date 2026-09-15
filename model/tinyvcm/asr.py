"""Independent character-CTC speech recognizer for parameter transcription."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .vcm import FrameNorm

from .text import VOCAB, decode_greedy


class ASRBlock(nn.Module):
    def __init__(self, width, kernel, repeat=3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv1d(width, width, kernel, padding=kernel // 2, groups=width, bias=False), nn.Conv1d(width, width, 1, bias=False), FrameNorm(width), nn.SiLU(), nn.Dropout(0.1)) for _ in range(repeat)])

    def forward(self, x, valid=None):
        residual = x
        for layer in self.layers:
            x = layer(x)
            if valid is not None:
                x = x * valid
        return (x + residual) * (2 ** -0.5)


class ParameterASR(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), width=256, kernels=(11, 13, 15, 17, 19, 21), repeat=3, mask_padding=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(40, width, 5, stride=2, padding=2, bias=False), FrameNorm(width), nn.SiLU())
        self.blocks = nn.ModuleList([ASRBlock(width, k, repeat) for k in kernels])
        self.head = nn.Conv1d(width, vocab_size, 1)
        self.mask_padding = mask_padding
        self.config = {"vocab_size": vocab_size, "width": width, "kernels": list(kernels), "repeat": repeat, "mask_padding": mask_padding}

    def forward(self, x, lengths=None):
        x = self.stem(x)
        valid = None
        if self.mask_padding and lengths is not None:
            valid = rearrange(torch.arange(x.shape[-1], device=x.device), "t -> 1 1 t") < rearrange(self.output_lengths(lengths), "b -> b 1 1")
            x = x * valid
        for block in self.blocks:
            x = block(x, valid)
        return rearrange(self.head(x), "b c t -> t b c")

    @staticmethod
    def output_lengths(lengths):
        return (lengths + 1) // 2


def encode_text(text):
    from .data import normalize_text
    lookup = {c: i for i, c in enumerate(VOCAB)}
    return [lookup[c] for c in normalize_text(text)]
