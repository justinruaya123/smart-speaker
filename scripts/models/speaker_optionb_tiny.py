"""A single <=300k-parameter audio-to-complete-OptionB-command classifier.

Reuse the existing separable temporal CNN. No CTC, transcript, teacher,
transformer, keyword rescue, separate intent head or separate slot head.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from scripts.models.vcm import VCM, valid_frames
from scripts.runtime.optionb_contract import COMMANDS

CONFIG: dict[str, Any] = dict(classes=len(COMMANDS), width=160, kernels=(7, 7, 7, 7),
              dilations=(1, 2, 4, 8), dropout=.1, pooling='mean_max')


class OptionBTinyVCM(nn.Module):
    """20ms temporal stride and one atomic 32-way command/UNKNOWN prediction."""
    def __init__(self):
        super().__init__()
        self.network = VCM(**CONFIG)
        self.config = {**self.network.config, 'input_temporal_stride': 2,
                       'output': 'complete_command_logits', 'unknown_class': 0}
        if len(COMMANDS) != 32 or sum(p.numel() for p in self.parameters()) > 300_000:
            raise ValueError('The complete OptionB classifier must stay within 300k parameters')

    def forward(self, features, lengths):
        valid_frames(features, lengths)  # Validate the unstrided input lengths too.
        return self.network(features[..., ::2], (lengths + 1) // 2)
