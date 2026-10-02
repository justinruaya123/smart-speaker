"""Offline finite-command teacher head; never part of the Pi runtime."""
import torch
from torch import nn
from scripts.models.pcv_residual import ChannelNorm, ResidualBlock
from scripts.runtime.pcv_schema import INTENTS


class FiniteTeacherHead(nn.Module):
    def __init__(self, width=256, dilations=(1, 2, 4, 8, 1, 2)):
        super().__init__()
        self.config = {'width': width, 'dilations': list(dilations)}
        self.stem = nn.Sequential(ChannelNorm(768), nn.Conv1d(768, width, 1))
        self.blocks = nn.ModuleList([ResidualBlock(width, d) for d in dilations])
        self.final_norm = ChannelNorm(width)
        self.ctc = nn.Conv1d(width, 40, 1)
        self.intent = nn.Conv1d(width, len(INTENTS)+1, 1)
        self.target = nn.Conv1d(width, 2, 1)

    def forward(self, features):
        x = self.stem(features)
        for block in self.blocks:
            state = torch.zeros(x.shape[0], x.shape[1], block.context,
                                device=x.device, dtype=x.dtype)
            x, _ = block(x, state)
        x = self.final_norm(x)
        return self.ctc(x), self.intent(x), self.target(x), []
