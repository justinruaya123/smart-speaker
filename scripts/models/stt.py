"""Finite-word CTC: a configurable CNN/BiGRU brightness speech recognizer."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange


class BrightnessSTT(nn.Module):
    def __init__(self, vocab_size, width=128, layers=2):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, width=width, layers=layers)
        self.conv1 = nn.Conv1d(80, width, 7, stride=2, padding=3)
        self.norm1 = nn.LayerNorm(width)
        self.conv2 = nn.Conv1d(width, width, 5, padding=2)
        self.norm2 = nn.LayerNorm(width)
        self.rnn = nn.GRU(width, width, num_layers=layers, bidirectional=True,
                          batch_first=True, dropout=.1 if layers > 1 else 0.)
        self.norm3 = nn.LayerNorm(2*width)
        self.head = nn.Linear(2*width, vocab_size)
        self.dropout = nn.Dropout(.1)
        with torch.no_grad():
            self.head.bias[0] = -1.

    def forward(self, features, lengths):
        x = self.conv1(rearrange(features, 'batch time mel -> batch mel time'))
        x = rearrange(x, 'batch channels time -> batch time channels')
        x = self.dropout(nn.functional.gelu(self.norm1(x)))
        lengths = (lengths+1)//2
        valid = rearrange(torch.arange(x.shape[1], device=x.device), 'time -> 1 time') < rearrange(lengths.to(x.device), 'batch -> batch 1')
        x = x.masked_fill(~rearrange(valid, 'batch time -> batch time 1'), 0.)
        residual = x
        x = rearrange(self.conv2(rearrange(x, 'batch time channels -> batch channels time')), 'batch channels time -> batch time channels')
        x = residual + self.dropout(nn.functional.gelu(self.norm2(x)))
        # Training batches are sorted, as required by the ONNX packed-GRU exporter.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        return self.head(self.dropout(self.norm3(x))), lengths
