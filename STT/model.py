"""Small randomly initialized character CTC model for the isolated experiment."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange
from common import CHARS

class AcousticCTC(nn.Module):
    def __init__(self, output_size=len(CHARS)):
        super().__init__()
        self.conv1 = nn.Conv1d(80, 192, 7, padding=3, stride=2)
        self.norm1 = nn.LayerNorm(192)
        self.conv2 = nn.Conv1d(192, 192, 5, padding=2)
        self.norm2 = nn.LayerNorm(192)
        self.rnn = nn.GRU(192, 192, num_layers=3, bidirectional=True, batch_first=True, dropout=.15)
        self.norm3 = nn.LayerNorm(384)
        self.head = nn.Linear(384, output_size)
        self.dropout = nn.Dropout(.15)
        with torch.no_grad():
            self.head.bias[0] = -1

    def forward(self, x, lengths):
        x = rearrange(x, 'batch time mel -> batch mel time')
        x = self.conv1(x)
        x = rearrange(x, 'batch channels time -> batch time channels')
        x = self.dropout(torch.nn.functional.gelu(self.norm1(x)))
        lengths = (lengths + 1) // 2
        valid = torch.arange(x.shape[1], device=x.device) < rearrange(lengths.to(x.device), 'batch -> batch 1')
        # Keep convolution boundary behavior identical for a single recording
        # and the same recording inside a padded batch.
        x = x.masked_fill(~rearrange(valid, 'batch time -> batch time 1'), 0)
        residual = x
        x = self.conv2(rearrange(x, 'batch time channels -> batch channels time'))
        x = rearrange(x, 'batch channels time -> batch time channels')
        x = residual + self.dropout(torch.nn.functional.gelu(self.norm2(x)))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        return self.head(self.dropout(self.norm3(x))), lengths

def greedy_decode(logits, lengths):
    out = []
    ids = logits.argmax(-1).detach().cpu().tolist()
    for row, length in zip(ids, lengths.tolist()):
        last, text = -1, []
        for token in row[:length]:
            if token != last and token:
                text.append(CHARS[token])
            last = token
        out.append(''.join(text).strip())
    return out
