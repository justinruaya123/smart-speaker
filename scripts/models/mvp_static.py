"""Our contextual encoder with one atomic, finite command output."""
import torch
from torch import nn
from scripts.models.pcv_contextual import ContextualVCM


class StaticCommandNet(nn.Module):
    def __init__(self, classes, width=128):
        super().__init__()
        base = ContextualVCM(width=width, layers=6, heads=4, feedforward=2*width, dropout=.1)
        self.stem, self.blocks, self.norm = base.stem, base.blocks, base.norm
        self.register_buffer('position_frequency', base.position_frequency.clone())
        self.classifier = nn.Linear(width, classes)
        self.config = {'classes': classes, 'width': width}

    def forward(self, features, lengths):
        steps = torch.arange(features.shape[2], device=features.device)[None, :]
        valid_input = steps < lengths[:, None]
        x = self.stem(features * valid_input[:, None, :]).transpose(1, 2)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < ((lengths+1)//2)[:, None]
        positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None] * self.position_frequency[None, :]
        encoding = torch.stack((positions.sin(), positions.cos()), dim=-1).flatten(-2)[None]
        x = (x + encoding) * valid[:, :, None]
        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x) * valid[:, :, None]
        pooled = x.sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return self.classifier(pooled)

    def initialize_our_encoder(self, checkpoint):
        """Declared continuation from our own training; classifier stays random."""
        weights = checkpoint['weights']
        own = self.state_dict()
        transferred = {key: weights['encoder.'+key] for key in own
                       if not key.startswith('classifier.') and 'encoder.'+key in weights}
        if len(transferred) != len(own)-2:
            raise ValueError('Incomplete encoder transfer')
        self.load_state_dict({**own, **transferred})
        return len(transferred)


class ExportStatic(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, features):
        lengths = torch.ones(features.shape[0], dtype=torch.long, device=features.device) * features.shape[2]
        return self.model(features, lengths)
