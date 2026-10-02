"""Research-only causal residual VCM; no unconstrained text decoder."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange, pack

from scripts.runtime.pcv_schema import INTENTS


class ChannelNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return rearrange(self.norm(rearrange(x, 'b c t -> b t c')), 'b t c -> b c t')


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.before = nn.Sequential(ChannelNorm(width), nn.Conv1d(width, width, 1), nn.SiLU())
        self.temporal = nn.Conv1d(width, width, 5, dilation=dilation, groups=width)
        self.after = nn.Sequential(ChannelNorm(width), nn.SiLU(), nn.Conv1d(width, width, 1))
        self.context = 4 * dilation

    def forward(self, x, state):
        joined, _ = pack([state, self.before(x)], 'b c *')
        return x + 0.5 * self.after(self.temporal(joined)), joined[..., -self.context:]


class ResidualVCM(nn.Module):
    def __init__(self, width=128, dilations=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16)):
        super().__init__()
        self.config = {'width': width, 'dilations': list(dilations)}
        self.state_shapes = [(1, 40, 2), *[(1, width, 4*d) for d in dilations]]
        self.stem = nn.Conv1d(40, width, 3)
        self.blocks = nn.ModuleList([ResidualBlock(width, d) for d in dilations])
        self.final_norm = ChannelNorm(width)
        self.ctc = nn.Conv1d(width, 40, 1)
        self.intent = nn.Conv1d(width, len(INTENTS)+1, 1)
        self.target = nn.Conv1d(width, 2, 1)

    def initial_states(self, batch=1, device='cpu'):
        return [torch.zeros(batch, c, n, device=device) for _, c, n in self.state_shapes]

    def forward(self, features, states=None):
        if states is None:
            states = self.initial_states(features.shape[0], features.device)
        joined, _ = pack([states[0], features], 'b c *')
        next_states = [joined[..., -2:]]
        x = self.stem(joined)
        for block, state in zip(self.blocks, states[1:]):
            x, next_state = block(x, state)
            next_states.append(next_state)
        x = self.final_norm(x)
        return self.ctc(x), self.intent(x), self.target(x), next_states


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, *states):
        ctc, intent, target, next_states = self.model(features, states)
        return (rearrange(ctc, 'b c t -> b t c'), rearrange(intent, 'b c t -> b t c'),
                rearrange(target, 'b c t -> b t c'), *next_states)


def export_onnx(model, path):
    import onnx
    import onnxruntime as ort
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model.cpu().eval())
    generator = torch.Generator().manual_seed(921)
    names = ['features'] + [f'state_{i}' for i in range(len(model.state_shapes))]
    outputs = ['ctc', 'intent', 'target'] + [f'next_state_{i}' for i in range(len(model.state_shapes))]
    inputs = (torch.randn(1, 40, 4, generator=generator), *[torch.randn(s, generator=generator) for s in model.state_shapes])
    torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=outputs,
                      dynamic_axes={'features': {2: 'frames'}, **{k: {1: 'frames'} for k in outputs[:3]}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    errors = []
    for frames in (1, 3, 4, 8, 17, 41):
        sample = (torch.randn(1, 40, frames, generator=generator), *[torch.randn(s, generator=generator) for s in model.state_shapes])
        with torch.no_grad():
            expected = wrapper(*sample)
        actual = session.run(None, {n: x.numpy() for n, x in zip(names, sample, strict=True)})
        errors.append(max(float(np.max(np.abs(a-b.numpy()))) for a, b in zip(actual, expected, strict=True)))
    passed = max(errors) < 2e-5 and path.stat().st_size < 2*1024*1024
    return {'pass': passed, 'max_absolute_error': max(errors), 'errors': errors, 'tolerance': 2e-5,
            'frame_counts': [1, 3, 4, 8, 17, 41], 'nonzero_cache_tested': True,
            'state_count': len(model.state_shapes), 'state_shapes': model.state_shapes,
            'parameters': sum(p.numel() for p in model.parameters()), 'onnx_bytes': path.stat().st_size,
            'output_heads': outputs[:3], 'logits': 'raw; runtime applies log-softmax once'}
