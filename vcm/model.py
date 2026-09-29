"""Training/export only. 113,064 randomly initialized parameters."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from einops import pack, rearrange

STATE_SHAPES = [(1,40,2), (1,64,4), (1,64,8), (1,64,16), (1,64,32), (1,64,64)]


class CausalVCM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(40,64,3)] + [nn.Conv1d(64,64,5,dilation=d) for d in (1,2,4,8,16)])
        self.projection = nn.Conv1d(64,40,1)

    def initial_states(self, batch=1, device='cpu'):
        return [torch.zeros(batch, c, t, device=device) for _, c, t in STATE_SHAPES]

    def forward(self, features, states=None):
        if states is None:
            states = self.initial_states(features.shape[0], features.device)
        x = features
        next_states = []
        for layer, state, (_, _, width) in zip(self.layers, states, STATE_SHAPES):
            joined, _ = pack([state, x], 'b c *')
            next_states.append(joined[..., -width:])
            x = torch.relu(layer(joined))
        return self.projection(x), next_states


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, features, *states):
        logits, next_states = self.model(features, states)
        return (rearrange(logits, 'b c t -> b t c'), *next_states)


def export_model(model, path):
    import onnx
    import onnxruntime as ort
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model.cpu().eval())
    generator = torch.Generator().manual_seed(921)
    inputs = (torch.randn(1,40,4,generator=generator), *[torch.randn(s,generator=generator) for s in STATE_SHAPES])
    names = ['features']+[f'state_{i}' for i in range(6)]
    outputs = ['ctc']+[f'next_state_{i}' for i in range(6)]
    torch.onnx.export(wrapper, inputs, str(path), input_names=names, output_names=outputs,
                      dynamic_axes={'features':{2:'frames'}, 'ctc':{1:'frames'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions(); options.intra_op_num_threads=1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    error = 0.
    for frames in (1,3,4,8,17):
        sample = (torch.randn(1,40,frames,generator=generator), *inputs[1:])
        with torch.no_grad(): expected = wrapper(*sample)
        actual = session.run(None, dict(zip(names,[x.numpy() for x in sample])))
        error=max(error,max(float(np.max(np.abs(a-b.numpy()))) for a,b in zip(actual,expected)))
    if error >= 2e-5:
        raise ValueError(f'ORT nonzero-cache parity failed: {error}')
    return {'max_absolute_error':error,'nonzero_cache_tested':True,'frame_counts':[1,3,4,8,17],
            'outputs':'raw logits; runtime applies log-softmax once','state_shapes':STATE_SHAPES,
            'parameters':sum(p.numel() for p in model.parameters()),'onnx_bytes':path.stat().st_size}
