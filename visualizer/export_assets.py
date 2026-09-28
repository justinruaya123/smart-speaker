"""Export an explicit tensor-inspection recipe from the supplied own checkpoint.

The original ONNX graph remains the interchange artifact. This derivative is
for transparent layer-by-layer browser inspection, not an arbitrary ONNX parser.
"""
from __future__ import annotations
import hashlib
import json
from math import gcd
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'experiments/vcm_streaming_v3'
V6 = ROOT / 'experiments/vcm_services_v6'
OUT = ROOT / 'visualizer/assets'
sys.path.insert(0, str(V3 / 'runtime'))
from frontend import Frontend


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(V3 / 'models/phone_s53.pt', map_location='cpu', weights_only=True)
    normalization = np.load(V3 / 'runtime/normalization.npz')
    layers, tensors, offset = [], [], 0
    for index in range(7):
        key = f'blocks.{index}.conv' if index < 6 else 'head'
        weight = checkpoint['weights'][key + '.weight'].numpy().astype('<f4')
        bias = checkpoint['weights'][key + '.bias'].numpy().astype('<f4')
        co, ci, kernel = weight.shape
        dilation = 1 if index in (0, 6) else 2 ** (index - 1)
        layers.append({'id': 'stem' if index == 0 else ('ctc' if index == 6 else f'conv{index}'),
            'type': 'Conv1d', 'in_channels': ci, 'out_channels': co,
            'kernel': kernel, 'dilation': dilation, 'stride': 1,
            'activation': 'relu' if index < 6 else 'linear',
            'weight_offset': offset, 'bias_offset': offset + weight.size,
            'parameters': weight.size + bias.size,
            'history_frames': (kernel - 1) * dilation})
        tensors.extend([weight.ravel(), bias.ravel()])
        offset += weight.size + bias.size
    (OUT / 'weights.bin').write_bytes(np.concatenate(tensors).tobytes())
    shutil.copyfile(V3 / 'models/phone_s53_40ms.onnx', OUT / 'model.onnx')
    graph = json.loads((V6 / 'data/services_compiled.json').read_text())
    (OUT / 'catalog.json').write_text(json.dumps(graph, separators=(',', ':')))
    f = Frontend()
    recipe = {'schema_version': 'vcm-inspector-1',
        'name': 'PhoneTCN-64 · retained research checkpoint',
        'source_onnx': 'model.onnx', 'source_onnx_sha256': digest(OUT / 'model.onnx'),
        'weights': 'weights.bin', 'weights_sha256': digest(OUT / 'weights.bin'),
        'catalog': 'catalog.json', 'graph_hash': graph['graph_hash'],
        'catalog_body_sha256': hashlib.sha256(json.dumps(
            graph, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        'sample_rate': 16000, 'window': 400, 'hop': 160, 'fft': 512,
        'mel_bins': 40, 'fmin': 20, 'fmax': 7600, 'log_floor': 1e-8,
        'chunk_frames': 4, 'maximum_seconds': 8,
        'mean': normalization['mean'].ravel().tolist(),
        'std': normalization['std'].ravel().tolist(),
        'mel_filters': f.mel.ravel().tolist(), 'window_coefficients': f.window.tolist(),
        'vocabulary': checkpoint['phones'], 'layers': layers,
        'acceptance_enabled': False, 'action_execution_enabled': False,
        'notes': 'Real float32 layer activations; no general transcript. Browser timings are not Pi timings.'}
    (OUT / 'recipe.json').write_text(json.dumps(recipe, separators=(',', ':')))
    examples = []
    predictions = json.loads((V6 / 'reports/probe_predictions.json').read_text())
    sources = json.loads((V6 / 'reports/example_sources.json').read_text())
    print('example source type', type(sources).__name__)
    for path in sorted((V6 / 'examples').glob('*.wav')):
        audio, rate = sf.read(path, dtype='float32')
        factor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
        destination = OUT / path.name
        # FLOAT WAV avoids adding a second PCM16 rounding after resampling.
        sf.write(destination, audio, 16000, subtype='FLOAT')
        examples.append({'file': path.name, 'name': path.stem.replace('_', ' '),
            'verbatim': sources.get(path.name, path.stem),
            'samples': len(audio), 'source_sha256': digest(path),
            'sha256': digest(destination)})
    (OUT / 'examples.json').write_text(json.dumps(examples, indent=2))
    print('Exported', len(layers), 'layers;', offset, 'parameters;', len(examples), 'WAVs')


if __name__ == '__main__':
    export()
