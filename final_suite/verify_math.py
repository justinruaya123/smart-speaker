"""Compare actual browser-WASM tensors to PyTorch and serialized ONNX/OpenCV."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.special import log_softmax
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'experiments/vcm_streaming_v3'
sys.path.insert(0, str(V3 / 'src'))
sys.path.insert(0, str(V3 / 'runtime'))
from phone_model import PhoneTCN
from frontend import Frontend
from run_stream import Engine
from stream_policy import CommandRanker
from trie_ctc import ScaledTrieCTC


def verify() -> dict:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    checkpoint = torch.load(V3 / 'models/phone_s53.pt', map_location='cpu', weights_only=True)
    model = PhoneTCN().eval()
    model.load_state_dict(checkpoint['weights'])
    norm = np.load(V3 / 'runtime/normalization.npz')
    graph = json.loads((ROOT / 'visualizer/assets/catalog.json').read_text())
    results = json.loads((ROOT / 'final_suite/reports/browser_math.json').read_text())
    engine = Engine(V3 / 'models/phone_s53_40ms.onnx')
    reports = []
    for result in results:
        pcm, rate = sf.read(ROOT / 'visualizer/assets' / result['file'], dtype='float32')
        assert rate == 16000
        pcm = np.r_[pcm, np.zeros(1600, np.float32)]
        mel = Frontend().push(pcm)
        folder = ROOT / 'final_suite/reports/browser_parity'
        actual_mel = np.fromfile(folder / (result['file'] + '.mel.f32'), np.float32).reshape(mel.shape)
        np.testing.assert_allclose(mel, actual_mel, atol=1e-4, rtol=1e-4)
        features = (mel - norm['mean'][:, None]) / norm['std'][:, None]
        with torch.inference_mode():
            expected = model(torch.from_numpy(features[None])).numpy()[0]
        raw = np.fromfile(folder / (result['file'] + '.ctc.f32'), np.float32)
        actual = np.concatenate(raw.reshape(-1, 40, 4), axis=-1)[:, :expected.shape[-1]]
        np.testing.assert_allclose(expected, actual, atol=1e-3, rtol=1e-4)
        scorer = ScaledTrieCTC([f['target'] for f in graph['forms']], 40, 'native')
        scores = scorer.push(log_softmax(expected.T.astype(np.float64), axis=-1))
        ranked = CommandRanker(graph).rank(scores)
        assert ranked['candidate'] == result['final']['candidate']
        assert ranked['ties'] == result['final']['ties']
        engine.states = [np.zeros_like(s) for s in engine.states]
        outputs = []
        for start in range(0, features.shape[-1], 4):
            chunk = features[:, start:start + 4]
            valid = chunk.shape[1]
            outputs.append(engine.step(np.pad(chunk, ((0, 0), (0, 4 - valid)))[None])[:valid])
        onnx = np.concatenate(outputs).T
        np.testing.assert_allclose(onnx, actual, atol=1e-3, rtol=1e-4)
        reports.append({
            'file': result['file'], 'frames': expected.shape[-1],
            'frontend_max_abs_error': float(np.max(np.abs(mel - actual_mel))),
            'torch_wasm_max_logit_error': float(np.max(np.abs(expected - actual))),
            'onnx_wasm_max_logit_error': float(np.max(np.abs(onnx - actual))),
            'changed_argmax_frames': int(np.sum(expected.argmax(0) != actual.argmax(0))),
            'candidate_equal': True, 'ties_equal': True,
        })
    report = {'passed': True, 'samples': len(reports), 'reports': reports,
              'wasm_executed': True, 'onnx_backend': 'OpenCV CPU',
              'onnx_runtime_web_tested': False, 'webgpu_compute_tested': False,
              'physical_pi_tested': False}
    (ROOT / 'final_suite/reports/math_verification.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
