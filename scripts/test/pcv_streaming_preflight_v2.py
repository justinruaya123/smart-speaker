"""Numerical/cost preflight; random tensors are NOT speech or Pi evaluation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from scripts.models.pcv_streaming_v2 import StreamingExport, StreamingVCM


def summary(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values)
    return {'n': len(values), 'p50_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)),
            'p99_ms': float(np.percentile(a, 99)), 'max_ms': float(a.max())}


def run() -> dict:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(231)
    old = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    if old:
        os.sched_setaffinity(0, {min(old)})
    try:
        model = StreamingVCM().eval()
        data = {
            'status': 'untrained_tensor_preflight_not_recognition',
            'architecture': 'Causal-Slot-TCN-96-v2',
            'python': platform.python_version(), 'torch': torch.__version__,
            'machine': platform.machine(),
            'affinity': sorted(os.sched_getaffinity(0)) if old else None,
            'threads': torch.get_num_threads(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'speaker_parameters': sum(
                p.numel() for p in model.speaker.parameters()),
            'weight_bytes_fp32': sum(p.numel() * p.element_size()
                                     for p in model.parameters()),
            'state_bytes_fp32': sum(s.numel() * s.element_size()
                                    for s in model.initial_state()),
            'state_shapes': [list(s.shape) for s in model.initial_state()],
            'physical_pi_measured': False, 'waveform_accuracy_measured': False,
            'onnx_available': importlib.util.find_spec('onnx') is not None,
            'onnxruntime_available': bool(
                importlib.util.find_spec('onnxruntime')),
            'latency_scope': 'model only; excludes features, decoder, '
                             'capture, endpointer, IPC, UI, file loading',
            'timings': {},
        }
        with torch.inference_mode():
            for profile in ('benchmark', 'demo'):
                wrapper = StreamingExport(model, profile).eval()
                probe = torch.randn(1, 40, 4)
                traced = torch.jit.trace(
                    wrapper, (probe, *model.initial_state()))
                traced = torch.jit.freeze(traced.eval())
                for engine, runner in (('pytorch_eager', wrapper),
                                        ('torchscript_frozen', traced)):
                    state = model.initial_state()
                    for _ in range(30):
                        state = runner(probe, *state)[-14:]
                    times = []
                    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    for i in range(1000):
                        if i % 100 == 0:
                            state = model.initial_state()
                        start = time.perf_counter_ns()
                        result = runner(probe, *state)
                        times.append((time.perf_counter_ns() - start) / 1e6)
                        state = result[-14:]
                    key = profile + '/' + engine
                    data['timings'][key] = {
                        **summary(times), 'chunk_audio_ms': 40,
                        'p95_compute_to_chunk_ratio':
                            float(np.percentile(times, 95)) / 40,
                        'peak_rss_before_kib_linux': before,
                        'peak_rss_after_kib_linux': resource.getrusage(
                            resource.RUSAGE_SELF).ru_maxrss,
                    }
        data['peak_rss_scope'] = ('training-capable PyTorch process, '
                                  'NOT Pi runtime RSS')
        return data
    finally:
        if old:
            os.sched_setaffinity(0, old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing receipt')
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
