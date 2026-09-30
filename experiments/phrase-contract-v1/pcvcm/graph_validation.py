"""Exact FP32 graph/cache parity and human-development waveform decisions."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from .contract import canonical, file_hash, read_jsonl, write_json


def check_graph(model, graph):
    import onnx
    import onnxruntime as ort
    import torch
    from vcm.model import ExportWrapper, STATE_SHAPES
    onnx.checker.check_model(str(graph))
    loaded = onnx.load(str(graph), load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in loaded.graph.initializer):
        raise ValueError('External graph weights prohibited')
    if any(t.data_type != onnx.TensorProto.FLOAT for t in loaded.graph.initializer):
        raise ValueError('Only FP32 reference weights are permitted')
    if Path(graph).stat().st_size >= 1024 * 1024:
        raise ValueError('Graph exceeds 1 MiB')
    ort.disable_telemetry_events()
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(graph), options, providers=['CPUExecutionProvider'])
    wrapper = ExportWrapper(model.cpu().eval())
    generator = torch.Generator().manual_seed(921)
    names = ['features'] + [f'state_{i}' for i in range(6)]
    trials = []
    for frames in (1, 3, 4, 8, 17):
        values = (torch.randn(1, 40, frames, generator=generator),
                  *[torch.randn(shape, generator=generator) for shape in STATE_SHAPES])
        with torch.no_grad():
            expected = wrapper(*values)
        actual = session.run(None, dict(zip(names, [v.numpy() for v in values])))
        errors = [float(np.max(np.abs(a - b.numpy()))) for a, b in zip(actual, expected)]
        if len(actual) != 7 or any(not np.isfinite(a).all() for a in actual) or max(errors) >= 2e-5:
            raise ValueError(f'Nonzero-cache parity failed: {frames}, {errors}')
        trials.append({'frames': frames, 'output_errors': errors})
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 113064:
        raise ValueError('Primary architecture changed')
    return {'graph_sha256': file_hash(graph), 'graph_bytes': Path(graph).stat().st_size,
            'parameters': parameters, 'precision': 'FP32', 'trials': trials,
            'max_absolute_error': max(max(t['output_errors']) for t in trials),
            'onnx': onnx.__version__, 'onnxruntime': ort.__version__,
            'evidence_type': 'software_nonzero_cache_parity', 'qualified': False}


def waveform_parity(model, bundle, manifest, audio_root, contract, output):
    import soundfile as sf
    from .bundle import verify_bundle
    from .data import below
    from .endpoint import EndpointConfig
    from .endpoint_selection import TorchEncoder
    from .evidence import audit_partitions, validate_reference
    from .runtime import Runtime, StreamEngine
    from .training import native_cpu_convolution
    content = verify_bundle(bundle, contract)
    rows = read_jsonl(manifest); audit_partitions(rows)
    if not rows or any(r.get('split') != 'dev' or r.get('synthetic') is not False or r.get('human_reviewed') is not True for r in rows):
        raise ValueError('Waveform parity requires reviewed original development audio')
    native = StreamEngine(contract, content['catalog'], TorchEncoder(model), content['normalization'],
        EndpointConfig(**content['endpoint']), content['policy'], content['binding'])
    exported = Runtime.from_bundle(bundle, contract)
    results = []
    with native_cpu_convolution():
        for row in rows:
            validate_reference(row, contract)
            path = below(audio_root, row['audio'])
            if file_hash(path) != row['audio_sha256']:
                raise ValueError('Waveform hash changed')
            pcm, rate = sf.read(path, dtype='float32')
            if rate != 16000 or pcm.ndim != 1 or hashlib.sha256(pcm.tobytes()).hexdigest() != row['pcm_sha256']:
                raise ValueError('Decoded waveform identity changed')
            decisions, candidates, endpoints = [], [], []
            for runtime in (native, exported):
                runtime.reset()
                for start in range(0, len(pcm), 640):
                    runtime.feed(pcm[start:start + 640], start)
                    if runtime.finalized or runtime.endpoint.fault:
                        break
                decisions.append(runtime.eof())
                candidates.append(runtime.candidate.get('command'))
                endpoints.append(runtime.endpoint.end_sample)
            passed = decisions[0] == decisions[1] and canonical(candidates[0]) == canonical(candidates[1]) and endpoints[0] == endpoints[1]
            results.append({'recording_id': row['recording_id'], 'passed': passed,
                            'decisions': decisions, 'candidates': candidates, 'endpoints': endpoints})
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'rows.json', results)
    report = {'passed': all(r['passed'] for r in results), 'recordings': len(results),
        'binding': content['binding'], 'policy_sha256': file_hash(Path(bundle) / 'policy.json'),
        'manifest_sha256': file_hash(manifest), 'rows_sha256': file_hash(output / 'rows.json'),
        'evidence_type': 'human_development_waveform_parity', 'qualified': False}
    write_json(output / 'parity.json', report)
    return report
