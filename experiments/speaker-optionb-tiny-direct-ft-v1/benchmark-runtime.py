"""Fresh Torch-free CPU probe for the direct tiny continuation."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime.speaker_optionb_tiny import Runtime
from scripts.test.speaker_demo_suite import read_json, read_jsonl, read_audio, write_new, file_digest


def main():
    study = ROOT / 'experiments/speaker-optionb-tiny-direct-ft-v1'
    run = ROOT / read_json(study / 'protocol.json')['outputs']['run']
    if not (study / 'selection-before-test.json').is_file():
        raise FileNotFoundError('Frozen model and policy required')
    path = ROOT / 'experiments/speaker-optionb-tiny-v1/benchmark-runtime.py'
    spec = importlib.util.spec_from_file_location('frozen_memory_reader', path)
    if spec is None or spec.loader is None:
        raise ImportError('Frozen Windows memory reader unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module.memory_counters()
    runtime = Runtime(run / 'bundle', threads=1)
    measured = []
    for row in read_jsonl(run / 'dev.jsonl'):
        if row['kind'] == 'human' and row['metric_eligible']:
            event = runtime.recognize(read_audio(row))
            assert event['actions_enabled'] is False and event['model_qualified'] is False
            measured.append(event['latency_ms'])
    assert 'torch' not in sys.modules
    result = dict(platform=platform.platform(), python=platform.python_version(), logical_cpu_count=os.cpu_count(),
        threads=1, blas_threads=1, measured_original_human_dev_utterances=len(measured),
        p50_ms=float(np.percentile(measured, 50)), p95_ms=float(np.percentile(measured, 95)),
        session_load_ms=runtime.load_ms, memory_before_model=before, memory_after_replay=module.memory_counters(),
        torch_imported=False, model_sha256=file_digest(run / 'bundle/model.onnx'),
        excludes=['WAV loading', 'speaking time', 'wake/VAD/endpoint waits', 'UI/TTS/device providers'], physical_pi_evidence=False)
    write_new(study / 'desktop-resource-results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
