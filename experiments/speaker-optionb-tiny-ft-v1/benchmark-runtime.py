"""Fresh Torch-free CPU probe, reusing the frozen Windows memory reader."""
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
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new


def main():
    study = ROOT / 'experiments/speaker-optionb-tiny-ft-v1'
    run = ROOT / read_json(study / 'protocol.json')['outputs']['run']
    if not (study / 'selection-before-test.json').is_file():
        raise FileNotFoundError('A frozen selection is required')
    prior = ROOT / 'experiments/speaker-optionb-tiny-v1/benchmark-runtime.py'
    spec = importlib.util.spec_from_file_location('frozen_windows_memory_reader', prior)
    if spec is None or spec.loader is None:
        raise ImportError('Frozen memory-reader module could not be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = module.memory_counters()
    runtime = Runtime(run / 'bundle', threads=1)
    samples = [r for r in read_jsonl(run / 'dev.jsonl') if r['kind'] == 'human' and r['metric_eligible']]
    times = []
    for row in samples:
        event = runtime.recognize(read_audio(row))
        times.append(event['latency_ms'])
        assert event['actions_enabled'] is False and event['model_qualified'] is False
    assert 'torch' not in sys.modules
    receipt = dict(platform=platform.platform(), python=platform.python_version(),
        logical_cpu_count=os.cpu_count(), threads=1, blas_threads=1,
        measured_original_human_dev_utterances=len(times),
        p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
        session_load_ms=runtime.load_ms, memory_before_model=before,
        memory_after_replay=module.memory_counters(), torch_imported=False,
        model_sha256=file_digest(run / 'bundle/model.onnx'), memory_reader_sha256=file_digest(prior),
        excludes=['audio file loading', 'utterance speaking time', 'wake/VAD/endpoint wait', 'UI/TTS/device providers'],
        physical_pi_evidence=False)
    write_new(study / 'desktop-resource-results.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
