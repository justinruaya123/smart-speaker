"""Matched desktop timing on a fixed development workload; no quality selection."""
from pathlib import Path
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources
register_sources()
from tinyvcm.complete_data import DATA, EXP, CACHE, read_rows, save, digest
from tinyvcm.inference import Recognizer
from tinyvcm.audio import load_audio


def main():
    protocol = json.loads((ROOT / 'experiments/wake-pipeline-v1/quantization-protocol.json').read_text())
    wanted = set(protocol['ids'])
    by_id = {r['id']: r for r in read_rows(DATA / 'manifest.jsonl') if r['id'] in wanted}
    assert len(by_id) == 150 and all(r['split'] == 'dev' for r in by_id.values())
    folders = {'installed_baseline': ROOT / 'artifacts', 'human_keywords': CACHE / 'human_keywords', 'v3_pooled': CACHE / 'v3_pooled'}
    models, loads = {}, {}
    for name, folder in folders.items():
        started = time.perf_counter(); models[name] = Recognizer(folder, threads=2)
        loads[name] = (time.perf_counter() - started) * 1000
    timings = {name: [] for name in models}
    for i, id in enumerate(protocol['ids']):
        pcm, _ = load_audio(ROOT / by_id[id]['audio'])
        names = list(models) if i % 2 == 0 else list(reversed(models))
        for name in names:
            models[name].predict(pcm, stream=False)
            sample = []
            for _ in range(3):
                started = time.perf_counter(); models[name].predict(pcm, stream=False)
                sample.append((time.perf_counter() - started) * 1000)
            timings[name].append(float(np.median(sample)))
    result = {'scope': 'Desktop CPU ONNX, identical 150 development PCM recordings, 2 threads, one warmup and 3 repeated calls per model/recording, alternating model order; no training or TTS generation running.',
        'excludes': 'disk, utterance duration, wake inference and 400 ms endpoint wait', 'Pi_measured': False,
        'ids_source_sha256': digest(ROOT / 'experiments/wake-pipeline-v1/quantization-protocol.json'),
        'models': {name: {'graph_sha256': digest(folders[name] / 'vcm.fp32.onnx'),
            'graph_bytes': (folders[name] / 'vcm.fp32.onnx').stat().st_size, 'session_load_ms': loads[name],
            'p50_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95))} for name, values in timings.items()}}
    save(EXP / 'matched-desktop-timing.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
