"""Published-artifact epoch44 benchmark; no fitting, microphone or playback.

Requires explicit original mono16kHz hash-bound audio. Reuses the existing typed
scorer, input checks, hardware, memory and timing implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

ROOT = Path(__file__).resolve().parents[1]


def benchmark(bundle, manifest, output, *, threads=1, engine='onnxruntime', repeats=3, root=None):
    from scripts.runtime.iris_epoch44 import Runtime
    from scripts.runtime.iris_pi_benchmark import benchmark as replay
    if engine != 'onnxruntime':
        raise ValueError('The published epoch44 artifact supports only onnxruntime CPU')
    return replay(bundle, manifest, output, threads=threads, engine=engine, repeats=repeats,
                  root=root or ROOT, runtime_factory=Runtime)


def main(argv=None, *, root=ROOT):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-choice', choices=['epoch44'], default='epoch44', help='Frozen published artifact (default)')
    parser.add_argument('--bundle', type=Path, default=root/'submission/models/optionb-tiny')
    parser.add_argument('--manifest', type=Path, required=True, help='Hash-bound original-audio JSONL with explicit typed references')
    parser.add_argument('--data-root', type=Path, default=root, help='Root containing the manifest audio paths; no download or fitting')
    parser.add_argument('--output', type=Path, help='New immutable result directory; existing paths are rejected')
    parser.add_argument('--threads', type=int, choices=range(1,9), default=1)
    parser.add_argument('--engine', choices=['onnxruntime'], default='onnxruntime')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-setup', action='store_true', help='Use this interpreter without installing dependencies')
    args = parser.parse_args(argv)
    if not args.manifest.is_file():
        parser.error('Original-audio manifest file not found: ' + str(args.manifest))
    output = args.output or root/'.cache/iris-publication-v1'/datetime.now(timezone.utc).strftime('pc-%Y%m%dT%H%M%S.%fZ')
    if not args.no_setup:
        from scripts.runtime.speaker_tools import sync_python
        python = sync_python(root, source=True)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(ACTIONS_ENABLED='false', IRIS_FEEDBACK_ENABLED='false')
        return subprocess.call([str(python), '-B', str(root/'benchmark.py'), '--no-setup',
            '--bundle', str(args.bundle.resolve()), '--manifest', str(args.manifest.resolve()),
            '--data-root', str(args.data_root.resolve()), '--output', str(output.resolve()),
            '--threads', str(args.threads), '--engine', args.engine, '--repeats', str(args.repeats)],
            cwd=root, env=env)
    try:
        result = benchmark(args.bundle, args.manifest, output, threads=args.threads,
                           engine=args.engine, repeats=args.repeats, root=args.data_root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(output), artifact_provenance=result['artifact_provenance'],
        model_sha256=result['model_sha256'], parameters=result['parameters'], selected_epoch=result['selected_epoch'],
        measurement_host=result['hardware']['measurement_host'], physical_pi_measured=result['physical_pi_measured'],
        actions_enabled=False, source_n=result['source_n'], observation_n=result['observation_n'],
        excluded_reference_n=result['excluded_reference_n'], raw=result['raw'], accepted=result['accepted'],
        latency_ms=result['latency_ms']['end_to_end']), allow_nan=False))
    return result


if __name__ == '__main__':
    outcome = main()
    raise SystemExit(outcome if isinstance(outcome, int) else 0)
