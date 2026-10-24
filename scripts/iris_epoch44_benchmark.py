"""Frozen epoch44 replay; no fitting, microphone, playback or actions.

Default: download only the pinned official 196-recording exposed holdout on demand.
This is not fresh blind/full-corpus evaluation; timing measures the actual host,
not inferred Pi performance. Explicit --manifest reuses local original audio.
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


def benchmark(bundle, manifest, output, *, threads=1, engine='onnxruntime', repeats=3, root=None, default_holdout=False):
    from scripts.runtime.iris_epoch44 import Runtime
    from scripts.runtime.iris_pi_benchmark import benchmark as replay
    if engine != 'onnxruntime':
        raise ValueError('The published epoch44 artifact supports only onnxruntime CPU')
    result = replay(bundle, manifest, output, threads=threads, engine=engine, repeats=repeats,
                    root=root or ROOT, runtime_factory=Runtime)
    scope: dict[str, object] = dict(population='explicit_manifest', measurement_scope='actual_host_only',
                 full_corpus=False, fresh_blind_claim=False,
                 description='Explicit local manifest replay; no fresh-blind/full-corpus qualification or inferred Pi performance claim.')
    if default_holdout:
        from scripts.runtime import iris_benchmark_assets as assets
        scope.update(population='existing_exposed_fixed_holdout', fresh_blind=False,
            description='Existing exposed fixed official holdout, not fresh blind or full-corpus evaluation. '
                        'Frozen ambiguous references remain in replay but are excluded from accuracy; hardware/timing are actual-host only, not inferred Pi results.',
            gold_reference_sha256=assets.GOLD_SHA256,
            source=dict(repository=assets.REPOSITORY, revision=assets.REVISION, file=assets.PARQUET_FILE,
                        url=assets.URL, bytes=assets.PARQUET_BYTES, sha256=assets.PARQUET_SHA256, recording_n=assets.RECORDINGS))
    result['benchmark_scope'] = scope
    folder = Path(output)
    sidecar = dict(benchmark_scope=scope, manifest=result['manifest'], manifest_sha256=result['manifest_sha256'],
                   source_n=result['source_n'], excluded_reference_n=result['excluded_reference_n'])
    (folder/'benchmark-inputs.json').write_text(json.dumps(sidecar, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (folder/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main(argv=None, *, root=ROOT):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-choice', choices=['epoch44'], default='epoch44', help='Frozen published artifact (default)')
    parser.add_argument('--bundle', type=Path, default=root/'submission/models/optionb-tiny')
    parser.add_argument('--manifest', type=Path, help='Explicit hash-bound local-audio JSONL (no downloads); default: pinned official 196-recording holdout')
    parser.add_argument('--data-root', type=Path, default=root, help='Explicit-manifest audio root; default holdout inputs are cache-local')
    parser.add_argument('--output', type=Path, help='New immutable result directory; existing paths are rejected')
    parser.add_argument('--threads', type=int, choices=range(1,9), default=1)
    parser.add_argument('--engine', choices=['onnxruntime'], default='onnxruntime')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--no-setup', action='store_true', help='Use this interpreter without installs or missing-asset downloads; cached default inputs work offline')
    parser.add_argument('--_managed', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.manifest is not None and not args.manifest.is_file():
        parser.error('Original-audio manifest file not found: ' + str(args.manifest))
    output = args.output or root/'.cache/iris-publication-v1'/datetime.now(timezone.utc).strftime('pc-%Y%m%dT%H%M%S.%fZ')
    try:
        output, bundle = output.resolve(), args.bundle.resolve()
        if output.exists():
            raise FileExistsError('Benchmark output already exists; choose a new --output directory: ' + str(output))
        if output == bundle or output.is_relative_to(bundle):
            raise ValueError('Benchmark output must be outside the immutable model bundle')
        if any(parent.exists() and not parent.is_dir() for parent in output.parents):
            raise ValueError('Benchmark output parent is not a directory')
        if args.repeats < 1:
            raise ValueError('Use a positive integer repeat count')
        if args.manifest is not None:
            from scripts.runtime.pcv_schema import validate_command
            count = 0
            with args.manifest.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or 'id' not in row or not {'expected', 'command'} & row.keys() or not {'sha256', 'audio_sha256'} & row.keys():
                        raise ValueError('Manifest requires id, recording hash and explicit typed expected reference')
                    expected = row.get('expected', row.get('command'))
                    if expected is not None:
                        validate_command(expected)
                    count += 1
            if not count:
                raise ValueError('Manifest has no recording rows')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not args.no_setup and not args._managed:
        from scripts.runtime.epoch44_setup import sync_epoch44
        python = sync_epoch44(root, extras=('benchmark',)) if args.manifest is None else sync_epoch44(root)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(ACTIONS_ENABLED='false', IRIS_FEEDBACK_ENABLED='false')
        mode = ['--_managed'] if args.manifest is None else ['--no-setup', '--manifest', str(args.manifest.resolve())]
        return subprocess.call([str(python), '-B', str(root/'benchmark.py'), *mode,
            '--bundle', str(args.bundle.resolve()), '--data-root', str(args.data_root.resolve()), '--output', str(output.resolve()),
            '--threads', str(args.threads), '--engine', args.engine, '--repeats', str(args.repeats)],
            cwd=root, env=env)
    try:
        manifest, data_root = args.manifest, args.data_root
        if manifest is None:
            from scripts.runtime.iris_benchmark_assets import default_manifest
            manifest = default_manifest(root, allow_download=not args.no_setup)
            data_root = manifest.parent
        result = benchmark(args.bundle, manifest, output, threads=args.threads,
                           engine=args.engine, repeats=args.repeats, root=data_root, default_holdout=args.manifest is None)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(output), artifact_provenance=result['artifact_provenance'],
        model_sha256=result['model_sha256'], parameters=result['parameters'], selected_epoch=result['selected_epoch'],
        measurement_host=result['hardware']['measurement_host'], physical_pi_measured=result['physical_pi_measured'],
        actions_enabled=False, benchmark_scope=result.get('benchmark_scope'), source_n=result['source_n'], observation_n=result['observation_n'],
        excluded_reference_n=result['excluded_reference_n'], raw=result['raw'], accepted=result['accepted'],
        latency_ms=result['latency_ms']['end_to_end']), allow_nan=False))
    return result


if __name__ == '__main__':
    outcome = main()
    raise SystemExit(outcome if isinstance(outcome, int) else 0)
