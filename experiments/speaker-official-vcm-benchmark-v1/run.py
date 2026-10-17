"""Fresh recorded holdout inference followed by the unmodified official scorer.

This does not simulate the live acoustic path or invent Raspberry Pi telemetry.
Run prepare, infer --model full, infer --model compact, then report.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STUDY = HERE.name
BENCHMARK_COMMIT = 'ab39857cf73dd067d068fed388c321ab160dc191'
BENCH = ROOT / '.cache/vcm-benchmark' / BENCHMARK_COMMIT
CACHE = ROOT / '.cache' / STUDY / '20261016-recorded-01'
DATA = ROOT / '.cache/pcv-source-refresh-v1/source/data/holdout-00000-of-00001.parquet'
DATA_SHA256 = '9917e336ec596eaaddb476e266f36451326e3f8e5a2cb3dafbb675c33be133aa'
ADAPTER = ROOT / 'scripts/runtime/pcv_benchmark_adapter.py'
MODELS = {'full': 'speaker-final-refinement-v1', 'compact': 'speaker-pipeline-repair-v1'}
MODES = ('shipped', 'demo', 'fixed_demo', 'qualified', 'raw_diagnostic')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def setup_imports(package=None):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(BENCH))
    if package is not None:
        sys.path.insert(0, str(package))


def load_adapter():
    spec = importlib.util.spec_from_file_location('official_run_adapter', ADAPTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_clips():
    from vcmbench.dataset import load_holdout
    return load_holdout(str(DATA), CACHE / 'unused-download-cache')


def parsed_trial(clip, event, adapter):
    from vcmbench import schema as schema
    from vcmbench.pi import LineParser
    record = adapter.log_record(event)
    parsed = LineParser().parse(json.dumps(record), 0.0)
    assert parsed is not None and parsed.kind == 'command'
    intent, slot, known, variation = schema.resolve_prediction(
        parsed.intent, parsed.slot, parsed.variation, schema.build_alias_table(),
        schema.id_orders()['manifest'])
    return dict(order=clip.idx, clip_idx=clip.idx, kind='wake',
                input_kind='recorded_command_only_no_wake', wake_present=False,
                transcript=clip.transcript, true_intent=clip.intent,
                true_variation=clip.variation, true_slot=clip.slot_value,
                speaker_id=clip.speaker_id, is_synthetic=clip.is_synthetic,
                accent_group=clip.accent_group, pred_intent=intent, pred_slot=slot,
                pred_variation=variation, pred_variation_id=None,
                pred_raw=json.dumps(record), prediction_known=known,
                infer_ms=parsed.infer_ms, audio_ms=parsed.audio_ms,
                latency_s=None, play_t0=None, n_command_events=1, wake_logged=False)


def verify_protocol():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for name, digest in protocol['bindings'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    return protocol


def prepare():
    CACHE.mkdir(parents=True, exist_ok=False)
    setup_imports()
    from vcmbench import audio, report
    from scripts.runtime.pcv_schema import baseline_cases, output
    from types import SimpleNamespace
    adapter = load_adapter()
    assert sha(DATA) == DATA_SHA256
    clips = read_clips()
    assert len(clips) == 196
    assert sum(not c.is_synthetic for c in clips) == 96
    assert sum(c.intent == 'OUT_OF_SCOPE' for c in clips) == 10
    assert len({c.variation for c in clips if c.variation}) == 93
    variations = {v['phrase'].casefold(): v for v in csv.DictReader(
        (BENCH / 'vcmbench/variations.csv').open())}
    fixtures = []
    for case in baseline_cases():
        v = variations[case['text'].casefold()]
        c = SimpleNamespace(idx=len(fixtures), transcript=case['text'], intent=v['label'],
                            variation=v['phrase'], slot_value=v['value'], speaker_id='fixture',
                            is_synthetic=False, accent_group='fixture')
        event = {'output': output(case['command'], None), 'timing': {'total_ms': 1.0},
                 'audio_seconds': 1.0}
        fixtures.append(parsed_trial(c, event, adapter))
    fixture_score = report.score(fixtures, [], {}, None, 0, 1, {'mode': 'serializer_fixtures'})
    assert len(fixtures) == 93 and fixture_score['command_level']['accuracy'] == 1.0
    rejection = {'output': output(None, 'unqualified_model'),
                 'diagnostic_candidate': {'candidate': baseline_cases()[0]['command']},
                 'timing': {'total_ms': 1.0}, 'audio_seconds': 1.0}
    assert adapter.log_record(rejection)['intent'] == 'OUT_OF_SCOPE'
    reject_trials = [parsed_trial(c, rejection, adapter) for c in clips]
    reject_score = report.score(reject_trials, [], {}, None, 0, 1, {'mode': 'reject_control'})
    assert reject_score['command_level']['accuracy'] == 10 / 196
    assert reject_score['breakdowns']['synthetic voice']['command_accuracy'] == 0
    assert reject_score['command_level']['false_accepts'] == 0
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in BENCH.rglob('*')
                if p.is_file() and (p.suffix in {'.py', '.csv'} or p.name == 'README.md')}
    for p in [Path(__file__), ADAPTER, DATA,
              ROOT / 'data/sources/ai231-primary.source.json']:
        bindings[str(p.relative_to(ROOT))] = sha(p)
    models = {}
    for name, study in MODELS.items():
        receipt_path = ROOT / 'experiments' / study / 'package.json'
        results_path = ROOT / 'experiments' / study / 'results.json'
        receipt = json.loads(receipt_path.read_text())
        archive = Path(receipt['path'])
        assert sha(archive) == receipt['sha256']
        results = json.loads(results_path.read_text())
        for p in [receipt_path, results_path, archive]:
            bindings[str(p.relative_to(ROOT))] = sha(p)
        models[name] = {'study': study, 'archive': str(archive.relative_to(ROOT)),
                        'model_binding': results['model_binding'],
                        'parameters': results['parameters'], 'graph_bytes': results['graph_bytes'],
                        'demo_policy': 'fixed_.95_intent_.90_slots' if name == 'full'
                                       else 'frozen_per_intent_recall_policy_.90_slots'}
    installed = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'artifacts').rglob('*') if p.is_file()}
    bindings.update(installed)
    preflight = {'serializer_fixtures_passed': 93, 'rejected_diagnostic_not_promoted': True,
                 'all_reject_control_accuracy': 10 / 196, 'original_holdout_rows': 196,
                 'human_positive': 86, 'human_out_of_scope': 10, 'synthetic_positive': 100}
    write_json(HERE / 'protocol.json', {
        'experiment_id': STUDY, 'date': '2026-10-16',
        'authorization': 'User requested running the official vcm-benchmark with both models; no Pi available.',
        'requirements': ['ME-03', 'ME-04', 'ME-06', 'ME-07', 'ME-08'],
        'mode': 'recorded command-only inference and official offline rescoring',
        'benchmark_repository': 'https://github.com/airimonda/vcm-benchmark',
        'benchmark_commit': BENCHMARK_COMMIT,
        'benchmark_source': 'Unmodified preserved source; remote verification unavailable (web cache miss).',
        'dataset_repository': 'airimonda/ai231-me2-voice-commands',
        'dataset_revision': '25111444af3adff7588d86ab27c304bb080895cf',
        'dataset_sha256': DATA_SHA256,
        'preprocessing': 'Official audio.normalize(audio.trim(clip.audio)) with unchanged defaults.',
        'modes': list(MODES), 'models': models, 'seed': 53, 'threads': 1,
        'cpu_affinity': 'First available logical CPU; not exclusively reserved.',
        'selection': 'No training, selection, calibration, threshold changes, or result-driven rerun.',
        'official_scoring': 'Actual LineParser and resolve_prediction; benchmark.py --rescore per stream.',
        'official_trial_bucket': 'kind=wake is the upstream scored-command bucket; no wake audio is supplied.',
        'exclusions': 'None: all 196 publisher references, including typed-reference ambiguity, are scored.',
        'physical_pi_measured': False, 'live_audio_measured': False, 'wake_vad_measured': False,
        'published_data_previously_examined': True, 'actions_enabled': False,
        'bindings': bindings, 'installed_bindings': installed, 'preflight': preflight,
        'output': str(CACHE.relative_to(ROOT))})
    write_json(CACHE / 'preflight.json', preflight)
    print(json.dumps(preflight), flush=True)


def infer(name):
    protocol = verify_protocol()
    folder = CACHE / name
    folder.mkdir(exist_ok=False)
    package = folder / 'package'
    package.mkdir()
    with zipfile.ZipFile(ROOT / protocol['models'][name]['archive']) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        for path, digest in manifest['files'].items():
            assert (package / path).resolve().is_relative_to(package.resolve())
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
        archive.extractall(package)
    setup_imports(package)
    from vcmbench import audio
    from scripts.runtime.pcv_final_runtime import Runtime
    from scripts.runtime.pcv_schema import output
    from scripts.runtime.speaker_confidence import application_decision
    adapter = load_adapter()
    if name == 'compact':
        from scripts.runtime.speaker_compact import CompactRuntime
        runtime = CompactRuntime(package / 'models/default', threads=1, mode='demo')
    else:
        runtime = Runtime(package / 'models/default', threads=1, diagnostic=True)
    assert runtime.recipe['binding'] == protocol['models'][name]['model_binding']
    assert runtime.policy['enabled'] is False
    assert Path(sys.modules['scripts.runtime.pcv_final_runtime'].__file__).is_relative_to(package)
    clips = read_clips()
    random.Random(53).shuffle(clips)
    streams = {}
    for mode in MODES:
        path = folder / mode
        path.mkdir()
        streams[mode] = (path / 'trials.jsonl').open('x')
    began = time.perf_counter()
    events_path = folder / 'events.jsonl'
    with events_path.open('x') as events:
        for i, clip in enumerate(clips, 1):
            pcm = audio.normalize(audio.trim(clip.audio))
            event = runtime.recognize(pcm)
            assert event['model_sha256'] == protocol['models'][name]['model_binding']['model_sha256']
            fixed = application_decision(event, {'mode': 'demo', 'intent_threshold': .95, 'slot_threshold': .9})
            qualified = event.get('qualified_output', event['output'])
            commands = {'shipped': event['output'].get('command'),
                        'qualified': qualified.get('command'),
                        'fixed_demo': fixed['command'],
                        'demo': event['application']['command'] if name == 'compact' else fixed['command'],
                        'raw_diagnostic': (event.get('diagnostic_candidate') or {}).get('candidate')}
            events.write(json.dumps({'clip_idx': clip.idx, 'file': clip.file,
                         'input_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
                         'input_samples': len(pcm), 'capture_fault': event['output'].get('reason') == 'capture_fault',
                         'event': event}, allow_nan=False) + '\n')
            for mode, command in commands.items():
                selected = {**event, 'output': output(command, None if command else 'OOV')}
                trial = parsed_trial(clip, selected, adapter)
                streams[mode].write(json.dumps(trial, allow_nan=False) + '\n')
            if i % 25 == 0 or i == len(clips):
                print(f'{name}: {i}/{len(clips)} fresh recordings inferred', flush=True)
    for stream in streams.values():
        stream.close()
    elapsed = time.perf_counter() - began
    host = {'model': 'x86_64 compute host; NOT a Raspberry Pi',
            'machine': platform.machine(), 'os': platform.platform(), 'python': platform.python_version(),
            'physical_pi_measured': False, 'cpu_affinity': sorted(os.sched_getaffinity(0))}
    for mode in MODES:
        write_json(folder / mode / 'config.json', {
            'student': f'{STUDY}/{name}/{mode}', 'started': '2026-10-16',
            'mode': 'OFFLINE_RECORDED_COMMAND_ONLY_NO_PI_NO_WAKE/' + mode,
            'wake_word': None, 'wake_gap': None, 'size': 'full', 'seed': 53,
            'id_order': 'manifest', 'holdout': str(DATA), 't_start': 0.0, 't_end': elapsed})
        write_json(folder / mode / 'pi_specs.json', host)
    forbidden = [m for m in sys.modules if m == 'torch' or m.startswith(('torch.', 'scripts.train.'))]
    assert not forbidden
    rss = next(int(line.split()[1]) / 1024 for line in Path('/proc/self/status').read_text().splitlines()
               if line.startswith('VmHWM:'))
    write_json(folder / 'inference.json', {
        'model': name, 'rows': len(clips), 'seconds_including_serialization': elapsed,
        'parameters': protocol['models'][name]['parameters'], 'runtime_load_ms': runtime.load_ms,
        'process_peak_rss_mib': rss, 'hardware': host,
        'versions': {n: importlib.metadata.version(n) for n in ['numpy', 'onnxruntime', 'soundfile']},
        'model_binding': runtime.recipe['binding'], 'events_sha256': sha(events_path),
        'forbidden_training_imports': forbidden, 'wake_vad_sessions_loaded': False})
    print(f'{name}: inference complete; {rss:.2f} MiB host peak RSS', flush=True)


def report_results():
    verify_protocol()
    setup_imports()
    from vcmbench.report import score
    from vcmbench.metrics import wilson
    results = {'experiment_id': STUDY, 'benchmark_commit': BENCHMARK_COMMIT,
               'mode': 'recorded holdout / official scorer; NOT live or physical Pi', 'models': {}}
    env = {**os.environ, 'MPLCONFIGDIR': str(CACHE / 'matplotlib')}
    for name in MODELS:
        model_results = {'inference': json.loads((CACHE / name / 'inference.json').read_text()), 'policies': {}}
        for mode in MODES:
            folder = CACHE / name / mode
            with (folder / 'official-cli.log').open('x') as log:
                subprocess.run([sys.executable, str(BENCH / 'benchmark.py'), '--rescore', str(folder)],
                               cwd=CACHE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            official = json.loads((folder / 'metrics.json').read_text())
            trials = [json.loads(line) for line in (folder / 'trials.jsonl').read_text().splitlines()]
            check = score(trials, [], {}, None, 0, 1, official['meta'])
            assert check['command_level']['accuracy'] == official['command_level']['accuracy']
            summary = {'official_command': official['command_level'], 'official_intent': official['intent_level'],
                       'official_breakdowns': official['breakdowns'], 'strata': {}, 'per_intent': {}}
            for kind in ['overall', 'human', 'synthetic']:
                sub = [t for t in trials if kind == 'overall' or bool(t['is_synthetic']) == (kind == 'synthetic')]
                positive = [t for t in sub if t['true_intent'] != 'OUT_OF_SCOPE']
                negative = [t for t in sub if t['true_intent'] == 'OUT_OF_SCOPE']
                accepted = [t for t in sub if t['pred_intent'] not in ('OUT_OF_SCOPE', 'NO_RESPONSE')]
                correct = sum(t['correct_command'] for t in positive)
                false_accepts = sum(t['pred_intent'] not in ('OUT_OF_SCOPE', 'NO_RESPONSE') for t in negative)
                summary['strata'][kind] = {'n': len(sub), 'positive_n': len(positive), 'negative_n': len(negative),
                    'correct_accepts': correct, 'accepted': len(accepted),
                    'precision': correct / len(accepted) if accepted else None,
                    'recall': correct / len(positive) if positive else None,
                    'recall_ci95': wilson(correct, len(positive)), 'false_accepts': false_accepts,
                    'false_accept_rate': false_accepts / len(negative) if negative else None}
            for intent in sorted({t['true_intent'] for t in trials}):
                summary['per_intent'][intent] = {}
                for kind in ['human', 'synthetic']:
                    sub = [t for t in trials if t['true_intent'] == intent and bool(t['is_synthetic']) == (kind == 'synthetic')]
                    summary['per_intent'][intent][kind] = {'n': len(sub), 'correct': sum(t['correct_command'] for t in sub)}
            assert len(trials) == 196
            assert summary['strata']['human']['positive_n'] == 86
            assert summary['strata']['synthetic']['positive_n'] == 100
            if mode == 'qualified':
                assert summary['strata']['overall']['accepted'] == 0
            model_results['policies'][mode] = summary
            print(name, mode, json.dumps(summary['strata']), flush=True)
        results['models'][name] = model_results
    verify_protocol()
    results['source_and_installed_bindings_unchanged'] = True
    results['bindings'] = {str(p.relative_to(ROOT)): sha(p) for p in CACHE.rglob('*')
                           if p.is_file() and p.name in ('trials.jsonl', 'metrics.json', 'report.md', 'inference.json', 'events.jsonl')}
    write_json(HERE / 'results.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'infer', 'report'])
    parser.add_argument('--model', choices=list(MODELS))
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'infer':
        if not args.model:
            parser.error('infer requires --model')
        infer(args.model)
    else:
        report_results()


if __name__ == '__main__':
    main()
