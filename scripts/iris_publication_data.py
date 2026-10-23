"""Build deterministic public Iris data from frozen inputs; stdlib only.

No capture, fitting, acoustic inference or provider calls. Historical receipts
stay separate from optional current desktop and Pi receipts. Fail closed on
wrong-model evidence; never guess missing performance/hardware data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.runtime.pcv_schema import baseline_cases

MODEL_HASHES = {
    'model.onnx': 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce',
    'commands.json': '71017cf22366d15557e8878636135f5ff2d0ac81fc4dfd6f058fc88f0f2d4b5d',
    'normalization.json': '28f7ef1b2c926c587d611d17eef91184c2c7e2de66abe352d1fc2365e45d69ed',
    'policy.json': 'b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0',
    'recipe.json': '020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621',
}
SNAPSHOT_HASH = 'd0ac31ea3d5f771cbe983e2ae9bd06303267c4b68723870c28b9904b17d74a21'
CATEGORIES = [
    ('UC-01', 'Music', ['PLAY_MUSIC'], ['Play music'], ['Typed PLAY_MUSIC', 'Configured player', 'Playback receipt'],
     'Synthetic playback', 'Playing · demonstration only', 'No audio is playing. A configured provider must supply its actual result.'),
    ('UC-02', 'Questions', ['WEATHER', 'TIME'], ['Weather', 'Time'], ['Typed WEATHER / TIME', 'OpenWeather / clock', 'Answer + source'],
     'Synthetic weather', 'Weather unavailable', 'No current observation. OpenWeather → honestly dated cache → labeled synthetic fallback. Time uses the local Manila clock.'),
    ('UC-03', 'Lights', ['LIGHT_ON', 'LIGHT_OFF'], ['Turn on the lights'], ['Typed LIGHT_ON', 'Hue lamp binding', 'Receipt / observation'],
     'Synthetic light', 'Lamp on · simulated', 'Target light.hue_color_lamp_1. No device changed; a provider acknowledgment is not an observed state.'),
    ('UC-04', 'Dim / color', ['BRIGHTNESS', 'COLOR'], ['Brightness 20 percent', 'Change color to Red'], ['Typed percent / color', 'Hue lamp binding', 'Value + source'],
     'Synthetic light', '20% · red · simulated', 'Finite brightness 20 / 60 / 100%; colors red / blue / green. No device changed.'),
    ('UC-05', 'Timer', ['TIMER'], ['Timer 10 seconds'], ['Typed TIMER', 'Persistent local job', 'Countdown → due alert'],
     'Local job example', '0:10', 'Example of a saved ten-second countdown. This static deck starts no job; the application persists jobs and due alerts.'),
    ('UC-06', 'Alarm', ['ALARM'], ['Alarm 6:00 AM'], ['Typed ALARM', 'Next Manila deadline', 'Saved alarm → alert'],
     'Local job example', '06:00 · Asia/Manila', 'Next matching Manila deadline, not a fixed UTC offset guessed by the deck. No alarm is scheduled here.'),
    ('UC-07', 'Temperature', ['TEMPERATURE'], ['Temperature 22 degrees'], ['Typed TEMPERATURE', 'Synthetic target only', 'Separate sensor view'],
     'Synthetic target', '22°C target', 'Observed home sensor: unavailable. No climate service is called. Target and real measured temperature stay separate.'),
    ('UC-08', 'Media', ['PAUSE', 'STOP', 'NEXT', 'VOLUME'], ['Pause', 'Next song', 'Volume up'], ['Typed media command', 'Configured player', 'Final result + source'],
     'Synthetic playback', 'Paused · simulated', 'Provider semantics apply: Spotify STOP is pause, not a claimed device stop. No playback changed here.'),
    ('UC-09', 'Reminders', ['CREATE_REMINDER', 'LIST_REMINDERS'], ['Remind me to Study', 'List my reminders'], ['Typed task ID', 'Persistent local store', 'Saved task / list'],
     'Local job example', 'Study', 'Finite tasks: drink water / study / exercise. The application persists the list; this example writes nothing.'),
    ('UC-10', 'Calls / messages', ['CALL', 'MESSAGE'], ['Call', 'Message'], ['Typed CALL / MESSAGE', 'Camera / preset notice', 'Video / delivery receipt'],
     'Synthetic camera / notice', 'Home camera · video only', 'Target camera.dining_area_main_camera_tc70_mainstream. No telephone call, stream or notification is started by this deck.'),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def checked(path: Path, expected: str):
    if path.is_symlink() or digest(path) != expected:
        raise ValueError(f'identity mismatch: {path.name}')
    return path


def historical(root: Path):
    folder = root / 'presentation/evidence'
    manifest = load(folder / 'manifest.json')
    for name, binding in manifest['files'].items():
        checked(folder / name, binding['sha256'])
    summary = load(folder / 'training-summary.json')
    curve = load(folder / 'training-curve.json')
    if summary['parameters'] != 258912 or summary['selected_epoch'] != 44 or len(curve) != 48:
        raise ValueError('historical training belongs to another model')
    runs = []
    for name, label in [('pc-original.json', 'Historical original PC run'), ('pc-repeat-v2.json', 'Historical later PC repeat v2')]:
        receipt = load(folder / name)
        for key, value in receipt['model']['binding'].items():
            if MODEL_HASHES.get(key) != value:
                raise ValueError('historical benchmark model mismatch')
        runs.append({
            'label': label, 'historical': True, 'receipt': f'presentation/evidence/{name}',
            'receipt_sha256': manifest['files'][name]['sha256'],
            'source_revision': manifest['source_revision'],
            'original_recordings': receipt['independent_original_human_recordings'],
            'timed_calls': receipt['observations'], 'repeats': receipt['repeats'],
            'hardware': receipt['hardware'], 'latency': receipt['latency_all'],
            'timing_includes': receipt['timing_includes'], 'timing_excludes': receipt['timing_excludes'],
            'cold_runtime_init_ms': receipt['cold_runtime_init_ms'],
            'onnx_session_load_ms': receipt['onnx_session_load_ms'],
        })
    return summary, [{'epoch': r['epoch'], 'loss': r['loss'], 'phase': r['phase']} for r in curve], runs, manifest


def use_cases(commands):
    definitions = baseline_cases()
    if len(definitions) != 93:
        raise ValueError('finite wording definition drift')
    cases = []
    for ident, title, intents, texts, diagram, badge, headline, detail in CATEGORIES:
        examples = []
        for text in texts:
            matches = [r for r in definitions if r['text'] == text]
            if len(matches) != 1 or matches[0]['command'] not in commands:
                raise ValueError(f'unsupported example: {text}')
            examples.append({'wording': text, 'command': matches[0]['command']})
        cases.append({
            'id': ident, 'title': title, 'intents': intents,
            'class_indices': [i for i, c in enumerate(commands) if c and c['intent'] in intents],
            'examples': examples, 'diagram': diagram,
            'output': {'badge': badge, 'headline': headline, 'detail': detail, 'example_only': True,
                       'source': 'local' if ident in {'UC-05', 'UC-06', 'UC-09'} else 'synthetic',
                       'provider_accepted': False, 'device_state_confirmed': False,
                       'observed_at': None, 'fetched_at': None},
        })
    indices = [i for case in cases for i in case['class_indices']]
    if sorted(indices) != list(range(1, 32)):
        raise ValueError('ten categories must cover each of the 31 commands exactly once')
    return cases, definitions


def build_publication(root: Path = ROOT, desktop: Path | None = None, pi: Path | None = None):
    snapshot_path = checked(root / 'submission/optionb-tiny/snapshot.json', SNAPSHOT_HASH)
    snapshot = load(snapshot_path)
    folder = root / snapshot['bundle']
    for name, expected in MODEL_HASHES.items():
        checked(folder / name, expected)
    recipe, policy, commands = (load(folder / name) for name in ('recipe.json', 'policy.json', 'commands.json'))
    if len(commands) != 32 or commands[0] is not None or recipe['parameters'] != 258912:
        raise ValueError('not the frozen atomic model')
    splits, training_sources = {}, Counter()
    for split, view in sorted(snapshot['views'].items()):
        path = checked(root / view['path'], view['sha256'])
        rows = [json.loads(line) for line in gzip.open(path, 'rt', encoding='utf-8')]
        kinds = dict(Counter(row['kind'] for row in rows))
        if len(rows) != view['rows'] or kinds != view['kinds']:
            raise ValueError(f'{split} membership mismatch')
        splits[split] = {kind: kinds.get(kind, 0) for kind in ('human', 'synthetic')}
        if split == 'train':
            training_sources.update((r['source'], r['location']['revision'], r['kind']) for r in rows)
    cases, definitions = use_cases(commands)
    summary, curve, runs, evidence = historical(root)
    companions = []
    for component, metadata, graph in [
        ('Hey Iris wake', 'artifacts/iris/wake/iris.json', 'artifacts/iris/wake/iris.onnx'),
        ('Owner / overlap', 'artifacts/iris/owner/owner.json', 'artifacts/iris/owner/owner.onnx'),
        ('Silero VAD', 'artifacts/iris/wake/wake-models.json', 'artifacts/iris/wake/silero-vad.onnx'),
    ]:
        info = load(root / metadata)
        companions.append({'component': component, 'parameters': info.get('parameters', info.get('model', {}).get('runtime_parameters')),
                           'graph_bytes': (root / graph).stat().st_size, 'graph_sha256': digest(root / graph),
                           'source': metadata, 'included_in_vcm_count': False})
    reference_binding = load(root / 'presentation/evidence/pc-reference-source.json')
    reference_path = checked(root / 'presentation/evidence/pc-reference-under-load.json', reference_binding['sha256'])
    reference = load(reference_path)
    if reference['format'] != 'optionb-pc-benchmark-results-v1' or reference['model']['binding'] != {k: v for k, v in MODEL_HASHES.items() if k != 'recipe.json'}:
        raise ValueError('reference benchmark identity mismatch')
    if reference['cohorts'] != snapshot['expected_pc_metrics']:
        # Latencies are measured afresh; only acoustic outcomes must remain frozen.
        for split in ('test', 'holdout'):
            for field in ('selected', 'raw', 'exact_wording', 'original_rows', 'ambiguous_unscored'):
                if reference['cohorts'][split][field] != snapshot['expected_pc_metrics'][split][field]:
                    raise ValueError('reference benchmark cohort mismatch')
    reference_run = {'label': '2026-10-24 reference driver / under publication load', 'historical': False,
                     'receipt': 'presentation/evidence/pc-reference-under-load.json', 'receipt_sha256': digest(reference_path),
                     'original_recordings': reference['independent_original_human_recordings'],
                     'timed_calls': reference['observations'], 'repeats': reference['repeats'],
                     'hardware': reference['hardware'], 'latency': reference['latency_all'],
                     'measurement_not_quiesced': reference['publication_provenance']['measurement_not_quiesced'],
                     'context': reference['publication_provenance']['context'],
                     'measured_at': reference['publication_provenance']['measured_at_utc'],
                     'timing_includes': reference['timing_includes'], 'timing_excludes': reference['timing_excludes']}
    wake = load(checked(root / 'presentation/evidence/wake-smoke-status.json', '8f3e9fc933721d8b63ef5cb346b4ae6ee086a73c9bd6b769b676db5375891f46'))
    result = {
        'schema_version': 1, 'wake': wake, 'product': {'id': 'iris-epoch44', 'name': 'Iris', 'wake_phrase': 'Hey Iris'},
        'model': {'id': 'OptionBTinyVCM-1', 'parameters': recipe['parameters'], 'selected_epoch': recipe['selected_epoch'],
                  'precision': 'FP32', 'graph_bytes': (folder / 'model.onnx').stat().st_size,
                  'bundle_bytes': sum((folder / name).stat().st_size for name in MODEL_HASHES),
                  'hashes': MODEL_HASHES, 'command_count': len(commands) - 1, 'definition_count': len(definitions),
                  'intent_count': len({c['intent'] for c in commands if c}), 'unknown_index': 0,
                  'policy': policy, 'qualified': False, 'architecture': snapshot['architecture'],
                  'snapshot_sha256': digest(snapshot_path)},
        'commands': [{'class_index': i, 'command': command} for i, command in enumerate(commands) if command],
        'definitions': definitions, 'use_cases': cases, 'companions': companions,
        'datasets': {'splits': splits, 'normalization_human_train': splits['train']['human'],
                     'training_sources': [{'repository': repo, 'revision': rev, 'kind': kind, 'rows': count}
                                          for (repo, rev, kind), count in sorted(training_sources.items())],
                     'source_terms': 'Source-specific research/education and consent restrictions remain; not a blanket MIT dataset.'},
        'training': {'recipe': snapshot['training'], 'device': snapshot['training_device'], 'optimizer': 'AdamW',
                     'optimizer_steps': summary['optimizer_steps'], 'wall_seconds': summary['elapsed_seconds'],
                     'curve': curve, 'curve_source': 'presentation/evidence/training-curve.json',
                     'artifact_loading': 'PASS (separate verifier)', 'original_cpu_exact_retraining': 'NOT VERIFIED / comparison failed',
                     'gpu_repeatability': 'Two historical CUDA runs matched each other; selected epoch28, not original epoch44'},
        'cohorts': snapshot['expected_pc_metrics'],
        'benchmarks': {'historical_desktop': runs, 'reference_desktop': reference_run, 'desktop': None, 'pi': None,
                       'timing_scope': 'recognize(PCM): validation + frontend + normalization + CPU ONNX + finite decision',
                       'excludes': runs[-1]['timing_excludes']},
        'provenance': {'snapshot': 'submission/optionb-tiny/snapshot.json', 'evidence': evidence,
                       'examples': 'Static storyboard only: no recognition, provider request or state mutation.',
                       'pi_attachment': 'Same graph/policy, named Pi hardware, actual receipt and matching timing scope required.'},
    }
    # Optional current receipts use a strict versioned bridge, not fuzzy field inference.
    for target, receipt_path in [('desktop', desktop), ('pi', pi)]:
        if receipt_path is not None:
            result['benchmarks'][target] = validate_measurement(load(receipt_path), target)
    return result


def validate_measurement(receipt, target):
    if target not in {'desktop', 'pi'}:
        raise ValueError('unsupported measurement target')
    if receipt.get('schema_version') != 1 or receipt.get('format') != 'iris-publication-measurement-v1':
        raise ValueError('unsupported measurement receipt')
    if receipt.get('target') != target:
        raise ValueError('measurement target mismatch')
    for key in ('model.onnx', 'policy.json'):
        if receipt.get('hashes', {}).get(key) != MODEL_HASHES[key]:
            raise ValueError('measurement graph/policy mismatch')
    hardware = receipt.get('hardware', {})
    if not hardware.get('cpu') or not hardware.get('platform') or not hardware.get('python'):
        raise ValueError('measurement requires named hardware, platform and Python')
    if target == 'pi' and 'raspberry pi' not in hardware.get('device', '').lower():
        raise ValueError('Pi measurement requires named Raspberry Pi model')
    if receipt.get('timing_scope') != 'recognize_pcm' or not receipt.get('timing_excludes'):
        raise ValueError('measurement timing scope missing or incompatible')
    for key in ('p50_ms', 'p95_ms'):
        value = receipt.get('latency', {}).get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid measured latency')
    if receipt['latency']['p95_ms'] < receipt['latency']['p50_ms']:
        raise ValueError('invalid latency percentiles')
    for key in ('original_recordings', 'timed_calls', 'repeats'):
        if type(receipt.get(key)) is not int or receipt[key] <= 0:
            raise ValueError('missing measured counts')
    if receipt['timed_calls'] != receipt['original_recordings'] * receipt['repeats']:
        raise ValueError('timed calls must not inflate acoustic denominator')
    if not receipt.get('receipt_path') or not receipt.get('measured_at'):
        raise ValueError('measurement requires source and time')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/site/assets/iris.json')
    parser.add_argument('--desktop', type=Path, help='Versioned current measurement bridge, not a historical repeat')
    parser.add_argument('--pi', type=Path)
    parser.add_argument('--check', action='store_true', help='Fail if committed JSON is stale; write nothing')
    args = parser.parse_args()
    data = build_publication(ROOT, args.desktop, args.pi)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.check:
        if args.output.read_text(encoding='utf-8') != text:
            raise SystemExit('publication JSON is stale; regenerate it with the same receipt arguments')
        print('publication_data_verified: 31 commands; 93 definitions; ten categories; frozen graph; Pi unmeasured' if data['benchmarks']['pi'] is None else 'publication_data_verified with measured Pi receipt')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8', newline='\n')
        print(f'publication_data_written: {args.output}')


if __name__ == '__main__':
    main()
