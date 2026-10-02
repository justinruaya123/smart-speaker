"""E18 locked historical regression; fixed, custom and numeric tests stay separate."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import time
import numpy as np
from einops import rearrange, reduce

from .prototype_data import ROOT, EXP, DATA, CACHE, dataset, verify
from .complete_data import read_rows, save, digest, finite_reference, custom_key
from .complete_eval import raw, raw_command, summarize_fixed, custom_metrics
from .inference import Recognizer
from .audio import load_audio

PARENT_EXP = ROOT / 'experiments/vcm-complete-v1'
PARENT_DATA = ROOT / 'data/vcm-complete-v1'
PARENT_CACHE = ROOT / '.cache/vcm-complete-v1'


def custom_records(folder, partition):
    catalog = json.loads((PARENT_EXP / 'custom-catalog-v3.json').read_text(encoding='utf-8'))['catalog']
    all_rows = read_rows(PARENT_DATA / 'manifest.jsonl')
    by_id = {r['id']: r for r in all_rows}
    enrollment = [by_id[i] for e in catalog for i in e['enrollment_ids']]
    training_speakers = {r['speaker_id'] for r in dataset({'train'})}
    query = [r for r in all_rows if r['kind'] == 'human' and custom_key(r) and
             (r['split'] == 'test' if partition == 'test' else r['split'] == 'dev' and r['fold'] == 'selection')]
    if {r['speaker_id'] for r in enrollment} & (training_speakers | {r['speaker_id'] for r in query}):
        raise ValueError('Custom enrollment speaker leakage')
    if {r['paired_group'] for r in enrollment} & {r['paired_group'] for r in query}:
        raise ValueError('Custom enrollment pair leakage')
    recognizer = Recognizer(folder, threads=2)
    prototypes = []
    for entry in catalog:
        vectors = [raw(recognizer, load_audio(ROOT / by_id[i]['audio'])[0], True)[2] for i in entry['enrollment_ids']]
        vector = reduce(rearrange(vectors, 'n c -> n c'), 'n c -> c', 'mean')
        prototypes.append(vector / max(np.linalg.norm(vector), 1e-8))
    matrix = rearrange(prototypes, 'n c -> n c')
    keys = [json.dumps(e['reference_arguments'], sort_keys=True) for e in catalog]
    records = []
    for row in query:
        family, slots, embedding = raw(recognizer, load_audio(ROOT / row['audio'])[0], True)
        similarity = matrix @ embedding
        order = np.argsort(similarity)[::-1]
        index = int(order[0]); name = recognizer.labels[int(family.argmax())]
        mode = recognizer.metadata['acoustic_slots']['vocabulary']['media_mode']
        key = custom_key(row)
        records.append({'id': row['id'], 'known': key in keys, 'correct': key == keys[index],
            'score': float(similarity[index]), 'margin': float(similarity[index] - similarity[order[1]]),
            'predicted_id': catalog[index]['id'],
            'family_accepted': name == 'MEDIA_PLAY' and float(family.max()) >= recognizer.routing_policy()['acceptance_thresholds'][name],
            'custom_mode': mode[int(slots['media_mode'].argmax())] == 'CUSTOM',
            'custom_probability': float(slots['media_mode'][mode.index('CUSTOM')]),
            'playlist': any(e['playlist_subset'] and key == keys[j] for j, e in enumerate(catalog))})
    return records


def calibrate_custom(folder):
    rows = custom_records(folder, 'dev')
    best, selected = None, {'score': 1.1, 'margin': 1.1, 'custom_probability': 1.1}
    for score in np.arange(.3, 1.001, .025):
        for margin in [0., .01, .02, .04, .08, .12, .2]:
            for probability in [.5, .7, .9]:
                policy = {'score': float(score), 'margin': margin, 'custom_probability': probability}
                metric = custom_metrics(rows, policy, True)
                if metric['accepted'] >= 5 and metric['precision'] >= .95 and metric['unknown_false_accept_rate'] <= .01:
                    rank = (metric['correct'], -metric['accepted'], score + margin)
                    if best is None or rank > best: best, selected = rank, policy
    receipt = {'policy': selected, 'gate_passed': best is not None, 'test_used': False,
        'catalog_sha256': digest(PARENT_EXP / 'custom-catalog-v3.json'),
        'conditional': custom_metrics(rows, selected, False), 'complete': custom_metrics(rows, selected, True)}
    save(EXP / 'custom-development.json', receipt)
    save(CACHE / 'custom-development-records.json', rows)
    return receipt


def score_views(predictions, revised, original):
    reports = {}
    for label, rows in [('original_silver', original), ('revised_silver', revised)]:
        if {p['id'] for p in predictions} != {r['id'] for r in rows} or len(predictions) != len(rows):
            raise ValueError('Test prediction coverage changed')
        by_id = {r['id']: r for r in rows}
        scored = [{**p, 'gold': by_id[p['id']]['command'], 'reference': finite_reference(by_id[p['id']])} for p in predictions]
        report = summarize_fixed(scored)
        report['per_source'] = {source: summarize_fixed([p for p in scored if by_id[p['id']]['source'] == source])['totals']
            for source in sorted({r['source'] for r in rows})}
        if not all('raw_family' in p for p in scored):
            for b in [report['totals'], *report['per_gold_family'].values(), *report['per_source'].values()]:
                b.pop('raw_family_correct', None); b.pop('raw_command_correct', None)
        reports[label] = report
    return reports


def fixed_test(folder):
    from .pipeline import Pipeline
    from .wake_models import SileroVAD
    from .config import load_config
    original, revised = dataset({'test'}, revised=False), dataset({'test'})
    if len(original) != 4314 or sum(finite_reference(r) is not None for r in original) != 1783:
        raise ValueError('Historical test denominator changed')
    start = time.perf_counter(); recognizer = Recognizer(folder, threads=2)
    load_ms = (time.perf_counter() - start) * 1000
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config = load_config(ROOT / 'pipeline.toml')
    pipeline.bundle, pipeline.recognizer = folder, recognizer
    pipeline.vad = SileroVAD(ROOT / 'artifacts', pipeline.config['audio'])
    output, timing = {'direct': [], 'silero': []}, []
    last = time.monotonic()
    for row in original:
        pcm, _ = load_audio(ROOT / row['audio'])
        result = recognizer.predict(pcm, stream=False)
        family, slots, _ = raw(recognizer, pcm)
        raw_family, raw_value = raw_command(recognizer, family, slots)
        common = {'id': row['id'], 'gold': row['command'], 'reference': finite_reference(row)}
        output['direct'].append({**common, 'result': result, 'outputs': [result], 'raw_family': raw_family, 'raw_command': raw_value})
        timing.append(result['processing_ms'])
        results = pipeline.process(pcm)
        endpoint = results[0] if len(results) == 1 else {'status': 'rejected', 'reason': 'multiple_endpoints'}
        output['silero'].append({**common, 'result': endpoint, 'outputs': results})
        if time.monotonic() - last > 20:
            print(f'LOCKED_TEST {len(output["direct"])}/{len(original)}', flush=True); last = time.monotonic()
    report = {'candidate': {}, 'frozen_E13_control': {}}
    for mode, values in output.items():
        path = CACHE / (mode + '-test.jsonl')
        with path.open('w', encoding='utf-8') as stream:
            for row in values: stream.write(json.dumps(row) + '\n')
        report['candidate'][mode] = score_views(values, revised, original)
        prior = read_rows(PARENT_CACHE / ('human_keywords-' + mode + '-test.jsonl'))
        report['frozen_E13_control'][mode] = score_views(prior, revised, original)
    report['desktop_cpu'] = {'threads': 2, 'session_load_ms': load_ms,
        'predict_p50_ms': float(np.median(timing[5:])), 'predict_p95_ms': float(np.quantile(timing[5:], .95)),
        'excludes': 'disk, utterance duration, wake inference, endpoint wait', 'Pi_measured': False}
    report['provenance'] = 'Previously examined historical human test; original and revised programmatic silver labels. No fresh blind claim.'
    save(EXP / 'fixed-test.json', report)
    return report


def number_test(folder):
    from .routing import parse_parameters
    recognizer = Recognizer(folder, threads=2)
    rows = []
    for row in read_rows(PARENT_DATA / 'manifest.jsonl'):
        amount = row['annotation']['labels'].get('amount')
        if row['split'] == 'test' and isinstance(amount, str) and amount.isdigit():
            rows.append({**row, 'value': int(amount), 'group': row['kind'] + '_connected_commands',
                'expected_family': row['command'], 'reference_value': finite_reference(row)})
    generation = json.loads((PARENT_EXP / 'number-stress-generation-v2.json').read_text(encoding='utf-8'))
    stress = PARENT_EXP / 'number-stress-manifest-v2.jsonl'
    if not generation['complete'] or digest(stress) != generation['manifest_sha256']:
        raise ValueError('Numeric stress manifest changed')
    for row in read_rows(stress):
        if digest(ROOT / row['audio']) != row['audio_sha256']: raise ValueError('Stress audio changed')
        reference = parse_parameters('VOLUME', row['transcript'], recognizer.catalog).get('command') if row['kind'] == 'volume_command' else None
        rows.append({**row, 'group': 'qwen_stress_' + row['kind'], 'reference_value': reference})
    order = json.loads((PARENT_DATA / 'number-test-plan.json').read_text(encoding='utf-8'))['randomized_values']
    rows.sort(key=lambda r: (order.index(r['value']), r['id']))
    groups, per_value, predictions = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), []
    vocab = recognizer.metadata['acoustic_slots']['vocabulary']['amount']
    for row in rows:
        pcm, _ = load_audio(ROOT / row['audio'])
        family, slots, _ = raw(recognizer, pcm)
        result = recognizer.predict(pcm, stream=False)
        amount = vocab[int(slots['amount'].argmax())]
        emitted = result.get('status') == 'accepted' and 'command' in result
        counts = {'examples': 1, 'raw_amount_correct': amount == str(row['value']),
            'raw_family_correct': recognizer.labels[int(family.argmax())] == row['expected_family'],
            'finite_references': row['reference_value'] is not None,
            'correct_complete': bool(row['reference_value'] is not None and emitted and result.get('command') == row['reference_value'] and result.get('vcm_command') == row['expected_family']),
            'emitted': emitted, 'expected_abstention': row['expected_family'] == 'UNKNOWN',
            'correct_abstention': row['expected_family'] == 'UNKNOWN' and not emitted}
        groups[row['group']].update(counts); per_value[row['group']][str(row['value'])].update(counts)
        predictions.append({'id': row['id'], 'group': row['group'], 'expected_amount': row['value'], 'raw_amount': amount, 'result': result})
    report = {'groups': groups, 'per_value': per_value, 'random_order': order,
        'scope': 'Separate human connected commands, existing synthetic connected commands, and 0-100 bare/distractor stress. No new synthesis or threshold fitting.',
        'labels': 'Human programmatic silver and requested TTS text, not independent human adjudication.'}
    save(EXP / 'numbers-test.json', report); save(CACHE / 'number-predictions.json', predictions)
    return report


def evaluate():
    verify()
    selection_path = EXP / 'selection-before-test.json'
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    for name, sha in selection['files'].items():
        if digest(ROOT / name) != sha: raise ValueError('Changed after selection: ' + name)
    if (EXP / 'results.json').exists(): raise ValueError('Preserve the completed test; use a new version')
    folder = CACHE / selection['selected_arm']
    fixed = fixed_test(folder)
    numbers = number_test(folder)
    rows = custom_records(folder, 'test'); policy = selection['custom_policy']['policy']
    custom = {'catalog_entries': 4, 'playlist_entries': 1, 'enrollments_per_entry': 3,
        'conditional': custom_metrics(rows, policy, False), 'complete': custom_metrics(rows, policy, True), 'policy': policy,
        'scope': 'Human media proxy catalog, not newly recorded household playlists; custom reader remains research-only.'}
    save(EXP / 'custom-test.json', custom); save(CACHE / 'custom-test-records.json', rows)
    gate = all(fixed['candidate'][mode][view]['joint_gate_passed'] for mode in ['direct', 'silero'] for view in ['original_silver', 'revised_silver'])
    results = {'selected_arm': selection['selected_arm'], 'selection_sha256': digest(selection_path),
        'fixed': {a: {m: {v: r['totals'] for v,r in reports.items()} for m,reports in modes.items()}
            for a,modes in fixed.items() if a in ['candidate', 'frozen_E13_control']},
        'numbers': numbers['groups'], 'custom': custom, 'fixed_joint_gate_passed': gate,
        'installed': False, 'physical_pi_validation': False, 'live_microphone_tested': False}
    save(EXP / 'results.json', results)
    release = {'format_version': 1, 'experiment': EXP.name, 'complete': True,
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'arm': selection['selected_arm'],
        'bundle_directory': folder.relative_to(ROOT).as_posix(),
        'selection_receipt': selection_path.relative_to(ROOT).as_posix(), 'selection_sha256': digest(selection_path),
        'test_receipt': (EXP / 'results.json').relative_to(ROOT).as_posix(), 'test_sha256': digest(EXP / 'results.json'),
        'files': {name: {'sha256': digest(folder / name), 'bytes': (folder / name).stat().st_size} for name in ['vcm.fp32.onnx', 'bundle.json']},
        'fixed_joint_gate_passed': gate, 'requirements_complete': False,
        'status': 'experimental_not_installed', 'physical_pi_validation': False,
        'resource_targets': {'threads': 2, 'peak_rss_mib': 512, 'recognition_p95_ms': 100},
        'unsupported': ['certified custom playlists', 'complete spoken numeric range 0-100', 'all ten assignment use cases', 'physical Pi and live microphone acceptance']}
    save(EXP / 'release.json', release)
    print(json.dumps({'arm': selection['selected_arm'], 'fixed_gate': gate, 'fixed': results['fixed']['candidate'], 'custom': custom, 'numbers': numbers['groups']}), flush=True)
