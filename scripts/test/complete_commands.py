"""Locked fixed-command, enrollment and numeric evaluation for E13/E14."""
from collections import Counter, defaultdict
import json
import time
import numpy as np
from einops import rearrange, reduce

from .complete_data import ROOT, DATA, EXP, CACHE, read_rows, save, digest, finite_reference, custom_key, verify_frozen
from .inference import Recognizer, softmax
from .audio import load_audio, features
from .config import load_config
from .family_contract import family_candidate
from .routing import PARAMETER_COMMANDS, parse_parameters
from .complete_training import experiment_for

ARMS = ['human_full', 'human_filtered', 'human_keywords', 'mixed_keywords', 'v2_background', 'v2_masked']


def raw(recognizer, pcm, embedding=False):
    vocab = recognizer.metadata['acoustic_slots']['vocabulary']
    names = ['logits', *['slot_' + k for k in vocab]] + (['embedding'] if embedding else [])
    x = recognizer.normalize(features(pcm))
    outputs = recognizer.vcm.run(names, {'features': x, 'lengths': np.array([x.shape[-1]], dtype=np.int64)})
    family = softmax(outputs[0][0], recognizer.routing_policy().get('temperature', 1.))
    slots = {name: softmax(output[0]) for name, output in zip(vocab, outputs[1:])}
    return family, slots, outputs[-1][0] if embedding else None


def raw_command(recognizer, probabilities, slots):
    family = recognizer.labels[int(probabilities.argmax())]
    if family == 'UNKNOWN':
        return family, None
    if family not in PARAMETER_COMMANDS:
        return family, parse_parameters(family, '', recognizer.catalog).get('command')
    candidate = family_candidate(family, slots, recognizer.metadata['acoustic_slots'], recognizer.catalog)
    return family, candidate['result'].get('command') if candidate['result'] else None


def custom_records(arm, partition):
    """No gold family in the complete custom path; conditional retrieval is separate."""
    catalog = json.loads((EXP / 'custom-catalog-v3.json').read_text())['catalog']
    all_rows = read_rows(DATA / 'manifest.jsonl')
    by_id = {r['id']: r for r in all_rows}
    recognizer = Recognizer(CACHE / arm, threads=2)
    prototypes = []
    enrollment = [by_id[id] for entry in catalog for id in entry['enrollment_ids']]
    training_speakers = {r['speaker_id'] for r in all_rows if r['kind'] == 'human' and r['split'] == 'train' and not r['unusable_reasons']}
    query = [r for r in all_rows if r['kind'] == 'human' and custom_key(r) and
             ((partition == 'dev' and r['split'] == 'dev' and r['fold'] == 'selection') or
              (partition == 'test' and r['split'] == 'test'))]
    assert not {r['speaker_id'] for r in enrollment} & (training_speakers | {r['speaker_id'] for r in query})
    assert not {r['paired_group'] for r in enrollment} & {r['paired_group'] for r in query}
    for entry in catalog:
        vectors = []
        for id in entry['enrollment_ids']:
            pcm, _ = load_audio(ROOT / by_id[id]['audio'])
            vectors.append(raw(recognizer, pcm, True)[2])
        prototype = reduce(rearrange(vectors, 'n c -> n c'), 'n c -> c', 'mean')
        prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-8))
    matrix = rearrange(prototypes, 'n c -> n c')
    keys = [json.dumps(entry['reference_arguments'], sort_keys=True) for entry in catalog]
    result = []
    for row in query:
        pcm, _ = load_audio(ROOT / row['audio'])
        probabilities, slots, vector = raw(recognizer, pcm, True)
        similarity = matrix @ vector
        order = np.argsort(similarity)[::-1]
        index = int(order[0])
        family = recognizer.labels[int(probabilities.argmax())]
        family_accepted = family == 'MEDIA_PLAY' and float(probabilities.max()) >= recognizer.routing_policy()['acceptance_thresholds'][family]
        mode = recognizer.metadata['acoustic_slots']['vocabulary']['media_mode']
        is_custom = mode[int(slots['media_mode'].argmax())] == 'CUSTOM'
        key = custom_key(row)
        result.append({'id': row['id'], 'known': key in keys, 'correct': key == keys[index],
            'score': float(similarity[index]), 'margin': float(similarity[order[0]] - similarity[order[1]]),
            'predicted_id': catalog[index]['id'], 'family_accepted': family_accepted,
            'custom_mode': is_custom, 'custom_probability': float(slots['media_mode'][mode.index('CUSTOM')]),
            'playlist': any(e['playlist_subset'] and key == keys[j] for j, e in enumerate(catalog))})
    return result


def custom_metrics(rows, policy, complete):
    counts = Counter(queries=len(rows), known=sum(r['known'] for r in rows), unknown=sum(not r['known'] for r in rows),
        playlist_queries=sum(r['playlist'] for r in rows), raw_top1_correct=sum(r['known'] and r['correct'] for r in rows))
    for r in rows:
        accept = r['score'] >= policy['score'] and r['margin'] >= policy['margin']
        if complete:
            accept = accept and r['family_accepted'] and r['custom_mode'] and r['custom_probability'] >= policy['custom_probability']
        counts['accepted'] += bool(accept)
        counts['correct'] += bool(accept and r['correct'])
        counts['unknown_false_accepts'] += bool(accept and not r['known'])
        counts['playlist_correct'] += bool(accept and r['correct'] and r['playlist'])
    return {**counts, 'recall': counts['correct'] / max(1, counts['known']),
        'precision': counts['correct'] / max(1, counts['accepted']),
        'unknown_false_accept_rate': counts['unknown_false_accepts'] / max(1, counts['unknown']),
        'conditional_on_gold_custom_family': not complete}


def calibrate_custom(arm):
    rows = custom_records(arm, 'dev')
    best, selected = None, {'score': 1.1, 'margin': 1.1, 'custom_probability': 1.1}
    # Fixed grid frozen in evaluation implementation before custom test access.
    for score in np.arange(.3, 1.001, .025):
        for margin in [0., .01, .02, .04, .08, .12, .2]:
            for custom_probability in [.5, .7, .9]:
                policy = {'score': float(score), 'margin': margin, 'custom_probability': custom_probability}
                metric = custom_metrics(rows, policy, True)
                if metric['accepted'] >= 5 and metric['precision'] >= .95 and metric['unknown_false_accept_rate'] <= .01:
                    rank = (metric['correct'], -metric['accepted'], score + margin)
                    if best is None or rank > best:
                        best, selected = rank, policy
    receipt = {'arm': arm, 'partition': 'human development selection only', 'policy': selected,
        'gate_passed': best is not None, 'catalog_sha256': digest(EXP / 'custom-catalog-v3.json'),
        'conditional': custom_metrics(rows, selected, False), 'complete': custom_metrics(rows, selected, True),
        'test_used': False, 'raw_top1_note': 'Closed-catalog ranking is diagnostic; unknown names require calibrated abstention.'}
    save(EXP / 'custom-development.json', receipt)
    save(CACHE / 'custom-development-records.json', rows)
    return receipt


def lock():
    verify_frozen()
    path = EXP / 'selection-before-test.json'
    if path.exists():
        raise ValueError('Selection is already locked; do not overwrite it')
    development = {arm: json.loads((experiment_for(arm) / (arm + '-development.json')).read_text()) for arm in ARMS}
    selected = max(ARMS, key=lambda arm: tuple(development[arm]['rank']))
    highest_recall = max(ARMS, key=lambda arm: development[arm]['commands']['exact_command_success'])
    custom = calibrate_custom(selected)
    files = [DATA / 'frozen.json', EXP / 'protocol.json', EXP.parent / 'vcm-complete-v2/protocol.json',
        EXP / 'custom-catalog-v3.json', EXP / 'custom-protocol-v3.json', EXP / 'custom-development.json',
        EXP / 'number-stress-plan-v2.json', ROOT / 'scripts/test/complete_commands.py']
    for arm in ARMS:
        files.extend(CACHE / arm / name for name in ['vcm.fp32.onnx', 'bundle.json'])
    files.extend(ROOT / 'artifacts' / name for name in ['vcm.fp32.onnx', 'bundle.json', 'silero-vad.onnx'])
    receipt = {'selected_arm': selected, 'development_ranks': {k: v['rank'] for k, v in development.items()},
        'custom_policy': custom['policy'], 'selection_rule': 'Declared dev gate pass, complete recall, minimum family recall, precision',
        'files': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files},
        'test_used_to_select': False, 'created_unix': time.time(),
        'highest_development_recall_arm': highest_recall,
        'test_arms': list(dict.fromkeys(['installed_baseline', selected, highest_recall])),
        'test_comparison_note': 'Report safest dev-qualified candidate and highest-recall dev candidate separately; test cannot change their selection status.',
        'numeric_diagnostic_arms': list(dict.fromkeys(['installed_baseline', selected, highest_recall, 'mixed_keywords'])),
        'scope': 'Original human fixed-command regression; custom and numeric evidence kept separate. No installation.'}
    save(path, receipt)
    print(json.dumps({'selected': selected, 'ranks': receipt['development_ranks'], 'custom_gate': custom['gate_passed']}), flush=True)
    return receipt


def summarize_fixed(rows):
    totals, families, predicted = Counter(), defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        family = r['gold']
        bucket = families[family]
        gold = r['reference']
        result = r['result']
        success = gold is not None and len(r['outputs']) == 1 and result.get('status') == 'accepted' and result.get('vcm_command') == family and result.get('command') == gold
        raw_correct = gold is not None and r.get('raw_family') == family and r.get('raw_command') == gold
        emitted = sum(out.get('status') == 'accepted' and 'command' in out for out in r['outputs'])
        updates = {'recordings': 1, 'finite_references': gold is not None, 'correct_complete': success,
            'emitted': emitted, 'raw_family_correct': r.get('raw_family') == family,
            'accepted_family_correct': result.get('intent_accepted', False) and result.get('vcm_command') == family,
            'raw_command_correct': raw_correct, 'unknown_recordings': family == 'UNKNOWN',
            'unknown_family_false_accepts': family == 'UNKNOWN' and any(out.get('intent_accepted', False) for out in r['outputs']),
            'unknown_command_false_accepts': emitted if family == 'UNKNOWN' else 0,
            'parameter_calls': sum(bool(out.get('parameter_invoked')) for out in r['outputs'])}
        totals.update(updates); bucket.update(updates)
        for out in r['outputs']:
            predicted[out.get('vcm_command', 'UNKNOWN')]['emitted'] += out.get('status') == 'accepted' and 'command' in out
        if success:
            predicted[family]['correct_complete'] += 1
    for bucket in [totals, *families.values()]:
        bucket['complete_recall'] = bucket['correct_complete'] / max(1, bucket['finite_references'])
        bucket['complete_precision'] = bucket['correct_complete'] / max(1, bucket['emitted'])
        bucket['unknown_family_false_accept_rate'] = bucket['unknown_family_false_accepts'] / max(1, bucket['unknown_recordings'])
    return {'totals': dict(totals), 'per_gold_family': families, 'per_predicted_family': predicted,
        'target_70_passed': totals['complete_recall'] >= .7,
        'joint_gate_passed': totals['complete_recall'] >= .7 and totals['complete_precision'] >= .95 and totals['unknown_family_false_accept_rate'] <= .01}


def fixed_test(arm):
    from .pipeline import Pipeline
    from .wake_models import SileroVAD
    folder = ROOT / 'artifacts' if arm == 'installed_baseline' else CACHE / arm
    started = time.perf_counter(); recognizer = Recognizer(folder, threads=2)
    load_ms = (time.perf_counter() - started) * 1000
    config = load_config(ROOT / 'pipeline.toml')
    # Compose canonical Pipeline methods with the unchanged installed VAD session.
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.config, pipeline.bundle, pipeline.recognizer = config, folder, recognizer
    pipeline.vad = SileroVAD(ROOT / 'artifacts', config['audio'])
    rows = [r for r in read_rows(DATA / 'manifest.jsonl') if r['kind'] == 'human' and r['split'] == 'test']
    assert len(rows) == 4314 and sum(finite_reference(r) is not None for r in rows) == 1783
    output, timings = {'direct': [], 'silero': []}, []
    last = time.monotonic()
    for row in rows:
        pcm, _ = load_audio(ROOT / row['audio'])
        result = recognizer.predict(pcm, stream=False)
        family, slots, _ = raw(recognizer, pcm)
        raw_family, raw_value = raw_command(recognizer, family, slots)
        timings.append(result.get('processing_ms', 0.))
        common = {'id': row['id'], 'gold': row['command'], 'reference': finite_reference(row)}
        output['direct'].append({**common, 'result': result, 'outputs': [result], 'raw_family': raw_family, 'raw_command': raw_value})
        results = pipeline.process(pcm)
        endpoint = results[0] if len(results) == 1 else {'status': 'rejected', 'reason': 'multiple_endpoints'}
        output['silero'].append({**common, 'result': endpoint, 'outputs': results})
        if time.monotonic() - last > 20:
            print(f'FIXED_TEST {arm} {len(output["direct"])}/{len(rows)}', flush=True); last = time.monotonic()
    report = {mode: summarize_fixed(values) for mode, values in output.items()}
    for mode, values in output.items():
        path = CACHE / (arm + '-' + mode + '-test.jsonl')
        with path.open('w', encoding='utf8') as stream:
            for value in values:
                stream.write(json.dumps(value) + '\n')
    report.update(arm=arm, graph_sha256=digest(folder / 'vcm.fp32.onnx'), graph_bytes=(folder / 'vcm.fp32.onnx').stat().st_size,
        desktop_cpu={'threads': 2, 'session_load_ms': load_ms, 'predict_p50_ms': float(np.median(timings[5:])),
            'predict_p95_ms': float(np.quantile(timings[5:], .95)), 'excludes': 'disk, utterance duration, wake inference and endpoint wait', 'Pi_measured': False},
        references='Original programmatic silver; historical regression, not new blind data', wake_accuracy_test=False)
    # Endpoint mode does not execute a second pre-gate classifier; raw metrics there are unavailable.
    for bucket in [report['silero']['totals'], *report['silero']['per_gold_family'].values()]:
        for field in ['raw_family_correct', 'raw_command_correct']:
            bucket.pop(field, None)
    save(EXP / (arm + '-test.json'), report)
    print(json.dumps({'arm': arm, 'direct': report['direct']['totals'], 'silero': report['silero']['totals']}), flush=True)
    return report


def number_test(arm):
    recognizer = Recognizer(ROOT / 'artifacts' if arm == 'installed_baseline' else CACHE / arm, threads=2)
    source = read_rows(DATA / 'manifest.jsonl')
    rows = []
    for r in source:
        amount = r['annotation']['labels'].get('amount')
        if r['split'] == 'test' and isinstance(amount, str) and amount.isdigit():
            rows.append({**r, 'value': int(amount), 'group': r['kind'] + '_connected_commands', 'expected_family': r['command'], 'reference_value': finite_reference(r)})
    generation = json.loads((EXP / 'number-stress-generation-v2.json').read_text())
    assert generation['complete'] and digest(EXP / 'number-stress-manifest-v2.jsonl') == generation['manifest_sha256']
    for r in read_rows(EXP / 'number-stress-manifest-v2.jsonl'):
        assert digest(ROOT / r['audio']) == r['audio_sha256']
        reference = parse_parameters('VOLUME', r['transcript'], recognizer.catalog).get('command') if r['kind'] == 'volume_command' else None
        rows.append({**r, 'group': 'qwen_stress_' + r['kind'], 'reference_value': reference})
    order = json.loads((DATA / 'number-test-plan.json').read_text())['randomized_values']
    rows.sort(key=lambda r: (order.index(r['value']), r['id']))
    groups, values, predictions = defaultdict(Counter), defaultdict(lambda: defaultdict(Counter)), []
    amount_vocab = recognizer.metadata['acoustic_slots']['vocabulary']['amount']
    for row in rows:
        pcm, _ = load_audio(ROOT / row['audio'])
        family, slots, _ = raw(recognizer, pcm)
        value = amount_vocab[int(slots['amount'].argmax())]
        result = recognizer.predict(pcm, stream=False)
        emitted = result.get('status') == 'accepted' and 'command' in result
        success = row['reference_value'] is not None and emitted and result.get('command') == row['reference_value'] and result.get('vcm_command') == row['expected_family']
        counts = {'examples': 1, 'raw_amount_correct': value == str(row['value']),
            'raw_family_correct': recognizer.labels[int(family.argmax())] == row['expected_family'],
            'finite_references': row['reference_value'] is not None, 'correct_complete': success,
            'emitted': emitted, 'expected_abstention': row['expected_family'] == 'UNKNOWN',
            'correct_abstention': row['expected_family'] == 'UNKNOWN' and not emitted}
        groups[row['group']].update(counts); values[row['group']][str(row['value'])].update(counts)
        predictions.append({'id': row['id'], 'group': row['group'], 'expected_amount': row['value'], 'raw_amount': value, 'result': result})
    report = {'arm': arm, 'groups': groups, 'per_value': values, 'random_order': order,
        'bare_amount_note': 'Amount head trained on full commands; bare-number accuracy is an out-of-distribution diagnostic. Bare numbers must not emit actions.',
        'reference_note': 'Human silver labels and synthesis requested text; synthetic speech not independently human-adjudicated. No numeric test threshold fitting.'}
    save(EXP / (arm + '-numbers.json'), report)
    save(CACHE / (arm + '-number-predictions.json'), predictions)
    print(json.dumps({'arm': arm, 'number_groups': groups}), flush=True)
    return report


def evaluate():
    verify_frozen()
    selection = json.loads((EXP / 'selection-before-test.json').read_text())
    for path, sha in selection['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Changed since selection lock: ' + path)
    if (EXP / 'results.json').exists():
        raise ValueError('Results already exist; preserve this test receipt')
    fixed = {arm: fixed_test(arm) for arm in selection['test_arms']}
    numbers = {arm: number_test(arm) for arm in dict.fromkeys(selection['numeric_diagnostic_arms'])}
    custom_rows = custom_records(selection['selected_arm'], 'test')
    custom = {'arm': selection['selected_arm'], 'catalog_entries': 4, 'playlist_entries': 1,
        'enrollments_per_entry': 3, 'conditional': custom_metrics(custom_rows, selection['custom_policy'], False),
        'complete': custom_metrics(custom_rows, selection['custom_policy'], True),
        'policy': selection['custom_policy'], 'scope': 'Small original-human media proxy catalog; new household playlists not recorded or certified.'}
    save(EXP / 'custom-test.json', custom)
    save(CACHE / 'custom-test-records.json', custom_rows)
    summary = {'selected_arm': selection['selected_arm'], 'selection_sha256': digest(EXP / 'selection-before-test.json'),
        'fixed': {a: {m: r[m]['totals'] for m in ['direct', 'silero']} for a, r in fixed.items()},
        'custom': custom, 'numbers': {a: r['groups'] for a, r in numbers.items()}, 'installed': False,
        'unchanged_installed_graph': digest(ROOT / 'artifacts/vcm.fp32.onnx') == json.loads((EXP / 'protocol.json').read_text())['baseline']['vcm_sha256']}
    save(EXP / 'results.json', summary)
    print(json.dumps(summary), flush=True)
