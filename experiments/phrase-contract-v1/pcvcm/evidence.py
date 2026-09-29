"""Recording plans, leakage checks, and exhaustive non-compensatory gates."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
from pathlib import Path

from .contract import Contract, canonical, write_json

SPLITS = {'train': (24, 12), 'dev': (8, 4),
          'calibration': (8, 4), 'confirmation': (20, 10)}
CONDITIONS = ('C1', 'C2', 'C3', 'C4')


def recording_plan(contract: Contract) -> list[dict]:
    """Empty recording slots, not fabricated examples or speaker identities."""
    output = []
    for split, (count, speakers) in SPLITS.items():
        for phrase in contract.positive + contract.negative:
            phrase_id = phrase.get('phrase_id', phrase.get('rejection_id'))
            for i in range(count):
                output.append({
                    'recording_id': f'{split}-{phrase_id}-{i + 1:02}',
                    'phrase_id': phrase_id, 'split': split,
                    'use_case': phrase['use_case'], 'text': phrase['exact_text'],
                    'expected_command': phrase.get('expected_command'),
                    'reference_class': phrase.get('reference_class', 'supported_positive'),
                    'speaker_slot': f'{split}-speaker-{i % speakers + 1:02}',
                    'condition_id': CONDITIONS[i // (count // 4)],
                    'status': 'UNRECORDED', 'human_reviewed': False,
                })
    for split, count in [('train', 1000), ('dev', 200),
                         ('calibration', 400), ('confirmation', 1000)]:
        for i in range(count):
            output.append({
                'recording_id': f'{split}-natural-{i + 1:04}', 'phrase_id': None,
                'split': split, 'use_case': f'UC-{i % 10 + 1:02}',
                'text': None, 'expected_command': None, 'natural_unknown': True,
                'reference_class': 'unknown_no_action',
                'speaker_slot': f'{split}-speaker-{i % SPLITS[split][1] + 1:02}',
                'condition_id': CONDITIONS[i % 4], 'status': 'UNRECORDED',
                'human_reviewed': False,
            })
    return output


def audit_partitions(rows: list[dict]) -> dict:
    mappings = {k: {} for k in ('speaker_id', 'utterance_parent_id', 'pcm_sha256')}
    seen_ids = set()
    for row in rows:
        if row['recording_id'] in seen_ids:
            raise ValueError('Repeated recording identity')
        seen_ids.add(row['recording_id'])
        if row['split'] not in SPLITS:
            raise ValueError('Unrecognized split')
        for key, mapping in mappings.items():
            value = row.get(key)
            if not value or value in ('unknown', 'pending'):
                raise ValueError(f'Missing provenance: {key}')
            if key != 'speaker_id' and value in mapping:
                raise ValueError(f'Repeated independent-event identity: {key}')
            previous = mapping.setdefault(value, row['split'])
            if previous != row['split']:
                raise ValueError(f'Cross-split {key} leakage')
    return {'rows': len(rows), 'recording_ids_unique': True,
            'cross_split_speaker_parent_pcm_overlap': 0}


def _exact(a: dict | None, b: dict | None) -> bool:
    return a is not None and canonical(a) == canonical(b)


def validate_reference(row: dict, contract: Contract) -> None:
    """Predictions never determine their own reference labels or scope."""
    phrase_id = row.get('phrase_id')
    positive = contract.by_id.get(phrase_id)
    negative = contract.negative_by_id.get(phrase_id)
    reference = positive or negative
    if reference is None:
        if (phrase_id is not None
                or row.get('natural_unknown') is not True
                or row.get('reference_class') != 'unknown_no_action'
                or row.get('expected_command') is not None):
            raise ValueError('Unrecognized recording reference')
        return
    reference_class = ('supported_positive' if positive else
                       reference['reference_class'])
    if (row.get('reference_class') != reference_class
            or row.get('use_case') != reference['use_case']
            or canonical(row.get('expected_command')) !=
            canonical(reference.get('expected_command'))):
        raise ValueError('Reference conflicts with the frozen phrase contract')


def summarize(rows: list[dict], contract: Contract) -> dict:
    seen = set()
    by_phrase = defaultdict(list)
    for row in rows:
        validate_reference(row, contract)
        if row['recording_id'] in seen:
            raise ValueError('Recording counted more than once')
        seen.add(row['recording_id'])
        by_phrase[row.get('phrase_id')].append(row)
    positives = [r for r in rows if r.get('reference_class') == 'supported_positive']
    negatives = [r for r in rows if r.get('reference_class') != 'supported_positive']
    emitted = [r for r in rows if r.get('accepted') is True]
    correct = lambda r: r.get('accepted') is True and _exact(r.get('expected_command'), r.get('command'))
    correct_count = sum(correct(r) for r in positives)
    failures, per_phrase, per_rejection, per_uc = [], [], [], {}
    for phrase in contract.positive:
        selected = by_phrase[phrase['phrase_id']]
        total = len(selected)
        wrong = sum(r.get('accepted') is True and not correct(r) for r in selected)
        conditions = {}
        for condition in CONDITIONS:
            subset = [r for r in selected if r.get('condition_id') == condition]
            conditions[condition] = {'total': len(subset), 'correct': sum(correct(r) for r in subset)}
        speakers = len({r.get('speaker_id') for r in selected if r.get('speaker_id')})
        passed = (total >= 20 and speakers >= 10 and wrong == 0
                  and sum(correct(r) for r in selected) >= math.ceil(.9 * total)
                  and all(c['total'] >= 5 and c['correct'] >= math.ceil(.8 * c['total'])
                          for c in conditions.values()))
        per_phrase.append({'phrase_id': phrase['phrase_id'], 'use_case': phrase['use_case'],
                           'recordings': total, 'speakers': speakers,
                           'exact_accepted': sum(correct(r) for r in selected),
                           'wrong_accepted': wrong, 'conditions': conditions, 'passed': passed})
    if not all(r['passed'] for r in per_phrase):
        failures.append('required_phrase_coverage_or_exact_acceptance')
    for phrase in contract.negative:
        selected = by_phrase[phrase['rejection_id']]
        speakers = len({r.get('speaker_id') for r in selected if r.get('speaker_id')})
        accepts = sum(r.get('accepted') is True for r in selected)
        passed = (len(selected) >= 20 and speakers >= 10
                  and (not phrase['critical_zero_accept'] or accepts == 0))
        per_rejection.append({'rejection_id': phrase['rejection_id'],
                              'recordings': len(selected), 'speakers': speakers,
                              'accepted': accepts, 'passed': passed})
    if not all(r['passed'] for r in per_rejection):
        failures.append('required_rejection_coverage_or_critical_acceptance')
    for uc in [f'UC-{i:02}' for i in range(1, 11)]:
        positive = [r for r in positives if r['use_case'] == uc]
        negative = [r for r in negatives if r['use_case'] == uc]
        predicted = [r for r in emitted if contract.intent_use_case.get(
            (r.get('command') or {}).get('intent')) == uc]
        accepted_correct = sum(correct(r) for r in positive)
        recall = accepted_correct / len(positive) if positive else None
        precision = sum(correct(r) for r in predicted) / len(predicted) if predicted else None
        negative_slices = {}
        for kind in ('unknown_no_action', 'unsupported_valid_request'):
            subset = [r for r in negative if r['reference_class'] == kind]
            count = sum(r.get('accepted') is True for r in subset)
            negative_slices[kind] = {'total': len(subset), 'accepted': count,
                                     'far': count / len(subset) if subset else None}
        acoustic_slices = {}
        for label, conditions, raw_gate in [('quiet', ('C1', 'C2'), .95),
                                            ('noisy', ('C3', 'C4'), .9)]:
            subset = [r for r in positive if r.get('condition_id') in conditions]
            accepted = sum(correct(r) for r in subset)
            raw = sum(_exact(r.get('expected_command'), r.get('candidate')) for r in subset)
            acoustic_slices[label] = {
                'total': len(subset), 'accepted_exact': accepted, 'raw_exact': raw,
                'passed': bool(subset) and accepted / len(subset) >= .9 and raw / len(subset) >= raw_gate}
        passed = (recall is not None and recall >= .9
                  and precision is not None and precision >= .99
                  and all(s['total'] >= 100 and s['far'] <= .01 for s in negative_slices.values())
                  and all(s['passed'] for s in acoustic_slices.values()))
        per_uc[uc] = {'positive_recordings': len(positive), 'exact_accepted': accepted_correct,
                      'predicted_emissions': len(predicted), 'recall': recall,
                      'precision': precision, 'negative_slices': negative_slices,
                      'acoustic_slices': acoustic_slices, 'passed': passed}
    if not all(r['passed'] for r in per_uc.values()):
        failures.append('per_use_case_quality')
    if sum(r.get('natural_unknown') is True for r in negatives) < 1000:
        failures.append('natural_unknown_coverage')
    if not rows or any(r.get('evidence_type') != 'human_waveform'
                       or r.get('replay_verified') is not True
                       or r.get('human_reviewed') is not True for r in rows):
        failures.append('nonhuman_or_unverified_evidence')
    if any(r.get('capture_fault') and r.get('accepted') for r in rows):
        failures.append('accepted_capture_fault')
    if any(r.get('accepted') and not r.get('endpoint_detected') for r in rows):
        failures.append('pre_endpoint_acceptance')
    return {'rows': len(rows), 'positives': len(positives), 'emissions': len(emitted),
            'exact_accepted': correct_count,
            'recall': correct_count / len(positives) if positives else None,
            'precision': correct_count / len(emitted) if emitted else None,
            'negative_false_accepts': sum(r.get('accepted') is True for r in negatives),
            'per_phrase': per_phrase, 'per_rejection': per_rejection,
            'per_use_case': per_uc, 'failed_gates': failures,
            'acoustic_gates_passed': not failures,
            'qualified': False, 'hardware_qualification': 'NOT_EVALUATED'}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or sorted({key for row in rows for key in row}) or ['recording_id']
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def write_reports(rows: list[dict], contract: Contract, output: Path) -> dict:
    Path(output).mkdir(parents=True, exist_ok=False)
    report = summarize(rows, contract)
    correct = lambda r: r.get('accepted') is True and _exact(r.get('expected_command'), r.get('command'))
    groups = {'all_commands': rows,
              'accepted_commands': [r for r in rows if correct(r)],
              'wrong_acceptances': [r for r in rows if r.get('accepted') and not correct(r)],
              'failed_commands': [r for r in rows if r.get('reference_class') == 'supported_positive' and not correct(r)],
              'negative_results': [r for r in rows if r.get('reference_class') != 'supported_positive'],
              'unsupported_valid_requests': [r for r in rows if r.get('reference_class') == 'unsupported_valid_request']}
    fields = sorted({key for row in rows for key in row}) or ['recording_id']
    for name, selected in groups.items():
        write_csv(Path(output) / f'{name}.csv', selected, fields)
    write_csv(Path(output) / 'per_phrase.csv', report['per_phrase'])
    write_csv(Path(output) / 'per_rejection.csv', report['per_rejection'])
    write_csv(Path(output) / 'per_use_case.csv', [{'use_case': k, **v} for k, v in report['per_use_case'].items()])
    write_json(Path(output) / 'summary.json', report)
    supplementary_reports(rows, Path(output))
    return report


def supplementary_reports(rows: list[dict], output: Path) -> None:
    """Disaggregated observed results; intervals do not upgrade qualification."""
    import numpy as np

    positive = [r for r in rows if r.get('reference_class') == 'supported_positive']
    correct = lambda r: r.get('accepted') is True and _exact(
        r.get('expected_command'), r.get('command'))
    for field, name in [('condition_id', 'per_condition'), ('speaker_id', 'per_speaker')]:
        records = []
        for group in sorted({str(r.get(field)) for r in positive}):
            selected = [r for r in positive if str(r.get(field)) == group]
            successes = sum(correct(r) for r in selected)
            records.append({field: group, 'positive_recordings': len(selected),
                            'accepted_exact': successes,
                            'recall': successes / len(selected)})
        write_csv(output / f'{name}.csv', records)
    confusions = []
    for row in positive:
        if correct(row):
            continue
        expected, predicted = row['expected_command'], row.get('command') or {}
        keys = sorted(set(expected) | set(predicted))
        confusions.append({'recording_id': row['recording_id'],
                           'phrase_id': row['phrase_id'],
                           'expected': expected, 'accepted_output': predicted,
                           'raw_candidate': row.get('candidate'),
                           'mismatched_fields': [k for k in keys if
                                                 canonical(expected.get(k)) !=
                                                 canonical(predicted.get(k))]})
    write_csv(output / 'parameter_confusions.csv', confusions)
    speakers = sorted({r.get('speaker_id') for r in positive if r.get('speaker_id')})
    totals = np.array([sum(r.get('speaker_id') == s for r in positive) for s in speakers])
    successes = np.array([sum(correct(r) for r in positive if r.get('speaker_id') == s)
                          for s in speakers])
    interval = None
    if len(speakers) >= 2:
        index = np.random.default_rng(53).integers(0, len(speakers), (2000, len(speakers)))
        draws = successes[index].sum(axis=1) / totals[index].sum(axis=1)
        interval = np.quantile(draws, [.025, .975]).tolist()
    write_json(output / 'uncertainty.json', {
        'speaker_clusters': len(speakers),
        'complete_recall_speaker_bootstrap_95_percentile': interval,
        'bootstrap_draws': 2000 if interval else 0,
        'seed': 53, 'gates_use_observed_rates_not_confidence_bounds': True,
        'limitations': ['Not simultaneous confidence for 570 phrases',
                        'Few speakers and nonrepresentative sampling limit inference',
                        'Synthetic results do not establish human performance']})
