"""Recompute evidence without equating raw ranking, acceptance, or deployment.

The independent unit is a recording, not a re-evaluation of that recording.
Selection reads development metrics only. Release gates fail closed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def select_research(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError('No evaluated candidates')
    protocols = {c.get('protocol_id') for c in candidates}
    if len(protocols) != 1 or not next(iter(protocols)):
        raise ValueError('A shared development/catalog/frontend protocol is required')
    if len({c.get('id') for c in candidates}) != len(candidates):
        raise ValueError('Candidate IDs must be unique')
    for candidate in candidates:
        metric = candidate.get('dev_error')
        if (type(metric) not in (float, int) or not math.isfinite(metric)
                or not 0 <= metric <= 1 or not candidate.get('id')):
            raise ValueError('A finite numeric development error in [0,1] is required')
    return min(candidates, key=lambda item: (item['dev_error'], item['id']))


def normalize_row(row: dict, trial: str, dataset: str) -> dict:
    for flag in ('accepted', 'acceptance_enabled', 'action_executed'):
        if flag in row and type(row[flag]) is not bool:
            raise ValueError(f'{flag} must be a JSON boolean')
    reference = row.get('target', row.get('reference'))
    candidate = row.get('candidate', row.get('raw_command'))
    is_ambiguous = row.get('category') == 'ambiguous'
    exact = (reference is not None and reference == candidate
             and not row.get('ties', []) and not row.get('ambiguous', False))
    saved = row.get('raw_correct', row.get('raw_exact', row.get('exact')))
    accepted = bool(row.get('accepted', row.get('status') == 'recognized'))
    if row.get('acceptance_enabled') is False:
        accepted = False
    if is_ambiguous:
        outcome = ('ambiguous_wrongly_accepted' if accepted else
                   'ambiguity_preserved' if row.get('ties') or row.get('ambiguous')
                   else 'ambiguous_rejected_without_tie')
    elif reference is None:
        outcome = 'negative_wrongly_accepted' if accepted else 'negative_rejected'
    elif exact:
        outcome = 'correct_accepted' if accepted else 'correct_candidate_rejected'
    else:
        outcome = 'wrong_accepted' if accepted else 'wrong_candidate_rejected'
    return {
        'trial': trial, 'dataset': dataset, 'id': row['id'],
        'pcm_sha256': row.get('pcm_sha256'),
        'voice': row.get('voice', 'not retained'),
        'verbatim': row.get('text', ''), 'category': row.get('category', ''),
        'partition': row.get('split', 'train' if dataset.endswith('_train') else
            'development' if '_dev' in dataset else 'calibration' if 'calibration' in dataset else 'regression'),
        'expected': reference, 'raw_output': candidate,
        'raw_exact': exact, 'accepted': accepted, 'outcome': outcome,
        'ties': row.get('ties', []),
        'inconsistent_saved_flag': saved is not None and bool(saved) != exact,
        'action_executed': bool(row.get('action_executed', False)),
        'recorded_status': row.get('status', 'not retained'),
        'decision_scope': 'diagnostic only; no external actions',
        'evidence_kind': 'saved_synthetic_prediction',
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row['expected'] is not None]
    negatives = [row for row in rows if row['expected'] is None
                 and row['category'] != 'ambiguous']
    ambiguous = [row for row in rows if row['category'] == 'ambiguous']
    correct = sum(row['raw_exact'] for row in valid)
    accepted_correct = sum(row['accepted'] and row['raw_exact'] for row in valid)
    accepted = sum(row['accepted'] for row in rows)
    wrong_accepts = accepted - accepted_correct
    false_accepts = sum(row['accepted'] for row in negatives)
    return {'recordings': len(rows), 'valid': len(valid),
            'negatives': len(negatives), 'ambiguous': len(ambiguous),
            'raw_correct': correct, 'accepted_correct': accepted_correct,
            'accepted_total': accepted, 'wrong_accepts': wrong_accepts,
            'negative_false_accepts': false_accepts,
            'raw_accuracy': correct / len(valid) if valid else None,
            'accepted_recall': accepted_correct / len(valid) if valid else None,
            'accepted_precision': accepted_correct / accepted if accepted else None,
            'negative_far': false_accepts / len(negatives) if negatives else None}


def duplicate_audit(rows: list[dict]) -> dict:
    hashes = [row['pcm_sha256'] for row in rows if row.get('pcm_sha256')]
    counts = Counter(hashes)
    return {'evaluation_rows': len(rows), 'rows_with_pcm_hash': len(hashes),
            'unique_recordings_by_hash': len(counts),
            'repeated_evaluations_by_hash': sum(n - 1 for n in counts.values()),
            'missing_pcm_hash': len(rows) - len(hashes),
            'caution': 'Repeated evaluations are regressions, not independent trials.'}


def release_decision(evidence: dict) -> dict:
    requirements = {
        'human_test_passed': True, 'pi4_test_passed': True,
        'all_ten_use_cases_passed': True, 'calibration_passed': True,
        'catalog_frozen_for_evaluation': True,
        'model_frontend_hashes_verified': True,
    }
    failures = [name for name, value in requirements.items()
                if evidence.get(name) is not value]
    for metric, limit, direction in [
        ('accepted_recall', .9, 'min'),
        ('accepted_precision', .99, 'min'),
        ('negative_far', .01, 'max')]:
        value = evidence.get(metric)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not 0 <= value <= 1
                or (value < limit if direction == 'min' else value > limit)):
            failures.append(metric)
    return {'eligible': not failures, 'failed_requirements': failures,
            'release_winner': None if failures else evidence.get('id')}


def read_predictions(root: Path) -> tuple[list[dict], list[dict]]:
    rows, sources = [], []
    sources_to_read = []
    for trial, folder in [
        ('v1', 'final_suite/archive/v1'), ('v2', 'final_suite/archive/v2'),
        ('v3', 'experiments/vcm_streaming_v3/reports'),
        ('v4', 'experiments/vcm_acceptance_v4/reports'),
        ('v5', 'experiments/vcm_acceptance_v5/reports'),
        ('v6', 'experiments/vcm_services_v6/reports')]:
        sources_to_read.extend((trial, str(path.relative_to(root))) for path in
                              sorted((root / folder).glob('*_predictions.json')))
    for trial, relative in sources_to_read:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required evidence is absent: {path}')
        raw = path.read_bytes()
        payload = json.loads(raw)
        dataset = path.stem.replace('_predictions', '')
        seen = set()
        for row in payload:
            if row['id'] in seen:
                raise ValueError(f'Duplicate recording ID within {relative}')
            seen.add(row['id'])
            rows.append(normalize_row(row, trial, dataset))
        sources.append({'trial': trial, 'path': relative, 'rows': len(payload),
                        'sha256': hashlib.sha256(raw).hexdigest()})
    scope = root / 'final_suite/reports/scope_probe.json'
    if scope.exists():
        raw = scope.read_bytes()
        rows.extend(normalize_row(row, 'scope', 'all_ten_use_cases')
                    for row in json.loads(raw)['rows'])
        sources.append({'trial': 'scope', 'path': str(scope.relative_to(root)),
                        'sha256': hashlib.sha256(raw).hexdigest(),
                        'rows': len(json.loads(raw)['rows'])})
    return rows, sources


def build(root: Path = ROOT) -> dict:
    out = root / 'final_suite/data'
    out.mkdir(parents=True, exist_ok=True)
    rows, sources = read_predictions(root)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['trial']}:{row['dataset']}"].append(row)
    summaries = {name: summarize(value) for name, value in groups.items()}
    training = json.loads((root / 'experiments/vcm_services_v6/reports/training.json').read_text())
    selection = json.loads((root / 'experiments/vcm_services_v6/reports/selection.json').read_text())
    missing = []
    for relative in [
        'experiments/vcm_streaming_v3/data/phone_manifest.jsonl',
        'experiments/vcm_services_v6/data/probe_manifest.jsonl']:
        manifest = [json.loads(line) for line in (root / relative).read_text().splitlines() if line]
        fields = Counter(key for item in manifest for key in item)
        path_field = 'wav' if 'wav' in fields else 'audio'
        base = (root / relative).parent.parent
        missing_count = sum(not (base / item.get(path_field, '__missing__')).is_file()
                            for item in manifest)
        missing.append({'path': relative, 'rows': len(manifest),
                        'missing_audio_at_manifest_paths': missing_count,
                        'split_counts': dict(Counter(item.get('split') for item in manifest))})
    artifact = {
        'schema_version': 1, 'sources': sources, 'rows': rows,
        'summaries': summaries, 'duplicates': duplicate_audit(rows),
        'saved_flag_disagreements': [row for row in rows if row['inconsistent_saved_flag']],
        'manifest_audio_audit': missing,
        'research_choice': {'id': 'phone_s53', 'selection': 'V6 epoch-zero parent',
            'dev_error': selection['development']['phone_error_rate'],
            'parameters': 113064, 'onnx_bytes': 458230, 'cache_bytes': 32064,
            'warning': 'Best retained research candidate, not a qualified ten-use-case winner.'},
        'release': release_decision({
            'id': 'phone_s53', 'model_frontend_hashes_verified': True,
            'catalog_frozen_for_evaluation': True, 'human_test_passed': False,
            'pi4_test_passed': False, 'all_ten_use_cases_passed': False,
            'calibration_passed': False, 'accepted_recall': 0.0,
            'accepted_precision': None, 'negative_far': 0.0}),
        'limits': ['Synthetic-only acoustic evidence.',
                   'Most historical WAVs are absent from these archives.',
                   'Inspected tests are regression sets, not sealed selection sets.',
                   'Final ten-use-case human/Pi qualification is incomplete.'],
    }
    compact = json.dumps(artifact, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    (out / 'evidence.json.gz').write_bytes(gzip.compress(compact, mtime=0))
    summary = {key: value for key, value in artifact.items() if key not in {'rows', 'saved_flag_disagreements'}}
    summary['saved_flag_disagreements_count'] = len(artifact['saved_flag_disagreements'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    for name, selected in [('all_commands', rows),
        ('incorrect_raw_commands', [r for r in rows if r['expected'] is not None and not r['raw_exact']]),
        ('wrong_acceptances', [r for r in rows if r['accepted'] and not r['raw_exact']]),
        ('raw_successes', [r for r in rows if r['raw_exact']])]:
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = list(rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))
