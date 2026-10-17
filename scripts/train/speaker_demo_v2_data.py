"""Freeze the existing shared VCM's human-first comparison inputs.

Preparation never fits a model or extracts calibration/test features. Missing
human coverage yields a durable readiness report and exit 2, before extraction.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import sys

from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command
from scripts.test.speaker_demo_suite import (
    file_digest, read_audio, read_json, read_jsonl, write_new,
)

ROOT = Path(__file__).resolve().parents[2]
HUMAN_VIEWS = ('human/train', 'human/dev', 'human/calibration')
FIT_VIEWS = (*HUMAN_VIEWS, 'synthetic/train')
EVALUATION_VIEWS = ('evaluation/test', 'evaluation/holdout')


class PreparationBlocked(ValueError):
    """Valid inputs need additional human recordings or catalog review."""

    def __init__(self, report):
        self.report = report
        super().__init__('Preparation blocked; see readiness.json for human coverage and catalog gaps')


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _bound(path, sha256):
    path = _path(path)
    if file_digest(path) != sha256:
        raise ValueError(f'Input identity mismatch: {path}')
    return path


def _receipt(path, **extra):
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path), **extra}


def _copy_new(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(source.read_bytes())
    return _receipt(destination)


def _load_inputs(protocol, installed):
    config = read_json(protocol)
    required = config['required_intents']
    if len(required) != len(INTENTS) or set(required) != set(INTENTS):
        raise ValueError('The comparison must preserve all 18 shared-model intents')
    audit_path = _bound(Path(installed) / 'audit.json', config['installation_audit_sha256'])
    audit = read_json(audit_path)
    bindings = {str(Path(protocol).resolve()): file_digest(protocol),
                str(audit_path.resolve()): file_digest(audit_path)}
    for name, expected in {**audit['source_receipts'], **config['bindings']}.items():
        path = _bound(name, expected)
        bindings[str(path.resolve())] = expected
    inventory = _bound(audit['inventory'], audit['inventory_sha256'])
    bindings[str(inventory.resolve())] = audit['inventory_sha256']
    paths, views = {}, {}
    for name, entry in audit['views'].items():
        paths[name] = _bound(entry['path'], entry['sha256'])
        bindings[str(paths[name].resolve())] = entry['sha256']
        if name in (*FIT_VIEWS, *EVALUATION_VIEWS):
            views[name] = read_jsonl(paths[name])
            if len(views[name]) != entry['rows']:
                raise ValueError(f'Manifest row count mismatch: {name}')
    if not set((*FIT_VIEWS, *EVALUATION_VIEWS)) <= views.keys():
        raise ValueError('Missing required fitting/evaluation views')
    catalog_path = _bound(**config['catalog'])
    definitions_path = _bound(**config['definitions'])
    catalog, definitions = read_json(catalog_path), read_json(definitions_path)
    commands = {}
    for entry in catalog['entries']:
        if entry['command'] is not None:
            validate_command(entry['command'])
            commands[canonical(entry['command'])] = entry['command']
    if not commands:
        raise ValueError('Empty complete-command catalog')
    for case in definitions:
        if case['command'] is not None:
            validate_command(case['command'])
            if canonical(case['command']) not in commands:
                raise ValueError('Catalog omits an approved benchmark/superset command')
    return config, audit, paths, views, commands, definitions, catalog_path, definitions_path, bindings


def _check_rows(views):
    ids = set()
    identities = {key: {} for key in ('sha256', 'pcm_sha256', 'speaker_id', 'pair_id')}
    for name, rows in views.items():
        expected_split = name.split('/')[1]
        for row in rows:
            if row['split'] != expected_split or row['kind'] not in {'human', 'synthetic'}:
                raise ValueError(f'Incorrect split or kind in {name}: {row["id"]}')
            if row['id'] in ids:
                raise ValueError('Duplicate recording ID')
            ids.add(row['id'])
            for key, seen in identities.items():
                value = row.get(key)
                if not isinstance(value, str) or not value or value.endswith('/'):
                    # Evaluation membership remains intact, including deficient metadata;
                    # unidentified fitting rows may not enter the comparison.
                    if name in FIT_VIEWS:
                        raise ValueError(f'Missing {key} in fitting input')
                    continue
                if value in seen and (key in {'sha256', 'pcm_sha256'} or seen[value] != expected_split):
                    raise ValueError(f'Cross-split overlap or duplicate {key}: {row["id"]}')
                seen[value] = expected_split
            if name not in FIT_VIEWS:
                continue
            if name.startswith('human/') and row['kind'] != 'human':
                raise ValueError('Synthetic recording in human view')
            if name == 'synthetic/train' and row['kind'] != 'synthetic':
                raise ValueError('Non-synthetic recording in synthetic view')
            status, command = row['reference_status'], row['command']
            if (row['role'] != 'command' or row['audio_flags'] or row['qa_eligible'] is not True
                    or row['metric_eligible'] is not True or status not in {'supported', 'unsupported'}):
                raise ValueError(f'Ineligible fitting input: {row["id"]}')
            if (command is None) != (status == 'unsupported'):
                raise ValueError('Reference status must agree with complete command or true UNKNOWN')
            if command is not None:
                validate_command(command)


def _coverage(views, commands, definitions):
    baseline = {canonical(c['command']): c['command'] for c in definitions
                if c.get('tier') == 'baseline' and c['command'] is not None}
    missing_intents, missing_unknown, outside, coverage, collection = {}, [], {}, {}, []
    for name in FIT_VIEWS:
        rows = views[name]
        counts = Counter(r['command']['intent'] if r['command'] else 'UNKNOWN' for r in rows)
        present = {canonical(r['command']) for r in rows if r['command'] is not None}
        people = defaultdict(set)
        for row in rows:
            people[row['command']['intent'] if row['command'] else 'UNKNOWN'].add(row['speaker_id'])
        coverage[name] = {
            'rows': len(rows), 'intents': dict(sorted(counts.items())),
            'declared_speakers': len({r['speaker_id'] for r in rows}),
            'declared_speakers_per_intent': {k: len(v) for k, v in sorted(people.items())},
            'baseline_commands_present': len(present & baseline.keys()),
            'baseline_commands_required': len(baseline),
            'missing_baseline_commands': [baseline[k] for k in sorted(baseline.keys() - present)],
        }
        outside[name] = [json.loads(k) for k in sorted(present - commands.keys())]
        if name not in HUMAN_VIEWS:
            continue
        missing_intents[name] = sorted(set(INTENTS) - counts.keys())
        if not counts['UNKNOWN']:
            missing_unknown.append(name)
        for intent in [*missing_intents[name], *(['UNKNOWN'] if not counts['UNKNOWN'] else [])]:
            examples = [dict(text=c['text'], command=c['command']) for c in definitions
                        if (c['command']['intent'] if c['command'] else 'UNKNOWN') == intent]
            collection.append({'view': name, 'intent': intent, 'examples': examples,
                               'requirement': 'Original human recordings; independently review complete typed labels; use people disjoint from every other partition'})
    denominators = {}
    for name in EVALUATION_VIEWS:
        rows = views[name]
        denominators[name] = {
            'rows': len(rows), 'kinds': dict(Counter(r['kind'] for r in rows)),
            'reference_status': dict(Counter(r['reference_status'] for r in rows)),
            'metric_eligible': sum(r['metric_eligible'] is True for r in rows),
        }
    blocked = any(missing_intents.values()) or bool(missing_unknown) or any(outside.values())
    return {
        'status': 'blocked_missing_data_or_catalog' if blocked else 'metadata_ready_audio_unverified',
        'ready_for_fitting': False, 'metadata_ready': not blocked, 'model_qualified': False,
        'missing_human_intents': missing_intents, 'missing_human_unknown': missing_unknown,
        'outside_catalog': outside, 'coverage': coverage, 'evaluation_denominators': denominators,
        'recording_requests': collection,
        'collection_scope': 'These are missing intent cells, not a statistically sufficient sample-size prescription; typed-value and person coverage require further review',
        'unfamiliar_speaker_claim_qualified': False,
        'limitations': ['Declared identities only; synthetic-reference aliases are unresolved',
                       'Near duplicates and independent listening are not established by metadata',
                       'Existing test exposure is historical; no new blind holdout or wake-owner/Pi evidence'],
    }


def prepare_run(protocol: Path, installed: Path, output: Path) -> dict:
    """Bind an installation, audit metadata, then freeze fitting inputs.

    ``installed`` is the installer's output directory containing audit.json.
    Relative paths in bound artifacts resolve against the repository root.
    Only the successful preparation.json is a consumable preparation receipt.
    """
    protocol, installed, output = Path(protocol), Path(installed), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new preparation directory; prior evidence is immutable')
    (config, audit, paths, views, commands, definitions,
     catalog_path, definitions_path, bindings) = _load_inputs(protocol, installed)
    _check_rows(views)
    report = _coverage(views, commands, definitions)
    report.update(experiment_id=config['experiment_id'],
                  source_bindings=bindings, catalog=_receipt(catalog_path),
                  definitions=_receipt(definitions_path))
    output.mkdir(parents=True, exist_ok=False)
    if not report['metadata_ready']:
        write_new(output / 'readiness.json', report)
        raise PreparationBlocked(report)

    import numpy as np
    from vcm.frontend import SPEC, features

    manifests = {}
    for name in (*FIT_VIEWS, *EVALUATION_VIEWS):
        manifests[name] = _copy_new(paths[name], output / 'manifests' / (name + '.jsonl'))
        manifests[name]['rows'] = len(views[name])
    catalog = _copy_new(catalog_path, output / 'catalog.json')
    approved_cases = _copy_new(definitions_path, output / 'definitions.json')
    feature_root = output / 'features'
    feature_root.mkdir()
    frontend_digest = file_digest(ROOT / 'vcm/frontend.py')
    sums, squares, normalization_ids = [], [], []
    feature_receipts = []
    for name in ('human/train', 'human/dev', 'synthetic/train'):
        for row in views[name]:
            values = features(read_audio(row, ROOT))
            if (values.ndim != 2 or values.shape[0] != SPEC['bins']
                    or not values.shape[1] or not np.isfinite(values).all()):
                raise ValueError(f'Invalid frontend features: {row["id"]}')
            path = feature_root / (row['pcm_sha256'] + '.npy')
            with path.open('xb') as stream:
                np.save(stream, values, allow_pickle=False)
            feature_receipts.append({**_receipt(path), 'id': row['id'], 'kind': row['kind'],
                                     'split': row['split'], 'shape': list(values.shape),
                                     'pcm_sha256': row['pcm_sha256'], 'frontend_sha256': frontend_digest})
            if name == 'human/train':
                precise = values.astype('float64')
                sums.append(precise.mean(axis=1))
                squares.append((precise ** 2).mean(axis=1))
                normalization_ids.append(row['id'])
    mean = np.mean(sums, axis=0)
    std = np.sqrt(np.maximum(np.mean(squares, axis=0) - mean ** 2, 1e-6))
    normalization_path = output / 'normalization.json'
    write_new(normalization_path, {'mean': mean.tolist(), 'std': std.tolist(), 'frontend': SPEC,
                                   'source_ids': normalization_ids,
                                   'human_training_utterances': len(normalization_ids),
                                   'weighting': 'equal weight per eligible human training utterance'})
    feature_path = output / 'features.jsonl'
    write_new(feature_path, feature_receipts, jsonl=True)
    implementation = {str(path.relative_to(ROOT)): file_digest(path) for path in (
        Path(__file__), ROOT / 'vcm/frontend.py', ROOT / 'scripts/runtime/pcv_schema.py',
        ROOT / 'scripts/test/speaker_demo_suite.py')}
    report.update(status='prepared_unqualified', ready_for_fitting=True,
                  manifests=manifests, catalog=catalog, definitions=approved_cases,
                  normalization=_receipt(normalization_path),
                  feature_receipts=_receipt(feature_path, rows=len(feature_receipts)),
                  implementation_bindings=implementation,
                  environment={'python': platform.python_version(), 'numpy': np.__version__},
                  features_extracted_from=['human/train', 'human/dev', 'synthetic/train'],
                  calibration_test_holdout_audio_read=False)
    write_new(output / 'readiness.json', report)
    write_new(output / 'preparation.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'experiments/speaker-demo-v2/protocol.json')
    parser.add_argument('--installed', type=Path, default=ROOT / '.cache/speaker-dataset-v1/verified')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_run(args.protocol, args.installed, args.output)
    except PreparationBlocked as exc:
        print(json.dumps({'status': exc.report['status'], 'readiness': str(args.output / 'readiness.json'),
                          'missing_human_intents': exc.report['missing_human_intents'],
                          'missing_human_unknown': exc.report['missing_human_unknown'],
                          'outside_catalog_counts': {k: len(v) for k, v in exc.report['outside_catalog'].items()}}, indent=2))
        return 2
    print(json.dumps({'status': result['status'], 'receipt': str(args.output / 'preparation.json')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
