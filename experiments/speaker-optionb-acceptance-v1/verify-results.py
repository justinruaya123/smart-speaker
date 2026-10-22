"""Independent read-only reconciliation of the OptionB holdout investigation."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.runtime.pcv_schema import INTENTS, baseline_cases, canonical
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.test.speaker_optionb_acceptance import gate_audit, scope_records
from scripts.train.pcv_updated_prepare import spoken

D = ROOT / 'experiments/speaker-optionb-acceptance-v1'
R = ROOT / '.cache/speaker-optionb-acceptance-v1/replay-01'
ARMS = ('full_catalog_fixed', 'full_catalog_optionb_output_scope', 'optionb_catalog_fixed')
VARIANTS = ('silence_prefix_1s', 'silence_suffix_1s', 'silence_both_1s')
CASES = baseline_cases()
ALLOWED = {canonical(case['command']) for case in CASES}


def direct_metrics(rows, key, *, intent=None, command=None):
    eligible = [r for r in rows if r['metric_eligible']]
    reference = lambda r: r['expected'] is not None and (intent is None or r['expected']['intent'] == intent) and (command is None or r['expected'] == command)
    predicted = lambda r: r[key] is not None and (intent is None or r[key]['intent'] == intent) and (command is None or r[key] == command)
    positives = sum(reference(r) for r in eligible)
    predictions = sum(predicted(r) for r in eligible)
    tp = sum(reference(r) and predicted(r) and r[key] == r['expected'] for r in eligible)
    fp, fn = predictions - tp, positives - tp
    return dict(n=len(eligible), positive_n=positives, accepted=predictions, tp=tp, fp=fp, fn=fn,
                precision=tp / predictions if predictions else None,
                recall=tp / positives if positives else None,
                f1=2 * tp / (positives + predictions) if positives + predictions else None,
                negative_n=sum(r['expected'] is None for r in eligible),
                negative_errors=sum(r['expected'] is None and r[key] is not None for r in eligible))


def same(actual, expected):
    if actual is None or expected is None:
        assert actual is expected, (actual, expected)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def check_metric(measured, rows, key):
    direct = direct_metrics(rows, key)
    for field, value in direct.items():
        same(measured['overall'][field], value)
    for intent in INTENTS:
        direct = direct_metrics(rows, key, intent=intent)
        actual = measured['per_intent'][intent]
        for field in ('tp', 'fp', 'fn', 'f1'):
            same(actual[field], direct[field])
        same(actual['reference_n'], direct['positive_n'])
        same(actual['predicted_n'], direct['accepted'])
        same(actual['complete_precision'], direct['precision'])
        same(actual['complete_recall'], direct['recall'])


def independently_scope(records, allowlist):
    result = []
    for row in records:
        expected = row['expected']
        category = ('ambiguous' if not row['metric_eligible'] else 'true_unknown' if expected is None
                    else 'optionb' if canonical(expected) in ALLOWED else 'outside_optionb')
        accepted = row['application_command']
        result.append({**row, 'expected': None if category == 'outside_optionb' else expected,
                       'reference_scope': category,
                       'application_command': None if allowlist and accepted is not None and canonical(accepted) not in ALLOWED else accepted})
    library = scope_records(records, output_allowlist=allowlist)
    assert len(library) == len(result)
    for left, right in zip(library, result, strict=True):
        for field in ('id', 'expected', 'application_command', 'reference_scope', 'training_eligible', 'metric_eligible', 'split'):
            assert left[field] == right[field], (left['id'], field)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    parser.add_argument('--test-log', type=Path, default=R.parent / 'verification-01/focused-tests.log')
    args = parser.parse_args()
    selection = read_json(D / 'selection-before-replay.json')
    result = read_json(D / 'results.json')
    receipt = read_json(R / 'receipt.json')
    assert len(CASES) == 93 and len(ALLOWED) == 31
    assert selection['policy'] == dict(mode='demo', intent_threshold=.95, slot_threshold=.90)
    assert file_digest(D / 'selection-before-replay.json') == result['selection_sha256'] == receipt['selection_sha256']
    for name, digest in selection['bindings'].items():
        assert file_digest(ROOT / name) == digest, name
    for field in ('training', 'threshold_search', 'installation'):
        assert result[field] is False and receipt[field] is False
    assert result['blind'] is False and result['qualified'] is False
    assert result['parent_regression_reproduced'] and receipt['parent_baseline_parity']
    full_catalog = read_json(ROOT / 'submission/models/full/catalog.json')['entries']
    selected = read_json(D / 'optionb-catalog.json')['entries']
    assert selected == [e for e in full_catalog if e['command'] is None or canonical(e['command']) in ALLOWED]
    assert [e for e in selected if e['command'] is None] == [e for e in full_catalog if e['command'] is None]
    assert {canonical(e['command']) for e in selected if e['command'] is not None} == ALLOWED
    outputs = {}
    for arm, kinds in result['raw_files'].items():
        outputs[arm] = {}
        for kind, info in kinds.items():
            path = ROOT / info['path']
            assert file_digest(path) == info['sha256'] == receipt['raw_files'][arm][kind]['sha256']
            rows = read_jsonl(path)
            assert len(rows) == info['rows'] and len({r['id'] for r in rows}) == len(rows)
            assert all(r['split'] == 'holdout' and r['training_eligible'] is False for r in rows)
            outputs[arm][kind] = rows
        original = outputs[arm]['original']
        stress = outputs[arm]['stress']
        assert len(original) == 196 and len(stress) == 282
        parent = {r['id']: r for r in original}
        eligible_human_ids = {r['id'] for r in original if r['kind'] == 'human' and r['metric_eligible']}
        assert len(eligible_human_ids) == 94
        assert Counter(r['parent_id'] for r in stress) == Counter({i: 3 for i in eligible_human_ids})
        for row in stress:
            assert row['kind'] == 'constructed_silence_stress' and row['synthetic'] is None
            assert row['audio'] is None and row['features'] is None and row['sha256'] is None and row['features_sha256'] is None
            assert row['parent_pcm_sha256'] == parent[row['parent_id']]['pcm_sha256']
            assert row['expected'] == parent[row['parent_id']]['expected']
    for cohort, filename in (('original', 'full-original.jsonl'), ('stress', 'full-silence-stress.jsonl')):
        parent_rows = read_jsonl(ROOT / '.cache/speaker-optionb-holdout-v1/replay-01' / filename)
        control = outputs['full_catalog_fixed'][cohort]
        restricted = outputs['optionb_catalog_fixed'][cohort]
        for a, b, c in zip(control, parent_rows, restricted, strict=True):
            for field in ('id', 'expected', 'candidate', 'application_command', 'application_reason', 'qualified_command'):
                assert a[field] == b[field], (a['id'], field)
            assert a['id'] == c['id'] and a['expected'] == c['expected']
            x, y = a['diagnostic_candidate'], c['diagnostic_candidate']
            assert x.get('intent_index') == y.get('intent_index')
            for field in ('top_intent_probability', 'target_named_probability'):
                if x.get(field) is None or y.get(field) is None:
                    assert x.get(field) == y.get(field)
                else:
                    assert math.isclose(x[field], y[field], rel_tol=selection['baseline_parity']['rtol'],
                                        abs_tol=selection['baseline_parity']['atol'])
    for arm in ARMS:
        base = outputs['full_catalog_fixed' if arm == 'full_catalog_optionb_output_scope' else arm]
        allowlist = arm == 'full_catalog_optionb_output_scope'
        for cohort in ('original', 'stress'):
            groups = ('human', 'synthetic') if cohort == 'original' else VARIANTS
            for group in groups:
                records = [r for r in base[cohort] if (r['kind'] == group if cohort == 'original' else r['variant'] == group)]
                rows = independently_scope(records, allowlist)
                actual = result['arms'][arm][cohort][group]
                check_metric(actual['application'], rows, 'application_command')
                check_metric(actual['raw'], rows, 'candidate')
                audit = gate_audit(records, selection['policy'])
                details = audit.pop('records')
                assert audit == actual['gate_audit_before_output_scope']
                saved = read_jsonl(R / (arm + '-' + group + '-gate-audit.jsonl'))
                assert saved == details
                assert sum(audit['disjoint_outcomes'].values()) == audit['reference_n']
                for scope in ('true_unknown', 'outside_optionb'):
                    members = [r for r in rows if r['metric_eligible'] and r['reference_scope'] == scope]
                    neg = actual['application']['negative_groups'][scope]
                    assert neg['n'] == len(members)
                    assert neg['false_accepts'] == sum(r['application_command'] is not None for r in members)
    table_counts = {}
    for filename, expected_per_group, key in (('per-intent.csv', len(INTENTS), 'intent'),
                                            ('per-command.csv', 31, 'command'), ('per-verbatim.csv', 93, 'case_id')):
        with (D / filename).open(encoding='utf-8', newline='') as stream:
            table = list(csv.DictReader(stream))
        assert len(table) == len(ARMS) * 2 * expected_per_group
        assert len({(r['arm'], r['kind'], r[key]) for r in table}) == len(table)
        table_counts[filename] = len(table)
        for arm in ARMS:
            base = outputs['full_catalog_fixed' if arm == 'full_catalog_optionb_output_scope' else arm]['original']
            for kind in ('human', 'synthetic'):
                members = [r for r in base if r['kind'] == kind]
                rows = independently_scope(members, arm == 'full_catalog_optionb_output_scope')
                subset = [r for r in table if r['arm'] == arm and r['kind'] == kind]
                assert len(subset) == expected_per_group
                keys = {r[key] for r in subset}
                assert keys == (set(INTENTS) if key == 'intent' else ALLOWED if key == 'command' else {c['id'] for c in CASES})
                for cell in subset:
                    if key != 'case_id':
                        direct = direct_metrics(rows, 'application_command', **({key: cell[key]} if key == 'intent' else {'command': json.loads(cell[key])}))
                        assert int(cell['reference_n']) == direct['positive_n']
                        for field in ('fp', 'fn'):
                            assert int(cell[field]) == direct[field]
                        assert int(cell['accepted_correct']) == direct['tp']
                        for field in ('precision', 'recall', 'f1'):
                            same(float(cell[field]) if cell[field] else None, direct[field])
                    else:
                        definition = next(c for c in CASES if c['id'] == cell['case_id'])
                        exact = [r for r in rows if r['metric_eligible'] and r['expected'] == definition['command']
                                 and spoken(r.get('text', '')) == spoken(definition['spoken_text'])]
                        assert int(cell['exact_reference_n']) == len(exact)
                        assert int(cell['raw_correct']) == sum(r['candidate'] == r['expected'] for r in exact)
                        assert int(cell['accepted_correct']) == sum(r['application_command'] == r['expected'] for r in exact)
    fields = read_json(D / 'reference-field-check.json')
    assert fields['changes_to_source_labels_or_fitting'] is False
    assert fields['driver_sha256'] == file_digest(D / 'reference-field-check.py')
    assert fields['selection_sha256'] == result['selection_sha256']
    assert fields['preserved_initial_audit_sha256'] == file_digest(D / 'fit-reference-audit.json')
    for name, values in fields['views'].items():
        for counts in values['fields'].values():
            assert sum(counts.values()) == values['rows']
            assert counts.get('present_disagreement', 0) == 0 and counts.get('invalid_present_type', 0) == 0
            if name in ('train', 'dev'):
                assert counts.get('present_consistent') == values['rows']
    test_log = args.test_log.read_text(encoding='utf-8')
    match = re.search(r'(\d+) passed', test_log)
    assert match and not re.search(r'\d+ (?:failed|error)', test_log)
    artifacts = [p for p in D.iterdir() if p.is_file() and p.name != 'verification.json']
    artifacts += [ROOT / 'scripts/test/speaker_optionb_acceptance.py', ROOT / 'scripts/test/test_speaker_optionb_acceptance.py',
                  ROOT / 'docs/EXPERIMENT-speaker-optionb-acceptance-v1.md']
    verified = dict(format='optionb-acceptance-verification-v1', pass_=True, protected_inputs=len(selection['bindings']),
                    original_recordings_per_inference_arm=196, dependent_stress_trials_per_arm=282,
                    inference_arms=len(outputs), reported_arms=len(ARMS), csv_rows=table_counts,
                    optionb_verbatims=len(CASES), optionb_commands=len(ALLOWED),
                    focused_tests=int(match.group(1)), test_log_sha256=file_digest(args.test_log),
                    training=False, threshold_search=False, deployment_changed=False,
                    archived_parent_inputs_unchanged=True, fresh_blind_holdout=False,
                    artifacts={p.relative_to(ROOT).as_posix(): file_digest(p) for p in artifacts})
    if args.write_receipt:
        write_new(D / 'verification.json', verified)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
