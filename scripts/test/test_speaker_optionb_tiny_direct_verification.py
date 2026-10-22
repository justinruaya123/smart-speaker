"""Verifier software fixtures only; no audio, inference or accuracy evidence."""
from copy import deepcopy
import ast
import builtins
import importlib.util
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scripts.runtime.optionb_contract import COMMANDS
from scripts.runtime.pcv_schema import INTENTS, baseline_cases


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/speaker-optionb-tiny-direct-ft-v1'
POLICY = dict(enabled=True, min_probability=.5, min_margin=0.)
FIXED = dict(enabled=True, min_probability=.95, min_margin=0.)
ARMS = (('raw', 'raw_index'), ('selected', 'accepted_index'), ('fixed_095', 'fixed_index'))
STRESS = ('silence_prefix_1s', 'silence_suffix_1s', 'silence_both_1s')


def load_module(path, name):
    if not path.is_file():
        pytest.fail('Missing versioned verifier: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def verifier():
    return load_module(STUDY / 'verify-results-v2.py', 'direct_verifier_v2_tests')


@pytest.fixture(scope='module')
def helpers():
    # Use the actual independent NumPy softmax/metrics, not a decision stub.
    return load_module(ROOT / 'experiments/speaker-optionb-tiny-ft-v1/verify-results.py',
                       'frozen_metric_helpers_tests')


def reference(parent='human-holdout', *, split='holdout', kind='human', eligible=True) -> dict[str, Any]:
    command = deepcopy(COMMANDS[1]) if eligible else None
    return dict(id=parent, split=split, kind=kind, metric_eligible=eligible,
                class_index=1 if eligible else 0, expected=deepcopy(command),
                optionb_command=deepcopy(command), scope='optionb' if eligible else 'ambiguous',
                sha256='a' * 64, pcm_sha256='b' * 64)


def prediction(ref, helpers, variant='original', *, peak=4., fault=False) -> dict[str, Any]:
    logits = np.zeros(32, dtype=np.float64)
    logits[1] = peak
    raw, accepted = (0, 0) if fault else helpers.predicted(logits, POLICY)
    fixed = 0 if fault else helpers.predicted(logits, FIXED)[1]
    event = dict(actions_enabled=False, model_qualified=False,
                 output=dict(accepted=bool(accepted), reason='capture_fault' if fault else None))
    if not fault:
        event['logits'] = logits.tolist()
    return dict(id=ref['id'] if variant == 'original' else ref['id'] + '::' + variant,
                parent_id=ref['id'], variant=variant,
                **{key: deepcopy(ref[key]) for key in
                   ('split', 'kind', 'metric_eligible', 'class_index', 'expected', 'optionb_command', 'scope')},
                source_audio_sha256=ref['sha256'], source_pcm_sha256=ref['pcm_sha256'],
                raw_index=raw, accepted_index=accepted, fixed_index=fixed, event=event)


def reported(records, helpers) -> dict[str, Any]:
    groups = {}
    for record in records:
        key = '/'.join((record['split'], record['kind'], record['variant']))
        groups.setdefault(key, []).append(record)
    return dict(prediction_rows=len(records), results={
        key: dict(total_rows=len(rows), ambiguous_unscored=sum(not r['metric_eligible'] for r in rows),
                  metrics={arm: helpers.metrics([r for r in rows if r['metric_eligible']], field)
                           for arm, field in ARMS})
        for key, rows in groups.items()})


def test_forged_fixed_decision_rejected_even_with_matching_forged_metrics(helpers):
    verifier = load_module(STUDY / 'verify-results-v2.py', 'direct_verifier_v2_first_guard')
    ref = reference(split='test')
    row = prediction(ref, helpers)
    assert helpers.predicted(row['event']['logits'], POLICY) == (1, 1)
    assert helpers.predicted(row['event']['logits'], FIXED) == (1, 0)
    row['fixed_index'] = 1
    final = reported([row], helpers)  # The old verifier trusted this forged TP.
    with pytest.raises(AssertionError, match='fixed_index'):
        verifier.reconcile_predictions([row], dict(test=[ref], holdout=[]), POLICY, final, helpers)


def test_capture_fault_cannot_forge_fixed_acceptance(verifier, helpers):
    ref = reference(split='test')
    row = prediction(ref, helpers, fault=True)
    assert row['raw_index'] == row['accepted_index'] == row['fixed_index'] == 0
    row['fixed_index'] = 1
    with pytest.raises(AssertionError, match='fixed_index'):
        verifier.reconcile_predictions([row], dict(test=[ref], holdout=[]), POLICY,
                                       reported([row], helpers), helpers)


@pytest.mark.parametrize('drift', ['declared_digest', 'actual_file_bytes'])
def test_predictions_digest_must_bind_actual_file(verifier, helpers, tmp_path, drift):
    assert callable(getattr(verifier, 'verify_prediction_artifact', None)), 'missing predictions digest guard'
    ref = reference(split='test')
    records = [prediction(ref, helpers)]
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(records[0]) + '\n', encoding='utf-8')
    final = reported(records, helpers)
    final['predictions_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Whitespace drift leaves parsed rows and all declared aggregate metrics unchanged.
    if drift == 'actual_file_bytes':
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        final['predictions_sha256'] = '0' * 64
    with pytest.raises(AssertionError, match='predictions_sha256'):
        verifier.verify_prediction_artifact(path, dict(test=[ref], holdout=[]), POLICY, final, helpers)


def test_valid_predictions_digest_and_decisions_pass(verifier, helpers, tmp_path):
    assert callable(getattr(verifier, 'verify_prediction_artifact', None)), 'missing predictions digest guard'
    ref = reference(split='test')
    records = [prediction(ref, helpers)]
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(records[0]) + '\n', encoding='utf-8')
    final = reported(records, helpers)
    final['predictions_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual, groups = verifier.verify_prediction_artifact(path, dict(test=[ref], holdout=[]), POLICY,
                                                       final, helpers)
    assert actual == records
    assert set(groups) == {'test/human/original'}
    assert groups['test/human/original'][0]['fixed_index'] == 0


@pytest.fixture
def complete_case(helpers):
    data = dict(test=[reference('test-human', split='test'),
                      reference('test-ambiguous', split='test', eligible=False),
                      reference('test-synthetic', split='test', kind='synthetic')],
                holdout=[reference('holdout-human-a'), reference('holdout-human-b'),
                         reference('holdout-ambiguous', eligible=False),
                         reference('holdout-synthetic', kind='synthetic')])
    records = []
    for split in ('test', 'holdout'):
        for ref in data[split]:
            variants = ('original', *STRESS) if split == 'holdout' and ref['kind'] == 'human' and ref['metric_eligible'] else ('original',)
            records.extend(prediction(ref, helpers, variant) for variant in variants)
    return data, records, reported(records, helpers)


def test_exact_original_and_eligible_human_stress_coverage_passes(verifier, helpers, complete_case):
    data, records, final = complete_case
    groups = verifier.reconcile_predictions(records, data, POLICY, final, helpers)
    assert set(groups) == set(final['results'])
    assert sum(len(rows) for rows in groups.values()) == sum(r['metric_eligible'] for r in records)
    assert sum(not r['metric_eligible'] for r in records) == 2


@pytest.mark.parametrize('attack', [
    'substituted_original', 'stress_for_synthetic', 'stress_for_test',
    'duplicate_canonical_with_distinct_id', 'unexpected_variant', 'stress_for_ambiguous',
])
def test_exact_canonical_coverage_rejects_same_count_substitution(verifier, helpers, complete_case, attack):
    data, records, _ = complete_case
    before_count = len(records)
    if attack in ('substituted_original', 'duplicate_canonical_with_distinct_id'):
        missing = next(r for r in records if r['parent_id'] == 'holdout-human-a' and r['variant'] == 'original')
        duplicate_variant = STRESS[0] if attack == 'substituted_original' else 'original'
        duplicate = deepcopy(next(r for r in records if r['parent_id'] == 'holdout-human-b' and r['variant'] == duplicate_variant))
        duplicate['id'] = missing['id']  # Distinct IDs, valid parent, unchanged total.
        records[records.index(missing)] = duplicate
    else:
        parent = {'stress_for_synthetic': 'holdout-synthetic', 'stress_for_test': 'test-human',
                  'stress_for_ambiguous': 'holdout-ambiguous', 'unexpected_variant': 'holdout-human-a'}[attack]
        row = next(r for r in records if r['parent_id'] == parent and r['variant'] == 'original')
        row['variant'] = 'unregistered_variant' if attack == 'unexpected_variant' else STRESS[0]
        row['id'] = row['parent_id'] + '::' + row['variant']
    assert len(records) == before_count == len({r['id'] for r in records})
    with pytest.raises(AssertionError, match='canonical'):
        verifier.reconcile_predictions(records, data, POLICY, reported(records, helpers), helpers)


@pytest.mark.parametrize('variant', ['original', STRESS[0]])
def test_prediction_id_must_match_frozen_replay_format(verifier, helpers, complete_case, variant):
    data, records, final = complete_case
    row = next(r for r in records if r['parent_id'] == 'holdout-human-a' and r['variant'] == variant)
    row['id'] = row['parent_id'] + '/' + variant
    with pytest.raises(AssertionError, match='prediction id'):
        verifier.reconcile_predictions(records, data, POLICY, final, helpers)


@pytest.mark.parametrize(('field', 'forged'), [
    ('class_index', 0), ('expected', None), ('optionb_command', None),
    ('metric_eligible', False), ('scope', 'outside_optionb'), ('kind', 'synthetic'),
    ('source_audio_sha256', 'c' * 64), ('source_pcm_sha256', 'd' * 64),
])
def test_prediction_reference_fields_are_bound_to_frozen_view(verifier, helpers, complete_case, field, forged):
    data, records, final = complete_case
    records[0][field] = forged
    with pytest.raises(AssertionError, match=field):
        verifier.reconcile_predictions(records, data, POLICY, final, helpers)


@pytest.mark.parametrize('flag', ['actions_enabled', 'model_qualified'])
@pytest.mark.parametrize('fault', [False, True])
def test_every_event_remains_unqualified_and_actions_disabled(verifier, helpers, flag, fault):
    ref = reference(split='test')
    row = prediction(ref, helpers, fault=fault)
    row['event'][flag] = True
    with pytest.raises(AssertionError, match=flag):
        verifier.reconcile_predictions([row], dict(test=[ref], holdout=[]), POLICY,
                                       reported([row], helpers), helpers)


@pytest.mark.parametrize('attack', ['extra', 'missing'])
def test_result_metric_group_keyset_is_exact(verifier, helpers, complete_case, attack):
    data, records, final = complete_case
    if attack == 'extra':
        final['results']['test/human/unregistered_variant'] = deepcopy(final['results']['test/human/original'])
    else:
        del final['results']['test/human/original']
    with pytest.raises(AssertionError, match='metric group keys'):
        verifier.reconcile_predictions(records, data, POLICY, final, helpers)


def test_ambiguous_only_group_is_present_but_unscored(verifier, helpers):
    ref = reference(split='test', eligible=False)
    records = [prediction(ref, helpers)]
    groups = verifier.reconcile_predictions(records, dict(test=[ref], holdout=[]), POLICY,
                                            reported(records, helpers), helpers)
    assert groups == {'test/human/original': []}


@pytest.mark.parametrize('field', ['total_rows', 'ambiguous_unscored', 'extra_metric_arm'])
def test_reported_aggregate_counts_and_arms_are_exact(verifier, helpers, complete_case, field):
    data, records, final = complete_case
    cohort = final['results']['test/human/original']
    if field == 'extra_metric_arm':
        cohort['metrics']['unregistered_arm'] = deepcopy(cohort['metrics']['raw'])
    else:
        cohort[field] += 1
    with pytest.raises(AssertionError, match='aggregate|metric arms'):
        verifier.reconcile_predictions(records, data, POLICY, final, helpers)


@pytest.mark.parametrize('forged_output', [dict(accepted=True, reason='capture_fault'),
                                           dict(accepted=False, reason='OOV')])
def test_missing_logits_requires_an_explicit_rejected_capture_fault(verifier, helpers, forged_output):
    ref = reference(split='test')
    row = prediction(ref, helpers, fault=True)
    row['event']['output'] = forged_output
    with pytest.raises(AssertionError, match='capture_fault event'):
        verifier.reconcile_predictions([row], dict(test=[ref], holdout=[]), POLICY,
                                       reported([row], helpers), helpers)


@pytest.fixture
def csv_case(tmp_path):
    counts = dict(per_command=len(COMMANDS) - 1, per_intent=len(INTENTS), per_verbatim=len(baseline_cases()))
    for key, count in counts.items():
        (tmp_path / (key.replace('_', '-') + '.csv')).write_text('fixture\n' + 'software-only\n' * count,
                                                              encoding='utf-8')
    return tmp_path, dict(csv_rows=counts)


@pytest.mark.parametrize('name', ['per_command', 'per_intent', 'per_verbatim'])
@pytest.mark.parametrize('drift', ['file_only', 'result_only', 'both'])
def test_csv_counts_bind_actual_exports_and_canonical_table_sizes(verifier, csv_case, name, drift):
    assert callable(getattr(verifier, 'verify_csv_counts', None)), 'missing CSV row count guard'
    study, final = csv_case
    if drift in ('file_only', 'both'):
        path = study / (name.replace('_', '-') + '.csv')
        path.write_text(path.read_text(encoding='utf-8') + 'extra-software-fixture\n', encoding='utf-8')
    if drift in ('result_only', 'both'):
        final['csv_rows'][name] += 1
    with pytest.raises(AssertionError, match='csv_rows|CSV coverage'):
        verifier.verify_csv_counts(study, final, group_count=1)


def test_valid_csv_export_counts_pass(verifier, csv_case):
    assert callable(getattr(verifier, 'verify_csv_counts', None)), 'missing CSV row count guard'
    study, final = csv_case
    assert verifier.verify_csv_counts(study, final, group_count=1) == final['csv_rows']


def test_versioned_receipt_is_exclusive_and_preserves_old_negative_evidence(verifier, tmp_path):
    assert callable(getattr(verifier, 'write_versioned_receipt', None)), 'missing exclusive versioned receipt writer'
    old = {'verify-results.py': b'preserved original verifier fixture\n',
           'verification.json': b'{"status":"old-negative-evidence-fixture"}\n'}
    for name, payload in old.items():
        (tmp_path / name).write_bytes(payload)
    receipt = dict(status='software-fixture-only', acoustic_evidence=False)
    verifier.write_versioned_receipt(tmp_path, receipt)
    target = tmp_path / 'verification-v2.json'
    before = target.read_bytes()
    assert json.loads(before) == receipt
    with pytest.raises(FileExistsError):
        verifier.write_versioned_receipt(tmp_path, dict(status='must-not-overwrite'))
    assert target.read_bytes() == before
    for name, payload in old.items():
        assert (tmp_path / name).read_bytes() == payload


def test_main_help_is_available_without_acoustic_imports(verifier, monkeypatch, capsys):
    assert callable(getattr(verifier, 'main', None)), 'missing full verifier entry point'
    original_import = builtins.__import__

    def no_acoustic_imports(name, *args, **kwargs):
        assert name not in ('torch', 'onnx', 'vcm.frontend', 'scripts.runtime.speaker_optionb_tiny'), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, 'argv', ['verify-results-v2.py', '--help'])
    monkeypatch.setattr(builtins, '__import__', no_acoustic_imports)
    with pytest.raises(SystemExit) as stopped:
        verifier.main()
    assert stopped.value.code == 0
    assert '--write-receipt' in capsys.readouterr().out


def test_full_main_wires_bounded_guards_and_only_versioned_writer(verifier):
    assert callable(getattr(verifier, 'main', None)), 'missing full verifier entry point'
    tree = ast.parse(inspect.getsource(verifier.main))
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    for guard in ('verify_prediction_artifact', 'verify_csv_counts', 'write_versioned_receipt'):
        assert calls.count(guard) == 1, guard
    assert 'write_new' not in calls


@pytest.fixture
def final_artifact_case(csv_case, helpers):
    study, final = csv_case
    run = study / 'run'
    run.mkdir()
    ref = reference(split='test')
    records = [prediction(ref, helpers)]
    final.update(reported(records, helpers))
    prediction_path = run / 'final-predictions.jsonl'
    prediction_path.write_text(json.dumps(records[0]) + '\n', encoding='utf-8')
    selection_path = study / 'selection-before-test.json'
    selection_path.write_text(json.dumps(dict(policy=POLICY, selected_on=['dev', 'calibration'])), encoding='utf-8')
    final.update(predictions_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                 selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest())
    (study / 'results.json').write_text(json.dumps(final), encoding='utf-8')
    return study, run, final


@pytest.mark.parametrize('field', ['predictions_sha256', 'selection_sha256'])
@pytest.mark.parametrize('drift', ['declared_digest', 'actual_file_bytes'])
def test_final_bindings_reject_declared_or_actual_digest_drift(verifier, final_artifact_case, field, drift):
    assert callable(getattr(verifier, 'verify_final_bindings', None)), 'missing final artifact bindings guard'
    study, run, final = final_artifact_case
    if drift == 'declared_digest':
        final[field] = '0' * 64
        (study / 'results.json').write_text(json.dumps(final), encoding='utf-8')
    else:
        path = run / 'final-predictions.jsonl' if field == 'predictions_sha256' else study / 'selection-before-test.json'
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(AssertionError, match=field):
        verifier.verify_final_bindings(study, run, final)


def test_final_bindings_cover_exactly_all_six_final_artifacts(verifier, final_artifact_case):
    assert callable(getattr(verifier, 'verify_final_bindings', None)), 'missing final artifact bindings guard'
    study, run, final = final_artifact_case
    paths = [run / 'final-predictions.jsonl', study / 'results.json', study / 'selection-before-test.json',
             *[study / name for name in ('per-command.csv', 'per-intent.csv', 'per-verbatim.csv')]]
    actual = verifier.verify_final_bindings(study, run, final)
    assert actual == {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_main_rechecks_final_bindings_and_includes_them_in_receipt(verifier):
    tree = ast.parse(inspect.getsource(verifier.main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert sum(isinstance(node.func, ast.Name) and node.func.id == 'verify_final_bindings' for node in calls) == 2, 'missing pre/post final artifact bindings'
    receipt = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'receipt' for target in node.targets))
    assert isinstance(receipt, ast.Call)
    bound = next((keyword.value for keyword in receipt.keywords if keyword.arg == 'bindings'), None)
    assert isinstance(bound, ast.Name) and bound.id == 'final_bindings', 'receipt does not bind final artifacts'


# Positive controls and inherited-metric regressions accompany the new guards.
@pytest.mark.parametrize(('peak', 'fault', 'indices'), [
    (0., False, (0, 0, 0)), (4., False, (1, 1, 0)),
    (10., False, (1, 1, 1)), (4., True, (0, 0, 0)),
])
def test_valid_raw_selected_fixed_or_capture_fault_decisions_pass(verifier, helpers, peak, fault, indices):
    ref = reference(split='test')
    row = prediction(ref, helpers, peak=peak, fault=fault)
    assert (row['raw_index'], row['accepted_index'], row['fixed_index']) == indices
    groups = verifier.reconcile_predictions([row], dict(test=[ref], holdout=[]), POLICY,
                                           reported([row], helpers), helpers)
    actual = groups['test/human/original'][0]
    assert (actual['raw_index'], actual['accepted_index'], actual['fixed_index']) == indices


@pytest.mark.parametrize('arm', ['raw', 'selected', 'fixed_095'])
def test_each_reported_metric_arm_is_recomputed(verifier, helpers, complete_case, arm):
    data, records, final = complete_case
    final['results']['test/human/original']['metrics'][arm]['tp'] += 1
    with pytest.raises(AssertionError, match=arm):
        verifier.reconcile_predictions(records, data, POLICY, final, helpers)
