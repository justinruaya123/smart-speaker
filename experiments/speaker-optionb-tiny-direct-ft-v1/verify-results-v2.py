"""Versioned direct-study verification; preserve the original source and receipt.

The public reconciliation seam consumes metadata/logits only. The full verifier
entry point is separate from these bounded software checks.
"""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'experiments/speaker-optionb-tiny-direct-ft-v1'


def verify_prediction_artifact(path, data, policy, final, helpers):
    """Hash and parse the same byte image, rather than trusting the declared SHA."""
    payload = Path(path).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == final['predictions_sha256'], 'predictions_sha256'
    predictions = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    groups = reconcile_predictions(predictions, data, policy, final, helpers)
    return predictions, groups


def verify_csv_counts(study, final, *, group_count):
    """Reconcile export row counts, not the semantic contents of CSV cells."""
    from scripts.runtime.optionb_contract import COMMANDS
    from scripts.runtime.pcv_schema import INTENTS, baseline_cases

    counts = {}
    for name in ('per_command', 'per_intent', 'per_verbatim'):
        path = Path(study) / (name.replace('_', '-') + '.csv')
        with path.open(newline='', encoding='utf-8') as stream:
            counts[name] = sum(1 for _ in csv.DictReader(stream))
    assert counts == final['csv_rows'], 'csv_rows'
    expected = dict(per_command=group_count * (len(COMMANDS) - 1),
                    per_intent=group_count * len(INTENTS), per_verbatim=len(baseline_cases()))
    assert counts == expected, 'canonical CSV coverage'
    return counts


def write_versioned_receipt(study, receipt):
    """Never overwrite an earlier receipt, including the version being replaced."""
    with (Path(study) / 'verification-v2.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_final_bindings(study, run, final):
    """Bind the report, selected lock, predictions and all three final CSV exports."""
    study, run = Path(study), Path(run)
    prediction_path = run / 'final-predictions.jsonl'
    selection_path = study / 'selection-before-test.json'
    result_path = study / 'results.json'
    paths = (prediction_path, selection_path, result_path,
             *[study / name for name in ('per-command.csv', 'per-intent.csv', 'per-verbatim.csv')])
    images = {path: path.read_bytes() for path in paths}
    bindings = {path.as_posix(): hashlib.sha256(payload).hexdigest() for path, payload in images.items()}
    assert json.loads(images[result_path].decode('utf-8')) == final, 'results.json byte image'
    assert bindings[prediction_path.as_posix()] == final['predictions_sha256'], 'predictions_sha256'
    assert bindings[selection_path.as_posix()] == final['selection_sha256'], 'selection_sha256'
    return bindings


def reconcile_predictions(predictions, data, policy, final, helpers):
    """Independently reconstruct stored decisions and reconcile reported metrics."""
    expected = {}
    stress = ('silence_prefix_1s', 'silence_suffix_1s', 'silence_both_1s')
    for split in ('test', 'holdout'):
        for reference in data[split]:
            assert reference['split'] == split, 'frozen view split'
            variants = ('original', *stress) if split == 'holdout' and reference['kind'] == 'human' and reference['metric_eligible'] else ('original',)
            for variant in variants:
                key = (split, reference['id'], variant)
                assert key not in expected, 'duplicate frozen canonical key'
                expected[key] = reference
    canonical = [(r['split'], r['parent_id'], r['variant']) for r in predictions]
    assert len(canonical) == len(set(canonical)), 'duplicate prediction canonical key'
    assert set(canonical) == set(expected), 'canonical prediction coverage'
    assert len(predictions) == final['prediction_rows'] == len(expected), 'prediction_rows'
    assert len({r['id'] for r in predictions}) == len(predictions), 'duplicate prediction ID'
    groups = {'/'.join((split, ref['kind'], variant)): []
              for (split, _, variant), ref in expected.items()}
    totals = Counter('/'.join((split, ref['kind'], variant))
                     for (split, _, variant), ref in expected.items())
    assert set(final['results']) == set(groups), 'metric group keys'
    fixed_policy = dict(enabled=True, min_probability=.95, min_margin=0.)
    for row in predictions:
        reference = expected[row['split'], row['parent_id'], row['variant']]
        # Preserve the existing replay writer's IDs; canonical membership is separate.
        expected_id = reference['id'] if row['variant'] == 'original' else reference['id'] + '::' + row['variant']
        assert row['id'] == expected_id, 'prediction id'
        for field in ('class_index', 'expected', 'optionb_command', 'metric_eligible', 'scope', 'kind'):
            assert row[field] == reference[field], (row['id'], field)
        for field, source in (('source_audio_sha256', 'sha256'), ('source_pcm_sha256', 'pcm_sha256')):
            assert row[field] == reference[source], (row['id'], field)
        assert row['event']['actions_enabled'] is False, 'actions_enabled'
        assert row['event']['model_qualified'] is False, 'model_qualified'
        if 'logits' in row['event']:
            raw, accepted = helpers.predicted(row['event']['logits'], policy)
            fixed = helpers.predicted(row['event']['logits'], fixed_policy)[1]
            assert row['raw_index'] == raw, 'raw_index'
            assert row['accepted_index'] == accepted, 'accepted_index'
            assert row['fixed_index'] == fixed, 'fixed_index'
        else:
            raw = accepted = fixed = 0
            output = row['event']['output']
            assert output['accepted'] is False and output['reason'] == 'capture_fault', 'capture_fault event'
            assert row['raw_index'] == row['accepted_index'] == 0, 'capture_fault indices'
            assert row['fixed_index'] == 0, 'capture_fault fixed_index'
        if row['metric_eligible']:
            reconstructed = dict(row, raw_index=raw, accepted_index=accepted, fixed_index=fixed)
            groups['/'.join((row['split'], row['kind'], row['variant']))].append(reconstructed)
    for key, records in groups.items():
        cohort = final['results'][key]
        assert cohort['total_rows'] == totals[key], (key, 'aggregate total_rows')
        assert cohort['ambiguous_unscored'] == totals[key] - len(records), (key, 'aggregate ambiguous_unscored')
        assert set(cohort['metrics']) == {'raw', 'selected', 'fixed_095'}, (key, 'metric arms')
        for arm, field in (('raw', 'raw_index'), ('selected', 'accepted_index'), ('fixed_095', 'fixed_index')):
            assert helpers.metrics(records, field) == cohort['metrics'][arm], (key, arm)
    return dict(groups)


def main():
    """Retain v1's lineage/normalization/replay checks with versioned reconciliation."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    import torch
    import onnx
    from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, pcm_digest
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.runtime.optionb_contract import COMMANDS
    from vcm.frontend import features
    from threadpoolctl import threadpool_limits

    prior = ROOT / 'experiments/speaker-optionb-tiny-ft-v1'
    verifier = prior / 'verify-results.py'
    assert file_digest(verifier) == read_json(prior / 'verification.json')['verifier_sha256']
    spec = importlib.util.spec_from_file_location('frozen_metric_verifier', verifier)
    if spec is None or spec.loader is None:
        raise ImportError('Frozen verifier helper unavailable')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    protocol, freeze = read_json(STUDY / 'protocol.json'), read_json(STUDY / 'data-freeze.json')
    run = ROOT / protocol['outputs']['run']
    fit = read_json(run / 'fit-verification.json')
    lock = read_json(STUDY / 'selection-before-test.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings'], fit['bindings'], lock['bindings']):
        helpers.check(bindings)
    data = {s: read_jsonl(run / f'{s}.jsonl') for s in ('train', 'dev', 'calibration', 'test', 'holdout')}
    source = ROOT / protocol['data']['source_run']
    for split in ('dev', 'calibration', 'test', 'holdout'):
        assert file_digest(run / f'{split}.jsonl') == file_digest(source / f'{split}.jsonl')
    original = {r['id']: r for r in read_jsonl(ROOT / protocol['parent']['original_training_membership'])}
    admitted = {r['id']: r for r in read_jsonl(source / 'train.jsonl')}
    train = data['train']
    assert all(r == admitted[r['id']] == original[r['id']] for r in train)
    human = [r for r in train if r['kind'] == 'human']
    assert len(human) == 1889 and {r['class_index'] for r in train if r['class_index']} == set(range(1, 32))
    from collections import defaultdict
    groups = defaultdict(list)
    for row in admitted.values():
        if row['kind'] == 'synthetic':
            groups[row['class_index'], row['speaker']].append(row)
    expected_synthetic = {r['id'] for members in groups.values() for r in sorted(members,
        key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:2]}
    assert {r['id'] for r in train if r['kind'] == 'synthetic'} == expected_synthetic
    reserved = [r for s in ('dev', 'calibration', 'test', 'holdout') for r in data[s]]
    for field in ('id', 'speaker', 'sha256', 'pcm_sha256', 'pair', 'source_speaker_key', 'session_id', 'parent_recording_id', 'speaker_alias_group'):
        left = {json.dumps(r[field], sort_keys=True) for r in original.values() if r.get(field) not in (None, '')}
        right = {json.dumps(r[field], sort_keys=True) for r in reserved if r.get(field) not in (None, '')}
        assert not left & right, field
    parent = torch.load(ROOT / protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    assert parent['epoch'] == 44 and set(parent['sample_counts']) <= set(original)
    assert freeze['parent_lineage']['human_draws'] == sum(n for key, n in parent['sample_counts'].items() if original[key]['kind'] == 'human')
    assert freeze['parent_lineage']['synthetic_draws'] == sum(n for key, n in parent['sample_counts'].items() if original[key]['kind'] == 'synthetic')
    result = read_json(run / 'training-result.json')
    curve = read_jsonl(run / 'training-curve.jsonl')
    assert [r['epoch'] for r in curve] == list(range(protocol['training']['epochs'] + 1))
    baseline = curve[0]
    for event in curve:
        competent = (event['human_dev']['raw']['recall'] >= baseline['human_dev']['raw']['recall']
            and event['synthetic_dev_raw']['recall'] >= baseline['synthetic_dev_raw']['recall'] - .01)
        assert int(competent) == event['rank'][0]
    chosen = max(curve, key=lambda r: tuple(r['rank']))
    assert chosen['epoch'] == result['selected_epoch'] == lock['selected_epoch'] and chosen['competence_pass']
    assert chosen['rank'] == result['selection_rank']
    updates = protocol['training']['epochs'] * protocol['training']['steps_per_epoch']
    assert result['optimizer_steps'] == fit['total_optimizer_steps'] == updates
    assert sum(result['sample_counts'].values()) == updates * 32
    assert set(result['sample_counts']) <= {r['id'] for r in train}
    selected = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    assert sum(selected['sample_counts'].values()) == selected['epoch'] * protocol['training']['steps_per_epoch'] * 32
    if selected['epoch'] == 0:
        assert all(torch.equal(v, parent['state_dict'][key]) for key, v in selected['state_dict'].items())
    last = torch.load(run / 'arms/direct/last.pt', map_location='cpu', weights_only=True)
    assert any(not torch.equal(v, parent['state_dict'][key]) for key, v in last['state_dict'].items())
    inputs = read_jsonl(run / 'raw-feature-inputs.jsonl')
    assert {r['id'] for r in inputs} == {r['id'] for r in [*train, *data['dev']]}
    assert {r['split'] for r in inputs} == {'train', 'dev'}
    norm = read_json(run / 'normalization.json')
    assert set(norm['ids']) == {r['id'] for r in human}
    first, second = [], []
    with threadpool_limits(limits=1, user_api='blas'):
        for row in human:
            values = features(read_audio(row))
            assert hashlib.sha256(f'f32le:{values.shape}\0'.encode() + np.asarray(values, dtype='<f4').tobytes()).hexdigest() == norm['raw_feature_tensor_sha256'][row['id']]
            values = values.astype(np.float64)
            first.append(values.mean(axis=1))
            second.append(np.square(values).mean(axis=1))
    mean = np.mean(first, axis=0)
    std = np.sqrt(np.maximum(np.mean(second, axis=0) - mean ** 2, 1e-8))
    np.testing.assert_array_equal(mean.astype(np.float32), np.asarray(norm['mean'], dtype=np.float32))
    np.testing.assert_array_equal(std.astype(np.float32), np.asarray(norm['std'], dtype=np.float32))
    hard = read_json(run / 'hard-negative-pool.json')
    assert set(hard['ids']) <= {r['id'] for r in human if r['class_index'] == 0 and r.get('transcript_source') != 'whisper'}
    final = read_json(STUDY / 'results.json')
    final_bindings = verify_final_bindings(STUDY, run, final)
    predictions, metric_groups = verify_prediction_artifact(run / 'final-predictions.jsonl', data,
                                                            lock['policy'], final, helpers)
    csv_counts = verify_csv_counts(STUDY, final, group_count=len(metric_groups))
    refs = {(r['split'], r['id']): r for s in ('test', 'holdout') for r in data[s]}
    runtime = Runtime(run / 'bundle')
    for index in np.random.default_rng(103).choice(len(predictions), 32, replace=False):
        row = predictions[int(index)]
        pcm = read_audio(refs[row['split'], row['parent_id']])
        zeros = np.zeros(16000, dtype=np.float32)
        if row['variant'] in ('silence_prefix_1s', 'silence_both_1s'):
            pcm = np.concatenate([zeros, pcm])
        if row['variant'] in ('silence_suffix_1s', 'silence_both_1s'):
            pcm = np.concatenate([pcm, zeros])
        assert pcm_digest(pcm) == row['actual_input_pcm_sha256']
        event = runtime.recognize(pcm)
        assert event.get('command_index', 0) == row['raw_index']
        assert (event.get('command_index', 0) if event['output']['accepted'] else 0) == row['accepted_index']
    graph = onnx.load(str(run / 'bundle/model.onnx'))
    assert sum(int(np.prod(t.dims)) for t in graph.graph.initializer) == lock['parameters'] == 258912
    assert read_json(run / 'bundle/commands.json') == list(COMMANDS)
    for bindings in (freeze['bindings'], freeze['historical_bindings'], fit['bindings'], lock['bindings']):
        helpers.check(bindings)
    assert verify_final_bindings(STUDY, run, final) == final_bindings, 'final artifacts changed during verification'
    receipt = dict(status='verified', parameters=lock['parameters'], parent_epoch=parent['epoch'], selected_continuation_epoch=selected['epoch'],
        selected_parent_unchanged=selected['epoch'] == 0, optimizer_updates=updates, sample_draws=sum(result['sample_counts'].values()),
        human_normalization_rows_reconstructed=len(human), inference_rows=len(predictions), independently_replayed_rows=32,
        current_reserved_membership_unchanged=True, current_declared_parent_training_overlap=False,
        original_historical_inputs=freeze['original_historical_bindings'], protected_inputs_unchanged=len(freeze['historical_bindings']),
        inherited_historical_gap_recordings=freeze['inherited_historical_gap_recordings'], exposed_regression_not_blind=True,
        labels_changed=False, installed=False, verifier_sha256=file_digest(Path(__file__)),
        verification_version=2, exact_canonical_prediction_coverage=True, csv_rows=csv_counts,
        predictions_sha256=final['predictions_sha256'], selection_sha256=final['selection_sha256'], bindings=final_bindings,
        preserved_original_verifier_sha256=file_digest(STUDY / 'verify-results.py'),
        preserved_original_receipt_sha256=file_digest(STUDY / 'verification.json'))
    if args.write_receipt:
        write_versioned_receipt(STUDY, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
