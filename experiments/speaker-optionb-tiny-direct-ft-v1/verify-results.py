"""Independent direct-continuation lineage, selection and replay verification."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'experiments/speaker-optionb-tiny-direct-ft-v1'


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    import torch
    import onnx
    from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new, pcm_digest
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
    import hashlib
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
    predictions = read_jsonl(run / 'final-predictions.jsonl')
    assert len(predictions) == final['prediction_rows'] == len(data['test']) + len(data['holdout']) + 3 * sum(r['kind'] == 'human' and r['metric_eligible'] for r in data['holdout'])
    assert len({r['id'] for r in predictions}) == len(predictions)
    refs = {(r['split'], r['id']): r for s in ('test', 'holdout') for r in data[s]}
    metric_groups = defaultdict(list)
    for row in predictions:
        reference = refs[row['split'], row['parent_id']]
        for key in ('class_index', 'expected', 'optionb_command', 'metric_eligible', 'scope', 'kind'):
            assert row[key] == reference[key]
        assert row['event']['actions_enabled'] is False
        if 'logits' in row['event']:
            raw, accepted = helpers.predicted(row['event']['logits'], lock['policy'])
            assert row['raw_index'] == raw and row['accepted_index'] == accepted
        else:
            assert row['raw_index'] == row['accepted_index'] == 0
        if row['metric_eligible']:
            metric_groups['/'.join((row['split'], row['kind'], row['variant']))].append(row)
    for key, records in metric_groups.items():
        for arm, field in (('raw', 'raw_index'), ('selected', 'accepted_index'), ('fixed_095', 'fixed_index')):
            assert helpers.metrics(records, field) == final['results'][key]['metrics'][arm]
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
    receipt = dict(status='verified', parameters=lock['parameters'], parent_epoch=parent['epoch'], selected_continuation_epoch=selected['epoch'],
        selected_parent_unchanged=selected['epoch'] == 0, optimizer_updates=updates, sample_draws=sum(result['sample_counts'].values()),
        human_normalization_rows_reconstructed=len(human), inference_rows=len(predictions), independently_replayed_rows=32,
        current_reserved_membership_unchanged=True, current_declared_parent_training_overlap=False,
        original_historical_inputs=freeze['original_historical_bindings'], protected_inputs_unchanged=len(freeze['historical_bindings']),
        inherited_historical_gap_recordings=freeze['inherited_historical_gap_recordings'], exposed_regression_not_blind=True,
        labels_changed=False, installed=False, verifier_sha256=file_digest(Path(__file__)))
    if args.write_receipt:
        write_new(STUDY / 'verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
