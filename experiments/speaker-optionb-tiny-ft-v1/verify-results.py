"""Independent artifact, membership and metric reconciliation; no fitting."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'experiments/speaker-optionb-tiny-ft-v1'
PARENT = ROOT / '.cache/speaker-optionb-tiny-v1/run-01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(bindings):
    for path, expected in bindings.items():
        assert digest(ROOT / path) == expected, path
    return len(bindings)


def metrics(records, field):
    tp = sum(r['class_index'] > 0 and r[field] == r['class_index'] for r in records)
    fp = sum(r[field] > 0 and r[field] != r['class_index'] for r in records)
    fn = sum(r['class_index'] > 0 and r[field] != r['class_index'] for r in records)
    negatives = sum(r['class_index'] == 0 for r in records)
    false_accepts = sum(r['class_index'] == 0 and r[field] > 0 for r in records)
    return dict(rows=len(records), positives=tp + fn, negatives=negatives,
        tp=tp, fp=fp, fn=fn, correct_accepted=tp, wrong_accepted=fp,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        negative_false_accepts=false_accepts,
        negative_false_accept_rate=false_accepts / negatives if negatives else None)


def predicted(logits, policy):
    values = np.asarray(logits, dtype=np.float64)
    assert values.shape == (32,) and np.isfinite(values).all()
    p = np.exp(values - values.max())
    p /= p.sum()
    index = int(p.argmax())
    margin = float(p[index] - np.partition(p, -2)[-2])
    accepted = bool(policy['enabled'] and index and p[index] >= policy['min_probability'] and margin >= policy['min_margin'])
    return index, index if accepted else 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    import torch
    import onnx
    from threadpoolctl import threadpool_limits
    from scripts.runtime.optionb_contract import COMMANDS
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.runtime.pcv_schema import baseline_cases
    from scripts.test.speaker_demo_suite import read_audio, pcm_digest, write_new
    from vcm.frontend import features

    protocol = read(STUDY / 'protocol.json')
    run = ROOT / protocol['outputs']['run']
    freeze = read(STUDY / 'data-freeze.json')
    fit = read(run / 'fit-verification.json')
    lock = read(STUDY / 'selection-before-test.json')
    model_lock = read(STUDY / 'model-selection-before-calibration.json')
    protected_count = check(freeze['historical_bindings'])
    for group in (freeze['bindings'], fit['bindings'], lock['bindings'], model_lock['bindings']):
        check(group)
    assert datetime.fromisoformat(model_lock['created_utc']) <= datetime.fromisoformat(lock['created_utc'])
    data = {split: rows(run / f'{split}.jsonl') for split in ('train', 'dev', 'calibration', 'test', 'holdout')}
    for split in ('dev', 'calibration', 'test', 'holdout'):
        assert digest(run / f'{split}.jsonl') == digest(PARENT / f'{split}.jsonl'), split
    old_train = {r['id']: r for r in rows(PARENT / 'train.jsonl')}
    train = data['train']
    train_ids = {r['id'] for r in train}
    assert len(train_ids) == len(train) and all(r == old_train[r['id']] for r in train)
    excluded = rows(STUDY / 'excluded-historical-gap-rows.jsonl')
    assert set(old_train) - train_ids == {r['id'] for r in excluded}
    assert all(r['historical_manifest_split'] in ('val', 'test', '', None) for r in excluded)
    assert {r['class_index'] for r in train if r['class_index']} == set(range(1, 32))
    human = [r for r in train if r['kind'] == 'human']
    human_ids = {r['id'] for r in human}
    reserved = [r for split in ('dev', 'calibration', 'test', 'holdout') for r in data[split]]
    fields = ('id', 'speaker', 'sha256', 'pcm_sha256', 'pair', 'source_speaker_key',
              'session_id', 'parent_recording_id', 'speaker_alias_group')
    for field in fields:
        left = {json.dumps(r[field], sort_keys=True) for r in train if r.get(field) not in (None, '')}
        right = {json.dumps(r[field], sort_keys=True) for r in reserved if r.get(field) not in (None, '')}
        assert not left & right, field
    parent = torch.load(ROOT / protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    assert parent['epoch'] == protocol['parent']['epoch'] == parent['config']['human_epochs']
    assert set(parent['sample_counts']) <= human_ids
    arms = [read(run / 'arms' / arm['name'] / 'training-result.json') for arm in protocol['arms']]
    updates = protocol['training']['epochs'] * protocol['training']['steps_per_epoch']
    for arm in arms:
        assert arm['optimizer_steps'] == updates and sum(arm['sample_counts'].values()) == updates * 32
        assert set(arm['sample_counts']) <= train_ids
        curve = rows(run / 'arms' / arm['arm']['name'] / 'training-curve.jsonl')
        assert len(curve) == protocol['training']['epochs'] and curve[-1]['optimizer_steps'] == updates
        chosen = max(curve, key=lambda e: tuple(e['rank']))
        assert chosen['epoch'] == arm['selected_epoch'] and chosen['rank'] == arm['selection_rank']
        checkpoint = torch.load(run / 'arms' / arm['arm']['name'] / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == chosen['epoch'] and set(checkpoint['sample_counts']) <= train_ids
        assert sum(checkpoint['sample_counts'].values()) == chosen['epoch'] * protocol['training']['steps_per_epoch'] * 32
        assert any(not torch.equal(v, parent['state_dict'][k]) for k, v in checkpoint['state_dict'].items())
    assert arms[0]['sample_counts'] == arms[1]['sample_counts'], 'Matched smoothing-control exposure diverged'
    chosen_arm = max(arms, key=lambda r: tuple(r['selection_rank']))
    assert chosen_arm['arm']['name'] == lock['selected_arm'] == model_lock['selected_arm']
    assert sum(r['optimizer_steps'] for r in arms) == fit['total_optimizer_steps']
    norm = read(run / 'normalization.json')
    assert set(norm['ids']) == human_ids and len(norm['ids']) == len(human)
    first, second = [], []
    with threadpool_limits(limits=1, user_api='blas'):
        for row in human:
            raw = features(read_audio(row))
            raw_digest = hashlib.sha256(f'f32le:{raw.shape}\0'.encode() + np.asarray(raw, dtype='<f4').tobytes()).hexdigest()
            assert raw_digest == norm['raw_feature_tensor_sha256'][row['id']]
            raw = raw.astype(np.float64)
            first.append(raw.mean(axis=1))
            second.append(np.square(raw).mean(axis=1))
    mean = np.mean(np.stack(first), axis=0)
    std = np.sqrt(np.maximum(np.mean(np.stack(second), axis=0) - mean ** 2, 1e-8))
    np.testing.assert_array_equal(mean.astype(np.float32), np.asarray(norm['mean'], dtype=np.float32))
    np.testing.assert_array_equal(std.astype(np.float32), np.asarray(norm['std'], dtype=np.float32))
    inputs = rows(run / 'raw-feature-inputs.jsonl')
    assert {r['id'] for r in inputs} == train_ids | {r['id'] for r in data['dev']}
    assert {r['split'] for r in inputs} == {'train', 'dev'}
    hard = read(run / 'hard-negative-pool.json')
    assert hard['parent_checkpoint_sha256'] == freeze['parent_checkpoint_sha256']
    assert set(hard['ids']) <= {r['id'] for r in human if r['class_index'] == 0 and r.get('transcript_source') != 'whisper'}
    assert not hard['labels_changed'] and not hard['asr_derived_amplification']
    policy = read(run / 'bundle/policy.json')
    selection = read(STUDY / 'policy-selection.json')
    assert selection['policy'] == policy == lock['policy']
    if selection['feasible']:
        for split in ('dev', 'calibration'):
            m = selection['metrics'][split]
            assert m['tp'] > 0 and m['precision'] >= .99 and m['negative_false_accept_rate'] <= .01
    result = read(STUDY / 'results.json')
    predictions = rows(run / 'final-predictions.jsonl')
    expected_count = len(data['test']) + len(data['holdout']) + 3 * sum(r['kind'] == 'human' and r['metric_eligible'] for r in data['holdout'])
    assert len(predictions) == result['prediction_rows'] == expected_count
    assert len({r['id'] for r in predictions}) == len(predictions)
    references = {(r['split'], r['id']): r for split in ('test', 'holdout') for r in data[split]}
    groups = defaultdict(list)
    fixed = dict(enabled=True, min_probability=.95, min_margin=0.)
    for row in predictions:
        reference = references[row['split'], row['parent_id']]
        for field in ('class_index', 'expected', 'optionb_command', 'metric_eligible', 'scope', 'kind'):
            assert row[field] == reference[field], (row['id'], field)
        assert row['source_audio_sha256'] == reference['sha256'] and row['source_pcm_sha256'] == reference['pcm_sha256']
        if 'logits' in row['event']:
            raw, accepted = predicted(row['event']['logits'], policy)
            assert row['raw_index'] == raw and row['accepted_index'] == accepted
            assert row['fixed_index'] == predicted(row['event']['logits'], fixed)[1]
        else:
            assert row['raw_index'] == row['accepted_index'] == row['fixed_index'] == 0
        assert row['event']['actions_enabled'] is False and row['event']['model_qualified'] is False
        if row['metric_eligible']:
            groups['/'.join((row['split'], row['kind'], row['variant']))].append(row)
    for key, records in groups.items():
        for arm, field in (('raw', 'raw_index'), ('selected', 'accepted_index'), ('fixed_095', 'fixed_index')):
            assert metrics(records, field) == result['results'][key]['metrics'][arm], (key, arm)
    runtime = Runtime(run / 'bundle')
    for index in np.random.default_rng(97).choice(len(predictions), size=32, replace=False):
        row = predictions[int(index)]
        pcm = read_audio(references[row['split'], row['parent_id']])
        silence = np.zeros(16000, dtype=np.float32)
        if row['variant'] in ('silence_prefix_1s', 'silence_both_1s'):
            pcm = np.concatenate([silence, pcm])
        if row['variant'] in ('silence_suffix_1s', 'silence_both_1s'):
            pcm = np.concatenate([pcm, silence])
        assert pcm_digest(pcm) == row['actual_input_pcm_sha256']
        event = runtime.recognize(pcm)
        assert event.get('command_index', 0) == row['raw_index']
        assert (event.get('command_index', 0) if event['output']['accepted'] else 0) == row['accepted_index']
        if 'logits' in event:
            np.testing.assert_allclose(event['logits'], row['event']['logits'], rtol=3e-5, atol=3e-5)
    graph = onnx.load(str(run / 'bundle/model.onnx'))
    parameters = sum(int(np.prod(tensor.dims)) for tensor in graph.graph.initializer)
    assert parameters == lock['parameters'] == 258912
    assert read(run / 'bundle/commands.json') == list(COMMANDS) and len(baseline_cases()) == 93
    import csv
    csv_counts = {}
    for name in ('per-command', 'per-intent', 'per-verbatim'):
        with (STUDY / (name + '.csv')).open(newline='', encoding='utf-8') as stream:
            csv_counts[name] = len(list(csv.DictReader(stream)))
    assert csv_counts['per-command'] == len(groups) * 31
    assert csv_counts['per-verbatim'] == 93 and csv_counts['per-intent'] == result['csv_rows']['per_intent']
    for group in (freeze['historical_bindings'], freeze['bindings'], fit['bindings'], lock['bindings']):
        check(group)
    receipt = dict(status='verified', selected_arm=lock['selected_arm'], selected_epoch=lock['selected_epoch'],
        parameters=parameters, inference_rows=len(predictions), independently_replayed_original_or_stress_rows=32,
        human_normalization_rows_reconstructed=len(human), exact_human_normalization_match=True,
        optimizer_updates=sum(r['optimizer_steps'] for r in arms), matched_smoothing_exposure=True,
        protected_inputs_unchanged=protected_count, original_historical_inputs=freeze['original_historical_bindings'],
        metadata_label_rechecks=freeze['metadata_label_rechecks'], labels_changed=False,
        evaluation_membership_unchanged=True, exposed_regression_not_blind=True, installed=False,
        csv_rows=csv_counts, predictions_sha256=digest(run / 'final-predictions.jsonl'),
        verifier_sha256=digest(Path(__file__)))
    if args.write_receipt:
        write_new(STUDY / 'verification.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
