"""Read-only train/development diagnosis for the versioned tiny-model fine-tune."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np

from scripts.runtime.optionb_contract import COMMANDS
from scripts.runtime.pcv_schema import normalize_text
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new
from scripts.train.speaker_human_data import split_leakage
from scripts.train.speaker_optionb_tiny import (
    array_digest, check_bindings, complete_class_metrics, normalized,
    predict_logits, train_normalization, training_view,
)

PARENT = Path('.cache/speaker-optionb-tiny-v1/run-01')
STUDY = Path('experiments/speaker-optionb-tiny-ft-v1')
AUDIT = Path('.cache/speaker-optionb-tiny-ft-v1/audit-01')
POLICY_GRID = dict(
    probability_grid=[.5, .6, .7, .8, .85, .9, .92, .94, .95, .96, .97,
                      .975, .98, .985, .99, .995, .999, .9995, .9999],
    margin_grid=[0., .05, .1, .2, .3],
)


def audit_training_rows(rows, reserved):
    projected = training_view(rows, reserved)
    if split_leakage([*rows, *reserved]):
        raise ValueError('Declared train/reserved identity overlap, including synthetic groups')
    required = {'class_index', 'optionb_command', 'scope'}
    for source, rebuilt in zip(rows, projected):
        if (not required <= source.keys() or type(source['class_index']) is not int
                or any(source[key] != rebuilt[key] for key in required)
                or source.get('reference_status') in {'ambiguous', 'auxiliary'}
                or source.get('capture_fault') is True):
            raise ValueError('Invalid, ambiguous or stale training reference: ' + source['id'])
    text_groups = defaultdict(list)
    for row in rows:
        text = normalize_text(row.get('text') or '')
        if text:
            text_groups[text].append(row)
    conflicts = [dict(text=text, class_indices=sorted({r['class_index'] for r in members}),
                      ids=sorted(r['id'] for r in members), action='review_only')
                 for text, members in sorted(text_groups.items())
                 if len({r['class_index'] for r in members}) > 1]
    counts, missing = {}, {}
    for kind in ('human', 'synthetic'):
        selected = [r for r in rows if r['kind'] == kind]
        counts[kind] = dict(Counter(r['scope'] for r in selected))
        covered = {r['class_index'] for r in selected if r['class_index']}
        missing[kind] = [dict(class_index=i, command=COMMANDS[i])
                         for i in range(1, 32) if i not in covered]
    return dict(rows=len(rows), counts=counts, missing_commands=missing,
                text_reference_conflicts=conflicts, labels_changed=False,
                independent_listening=False, reserved_metadata_rows_checked=len(reserved),
                all_declared_cross_split_groups_disjoint=True)


def confidence_arrays(scores, count):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (count, 32) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite 32-way logit vector per reference')
    probabilities = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    raw = np.argmax(probabilities, axis=1)
    maximum = np.max(probabilities, axis=1)
    margin = maximum - np.partition(probabilities, -2, axis=1)[:, -2]
    return probabilities, raw, maximum, margin


def score_logits(rows, scores, policy=None):
    from scripts.runtime.speaker_optionb_tiny import decision
    _, predicted, maximum, margin = confidence_arrays(scores, len(rows))
    if policy is not None:
        decision(np.zeros(32), policy)
        accept = (policy['enabled'] & (predicted > 0)
                  & (maximum >= policy['min_probability']) & (margin >= policy['min_margin']))
        predicted = np.where(accept, predicted, 0)
    eligible = np.asarray([r.get('metric_eligible') is True for r in rows], dtype=bool)
    return complete_class_metrics([r for r, keep in zip(rows, eligible) if keep], predicted[eligible])


def hard_negative_ids(rows, scores, fraction=.25):
    """Parent-score mining changes train exposure only, never a reference label."""
    audit_training_rows(rows, [])
    if isinstance(fraction, bool) or not 0 < fraction <= 1:
        raise ValueError('Invalid hard-negative fraction')
    probabilities, _, _, _ = confidence_arrays(scores, len(rows))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row['kind'] == 'human' and row['class_index'] == 0:
            groups[row['scope']].append((float(probabilities[index, 1:].max()), row['id']))
    result = []
    for members in groups.values():
        ordered = sorted(members, key=lambda item: (-item[0], item[1]))
        result.extend(identifier for _, identifier in ordered[:math.ceil(len(ordered) * fraction)])
    return sorted(result)


def dev_operating_point(rows, scores, config):
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or any(r['split'] != 'dev' or r['kind'] != 'human'
                   or r.get('training_eligible') is True for r in rows)):
        raise ValueError('Weight selection requires unique nonfitting human development rows')
    raw = score_logits(rows, scores)
    feasible = []
    for probability in config['probability_grid']:
        for margin in config['margin_grid']:
            policy = dict(enabled=True, min_probability=probability, min_margin=margin)
            metrics = score_logits(rows, scores, policy)
            if (metrics['tp'] > 0 and metrics['precision'] is not None and metrics['precision'] >= .99
                    and metrics['negative_false_accept_rate'] is not None
                    and metrics['negative_false_accept_rate'] <= .01):
                feasible.append(((metrics['recall'], probability, margin), policy, metrics))
    if feasible:
        _, policy, metrics = max(feasible, key=lambda item: item[0])
    else:
        policy = dict(enabled=False, min_probability=.9999, min_margin=.3)
        metrics = score_logits(rows, scores, policy)
    return dict(policy=policy, metrics=metrics, raw=raw, feasible=bool(feasible),
                rank=(int(bool(feasible)), metrics['recall'] or 0., raw['f1'] or 0., raw['precision'] or 0.),
                selected_on='dev', calibration_used=False, fitting_splits=[])


def run_audit():
    import torch
    from threadpoolctl import threadpool_limits
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.test.test_speaker_optionb_tiny import select_policy
    from vcm.frontend import features, SPEC

    if (STUDY / 'audit-results.json').exists() or (AUDIT / 'train-human-logits.jsonl').exists():
        raise FileExistsError('Preserve the earlier audit and select a new audit version')
    freeze = read_json('experiments/speaker-optionb-tiny-v1/data-freeze.json')
    parent_lock = read_json('experiments/speaker-optionb-tiny-v1/selection-before-test.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings'], parent_lock['bindings']):
        check_bindings(bindings)
    views = {s: read_jsonl(PARENT / f'{s}.jsonl') for s in ('train', 'dev', 'calibration', 'test', 'holdout')}
    reserved = [r for s in ('dev', 'calibration', 'test', 'holdout') for r in views[s]]
    audit = audit_training_rows(views['train'], reserved)
    human = [r for r in views['train'] if r['kind'] == 'human']
    norm = read_json(PARENT / 'normalization.json')
    arrays, acoustic = {}, []
    with threadpool_limits(limits=1, user_api='blas'):
        for index, row in enumerate(human):
            pcm = read_audio(row)
            arrays[row['id']] = features(pcm)
            if array_digest(arrays[row['id']]) != norm['raw_feature_tensor_sha256'][row['id']]:
                raise ValueError('Parent training feature mismatch: ' + row['id'])
            acoustic.append(dict(id=row['id'], seconds=len(pcm) / 16000.,
                peak=float(np.max(np.abs(pcm))), rms=float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))),
                clipped_fraction=float(np.mean(np.abs(pcm) >= .999))))
            if (index + 1) % 500 == 0:
                print(f'Original human train recordings audited: {index + 1}/{len(human)}', flush=True)
    reconstructed = train_normalization(human, arrays)
    if reconstructed != norm:
        raise ValueError('Exact parent human-train normalization reconstruction failed')
    torch.set_num_threads(2)
    model = OptionBTinyVCM().eval()
    checkpoint = torch.load(PARENT / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    scores = predict_logits(model, human, arrays, norm)
    runtime = Runtime(PARENT / 'bundle')
    parity = []
    for index in np.random.default_rng(71).choice(len(human), size=min(16, len(human)), replace=False):
        values = normalized(arrays[human[index]['id']], norm)[None]
        actual = runtime.session.run(['logits'], {'features': values,
            'lengths': np.asarray([values.shape[-1]], dtype=np.int64)})[0][0]
        np.testing.assert_allclose(actual, scores[index], atol=3e-5, rtol=3e-5)
        parity.append(float(np.max(np.abs(actual - scores[index]))))
    probabilities, raw, confidence, margin = confidence_arrays(scores, len(human))
    hard_ids = hard_negative_ids(human, scores)
    wrong = [dict(id=r['id'], scope=r['scope'], expected=r['expected'], class_index=r['class_index'],
                  candidate=COMMANDS[int(raw[i])], probability=float(confidence[i]), margin=float(margin[i]),
                  text=r.get('text'), transcript_source=r.get('transcript_source'),
                  reference_reason=r.get('reference_reason'), review_only=True)
             for i, r in enumerate(human) if raw[i] > 0 and raw[i] != r['class_index']]
    wrong.sort(key=lambda r: (-r['probability'], r['id']))
    train_by_scope = {}
    for scope in ('optionb', 'original_unknown', 'outside_optionb'):
        keep = [i for i, row in enumerate(human) if row['scope'] == scope]
        train_by_scope[scope] = score_logits([human[i] for i in keep], scores[keep])
    cached = read_jsonl(PARENT / 'calibration-predictions.jsonl')
    selection_rows = [r for s in ('dev', 'calibration') for r in views[s] if r['kind'] == 'human']
    by_id = {r['id']: r for r in selection_rows}
    assert set(by_id) == {r['id'] for r in cached}
    assert all(r['class_index'] == by_id[r['id']]['class_index'] and r['split'] == by_id[r['id']]['split'] for r in cached)
    baseline_policy = select_policy(selection_rows, {r['id']: r['logits'] for r in cached}, POLICY_GRID)
    fine_policy = {k: v for k, v in baseline_policy.items() if k != 'grid'}
    receptive_frames = 5 + sum(2 * (k - 1) * d for k, d in zip(model.network.config['kernels'], model.network.config['dilations']))
    receptive_seconds = ((receptive_frames - 1) * 2 * SPEC['hop'] + SPEC['window']) / SPEC['sample_rate']
    summary = dict(data=audit, parent_epoch=checkpoint['epoch'], parameters=sum(p.numel() for p in model.parameters()),
        human_train_raw=score_logits(human, scores), human_train_by_scope=train_by_scope,
        parent_finer_grid_policy=fine_policy, policy_grid=POLICY_GRID,
        parent_policy=read_json(PARENT / 'bundle/policy.json'),
        smoothing_target_correct_class=1 - .02 + .02 / 32,
        smoothing_not_a_hard_probability_ceiling=True,
        receptive_field_encoded_frames=receptive_frames, receptive_field_seconds=receptive_seconds,
        recording_duration_percentiles={str(q): float(np.percentile([r['seconds'] for r in acoustic], q)) for q in (0, 25, 50, 75, 95, 100)},
        human_train_recordings_longer_than_receptive_field=sum(r['seconds'] > receptive_seconds for r in acoustic),
        recordings_exceeding_runtime_15s=sum(r['seconds'] > 15 for r in acoustic),
        near_silent_recordings=sum(r['peak'] < 1e-6 for r in acoustic),
        heavy_clipping_recordings=sum(r['clipped_fraction'] > .01 for r in acoustic),
        transcript_source_counts=dict(Counter(r.get('transcript_source', '<absent>') for r in human)),
        parent_torch_onnx_parity=dict(recordings=len(parity), max_abs_logit_error=max(parity)),
        hard_negative_ids=hard_ids, hard_negative_count=len(hard_ids),
        wrong_human_train_proposals=len(wrong), highest_confidence_train_wrong_proposals=wrong[:24],
        labels_changed=False, independent_listening=False, optimizer_updates=0,
        decoded_splits=['train'], reserved_pcm_decoded_or_inferred=False,
        reserved_parent_scores_reused_splits=['dev', 'calibration'],
        normalization_reconstructed_exactly=True, historical_bindings_checked=len(freeze['historical_bindings']),
        source_sha256=file_digest(Path(__file__)))
    write_new(AUDIT / 'train-human-logits.jsonl', [dict(id=r['id'], class_index=r['class_index'], logits=s.tolist())
        for r, s in zip(human, scores)], jsonl=True)
    write_new(AUDIT / 'train-acoustic-audit.jsonl', acoustic, jsonl=True)
    write_new(AUDIT / 'train-reference-review.jsonl', wrong, jsonl=True)
    write_new(STUDY / 'audit-results.json', summary)
    write_new(STUDY / 'parent-finer-grid-policy.json', baseline_policy)
    for bindings in (freeze['bindings'], freeze['historical_bindings'], parent_lock['bindings']):
        check_bindings(bindings)
    import json
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    run_audit()
