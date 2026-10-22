"""Bounded direct continuation of the trained tiny checkpoint; no installation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.test.speaker_optionb_tiny_ft_audit import audit_training_rows, dev_operating_point, hard_negative_ids, score_logits
from scripts.train.speaker_optionb_tiny import array_digest, augmented, check_bindings, extract_arrays, predict_logits, tensor_batch

STUDY = Path('experiments/speaker-optionb-tiny-direct-ft-v1')


def validate_parent_lineage(parent, original_training, reserved, *, expected_epoch):
    audit_training_rows(original_training, reserved)
    by_id = {r['id']: r for r in original_training}
    counts = parent.get('sample_counts') or {}
    if (parent.get('epoch') != expected_epoch or not counts or not set(counts) <= set(by_id)
            or any(type(n) is not int or n <= 0 for n in counts.values())):
        raise ValueError('Parent ancestry must match its declared TRAIN-only fitting cohort and epoch')
    return dict(parent_epoch=expected_epoch,
        human_draws=sum(n for key, n in counts.items() if by_id[key]['kind'] == 'human'),
        synthetic_draws=sum(n for key, n in counts.items() if by_id[key]['kind'] == 'synthetic'),
        actual_parent_training_ids=len(counts), current_reserved_overlap=False,
        human_only_parent=False, alias_resolved_voice_independence_established=False)


def bounded_replay_view(rows):
    audit_training_rows(rows, [])
    groups = defaultdict(list)
    human = [r for r in rows if r['kind'] == 'human']
    for row in rows:
        if row['kind'] == 'synthetic':
            groups[row['class_index'], row['speaker']].append(row)
    synthetic = [row for key in sorted(groups) for row in sorted(groups[key],
        key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:2]]
    return [*human, *synthetic]


def fit_direct(rows, dev, arrays, norm, parent, original_training, hard_ids, output, config, arm) -> dict[str, Any]:
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    validate_parent_lineage(parent, original_training, dev, expected_epoch=parent['epoch'])
    audit_training_rows(rows, dev)
    if any(r['split'] != 'dev' or r.get('training_eligible') is True for r in dev):
        raise ValueError('Only nonfitting development selects continuation weights')
    human_ids = {r['id'] for r in rows if r['kind'] == 'human'}
    if set(norm['ids']) != human_ids or norm['fit_split'] != 'train':
        raise ValueError('Inherited human-train normalizer membership changed')
    allowed_hard = {r['id'] for r in rows if r['kind'] == 'human' and not r['class_index'] and r.get('transcript_source') != 'whisper'}
    if not set(hard_ids) <= allowed_hard:
        raise ValueError('Invalid train-only hard-negative pool')
    if sum(arm[k] for k in ('human_positive', 'human_negative', 'synthetic_positive')) != 32:
        raise ValueError('Expected the declared32example sampler')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'training-curve.jsonl').exists():
        raise FileExistsError('Preserve the previous direct continuation')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    model = OptionBTinyVCM()
    model.load_state_dict(parent['state_dict'])
    initial = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    pools = {key: defaultdict(lambda: defaultdict(list)) for key in ('hp', 'hn', 'sp', 'hard')}
    for index, row in enumerate(rows):
        group = 'sp' if row['kind'] == 'synthetic' else ('hp' if row['class_index'] else 'hn')
        if group == 'sp' and not row['class_index']:
            raise ValueError('No synthetic-negative pool is declared')
        key = row['class_index'] if row['class_index'] else row['scope']
        pools[group][key][row['speaker']].append(index)
        if row['id'] in set(hard_ids):
            pools['hard'][key][row['speaker']].append(index)
    if any(not pools[group] for group in ('hp', 'hn', 'sp')) or (arm['hard_negative_draws'] and not pools['hard']):
        raise ValueError('Missing declared sampling group')

    def sample(group, count):
        result = []
        keys = list(pools[group])
        for _ in range(count):
            speakers = pools[group][keys[int(rng.integers(len(keys)))]]
            names = list(speakers)
            indices = speakers[names[int(rng.integers(len(names)))]]
            result.append(indices[int(rng.integers(len(indices)))])
        return result

    h_indices = [i for i, r in enumerate(dev) if r['kind'] == 'human']
    s_indices = [i for i, r in enumerate(dev) if r['kind'] == 'synthetic']
    h_rows, s_rows = [dev[i] for i in h_indices], [dev[i] for i in s_indices]
    baseline_scores = predict_logits(model, dev, arrays, norm)
    baseline_h = score_logits(h_rows, baseline_scores[h_indices])
    baseline_s = score_logits(s_rows, baseline_scores[s_indices]) if s_rows else dict(recall=None)
    draws, steps = Counter(), 0
    started = time.perf_counter()

    def measure(epoch, scores, loss, lr):
        human: dict[str, Any] = dev_operating_point(h_rows, scores[h_indices], config['selection'])
        synthetic: dict[str, Any] = score_logits(s_rows, scores[s_indices]) if s_rows else dict(recall=None)
        competent = ((human['raw']['recall'] or 0.) >= (baseline_h['recall'] or 0.)
            and (not s_rows or (synthetic['recall'] or 0.) >= (baseline_s['recall'] or 0.) - .01))
        key = (int(competent), *human['rank'], synthetic['recall'] or 0.)
        event = dict(arm=arm['name'], epoch=epoch, loss=loss, learning_rate=lr,
            human_dev=human, synthetic_dev_raw=synthetic, competence_pass=bool(competent), rank=list(key),
            optimizer_steps=steps, elapsed_seconds=time.perf_counter() - started)
        return key, event

    best_key, best_event = measure(0, baseline_scores, None, 0.)
    best_epoch = 0

    def save(epoch, scores):
        torch.save(dict(state_dict=model.state_dict(), epoch=epoch, arm=arm['name'], config=config,
            arm_config=arm, parent_epoch=parent['epoch'], sample_counts=dict(draws)), output / 'best.pt')
        write_new(output / f'selected-dev-epoch-{epoch:02d}.jsonl',
            [dict(id=r['id'], class_index=r['class_index'], logits=s.tolist()) for r, s in zip(dev, scores)], jsonl=True)

    save(0, baseline_scores)
    with (output / 'training-curve.jsonl').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({**best_event, 'selected_epoch': 0}, allow_nan=False) + '\n')
        stream.flush()
        print(json.dumps(best_event, allow_nan=False), flush=True)
        for epoch in range(1, config['epochs'] + 1):
            position = (epoch - 1) / max(1, config['epochs'] - 1)
            lr = config['minimum_learning_rate'] + .5 * (config['learning_rate'] - config['minimum_learning_rate']) * (1 + math.cos(math.pi * position))
            optimizer.param_groups[0]['lr'] = lr
            model.train()
            losses = []
            for _ in range(config['steps_per_epoch']):
                indices = (sample('hp', arm['human_positive']) + sample('hn', arm['human_negative'] - arm['hard_negative_draws'])
                    + sample('hard', arm['hard_negative_draws']) + sample('sp', arm['synthetic_positive']))
                rng.shuffle(indices)
                batch = [rows[i] for i in indices]
                values = [augmented(arrays[r['id']], norm, rng, r['kind'], config['context_max_frames']) for r in batch]
                targets = torch.tensor([r['class_index'] for r in batch], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(*tensor_batch(values)), targets, label_smoothing=arm['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite direct-continuation loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                draws.update(r['id'] for r in batch)
                steps += 1
                losses.append(float(loss.detach()))
            scores = predict_logits(model, dev, arrays, norm)
            key, event = measure(epoch, scores, float(np.mean(losses)), lr)
            if key > best_key:
                best_key, best_event, best_epoch = key, event, epoch
                save(epoch, scores)
            event['selected_epoch'] = best_epoch
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps(event, allow_nan=False), flush=True)
    torch.save(dict(state_dict=model.state_dict(), epoch=config['epochs'], parent_epoch=parent['epoch']), output / 'last.pt')
    result = dict(arm=arm, parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=steps,
        sample_counts=dict(draws), sample_draws=sum(draws.values()), selected_epoch=best_epoch,
        selected_on='dev', selection_rank=list(best_key), selected_development=best_event,
        parent_epoch=parent['epoch'], baseline_human_raw=baseline_h, baseline_synthetic_raw=baseline_s,
        trained_weights_changed=not torch.equal(initial, next(model.parameters()).detach()),
        selected_parent_unchanged=best_epoch == 0, elapsed_seconds=time.perf_counter() - started,
        fitting_splits=['train'], calibration_used=False, test_holdout_used=False)
    write_new(output / 'training-result.json', result)
    return result


def prepare_dataset():
    import torch
    if (STUDY / 'data-freeze.json').exists():
        raise FileExistsError('Preserve the earlier direct preparation')
    protocol = read_json(STUDY / 'protocol.json')
    previous = Path('experiments/speaker-optionb-tiny-ft-v1')
    previous_run = Path(protocol['data']['source_run'])
    prior_freeze = read_json(previous / 'data-freeze.json')
    protected = {}
    for group in (prior_freeze['historical_bindings'], prior_freeze['bindings'],
                  read_json(previous_run / 'fit-verification.json')['bindings'],
                  read_json(previous / 'selection-before-test.json')['bindings']):
        check_bindings(group)
        for path, value in group.items():
            if path in protected and protected[path] != value:
                raise ValueError('Conflicting protected input identities')
            protected[path] = value
    for path in (previous / 'results.json', previous / 'verification.json', previous / 'baseline-controls.json'):
        protected[path.as_posix()] = file_digest(path)
    original = {s: read_jsonl(previous_run / f'{s}.jsonl') for s in ('train', 'dev', 'calibration', 'test', 'holdout')}
    reserved = [r for s in ('dev', 'calibration', 'test', 'holdout') for r in original[s]]
    parent_training = read_jsonl(Path(protocol['parent']['original_training_membership']))
    parent = torch.load(protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    ancestry = validate_parent_lineage(parent, parent_training, reserved, expected_epoch=protocol['parent']['epoch'])
    selected = bounded_replay_view(original['train'])
    if {r['class_index'] for r in selected if r['class_index']} != set(range(1, 32)):
        raise ValueError('Every complete command must remain represented')
    run = Path(protocol['outputs']['run'])
    paths = [STUDY / 'protocol.json', Path(__file__), previous / 'parent-finer-grid-policy.json',
             Path(protocol['parent']['checkpoint']), Path(protocol['parent']['original_training_membership']),
             Path('scripts/test/test_speaker_optionb_tiny_direct_ft.py')]
    bindings = {p.as_posix(): file_digest(p) for p in paths}
    for split, records in original.items():
        path = run / f'{split}.jsonl'
        write_new(path, selected if split == 'train' else records, jsonl=True)
        bindings[path.as_posix()] = file_digest(path)
    write_new(run / 'normalization.json', read_json(Path(protocol['parent']['normalization'])))
    write_new(STUDY / 'parent-finer-grid-policy.json', read_json(previous / 'parent-finer-grid-policy.json'))
    for path in (run / 'normalization.json', STUDY / 'parent-finer-grid-policy.json'):
        bindings[path.as_posix()] = file_digest(path)
    old_excluded = {r['id'] for r in read_jsonl(previous / 'excluded-historical-gap-rows.jsonl')}
    freeze = dict(bindings=bindings, historical_bindings=protected,
        original_historical_bindings=prior_freeze['original_historical_bindings'],
        counts=audit_training_rows(selected, reserved), parent_lineage=ancestry,
        inherited_historical_gap_recordings=len(set(parent['sample_counts']) & old_excluded),
        parent_checkpoint_sha256=file_digest(Path(protocol['parent']['checkpoint'])),
        parent_epoch=parent['epoch'], metadata_label_rechecks=prior_freeze['metadata_label_rechecks'],
        new_references_or_audio=False, current_reserved_membership_unchanged=True,
        heldout_payload_bytes_hashed_for_integrity=True,
        heldout_pcm_decoded_or_inferred_for_continuation_selection=False,
        exposed_regression_not_blind=True)
    check_bindings(protected)
    write_new(STUDY / 'data-freeze.json', freeze)
    print(json.dumps({k: v for k, v in freeze.items() if k not in ('bindings', 'historical_bindings')}, indent=2), flush=True)
    return freeze


def fit_study():
    import torch
    protocol, freeze = read_json(STUDY / 'protocol.json'), read_json(STUDY / 'data-freeze.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings']):
        check_bindings(bindings)
    run = Path(protocol['outputs']['run'])
    if (run / 'fit-verification.json').exists() or (run / 'hard-negative-pool.json').exists():
        raise FileExistsError('Preserve the earlier direct fit')
    train, dev = read_jsonl(run / 'train.jsonl'), read_jsonl(run / 'dev.jsonl')
    arrays = extract_arrays([*train, *dev])
    norm = read_json(run / 'normalization.json')
    for key in norm['ids']:
        if array_digest(arrays[key]) != norm['raw_feature_tensor_sha256'][key]:
            raise ValueError('Fresh human feature mismatch')
    write_new(run / 'raw-feature-inputs.jsonl', [dict(id=r['id'], split=r['split'], kind=r['kind'],
        audio_sha256=r['sha256'], pcm_sha256=r['pcm_sha256'],
        raw_feature_tensor_sha256=array_digest(arrays[r['id']])) for r in [*train, *dev]], jsonl=True)
    cached = {r['id']: r['logits'] for r in read_jsonl(Path('.cache/speaker-optionb-tiny-ft-v1/audit-01/train-human-logits.jsonl'))}
    trusted = [r for r in train if r['kind'] == 'human' and r.get('transcript_source') != 'whisper']
    hard_ids = hard_negative_ids(trusted, np.asarray([cached[r['id']] for r in trusted]))
    write_new(run / 'hard-negative-pool.json', dict(ids=hard_ids, parent_epoch=protocol['parent']['epoch'],
        parent_checkpoint_sha256=freeze['parent_checkpoint_sha256'], fitting_split='train',
        asr_derived_amplification=False, labels_changed=False))
    parent = torch.load(protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    original_training = read_jsonl(Path(protocol['parent']['original_training_membership']))
    config = {**protocol['training'], 'selection': protocol['selection']}
    arm = protocol['arms'][0]
    output = run / 'arms' / arm['name']
    result = fit_direct(train, dev, arrays, norm, parent, original_training, hard_ids, output, config, arm)
    chosen = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    torch.save(chosen, run / 'best.pt')
    write_new(run / 'training-result.json', {**result, 'total_experiment_optimizer_steps': result['optimizer_steps'],
        'total_experiment_sample_draws': result['sample_draws'], 'all_arm_results': [dict(name=arm['name'],
            selected_epoch=result['selected_epoch'], selection_rank=result['selection_rank'], optimizer_steps=result['optimizer_steps'])]})
    write_new(run / 'training-curve.jsonl', read_jsonl(output / 'training-curve.jsonl'), jsonl=True)
    for bindings in (freeze['bindings'], freeze['historical_bindings']):
        check_bindings(bindings)
    paths = [run / 'best.pt', run / 'training-result.json', run / 'training-curve.jsonl',
             run / 'raw-feature-inputs.jsonl', run / 'hard-negative-pool.json',
             *[p for p in output.glob('*') if p.is_file()]]
    receipt = dict(completed_arms=1, full_updates_per_arm=config['epochs'] * config['steps_per_epoch'],
        total_optimizer_steps=result['optimizer_steps'], selected_arm=arm['name'], selected_epoch=result['selected_epoch'],
        selected_parent_unchanged=result['selected_parent_unchanged'], frozen_inputs_unchanged=True,
        calibration_used_for_weights_or_arm=False, heldout_pcm_decoded_or_inferred_before_selection=False,
        bindings={p.as_posix(): file_digest(p) for p in paths}, data_freeze_sha256=file_digest(STUDY / 'data-freeze.json'))
    write_new(run / 'fit-verification.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'bindings'}, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'fit'])
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare_dataset()
    else:
        fit_study()
