"""Treated human-first training for one bounded OptionB joint-command CNN."""
from __future__ import annotations

from collections import Counter
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts.runtime.optionb_contract import class_index, command_for_index
from scripts.train.speaker_human_data import split_leakage
from scripts.test.speaker_demo_suite import write_new, read_json, read_jsonl, read_audio, file_digest


def projected_row(row):
    result = deepcopy(row)
    index = class_index(row['expected'])
    result.update(class_index=index, optionb_command=command_for_index(index),
                  scope='optionb' if index else ('original_unknown' if row['expected'] is None else 'outside_optionb'))
    return result


def training_view(rows, reserved):
    """Projection changes task membership, never a source reference or split."""
    for row in rows:
        human = row.get('kind') == 'human' and row.get('synthetic') is False
        synthetic = row.get('kind') == 'synthetic' and row.get('synthetic') is True
        if (row.get('split') != 'train' or row.get('training_eligible') is not True
                or row.get('metric_eligible') is not True or not (human or synthetic)):
            raise ValueError('Only explicit, eligible train recordings may fit the model')
        if human and row.get('speaker') in (None, '', 'unknown'):
            raise ValueError('Human training requires a declared speaker')
    for field in ('id', 'sha256', 'pcm_sha256'):
        values = [r[field] for r in rows]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate training recording: ' + field)
        right = {r.get(field) for r in reserved if r.get(field)}
        if set(values) & right:
            raise ValueError('Reserved recording overlap: ' + field)
    human = [r for r in [*rows, *reserved] if r.get('kind') == 'human']
    if split_leakage(human):
        raise ValueError('Declared human cross-split identity overlap')
    return [projected_row(row) for row in rows]


def array_digest(values):
    values = np.asarray(values, dtype='<f4')
    header = f'f32le:{values.shape}\0'.encode()
    return hashlib.sha256(header + values.tobytes()).hexdigest()


def train_normalization(rows, arrays):
    training_view(rows, [])
    if not rows or any(r['kind'] != 'human' for r in rows):
        raise ValueError('Normalization uses original human training only')
    means, seconds = [], []
    for row in rows:
        values = np.asarray(arrays[row['id']], dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 40 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError('Invalid raw log-mel feature array')
        means.append(values.mean(axis=1))
        seconds.append(np.square(values).mean(axis=1))
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.maximum(np.mean(seconds, axis=0) - np.square(mean), 1e-8))
    return dict(mean=mean.astype(np.float32).tolist(), std=std.astype(np.float32).tolist(),
                ids=sorted(r['id'] for r in rows), fit_split='train', weighting='equal_utterance',
                raw_feature_tensor_sha256={r['id']: array_digest(arrays[r['id']]) for r in rows})


def pad_context(values, prefix, suffix, floor):
    if type(prefix) is not int or type(suffix) is not int or min(prefix, suffix) < 0:
        raise ValueError('Context lengths must be nonnegative integers')
    return np.concatenate([np.repeat(np.asarray(floor, dtype=np.float32)[:, None], prefix, axis=1),
                           values.copy(), np.repeat(np.asarray(floor, dtype=np.float32)[:, None], suffix, axis=1)], axis=1)


def complete_class_metrics(rows, predicted):
    truth = np.asarray([r['class_index'] for r in rows], dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if predicted.shape != truth.shape:
        raise ValueError('Prediction/reference count mismatch')
    positive = truth > 0
    tp = int(np.sum(positive & (predicted == truth)))
    fp = int(np.sum((predicted > 0) & (predicted != truth)))
    fn = int(np.sum(positive & (predicted != truth)))
    negatives = int(np.sum(~positive))
    false_accepts = int(np.sum(~positive & (predicted > 0)))
    return dict(rows=len(rows), positives=int(np.sum(positive)), negatives=negatives,
                tp=tp, fp=fp, fn=fn, correct_accepted=tp, wrong_accepted=fp,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                negative_false_accepts=false_accepts,
                negative_false_accept_rate=false_accepts / negatives if negatives else None)


def normalized(values, norm):
    return ((values - np.asarray(norm['mean'], dtype=np.float32)[:, None]) /
            np.asarray(norm['std'], dtype=np.float32)[:, None]).astype(np.float32)


def augmented(values, norm, rng, kind, max_context):
    result = values.copy()
    # Every synthetic sample gets channel/gain perturbation; humans usually do.
    if kind == 'synthetic' or rng.random() < .7:
        curve = np.interp(np.arange(40), np.linspace(0, 39, 4), rng.normal(0, .25, 4))
        result += np.asarray(curve + rng.normal(0, .3), dtype=np.float32)[:, None]
    if rng.random() < .65:
        prefix, suffix = map(int, rng.integers(0, max_context + 1, size=2))
        result = pad_context(result, prefix, suffix, np.full(40, np.log(1e-8), dtype=np.float32))
    result = normalized(result, norm)
    result += rng.normal(0, .015, result.shape).astype(np.float32)
    if rng.random() < .3:
        start, width = int(rng.integers(0, 36)), int(rng.integers(1, 5))
        result[start:start + width] = 0
    return result


def tensor_batch(values):
    import torch
    lengths = np.asarray([v.shape[1] for v in values], dtype=np.int64)
    batch = np.zeros((len(values), 40, int(lengths.max())), dtype=np.float32)
    for index, value in enumerate(values):
        batch[index, :, :value.shape[1]] = value
    return torch.from_numpy(batch), torch.from_numpy(lengths)


def predict_logits(model, rows, arrays, norm):
    import torch
    result = np.zeros((len(rows), 32), dtype=np.float32)
    order = sorted(range(len(rows)), key=lambda i: arrays[rows[i]['id']].shape[1])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            values = [normalized(arrays[rows[i]['id']], norm) for i in indices]
            result[indices] = model(*tensor_batch(values)).numpy()
    return result


def fit_model(rows, dev, arrays, output, config):
    """Random initialization, human foundation, then human-majority mixed fit."""
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    rows = training_view(rows, dev)
    if any(r['split'] != 'dev' for r in dev):
        raise ValueError('Checkpoint selection must use dev only')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'training-curve.jsonl').exists():
        raise FileExistsError('Preserve the earlier fit; choose a new run ID')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    human = [r for r in rows if r['kind'] == 'human']
    norm = train_normalization(human, arrays)
    write_new(output / 'normalization.json', norm)
    model = OptionBTinyVCM()
    initial = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    pools = {key: defaultdict(lambda: defaultdict(list)) for key in ('human_positive', 'human_negative', 'synthetic_positive')}
    for index, row in enumerate(rows):
        if row['kind'] == 'synthetic' and not row['class_index']:
            raise ValueError('This protocol has no synthetic-negative sampling arm')
        group = 'synthetic_positive' if row['kind'] == 'synthetic' else ('human_positive' if row['class_index'] else 'human_negative')
        key = row['class_index'] if row['class_index'] else row['scope']
        pools[group][key][str(row.get('speaker', 'tts_unknown'))].append(index)
    if any(not value for value in pools.values()):
        raise ValueError('Missing required human-positive, human-negative or synthetic-positive pool')

    def sample(group, count):
        choices = list(pools[group])
        result = []
        for _ in range(count):
            identities = pools[group][choices[int(rng.integers(len(choices)))]]
            speakers = list(identities)
            indices = identities[speakers[int(rng.integers(len(speakers)))]]
            result.append(indices[int(rng.integers(len(indices)))])
        return result

    best_key, selected_epoch, steps, draws = None, None, 0, Counter()
    last_loss = 0.
    started = time.perf_counter()
    human_count, synthetic_count = config['human_epochs'], config['mixed_epochs']
    if config['batch_size'] != 32 or min(human_count, synthetic_count, config['steps_per_epoch']) < 1:
        raise ValueError('Expected positive phase lengths and the declared 32-example sampler')
    with (output / 'training-curve.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, human_count + synthetic_count + 1):
            mixed = epoch > human_count
            position = (epoch - human_count - 1) / max(1, synthetic_count - 1)
            lr = (.00005 + .5 * (config['learning_rate'] - .00005) * (1 + math.cos(math.pi * position))) if mixed else config['learning_rate']
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            losses = []
            for _ in range(config['steps_per_epoch']):
                indices = sample('human_positive', 12 if mixed else 24) + sample('human_negative', 8)
                if mixed:
                    indices += sample('synthetic_positive', 12)
                rng.shuffle(indices)
                selected = [rows[i] for i in indices]
                values = [augmented(arrays[r['id']], norm, rng, r['kind'], config['context_max_frames']) for r in selected]
                targets = torch.tensor([r['class_index'] for r in selected], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*tensor_batch(values))
                loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=config['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                last_loss = losses[-1]
                draws.update(r['id'] for r in selected)
                steps += 1
            dev_logits = predict_logits(model, dev, arrays, norm)
            metrics = {}
            for kind in ('human', 'synthetic'):
                keep = [i for i, r in enumerate(dev) if r['kind'] == kind and r['metric_eligible']]
                metrics[kind] = complete_class_metrics([dev[i] for i in keep], np.argmax(dev_logits[keep], axis=1))
            key = (metrics['human']['f1'] or 0., metrics['human']['precision'] or 0., metrics['synthetic']['recall'] or 0.)
            checkpoint = dict(state_dict=model.state_dict(), epoch=epoch, config=config, sample_counts=dict(draws))
            if mixed and (best_key is None or key > best_key):
                best_key, selected_epoch = key, epoch
                torch.save(checkpoint, output / 'best.pt')
            if epoch == human_count:
                torch.save(checkpoint, output / 'human-foundation.pt')
            event = dict(epoch=epoch, phase='mixed' if mixed else 'human', loss=float(np.mean(losses)),
                         learning_rate=lr, dev=metrics, selected_epoch=selected_epoch,
                         elapsed_seconds=time.perf_counter() - started)
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(event, allow_nan=False), flush=True)
    result = dict(parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=steps,
                  selected_epoch=selected_epoch, selected_on='dev', fitting_splits=['train'],
                  normalization_ids=norm['ids'], last_loss=last_loss,
                  weights_changed=not torch.equal(initial, next(model.parameters()).detach()),
                  elapsed_seconds=time.perf_counter() - started,
                  unique_sampled_by_kind={kind: sum(r['id'] in draws for r in rows if r['kind'] == kind) for kind in ('human', 'synthetic')},
                  sample_counts=dict(draws), training_config=config)
    write_new(output / 'training-result.json', result)
    return result


STUDY = Path('experiments/speaker-optionb-tiny-v1')


def check_bindings(bindings):
    changed = [path for path, digest in bindings.items() if not Path(path).is_file() or file_digest(Path(path)) != digest]
    if changed:
        raise ValueError('Frozen input identity changed: ' + ', '.join(changed[:8]))
    return len(bindings)


def prepare_dataset():
    protocol_path = STUDY / 'protocol.json'
    protocol = read_json(protocol_path)
    prior_path = Path(protocol['sources']['historical_protected_inputs'])
    prior = read_json(prior_path)['bindings']
    check_bindings(prior)
    human_path = Path(protocol['sources']['treated_human'])
    original_path = Path(protocol['sources']['provided_synthetic_train_and_reserved_views'])
    run = Path(protocol['outputs']['run'])
    human = {split: read_jsonl(human_path / f'{split}.jsonl') for split in ('train', 'dev', 'calibration', 'test', 'holdout')}
    original = {split: read_jsonl(original_path / f'{split}.jsonl') for split in human}
    reserved = [r for split in ('dev', 'calibration', 'test', 'holdout') for r in [*original[split], *human[split]]]
    synthetic = [r for r in original['train'] if r['kind'] == 'synthetic']
    if len(human['train']) != protocol['data_rules']['human_training_rows_expected'] or len(synthetic) != protocol['data_rules']['synthetic_training_rows_expected']:
        raise ValueError('The declared source membership changed')
    views = {'train': training_view([*human['train'], *synthetic], reserved)}
    for split in ('dev', 'calibration'):
        views[split] = [projected_row(r) for r in [*human[split], *(r for r in original[split] if r['kind'] == 'synthetic')]]
    for split in ('test', 'holdout'):
        views[split] = [projected_row(r) for r in original[split]]
    if {r['class_index'] for r in views['train'] if r['class_index']} != set(range(1, 32)):
        raise ValueError('Every complete OptionB command must remain represented')
    source_paths = [protocol_path, prior_path,
        *[base / f'{split}.jsonl' for base in (human_path, original_path) for split in human],
        Path(__file__), Path('scripts/models/speaker_optionb_tiny.py'), Path('scripts/models/vcm.py'),
        Path('scripts/runtime/optionb_contract.py'), Path('scripts/runtime/pcv_schema.py'),
        Path('vcm/frontend.py'), Path('pyproject.toml'), Path('uv.lock')]
    bindings = {str(path).replace('\\', '/'): file_digest(path) for path in source_paths}
    counts = {}
    for split, rows in views.items():
        write_new(run / f'{split}.jsonl', rows, jsonl=True)
        bindings[str(run / f'{split}.jsonl').replace('\\', '/')] = file_digest(run / f'{split}.jsonl')
        counts[split] = {}
        for kind in ('human', 'synthetic'):
            selected = [r for r in rows if r['kind'] == kind]
            eligible = [r for r in selected if r['metric_eligible']]
            counts[split][kind] = dict(rows=len(selected), eligible=len(eligible), ambiguous=len(selected)-len(eligible),
                positives=sum(r['class_index'] > 0 for r in eligible),
                scopes=dict(Counter(r['scope'] for r in eligible)),
                covered_commands=sorted({r['class_index'] for r in eligible if r['class_index']}),
                declared_speakers=len({r.get('speaker') for r in eligible}))
    freeze = dict(protocol_sha256=file_digest(protocol_path), bindings=bindings, historical_bindings=prior,
                  counts=counts, recording_overlap=0, declared_human_cross_split_overlap=0,
                  fitting_splits=['train'], normalization_split='train', heldout_audio_read=False,
                  baseline_git_head=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write_new(STUDY / 'data-freeze.json', freeze)
    print(json.dumps(dict(status='prepared', counts=counts, protected_inputs_checked=len(prior)), indent=2), flush=True)
    return freeze


def extract_arrays(rows):
    from threadpoolctl import threadpool_limits
    from vcm.frontend import features
    arrays = {}
    with threadpool_limits(limits=1, user_api='blas'):
        for index, row in enumerate(rows):
            values = features(read_audio(row))
            if values.ndim != 2 or values.shape[0] != 40 or not values.shape[1] or not np.isfinite(values).all():
                raise ValueError('Invalid fresh features: ' + row['id'])
            arrays[row['id']] = values
            if (index + 1) % 2000 == 0:
                print(json.dumps(dict(feature_arrays=index + 1, total=len(rows))), flush=True)
    return arrays


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'fit'])
    args = parser.parse_args()
    if args.operation == 'prepare':
        prepare_dataset()
        return
    protocol = read_json(STUDY / 'protocol.json')
    freeze = read_json(STUDY / 'data-freeze.json')
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    run = Path(protocol['outputs']['run'])
    rows, dev = read_jsonl(run / 'train.jsonl'), read_jsonl(run / 'dev.jsonl')
    arrays = extract_arrays([*rows, *dev])
    write_new(run / 'raw-feature-inputs.jsonl', [dict(id=r['id'], split=r['split'], kind=r['kind'],
        audio_sha256=r['sha256'], pcm_sha256=r['pcm_sha256'],
        raw_feature_tensor_sha256=array_digest(arrays[r['id']])) for r in [*rows, *dev]], jsonl=True)
    fit_model(rows, dev, arrays, run, protocol['training'])
    check_bindings(freeze['bindings'])
    check_bindings(freeze['historical_bindings'])
    write_new(run / 'fit-verification.json', dict(frozen_inputs_unchanged=True, heldout_audio_read=False,
        feature_arrays=len(arrays), fitting_splits=['train'], normalization_split='train',
        selected_on='dev', data_freeze_sha256=file_digest(STUDY / 'data-freeze.json')))


if __name__ == '__main__':
    main()
