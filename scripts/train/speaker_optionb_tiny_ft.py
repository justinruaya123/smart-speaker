"""Versioned <=300k OptionB fine-tuning; never installs or actuates anything."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.test.speaker_optionb_holdout import annotation_review
from scripts.test.speaker_optionb_tiny_ft_audit import (
    AUDIT, PARENT, POLICY_GRID, STUDY, audit_training_rows, confidence_arrays,
    dev_operating_point, hard_negative_ids, score_logits,
)
from scripts.train.speaker_optionb_tiny import (
    array_digest, augmented, check_bindings, extract_arrays, predict_logits, tensor_batch,
)

SOURCE_BASE = Path('.cache/frontend-repro-20261019/source-view/.cache')
SOURCE_ROOTS = {
    'airimonda/ai231-me2-voice-commands': SOURCE_BASE / 'pcv-source-refresh-v1/source',
    'martinnavs/ai231-fil-supplemental-data': SOURCE_BASE / 'fil-supplemental-v1/source',
}


def is_gap_fill(row):
    location = row.get('location') or {}
    return (location.get('repository') == 'martinnavs/ai231-fil-supplemental-data'
            and location.get('file', '').startswith('gap_fill/'))


def fine_tuning_view(rows, reserved, metadata):
    audit_training_rows(rows, reserved)
    kept, excluded = [], []
    for row in rows:
        if is_gap_fill(row):
            if row['id'] not in metadata:
                raise ValueError('Missing historical source split: ' + row['id'])
            historical = metadata[row['id']].get('manifest_split')
            if historical not in ('train', 'val', 'test', '', None):
                raise ValueError('Unrecognized historical source split; do not normalize it')
            if historical != 'train':
                excluded.append(dict(id=row['id'], historical_manifest_split=historical,
                    reason='conservative_historical_gap_split_quarantine', label_changed=False))
                continue
        kept.append(row)
    return kept, excluded


def checked_human_parent(parent, rows):
    human_ids = {r['id'] for r in rows if r['kind'] == 'human'}
    if (parent['epoch'] != parent['config']['human_epochs'] or not parent['sample_counts']
            or not set(parent['sample_counts']) <= human_ids
            or any(type(n) is not int or n <= 0 for n in parent['sample_counts'].values())):
        raise ValueError('A verified human-only foundation is required; no synthetic continuation parent')


def prepare_dataset():
    import pyarrow.parquet as pq
    import torch
    if (STUDY / 'data-freeze.json').exists():
        raise FileExistsError('Preserve the earlier preparation')
    protocol = read_json(STUDY / 'protocol.json')
    if {k: protocol['selection'][k] for k in POLICY_GRID} != POLICY_GRID:
        raise ValueError('The audit and predeclared policy grids disagree')
    old = read_json(Path('experiments/speaker-optionb-tiny-v1/data-freeze.json'))
    old_lock = read_json(Path('experiments/speaker-optionb-tiny-v1/selection-before-test.json'))
    protected = {}
    for group in (old['historical_bindings'], old['bindings'], old_lock['bindings']):
        check_bindings(group)
        for path, digest in group.items():
            if path in protected and protected[path] != digest:
                raise ValueError('Contradictory protected input identities')
            protected[path] = digest
    prior_files = [*Path('experiments/speaker-optionb-tiny-v1').glob('*'),
                   *PARENT.glob('*'), *(PARENT / 'bundle').glob('*'),
                   Path('docs/EXPERIMENT-speaker-optionb-tiny-v1.md')]
    for path in prior_files:
        if path.is_file():
            protected[path.as_posix()] = file_digest(path)
    original = {s: read_jsonl(PARENT / f'{s}.jsonl') for s in ('train', 'dev', 'calibration', 'test', 'holdout')}
    reserved = [r for s in ('dev', 'calibration', 'test', 'holdout') for r in original[s]]
    parent = torch.load(protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    checked_human_parent(parent, original['train'])
    approval_path = Path('experiments/pcv-release-train-v1/annotation-approval.json')
    approved = read_json(approval_path)
    approval_freeze = read_json(Path('experiments/pcv-release-train-v1/data-freeze.json'))
    approval_key = next(k for k in approval_freeze['bindings'] if k.replace('\\', '/') == approval_path.as_posix())
    if file_digest(approval_path) != approval_freeze['bindings'][approval_key]:
        raise ValueError('Archived reference approval changed')
    groups = defaultdict(list)
    for row in original['train']:
        location = row['location']
        groups[(location['repository'], location['revision'], location['file'], location['sha256'])].append(row)
    metadata, source_bindings, rechecks = {}, {}, []
    for (repo, revision, name, digest), members in groups.items():
        if repo not in SOURCE_ROOTS:
            raise ValueError('Unknown provided source repository')
        base = SOURCE_ROOTS[repo].resolve()
        path = (base / name).resolve()
        if not path.is_relative_to(base) or file_digest(path) != digest:
            raise ValueError('Source shard identity mismatch: ' + name)
        source_bindings[path.as_posix()] = digest
        parquet = pq.ParquetFile(path)
        items = parquet.read(columns=[c for c in parquet.schema_arrow.names
                                      if c not in ('audio', 'whisper_transcript')]).to_pylist()
        for row in members:
            number = row['location']['row']
            if type(number) is not int or not 0 <= number < len(items):
                raise ValueError('Invalid pinned publisher row')
            metadata[row['id']] = items[number]
            result = annotation_review(row, items[number], policy=approved)
            serious = set(result['flags']) - {'asr_derived_not_independently_reviewed'}
            if serious:
                raise ValueError('Unresolved training annotation conflict: ' + row['id'] + ' ' + str(sorted(serious)))
            rechecks.append(result)
        print(json.dumps(dict(source_metadata_checked=name, repository=repo,
            revision=revision, training_rows=len(members))), flush=True)
    rows, excluded = fine_tuning_view(original['train'], reserved, metadata)
    if len([r for r in rows if r['kind'] == 'human']) != protocol['data']['human_train_rows']:
        raise ValueError('Human training membership changed')
    if {r['class_index'] for r in rows if r['class_index']} != set(range(1, 32)):
        raise ValueError('Every complete OptionB command must remain in training')
    run = Path(protocol['outputs']['run'])
    paths = [STUDY / 'protocol.json', STUDY / 'audit-results.json',
             STUDY / 'parent-finer-grid-policy.json', Path(__file__),
             Path('scripts/test/speaker_optionb_tiny_ft_audit.py'),
             AUDIT / 'train-human-logits.jsonl', approval_path,
             Path('data/sources/fil-supplemental-NOTICE.txt')]
    bindings = {p.as_posix(): file_digest(p) for p in paths}
    bindings.update(source_bindings)
    for split, records in original.items():
        path = run / f'{split}.jsonl'
        write_new(path, rows if split == 'train' else records, jsonl=True)
        bindings[path.as_posix()] = file_digest(path)
    write_new(run / 'normalization.json', read_json(Path(protocol['parent']['normalization'])))
    bindings[(run / 'normalization.json').as_posix()] = file_digest(run / 'normalization.json')
    write_new(run / 'label-rechecks.jsonl', rechecks, jsonl=True)
    write_new(STUDY / 'excluded-historical-gap-rows.jsonl', excluded, jsonl=True)
    for path in (run / 'label-rechecks.jsonl', STUDY / 'excluded-historical-gap-rows.jsonl'):
        bindings[path.as_posix()] = file_digest(path)
    counts = audit_training_rows(rows, reserved)
    freeze = dict(bindings=bindings, historical_bindings=protected, counts=counts,
        original_historical_bindings=len(old['historical_bindings']), parent_checkpoint_sha256=file_digest(Path(protocol['parent']['checkpoint'])),
        parent_epoch=parent['epoch'], parent_sampled_original_human_ids=len(parent['sample_counts']),
        parent_sample_draws=sum(parent['sample_counts'].values()), parent_synthetic_sample_draws=0,
        historical_gap_exclusions=dict(Counter(r['historical_manifest_split'] for r in excluded)),
        metadata_label_rechecks=len(rechecks), label_recheck_flags=dict(Counter(f for r in rechecks for f in r['flags'])),
        labels_changed=False, new_audio_generated=False, independent_listening=False,
        fitting_splits=['train'], normalization_fit_split='train',
        metadata_only_preparation=True, heldout_payload_bytes_hashed_for_integrity=True,
        heldout_pcm_decoded_or_inferred_before_selection=False,
        baseline_git_head=__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    check_bindings(protected)
    write_new(STUDY / 'data-freeze.json', freeze)
    print(json.dumps({k: v for k, v in freeze.items() if k not in ('bindings', 'historical_bindings')}, indent=2), flush=True)
    return freeze


def fit_continuation(rows, dev, arrays, norm, parent, hard_ids, output, config, arm) -> dict[str, Any]:
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    audit_training_rows(rows, dev)
    checked_human_parent(parent, rows)
    if any(r['split'] != 'dev' or r.get('training_eligible') is True for r in dev):
        raise ValueError('Checkpoint selection uses nonfitting development only')
    human_ids = {r['id'] for r in rows if r['kind'] == 'human'}
    if norm.get('fit_split') != 'train' or set(norm['ids']) != human_ids or len(norm['ids']) != len(human_ids):
        raise ValueError('Exact inherited human-train normalizer membership mismatch')
    if (sum(arm[k] for k in ('human_positive', 'human_negative', 'synthetic_positive')) != 32
            or not 0 <= arm['hard_negative_draws'] <= arm['human_negative']
            or min(config['epochs'], config['steps_per_epoch']) < 1):
        raise ValueError('Invalid predeclared continuation sampler')
    allowed_hard = {r['id'] for r in rows if r['kind'] == 'human' and not r['class_index']
                    and r.get('transcript_source') != 'whisper'}
    if not set(hard_ids) <= allowed_hard:
        raise ValueError('Hard-example amplification requires non-ASR-derived human train negatives')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'training-curve.jsonl').exists():
        raise FileExistsError('Preserve the earlier continuation')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    model = OptionBTinyVCM()
    model.load_state_dict(parent['state_dict'])
    initial = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    pools = {group: defaultdict(lambda: defaultdict(list))
             for group in ('human_positive', 'human_negative', 'synthetic_positive', 'hard_negative')}
    hard_set = set(hard_ids)
    for index, row in enumerate(rows):
        if row['kind'] == 'synthetic' and not row['class_index']:
            raise ValueError('No synthetic-negative pool is authorized in this protocol')
        group = 'synthetic_positive' if row['kind'] == 'synthetic' else ('human_positive' if row['class_index'] else 'human_negative')
        key = row['class_index'] if row['class_index'] else row['scope']
        pools[group][key][str(row['speaker'])].append(index)
        if row['id'] in hard_set:
            pools['hard_negative'][key][str(row['speaker'])].append(index)
    if any(not pools[k] for k in ('human_positive', 'human_negative', 'synthetic_positive')):
        raise ValueError('Missing required sampling stratum')
    if arm['hard_negative_draws'] and not pools['hard_negative']:
        raise ValueError('Missing predeclared hard-negative pool')

    def sample(group, count):
        result = []
        keys = list(pools[group])
        for _ in range(count):
            speakers = pools[group][keys[int(rng.integers(len(keys)))]]
            names = list(speakers)
            indices = speakers[names[int(rng.integers(len(names)))]]
            result.append(indices[int(rng.integers(len(indices)))])
        return result

    human_dev = [r for r in dev if r['kind'] == 'human']
    synthetic_dev = [r for r in dev if r['kind'] == 'synthetic']
    best_key: tuple[Any, ...] | None = None
    best_epoch: int | None = None
    best_event: dict[str, Any] | None = None
    draws, steps = Counter(), 0
    started = time.perf_counter()
    with (output / 'training-curve.jsonl').open('x', encoding='utf-8') as stream:
        for epoch in range(1, config['epochs'] + 1):
            position = (epoch - 1) / max(1, config['epochs'] - 1)
            lr = config['minimum_learning_rate'] + .5 * (config['learning_rate'] - config['minimum_learning_rate']) * (1 + math.cos(math.pi * position))
            optimizer.param_groups[0]['lr'] = lr
            model.train()
            losses = []
            for _ in range(config['steps_per_epoch']):
                indices = (sample('human_positive', arm['human_positive'])
                    + sample('human_negative', arm['human_negative'] - arm['hard_negative_draws'])
                    + sample('hard_negative', arm['hard_negative_draws'])
                    + sample('synthetic_positive', arm['synthetic_positive']))
                rng.shuffle(indices)
                selected = [rows[i] for i in indices]
                values = [augmented(arrays[r['id']], norm, rng, r['kind'], config['context_max_frames']) for r in selected]
                targets = torch.tensor([r['class_index'] for r in selected], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                logits = model(*tensor_batch(values))
                loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=arm['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite continuation loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                draws.update(r['id'] for r in selected)
                steps += 1
            scores = predict_logits(model, dev, arrays, norm)
            keep_h = [i for i, r in enumerate(dev) if r['kind'] == 'human']
            keep_s = [i for i, r in enumerate(dev) if r['kind'] == 'synthetic']
            operating: dict[str, Any] = dev_operating_point(human_dev, scores[keep_h], config['selection'])
            synthetic = score_logits(synthetic_dev, scores[keep_s]) if synthetic_dev else dict(recall=None)
            key = (*operating['rank'], synthetic['recall'] or 0.)
            event: dict[str, Any] = dict(arm=arm['name'], epoch=epoch, loss=float(np.mean(losses)), learning_rate=lr,
                human_dev=operating, synthetic_dev_raw=synthetic, rank=list(key),
                optimizer_steps=steps, elapsed_seconds=time.perf_counter() - started)
            if best_key is None or key > best_key:
                best_key, best_epoch, best_event = key, epoch, event
                torch.save(dict(state_dict=model.state_dict(), epoch=epoch, arm=arm['name'],
                    config=config, arm_config=arm, parent_epoch=parent['epoch'], sample_counts=dict(draws)), output / 'best.pt')
                write_new(output / f'selected-dev-epoch-{epoch:02d}.jsonl', [dict(id=r['id'], class_index=r['class_index'], logits=s.tolist())
                    for r, s in zip(dev, scores)], jsonl=True)
            event['selected_epoch'] = best_epoch
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps(event, allow_nan=False), flush=True)
    assert best_key is not None and best_epoch is not None and best_event is not None
    result = dict(arm=arm, parameters=sum(p.numel() for p in model.parameters()),
        optimizer_steps=steps, selected_epoch=best_epoch, selected_on='dev',
        selection_rank=list(best_key), selected_development=best_event,
        fitting_splits=['train'], normalization_fit_split='train', parent_epoch=parent['epoch'],
        sample_counts=dict(draws), sample_draws=sum(draws.values()),
        draws_by_kind={kind: sum(draws[r['id']] for r in rows if r['kind'] == kind) for kind in ('human', 'synthetic')},
        weights_changed=not torch.equal(initial, next(model.parameters()).detach()),
        elapsed_seconds=time.perf_counter() - started, training_config=config,
        calibration_used=False, test_holdout_used=False)
    write_new(output / 'training-result.json', result)
    return result


def fit_study():
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    protocol, freeze = read_json(STUDY / 'protocol.json'), read_json(STUDY / 'data-freeze.json')
    for bindings in (freeze['bindings'], freeze['historical_bindings']):
        check_bindings(bindings)
    run = Path(protocol['outputs']['run'])
    if (run / 'fit-verification.json').exists() or (run / 'hard-negative-pool.json').exists():
        raise FileExistsError('Preserve the earlier fit or attempted fit')
    rows, dev = read_jsonl(run / 'train.jsonl'), read_jsonl(run / 'dev.jsonl')
    arrays = extract_arrays([*rows, *dev])
    norm = read_json(run / 'normalization.json')
    for identifier in norm['ids']:
        if array_digest(arrays[identifier]) != norm['raw_feature_tensor_sha256'][identifier]:
            raise ValueError('Fresh human feature identity changed')
    write_new(run / 'raw-feature-inputs.jsonl', [dict(id=r['id'], split=r['split'], kind=r['kind'],
        audio_sha256=r['sha256'], pcm_sha256=r['pcm_sha256'],
        raw_feature_tensor_sha256=array_digest(arrays[r['id']])) for r in [*rows, *dev]], jsonl=True)
    parent = torch.load(protocol['parent']['checkpoint'], map_location='cpu', weights_only=True)
    checked_human_parent(parent, rows)
    torch.set_num_threads(protocol['training']['threads'])
    model = OptionBTinyVCM().eval()
    model.load_state_dict(parent['state_dict'])
    trusted = [r for r in rows if r['kind'] == 'human' and r.get('transcript_source') != 'whisper']
    parent_scores = predict_logits(model, trusted, arrays, norm)
    hard_ids = hard_negative_ids(trusted, parent_scores)
    probabilities = confidence_arrays(parent_scores, len(trusted))[0]
    write_new(run / 'hard-negative-pool.json', dict(ids=hard_ids, parent_epoch=parent['epoch'],
        parent_checkpoint_sha256=freeze['parent_checkpoint_sha256'], fitting_split='train',
        scores={r['id']: float(probabilities[i, 1:].max()) for i, r in enumerate(trusted) if r['id'] in set(hard_ids)},
        asr_derived_amplification=False, labels_changed=False))
    config = {**protocol['training'], 'selection': protocol['selection']}
    results = [fit_continuation(rows, dev, arrays, norm, parent, hard_ids, run / 'arms' / arm['name'], config, arm)
               for arm in protocol['arms']]
    chosen = max(results, key=lambda result: tuple(result['selection_rank']))
    checkpoint = torch.load(run / 'arms' / chosen['arm']['name'] / 'best.pt', map_location='cpu', weights_only=True)
    torch.save(checkpoint, run / 'best.pt')
    write_new(run / 'training-result.json', {**chosen, 'total_experiment_optimizer_steps': sum(r['optimizer_steps'] for r in results),
        'total_experiment_sample_draws': sum(r['sample_draws'] for r in results),
        'all_arm_results': [dict(name=r['arm']['name'], optimizer_steps=r['optimizer_steps'], selected_epoch=r['selected_epoch'],
                                selection_rank=r['selection_rank'], elapsed_seconds=r['elapsed_seconds']) for r in results]})
    selected_curve = read_jsonl(run / 'arms' / chosen['arm']['name'] / 'training-curve.jsonl')
    write_new(run / 'training-curve.jsonl', selected_curve, jsonl=True)
    for bindings in (freeze['bindings'], freeze['historical_bindings']):
        check_bindings(bindings)
    receipt_paths = [run / 'best.pt', run / 'training-result.json', run / 'training-curve.jsonl',
        run / 'hard-negative-pool.json', run / 'raw-feature-inputs.jsonl',
        *[p for arm in protocol['arms'] for p in (run / 'arms' / arm['name']).glob('*') if p.is_file()]]
    receipt = dict(completed_arms=len(results), full_updates_per_arm=config['epochs'] * config['steps_per_epoch'],
        total_optimizer_steps=sum(r['optimizer_steps'] for r in results), fitting_splits=['train'],
        calibration_used_for_weights_or_arm=False, heldout_pcm_decoded_or_inferred_before_selection=False,
        bindings={p.as_posix(): file_digest(p) for p in receipt_paths}, selected_arm=chosen['arm']['name'],
        selected_epoch=chosen['selected_epoch'], frozen_inputs_unchanged=True,
        data_freeze_sha256=file_digest(STUDY / 'data-freeze.json'))
    write_new(run / 'fit-verification.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'bindings'}, indent=2), flush=True)
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'fit'])
    arguments = parser.parse_args()
    if arguments.operation == 'prepare':
        prepare_dataset()
    else:
        fit_study()
