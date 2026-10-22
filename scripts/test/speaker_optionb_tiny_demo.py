"""Finite OptionB demo acceptance; unchanged model, no fitting or device actions."""
from pathlib import Path

from scripts.test.speaker_optionb_tiny_ft_audit import confidence_arrays, score_logits

STUDY = Path('experiments/speaker-optionb-tiny-demo-v1')


def validated_cached_scores(rows, cached):
    """Recorded scores are usable only for their exact original references."""
    references = {r['id']: r for r in rows}
    scores = {r['id']: r['logits'] for r in cached}
    if len(references) != len(rows) or len(scores) != len(cached) or set(scores) != set(references):
        raise ValueError('Cached score membership differs from references')
    for row in cached:
        ref = references[row['id']]
        if (type(row['class_index']) is not int or row['class_index'] != ref['class_index']
                or row['split'] != ref['split']):
            raise ValueError('Cached score reference class/split changed')
    confidence_arrays([scores[r['id']] for r in rows], len(rows))
    return scores


def select_demo_policy(rows, scores, config):
    """Choose complete-command F1, not a .99 precision feasibility constraint."""
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or set(scores) != {r['id'] for r in rows}
            or {r['split'] for r in rows} != {'dev', 'calibration'}
            or any(r['kind'] != 'human' or r.get('training_eligible') is not False for r in rows)):
        raise ValueError('Expected unique nonfitting human dev/calibration rows and exact scores')
    grid = []
    for probability in config['probability_grid']:
        for margin in config['margin_grid']:
            policy = dict(enabled=True, min_probability=probability, min_margin=margin)
            metrics = {split: score_logits([r for r in rows if r['split'] == split],
                [scores[r['id']] for r in rows if r['split'] == split], policy)
                for split in ('dev', 'calibration')}
            metrics['pooled'] = score_logits(rows, [scores[r['id']] for r in rows], policy)
            pooled = metrics['pooled']
            rank = (pooled['f1'] or 0., min(metrics[s]['f1'] or 0. for s in ('dev', 'calibration')),
                    pooled['precision'] or 0., pooled['recall'] or 0., probability, margin)
            grid.append(dict(policy=policy, metrics=metrics, rank=rank))
    chosen = max(grid, key=lambda point: tuple(point['rank']))
    return dict(**chosen, grid=grid, selected_on=['dev', 'calibration'],
                objective='human_pooled_complete_command_micro_f1', optimizer_updates=0)


def select_study(study=STUDY):
    """Rebind a new demo policy without changing acoustic weights or old assets."""
    from datetime import datetime, timezone
    import json
    import math
    import shutil
    import onnx
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.test.speaker_demo_suite import read_json, read_jsonl, write_new, file_digest
    from scripts.train.speaker_optionb_tiny import check_bindings

    study = Path(study)
    protocol = read_json(study / 'protocol.json')
    run = Path(protocol['outputs']['run'])
    parent = Path(protocol['parent']['run'])
    source_bundle = Path(protocol['parent']['bundle'])
    if run.exists() or (study / 'selection-before-test.json').exists():
        raise FileExistsError('Preserve the earlier demo selection/run')
    historical = dict(read_json(Path('experiments/speaker-optionb-tiny-direct-ft-v1/data-freeze.json'))['historical_bindings'])
    paths = [*[parent / (split + '.jsonl') for split in ('dev', 'calibration', 'test', 'holdout')],
             parent / 'calibration-predictions.jsonl', parent / 'normalization.json',
             *[source_bundle / name for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json')],
             Path(protocol['parent']['verification']),
             Path('experiments/speaker-optionb-tiny-direct-ft-v1/verify-results-v2.py'),
             Path('experiments/speaker-optionb-tiny-ft-v1/verify-results.py'),
             Path('scripts/test/test_speaker_optionb_tiny.py')]
    for path in paths:
        digest = file_digest(path)
        if path.as_posix() in historical and historical[path.as_posix()] != digest:
            raise ValueError('Conflicting frozen parent input: ' + path.as_posix())
        historical[path.as_posix()] = digest
    check_bindings(historical)
    parent_runtime = Runtime(source_bundle)
    recipe = dict(parent_runtime.recipe)
    if recipe['selected_epoch'] != protocol['parent']['epoch']:
        raise ValueError('Unexpected parent checkpoint epoch')
    graph = onnx.load(str(source_bundle / 'model.onnx'))
    parameters = sum(math.prod(t.dims) for t in graph.graph.initializer)
    if parameters != recipe['parameters'] or parameters != protocol['scope']['parameters']:
        raise ValueError('Parent graph parameter count changed')
    rows = [r for split in ('dev', 'calibration') for r in read_jsonl(parent / (split + '.jsonl')) if r['kind'] == 'human']
    scores = validated_cached_scores(rows, read_jsonl(parent / 'calibration-predictions.jsonl'))
    selection = select_demo_policy(rows, scores, protocol['selection'])
    run.mkdir(parents=True, exist_ok=False)
    for split in ('dev', 'calibration', 'test', 'holdout'):
        shutil.copyfile(parent / (split + '.jsonl'), run / (split + '.jsonl'))
    for name in ('normalization.json', 'calibration-predictions.jsonl'):
        shutil.copyfile(parent / name, run / name)
    bundle = run / 'bundle'
    bundle.mkdir()
    for name in ('model.onnx', 'commands.json', 'normalization.json'):
        shutil.copyfile(source_bundle / name, bundle / name)
    write_new(bundle / 'policy.json', selection['policy'])
    recipe.update(binding={name: file_digest(bundle / name) for name in
                           ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')},
                  acceptance_scope='finite_optionb_demo_balanced_f1', qualified=False,
                  parent_recipe_sha256=file_digest(source_bundle / 'recipe.json'))
    write_new(bundle / 'recipe.json', recipe)
    Runtime(bundle)
    write_new(study / 'policy-selection.json', selection)
    bound = [study / 'protocol.json', Path(__file__), Path('scripts/test/test_speaker_optionb_tiny_demo.py'),
             *[run / (s + '.jsonl') for s in ('dev', 'calibration', 'test', 'holdout')],
             run / 'normalization.json', run / 'calibration-predictions.jsonl']
    freeze = dict(bindings={p.as_posix(): file_digest(p) for p in bound}, historical_bindings=historical,
                  fitting_splits=[], optimizer_updates=0, labels_changed=False, normalizer_unchanged=True,
                  current_reserved_membership_unchanged=True, inherited_provenance_caveats=protocol['parent']['inheritance'])
    write_new(study / 'data-freeze.json', freeze)
    lock_paths = [*bound, study / 'data-freeze.json', study / 'policy-selection.json',
                  *[bundle / name for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json')]]
    lock = dict(bindings={p.as_posix(): file_digest(p) for p in lock_paths}, parameters=parameters,
                selected_epoch=recipe['selected_epoch'], policy=selection['policy'],
                selected_on=['dev', 'calibration'], model_weights_changed=False,
                policy_only=True, optimizer_updates=0, normalization_split='inherited_train_only',
                test_holdout_used_for_selection=False, exposed_regression_not_blind=True,
                inherited_heldout_history_disclosed=True, cached_scores_reused=True,
                heldout_pcm_decoded_or_inferred_for_this_policy_selection=False,
                created_utc=datetime.now(timezone.utc).isoformat())
    check_bindings(historical)
    check_bindings(freeze['bindings'])
    write_new(study / 'selection-before-test.json', lock)
    print(json.dumps(dict(status='demo_policy_locked', policy=selection['policy'],
                          metrics=selection['metrics'], parameters=parameters,
                          optimizer_updates=0, installed=False), indent=2))
    return lock


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['select'])
    parser.parse_args()
    select_study()
