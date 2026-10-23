"""Additive source-only CPU fit comparison for the frozen tiny OptionB model.

Nothing here installs a model or reads test/holdout PCM. Failed attempts are
preserved in the requested run's fit/ directory; use a new run ID to retry.

Fitting requires the locked training+benchmark environment AND the hashed
export/test helpers: the unchanged canonical exporter imports pytest.
From checkout root, install the required helpers into the same root environment:
uv pip install --python .venv/Scripts/python.exe --require-hashes --no-deps --requirement submission/optionb-tiny/testing.lock
Use .venv/bin/python on POSIX. This stage never installs packages itself.
"""
from pathlib import Path
from itertools import zip_longest

from scripts.test.speaker_demo_suite import canonical, file_digest, read_json, read_jsonl, write_new


def _fit_target(fit, relative):
    """Check each fully derived path; never follow writer symlinks/junctions."""
    fit = Path(fit).absolute()
    path = fit / relative
    if (fit.resolve() != fit or not path.is_relative_to(fit) or path.resolve() != path
            or path.is_symlink() or path.is_junction()):
        raise ValueError('Fit artifact containment changed: ' + str(path))
    return path


def _fit_files(fit):
    """Walk top-down so a linked directory is rejected before traversing it."""
    pending, files = [_fit_target(fit, '.')], []
    while pending:
        for path in pending.pop().iterdir():
            path = _fit_target(fit, path.relative_to(fit))
            if path.is_dir():
                pending.append(path)
            elif path.is_file():
                files.append(path)
    return sorted(files)


def _write_new(fit, relative, value, *, jsonl=False):
    write_new(_fit_target(fit, relative), value, jsonl=jsonl)


def _prepared_rows(root, run, split, expected):
    from scripts.submission.optionb_reproduce import pinned_path

    rows = read_jsonl(_fit_target(run, f'prepared/{split}.jsonl'))
    if (not expected or len(rows) != len(expected)
            or len({r['id'] for r in rows}) != len(rows)):
        raise ValueError('Prepared membership changed: ' + split)
    result = []
    for index, (row, reference) in enumerate(zip(rows, expected)):
        if (canonical({k: v for k, v in row.items() if k != 'audio'}) != canonical(reference)
                or row['split'] != split):
            raise ValueError(f'Prepared metadata/order changed: {split}:{index}')
        audio = pinned_path(root, row['audio'])
        if not audio.is_relative_to(run) or audio != (run / 'audio' / (row['sha256'] + '.wav')).resolve():
            raise ValueError('Prepared audio must be this source run/audio/<sha256>.wav')
        result.append({**row, 'audio': str(audio)})
    return result


def _feature_preflight(rows, dev, golden, expected_norm, fit):
    """Canonical BLAS1 extraction hashes every TRAIN+DEV tensor before fitting."""
    from scripts.train import speaker_optionb_tiny as training

    all_rows = [*rows, *dev]
    if len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Feature preflight requires unique TRAIN+DEV IDs')
    arrays = training.extract_arrays(all_rows)
    actual = [dict(id=r['id'], split=r['split'], kind=r['kind'], audio_sha256=r['sha256'],
                   pcm_sha256=r['pcm_sha256'],
                   raw_feature_tensor_sha256=training.array_digest(arrays[r['id']])) for r in all_rows]
    _write_new(fit, 'raw-feature-inputs.jsonl', actual, jsonl=True)
    differences = [dict(index=i, expected=left, actual=right)
                   for i, (left, right) in enumerate(zip_longest(golden, actual)) if left != right]
    norm = training.train_normalization([r for r in rows if r['kind'] == 'human'], arrays)
    _write_new(fit, 'recomputed-normalization.json', norm)
    comparison = dict(exact=not differences and norm == expected_norm,
        feature_rows=len(actual), golden_rows=len(golden), differences=differences,
        normalization_equal=norm == expected_norm,
        normalization_different_fields=sorted(k for k in set(norm) | set(expected_norm)
                                            if norm.get(k) != expected_norm.get(k)),
        extraction='canonical_extract_arrays_BLAS1', fitting_splits=['train'],
        normalization_split='human_train_only', optimizer_steps_before_gate=0,
        heldout_pcm_decoded_or_inferred_before_selection=False)
    _write_new(fit, 'input-comparison.json', comparison)
    if not comparison['exact']:
        raise ValueError('Exact feature/normalization preflight failed; evidence preserved')
    return arrays, norm


def _compare_checkpoints(actual, reference, snapshot):
    """Tensor identity is exact; numerical differences are diagnostics, not gates."""
    import torch

    tensors = {}
    for name in sorted(set(actual['state_dict']) | set(reference['state_dict'])):
        left, right = actual['state_dict'].get(name), reference['state_dict'].get(name)
        compatible = (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                      and left.shape == right.shape)
        finite = bool(compatible and torch.isfinite(left).all() and torch.isfinite(right).all())
        difference = float((left.double() - right.double()).abs().max()) if finite else None
        tensors[name] = dict(equal=bool(finite and left.dtype == right.dtype and torch.equal(left, right)),
            max_abs_difference=difference,
            actual_shape=list(left.shape) if isinstance(left, torch.Tensor) else None,
            expected_shape=list(right.shape) if isinstance(right, torch.Tensor) else None,
            actual_dtype=str(left.dtype) if isinstance(left, torch.Tensor) else None,
            expected_dtype=str(right.dtype) if isinstance(right, torch.Tensor) else None)
    epoch_equal = actual['epoch'] == reference['epoch'] == snapshot['selected_epoch']
    config_equal = actual['config'] == reference['config'] == snapshot['training']
    counts_equal = actual['sample_counts'] == reference['sample_counts']
    state_equal = bool(tensors) and all(t['equal'] for t in tensors.values())
    differences = {key: dict(actual=actual['sample_counts'].get(key), expected=reference['sample_counts'].get(key))
                   for key in sorted(set(actual['sample_counts']) | set(reference['sample_counts']))
                   if actual['sample_counts'].get(key) != reference['sample_counts'].get(key)}
    return dict(exact=epoch_equal and config_equal and counts_equal and state_equal,
        selected_epoch_equal=epoch_equal, actual_epoch=actual['epoch'], expected_epoch=reference['epoch'],
        config_equal=config_equal, sample_counts_equal=counts_equal, sample_count_differences=differences,
        selected_draws=sum(actual['sample_counts'].values()), expected_selected_draws=sum(reference['sample_counts'].values()),
        state_dict_equal=state_equal, tensors=tensors,
        max_abs_tensor_difference=max((t['max_abs_difference'] for t in tensors.values()
                                       if t['max_abs_difference'] is not None), default=None))


def _fit_and_compare(root, fit, views, arrays, norm, snapshot):
    import torch
    from copy import deepcopy
    from scripts.submission.optionb_reproduce import pinned_path
    from scripts.train import speaker_optionb_tiny as training

    if torch.get_default_device().type != 'cpu' or torch.get_default_dtype() != torch.float32:
        raise ValueError('Canonical fitting requires untouched CPU/FP32 Torch defaults')
    _fit_files(fit)
    for name in ('best.pt', 'human-foundation.pt', 'normalization.json', 'training-curve.jsonl', 'training-result.json'):
        _fit_target(fit, name)
    config = deepcopy(snapshot['training'])
    # This is the only optimizer entry point. No recipe/device/thread overrides.
    result = training.fit_model(views['train'], views['dev'], arrays, fit, config)
    actual = torch.load(_fit_target(fit, 'best.pt'), map_location='cpu', weights_only=True)
    reference = torch.load(pinned_path(root, snapshot['checkpoint']), map_location='cpu', weights_only=True)
    checkpoint = _compare_checkpoints(actual, reference, snapshot)
    history = read_jsonl(_fit_target(fit, 'training-curve.jsonl'))
    human, mixed, steps = config['human_epochs'], config['mixed_epochs'], config['steps_per_epoch']
    expected_draws = dict(human_positive=(24 * human + 12 * mixed) * steps,
                          human_negative=8 * (human + mixed) * steps,
                          synthetic_positive=12 * mixed * steps)
    members = {r['id']: r for r in views['train']}
    draws = dict.fromkeys(expected_draws, 0)
    valid_draws = True
    for identifier, count in result['sample_counts'].items():
        row = members.get(identifier)
        if row is None or type(count) is not int or count <= 0:
            valid_draws = False
            continue
        group = 'synthetic_positive' if row['kind'] == 'synthetic' else (
            'human_positive' if row['class_index'] else 'human_negative')
        draws[group] += count
    cpu_only = all(t.device.type == 'cpu' for t in actual['state_dict'].values())
    checks = dict(checkpoint_exact=checkpoint['exact'],
        full_optimizer_steps=result['optimizer_steps'] == (human + mixed) * steps,
        exact_training_config=result['training_config'] == snapshot['training'],
        result_file_equal=read_json(_fit_target(fit, 'training-result.json')) == result,
        full_epoch_history=[e['epoch'] for e in history] == list(range(1, human + mixed + 1)),
        original_phases=[e['phase'] for e in history] == ['human'] * human + ['mixed'] * mixed,
        normalization_equal=read_json(_fit_target(fit, 'normalization.json')) == norm,
        final_draw_counts_equal=valid_draws and draws == expected_draws,
        selected_epoch_equal=result['selected_epoch'] == actual['epoch'] == snapshot['selected_epoch'],
        train_only=result['fitting_splits'] == ['train'] and result['selected_on'] == 'dev',
        normalization_ids_equal=result['normalization_ids'] == norm['ids'],
        parameters_equal=result['parameters'] == snapshot['parameters'],
        weights_changed=result['weights_changed'] is True,
        cpu_only=cpu_only, threads_equal=torch.get_num_threads() == config['threads'])
    comparison = dict(exact=all(checks.values()), checks=checks, checkpoint=checkpoint,
        optimizer_steps=result['optimizer_steps'], expected_optimizer_steps=(human + mixed) * steps,
        final_draw_counts=draws, expected_final_draw_counts=expected_draws,
        cpu_only=cpu_only, torch_threads=torch.get_num_threads())
    _write_new(fit, 'training-comparison.json', comparison)
    return result, actual, comparison


def _selection_contract(root):
    """Recover the existing grid and helper hashes, without legacy cache access."""
    from scripts.submission.optionb_reproduce import pinned_path

    study = 'experiments/speaker-optionb-tiny-demo-v1/'
    lock = read_json(pinned_path(root, study + 'selection-before-test.json'))
    freeze = read_json(pinned_path(root, study + 'data-freeze.json'))
    expected = {**freeze['historical_bindings'], **lock['bindings']}
    targets = [study + 'protocol.json', 'scripts/test/test_speaker_optionb_tiny.py',
               'scripts/test/speaker_optionb_tiny_demo.py', 'scripts/test/speaker_optionb_tiny_ft_audit.py']
    bindings = {}
    for name in targets:
        # One historical writer bound __file__ absolutely. Match only this exact
        # checkout-relative suffix; never consume its old machine/cache path.
        digests = {value for path, value in expected.items()
                   if path.replace('\\', '/') == name or path.replace('\\', '/').endswith('/' + name)}
        if len(digests) != 1 or file_digest(pinned_path(root, name)) not in digests:
            raise ValueError('Frozen export/selection source changed: ' + name)
        bindings[name] = next(iter(digests))
    for name in (study + 'selection-before-test.json', study + 'data-freeze.json'):
        bindings[name] = file_digest(pinned_path(root, name))
    return read_json(pinned_path(root, study + 'protocol.json'))['selection'], bindings


def _export_dependency_preflight(root):
    """Import the canonical exporter before any feature extraction or fitting."""
    import sys
    from scripts.submission.optionb_reproduce import pinned_path

    requirements = 'submission/optionb-tiny/testing.lock'
    try:
        from scripts.test.test_speaker_optionb_tiny import export_cpu_model
    except ImportError as exc:
        raise RuntimeError(
            f'Canonical CPU exporter import failed ({type(exc).__name__}: {exc}). '
            'Required export/test helpers are separate from the frozen scientific lock. '
            'From checkout root, install the checked-in hashed helper requirements into this same environment: '
            f'uv pip install --python "{sys.executable}" --require-hashes --no-deps --requirement {requirements}'
        ) from exc
    return dict(exporter=export_cpu_model.__module__ + '.' + export_cpu_model.__name__,
        import_verified=True, checked_before_feature_extraction=True, optimizer_steps=0,
        helper_requirements=requirements, helper_requirements_sha256=file_digest(pinned_path(root, requirements)))


def _export_and_select(root, fit, views, arrays, norm, checkpoint, snapshot):
    import numpy as np
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.submission.optionb_reproduce import pinned_path
    from scripts.train import speaker_optionb_tiny as training

    config, source_bindings = _selection_contract(root)
    # The original exporter actually lives beside the original model tests, not
    # in the architecture module. Reuse that exact implementation, including its
    # opset, dynamic axes and warning capture; do not copy it into this stage.
    from scripts.test.test_speaker_optionb_tiny import export_cpu_model
    from scripts.test.speaker_optionb_tiny_demo import select_demo_policy

    torch.set_num_threads(1)  # Original CPU export/parity recipe; fitting used four.
    model = OptionBTinyVCM().eval().cpu()
    model.load_state_dict(checkpoint['state_dict'])
    _fit_files(fit)
    bundle = _fit_target(fit, 'bundle')
    bundle.mkdir()
    exported = export_cpu_model(model, _fit_target(fit, 'bundle/model.onnx'))
    frozen_bundle = pinned_path(root, snapshot['bundle'])
    frozen_norm = read_json(frozen_bundle / 'normalization.json')
    if any(norm[key] != frozen_norm[key] for key in ('mean', 'std')):
        raise ValueError('Refuse to export changed normalization semantics')
    for name in ('commands.json', 'normalization.json', 'policy.json'):
        with _fit_target(fit, 'bundle/' + name).open('xb') as stream:
            stream.write((frozen_bundle / name).read_bytes())
    recipe = read_json(frozen_bundle / 'recipe.json')
    recipe.update(binding={name: file_digest(bundle / name) for name in
                           ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')},
                  selected_epoch=checkpoint['epoch'], actions_enabled=False, qualified=False,
                  model_qualified=False, installed=False, research_only=True,
                  reproduction_comparison_only=True,
                  parent_recipe_sha256=file_digest(frozen_bundle / 'recipe.json'))
    _write_new(fit, 'bundle/recipe.json', recipe)
    own, frozen = Runtime(bundle, threads=1), Runtime(frozen_bundle, threads=1)
    rows = [r for split in ('dev', 'calibration') for r in views[split] if r['kind'] == 'human']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected unique human DEV/calibration for fresh scoring')
    calibration = [r for r in views['calibration'] if r['kind'] == 'human']
    fresh = {**arrays, **training.extract_arrays(calibration)}
    predictions, scores = [], {}
    own_errors, frozen_errors = [], []
    own_passed, frozen_passed = True, True
    for row in rows:
        values = training.normalized(fresh[row['id']], norm)[None]
        lengths = np.asarray([values.shape[-1]], dtype=np.int64)
        with torch.inference_mode():
            expected = model(torch.from_numpy(values), torch.from_numpy(lengths)).numpy()[0]
        inputs = {'features': values, 'lengths': lengths}
        actual = own.session.run(['logits'], inputs)[0][0]
        tracked = frozen.session.run(['logits'], inputs)[0][0]
        finite = bool(np.isfinite(expected).all() and np.isfinite(actual).all() and np.isfinite(tracked).all())
        if not finite:
            raise ValueError('Nonfinite fresh DEV/calibration inference: ' + row['id'])
        own_equal = bool(np.allclose(actual, expected, atol=3e-5, rtol=3e-5)
                         and np.argmax(actual) == np.argmax(expected))
        frozen_equal = bool(np.allclose(actual, tracked, atol=3e-5, rtol=3e-5)
                            and np.argmax(actual) == np.argmax(tracked))
        own_passed &= own_equal
        frozen_passed &= frozen_equal
        own_errors.append(float(np.max(np.abs(actual - expected))))
        frozen_errors.append(float(np.max(np.abs(actual - tracked))))
        scores[row['id']] = actual.tolist()
        predictions.append(dict(id=row['id'], split=row['split'], kind=row['kind'],
            class_index=row['class_index'], audio_sha256=row['sha256'], pcm_sha256=row['pcm_sha256'],
            raw_feature_tensor_sha256=training.array_digest(fresh[row['id']]), logits=actual.tolist(),
            checkpoint_logits=expected.tolist(), tracked_graph_logits=tracked.tolist(),
            checkpoint_parity=own_equal, tracked_graph_parity=frozen_equal))
    _write_new(fit, 'calibration-predictions.jsonl', predictions, jsonl=True)
    selection = select_demo_policy(rows, scores, config)
    selection.update(cached_scores_reused=False, source='fresh_source_WAV_PCM_canonical_BLAS1_features',
                     test_holdout_used_for_selection=False)
    _write_new(fit, 'policy-selection.json', selection)
    parameters = sum(p.numel() for p in model.parameters())
    graph_digest = file_digest(bundle / 'model.onnx')
    expected_digest = snapshot['files'][snapshot['bundle'] + '/model.onnx']['sha256']
    checkpoint_parity = dict(passed=own_passed, rows=len(rows), max_abs_logit_difference=max(own_errors),
                             atol=3e-5, rtol=3e-5, class_equality_required=True)
    graph_parity = dict(passed=frozen_passed, rows=len(rows), max_abs_logit_difference=max(frozen_errors),
                        atol=3e-5, rtol=3e-5, class_equality_required=True)
    policy_equal = selection['policy'] == snapshot['policy'] == read_json(bundle / 'policy.json')
    comparison = dict(exact=own_passed and frozen_passed and policy_equal and graph_digest == expected_digest
                            and parameters == snapshot['parameters'] == exported['initializer_elements'],
        checkpoint_parity=checkpoint_parity, tracked_graph_parity=graph_parity,
        graph_sha256=graph_digest, frozen_graph_sha256=expected_digest,
        graph_bytes_equal=graph_digest == expected_digest, parameters=parameters,
        selected_policy=selection['policy'], frozen_policy=snapshot['policy'], policy_equal=policy_equal,
        policy_grid_points=len(selection['grid']), export=exported,
        selected_on=['dev', 'calibration'], cached_scores_reused=False,
        actions_enabled=False, qualified=False, installed=False)
    _write_new(fit, 'export-comparison.json', comparison)
    return comparison, source_bindings


def _assert_bindings(root, bindings):
    from scripts.submission.optionb_reproduce import pinned_path
    changed = [name for name, expected in bindings.items()
               if not pinned_path(root, name).is_file() or file_digest(pinned_path(root, name)) != expected]
    if changed:
        raise ValueError('Consumed input/output identity changed: ' + ', '.join(changed[:8]))


def fit_source(root, output):
    """Fit prepared .cache/... inputs; require locked science + hashed export/test helpers."""
    from datetime import datetime, timezone
    import platform
    import time
    from scripts.submission.optionb_reproduce import SNAPSHOT, pinned_path, read_snapshot_rows, verify_snapshot

    root = Path(root).resolve()
    run = pinned_path(root, output)
    parts = run.relative_to(root).parts
    if len(parts) < 2 or parts[0] != '.cache':
        raise ValueError('Fit outputs must be a named run under .cache/')
    if run != root / output:
        raise ValueError('Run containment changed by a symlink/junction')
    fit = _fit_target(run, 'fit')
    if fit.exists():
        raise FileExistsError('Preserve the earlier fit; choose a new run ID')
    fit.mkdir(parents=True)
    started = time.perf_counter()
    stage = 'snapshot_verification'
    try:
        snapshot = verify_snapshot(root)
        protected = {name: record['sha256'] for name, record in snapshot['files'].items()}
        protected.update(snapshot['source_bindings'])
        protected[SNAPSHOT] = file_digest(pinned_path(root, SNAPSHOT))
        for name in ('scripts/submission/optionb_fit.py', 'scripts/submission/optionb_reproduce.py',
                     'scripts/test/test_optionb_reproduction_fit.py', 'scripts/test/speaker_demo_suite.py',
                     'scripts/train/speaker_human_data.py'):
            protected[name] = file_digest(pinned_path(root, name))
        stage = 'prepared_inputs'
        for split in ('train', 'dev', 'calibration'):
            path = _fit_target(run, f'prepared/{split}.jsonl')
            protected[path.relative_to(root).as_posix()] = file_digest(path)
        frozen = {split: read_snapshot_rows(root, snapshot['views'][split])
                  for split in ('train', 'dev', 'calibration')}
        views = {split: _prepared_rows(root, run, split, rows) for split, rows in frozen.items()}
        preparation_path = _fit_target(run, 'prepare-train-dev-calibration.json')
        if preparation_path.exists():
            preparation = read_json(preparation_path)
            if (preparation.get('snapshot_sha256') != protected[SNAPSHOT]
                    or preparation.get('wav_and_pcm_verified') is not True
                    or preparation.get('old_audio_or_features_copied') is not False
                    or preparation.get('derived_feature_cache_reused') is not False
                    or preparation.get('heldout_pcm_decoded') is not False):
                raise ValueError('Source preparation receipt does not describe this source-only run')
            for name, digest in preparation['bindings'].items():
                if name in protected and protected[name] != digest:
                    raise ValueError('Conflicting independently protected preparation binding: ' + name)
                protected[name] = digest
            protected[preparation_path.relative_to(root).as_posix()] = file_digest(preparation_path)
        stage = 'training_admission'
        from scripts.train import speaker_optionb_tiny as training
        reserved = {**{s: frozen[s] for s in ('dev', 'calibration')},
                    **{s: read_snapshot_rows(root, snapshot['views'][s]) for s in ('test', 'holdout')}}
        training.training_view(views['train'], [r for rows in reserved.values() for r in rows])
        _write_new(fit, 'training-admission.json', dict(
            training_rows=len(views['train']), reserved_metadata_counts={s: len(r) for s, r in reserved.items()},
            canonical_training_view_checked=True, reserved_pcm_decoded=False,
            fitting_splits=['train'], checkpoint_selection_split='dev', normalization_split='human_train_only',
            inherited_history='Frozen historical gap-fill exposures and synthetic voice-alias caveats are not erased'))
        _, selection_sources = _selection_contract(root)
        protected.update(selection_sources)
        _assert_bindings(root, protected)
        stage = 'export_dependency_preflight'
        export_prerequisite = _export_dependency_preflight(root)
        protected[export_prerequisite['helper_requirements']] = export_prerequisite['helper_requirements_sha256']
        _write_new(fit, 'export-dependency-preflight.json', export_prerequisite)
        stage = 'feature_normalization_preflight'
        golden = read_snapshot_rows(root, snapshot['raw_feature_inputs'])
        arrays, norm = _feature_preflight(views['train'], views['dev'], golden,
            read_json(pinned_path(root, snapshot['training_normalization'])), fit)
        _assert_bindings(root, protected)
        stage = 'canonical_cpu_fit'
        result, checkpoint, fit_comparison = _fit_and_compare(root, fit, views, arrays, norm, snapshot)
        stage = 'cpu_export_and_fresh_demo_selection'
        export_comparison, export_sources = _export_and_select(root, fit, views, arrays, norm, checkpoint, snapshot)
        if export_sources != selection_sources:
            raise ValueError('Export/selection sources changed during fitting')
        stage = 'exact_comparison_and_lock'
        verify_snapshot(root)
        _assert_bindings(root, protected)
        scope = ('software_fixture_not_corpus_reproduction' if snapshot.get('software_fixture_not_corpus_evidence')
                 else 'source_only_frozen_corpus_fit_comparison')
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        comparison = dict(format='optionb-source-fit-comparison-v1',
            exact=fit_comparison['exact'] and export_comparison['exact'],
            evidence_scope=scope, training=fit_comparison, export=export_comparison,
            input_comparison_sha256=file_digest(fit / 'input-comparison.json'),
            snapshot_sha256=protected[SNAPSHOT], protected_inputs_unchanged=True,
            feature_rows=len(golden), human_normalization_rows=len(norm['ids']),
            optimizer_steps=result['optimizer_steps'], original_training_config=snapshot['training'],
            created_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
            environment=dict(python=platform.python_version(), platform=platform.platform(),
                             torch=torch.__version__, numpy=np.__version__, onnx=onnx.__version__,
                             onnxruntime=ort.__version__, fit_torch_threads=fit_comparison['torch_threads'],
                             export_torch_threads=torch.get_num_threads(), frontend_blas_threads=1),
            heldout_metadata_read_for_training_admission=True,
            heldout_pcm_decoded_or_inferred_before_selection=False,
            test_holdout_used_for_selection=False, exposed_regression_not_blind=True,
            actions_enabled=False, qualified=False, installed=False)
        _write_new(fit, 'fit-comparison.json', comparison)
        if not comparison['exact']:
            raise ValueError('Fit/export/policy comparison mismatch; preserve this run, never replace the snapshot')
        bindings = dict(protected)
        for path in _fit_files(fit):
            relative = path.relative_to(root).as_posix()
            bindings[relative] = file_digest(pinned_path(root, relative))
        lock = dict(format='optionb-source-fit-selection-before-test-v1', status='selection_locked',
            evidence_scope=scope, bindings=bindings, snapshot_sha256=protected[SNAPSHOT],
            selected_epoch=checkpoint['epoch'], policy=snapshot['policy'], parameters=snapshot['parameters'],
            optimizer_steps=result['optimizer_steps'], fitting_splits=['train'],
            checkpoint_selected_on='dev', selected_on=['dev', 'calibration'],
            normalization_split='original_human_train_only', feature_rows=len(golden),
            source_preparation_receipt=preparation_path.relative_to(root).as_posix() if preparation_path.exists() else None,
            export_dependency_preflight=(fit / 'export-dependency-preflight.json').relative_to(root).as_posix(),
            fit_comparison=(fit / 'fit-comparison.json').relative_to(root).as_posix(),
            test_holdout_used_for_selection=False, cached_scores_reused=False,
            heldout_metadata_read_for_training_admission=True,
            heldout_pcm_decoded_or_inferred_before_selection=False,
            exposed_regression_not_blind=True, installed_defaults_changed=False,
            actions_enabled=False, qualified=False, installed=False,
            created_utc=datetime.now(timezone.utc).isoformat())
        _assert_bindings(root, bindings)
        _write_new(fit, 'selection-before-test.json', lock)
        if read_json(_fit_target(fit, 'selection-before-test.json')) != lock:
            raise ValueError('Selection receipt readback mismatch')
        _assert_bindings(root, bindings)
        return lock
    except BaseException as exc:
        preserved = {path.relative_to(root).as_posix(): file_digest(path) for path in _fit_files(fit)}
        comparison_path = _fit_target(fit, 'fit-comparison.json')
        _write_new(fit, 'failure.json', dict(stage=stage, error_type=type(exc).__name__, error=str(exc),
            bindings=preserved,
            comparison_receipt=comparison_path.relative_to(root).as_posix() if comparison_path.exists() else None,
            selection_lock_written=_fit_target(fit, 'selection-before-test.json').exists(),
            actions_enabled=False, qualified=False, installed=False,
            heldout_pcm_decoded_or_inferred_before_selection=False))
        raise


def main(argv=None, *, root=None):
    import argparse
    import json
    from scripts.submission.common import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.cache/optionb-source-repro-v1/run-01',
                        help='prepared source run under .cache/; fit/ must not already exist')
    arguments = parser.parse_args(argv)
    receipt = fit_source(ROOT if root is None else root, arguments.output)
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
    return receipt


if __name__ == '__main__':
    main()
