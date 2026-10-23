"""Source-fit guards and bounded real CPU fixtures, not corpus-fit evidence."""
from copy import deepcopy
import gzip
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = ROOT / 'submission/optionb-tiny/snapshot.json'


def module():
    assert importlib.util.find_spec('scripts.submission.optionb_fit') is not None, (
        'Missing additive source-only OptionB fit stage')
    return importlib.import_module('scripts.submission.optionb_fit')


def test_fit_source_preserves_an_existing_fit_even_when_inputs_are_missing(tmp_path):
    fit = tmp_path / '.cache/fixture/fit'
    fit.mkdir(parents=True)
    evidence = fit / 'failed-earlier.json'
    evidence.write_bytes(b'{"fixture":true}\n')
    with pytest.raises(FileExistsError, match='Preserve'):
        module().fit_source(tmp_path, '.cache/fixture')
    assert evidence.read_bytes() == b'{"fixture":true}\n'
    assert sorted(p.name for p in fit.iterdir()) == ['failed-earlier.json']


def frozen_rows(split):
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    with gzip.open(ROOT / snapshot['views'][split]['path'], 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream]


@pytest.mark.parametrize('change', [None, 'label', 'order', 'extra', 'missing', 'duplicate', 'split',
                                  'audio_escape', 'audio_noncanonical', 'scalar_type'])
def test_prepared_metadata_is_the_exact_frozen_order_with_only_audio_added(tmp_path, change):
    """Two actual frozen metadata rows; no corpus PCM or fitted result."""
    expected = frozen_rows('train')[:2]
    run = tmp_path / '.cache/fixture'
    prepared = run / 'prepared'
    prepared.mkdir(parents=True)
    rows = [{**r, 'audio': f".cache/fixture/audio/{r['sha256']}.wav"} for r in expected]
    if change == 'label':
        rows[0]['expected'] = None
    elif change == 'order':
        rows.reverse()
    elif change == 'extra':
        rows[0]['features_sha256'] = 'not-tiny-training-features'
    elif change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif change == 'split':
        rows[0]['split'] = 'holdout'
    elif change == 'audio_escape':
        rows[0]['audio'] = '../outside.wav'
    elif change == 'audio_noncanonical':
        rows[0]['audio'] = '.cache/fixture/not-a-hydrated-source.wav'
    elif change == 'scalar_type':
        rows[0]['class_index'] = float(rows[0]['class_index'])
    (prepared / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf8')
    stage = module()
    assert hasattr(stage, '_prepared_rows'), 'Missing exact prepared-view guard'
    if change:
        with pytest.raises(ValueError):
            stage._prepared_rows(tmp_path, run, 'train', expected)
    else:
        actual = stage._prepared_rows(tmp_path, run, 'train', expected)
        assert [{k: v for k, v in r.items() if k != 'audio'} for r in actual] == expected
        assert all(Path(r['audio']).is_absolute() for r in actual)


def real_audio_fixture(tmp_path):
    """New tones with real WAV/PCM/features: software plumbing, not speech gold."""
    import hashlib
    import numpy as np
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest
    from scripts.train import speaker_optionb_tiny as training

    rows = []
    specs = [('train', 'human', True), ('train', 'human', False), ('train', 'synthetic', True),
             ('dev', 'human', True), ('dev', 'human', False), ('dev', 'synthetic', True),
             ('calibration', 'human', True), ('calibration', 'human', False)]
    for i, (split, kind, positive) in enumerate(specs):
        path = tmp_path / f'software-tone-{i}.wav'
        sf.write(path, np.sin(np.arange(1600 + i * 400) * (.03 + i * .007)) * .2,
                 16000, subtype='PCM_16')
        pcm, rate = sf.read(path, dtype='float32')
        rows.append(training.projected_row(dict(
            id=f'software-fixture/{i}', split=split, kind=kind, synthetic=kind == 'synthetic',
            speaker=f'{kind}-{split}', training_eligible=split == 'train', metric_eligible=True,
            expected=dict(intent='WEATHER', slots={}) if positive else None,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), pcm_sha256=pcm_digest(pcm, rate),
            audio=str(path))))
    views = {split: [r for r in rows if r['split'] == split] for split in ('train', 'dev', 'calibration')}
    arrays = training.extract_arrays([*views['train'], *views['dev']])
    golden = [dict(id=r['id'], split=r['split'], kind=r['kind'], audio_sha256=r['sha256'],
                   pcm_sha256=r['pcm_sha256'], raw_feature_tensor_sha256=training.array_digest(arrays[r['id']]))
              for r in [*views['train'], *views['dev']]]
    norm = training.train_normalization([r for r in views['train'] if r['kind'] == 'human'], arrays)
    return views, arrays, golden, norm


@pytest.mark.parametrize('change', [None, 'id', 'kind', 'split', 'audio_sha256', 'pcm_sha256',
                                  'raw_feature_tensor_sha256', 'order', 'missing', 'normalization_ids',
                                  'normalization_hashes', 'normalization_mean'])
def test_feature_preflight_preserves_strict_tensor_and_normalization_evidence(tmp_path, change):
    views, _, golden, norm = real_audio_fixture(tmp_path)
    if change in {'id', 'kind', 'split', 'audio_sha256', 'pcm_sha256', 'raw_feature_tensor_sha256'}:
        golden[0][change] = 'deliberate-fixture-mismatch'
    elif change == 'order':
        golden.reverse()
    elif change == 'missing':
        golden.pop()
    elif change == 'normalization_ids':
        norm['ids'].pop()
    elif change == 'normalization_hashes':
        norm['raw_feature_tensor_sha256'][views['train'][0]['id']] = 'wrong'
    elif change == 'normalization_mean':
        norm['mean'][0] += .000001
    fit = tmp_path / 'fit'
    fit.mkdir()
    stage = module()
    assert hasattr(stage, '_feature_preflight'), 'Missing pre-optimizer exact-feature gate'
    if change:
        with pytest.raises(ValueError, match='preflight'):
            stage._feature_preflight(views['train'], views['dev'], golden, norm, fit)
    else:
        arrays, actual_norm = stage._feature_preflight(views['train'], views['dev'], golden, norm, fit)
        assert actual_norm == norm and set(arrays) == {r['id'] for r in golden}
    audit = json.loads((fit / 'input-comparison.json').read_text())
    assert audit['exact'] is (change is None)
    assert audit['feature_rows'] == 6
    assert audit['normalization_equal'] is (not str(change).startswith('normalization_'))
    assert (fit / 'raw-feature-inputs.jsonl').is_file()
    assert (fit / 'recomputed-normalization.json').is_file()
    assert not (fit / 'best.pt').exists()


@pytest.mark.parametrize('change', [None, 'epoch', 'config', 'draws', 'tensor', 'shape', 'dtype', 'missing_tensor'])
def test_checkpoint_comparison_is_exact_against_the_tracked_epoch44_tensors(change):
    import torch
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    reference = torch.load(ROOT / snapshot['checkpoint'], map_location='cpu', weights_only=True)
    actual = deepcopy(reference)
    name = next(iter(actual['state_dict']))
    if change == 'epoch':
        actual['epoch'] -= 1
    elif change == 'config':
        actual['config']['threads'] = 1
    elif change == 'draws':
        actual['sample_counts'][next(iter(actual['sample_counts']))] += 1
    elif change == 'tensor':
        actual['state_dict'][name].view(-1)[0] += .125
    elif change == 'shape':
        actual['state_dict'][name] = actual['state_dict'][name].flatten()
    elif change == 'dtype':
        actual['state_dict'][name] = actual['state_dict'][name].double()
    elif change == 'missing_tensor':
        del actual['state_dict'][name]
    stage = module()
    assert hasattr(stage, '_compare_checkpoints'), 'Missing selected-checkpoint tensor comparison'
    comparison = stage._compare_checkpoints(actual, reference, snapshot)
    assert comparison['exact'] is (change is None)
    assert comparison['selected_epoch_equal'] is (change != 'epoch')
    assert comparison['sample_counts_equal'] is (change != 'draws')
    assert comparison['state_dict_equal'] is (change not in {'tensor', 'shape', 'dtype', 'missing_tensor'})
    if change == 'tensor':
        assert comparison['max_abs_tensor_difference'] == pytest.approx(.125, abs=1e-7)
        assert comparison['tensors'][name]['equal'] is False
    if change is None:
        assert comparison['max_abs_tensor_difference'] == 0.
        assert comparison['selected_draws'] == sum(reference['sample_counts'].values())


def test_two_step_real_cpu_optimizer_fixture_matches_an_independent_canonical_fit(tmp_path):
    """Two+two actual optimizer steps on tones, NOT the 48-epoch corpus job."""
    from scripts.train import speaker_optionb_tiny as training
    views, arrays, _, norm = real_audio_fixture(tmp_path)
    frozen = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    config = {**frozen['training'], 'human_epochs': 1, 'mixed_epochs': 1, 'steps_per_epoch': 1}
    baseline = tmp_path / 'fixture-baseline'
    expected = training.fit_model(views['train'], views['dev'], arrays, baseline, config)
    fixture_snapshot = {**frozen, 'training': config, 'checkpoint': 'fixture-baseline/best.pt',
                        'selected_epoch': expected['selected_epoch']}
    fit = tmp_path / 'fixture-replay'
    fit.mkdir()
    stage = module()
    assert hasattr(stage, '_fit_and_compare'), 'Missing unchanged canonical CPU fitting stage'
    result, checkpoint, comparison = stage._fit_and_compare(tmp_path, fit, views, arrays, norm, fixture_snapshot, device='cpu')
    assert result['optimizer_steps'] == 2 and result['weights_changed'] is True
    assert checkpoint['config'] == config
    assert comparison['exact'] is True
    assert comparison['checkpoint']['state_dict_equal'] is True
    assert comparison['final_draw_counts'] == {'human_positive': 36, 'human_negative': 16, 'synthetic_positive': 12}
    assert comparison['cpu_only'] is True and comparison['torch_threads'] == 4
    assert (fit / 'training-comparison.json').is_file()
    assert (fit / 'training-curve.jsonl').is_file()
    assert not (fit / 'selection-before-test.json').exists()


def test_selection_contract_uses_the_original_hash_bound_138_point_demo_grid():
    stage = module()
    assert hasattr(stage, '_selection_contract'), 'Missing frozen demo-grid source binding'
    config, bindings = stage._selection_contract(ROOT)
    assert config == json.loads((ROOT / 'experiments/speaker-optionb-tiny-demo-v1/protocol.json').read_text())['selection']
    assert len(config['probability_grid']) * len(config['margin_grid']) == 138
    assert .5 in config['probability_grid'] and .1 in config['margin_grid']
    assert {'scripts/test/test_speaker_optionb_tiny.py', 'scripts/test/speaker_optionb_tiny_demo.py',
            'scripts/test/speaker_optionb_tiny_ft_audit.py'} <= bindings.keys()


def test_real_tracked_checkpoint_export_has_separate_parities_and_fresh_tone_policy_grid(tmp_path):
    """Real checkpoint/ONNX on four tones: export plumbing, not speech quality."""
    import torch
    from scripts.runtime.speaker_optionb_tiny import Runtime
    from scripts.test.speaker_optionb_tiny_demo import select_demo_policy
    frozen = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    checkpoint = torch.load(ROOT / frozen['checkpoint'], map_location='cpu', weights_only=True)
    views, arrays, _, _ = real_audio_fixture(tmp_path)
    norm = json.loads((ROOT / frozen['training_normalization']).read_text())
    fit = tmp_path / 'fit'
    fit.mkdir()
    stage = module()
    assert hasattr(stage, '_export_and_select'), 'Missing original CPU exporter/fresh policy comparison'
    comparison, bindings = stage._export_and_select(ROOT, fit, views, arrays, norm, checkpoint, frozen)
    assert comparison['checkpoint_parity']['passed'] is True
    assert comparison['tracked_graph_parity']['passed'] is True
    assert comparison['checkpoint_parity']['rows'] == 4
    assert comparison['tracked_graph_parity']['rows'] == 4
    assert comparison['parameters'] == 258912
    assert comparison['graph_sha256']
    assert comparison['frozen_graph_sha256'] == frozen['files'][frozen['bundle'] + '/model.onnx']['sha256']
    rows = [r for split in ('dev', 'calibration') for r in views[split] if r['kind'] == 'human']
    predictions = [json.loads(line) for line in (fit / 'calibration-predictions.jsonl').read_text().splitlines()]
    config, _ = stage._selection_contract(ROOT)
    expected_selection = select_demo_policy(rows, {r['id']: r['logits'] for r in predictions}, config)
    assert comparison['selected_policy'] == expected_selection['policy']
    assert comparison['policy_equal'] is (expected_selection['policy'] == frozen['policy'])
    selection = json.loads((fit / 'policy-selection.json').read_text())
    assert len(selection['grid']) == 138 and selection['cached_scores_reused'] is False
    assert selection['selected_on'] == ['dev', 'calibration']
    for name in ('commands.json', 'normalization.json', 'policy.json'):
        assert (fit / 'bundle' / name).read_bytes() == (ROOT / frozen['bundle'] / name).read_bytes()
    runtime = Runtime(fit / 'bundle')
    assert runtime.recipe['actions_enabled'] is False and runtime.recipe['qualified'] is False
    assert 'scripts/test/test_speaker_optionb_tiny.py' in bindings
    assert not (fit / 'selection-before-test.json').exists()


def snapshot_checkout(tmp_path):
    """Copy only tracked immutable snapshot/source metadata, never PCM/caches."""
    import shutil
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding='utf8'))
    names = {SNAPSHOT_PATH.relative_to(ROOT).as_posix(), *snapshot['files'], *snapshot['source_bindings'],
             'scripts/submission/optionb_fit.py', 'scripts/submission/optionb_reproduce.py',
             'scripts/test/speaker_demo_suite.py', 'scripts/train/speaker_human_data.py',
             'scripts/test/test_optionb_reproduction_fit.py', 'scripts/test/test_speaker_optionb_tiny.py',
             'scripts/train/speaker_optionb_tiny_accelerated.py',
             'scripts/test/test_speaker_optionb_tiny_accelerated.py',
             'scripts/test/speaker_optionb_tiny_demo.py', 'scripts/test/speaker_optionb_tiny_ft_audit.py',
             'submission/optionb-tiny/testing.lock',
             'experiments/speaker-optionb-tiny-demo-v1/protocol.json',
             'experiments/speaker-optionb-tiny-demo-v1/data-freeze.json',
             'experiments/speaker-optionb-tiny-demo-v1/selection-before-test.json'}
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return snapshot


@pytest.mark.parametrize('output', ['../outside', 'submission/new-run', 'artifacts/new-run',
                                  'experiments/old-study', 'scripts/new-run', '.cache'])
def test_fit_source_rejects_non_run_paths_before_any_write(tmp_path, output):
    with pytest.raises(ValueError):
        module().fit_source(tmp_path, output)
    assert list(tmp_path.iterdir()) == []


def test_fit_source_seals_missing_preparation_failure_without_calling_the_optimizer(tmp_path, monkeypatch):
    from scripts.train import speaker_optionb_tiny as training
    snapshot_checkout(tmp_path)
    calls = []
    def forbidden_fit(*args, **kwargs):
        calls.append(True)
        pytest.fail('Optimizer must not run before exact source preflight')
    monkeypatch.setattr(training, 'fit_model', forbidden_fit)
    with pytest.raises(FileNotFoundError):
        module().fit_source(tmp_path, '.cache/fixture')
    failure = json.loads((tmp_path / '.cache/fixture/fit/failure.json').read_text())
    assert failure['stage'] == 'prepared_inputs'
    assert failure['error_type'] == 'FileNotFoundError'
    assert failure['actions_enabled'] is False
    assert calls == []
    assert not (tmp_path / '.cache/fixture/fit/selection-before-test.json').exists()
    with pytest.raises(FileExistsError, match='Preserve'):
        module().fit_source(tmp_path, '.cache/fixture')


def source_fixture_checkout(tmp_path):
    """Explicit software snapshot of real tones; original recipe/checkpoint stay pinned."""
    import hashlib
    snapshot = snapshot_checkout(tmp_path)
    run = tmp_path / '.cache/fixture'
    audio = run / 'audio'
    audio.mkdir(parents=True)
    views, arrays, golden, norm = real_audio_fixture(audio)
    for rows in views.values():
        for row in rows:
            path = Path(row['audio'])
            target = audio / (row['sha256'] + '.wav')
            path.rename(target)
            row['audio'] = str(target)
    changed = []
    for split, rows in views.items():
        record = snapshot['views'][split]
        path = tmp_path / record['path']
        with gzip.open(path, 'wt', encoding='utf8') as stream:
            for row in rows:
                stream.write(json.dumps({k: v for k, v in row.items() if k != 'audio'}) + '\n')
        record.update(rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        prepared = run / 'prepared' / f'{split}.jsonl'
        prepared.parent.mkdir(exist_ok=True)
        prepared.write_text(''.join(json.dumps({**r, 'audio': Path(r['audio']).relative_to(tmp_path).as_posix()})
                                    + '\n' for r in rows), encoding='utf8')
        changed.append(record['path'])
    record = snapshot['raw_feature_inputs']
    path = tmp_path / record['path']
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        stream.write(''.join(json.dumps(r) + '\n' for r in golden))
    record.update(rows=len(golden), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    changed.append(record['path'])
    norm_path = tmp_path / snapshot['training_normalization']
    norm_path.write_text(json.dumps(norm), encoding='utf8')
    changed.append(snapshot['training_normalization'])
    for name in changed:
        path = tmp_path / name
        snapshot['files'][name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    snapshot['software_fixture_not_corpus_evidence'] = True
    (tmp_path / SNAPSHOT_PATH.relative_to(ROOT)).write_text(json.dumps(snapshot), encoding='utf8')
    return snapshot, views, arrays, norm


def test_missing_pytest_export_dependency_fails_in_a_fresh_process_before_features_or_optimizer(tmp_path):
    """The pytest runner must not conceal the canonical exporter's import need."""
    import subprocess
    import sys
    import textwrap

    source_fixture_checkout(tmp_path)
    result = subprocess.run([sys.executable, '-c', textwrap.dedent('''
        import importlib.abc
        from pathlib import Path
        import sys
        from scripts.train import speaker_optionb_tiny as training
        from scripts.submission.optionb_fit import fit_source

        assert 'pytest' not in sys.modules
        assert 'scripts.test.test_speaker_optionb_tiny' not in sys.modules
        class BlockPytest(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'pytest' or fullname.startswith('pytest.'):
                    raise ModuleNotFoundError('pytest blocked by subprocess regression', name=fullname)
        sys.meta_path.insert(0, BlockPytest())
        root = Path(sys.argv[1])
        def forbidden_features(*args, **kwargs):
            (root / 'features.marker').write_text('unexpected feature extraction')
            raise AssertionError('Features ran without required export helpers')
        def forbidden_optimizer(*args, **kwargs):
            (root / 'optimizer.marker').write_text('unexpected optimizer entry')
            raise AssertionError('Optimizer ran without required export helpers')
        training.extract_arrays = forbidden_features
        training.fit_model = forbidden_optimizer
        fit_source(root, '.cache/fixture')
    '''), str(tmp_path)], cwd=ROOT, check=False, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0, result.stdout + result.stderr
    fit = tmp_path / '.cache/fixture/fit'
    failure = json.loads((fit / 'failure.json').read_text())
    assert failure['stage'] == 'export_dependency_preflight', result.stdout + result.stderr
    assert failure['error_type'] == 'RuntimeError'
    assert 'pytest blocked by subprocess regression' in failure['error']
    assert ('uv pip install --python "' + sys.executable + '" --require-hashes --no-deps '
            '--requirement submission/optionb-tiny/testing.lock') in failure['error']
    assert failure['selection_lock_written'] is False
    assert failure['comparison_receipt'] is None
    assert not (tmp_path / 'features.marker').exists()
    assert not (tmp_path / 'optimizer.marker').exists()
    assert not list(fit.rglob('*.pt'))
    assert not (fit / 'raw-feature-inputs.jsonl').exists()
    assert not (fit / 'recomputed-normalization.json').exists()
    assert not (fit / 'export-dependency-preflight.json').exists()
    assert not (fit / 'selection-before-test.json').exists()
    assert '.cache/fixture/fit/training-admission.json' in failure['bindings']
    with pytest.raises(FileExistsError, match='Preserve'):
        module().fit_source(tmp_path, '.cache/fixture')


def test_fit_source_reaches_only_the_same_recipe_port_after_real_preflight_and_reserved_metadata_audit(tmp_path, monkeypatch):
    """Only the expensive full fit is stopped; WAV/PCM/extraction gates really run."""
    from scripts.train import speaker_optionb_tiny_accelerated as training
    from scripts.test.speaker_demo_suite import read_jsonl
    snapshot, views, _, norm = source_fixture_checkout(tmp_path)
    seen = []
    class FixtureOptimizerStop(RuntimeError):
        pass
    def stop_full_fit(rows, dev, arrays, output, config, *, device):
        assert device == 'auto'
        assert config == json.loads(SNAPSHOT_PATH.read_text())['training']
        assert {r['split'] for r in rows} == {'train'}
        assert {r['split'] for r in dev} == {'dev'}
        assert set(arrays) == {r['id'] for r in [*views['train'], *views['dev']]}
        assert json.loads((output / 'input-comparison.json').read_text())['exact'] is True
        assert json.loads((output / 'recomputed-normalization.json').read_text()) == norm
        assert len(read_jsonl(output / 'raw-feature-inputs.jsonl')) == 6
        admission = json.loads((output / 'training-admission.json').read_text())
        assert admission['reserved_metadata_counts']['test'] == snapshot['views']['test']['rows']
        assert admission['reserved_metadata_counts']['holdout'] == snapshot['views']['holdout']['rows']
        assert admission['reserved_pcm_decoded'] is False
        seen.append(config)
        raise FixtureOptimizerStop('Software fixture stopped before the 48-epoch job')
    monkeypatch.setattr(training, 'fit_model', stop_full_fit)
    with pytest.raises(FixtureOptimizerStop):
        module().fit_source(tmp_path, '.cache/fixture')
    assert len(seen) == 1
    failure = json.loads((tmp_path / '.cache/fixture/fit/failure.json').read_text())
    assert failure['stage'] == 'device_explicit_fit'
    assert not (tmp_path / '.cache/fixture/fit/selection-before-test.json').exists()


def completed_fixture_checkout(tmp_path):
    """A real two-step reference, not an invented corpus training result."""
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.test.test_speaker_optionb_tiny import export_cpu_model
    from scripts.test.speaker_optionb_tiny_demo import select_demo_policy
    from scripts.test.speaker_demo_suite import file_digest, write_new
    from scripts.train import speaker_optionb_tiny as training
    snapshot, views, arrays, norm = source_fixture_checkout(tmp_path)
    config = {**snapshot['training'], 'human_epochs': 1, 'mixed_epochs': 1, 'steps_per_epoch': 1}
    baseline = tmp_path / '.cache/software-fixture-reference'
    result = training.fit_model(views['train'], views['dev'], arrays, baseline, config)
    checkpoint = torch.load(baseline / 'best.pt', map_location='cpu', weights_only=True)
    model = OptionBTinyVCM().eval()
    model.load_state_dict(checkpoint['state_dict'])
    bundle = baseline / 'bundle'
    bundle.mkdir()
    export_cpu_model(model, bundle / 'model.onnx')
    write_new(bundle / 'normalization.json', dict(mean=norm['mean'], std=norm['std'],
              source_sha256=file_digest(tmp_path / snapshot['training_normalization'])))
    with (bundle / 'commands.json').open('xb') as stream:
        stream.write((ROOT / snapshot['bundle'] / 'commands.json').read_bytes())
    selection_rows = [r for s in ('dev', 'calibration') for r in views[s] if r['kind'] == 'human']
    selection_arrays = training.extract_arrays(selection_rows)
    logits = training.predict_logits(model, selection_rows, selection_arrays, norm)
    selection_config, _ = module()._selection_contract(tmp_path)
    selected = select_demo_policy(selection_rows, {r['id']: v.tolist() for r, v in zip(selection_rows, logits)}, selection_config)
    write_new(bundle / 'policy.json', selected['policy'])
    recipe = json.loads((ROOT / snapshot['bundle'] / 'recipe.json').read_text())
    recipe.update(binding={n: file_digest(bundle / n) for n in
                           ('model.onnx', 'commands.json', 'normalization.json', 'policy.json')},
                  selected_epoch=checkpoint['epoch'])
    write_new(bundle / 'recipe.json', recipe)
    snapshot.update(training=config, selected_epoch=result['selected_epoch'], policy=selected['policy'],
                    checkpoint=(baseline / 'best.pt').relative_to(tmp_path).as_posix(),
                    bundle=bundle.relative_to(tmp_path).as_posix())
    for path in [baseline / 'best.pt', *bundle.iterdir()]:
        snapshot['files'][path.relative_to(tmp_path).as_posix()] = dict(bytes=path.stat().st_size, sha256=file_digest(path))
    (tmp_path / SNAPSHOT_PATH.relative_to(ROOT)).write_text(json.dumps(snapshot), encoding='utf8')
    return snapshot


@pytest.mark.parametrize('entrypoint', ['fit_source', 'main'])
def test_fit_source_returns_a_hash_sealed_receipt_after_a_real_two_step_fixture_run(tmp_path, entrypoint):
    from scripts.test.speaker_demo_suite import file_digest
    snapshot = completed_fixture_checkout(tmp_path)
    stage = module()
    assert hasattr(stage, entrypoint), 'Missing source-fit command entry point'
    receipt = (stage.fit_source(tmp_path, '.cache/fixture', device='cpu') if entrypoint == 'fit_source'
               else stage.main(['--output', '.cache/fixture', '--device', 'cpu'], root=tmp_path))
    assert isinstance(receipt, dict), 'Missing completed fit/export/selection receipt'
    fit = tmp_path / '.cache/fixture/fit'
    assert receipt == json.loads((fit / 'selection-before-test.json').read_text())
    assert receipt['status'] == 'selection_locked'
    assert receipt['evidence_scope'] == 'software_fixture_not_corpus_reproduction'
    assert receipt['optimizer_steps'] == 2
    assert receipt['policy'] == snapshot['policy']
    assert receipt['heldout_pcm_decoded_or_inferred_before_selection'] is False
    assert receipt['qualified'] is False and receipt['actions_enabled'] is False
    comparison = json.loads((fit / 'fit-comparison.json').read_text())
    assert comparison['exact'] is True
    assert comparison['training']['checkpoint']['state_dict_equal'] is True
    assert comparison['export']['graph_bytes_equal'] is True
    for name, expected in receipt['bindings'].items():
        assert file_digest(tmp_path / name) == expected
    assert '.cache/fixture/fit/fit-comparison.json' in receipt['bindings']
    assert '.cache/fixture/prepared/train.jsonl' in receipt['bindings']
    prerequisite = json.loads((tmp_path / receipt['export_dependency_preflight']).read_text())
    assert prerequisite['exporter'] == 'scripts.test.test_speaker_optionb_tiny.export_cpu_model'
    assert prerequisite['import_verified'] is True
    assert prerequisite['checked_before_feature_extraction'] is True
    assert prerequisite['optimizer_steps'] == 0
    requirements = 'submission/optionb-tiny/testing.lock'
    assert prerequisite['helper_requirements'] == requirements
    assert prerequisite['helper_requirements_sha256'] == file_digest(tmp_path / requirements)
    assert receipt['bindings'][requirements] == prerequisite['helper_requirements_sha256']
    assert receipt['export_dependency_preflight'] in receipt['bindings']


def test_preparation_receipt_cannot_override_independently_bound_input_hashes(tmp_path):
    from scripts.test.speaker_demo_suite import file_digest
    source_fixture_checkout(tmp_path)
    run = tmp_path / '.cache/fixture'
    receipt = dict(snapshot_sha256=file_digest(tmp_path / SNAPSHOT_PATH.relative_to(ROOT)),
                   wav_and_pcm_verified=True, old_audio_or_features_copied=False,
                   derived_feature_cache_reused=False, heldout_pcm_decoded=False,
                   bindings={'.cache/fixture/prepared/train.jsonl': '0' * 64})
    (run / 'prepare-train-dev-calibration.json').write_text(json.dumps(receipt), encoding='utf8')
    with pytest.raises(ValueError):
        module().fit_source(tmp_path, '.cache/fixture')
    failure = json.loads((run / 'fit/failure.json').read_text())
    assert failure['stage'] == 'prepared_inputs'
    assert not (run / 'fit/training-admission.json').exists()
    assert not (run / 'fit/best.pt').exists()


def make_directory_link(link, target):
    """Exercise real filesystem links; Windows junctions need no admin token."""
    import os
    import subprocess
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            raise
        result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)],
                                check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr


def test_derived_fit_targets_reject_real_nested_link_or_junction_escapes(tmp_path):
    fit, outside = tmp_path / 'fit', tmp_path / 'outside'
    fit.mkdir()
    outside.mkdir()
    make_directory_link(fit / 'bundle', outside)
    stage = module()
    assert hasattr(stage, '_fit_target'), 'Missing per-artifact fit containment guard'
    assert stage._fit_target(fit, 'normalization.json') == fit / 'normalization.json'
    with pytest.raises(ValueError, match='containment'):
        stage._fit_target(fit, 'bundle/model.onnx')
    assert list(outside.iterdir()) == []


def test_canonical_fitter_cannot_follow_a_replaced_fit_directory(tmp_path, monkeypatch):
    from scripts.train import speaker_optionb_tiny as training
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    views, arrays, _, norm = real_audio_fixture(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    fit = tmp_path / 'fit'
    make_directory_link(fit, outside)
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Fit directory containment must be checked before canonical optimization')
    monkeypatch.setattr(training, 'fit_model', forbidden_fit)
    with pytest.raises(ValueError, match='containment'):
        module()._fit_and_compare(ROOT, fit, views, arrays, norm, snapshot)
    assert list(outside.iterdir()) == []


def test_real_fixture_draw_mismatch_is_sealed_failed_and_never_replaces_the_snapshot(tmp_path):
    """Deliberately corrupt only a temporary fixture reference's draw counter."""
    import torch
    from scripts.test.speaker_demo_suite import file_digest
    snapshot = completed_fixture_checkout(tmp_path)
    reference_path = tmp_path / snapshot['checkpoint']
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    identifier = next(iter(reference['sample_counts']))
    reference['sample_counts'][identifier] += 1
    torch.save(reference, reference_path)
    snapshot['files'][snapshot['checkpoint']] = dict(bytes=reference_path.stat().st_size, sha256=file_digest(reference_path))
    (tmp_path / SNAPSHOT_PATH.relative_to(ROOT)).write_text(json.dumps(snapshot), encoding='utf8')
    expected_reference_hash = file_digest(reference_path)
    with pytest.raises(ValueError, match='comparison mismatch'):
        module().fit_source(tmp_path, '.cache/fixture', device='cpu')
    fit = tmp_path / '.cache/fixture/fit'
    comparison = json.loads((fit / 'fit-comparison.json').read_text())
    assert comparison['exact'] is False
    assert comparison['training']['checkpoint']['state_dict_equal'] is True
    assert comparison['training']['checkpoint']['sample_counts_equal'] is False
    assert not (fit / 'selection-before-test.json').exists()
    assert file_digest(reference_path) == expected_reference_hash
    failure = json.loads((fit / 'failure.json').read_text())
    assert failure['comparison_receipt'] == '.cache/fixture/fit/fit-comparison.json'
    assert failure['selection_lock_written'] is False
    assert failure['bindings']['.cache/fixture/fit/fit-comparison.json'] == file_digest(fit / 'fit-comparison.json')
    assert '.cache/fixture/fit/best.pt' in failure['bindings']


def test_false_execution_placement_cannot_pass_an_otherwise_exact_cpu_fit(tmp_path, monkeypatch):
    """Adversarial SOFTWARE FIXTURE: corrupt provenance, not fitted tensor values."""
    from scripts.train import speaker_optionb_tiny as canonical
    from scripts.train import speaker_optionb_tiny_accelerated as accelerated
    views, arrays, _, norm = real_audio_fixture(tmp_path)
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    config = {**snapshot['training'], 'human_epochs': 1, 'mixed_epochs': 1, 'steps_per_epoch': 1}
    result = canonical.fit_model(views['train'], views['dev'], arrays, tmp_path / 'reference', config)
    snapshot.update(training=config, checkpoint='reference/best.pt', selected_epoch=result['selected_epoch'])
    original_fit = accelerated.fit_model

    def corrupt_placement(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        result['device_placement']['gradient_devices'] = ['meta']
        (Path(args[3]) / 'training-result.json').write_text(json.dumps(result), encoding='utf8')
        return result

    monkeypatch.setattr(accelerated, 'fit_model', corrupt_placement)
    fit = tmp_path / 'candidate'
    fit.mkdir()
    _, _, comparison = module()._fit_and_compare(tmp_path, fit, views, arrays, norm, snapshot, device='cpu')
    assert comparison['checkpoint']['exact'] is True
    assert comparison['exact'] is False
    assert comparison['checks']['execution_placement_equal'] is False


def test_required_cuda_unavailable_is_sealed_with_requested_device_before_recipe_steps(tmp_path, monkeypatch):
    import torch
    from scripts.train import speaker_optionb_tiny_accelerated as accelerated
    source_fixture_checkout(tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(accelerated, '_probe_device', lambda device: pytest.fail('No CPU probe for required CUDA'))
    with pytest.raises(RuntimeError, match='Required CUDA'):
        module().fit_source(tmp_path, '.cache/fixture', device='cuda', fit_name='fit-gpu-01')
    fit = tmp_path / '.cache/fixture/fit-gpu-01'
    failure = json.loads((fit / 'failure.json').read_text())
    assert failure['stage'] == 'device_explicit_fit'
    assert failure['requested_device'] == 'cuda' and failure['fit_name'] == 'fit-gpu-01'
    assert failure['selection_lock_written'] is False
    assert not (fit / 'best.pt').exists() and not (fit / 'training-curve.jsonl').exists()


def test_real_cuda_cpu_reference_mismatch_preserves_comparison_and_never_locks(tmp_path):
    """Conditional two-step GPU SOFTWARE FIXTURE with a deliberately bad draw reference."""
    import torch
    from scripts.test.speaker_demo_suite import file_digest
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; this GPU comparison fixture did not execute')
    snapshot = completed_fixture_checkout(tmp_path)
    reference_path = tmp_path / snapshot['checkpoint']
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    reference['sample_counts'][next(iter(reference['sample_counts']))] += 1
    torch.save(reference, reference_path)
    snapshot['files'][snapshot['checkpoint']] = dict(bytes=reference_path.stat().st_size, sha256=file_digest(reference_path))
    (tmp_path / SNAPSHOT_PATH.relative_to(ROOT)).write_text(json.dumps(snapshot), encoding='utf8')
    reference_hash = file_digest(reference_path)
    with pytest.raises(ValueError, match='comparison mismatch'):
        module().fit_source(tmp_path, '.cache/fixture', device='cuda', fit_name='fit-gpu-01')
    fit = tmp_path / '.cache/fixture/fit-gpu-01'
    comparison = json.loads((fit / 'fit-comparison.json').read_text())
    assert comparison['exact'] is False and comparison['execution_device'] == 'cuda:0'
    training = comparison['training']
    assert training['execution_device'] == 'cuda:0' and training['checkpoint_tensor_device'] == 'cpu'
    assert training['cpu_only'] is True  # Not a claim that the optimizer ran on CPU.
    assert training['device_placement']['optimizer_moment_devices'] == ['cuda:0']
    actual = torch.load(fit / 'best.pt', weights_only=True)
    assert all(value.device.type == 'cpu' for value in actual['state_dict'].values())
    independent = module()._compare_checkpoints(actual, reference, snapshot)
    assert independent == training['checkpoint']
    assert training['checkpoint']['sample_counts_equal'] is False
    assert comparison['export']['checkpoint_parity']['passed'] is True
    assert file_digest(reference_path) == reference_hash
    assert not (fit / 'selection-before-test.json').exists()
    failure = json.loads((fit / 'failure.json').read_text())
    assert failure['selection_lock_written'] is False
    assert failure['comparison_receipt'] == '.cache/fixture/fit-gpu-01/fit-comparison.json'
    assert failure['qualified'] is False and failure['actions_enabled'] is False and failure['installed'] is False


@pytest.mark.parametrize('fit_name', ['', '.', '..', '../escaped', 'fit/new', 'fit\\new',
                                     'C:escape', '/absolute', 'fit:stream', 'fit.', ' fit', 'fit ',
                                     'CON', 'nul.txt', 'COM1', 'LPT9', 'audio', 'prepared', 'x' * 65])
def test_fit_name_is_one_safe_nonreserved_directory_component_before_any_write(tmp_path, fit_name):
    with pytest.raises(ValueError, match='fit.name'):
        module().fit_source(tmp_path, '.cache/fixture', fit_name=fit_name, device='cpu')
    assert list(tmp_path.iterdir()) == []


def test_named_fit_rejects_a_real_symlink_or_junction_before_writing(tmp_path):
    run, outside = tmp_path / '.cache/fixture', tmp_path / 'outside'
    run.mkdir(parents=True)
    outside.mkdir()
    make_directory_link(run / 'fit-gpu-01', outside)
    with pytest.raises(ValueError, match='containment'):
        module().fit_source(tmp_path, '.cache/fixture', fit_name='fit-gpu-01', device='cpu')
    assert list(outside.iterdir()) == []
    assert not (run / 'fit').exists()


def test_named_cpu_fit_reuses_prepared_source_run_without_touching_the_interrupted_fit(tmp_path):
    """Real two-step tone fixture and original CPU exporter, not corpus evidence."""
    snapshot = completed_fixture_checkout(tmp_path)
    old_fit = tmp_path / '.cache/fixture/fit'
    old_fit.mkdir()
    marker = old_fit / 'interrupted.json'
    marker.write_bytes(b'{"software_fixture":true,"preserve":true}\n')
    receipt = module().main(['--output', '.cache/fixture', '--device', 'cpu',
                             '--fit-name', 'fit-gpu-01'], root=tmp_path)
    fit = tmp_path / '.cache/fixture/fit-gpu-01'
    assert marker.read_bytes() == b'{"software_fixture":true,"preserve":true}\n'
    assert sorted(p.name for p in old_fit.iterdir()) == ['interrupted.json']
    assert receipt == json.loads((fit / 'selection-before-test.json').read_text())
    assert receipt['execution_device'] == 'cpu' and receipt['checkpoint_tensor_device'] == 'cpu'
    comparison = json.loads((fit / 'fit-comparison.json').read_text())
    assert comparison['training']['execution_device'] == 'cpu'
    assert comparison['training']['checkpoint_tensor_device'] == 'cpu'
    assert comparison['training']['device_selection']['reason'] == 'explicit_cpu'
    assert comparison['exact'] is True and receipt['optimizer_steps'] == 2
    assert receipt['qualified'] is False and receipt['actions_enabled'] is False and receipt['installed'] is False
    assert '.cache/fixture/prepared/train.jsonl' in receipt['bindings']
    for name in ('scripts/train/speaker_optionb_tiny_accelerated.py',
                 'scripts/test/test_speaker_optionb_tiny_accelerated.py'):
        assert name in receipt['bindings']


@pytest.mark.parametrize('changed_default', ['float64', 'meta'])
def test_canonical_fitter_rejects_non_cpu_fp32_defaults_before_optimization(tmp_path, monkeypatch, changed_default):
    import torch
    from scripts.train import speaker_optionb_tiny as training
    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    views, arrays, _, norm = real_audio_fixture(tmp_path)
    fit = tmp_path / 'fit'
    fit.mkdir()
    def forbidden_fit(*args, **kwargs):
        pytest.fail('Unapproved numerical/device defaults must fail before optimization')
    monkeypatch.setattr(training, 'fit_model', forbidden_fit)
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    try:
        if changed_default == 'float64':
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_device('meta')
        with pytest.raises(ValueError, match='CPU/FP32'):
            module()._fit_and_compare(ROOT, fit, views, arrays, norm, snapshot)
    finally:
        torch.set_default_dtype(old_dtype)
        torch.set_default_device(old_device)
