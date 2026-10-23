"""Epoch44 publication stays opt-in and never substitutes GPU/CPU identities."""
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def implementation():
    try:
        return importlib.import_module('scripts.optionb_epoch44')
    except ModuleNotFoundError as error:
        if error.name != 'scripts.optionb_epoch44':
            raise
        raise AssertionError('The opt-in epoch44 verification command is missing') from error


def copy_publication(destination):
    shutil.copytree(ROOT / 'submission', destination / 'submission')
    return destination


def test_original_epoch44_payload_verification_is_distinct_from_training():
    report = implementation().verify_frozen(ROOT)
    assert report['status'] == 'frozen_epoch44_verified'
    assert report['verified_payloads'] == 15
    assert report['selected_epoch'] == 44
    assert report['optimizer_updates'] == 0
    assert report['inference_checked'] is False
    assert report['original_CPU_model_retrained'] is False
    assert report['actions_enabled'] is False and report['qualified'] is False


def test_changed_original_graph_fails_before_loading(tmp_path):
    module = implementation()
    root = copy_publication(tmp_path / 'publication')
    (root / 'submission/models/optionb-tiny/model.onnx').write_bytes(b'not the original graph')
    with pytest.raises(ValueError, match='identity'):
        module.verify_frozen(root)


def test_rehashed_mutable_snapshot_cannot_approve_substituted_model(tmp_path):
    module = implementation()
    root = copy_publication(tmp_path / 'publication')
    name = 'submission/models/optionb-tiny/model.onnx'
    path = root / name
    path.write_bytes(b'substituted graph')
    snapshot = root / 'submission/optionb-tiny/snapshot.json'
    data = json.loads(snapshot.read_text())
    data['files'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'bytes': path.stat().st_size}
    snapshot.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='snapshot identity'):
        module.verify_frozen(root)


@pytest.mark.parametrize('relative', ['../outside', '/absolute', 'C:/outside', 'a/../../outside'])
def test_publication_paths_cannot_escape_root(tmp_path, relative):
    with pytest.raises(ValueError, match='path'):
        implementation().checked_path(tmp_path, relative)


def test_optional_load_executes_exact_graph_without_capture_or_training():
    report = implementation().verify_frozen(ROOT, load=True)
    assert report['inference_checked'] is True
    assert report['output_shape'] == [1, 32]
    assert report['providers'] == ['CPUExecutionProvider']
    assert report['optimizer_updates'] == 0 and report['actions_enabled'] is False
    assert report['acoustic_quality_claimed'] is False


def test_main_epoch44_command_emits_separate_frozen_artifact_result():
    process = subprocess.run([sys.executable, '-m', 'scripts.optionb_epoch44', 'verify'],
                             cwd=ROOT, capture_output=True, text=True, check=True)
    report = json.loads(process.stdout)
    assert report['status'] == 'frozen_epoch44_verified'
    assert report['verified_payloads'] == 15
    assert report['original_CPU_model_retrained'] is False


def gpu_fixture(tmp_path, monkeypatch):
    """Stored-evidence software fixture, never represented as actual training."""
    module = implementation()
    publication = copy_publication(tmp_path / 'publication')
    source = tmp_path / 'source'
    fit = source / '.cache/optionb-source-repro-v1/run-01/fit'
    fit.mkdir(parents=True)
    reference_path = publication / 'submission/optionb-tiny/gpu-reference.json'
    reference = json.loads(reference_path.read_text())
    for name in reference['required_failure_bindings']:
        path = fit / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('software fixture: ' + name).encode())
    curve = [{'epoch': index + 1, 'loss': 0.5, 'elapsed_seconds': index + 0.1}
             for index in range(48)]
    placement = {**{key: ['cuda:0'] for key in (
        'model_devices', 'input_devices', 'gradient_devices', 'optimizer_moment_devices')},
        'target_device': 'cuda:0', 'optimizer_step_devices': ['cpu']}
    probe = dict(placement, passed=True, finite_loss=True, synchronized=True,
                 weights_changed=True, optimizer_steps=1)
    selection = dict(requested='cuda', device='cuda:0', reason='cuda_probe_passed',
                     cuda_device_name='software fixture (not actual CUDA hardware)', probe=probe)
    placement = dict(placement, dev_model_devices=['cuda:0'], dev_input_devices=['cuda:0'],
                     dev_output_device='cuda:0')
    result = {'execution_device': 'cuda:0', 'optimizer_steps': 4608,
              'selected_epoch': 28, 'sample_counts': {'software_fixture': 147456},
              'device_selection': selection, 'device_placement': placement,
              'checkpoint_tensor_device': 'cpu', 'weights_changed': True}
    checks: dict[str, bool] = dict.fromkeys(('device_provenance_equal', 'execution_device_equal',
        'execution_placement_equal', 'requested_device_equal', 'requested_device_honored',
        'working_device_probe', 'checkpoint_handoff_cpu', 'weights_changed', 'cpu_only'), True)
    checks.update(checkpoint_exact=False, selected_epoch_equal=False)
    comparison = dict(execution_device='cuda:0', device_selection=selection,
        device_placement=placement, checkpoint_tensor_device='cpu', cpu_only=True,
        cpu_only_describes='loaded_checkpoint_tensors_not_optimizer_execution', exact=False,
        checks=checks)
    values = {'training-device.json': selection, 'training-result.json': result,
              'training-comparison.json': comparison,
              'input-comparison.json': {'exact': True, 'normalization_equal': True,
                                        'feature_rows': 24566},
              'policy-selection.json': {'policy': {'min_probability': 0.55, 'min_margin': 0.3},
                                        'grid': [{} for _ in range(138)]}}
    for name, value in values.items():
        (fit / name).write_text(json.dumps(value))
    (fit / 'training-curve.jsonl').write_text('\n'.join(json.dumps(row) for row in curve))
    (fit / 'calibration-predictions.jsonl').write_text('\n'.join(
        json.dumps({'id': 'software_fixture_' + str(index)}) for index in range(466)))
    for key in ('artifact_sha256', 'additional_artifact_sha256'):
        reference[key] = {name: module.digest(fit / name) for name in reference[key]}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    reference['epoch_curve_without_elapsed_seconds_sha256'] = hashlib.sha256(canonical(
        [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in curve])).hexdigest()
    reference['final_sample_counts_sha256'] = hashlib.sha256(canonical(result['sample_counts'])).hexdigest()
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(module, 'GPU_REFERENCE_SHA256', module.digest(reference_path), raising=False)
    monkeypatch.setattr(module, 'verify_source_checkout', lambda root: None, raising=False)
    failure = {'stage': 'exact_comparison_and_lock', 'selection_lock_written': False,
               'bindings': {path.relative_to(source).as_posix(): module.digest(path)
                            for path in fit.rglob('*') if path.is_file()}}
    (fit / 'failure.json').write_text(json.dumps(failure))
    return module, publication, source, fit


def test_GPU_reference_pass_is_not_original_CPU_retraining(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    report = module.verify_gpu_fit(source, fit, root=publication)
    assert report['status'] == 'GPU_repeatability_verified'
    assert report['selected_GPU_epoch'] == 28
    assert report['original_CPU_epoch44_exact'] is False
    assert report['original_success_lock_written'] is False
    assert report['actions_enabled'] is False
    result = json.loads((fit / 'training-result.json').read_text())
    comparison = json.loads((fit / 'training-comparison.json').read_text())
    assert result['device_placement']['optimizer_step_devices'] == ['cpu']
    assert result['checkpoint_tensor_device'] == 'cpu'
    assert comparison['cpu_only'] is comparison['checks']['cpu_only'] is True
    assert comparison['checks']['checkpoint_exact'] is False
    assert comparison['checks']['selected_epoch_equal'] is False
    assert not (fit / 'selection-before-test.json').exists()


def rebind_gpu_json(module, source, fit, name, value):
    """Model the reported attack: update mutable evidence and its failure binding."""
    path = fit / name
    path.write_text(json.dumps(value))
    failure_path = fit / 'failure.json'
    failure = json.loads(failure_path.read_text())
    failure['bindings'][path.relative_to(source).as_posix()] = module.digest(path)
    failure_path.write_text(json.dumps(failure))


def test_rebound_CPU_model_placement_cannot_pass_GPU_CLI(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    result = json.loads((fit / 'training-result.json').read_text())
    result['device_placement']['model_devices'] = ['cpu']
    rebind_gpu_json(module, source, fit, 'training-result.json', result)
    with pytest.raises(ValueError, match='GPU.*placement'):
        module.main(['verify-gpu', '--source-checkout', str(source), '--fit', str(fit)],
                    root=publication)


@pytest.mark.parametrize('field,value', [
    (field, value)
    for field in ('model_devices', 'input_devices', 'gradient_devices', 'optimizer_moment_devices')
    for value in (Ellipsis, None, [], ['cpu'], 'cuda:0')
] + [
    ('target_device', value) for value in (Ellipsis, None, 'cpu', ['cuda:0'])
] + [
    ('optimizer_step_devices', value) for value in (Ellipsis, None, [], 'cpu')
])
def test_rebound_matching_invalid_training_placements_reject(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    for name in ('training-result.json', 'training-comparison.json'):
        record = json.loads((fit / name).read_text())
        if value is Ellipsis:
            record['device_placement'].pop(field)
        else:
            record['device_placement'][field] = value
        rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*placement'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('field,value', [
    (field, value) for field in ('dev_model_devices', 'dev_input_devices')
    for value in (Ellipsis, None, [], ['cpu'])
] + [('dev_output_device', value) for value in (Ellipsis, None, 'cpu', ['cuda:0'])])
def test_rebound_matching_invalid_DEV_scoring_placements_reject(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    for name in ('training-result.json', 'training-comparison.json'):
        record = json.loads((fit / name).read_text())
        if value is Ellipsis:
            record['device_placement'].pop(field)
        else:
            record['device_placement'][field] = value
        rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*placement'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('name', ['training-result.json', 'training-comparison.json'])
@pytest.mark.parametrize('value', [Ellipsis, None, []])
def test_rebound_missing_placement_record_rejects(tmp_path, monkeypatch, name, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    record = json.loads((fit / name).read_text())
    if value is Ellipsis:
        record.pop('device_placement')
    else:
        record['device_placement'] = value
    rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*placement'):
        module.verify_gpu_fit(source, fit, root=publication)


def rebind_gpu_selection(module, source, fit, selection):
    rebind_gpu_json(module, source, fit, 'training-device.json', selection)
    for name in ('training-result.json', 'training-comparison.json'):
        record = json.loads((fit / name).read_text())
        record['device_selection'] = selection
        rebind_gpu_json(module, source, fit, name, record)


@pytest.mark.parametrize('name', ['training-device.json', 'training-result.json', 'training-comparison.json'])
@pytest.mark.parametrize('field,value', [
    ('requested', 'auto'), ('device', 'cuda:1'), ('reason', 'cuda_preflight_failed'),
    ('cuda_device_name', 'contradictory software fixture'),
])
def test_rebound_cross_file_device_selection_disagreement_rejects(tmp_path, monkeypatch, name, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    record = json.loads((fit / name).read_text())
    selection = record if name == 'training-device.json' else record['device_selection']
    selection[field] = value
    rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*provenance'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('name', ['training-device.json', 'training-result.json', 'training-comparison.json'])
@pytest.mark.parametrize('value', [Ellipsis, None, []])
def test_rebound_missing_device_selection_rejects(tmp_path, monkeypatch, name, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    record = json.loads((fit / name).read_text())
    if name == 'training-device.json':
        record = {} if value is Ellipsis else value
    elif value is Ellipsis:
        record.pop('device_selection')
    else:
        record['device_selection'] = value
    rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*provenance'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('field,value', [
    (field, value) for field in ('requested', 'device', 'reason', 'cuda_device_name')
    for value in (Ellipsis, None, '')
] + [('requested', 'cpu'), ('device', 'cpu'), ('reason', 'cuda_preflight_failed')])
def test_rebound_matching_invalid_device_selection_rejects(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    selection = json.loads((fit / 'training-device.json').read_text())
    if value is Ellipsis:
        selection.pop(field)
    else:
        selection[field] = value
    rebind_gpu_selection(module, source, fit, selection)
    with pytest.raises(ValueError, match='GPU.*provenance'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_coherent_auto_selected_CUDA_keeps_CPU_export_distinct(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    selection = json.loads((fit / 'training-device.json').read_text())
    selection['requested'] = 'auto'
    rebind_gpu_selection(module, source, fit, selection)
    assert module.verify_gpu_fit(source, fit, root=publication)['original_CPU_epoch44_exact'] is False


@pytest.mark.parametrize('field,value', [
    (field, value) for field in ('passed', 'finite_loss', 'synchronized', 'weights_changed')
    for value in (Ellipsis, None, False, 1, 'true')
] + [('optimizer_steps', value) for value in (Ellipsis, None, 0, True, 1.0, '1', 2)])
def test_rebound_matching_failed_or_malformed_compute_probe_rejects(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    selection = json.loads((fit / 'training-device.json').read_text())
    if value is Ellipsis:
        selection['probe'].pop(field)
    else:
        selection['probe'][field] = value
    rebind_gpu_selection(module, source, fit, selection)
    with pytest.raises(ValueError, match='GPU.*probe'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('field,value', [
    (field, value)
    for field in ('model_devices', 'input_devices', 'gradient_devices', 'optimizer_moment_devices')
    for value in (Ellipsis, None, [], ['cpu'], 'cuda:0')
] + [('target_device', value) for value in (Ellipsis, None, 'cpu', ['cuda:0'])]
  + [('optimizer_step_devices', value) for value in (Ellipsis, None, [], 'cpu')])
def test_rebound_matching_invalid_probe_placement_rejects(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    selection = json.loads((fit / 'training-device.json').read_text())
    if value is Ellipsis:
        selection['probe'].pop(field)
    else:
        selection['probe'][field] = value
    rebind_gpu_selection(module, source, fit, selection)
    with pytest.raises(ValueError, match='GPU.*placement'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('value', [Ellipsis, None, []])
def test_rebound_missing_compute_probe_rejects(tmp_path, monkeypatch, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    selection = json.loads((fit / 'training-device.json').read_text())
    if value is Ellipsis:
        selection.pop('probe')
    else:
        selection['probe'] = value
    rebind_gpu_selection(module, source, fit, selection)
    with pytest.raises(ValueError, match='GPU.*probe'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('name', ['training-result.json', 'training-comparison.json'])
@pytest.mark.parametrize('field,value', [('passed', 1), ('optimizer_steps', True)])
def test_rebound_probe_cross_file_type_disagreement_rejects(tmp_path, monkeypatch, name, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    record = json.loads((fit / name).read_text())
    record['device_selection']['probe'][field] = value
    rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*(provenance|probe)'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('field', [
    'device_provenance_equal', 'execution_device_equal', 'execution_placement_equal',
    'requested_device_equal', 'requested_device_honored', 'working_device_probe',
    'checkpoint_handoff_cpu', 'weights_changed', 'cpu_only',
])
@pytest.mark.parametrize('value', [Ellipsis, None, False, 1])
def test_rebound_stale_or_malformed_GPU_comparison_check_rejects(tmp_path, monkeypatch, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    comparison = json.loads((fit / 'training-comparison.json').read_text())
    if value is Ellipsis:
        comparison['checks'].pop(field)
    else:
        comparison['checks'][field] = value
    rebind_gpu_json(module, source, fit, 'training-comparison.json', comparison)
    with pytest.raises(ValueError, match='GPU.*checks'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('value', [Ellipsis, None, []])
def test_rebound_missing_GPU_comparison_checks_rejects(tmp_path, monkeypatch, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    comparison = json.loads((fit / 'training-comparison.json').read_text())
    if value is Ellipsis:
        comparison.pop('checks')
    else:
        comparison['checks'] = value
    rebind_gpu_json(module, source, fit, 'training-comparison.json', comparison)
    with pytest.raises(ValueError, match='GPU.*checks'):
        module.verify_gpu_fit(source, fit, root=publication)


@pytest.mark.parametrize('name,field,value', [
    ('training-comparison.json', 'execution_device', value) for value in (Ellipsis, None, 'cpu')
] + [
    (name, 'checkpoint_tensor_device', value)
    for name in ('training-result.json', 'training-comparison.json')
    for value in (Ellipsis, None, 'cuda:0')
] + [
    ('training-comparison.json', 'cpu_only', value) for value in (Ellipsis, None, False, 1)
] + [
    ('training-comparison.json', 'cpu_only_describes', value)
    for value in (Ellipsis, None, 'optimizer_execution')
] + [
    ('training-result.json', 'weights_changed', value) for value in (Ellipsis, None, False, 1)
])
def test_rebound_contradictory_GPU_comparison_facts_reject(tmp_path, monkeypatch, name, field, value):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    record = json.loads((fit / name).read_text())
    if value is Ellipsis:
        record.pop(field)
    else:
        record[field] = value
    rebind_gpu_json(module, source, fit, name, record)
    with pytest.raises(ValueError, match='GPU.*provenance'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_rebound_substituted_GPU_checkpoint_still_fails_independent_reference(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    (fit / 'best.pt').write_bytes(b'substituted GPU checkpoint')
    failure_path = fit / 'failure.json'
    failure = json.loads(failure_path.read_text())
    failure['bindings'][(fit / 'best.pt').relative_to(source).as_posix()] = module.digest(fit / 'best.pt')
    failure_path.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match='GPU reference'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_failed_training_stage_cannot_be_relabelled_GPU_success(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    failure_path = fit / 'failure.json'
    failure = json.loads(failure_path.read_text())
    failure['stage'] = 'training'
    failure_path.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match='stage'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_Iris_checkout_is_not_misidentified_as_the_original_training_source():
    with pytest.raises(ValueError, match='source revision'):
        implementation().verify_source_checkout(ROOT)


def test_GPU_verification_CLI_has_its_own_explicit_scope(tmp_path, monkeypatch, capsys):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    assert module.main(['verify-gpu', '--source-checkout', str(source), '--fit', str(fit)],
                       root=publication) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'GPU_repeatability_verified'
    assert report['original_CPU_epoch44_exact'] is False


@pytest.mark.parametrize('field,value,message', [
    ('optimizer_steps', 4607, 'optimizer'),
    ('execution_device', 'cpu', 'CUDA'),
    ('selected_epoch', 44, 'selected epoch'),
])
def test_rebound_incomplete_or_CPU_training_cannot_pass(tmp_path, monkeypatch, field, value, message):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    path = fit / 'training-result.json'
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    failure_path = fit / 'failure.json'
    failure = json.loads(failure_path.read_text())
    failure['bindings'][path.relative_to(source).as_posix()] = module.digest(path)
    failure_path.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match=message):
        module.verify_gpu_fit(source, fit, root=publication)


def test_missing_required_evidence_binding_rejects(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    failure_path = fit / 'failure.json'
    failure = json.loads(failure_path.read_text())
    failure['bindings'].pop(next(iter(failure['bindings'])))
    failure_path.write_text(json.dumps(failure))
    with pytest.raises(ValueError, match='bindings'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_forged_original_selection_lock_cannot_pass_GPU_verification(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    (fit / 'selection-before-test.json').write_text('{}')
    with pytest.raises(ValueError, match='success lock'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_mutable_GPU_reference_cannot_approve_new_weights(tmp_path, monkeypatch):
    module, publication, source, fit = gpu_fixture(tmp_path, monkeypatch)
    path = publication / 'submission/optionb-tiny/gpu-reference.json'
    data = json.loads(path.read_text())
    data['artifact_sha256']['best.pt'] = '0' * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='reference identity'):
        module.verify_gpu_fit(source, fit, root=publication)


def test_original_frozen_bytes_are_not_subject_to_Git_EOL_conversion():
    names = ['submission/models/optionb-tiny/recipe.json',
             'submission/optionb-tiny/snapshot.json', 'submission/optionb-tiny/gpu-reference.json']
    output = subprocess.check_output(['git', 'check-attr', 'text', '--', *names], cwd=ROOT, text=True)
    assert all(line.endswith(': unset') for line in output.splitlines())


def test_artifact_verification_supports_Python311_without_junction_API(monkeypatch):
    module = implementation()
    for cls in type(ROOT).__mro__:
        if 'is_junction' in cls.__dict__:
            monkeypatch.delattr(cls, 'is_junction')
    assert module.verify_frozen(ROOT)['status'] == 'frozen_epoch44_verified'
