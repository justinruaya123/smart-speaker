"""Bounded SOFTWARE FIXTURES only; no corpus-fit or GPU qualification claims."""
import importlib
import importlib.util
import json

import pytest


def module():
    name = 'scripts.train.speaker_optionb_tiny_accelerated'
    assert importlib.util.find_spec(name) is not None, 'Missing additive device-explicit trainer'
    return importlib.import_module(name)


def forbid_cuda(monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail('Explicit CPU execution must not call CUDA APIs')

    for name in ('is_available', 'manual_seed_all', 'synchronize', 'current_device',
                 'get_device_name', '_lazy_init', '_is_in_bad_fork', 'is_current_stream_capturing'):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_explicit_cpu_runs_a_real_forward_backward_optimizer_probe_without_cuda(monkeypatch):
    """Synthetic tensors only; an actual parameter must change on CPU."""
    stage = module()
    forbid_cuda(monkeypatch)
    selection = stage.select_device('cpu')
    assert selection['requested'] == 'cpu'
    assert selection['device'] == 'cpu'
    assert selection['reason'] == 'explicit_cpu'
    probe = selection['probe']
    assert probe['passed'] is True and probe['weights_changed'] is True
    assert probe['optimizer_steps'] == 1 and probe['finite_loss'] is True
    assert probe['model_devices'] == probe['input_devices'] == probe['gradient_devices'] == ['cpu']
    assert probe['target_device'] == 'cpu'
    assert probe['optimizer_moment_devices'] == ['cpu']


def test_cpu_probe_does_not_query_cuda_even_in_a_cuda_built_wheel(monkeypatch):
    import torch
    stage = module()
    forbid_cuda(monkeypatch)
    monkeypatch.setattr(torch.backends.cuda, 'is_built', lambda: True)
    assert stage.select_device('cpu')['probe']['weights_changed'] is True


def test_auto_unavailable_cuda_falls_back_to_a_real_cpu_weight_update(monkeypatch):
    import torch
    stage = module()
    forbid_cuda(monkeypatch)
    calls = []

    def unavailable():
        calls.append('availability')
        return False

    monkeypatch.setattr(torch.cuda, 'is_available', unavailable)
    selection = stage.select_device()
    assert calls == ['availability']
    assert selection['requested'] == 'auto' and selection['device'] == 'cpu'
    assert selection['reason'] == 'cuda_unavailable'
    assert selection['probe']['weights_changed'] is True
    assert selection['probe']['optimizer_moment_devices'] == ['cpu']


@pytest.mark.parametrize('requested', ['auto', 'cuda'])
def test_cuda_preflight_failure_is_recorded_or_required_cuda_fails_closed(monkeypatch, requested):
    """Mock only the missing hardware boundary; CPU fallback really computes."""
    import os
    import torch
    stage = module()
    original_probe = stage._probe_device
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def probe(device):
        calls.append(str(device))
        if str(device).startswith('cuda'):
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert torch.backends.fp32_precision == 'ieee'
            assert torch.backends.cuda.matmul.fp32_precision == 'ieee'
            assert torch.backends.cudnn.fp32_precision == 'ieee'
            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
            assert torch.are_deterministic_algorithms_enabled()
            raise RuntimeError('software fixture: unusable CUDA kernel')
        return original_probe(device)

    monkeypatch.setattr(stage, '_probe_device', probe)
    if requested == 'cuda':
        with pytest.raises(RuntimeError, match='Required CUDA.*unusable CUDA kernel'):
            stage.select_device(requested)
        assert calls == ['cuda:0']
    else:
        selection = stage.select_device(requested)
        assert calls == ['cuda:0', 'cpu']
        assert selection['device'] == 'cpu' and selection['reason'] == 'cuda_preflight_failed'
        assert 'unusable CUDA kernel' in selection['cuda_preflight_error']
        assert selection['probe']['weights_changed'] is True


def test_required_cuda_unavailable_never_probes_cpu(monkeypatch):
    import torch
    stage = module()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(stage, '_probe_device', lambda device: pytest.fail('No CPU fallback for required CUDA'))
    with pytest.raises(RuntimeError, match='Required CUDA.*unavailable'):
        stage.select_device('cuda')


@pytest.mark.parametrize('requested', ['auto', 'cuda'])
def test_working_cuda_selection_requires_real_original_model_optimizer_probe(requested):
    """Conditional real GPU SOFTWARE FIXTURE, never a fabricated CUDA success."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable in this test environment; actual GPU probe not executed')
    selection = module().select_device(requested)
    assert selection['device'] == 'cuda:0' and selection['reason'] == 'cuda_probe_passed'
    probe = selection['probe']
    assert probe['weights_changed'] is True and probe['synchronized'] is True
    assert probe['model_devices'] == probe['input_devices'] == probe['gradient_devices'] == ['cuda:0']
    assert probe['target_device'] == 'cuda:0'
    assert probe['optimizer_moment_devices'] == ['cuda:0']
    assert probe['optimizer_step_devices'] == ['cpu']  # AdamW's noncapturable scalar bookkeeping.


@pytest.mark.parametrize('requested', ['cpu', 'auto'])
def test_bounded_cpu_port_is_tensor_exact_through_the_original_cosine_schedule(tmp_path, monkeypatch, requested):
    """Five tone-fixture steps exercise both phases and the cosine endpoint."""
    import torch
    from scripts.train import speaker_optionb_tiny as canonical
    from scripts.test.test_optionb_reproduction_fit import SNAPSHOT_PATH, real_audio_fixture

    stage = module()
    assert hasattr(stage, 'fit_model'), 'Missing minimal same-recipe device-explicit loop'
    views, arrays, _, norm = real_audio_fixture(tmp_path)
    config = {**json.loads(SNAPSHOT_PATH.read_text())['training'],
              'human_epochs': 2, 'mixed_epochs': 3, 'steps_per_epoch': 1}
    expected_result = canonical.fit_model(views['train'], views['dev'], arrays, tmp_path / 'canonical', config)
    expected = torch.load(tmp_path / 'canonical/best.pt', map_location='cpu', weights_only=True)
    forbid_cuda(monkeypatch)
    monkeypatch.setattr(torch.backends.cuda, 'is_built', lambda: True)
    if requested == 'auto':
        # Availability is the sole permitted discovery call; no GPU compute/seed calls.
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    output = tmp_path / 'ported'
    actual_result = stage.fit_model(views['train'], views['dev'], arrays, output, config, device=requested)
    actual = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(actual['state_dict'][name], value) for name, value in expected['state_dict'].items())
    assert actual['epoch'] == expected['epoch'] and actual['config'] == expected['config'] == config
    assert actual['sample_counts'] == expected['sample_counts']
    for key in expected_result.keys() - {'elapsed_seconds'}:
        assert actual_result[key] == expected_result[key], key
    assert actual_result['execution_device'] == 'cpu'
    assert actual_result['checkpoint_tensor_device'] == 'cpu'
    assert actual_result['optimizer_steps'] == 5 and actual_result['weights_changed'] is True
    assert json.loads((output / 'normalization.json').read_text()) == norm
    placement = actual_result['device_placement']
    assert placement['model_devices'] == placement['input_devices'] == placement['gradient_devices'] == ['cpu']
    assert placement['target_device'] == 'cpu' and placement['optimizer_moment_devices'] == ['cpu']
    assert placement['dev_input_devices'] == placement['dev_model_devices'] == ['cpu']
    assert placement['dev_output_device'] == 'cpu'
    assert json.loads((output / 'training-device.json').read_text()) == actual_result['device_selection']
    assert json.loads((output / 'training-result.json').read_text()) == actual_result
    histories = [[{k: v for k, v in json.loads(line).items() if k != 'elapsed_seconds'}
                  for line in (tmp_path / directory / 'training-curve.jsonl').read_text().splitlines()]
                 for directory in ('canonical', 'ported')]
    assert histories[0] == histories[1]


def test_real_cuda_fit_places_training_dev_and_optimizer_on_gpu_and_hands_off_cpu_checkpoint(tmp_path, monkeypatch):
    """Conditional two-step GPU SOFTWARE FIXTURE, with live tensor/gradient spies."""
    import torch
    from scripts.test.test_optionb_reproduction_fit import SNAPSHOT_PATH, real_audio_fixture
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.test.test_speaker_optionb_tiny import export_cpu_model

    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; no GPU fitting evidence produced')
    stage = module()
    views, arrays, _, _ = real_audio_fixture(tmp_path)
    config = {**json.loads(SNAPSHOT_PATH.read_text())['training'],
              'human_epochs': 1, 'mixed_epochs': 1, 'steps_per_epoch': 1}
    forwards, steps = [], []
    original_step = stage._DeviceAdamW.step

    def forward_hook(model, inputs, logits):
        assert all(p.device.type == 'cuda' for p in model.parameters())
        assert all(value.device.type == 'cuda' for value in inputs)
        assert logits.device.type == 'cuda' and logits.dtype == torch.float32
        assert not torch.is_autocast_enabled('cuda')
        forwards.append('train' if model.training else 'dev')

    def step(optimizer, *args, **kwargs):
        assert all(p.device.type == 'cuda' and p.grad is not None and p.grad.device.type == 'cuda'
                   and torch.isfinite(p.grad).all() for group in optimizer.param_groups for p in group['params'])
        result = original_step(optimizer, *args, **kwargs)
        assert all(v.device.type == 'cuda' for state in optimizer.state.values()
                   for key, v in state.items() if key != 'step')
        steps.append(True)
        return result

    # Hook real modules and delegate the real optimizer; no fabricated GPU tensors.
    original_forward = OptionBTinyVCM.forward
    def forward(model, features, lengths):
        logits = original_forward(model, features, lengths)
        forward_hook(model, (features, lengths), logits)
        return logits
    monkeypatch.setattr(OptionBTinyVCM, 'forward', forward)
    monkeypatch.setattr(stage._DeviceAdamW, 'step', step)
    output = tmp_path / 'gpu-fit'
    result = stage.fit_model(views['train'], views['dev'], arrays, output, config)
    assert result['device_selection']['requested'] == 'auto'
    assert steps == [True, True, True]  # One disposable probe + two recipe steps.
    assert forwards.count('train') == 3 and forwards.count('dev') == 2
    assert result['execution_device'] == 'cuda:0' and result['checkpoint_tensor_device'] == 'cpu'
    assert result['device_placement']['target_device'] == 'cuda:0'
    assert result['device_placement']['gradient_devices'] == result['device_placement']['optimizer_moment_devices'] == ['cuda:0']
    assert result['device_placement']['dev_input_devices'] == result['device_placement']['dev_model_devices'] == ['cuda:0']
    assert result['device_placement']['dev_output_device'] == 'cuda:0'
    # Load without map_location to verify the serialized tensors really are CPU.
    checkpoint = torch.load(output / 'best.pt', weights_only=True)
    assert all(value.device.type == 'cpu' for value in checkpoint['state_dict'].values())
    assert checkpoint['config'] == config and result['weights_changed'] is True
    monkeypatch.setattr(OptionBTinyVCM, 'forward', original_forward)
    cpu_model = OptionBTinyVCM().eval().cpu()
    cpu_model.load_state_dict(checkpoint['state_dict'])
    exported = export_cpu_model(cpu_model, output / 'software-fixture.onnx')
    assert exported['initializer_elements'] == 258912
    assert (output / 'software-fixture.onnx').is_file()


@pytest.mark.parametrize('execution', ['cpu', 'cuda:0'])
def test_training_optimizer_failure_never_restarts_fitting_on_cpu(tmp_path, monkeypatch, execution):
    """Simulate OOM after a successful real probe; never restart the recipe."""
    import torch
    from scripts.test.test_optionb_reproduction_fit import SNAPSHOT_PATH, real_audio_fixture
    if execution == 'cuda:0' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; GPU training-error propagation not executed')
    stage = module()
    views, arrays, _, _ = real_audio_fixture(tmp_path)
    config = {**json.loads(SNAPSHOT_PATH.read_text())['training'],
              'human_epochs': 1, 'mixed_epochs': 1, 'steps_per_epoch': 1}
    original_step, calls = stage._DeviceAdamW.step, []
    if execution == 'cpu':
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    def step(optimizer, *args, **kwargs):
        calls.append(str(optimizer.param_groups[0]['params'][0].device))
        if len(calls) == 2:
            raise RuntimeError('software fixture: training OOM, do not restart')
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(stage._DeviceAdamW, 'step', step)
    with pytest.raises(RuntimeError, match='training OOM'):
        stage.fit_model(views['train'], views['dev'], arrays, tmp_path / 'failed', config)
    assert calls == [execution, execution]
    assert not (tmp_path / 'failed/training-result.json').exists()
    assert not (tmp_path / 'failed/best.pt').exists()
    assert json.loads((tmp_path / 'failed/training-device.json').read_text())['device'] == execution
