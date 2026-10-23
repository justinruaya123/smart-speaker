"""Frozen epoch44 decisions and actual CPU graph/frontend contract."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'submission/models/optionb-tiny'


def runtime_module():
    from scripts.runtime import iris_epoch44
    return iris_epoch44


@pytest.mark.parametrize('index', range(32))
def test_all_atomic_classes_use_shipped_order(index):
    module = runtime_module()
    commands = json.loads((BUNDLE / 'commands.json').read_bytes())
    logits = np.full(32, -10.)
    logits[index] = 10.
    event = module.decision(logits, {'enabled': True, 'min_probability': .5, 'min_margin': .1})
    assert event['candidate'] == commands[index]
    assert event['command_index'] == index
    assert event['output']['accepted'] is (index != 0)
    assert event['output']['command'] == commands[index]
    assert event['actions_enabled'] is False and event['model_qualified'] is False
    assert event['application']['mode'] == 'joint_complete_command'


def test_probability_and_margin_are_inclusive_and_atomic():
    module = runtime_module()
    logits = np.full(32, -30.)
    logits[1:3] = np.log([.55, .45])
    policy = {'enabled': True, 'min_probability': 0., 'min_margin': 0.}
    event = module.decision(logits, policy)
    policy.update(min_probability=event['command_probability'], min_margin=event['command_margin'])
    assert module.decision(logits, policy)['output']['accepted'] is True
    for field in ('min_probability', 'min_margin'):
        strict = {**policy, field: np.nextafter(policy[field], 1.).item()}
        assert module.decision(logits, strict)['output']['accepted'] is False
    assert module.decision(logits, {**policy, 'enabled': False})['output']['reason'] == 'unqualified_model'


@pytest.mark.parametrize('name', ['model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json'])
def test_self_consistent_rehashed_substitutions_still_fail(tmp_path, name):
    module = runtime_module()
    copied = tmp_path / 'bundle'
    shutil.copytree(BUNDLE, copied)
    (copied / name).write_bytes((copied / name).read_bytes() + b' ')
    # A rehashed mutable recipe is never an approval authority.
    import hashlib
    recipe = json.loads((copied / 'recipe.json').read_bytes())
    if name != 'recipe.json':
        recipe['binding'][name] = hashlib.sha256((copied / name).read_bytes()).hexdigest()
        (copied / 'recipe.json').write_text(json.dumps(recipe), encoding='utf8')
    with pytest.raises(ValueError, match='identity|frozen'):
        module.Runtime(copied)


@pytest.mark.parametrize('pcm', [np.empty(0), np.zeros(16000), np.ones(399)*.1,
    np.ones(240001)*.1, np.array([float('nan')]*400), np.full(400, float('inf')),
    np.full(400, 1.1), np.ones((400, 1))*.1])
def test_capture_faults_never_reach_graph(pcm):
    runtime = runtime_module().Runtime(BUNDLE)
    class Forbidden:
        def run(self, *args, **kwargs):
            pytest.fail('Capture faults must not call the graph')
    runtime.session = Forbidden()
    event = runtime.recognize(pcm)
    assert event['output']['reason'] == 'capture_fault'
    assert event['diagnostic_candidate'] is None


def test_real_onnx_features_lengths_and_decision_parity():
    from vcm.frontend import features, StreamingFrontend
    runtime = runtime_module().Runtime(BUNDLE)
    pcm = np.sin(np.arange(16000, dtype='float32') * .13) * .05
    f = features(pcm, runtime.norm['mean'], runtime.norm['std'])
    streaming = StreamingFrontend(runtime.norm['mean'], runtime.norm['std'])
    parts = [streaming.feed(pcm[i:i+512]) for i in range(0, len(pcm), 512)]
    # Chunked BLAS can differ by one float32 ULP without changing the frontend.
    np.testing.assert_allclose(np.concatenate(parts, axis=1), f, rtol=0, atol=2e-7)
    assert f.shape == (40, 98)
    logits = runtime.session.run(['logits'], {'features': f[None], 'lengths': np.array([f.shape[1]], dtype='int64')})[0][0]
    event = runtime.recognize(pcm)
    np.testing.assert_array_equal(event['logits'], logits)
    assert event['output'] == runtime_module().decision(logits, runtime.policy)['output']
    assert event['model_output'] == event['output']
    assert event['model_sha256'] == runtime.model_sha256
    assert event['timing']['total_ms'] >= event['frontend_ms'] >= 0
    assert runtime.session.get_providers() == ['CPUExecutionProvider']
    assert runtime.wake_root == ROOT / 'artifacts/iris/wake'
    assert runtime.owner_root == ROOT / 'artifacts/iris/owner'


def test_qualified_mode_never_accepts_and_unsupported_engine_fails():
    module = runtime_module()
    with pytest.raises(ValueError, match='onnxruntime'):
        module.Runtime(BUNDLE, engine='ncnn')
    runtime = module.Runtime(BUNDLE, mode='qualified')
    assert runtime.recognize(np.ones(16000, dtype='float32')*.1)['output']['reason'] == 'unqualified_model'
    assert runtime.recognize(np.ones(16000, dtype='float32')*.1, sample_rate=8000)['output']['reason'] == 'capture_fault'


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_thread_count_fails(value):
    with pytest.raises(ValueError, match='thread'):
        runtime_module().Runtime(BUNDLE, threads=value)
