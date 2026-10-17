"""Portable replay tests: real WAV/frontend/ORT, never acoustic qualification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


COMMAND = {'intent': 'TIMER', 'slots': {'duration_seconds': 123}}


def tiny_bundle(root, threshold=.8):
    """Constant finite logits exercise the actual CPU runtime without training."""
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    root.mkdir()
    from onnx import TensorProto, helper
    graph = helper.make_graph([helper.make_node('Constant', [], ['logits'], value=helper.make_tensor(
        'constant', TensorProto.FLOAT, [1, 2], [0., 8.]))], 'software_fixture',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, 40, 'time'])],
        [helper.make_tensor_value_info('logits', TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save(model, root / 'model.onnx')
    values = {
        'catalog.json': [{'command_id': 0, 'command': None}, {'command_id': 1, 'command': COMMAND}],
        'normalization.json': {'mean': [0.] * 40, 'std': [1.] * 40},
        'policy.json': {'enabled': True, 'qualified': False, 'trained_classes': [0, 1],
                        'threshold': threshold, 'margin': .1},
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ['model.onnx', *values]}
    (root / 'bundle.json').write_text(json.dumps({'format': 'mvp-static-v1', 'files': hashes}))
    return root


def manifest(root, *, silence=True):
    root.mkdir(exist_ok=True)
    tone = np.sin(np.arange(16000, dtype=np.float32) * .1) * .05
    sf.write(root / 'tone.wav', tone, 16000, subtype='PCM_16')
    sf.write(root / 'silence.wav', np.zeros(16000, dtype=np.float32), 16000, subtype='PCM_16')
    rows = [dict(id='correct', audio='tone.wav', sha256=hashlib.sha256((root / 'tone.wav').read_bytes()).hexdigest(),
                 expected=json.loads(json.dumps(COMMAND)), provenance='software_fixture_tone')]
    if silence:
        rows.append(dict(id='silence', audio='silence.wav', sha256=hashlib.sha256((root / 'silence.wav').read_bytes()).hexdigest(),
                         expected=None, provenance='software_fixture_silence'))
    path = root / 'commands.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path, rows


def test_real_pcm_runtime_scores_raw_and_accepted_separately_without_repeat_inflation(tmp_path):
    from scripts.runtime.iris_pi_benchmark import benchmark
    bundle = tiny_bundle(tmp_path / 'bundle', threshold=1.000001)
    source, _ = manifest(tmp_path / 'samples')
    result = benchmark(bundle, source, tmp_path / 'result', engine='onnxruntime', repeats=3)
    assert result['actions_enabled'] is False
    assert result['profile'] == 'benchmark-compatible'
    assert result['source_n'] == 2 and result['observation_n'] == 6
    assert result['raw']['exact_positive_n'] == 1
    assert result['raw']['unavailable_n'] == 1  # silence never ran the classifier
    assert result['accepted']['exact_positive_n'] == 0
    assert result['accepted']['positive_n'] == 1
    assert result['accepted']['negative_abstention_n'] == 1
    assert result['per_intent']['TIMER']['accepted']['exact_positive_n'] == 0
    assert result['latency_ms']['end_to_end']['n'] == 6
    assert result['rtf']['n'] == 6
    assert result['first_inference_included'] is True
    assert result['latency_ms']['end_to_end']['p95'] >= result['latency_ms']['end_to_end']['p50'] >= 0
    assert result['session_load_ms'] >= 0
    assert result['physical_microphone_measured'] is False
    saved = json.loads((tmp_path / 'result/metrics.json').read_text())
    assert saved['manifest_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    observations = [json.loads(line) for line in (tmp_path / 'result/trials.jsonl').read_text().splitlines()]
    assert len(observations) == 6
    assert all(row['sample_rate'] == 16000 and row['audio_seconds'] == 1. for row in observations)
    assert all(row['actions_enabled'] is False for row in observations)


def test_exact_slots_and_negative_false_accepts_use_original_recording_denominators(tmp_path):
    from scripts.runtime.iris_pi_benchmark import benchmark
    bundle = tiny_bundle(tmp_path / 'bundle')
    source, rows = manifest(tmp_path / 'samples')
    rows.extend([{**rows[0], 'id': 'wrong-value', 'expected': {'intent': 'TIMER', 'slots': {'duration_seconds': 124}}},
                 {**rows[0], 'id': 'negative-speech-fixture', 'expected': None}])
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = benchmark(bundle, source, tmp_path / 'result', engine='onnxruntime', repeats=2)
    assert result['accepted']['positive_n'] == 2
    assert result['accepted']['negative_n'] == 2
    assert result['accepted']['exact_positive_n'] == 1
    assert result['accepted']['negative_abstention_n'] == 1
    assert result['accepted']['false_accept_n'] == 1
    assert result['accepted']['complete_command_recall'] == .5
    assert result['accepted']['precision'] == pytest.approx(1 / 3)
    assert result['per_intent']['TIMER']['accepted']['precision'] == pytest.approx(1 / 3)
    assert result['capture_fault_source_n'] == 1


def test_cli_defaults_to_installed_bundle_and_missing_default_manifest_is_actionable(tmp_path, capsys):
    from scripts.runtime.iris_pi_benchmark import main
    with pytest.raises(SystemExit) as failure:
        main(['--output', str(tmp_path / 'result')], root=tmp_path)
    assert failure.value.code == 2
    assert '--manifest' in capsys.readouterr().err
    assert not (tmp_path / 'result').exists()


def test_changed_audio_or_invalid_typed_reference_fails_before_model_loading(tmp_path):
    from scripts.runtime.iris_pi_benchmark import benchmark
    source, rows = manifest(tmp_path / 'samples')
    def forbidden(*args, **kwargs):
        pytest.fail('Bad input must not initialize a model')
    (source.parent / 'tone.wav').write_bytes(b'changed payload')
    with pytest.raises(ValueError, match='identity|hash'):
        benchmark(tmp_path, source, tmp_path / 'result', runtime_factory=forbidden)
    source, rows = manifest(source.parent)
    rows[0]['expected']['slots']['duration_seconds'] = True
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='reference|numeric'):
        benchmark(tmp_path, source, tmp_path / 'result', runtime_factory=forbidden)


@pytest.mark.parametrize('rate,channels', [(8000, 1), (16000, 2)])
def test_replay_never_resamples_or_mixes_invalid_pcm(tmp_path, rate, channels):
    from scripts.runtime.iris_pi_benchmark import read_pcm
    path = tmp_path / 'invalid.wav'
    sf.write(path, np.ones((rate, channels), dtype=np.float32) * .1, rate, subtype='PCM_16')
    with pytest.raises(ValueError, match='mono.*16000|16000.*mono'):
        read_pcm(path)


def test_arm_machine_without_raspberry_pi_model_is_never_labelled_pi(monkeypatch):
    from scripts.runtime import iris_pi_benchmark as module
    monkeypatch.setattr(module, 'device_model', lambda: None)
    monkeypatch.setattr(module.platform, 'machine', lambda: 'aarch64')
    hardware = module.hardware()
    assert hardware['physical_pi_measured'] is False
    assert hardware['measurement_host'] == 'non_pi_host'


def test_runtime_module_does_not_require_reproduction_tool(tmp_path, monkeypatch):
    import sys
    from scripts.runtime.iris_pi_benchmark import benchmark
    bundle = tiny_bundle(tmp_path / 'bundle')
    source, _ = manifest(tmp_path / 'samples')
    monkeypatch.setitem(sys.modules, 'scripts.iris_release', None)
    result = benchmark(bundle, source, tmp_path / 'result', engine='onnxruntime', repeats=1)
    assert result['accepted']['exact_positive_n'] == 1


def test_cli_uses_the_transactional_installed_bundle_pointer(tmp_path, capsys):
    from scripts.runtime.iris_pi_benchmark import main
    (tmp_path/'models').mkdir()
    selected=tiny_bundle(tmp_path/'models/selected')
    manifest(tmp_path/'samples')
    (tmp_path/'state').mkdir()
    (tmp_path/'state/install.json').write_text(json.dumps({'bundle':'models/selected'}))
    result=main(['--engine','onnxruntime','--repeats','1','--output',str(tmp_path/'results')],root=tmp_path)
    assert result['bundle']==str(selected)
    assert json.loads(capsys.readouterr().out)['actions_enabled'] is False


def test_malformed_runtime_output_cannot_be_scored_as_a_successful_negative(tmp_path):
    from types import SimpleNamespace
    from scripts.runtime.iris_pi_benchmark import benchmark
    source,_=manifest(tmp_path/'samples')
    calls=[]
    def factory(bundle,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(recognize=lambda pcm,**kw:{'output':{'accepted':False,'command':None}},
                               make_gate=lambda:pytest.fail('Replay cannot start wake or action paths'))
    with pytest.raises(ValueError,match='schema-v3'):
        benchmark(tmp_path/'unused-model',source,tmp_path/'results',runtime_factory=factory,repeats=1)
    assert calls==[dict(threads=1,engine='ncnn',mode='demo',profile='benchmark-compatible')]
    assert not (tmp_path/'results/metrics.json').exists()
