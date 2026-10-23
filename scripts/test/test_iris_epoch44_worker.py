"""Existing JSONL worker with the frozen adapter, never a second dispatcher."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from scripts.runtime.speaker_worker import AudioService
from scripts.runtime.speaker_session import SpeakerEvidence, SpeakerPolicy

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'submission/models/optionb-tiny'


def worker(*controls, args=()):
    env = {**os.environ, 'SMART_SPEAKER_ROOT': str(ROOT), 'ACTIONS_ENABLED': 'true',
           'IRIS_FEEDBACK_ENABLED': 'true', 'IRIS_SPEAKER_INDEX': '0'}
    return subprocess.run([sys.executable, '-B', '-m', 'scripts.runtime.speaker_worker',
        '--bundle', str(BUNDLE), *args], input=''.join(json.dumps(c)+'\n' for c in controls),
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=30)


def test_worker_ready_eof_identifies_selected_model_policy_and_actual_actions():
    result = worker()
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    ready = next(e for e in events if e['event'] == 'ready')
    assert ready['model_sha256'] == 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'
    assert ready['parameters'] == 258912 and ready['selected_epoch'] == 44
    assert ready['application_config']['mode'] == 'joint_complete_command'
    assert ready['application_config']['min_probability'] == .5
    assert ready['application_config']['min_margin'] == .1
    assert 'intent_threshold' not in ready['application_config']
    assert ready['wake_phrase'] == 'hey iris'
    assert ready['actions_enabled'] is False  # benchmark-compatible profile
    assert ready['model_qualified'] is False


def test_frozen_check_loads_wake_vad_owner_companions_without_microphone():
    result = worker(args=('--check', '--profile', 'owner-protected'))
    assert result.returncode == 0, result.stderr
    checked = json.loads(result.stdout)
    assert checked['parameters'] == 258912 and checked['selected_epoch'] == 44
    assert checked['owner_gating_enabled'] is True
    assert checked['companions']['wake']['model_sha256']
    assert checked['companions']['owner']['model_sha256']
    assert checked['actions_executed'] is False


def test_owner_veto_preserves_model_output_and_atomic_mode():
    from scripts.runtime.iris_epoch44 import Runtime
    runtime = Runtime(BUNDLE)
    # Decision fixture above the frozen .50/.10 but below legacy .95, not speech accuracy.
    from scripts.runtime.iris_epoch44 import decision
    logits = np.full(32, -30.); logits[1:3] = np.log([.6, .4])
    model = decision(logits, runtime.policy)
    runtime.recognize = lambda pcm: model.copy()
    ev = lambda vector: SpeakerEvidence(vector, 'a'*64, 1., False)
    owner = SimpleNamespace(policy=SpeakerPolicy(.8), evidence=lambda pcm: ev([0., 1.]),
                            continuity=lambda pcm: [ev([0., 1.])])
    events = []
    service = AudioService(runtime, events.append, {'inference': {'mode': 'demo'}, 'actions': {'enabled': True}},
                           owner=owner, profile='owner-protected')
    service.arm_owner(np.ones(16000)*.1, reference=ev([1., 0.]))
    service.forward(service.recognize(np.ones(16000)*.1))
    final = next(e for e in events if e['event'] == 'command')
    assert final['model_output']['accepted'] is True
    assert final['output']['reason'] == 'speaker_mismatch'
    assert final['application']['mode'] == 'joint_complete_command'
    assert final['application']['accepted'] is False
    assert final['qualified_output']['accepted'] is False


def test_no_audio_uploaded_wav_and_unknown_control_remain_coherent_and_nonactuating():
    stream = io.BytesIO()
    sf.write(stream, np.zeros(16000), 16000, format='WAV', subtype='PCM_16')
    result = worker({'id': 'bad', 'op': 'not_supported'},
        {'id': 'wav', 'op': 'wav', 'data': base64.b64encode(stream.getvalue()).decode()},
        {'op': 'shutdown'}, args=('--no-audio', '--profile', 'owner-protected'))
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert next(e for e in events if e.get('id') == 'bad')['ok'] is False
    command = next(e for e in events if e['event'] == 'command')
    assert command['source'] == 'wav' and command['actions_enabled'] is False
    assert command['output']['reason'] == 'capture_fault'
    assert all(e['actions_enabled'] is False for e in events)
    assert next(e for e in events if e.get('id') == 'wav')['ok'] is True


def test_runtime_factory_keeps_legacy_loader(monkeypatch):
    from scripts.runtime import speaker_worker, mvp_static
    sentinel = object()
    monkeypatch.setattr(mvp_static, 'Runtime', lambda *a, **k: sentinel)
    assert speaker_worker.load_runtime(ROOT / 'artifacts/iris', engine='ncnn') is sentinel
