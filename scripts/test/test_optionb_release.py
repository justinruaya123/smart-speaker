"""Release adapter contracts, not recognition-quality or physical-device evidence."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def adapter():
    try:
        return importlib.import_module('scripts.runtime.optionb_worker')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.runtime.optionb_worker':
            raise
        pytest.fail('Pinned OptionB SpeakerUI worker is missing')


def test_atomic_decision_survives_old_independent_gates_without_enabling_actions():
    from scripts.runtime.speaker_optionb_tiny import decision
    logits = np.zeros(32)
    logits[1] = 4.
    event = decision(logits, dict(enabled=True, min_probability=.50, min_margin=.10))
    assert event['output']['accepted'] and event['command_probability'] < .95
    before = deepcopy(event)
    delivered = []
    service = adapter().OptionBAudioService(None, delivered.append,
        settings={'inference': {'intent_threshold': .99, 'slot_threshold': .99},
                  'actions': {'enabled': True}})
    service.emit(event)
    actual = delivered[-1]
    assert event == before
    assert actual['output'] == before['output']
    assert actual['application'] == before['application']
    assert actual['qualified_output']['accepted'] is False
    assert actual['actions_enabled'] is False
    assert actual['inference_contract'] == 'audio_to_atomic_complete_optionb_command'
    assert actual['source'] == 'microphone'


def test_gate_capture_rejection_is_forwarded_without_rescuing_a_command():
    from scripts.runtime.pcv_schema import output
    delivered = []
    service = adapter().OptionBAudioService(None, delivered.append)
    service.forward({'output': output(None, 'capture_fault'), 'reason': 'capture_fault'})
    event = next(e for e in delivered if e['event'] == 'command')
    assert event['application'] == {**event['output'], 'mode': 'joint_complete_command'}
    assert event['application']['accepted'] is False
    assert event['application']['reason'] == 'capture_fault'
    assert event['actions_enabled'] is False


def test_inconsistent_or_missing_accepted_application_fails_closed():
    from scripts.runtime.speaker_optionb_tiny import decision
    logits = np.zeros(32); logits[1] = 8.
    event = decision(logits, dict(enabled=True, min_probability=.50, min_margin=.10))
    service = adapter().OptionBAudioService(None, lambda _: pytest.fail('invalid event delivered'))
    for changed in ({'application': {}}, {'application': {**event['application'], 'accepted': False}},
                    {'application': {**event['application'], 'command': {'intent': 'TIME', 'slots': {}}}}):
        with pytest.raises(ValueError, match='atomic decision'):
            service.emit({**event, **changed})


def test_pinned_bundle_rejects_a_self_consistent_substitute_before_inference(tmp_path):
    import json
    for name in ('model.onnx', 'commands.json', 'normalization.json', 'policy.json', 'recipe.json'):
        (tmp_path / name).write_text('{}' if name.endswith('.json') else 'not the epoch44 graph')
    with pytest.raises(ValueError, match='Frozen OptionB identity'):
        adapter().FrozenOptionBRuntime(tmp_path)


def test_worker_accepts_only_wav_controls_and_never_executes_provider_operations():
    module = adapter()
    class Service:
        def wav(self, data, *, wake=False): return {'bytes': len(data), 'wake': wake}
    assert module.control(Service(), {'op': 'wav', 'data': 'YWJj', 'wake': False}) == {'bytes': 3, 'wake': False}
    for op in ('speak', 'execute', 'fit', 'update_policy'):
        with pytest.raises(ValueError, match='Unknown runtime operation'):
            module.control(Service(), {'op': op})


def _worker_control_probe(failure):
    """Child-process hardware/model doubles; main/control/capture lifecycle stay real."""
    import json
    import sys
    from types import ModuleType, SimpleNamespace

    module = importlib.import_module('scripts.runtime.optionb_worker')
    streams, gates, services, transitions = [], [], [], []
    discovery_calls = 0

    class Stream:
        def __init__(self, *, device, **kwargs):
            if device == 1 and failure == 'open':
                raise RuntimeError('fixture open failure')
            self.device = device
            self.aborted = self.closed = False
            streams.append(self)

        def start(self):
            if self.device == 1 and failure == 'start':
                raise RuntimeError('fixture start failure')

        def read(self, frames):
            return np.zeros((frames, 1), dtype=np.float32), False

        def abort(self):
            self.aborted = True
            if failure in ('mute', 'cleanup_abort'):
                raise RuntimeError(f'fixture {failure} failure')

        def close(self):
            self.closed = True
            if failure == 'cleanup_close':
                raise RuntimeError('fixture cleanup_close failure')

    class Gate:
        state = 'wake'
        started = False

        def __init__(self):
            self.resets = 0
            gates.append(self)

        def feed(self, *args, **kwargs):
            return []

        def reset(self):
            self.resets += 1
            if failure == 'cleanup_gate':
                raise RuntimeError('fixture cleanup_gate failure')

    def devices():
        nonlocal discovery_calls
        discovery_calls += 1
        if discovery_calls > 1 and failure == 'devices':
            raise RuntimeError('fixture devices failure')
        return [dict(name=f'Fixture input {index}', hostapi=0, max_input_channels=1,
                     default_samplerate=16000) for index in range(2)]

    def check_settings(*, device, **kwargs):
        if device == 1 and failure == 'settings':
            raise RuntimeError('fixture settings failure')

    def new_gate():
        if gates and failure == 'gate':
            services[0].wake_root = None
            return service_class.new_gate(services[0])
        return Gate()

    service_class = module.OptionBAudioService

    def snapshot(service):
        return {'state': service.state, 'stream_stopped': service.stream is None,
                'streams': [dict(device=stream.device, aborted=stream.aborted, closed=stream.closed)
                            for stream in streams], 'gate_resets': [gate.resets for gate in gates]}

    def create_service(*args, **kwargs):
        service = service_class(*args, **kwargs)
        service.new_gate = new_gate
        emit = service.deliver

        def observed_emit(event):
            if event['event'] in ('state', 'reply'):
                transitions.append({**snapshot(service), 'event': event['event'], 'id': event.get('id')})
            emit(event)

        service.deliver = observed_emit
        services.append(service)
        return service

    sounddevice = ModuleType('sounddevice')
    vars(sounddevice).update(query_hostapis=lambda: [{'name': 'Fixture host'}], query_devices=devices,
                             check_input_settings=check_settings, InputStream=Stream)
    sys.modules['sounddevice'] = sounddevice
    setattr(module, 'OptionBAudioService', create_service)
    setattr(module, 'FrozenOptionBRuntime', lambda *args, **kwargs: SimpleNamespace(
        policy=dict(enabled=True, min_probability=.50, min_margin=.10), load_ms=0))
    module.main()
    print(json.dumps({'probe': 'cleanup', **snapshot(services[0]), 'transitions': transitions}), flush=True)


def worker_jsonl(messages, *, failure=None, shutdown=True):
    import json
    from pathlib import Path
    import subprocess
    import sys

    controls = [*messages, *([{'op': 'shutdown'}] if shutdown else [])]
    child = ('from scripts.test.test_optionb_release import _worker_control_probe; '
             f'_worker_control_probe({failure!r})')
    result = subprocess.run(
        [sys.executable, '-B', '-c', child, '--bundle', 'fixture-bundle-not-loaded'],
        cwd=Path(__file__).resolve().parents[2],
        input=''.join(json.dumps(message) + '\n' for message in controls),
        text=True, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ''
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    cleanup = rows.pop()
    assert cleanup['probe'] == 'cleanup'
    assert rows[0]['event'] == 'ready'
    assert all(event['actions_enabled'] is False for event in rows)
    assert all(event['model_qualified'] is False for event in rows)
    assert not any(event['event'] == 'command' for event in rows)
    assert [event['sequence'] for event in rows] == list(range(1, len(rows) + 1))
    assert len({event['session_id'] for event in rows}) == 1
    return rows, cleanup


@pytest.mark.parametrize('failure', ['settings', 'gate', 'open', 'start', 'devices', 'wav', 'unknown', 'mute'])
def test_worker_control_failure_publishes_state_and_cleans_real_service(failure):
    if failure in ('settings', 'gate', 'open', 'start'):
        control = {'op': 'listen', 'device': 1}
        expected_state = 'capture_error'
    else:
        control = {'op': failure, **({'data': 'bm90LXdhdg=='} if failure == 'wav' else {})}
        expected_state = 'runtime_error'
    events, cleanup = worker_jsonl([
        {'op': 'listen', 'device': 0, 'id': 'first'}, {**control, 'id': 'failed'}], failure=failure)
    assert [event['ok'] for event in events if event['event'] == 'reply'] == [True, False]
    assert [event['state'] for event in events if event['event'] == 'state'] == ['wake_listening', expected_state]
    error = next(event for event in events if event.get('state') == expected_state)
    reply = next(event for event in events if event.get('id') == 'failed')
    assert error['message'] == reply['message']
    assert error['sequence'] < reply['sequence'], 'PythonRuntime consumes replies; state must be explicit first'
    assert cleanup['state'] == expected_state
    for snapshot in [cleanup, *[transition for transition in cleanup['transitions'] if transition['state'] == expected_state]]:
        assert snapshot['stream_stopped'] is True
        assert snapshot['streams'] and all(stream['aborted'] and stream['closed'] for stream in snapshot['streams'])
        assert snapshot['gate_resets'] and all(resets > 0 for resets in snapshot['gate_resets'])


@pytest.mark.parametrize('failure', ['cleanup_abort', 'cleanup_close', 'cleanup_gate'])
def test_worker_listen_error_still_publishes_state_when_cleanup_fails(failure):
    events, cleanup = worker_jsonl([
        {'op': 'listen', 'device': 0, 'id': 'first'},
        {'op': 'listen', 'device': -1, 'id': 'failed'}], failure=failure)
    error = next(event for event in events if event.get('state') == 'capture_error')
    reply = next(event for event in events if event.get('id') == 'failed')
    assert error['message'] == reply['message'] == 'Choose a listed microphone index'
    assert error['cleanup_error'] == f'fixture {failure} failure'
    assert reply['ok'] is False and error['sequence'] < reply['sequence']
    assert cleanup['stream_stopped'] is True
    assert cleanup['streams'] == [{'device': 0, 'aborted': True, 'closed': True}]
    assert cleanup['gate_resets'] == [1], 'a failed reset must not be retried on shutdown'
    at_error = next(transition for transition in cleanup['transitions'] if transition['state'] == 'capture_error')
    assert at_error['stream_stopped'] is True and at_error['gate_resets'] == [1]


def test_worker_recovers_after_failed_relisten_and_can_mute():
    events, cleanup = worker_jsonl([
        {'op': 'listen', 'device': 0, 'id': 'first'},
        {'op': 'listen', 'device': 1, 'id': 'failed'},
        {'op': 'listen', 'device': 0, 'id': 'recovered'},
        {'op': 'mute', 'id': 'muted'}], failure='settings')
    assert [event['ok'] for event in events if event['event'] == 'reply'] == [True, False, True, True]
    assert [event['state'] for event in events if event['event'] == 'state'] == [
        'wake_listening', 'capture_error', 'wake_listening', 'muted']
    assert cleanup['state'] == 'muted' and cleanup['stream_stopped'] is True
    assert len(cleanup['streams']) == 2
    assert all(stream['aborted'] and stream['closed'] for stream in cleanup['streams'])


@pytest.mark.parametrize('shutdown', [False, True], ids=['eof', 'shutdown'])
@pytest.mark.parametrize('listening', [False, True], ids=['idle', 'listening'])
def test_worker_bounded_jsonl_eof_and_shutdown_close_capture(shutdown, listening):
    messages = [{'op': 'listen', 'device': 0, 'id': 'first'}] if listening else []
    events, cleanup = worker_jsonl(messages, shutdown=shutdown)
    assert cleanup['stream_stopped'] is True
    assert [event['ok'] for event in events if event['event'] == 'reply'] == ([True] if listening else [])
    assert [event['state'] for event in events if event['event'] == 'state'] == (['wake_listening'] if listening else [])
    assert len(cleanup['streams']) == int(listening)
    assert all(stream['aborted'] and stream['closed'] for stream in cleanup['streams'])
    assert all(resets > 0 for resets in cleanup['gate_resets'])


def test_release_check_refuses_to_overwrite_any_prior_evidence(tmp_path):
    try:
        module = importlib.import_module('scripts.test.optionb_release_check')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.optionb_release_check':
            raise
        pytest.fail('Read-only frozen release checker is missing')
    (tmp_path / 'sentinel').write_text('preserve')
    with pytest.raises(FileExistsError, match='Preserve'):
        module.run_check(tmp_path / 'no-protocol.json', tmp_path)
    assert (tmp_path / 'sentinel').read_text() == 'preserve'
