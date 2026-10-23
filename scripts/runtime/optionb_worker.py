"""Pinned epoch44 OptionB preview adapter; one decision, no provider actions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

from scripts.runtime.pcv_schema import output
from scripts.runtime.speaker_optionb_tiny import Runtime
from scripts.runtime.speaker_worker import AudioService

CONTRACT = 'audio_to_atomic_complete_optionb_command'
BUNDLE_SHA256 = {
    'model.onnx': 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce',
    'normalization.json': '28f7ef1b2c926c587d611d17eef91184c2c7e2de66abe352d1fc2365e45d69ed',
    'commands.json': '71017cf22366d15557e8878636135f5ff2d0ac81fc4dfd6f058fc88f0f2d4b5d',
    'policy.json': 'b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0',
    'recipe.json': '020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621',
}


class FrozenOptionBRuntime(Runtime):
    """Pin the original epoch44, class order, normalizer and separate demo policy."""
    engine = 'onnxruntime'

    def __init__(self, bundle, threads=1):
        root = Path(bundle)
        for name, expected in BUNDLE_SHA256.items():
            with (root / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Frozen OptionB identity mismatch: ' + name)
        super().__init__(root, threads=threads)


class OptionBAudioService(AudioService):
    """Reuse capture/JSONL semantics, not historical independent-head gates."""
    def __init__(self, runtime, emit, settings=None, *, wake_root=None):
        super().__init__(runtime, emit, settings=settings)
        self.wake_root = Path(wake_root) if wake_root is not None else None

    def new_gate(self):
        if self.wake_root is None:
            raise ValueError('Select an explicit --wake-root for wake/microphone validation')
        from scripts.runtime.pcv_final_runtime import make_gate
        service = self
        class Observed:
            root = service.wake_root
            def recognize(self, pcm):
                return service.recognize(pcm)
        return make_gate(Observed())

    def fail(self, state, error):
        """Publish failure even if capture cleanup raises; never leave a live gate."""
        details = {'message': str(error)}
        gate, self.gate = self.gate, None
        try:
            try:
                self.stop(announce=False)
            finally:
                if gate is not None:
                    gate.reset()
        except Exception as cleanup_error:
            details['cleanup_error'] = str(cleanup_error)
        self.set_state(state, **details)

    def emit(self, value):
        result = deepcopy(value)
        if result.get('event') == 'command':
            chosen = result['output']
            application = result.get('application')
            if application is None and chosen.get('accepted') is False:
                application = {**chosen, 'mode': 'joint_complete_command'}
            if (not isinstance(application, dict)
                    or application.get('mode') != 'joint_complete_command'
                    or {k: v for k, v in application.items() if k != 'mode'} != chosen):
                raise ValueError('Inconsistent atomic decision; no secondary acceptance gate')
            result.update(application=application, inference_contract=CONTRACT,
                          qualified_output=output(None, 'unqualified_model'))
        self.sequence += 1
        self.deliver({**result, 'session_id': self.session, 'sequence': self.sequence,
                      'source': self.source, 'actions_enabled': False, 'model_qualified': False})


def control(service, message):
    """The worker owns capture only. There is no action or policy-update operation."""
    import base64
    op = message.get('op')
    if op == 'devices':
        return {'devices': service.devices()}
    if op == 'listen':
        return {'microphone': service.listen(message.get('device'))}
    if op == 'mute':
        service.stop()
        return {}
    if op == 'wav':
        return service.wav(base64.b64decode(message['data'], validate=True), wake=message.get('wake') is True)
    raise ValueError('Unknown runtime operation')


def main():
    import argparse
    import json
    import os
    import queue
    import sys
    import threading

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--diagnostic', action='store_true', help='Compatibility flag; never changes acceptance')
    parser.add_argument('--wake-root', type=Path, default=os.environ.get('SMART_SPEAKER_WAKE_ROOT'))
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--wav', type=Path)
    source.add_argument('--list-mics', action='store_true')
    source.add_argument('--check', action='store_true')
    parser.add_argument('--wake', action='store_true')
    args = parser.parse_args()

    def emit(value):
        print(json.dumps(value, allow_nan=False), flush=True)

    if args.list_mics:
        emit({'event': 'devices', 'devices': OptionBAudioService(None, emit).devices(), 'actions_enabled': False})
        return
    runtime = FrozenOptionBRuntime(args.bundle, args.threads)
    service = OptionBAudioService(runtime, emit, wake_root=args.wake_root)
    if args.wav:
        emit({'event': 'done', **service.wav(args.wav.read_bytes(), wake=args.wake), 'actions_enabled': False})
        return
    service.emit({'event': 'ready', 'engine': 'onnx-cpu', 'loaded': True, 'nativeSession': True,
                  'state': service.state, 'model_sha256': BUNDLE_SHA256['model.onnx'],
                  'bundle_sha256': BUNDLE_SHA256, 'parameters': 258912, 'selected_epoch': 44,
                  'policy_enabled': True, 'application_config': {'mode': 'joint_complete_command', **runtime.policy},
                  'inference_contract': CONTRACT, 'session_load_ms': runtime.load_ms, 'threads': args.threads})
    if args.check:
        return
    incoming = queue.Queue(maxsize=8)

    def reader():
        while True:
            line = sys.stdin.readline(6 * 1024 * 1024)
            if not line:
                incoming.put({'op': 'shutdown'})
                return
            try:
                if not line.endswith('\n'):
                    raise ValueError('Control message too large')
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError('Expected control object')
                incoming.put(item, timeout=1)
            except Exception as error:
                emit({'event': 'error', 'message': str(error), 'actions_enabled': False})
    threading.Thread(target=reader, daemon=True).start()
    try:
        while True:
            try:
                message = incoming.get(timeout=0 if service.stream else .1)
            except queue.Empty:
                if service.stream:
                    try:
                        service.tick()
                    except Exception as error:
                        service.fail('capture_error', error)
                continue
            if message.get('op') == 'shutdown':
                break
            try:
                result = control(service, message)
                service.emit({'event': 'reply', 'id': message.get('id'), 'ok': True, **result})
            except Exception as error:
                service.fail('capture_error' if message.get('op') == 'listen' else 'runtime_error', error)
                service.emit({'event': 'reply', 'id': message.get('id'), 'ok': False, 'message': str(error)})
    finally:
        service.stop(announce=False)


if __name__ == '__main__':
    main()
