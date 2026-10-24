"""Frozen 258,912-parameter epoch44 VCM adapter; CPU only, never actuates.

The independent pins below, not a mutable recipe, approve the shipped bytes.
The frontend and atomic posterior/margin algorithm are the original snapshot's.
Wake/VAD and experimental ownership remain separate Iris companions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.runtime.pcv_schema import baseline_cases, canonical, output

ROOT = Path(__file__).resolve().parents[2]
FROZEN_HASHES = {
    'model.onnx': 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce',
    'commands.json': '71017cf22366d15557e8878636135f5ff2d0ac81fc4dfd6f058fc88f0f2d4b5d',
    'normalization.json': '28f7ef1b2c926c587d611d17eef91184c2c7e2de66abe352d1fc2365e45d69ed',
    'policy.json': 'b9242e46fb093642a41023e8ace3c0cadc2fe51ffede512aa09e399622f355f0',
    'recipe.json': '020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621',
}
COMMANDS = (None, *(json.loads(key) for key in sorted({canonical(c['command']) for c in baseline_cases()})))


def decision(logits, policy):
    """Original single-head complete-command acceptance; UNKNOWN always rejects."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (32,) or not np.isfinite(logits).all():
        raise ValueError('Expected 32 finite complete-command logits')
    if type(policy.get('enabled')) is not bool:
        raise ValueError('Policy enabled must be boolean')
    for field in ('min_probability', 'min_margin'):
        value = policy.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid joint-command policy: ' + field)
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    index = int(np.argmax(probabilities))
    probability = float(probabilities[index])
    margin = probability - float(np.partition(probabilities, -2)[-2])
    candidate = deepcopy(COMMANDS[index])
    accepted = bool(policy['enabled'] and index and probability >= policy['min_probability'] and margin >= policy['min_margin'])
    reason = None if accepted else ('unqualified_model' if not policy['enabled'] else 'OOV')
    result = output(candidate if accepted else None, reason)
    return dict(event='command', output=result, application={**result, 'mode': 'joint_complete_command'},
                qualified_output=output(None, 'unqualified_model'), candidate=candidate,
                command_index=index, command_probability=probability, command_margin=margin,
                probabilities=probabilities.tolist(), actions_enabled=False,
                model_qualified=False, inference_contract='audio_to_atomic_complete_optionb_command')


class Runtime:
    decision_authority = 'atomic_command'
    identity_filename = 'recipe.json'
    artifact_provenance = 'frozen_published_epoch44'
    source_path = Path(__file__)

    def __init__(self, bundle, threads=1, engine='onnxruntime', mode='demo', profile='benchmark-compatible'):
        if engine != 'onnxruntime':
            raise ValueError('Frozen epoch44 supports only onnxruntime CPU; no converted engine is approved')
        if type(threads) is not int or not 1 <= threads <= 8:
            raise ValueError('A CPU thread count in 1..8 is required')
        if mode not in ('demo', 'qualified'):
            raise ValueError('Unknown acceptance mode')
        if profile not in ('benchmark-compatible', 'owner-protected'):
            raise ValueError('Unknown recognition profile')
        self.root = Path(bundle).resolve()
        self.engine, self.mode, self.profile = engine, mode, profile
        self.wake_root = Path(os.environ.get('IRIS_WAKE_ROOT') or ROOT / 'artifacts/iris/wake').resolve()
        self.vad_root = ROOT / 'artifacts/iris/wake'
        self.owner_root = ROOT / 'artifacts/iris/owner'
        for name, expected in FROZEN_HASHES.items():
            path = self.root / name
            if (path.is_symlink() or getattr(path, 'is_junction', lambda: False)()
                    or path.resolve().parent != self.root
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise ValueError('Frozen epoch44 identity mismatch: ' + name)
        self.recipe = json.loads((self.root / 'recipe.json').read_bytes())
        self.commands = json.loads((self.root / 'commands.json').read_bytes())
        if canonical(self.commands) != canonical(list(COMMANDS)):
            raise ValueError('Frozen complete-command class order changed')
        self.norm = json.loads((self.root / 'normalization.json').read_bytes())
        from vcm.frontend import StreamingFrontend
        StreamingFrontend(self.norm['mean'], self.norm['std'])
        self.policy = json.loads((self.root / 'policy.json').read_bytes())
        decision(np.zeros(32), self.policy)
        self.model_sha256 = FROZEN_HASHES['model.onnx']
        self.engine_model_sha256 = self.model_sha256
        self.parameters, self.selected_epoch = 258912, 44
        # Compatibility metadata names actual verified files, never invents bundle.json.
        self.meta = {'files': dict(FROZEN_HASHES)}
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        started = time.perf_counter()
        self.session = ort.InferenceSession(str(self.root / 'model.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.load_ms = (time.perf_counter() - started) * 1000
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if ([v.name for v in inputs] != ['features', 'lengths']
                or inputs[0].type != 'tensor(float)' or inputs[1].type != 'tensor(int64)'
                or outputs[0].name != 'logits' or outputs[0].shape[-1] != 32):
            raise ValueError('Unexpected joint-command ONNX interface')

    def recognize(self, pcm, sample_rate=16000):
        from vcm.frontend import features
        started = time.perf_counter()
        event = dict(schema_version=1, event='command', engine=self.engine,
            model_sha256=self.model_sha256, engine_model_sha256=self.engine_model_sha256,
            parameters=self.parameters, selected_epoch=self.selected_epoch, profile=self.profile,
            decision_authority=self.decision_authority, diagnostic_candidate=None,
            actions_enabled=False, model_qualified=False, qualified=False,
            policy_enabled=self.mode == 'demo', qualified_output=output(None, 'unqualified_model'))
        try:
            x = np.asarray(pcm, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            x = np.empty(0, dtype=np.float32)
        if (sample_rate != 16000 or x.ndim != 1 or not 400 <= x.size <= 240000
                or not np.isfinite(x).all() or np.max(np.abs(x), initial=0) > 1
                or np.max(np.abs(x), initial=0) < 1e-6):
            event['output'] = output(None, 'capture_fault')
        else:
            values = features(x, self.norm['mean'], self.norm['std'])
            event['frontend_ms'] = (time.perf_counter() - started) * 1000
            logits = self.session.run(['logits'], {'features': values[None],
                'lengths': np.asarray([values.shape[1]], dtype=np.int64)})[0][0]
            event.update(decision(logits, self.policy))
            event['logits'] = logits.tolist()
            event['diagnostic_candidate'] = dict(candidate=event['candidate'],
                class_index=event['command_index'], class_probability=event['command_probability'],
                margin=event['command_margin'], logits=event['logits'],
                confidence_interpretation='model softmax; not a correctness guarantee')
            if self.mode == 'qualified':
                event['output'] = output(None, 'unqualified_model')
        event['model_output'] = deepcopy(event['output'])
        event['application'] = {'mode': 'joint_complete_command', **event['output']}
        event['audio_seconds'] = x.size / 16000
        event['latency_ms'] = (time.perf_counter() - started) * 1000
        event['timing'] = {'total_ms': event['latency_ms']}
        return event

    def make_gate(self, observed=None):
        from scripts.runtime.mvp_wake import make_gate
        return make_gate(self, observed)
