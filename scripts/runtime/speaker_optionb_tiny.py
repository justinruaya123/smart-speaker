"""Torch-free OptionB joint-command CPU inference; never executes devices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.runtime.optionb_contract import COMMANDS, command_for_index
from scripts.runtime.pcv_schema import output, canonical


def decision(logits, policy) -> dict[str, Any]:
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
    candidate = command_for_index(index)
    accepted = bool(policy['enabled'] and index and probability >= policy['min_probability'] and margin >= policy['min_margin'])
    reason = None if accepted else ('unqualified_model' if not policy['enabled'] else 'OOV')
    result = output(candidate if accepted else None, reason)
    return dict(event='command', output=result, application={**result, 'mode': 'joint_complete_command'},
                qualified_output=output(None, 'unqualified_model'), candidate=candidate,
                command_index=index, command_probability=probability, command_margin=margin,
                probabilities=probabilities.tolist(), actions_enabled=False,
                model_qualified=False, inference_contract='audio_to_atomic_complete_optionb_command')


class Runtime:
    """One FP32 ONNX graph; graph/norm/ontology/policy identities fail closed."""
    def __init__(self, bundle, threads=1):
        import onnxruntime as ort
        from vcm.frontend import StreamingFrontend
        self.root = Path(bundle)
        self.recipe = json.loads((self.root / 'recipe.json').read_text(encoding='utf-8'))
        if (self.recipe.get('architecture') != 'OptionBTinyVCM-1'
                or type(self.recipe.get('parameters')) is not int
                or not 0 < self.recipe['parameters'] <= 300_000
                or self.recipe.get('research_only') is not True):
            raise ValueError('Invalid tiny research architecture/budget contract')
        bindings = self.recipe['binding']
        names = {'model.onnx', 'commands.json', 'normalization.json', 'policy.json'}
        if set(bindings) != names:
            raise ValueError('Unexpected bundle identity targets')
        for name, expected in bindings.items():
            with (self.root / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Bundle identity mismatch: ' + name)
        commands = json.loads((self.root / 'commands.json').read_text(encoding='utf-8'))
        if canonical(commands) != canonical(list(COMMANDS)):
            raise ValueError('Complete-command class order changed')
        self.norm = json.loads((self.root / 'normalization.json').read_text(encoding='utf-8'))
        StreamingFrontend(self.norm['mean'], self.norm['std'])
        self.policy = json.loads((self.root / 'policy.json').read_text(encoding='utf-8'))
        decision(np.zeros(32), self.policy)  # Validate even for silent input.
        if type(threads) is not int or threads < 1:
            raise ValueError('A positive CPU thread count is required')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        started = time.perf_counter()
        self.session = ort.InferenceSession(str(self.root / 'model.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.load_ms = (time.perf_counter() - started) * 1000
        if [v.name for v in self.session.get_inputs()] != ['features', 'lengths'] or self.session.get_outputs()[0].shape[-1] != 32:
            raise ValueError('Unexpected joint-command ONNX interface')

    def recognize(self, pcm) -> dict[str, Any]:
        from threadpoolctl import threadpool_limits
        from vcm.frontend import features
        started = time.perf_counter()
        pcm = np.asarray(pcm, dtype=np.float32)
        if (pcm.ndim != 1 or not 400 <= pcm.size <= 15 * 16000 or not np.isfinite(pcm).all()
                or float(np.max(np.abs(pcm))) > 1 or float(np.max(np.abs(pcm))) < 1e-6):
            rejected = output(None, 'capture_fault')
            return dict(event='command', output=rejected, application={**rejected, 'mode': 'joint_complete_command'},
                        qualified_output=output(None, 'unqualified_model'), actions_enabled=False, model_qualified=False,
                        latency_ms=(time.perf_counter() - started) * 1000)
        with threadpool_limits(limits=1, user_api='blas'):
            values = features(pcm, self.norm['mean'], self.norm['std'])
        logits = self.session.run(['logits'], {'features': values[None],
            'lengths': np.asarray([values.shape[1]], dtype=np.int64)})[0][0]
        result = decision(logits, self.policy)
        result.update(logits=logits.tolist(), latency_ms=(time.perf_counter() - started) * 1000)
        return result


def main():
    import argparse
    import soundfile as sf
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    pcm, rate = sf.read(args.wav, dtype='float32', always_2d=False)
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Use mono 16kHz PCM WAV')
    event = Runtime(args.bundle, args.threads).recognize(pcm)
    event.update(source='wav', actions_enabled=False)
    print(json.dumps(event, allow_nan=False))


if __name__ == '__main__':
    main()
