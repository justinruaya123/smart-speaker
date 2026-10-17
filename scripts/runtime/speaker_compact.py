"""Explicit compact demo decisions; preserve the original qualified result."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

from scripts.runtime.pcv_schema import INTENTS, output
from scripts.runtime.speaker_confidence import application_decision

THRESHOLDS = (.50, .60, .70, .80, .85, .90, .95)


def load_bound_policy(folder, recipe):
    data = (Path(folder)/'recall-policy.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != recipe.get('recall_policy_sha256'):
        raise ValueError('Recall policy file identity mismatch')
    policy = json.loads(data)
    validate_policy(policy, recipe['binding'])
    return policy


def validate_policy(policy, binding):
    if policy.get('format') != 'speaker-compact-recall-v1' or policy.get('binding') != binding:
        raise ValueError('Recall policy binding mismatch')
    if policy.get('slot_threshold') != .9:
        raise ValueError('Required slot/target threshold must remain .90')
    values = policy.get('intent_thresholds', {})
    if set(values) != set(INTENTS) or any(type(v) not in (int, float) or not math.isfinite(v)
                                         or v not in THRESHOLDS for v in values.values()):
        raise ValueError('Invalid intent threshold map')
    if policy.get('qualified') is not False:
        raise ValueError('Demo recall fitting cannot grant qualification')


def apply_policy(event, policy):
    candidate = (event.get('diagnostic_candidate') or {}).get('candidate')
    intent = candidate.get('intent') if isinstance(candidate, dict) else None
    threshold = policy['intent_thresholds'].get(intent, .95)
    return application_decision(event, dict(mode='demo', intent_threshold=threshold, slot_threshold=.9))


def decorate_event(event, policy, *, mode):
    if mode not in ('demo', 'qualified'):
        raise ValueError('Decision mode must be demo or qualified')
    start = time.perf_counter()
    result = copy.deepcopy(event)
    result['qualified_output'] = copy.deepcopy(event['output'])
    result['application'] = apply_policy(event, policy)
    result['fixed_application'] = application_decision(event, dict(mode='demo', intent_threshold=.95, slot_threshold=.9))
    result['decision_mode'] = mode
    if mode == 'demo':
        app = result['application']
        reason = (None if app['command'] is not None else 'capture_fault' if app['reason']=='capture_fault'
                  else 'invalid_argument' if app['reason'] in ('slot_confidence','target_confidence','invalid_command')
                  else 'OOV')
        result['output'] = output(app['command'], reason)
    if 'timing' in result:
        elapsed = 1000 * (time.perf_counter()-start)
        result['timing']['recall_policy_ms'] = elapsed
        result['timing']['total_ms'] += elapsed
        seconds = result.get('audio_seconds', 0)
        result['timing']['rtf'] = result['timing']['total_ms']/1000/seconds if seconds else None
    return result


class CompactRuntime:
    def __init__(self, folder, *, diagnostic=True, threads=1, engine='onnxruntime', mode='demo'):
        from scripts.runtime.pcv_final_runtime import Runtime
        start = time.perf_counter()
        if mode not in ('demo', 'qualified'):
            raise ValueError('Decision mode must be demo or qualified')
        self.base = Runtime(folder, diagnostic=True, threads=threads, engine=engine)
        self.recall_policy = load_bound_policy(self.base.root, self.base.recipe)
        self.mode = mode
        self.load_ms = (time.perf_counter()-start)*1000

    def __getattr__(self, name):
        return getattr(self.base, name)

    def recognize(self, pcm, sample_rate=16000):
        return decorate_event(self.base.recognize(pcm, sample_rate), self.recall_policy, mode=self.mode)
