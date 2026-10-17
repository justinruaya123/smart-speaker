"""Stdlib-safe selection of bound compact or fixed-gate full VCM inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

PINNED_MODEL_METADATA = {
    '3dea2e25dd431c0b085033f04e0b3f16181016ed42c41a1b84ff6acd43d2ac7e':
        {'parameters':2066831,'precision':'int8'},
    'fc87c98c242ccb6f37ed9fcd84bad3869b8c90eaae6267fdab2a1a5af2737f6d':
        {'parameters':935567,'precision':'fp32'},
}


class FixedDemoRuntime:
    """Keep frozen qualification separate from explicit .95/.90 demo decisions."""

    def __init__(self, folder, *, threads=1, diagnostic=True, engine='onnxruntime', mode='demo'):
        if mode not in ('demo', 'qualified'):
            raise ValueError('Decision mode must be demo or qualified')
        from scripts.runtime.pcv_final_runtime import Runtime
        start = time.perf_counter()
        self.base = Runtime(folder, diagnostic=True, threads=threads, engine=engine)
        self.mode = mode
        self.load_ms = (time.perf_counter()-start)*1000

    def __getattr__(self, name):
        return getattr(self.base, name)

    def recognize(self, pcm, sample_rate=16000):
        from scripts.runtime.pcv_schema import output
        from scripts.runtime.speaker_confidence import application_decision
        event = self.base.recognize(pcm, sample_rate)
        start = time.perf_counter()
        result = copy.deepcopy(event)
        result['qualified_output'] = copy.deepcopy(event['output'])
        result['application'] = application_decision(event, dict(mode='demo', intent_threshold=.95, slot_threshold=.9))
        result['fixed_application'] = copy.deepcopy(result['application'])
        result['decision_mode'] = self.mode
        if self.mode == 'demo':
            app = result['application']
            reason = (None if app['command'] is not None else 'capture_fault' if app['reason']=='capture_fault'
                      else 'invalid_argument' if app['reason'] in ('slot_confidence','target_confidence','invalid_command')
                      else 'OOV')
            result['output'] = output(app['command'], reason)
        if 'timing' in result:
            result['timing']['demo_policy_ms'] = 1000*(time.perf_counter()-start)
            result['timing']['total_ms'] += result['timing']['demo_policy_ms']
            seconds = result.get('audio_seconds',0)
            result['timing']['rtf'] = result['timing']['total_ms']/1000/seconds if seconds else None
        return result


def load_runtime(folder, threads=1, *, diagnostic=True, engine='onnxruntime', mode='demo'):
    """Load only inference code; the recipe binds compact recall when present."""
    folder = Path(folder)
    recipe = json.loads((folder/'recipe.json').read_text(encoding='utf-8'))
    if recipe.get('recall_policy_sha256'):
        from scripts.runtime.speaker_compact import CompactRuntime
        runtime = CompactRuntime(folder, threads=threads, diagnostic=diagnostic, engine=engine, mode=mode)
    else:
        runtime = FixedDemoRuntime(folder, threads=threads, diagnostic=diagnostic, engine=engine, mode=mode)
    # Reporting metadata is keyed to a verified graph; frozen recipe bytes survive.
    metadata = PINNED_MODEL_METADATA.get(recipe.get('binding',{}).get('model_sha256'),{})
    runtime.recipe = {**metadata, **recipe}
    runtime.parameters = runtime.recipe.get('parameters')
    runtime.precision = runtime.recipe.get('precision')
    return runtime
