"""Original official-scoring serialization; no live logger, worker or CLI.

Extracted unchanged from the pre-migration 94c0021 producer source.
"""
import math

from scripts.train.iris_schema import DEFAULT_CUSTOM, validate_command


def class_prediction(command):
    if command is None:
        return 'OUT_OF_SCOPE', ''
    validate_command(command)
    intent, slots = command['intent'], command['slots']
    if intent == 'VOLUME':
        op = slots['operation']
        return ('VOLUME_'+op.upper(), '') if op in ('up', 'down') else ('UNMAPPED_VOLUME', op)
    if intent == 'TIMER':return intent, str(slots['duration_seconds'])+' seconds'
    if intent == 'ALARM':return intent, slots['time']
    if intent == 'TEMPERATURE':return intent, str(slots['degrees'])
    if intent == 'BRIGHTNESS':return intent, str(slots['percent'])
    if intent == 'COLOR':return intent, slots['color']
    if intent == 'CREATE_REMINDER':
        task = next((r['name'] for r in DEFAULT_CUSTOM['tasks'] if r['id'] == slots['task_id']), None)
        return intent, task or slots['task_id']
    return intent, ''

def log_record(event):
    result = event.get('output', {})
    accepted = result.get('accepted') is True
    command = result.get('command') if accepted else None
    if accepted and command is None:raise ValueError('Accepted output has no command')
    intent, slot = class_prediction(command)
    infer = event.get('timing', {}).get('total_ms')
    duration = event.get('audio_seconds')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (infer, duration)):
        raise ValueError('A benchmark command needs measured inference and audio duration')
    return {'intent':intent, 'slot':slot, 'infer_ms':infer, 'audio_ms':duration*1000}
