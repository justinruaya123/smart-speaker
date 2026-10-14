"""Translate authoritative schema-v3 events to the class live-benchmark log."""
import argparse
import json
import math
from pathlib import Path
import sys

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, validate_command


def class_prediction(command, *, custom=None):
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
        task = next((r['name'] for r in (custom if custom is not None else DEFAULT_CUSTOM)['tasks'] if r['id'] == slots['task_id']), None)
        return intent, task or slots['task_id']
    return intent, ''


def log_record(event, *, decision='output', custom=None):
    if decision not in ('output', 'application', 'final_application'):
        raise ValueError('Unknown authoritative decision')
    if decision != 'output' and decision not in event:
        raise ValueError('Missing declared application decision')
    result = event.get(decision, {})
    if result.get('reason') == 'capture_fault':
        return {'event': 'capture_fault', 'reason': 'capture_fault'}
    accepted = result.get('accepted') is True
    command = result.get('command') if accepted else None
    if accepted and command is None:raise ValueError('Accepted output has no command')
    intent, slot = class_prediction(command, custom=custom)
    infer = event.get('timing', {}).get('total_ms')
    duration = event.get('audio_seconds')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (infer, duration)):
        raise ValueError('A benchmark command needs measured inference and audio duration')
    return {'intent':intent, 'slot':slot, 'infer_ms':infer, 'audio_ms':duration*1000}


class BenchmarkLog:
    """Opt-in serving log, independent of UI diagnostics and transport messages."""
    def __init__(self, path, *, decision='output', custom=None):
        self.path = Path(path)
        self.decision = decision
        self.custom = custom
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event):
        if event.get(self.decision, {}).get('reason') == 'capture_fault':
            record = {'event': 'capture_fault', 'reason': 'capture_fault'}
        elif event.get('event') == 'command' and event.get('timing'):
            record = log_record(event, decision=self.decision, custom=self.custom)
        elif event.get('status') == 'detected':
            record = {'event':'wake'}
        else:
            return
        with self.path.open('a', encoding='utf8') as stream:
            print(json.dumps(record, allow_nan=False), file=stream, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Append to this log; default prints to stdout')
    args = parser.parse_args()
    stream = args.output.open('a', encoding='utf8') if args.output else sys.stdout
    try:
        for line in sys.stdin:
            event = json.loads(line)
            if event.get('event') == 'command' and event.get('timing'):
                print(json.dumps(log_record(event), allow_nan=False), file=stream, flush=True)
            elif event.get('status') == 'detected':
                print(json.dumps({'event':'wake'}), file=stream, flush=True)
    finally:
        if stream is not sys.stdout:stream.close()


if __name__ == '__main__':main()
