"""Versioned demo ontology, separate from all frozen OptionB catalogs."""
from copy import deepcopy
import json
import numpy as np
from scripts.runtime.pcv_schema import baseline_cases, canonical, command, validate_command
from vcm.catalog import number_words
PERCENTAGES = (0, 50, 100)
ALARM_TIMES = ('06:00', '12:30', '21:45')


def demo_cases():
    rows = []
    for old in baseline_cases():
        row = deepcopy(old)
        row['id'], row['source_definition_id'] = 'DEMO-' + old['id'], old['id']
        cmd = row['command']
        if cmd['intent'] == 'BRIGHTNESS':
            prior = cmd['slots']['percent']
            new = {20:0, 60:50, 100:100}[prior]
            row['text'] = row['text'].replace(f'{prior} percent', f'{new} percent')
            row['spoken_text'] = row['spoken_text'].replace(number_words(prior) + ' percent', number_words(new) + ' percent')
            cmd['slots']['percent'] = new
        elif cmd['intent'] == 'ALARM':
            prior = cmd['slots']['time']
            literal, spoken, time = {
                '06:00':('6:00 AM','six a m','06:00'),
                '08:00':('12:30 PM','twelve thirty p m','12:30'),
                '21:00':('9:45 PM','nine forty five p m','21:45')}[prior]
            old_literal, old_spoken = {'06:00':('6:00 AM','six a m'), '08:00':('8:00 AM','eight a m'),
                                      '21:00':('9:00 PM','nine p m')}[prior]
            row['text'] = row['text'].replace(old_literal, literal)
            row['spoken_text'] = row['spoken_text'].replace(old_spoken, spoken)
            cmd['slots']['time'] = time
        validate_command(cmd)
        rows.append(row)
    for n in PERCENTAGES:
        for i, template in enumerate(('Set volume to {} percent','Volume {} percent','Adjust volume to {} percent')):
            rows.append(dict(id=f'DEMO-VOLUME-{n}-{i}', source_definition_id=None,
                text=template.format(n), spoken_text=template.format(number_words(n)),
                command=command('VOLUME', operation='set', percent=n), tier='demo'))
    return rows


COMMANDS = (None, *(json.loads(k) for k in sorted({canonical(r['command']) for r in demo_cases()})))
_KEYS = tuple(canonical(c) for c in COMMANDS)


def action_template(cmd):
    value = deepcopy(cmd)
    slot = None
    if value['intent'] == 'BRIGHTNESS' or (value['intent'] == 'VOLUME' and value['slots']['operation'] == 'set'):
        value['slots'].pop('percent')
        slot = 'percent'
    elif value['intent'] == 'ALARM':
        value['slots'].pop('time')
        slot = 'alarm'
    return dict(command=value, slot=slot)


ACTIONS = (None, *(json.loads(k) for k in sorted({canonical(action_template(c)) for c in COMMANDS[1:]})))
_ACTION_KEYS = tuple(canonical(a) for a in ACTIONS)


def targets(cmd):
    if cmd is not None:
        validate_command(cmd)
    key = canonical(cmd)
    if key not in _KEYS or cmd is None:
        return dict(class_index=0, action=0, percent=0, alarm=0)
    template = action_template(cmd)
    return dict(class_index=_KEYS.index(key), action=_ACTION_KEYS.index(canonical(template)),
        percent=PERCENTAGES.index(cmd['slots']['percent']) + 1 if template['slot'] == 'percent' else 0,
        alarm=ALARM_TIMES.index(cmd['slots']['time']) + 1 if template['slot'] == 'alarm' else 0)


def command_for_heads(action, percent, alarm):
    if any(type(v) is not int or not 0 <= v < n for v,n in ((action,len(ACTIONS)),(percent,4),(alarm,4))):
        raise ValueError('Invalid demo head indices')
    if action == 0:
        return None
    template = ACTIONS[action]
    cmd = deepcopy(template['command'])
    if template['slot'] == 'percent':
        if percent == 0:
            return None
        cmd['slots']['percent'] = PERCENTAGES[percent - 1]
    elif template['slot'] == 'alarm':
        if alarm == 0:
            return None
        cmd['slots']['time'] = ALARM_TIMES[alarm - 1]
    validate_command(cmd)
    return cmd


def decode_scores(scores, mode):
    scores = np.asarray(scores)
    size = len(COMMANDS) if mode == 'flat' else len(ACTIONS) + 8
    if mode not in ('flat','heads') or scores.shape != (size,) or not np.isfinite(scores).all():
        raise ValueError('Invalid demo logits/mode')
    if mode == 'flat':
        index = int(scores.argmax())
        return dict(class_index=index, command=deepcopy(COMMANDS[index]))
    a = int(scores[:len(ACTIONS)].argmax())
    p = int(scores[len(ACTIONS):len(ACTIONS)+4].argmax())
    t = int(scores[-4:].argmax())
    cmd = command_for_heads(a,p,t)
    return dict(class_index=_KEYS.index(canonical(cmd)), command=cmd,
                heads=dict(action=a, percent=p, alarm=t))
