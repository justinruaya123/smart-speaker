"""Finite command schema and reference wordings; no audio transcription."""
from __future__ import annotations

import hashlib
import json
import re

def number_words(n):
    if type(n) is not int or not 0<=n<=100: raise ValueError('Integer 0..100 required')
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    if n<20: return units[n]
    if n==100: return 'one hundred'
    tens='zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    return tens[n//10]+(' '+units[n%10] if n%10 else '')


INTENTS = ['PLAY_MUSIC', 'WEATHER', 'TIME', 'LIGHT_ON', 'LIGHT_OFF', 'PAUSE', 'STOP', 'NEXT',
           'VOLUME', 'CALL', 'MESSAGE', 'LIST_REMINDERS', 'TIMER', 'ALARM', 'TEMPERATURE',
           'BRIGHTNESS', 'COLOR', 'CREATE_REMINDER']
DEFAULT_CUSTOM = {
    'version': 'PCV-B1-demo-1',
    'lights': [
        {'id': 'light.kitchen', 'name': 'kitchen lamp', 'aliases': ['kitchen lights', 'kitchen light']},
        {'id': 'light.bedroom', 'name': 'bedroom lamp', 'aliases': ['bedroom lights', 'bedroom light']},
        {'id': 'light.bathroom', 'name': 'bathroom light', 'aliases': ['bathroom lights']}],
    'songs': [{'id': 'song.dynamite', 'name': 'Dynamite', 'artist': 'BTS', 'artist_id': 'artist.bts'},
              {'id': 'song.butter', 'name': 'Butter', 'artist': 'BTS', 'artist_id': 'artist.bts'}],
    'contacts': [{'id': 'contact.home', 'name': 'Home', 'aliases': []},
                 {'id': 'contact.mom', 'name': 'Mom', 'aliases': ['mother']}],
    'tasks': [{'id': 'task.drink_water', 'name': 'Drink water'},
              {'id': 'task.study', 'name': 'Study'}, {'id': 'task.exercise', 'name': 'Exercise'}],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def normalize_text(text):
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()))


def command(intent, **slots):
    result = {'intent': intent, 'slots': slots}
    validate_command(result)
    return result


def default_slots(intent, explicit_name, custom=None):
    custom = custom or DEFAULT_CUSTOM
    kind, field, fallback = ('contacts', 'contact_id', 'contact.home') if intent == 'CALL' else ('lights', 'light_id', 'light.all')
    if explicit_name is None:
        return {field: fallback}
    matches = [r for r in custom[kind] if normalize_text(explicit_name) in
               {normalize_text(n) for n in [r['name'], *r.get('aliases', [])]}]
    if len(matches) != 1:
        raise ValueError('invalid_argument: unknown or ambiguous name')
    return {field: matches[0]['id']}


def validate_command(value):
    if not isinstance(value, dict) or set(value) != {'intent', 'slots'} or value['intent'] not in INTENTS:
        raise ValueError('Invalid command envelope')
    intent, slots = value['intent'], value['slots']
    if not isinstance(slots, dict):
        raise ValueError('Slots must be an object')
    required = {'LIGHT_ON': {'light_id'}, 'LIGHT_OFF': {'light_id'}, 'CALL': {'contact_id'},
                'VOLUME': {'operation'}, 'TIMER': {'duration_seconds'}, 'ALARM': {'time', 'timezone'},
                'TEMPERATURE': {'degrees', 'unit'}, 'BRIGHTNESS': {'light_id', 'percent'},
                'COLOR': {'light_id', 'color'}, 'CREATE_REMINDER': {'task_id'}}.get(intent, set())
    optional = {'PLAY_MUSIC': {'song_id', 'artist_id'}, 'VOLUME': {'percent'}}.get(intent, set())
    if not required <= set(slots) or not set(slots) <= required | optional:
        raise ValueError('Invalid required or extra slots')
    for field, item in slots.items():
        if field in {'percent', 'duration_seconds', 'degrees'}:
            low, high = (1, 86400) if field == 'duration_seconds' else ((-50, 60) if field == 'degrees' else (0, 100))
            if type(item) is not int or not low <= item <= high:
                raise ValueError('Invalid numeric type or bound')
        elif type(item) is not str or not item:
            raise ValueError('Invalid text/identifier type')
        elif field.endswith('_id') and not re.fullmatch(r'[a-z]+\.[a-z0-9_.-]+', item):
            raise ValueError('Invalid stable identifier')
        elif field.endswith('_id') and not item.startswith(field[:-3] + '.'):
            raise ValueError('Identifier has the wrong role')
    if intent == 'VOLUME' and (slots['operation'] not in {'up', 'down', 'mute', 'set'} or
                              (slots['operation'] == 'set') != ('percent' in slots)):
        raise ValueError('Invalid volume operation/percentage')
    if intent == 'PLAY_MUSIC' and 'artist_id' in slots and 'song_id' not in slots:
        raise ValueError('Artist requires a song')
    if intent == 'ALARM' and (not re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]', slots['time']) or slots['timezone'] != 'Asia/Manila'):
        raise ValueError('Invalid clock time or benchmark timezone')
    if intent == 'TEMPERATURE' and slots['unit'] != 'C':
        raise ValueError('Unsupported temperature unit')
    if intent == 'COLOR' and slots['color'] not in {'red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'pink', 'cyan', 'magenta', 'warm white', 'cool white'}:
        raise ValueError('Unsupported color')


def output(value, reason=None):
    if value is not None:
        validate_command(value)
        if reason is not None:
            raise ValueError('Accepted commands have no rejection reason')
    elif reason not in {'OOV', 'invalid_argument', 'capture_fault', 'unqualified_model'}:
        raise ValueError('A rejection requires an explicit reason')
    return {'schema_version': 3, 'accepted': value is not None, 'command': value, 'reason': reason}


def baseline_cases():
    fixed = [
        ('PLAY_MUSIC', ['Play music', 'Start music', 'Play some music'], {}),
        ('WEATHER', ['Weather', "What's the weather?", 'Tell me the weather'], {}),
        ('TIME', ['Time', 'What time is it?', 'Tell me the time'], {}),
        ('LIGHT_ON', ['Lights on', 'Power on the lights', 'Turn on the lights'], {'light_id': 'light.all'}),
        ('LIGHT_OFF', ['Lights out', 'Kill the lights', 'Shut off the lights'], {'light_id': 'light.all'}),
        ('PAUSE', ['Pause', 'Pause audio', 'Pause song'], {}),
        ('STOP', ['Stop', 'Stop playing', 'End playback'], {}),
        ('NEXT', ['Next song', 'Skip song', 'Play next song'], {}),
        ('VOLUME', ['Volume up', 'Increase the volume', 'Turn the volume up'], {'operation': 'up'}),
        ('VOLUME', ['Volume down', 'Lower the volume', 'Turn the volume down'], {'operation': 'down'}),
        ('CALL', ['Call', 'Make a call', 'Make a phone call'], {'contact_id': 'contact.home'}),
        ('MESSAGE', ['Message', 'Send a message', 'Send my message'], {}),
        ('LIST_REMINDERS', ['Reminders', 'Show my reminders', 'List my reminders'], {}),
    ]
    rows = []
    def add(text, cmd, spoken=None):
        rows.append({'id': f'B1-{len(rows)+1:03d}', 'text': text, 'spoken_text': spoken or text,
                     'command': cmd, 'tier': 'baseline'})
    for intent, phrases, slots in fixed:
        for phrase in phrases:
            add(phrase, command(intent, **slots))
    slotted = [
        ('TIMER', ['Timer {}', 'Countdown for {}', 'Start a timer for {}'],
         [('10 seconds', 'ten seconds', {'duration_seconds': 10}), ('30 seconds', 'thirty seconds', {'duration_seconds': 30}), ('1 minute', 'one minute', {'duration_seconds': 60})]),
        ('ALARM', ['Alarm {}', 'Wake me up at {}', 'Set an alarm for {}'],
         [('6:00 AM', 'six a m', {'time': '06:00', 'timezone': 'Asia/Manila'}), ('8:00 AM', 'eight a m', {'time': '08:00', 'timezone': 'Asia/Manila'}), ('9:00 PM', 'nine p m', {'time': '21:00', 'timezone': 'Asia/Manila'})]),
        ('TEMPERATURE', ['Temperature {}', 'Change the temperature to {}', 'Set the temperature to {}'],
         [(f'{n} degrees', number_words(n) + ' degrees', {'degrees': n, 'unit': 'C'}) for n in (18, 22, 26)]),
        ('BRIGHTNESS', ['Brightness {}', 'Adjust brightness to {}', 'Brightness level {}'],
         [(f'{n} percent', number_words(n) + ' percent', {'light_id': 'light.all', 'percent': n}) for n in (20, 60, 100)]),
        ('COLOR', ['Change color to {}', 'Switch color to {}', 'Set color to {}'],
         [(c.title(), c, {'light_id': 'light.all', 'color': c}) for c in ('red', 'blue', 'green')]),
        ('CREATE_REMINDER', ['Reminder {}', 'Remind me to {}', 'Create a reminder to {}'],
         [(r['name'], r['name'], {'task_id': r['id']}) for r in DEFAULT_CUSTOM['tasks']]),
    ]
    for intent, templates, values in slotted:
        for template in templates:
            for literal, spoken, slots in values:
                add(template.format(literal), command(intent, **slots), template.format(spoken))
    return rows


def extended_cases(custom=None):
    custom = custom or DEFAULT_CUSTOM
    rows = []
    def add(text, cmd, reason=None):
        rows.append({'id': 'X-' + hashlib.sha256(text.lower().encode()).hexdigest()[:14],
                     'text': text, 'command': cmd, 'tier': 'extension', 'reason': reason})
    for light in custom['lights']:
        for name in [light['name'], *light.get('aliases', [])]:
            for verb, intent in [('turn on', 'LIGHT_ON'), ('turn off', 'LIGHT_OFF')]:
                add(f'{verb} the {name}', command(intent, light_id=light['id']))
                add(f'{verb} {name}', command(intent, light_id=light['id']))
            for n in (0, 20, 60, 100):
                for verb in ('adjust', 'set'):
                    add(f'{verb} brightness of {name} to {number_words(n)} percent', command('BRIGHTNESS', light_id=light['id'], percent=n))
    for n in range(101):
        add(f'set volume to {number_words(n)} percent', command('VOLUME', operation='set', percent=n))
        add(f'set brightness to {number_words(n)} percent', command('BRIGHTNESS', light_id='light.all', percent=n))
    for phrase in ('mute', 'mute volume', 'mute the volume'):
        add(phrase, command('VOLUME', operation='mute'))
    for song in custom['songs']:
        add('play ' + song['name'], command('PLAY_MUSIC', song_id=song['id']))
        add(f"play {song['name']} by {song['artist']}", command('PLAY_MUSIC', song_id=song['id'], artist_id=song['artist_id']))
    for contact in custom['contacts']:
        add('call ' + contact['name'], command('CALL', contact_id=contact['id']))
    for phrase in ('turn on the mystery light', 'turn off the attic light', 'set brightness of garage light to twenty percent',
                   'set brightness to one hundred and one percent', 'set volume to minus one percent',
                   'call unknown contact', 'play unknown song', 'remind me to unknown task'):
        add(phrase, None, 'invalid_argument')
    for phrase in ('buy me a car', 'tell me a joke', 'this is not a command', 'do not turn on the lights',
                   'what is the population of mars', 'delete all my files'):
        add(phrase, None, 'OOV')
    return rows
