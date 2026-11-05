"""Training/reference curation only. Never import this as an audio text decoder.

Match complete, finite phrases. Unknown tokens, slots, negation and units are
never erased on a heuristic word-count rule. Harmless filler is allowlisted.
"""
from collections import defaultdict
from functools import lru_cache
import re

from scripts.runtime.optionb_contract import class_index, command_for_index
from scripts.runtime.pcv_schema import baseline_cases
from vcm.catalog import number_words

FILLERS = frozenset({'um', 'uh', 'please'})
PREFIXES = (('hey', 'iris'), ('can', 'you'), ('could', 'you'), ('would', 'you'),
            ('okay',), ('ok',))


def spoken_tokens(text):
    text = text.lower().replace('\u2019', "'").replace('%', ' percent ')
    text = re.sub(r'(?<=\d):00\b', '', text)
    text = re.sub(r'\b([ap])\s*\.?\s*m\.?\b', r'\1 m', text)
    def numeric(match):
        value = int(match.group())
        return ' ' + (number_words(value) if value <= 100 else match.group()) + ' '
    text = re.sub(r'\d+', numeric, text)
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return tuple({'light': 'lights', 'degree': 'degrees'}.get(t, t) for t in tokens)


@lru_cache(maxsize=1)
def wording_inventory():
    index = {}
    for case in baseline_cases():
        for text in (case['text'], case['spoken_text']):
            tokens = spoken_tokens(text)
            target = dict(class_index=class_index(case['command']), definition_id=case['id'])
            if tokens in index and index[tokens]['class_index'] != target['class_index']:
                raise ValueError('Conflicting canonical phrase')
            index[tokens] = target
    return index


def wording_target(text):
    tokens = spoken_tokens(text)
    inventory = wording_inventory()
    if tokens in inventory:
        return {**inventory[tokens], 'tier': 'exact_or_literal_surface_variant'}
    stripped = tuple(token for token in tokens if token not in FILLERS)
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if stripped[:len(prefix)] == prefix:
                stripped = stripped[len(prefix):]
                changed = True
                break
    if stripped in inventory:
        return {**inventory[stripped], 'tier': 'explicit_benign_filler'}
    return None


class SourceVoiceSampler:
    """Source quotas, then command, voice and available wording uniformly.

    A thousand repeats from one voice do not outvote a single other voice.
    Synthetic voice identity still requires audited upstream provenance.
    """
    def __init__(self, rows):
        self.pools = {}
        for index, row in enumerate(rows):
            group = ('synthetic' if row['kind'] == 'synthetic' else 'human')
            group += '_positive' if row['class_index'] else '_negative'
            group = row.get('sampling_group', group)
            cell = row['class_index'] if row['class_index'] else row.get('scope', 'negative')
            voice = str(row['speaker'])
            wording = row.get('definition_id', 'unavailable')
            pool = self.pools.setdefault(group, {})
            pool.setdefault(cell, {}).setdefault(voice, {}).setdefault(wording, []).append(index)

    def batch(self, quotas, rng):
        if sum(quotas.values()) != 32 or any(type(n) is not int or n < 0 for n in quotas.values()):
            raise ValueError('Declare exactly 32 nonnegative source draws')
        result = []
        def choose(values):
            return values[int(rng.integers(len(values)))]
        for group, count in quotas.items():
            if count and not self.pools.get(group):
                raise ValueError('Missing required source pool: ' + group)
            for _ in range(count):
                cells = self.pools[group]
                voices = cells[choose(list(cells))]
                wordings = voices[choose(list(voices))]
                members = wordings[choose(list(wordings))]
                result.append(choose(members))
        rng.shuffle(result)
        return result


def synthesis_prompts():
    """Complete connected commands plus bounded, explicitly unsupported phrases."""
    prompts = []
    wrappers = (('{text}', 'clean'), ('Please, {text}', 'please_prefix'),
                ('{text}, please', 'please_suffix'), ('Um, {text}', 'um_prefix'))
    for case in baseline_cases():
        for template, form in wrappers:
            prompts.append(dict(id=case['id'] + '-' + form,
                                text=template.format(text=case['spoken_text']),
                                definition_id=case['id'], form=form,
                                class_index=class_index(case['command'])))
    groups = {
        'negation': ['Do not play music', "Don't start music", 'Do not pause audio',
                     'Do not stop playing', "Don't skip song", 'Do not turn on the lights',
                     'Do not shut off the lights', 'Do not increase the volume',
                     'Do not lower the volume', 'Do not make a call',
                     'Do not send a message', 'Do not show my reminders',
                     'Do not start a timer for ten seconds', 'Do not set an alarm for six a m',
                     'Do not set the temperature to twenty two degrees',
                     'Do not adjust brightness to sixty percent', 'Do not change color to red',
                     'Do not remind me to drink water'],
        'unsupported_slot': ['Timer twenty seconds', 'Timer thirteen seconds',
                             'Timer thirty minutes', 'Timer ten hours',
                             'Countdown for two minutes', 'Start a timer for fifteen seconds',
                             'Alarm seven a m', 'Alarm nine a m', 'Alarm six p m',
                             'Wake me up at five a m', 'Set an alarm for eight p m',
                             'Temperature nineteen degrees', 'Temperature twenty degrees',
                             'Temperature twenty three degrees', 'Temperature sixty degrees',
                             'Set the temperature to twenty two Fahrenheit',
                             'Brightness forty percent', 'Brightness sixteen percent',
                             'Brightness eighty percent', 'Brightness sixty degrees',
                             'Change color to yellow', 'Switch color to purple',
                             'Call Jane', 'Call James', 'Call Mom',
                             'Turn on the kitchen lights', 'Shut off the bedroom lights',
                             'Remind me to buy groceries'],
        'missing_slot': ['Timer', 'Countdown for', 'Start a timer for', 'Alarm',
                         'Wake me up at', 'Set an alarm for', 'Temperature',
                         'Set the temperature to', 'Brightness', 'Adjust brightness to',
                         'Change color to', 'Switch color to', 'Create a reminder to'],
        'general_oos': ['Tell me a joke', 'Buy me a car', 'This is not a command',
                        'What is the population of Mars', 'Delete all my files',
                        'Please', 'Um', 'Thank you', 'Good morning', 'I am reading a book'],
    }
    for category, texts in groups.items():
        for i, text in enumerate(texts):
            if wording_target(text) is not None:
                raise ValueError('A negative collided with a supported command: ' + text)
            prompts.append(dict(id=f'NEG-{category}-{i:03d}', text=text,
                                class_index=0, negative_type=category,
                                definition_id='NEG-' + category))
    return prompts


def sapi_tasks(config):
    tasks = []
    for split in ('train', 'dev'):
        voice = config[split + '_voice']
        slug = re.sub(r'[^a-z0-9]+', '-', voice.lower()).strip('-')
        for rate in config[split + '_rates']:
            for prompt in synthesis_prompts():
                identifier = f'sapi-{slug}-rate{rate}-{prompt["id"]}'
                positive = prompt['class_index'] > 0
                tasks.append({**prompt, 'id': identifier, 'voice': voice, 'rate': rate,
                    'relative_audio': f'{split}/{slug}/rate{rate}/{prompt["id"]}.wav',
                    'split': split, 'kind': 'synthetic', 'synthetic': True,
                    'speaker': 'sapi:' + voice, 'speaker_id': 'sapi:' + voice,
                    'source': 'System.Speech_offline',
                    'expected': command_for_index(prompt['class_index']),
                    'optionb_command': command_for_index(prompt['class_index']),
                    'training_eligible': split == 'train', 'metric_eligible': True,
                    'scope': 'optionb' if positive else 'synthetic_' + prompt['negative_type'],
                    'sampling_group': 'synthetic_filler_positive' if positive else 'synthetic_negative',
                    'reference_status': 'authored_synthetic_prompt_not_independent_human_listening'})
    return tasks
