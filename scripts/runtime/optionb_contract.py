"""Shared finite class order for OptionB-only joint-command research.

Class zero rejects. A valid command outside the 31 allowed complete commands
is a task negative, never a partial intent match. No speech text is parsed.
"""
from copy import deepcopy
import json

from scripts.runtime.pcv_schema import baseline_cases, canonical, validate_command

COMMANDS = (None, *(json.loads(key) for key in sorted({canonical(c['command']) for c in baseline_cases()})))
_KEYS = tuple(canonical(command) for command in COMMANDS)


def class_index(command):
    if command is not None:
        validate_command(command)
    key = canonical(command)
    return _KEYS.index(key) if key in _KEYS else 0


def command_for_index(index):
    if type(index) is not int or not 0 <= index < len(COMMANDS):
        raise ValueError('Invalid complete-command class index')
    return deepcopy(COMMANDS[index])
