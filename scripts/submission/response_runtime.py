"""Pinned fixed recognition replies; inference requires only verified WAV playback."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

from scripts.submission.common import ROOT, fetch, read, sha

RESPONSE_TEXTS = {
    'recognized_play_music':'I recognized a request to play music.',
    'recognized_weather':'I recognized a weather request.',
    'recognized_time':'I recognized a request for the time.',
    'recognized_light_on':'I recognized a request to turn the lights on.',
    'recognized_light_off':'I recognized a request to turn the lights off.',
    'recognized_pause':'I recognized a request to pause playback.',
    'recognized_stop':'I recognized a request to stop playback.',
    'recognized_next':'I recognized a request for the next song.',
    'recognized_volume':'I recognized a volume request.',
    'recognized_call':'I recognized a call request.',
    'recognized_message':'I recognized a message request.',
    'recognized_list_reminders':'I recognized a request to list reminders.',
    'recognized_timer':'I recognized a timer request.',
    'recognized_alarm':'I recognized an alarm request.',
    'recognized_temperature':'I recognized a temperature request.',
    'recognized_brightness':'I recognized a brightness request.',
    'recognized_color':'I recognized a color request.',
    'recognized_create_reminder':'I recognized a request to create a reminder.',
    'not_recognized':"I couldn't recognize that command. Please try again.",
    'capture_fault':"I couldn't hear a usable command. Please check the microphone.",
}


def response_id(event):
    if event.get('event') not in (None, 'command') or 'output' not in event:
        return None
    result = event['output']
    app = event.get('application')
    if result.get('reason') == 'capture_fault':
        return 'capture_fault'
    accepted = result.get('accepted') is True and (app is None or app.get('accepted') is True)
    command = result.get('command')
    if accepted and isinstance(command, dict):
        key = 'recognized_' + str(command.get('intent', '')).lower()
        if key in RESPONSE_TEXTS:
            return key
    return 'not_recognized'


def validate_manifest(manifest):
    if manifest.get('format') != 'smart-speaker-fixed-responses-v1':
        raise ValueError('Unsupported recorded-response manifest')
    hub = manifest.get('hub', {})
    if hub.get('repo_type', 'model') not in ('model', 'dataset'):
        raise ValueError('Response repository must be a model or dataset')
    if not re.fullmatch(r'[\w.-]+/[\w.-]+', hub.get('repo_id', '')):
        raise ValueError('Invalid response repository')
    if not re.fullmatch(r'[a-f0-9]{40}', hub.get('revision', '')):
        raise ValueError('Response download must use a pinned 40-character commit')
    if hub.get('subdirectory') != 'responses':
        raise ValueError('Only the response subdirectory may be downloaded')
    rows = manifest.get('responses')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing recorded responses')
    seen = set()
    for row in rows:
        key = row.get('id'); path = PurePosixPath(row.get('path', ''))
        if key not in RESPONSE_TEXTS or key in seen:
            raise ValueError('Unknown or repeated recorded-response ID')
        seen.add(key)
        if (path.parts != ('audio', key + '.wav') or '\\' in row.get('path', '')
                or not re.fullmatch(r'[a-f0-9]{64}', row.get('sha256', ''))
                or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 5_000_000
                or row.get('text') != RESPONSE_TEXTS[key]):
            raise ValueError('Invalid recorded-response path or identity')
    return manifest


def ensure_responses(root=ROOT, manifest=None):
    root = Path(root)
    manifest = validate_manifest(manifest or read(root/'submission/responses/manifest.json'))
    hub = manifest['hub']; result = {}
    prefix = 'datasets/' if hub.get('repo_type') == 'dataset' else ''
    for row in manifest['responses']:
        packaged = root/'submission/responses'/row['path']
        cached = root/'.cache/submission/responses'/row['path']
        path = packaged if packaged.exists() else cached
        if not path.exists():
            url = (f"https://huggingface.co/{prefix}{hub['repo_id']}/resolve/{hub['revision']}/"
                   f"responses/{row['path']}")
            fetch(url, path, row['sha256'])
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Recorded response checksum mismatch: ' + row['id'])
        result[row['id']] = path
    return result

