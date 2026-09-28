"""Action-free ten-use-case scope probe; synthetic, never used for selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'experiments/vcm_streaming_v3'
V6 = ROOT / 'experiments/vcm_services_v6'
sys.path.insert(0, str(V6 / 'src'))
from service_stream import ServiceStream, json_safe, load_pcm

CASES = [
    (1, 'Music', 'play quiet river by luna rose', {'intent': 'PLAY_TRACK', 'track_id': 'dummy_track_001', 'artist_id': 'dummy_artist_006'}, True),
    (1, 'Music', 'quiet river', None, True),
    (2, 'Questions/search', 'what is the weather today', {'intent': 'WEATHER_QUERY'}, False),
    (2, 'Questions/search', 'what time is it', {'intent': 'TIME_QUERY'}, False),
    (3, 'Lights on/off', 'turn on bedroom lamp', {'intent': 'LIGHT_ON', 'entity_id': 'light.bedroom_lamp'}, True),
    (3, 'Lights on/off', 'turn off kitchen lamp', {'intent': 'LIGHT_OFF', 'entity_id': 'light.kitchen_lamp'}, True),
    (4, 'Brightness/color', 'set brightness of lamp two to seventy three percent', {'intent': 'SET_BRIGHTNESS', 'entity_id': 'light.lamp_two', 'value': 73, 'unit': 'percent'}, True),
    (4, 'Brightness/color', 'set bedroom lamp to red', {'intent': 'SET_COLOR', 'entity_id': 'light.bedroom_lamp', 'color': 'red'}, False),
    (5, 'Timers', 'set a timer for five minutes', {'intent': 'TIMER_SET', 'duration_seconds': 300}, False),
    (5, 'Timers', 'cancel the timer', {'intent': 'TIMER_CANCEL'}, False),
    (6, 'Alarms', 'set an alarm for seven thirty am', {'intent': 'ALARM_SET', 'hour': 7, 'minute': 30, 'period': 'am'}, False),
    (6, 'Alarms', 'snooze the alarm', {'intent': 'ALARM_SNOOZE'}, False),
    (7, 'Thermostat', 'set the thermostat to twenty two degrees celsius', {'intent': 'SET_TEMPERATURE', 'value': 22, 'unit': 'celsius'}, False),
    (7, 'Thermostat', 'increase the temperature by three degrees', {'intent': 'ADJUST_TEMPERATURE', 'delta': 3}, False),
    (8, 'Media controls/volume', 'set volume to forty two percent', {'intent': 'SET_VOLUME', 'entity_id': None, 'value': 42, 'unit': 'percent'}, True),
    (8, 'Media controls/volume', 'resume the music', {'intent': 'MEDIA_RESUME'}, False),
    (9, 'Reminders/lists', 'remind me to water the plants', {'intent': 'REMINDER_ADD', 'text': 'water the plants'}, False),
    (9, 'Reminders/lists', 'add milk to my shopping list', {'intent': 'LIST_ADD', 'item': 'milk'}, False),
    (10, 'Calls/messages', 'call alice', {'intent': 'CALL', 'contact': 'alice'}, False),
    (10, 'Calls/messages', 'send a message to bob', {'intent': 'MESSAGE_COMPOSE', 'contact': 'bob'}, False),
    (2, 'Questions/search', 'what is the date today', {'intent': 'DATE_QUERY'}, False),
    (8, 'Media controls/volume', 'do not set volume to fifty', None, True),
]


def run() -> dict:
    audio = ROOT / 'final_suite/data/scope_audio'
    audio.mkdir(parents=True, exist_ok=True)
    graph = json.loads((V6 / 'data/services_compiled.json').read_text())
    forms = {f['text']: f['command'] for f in graph['forms']}
    model = V3 / 'models/phone_s53_40ms.onnx'
    stream = ServiceStream(model, V6 / 'data/services_compiled.json',
                           V6 / 'data/ha_snapshot.json', V6 / 'data/spotify_tracks.json')
    rows = []
    for voice in ['en-us', 'en-gb+f3']:
        for number, family, text, wanted, supported in CASES:
            if supported and text in forms and wanted is not None:
                wanted = forms[text]
            identity = hashlib.sha256(f'{voice}|165|51|{text}'.encode()).hexdigest()[:20]
            wav = audio / f'{identity}.wav'
            subprocess.run(['espeak', '-v', voice, '-s', '165', '-p', '51',
                            '-w', str(wav), text], check=True, capture_output=True)
            pcm = load_pcm(wav)
            stream.reset()
            for start in range(0, len(pcm), 640):
                stream.push(pcm[start:start + 640])
            result = json_safe(stream.finish('push_to_talk_release'))
            rows.append({
                'id': identity, 'trial': 'final_scope_probe',
                'split': 'diagnostic_not_selection', 'voice': voice,
                'text': text, 'use_case': number, 'family': family,
                'target': wanted, 'candidate': result.get('candidate'),
                'status': result['status'], 'ties': result.get('ties', []),
                'supported_by_grammar': supported,
                'category': 'ambiguous' if text == 'quiet river' else
                            'supported' if supported else 'out_of_scope',
                'accepted': False, 'action_executed': False,
                'pcm_sha256': hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
                'wav': str(wav.relative_to(ROOT)), 'result': result,
            })
    output = {'protocol': 'Two synthetic voice settings; no fitting or selection',
              'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
              'graph_hash': graph['graph_hash'], 'recordings': len(rows),
              'all_ten_use_cases_passed': False, 'rows': rows}
    (ROOT / 'final_suite/reports/scope_probe.json').write_text(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    for row in run()['rows']:
        print(row['voice'], row['text'], '=>', row['candidate'], row['status'])
