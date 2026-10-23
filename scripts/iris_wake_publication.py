"""Explicit opt-in synthetic Hey Iris study; no runtime/actions or VCM fitting.

Raw provider output, decoded audio and feature/checkpoint intermediates belong
under the ignored .cache/iris-wake-publication-v1 directory, never in Git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/iris-wake-publication-v1'
CACHE = ROOT / '.cache/iris-wake-publication-v1'
BASELINE_VOICES = (
    'en-US-AndrewNeural', 'en-US-JennyNeural',
    'en-GB-RyanNeural', 'en-GB-LibbyNeural',
    'en-PH-JamesNeural', 'en-PH-RosaNeural',
    'fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural',
)
NEGATIVES = ('Hey Siri', 'Hey Jarvis', 'Hi Iris', 'Hey Irish', 'Hey Alice',
             'Iris flower', 'turn on the lights', 'ordinary conversation')


def baseline_cases(available):
    """Declare the whole cohort before touching any audio or model scores."""
    inventory = {v['ShortName']: v for v in available}
    missing = set(BASELINE_VOICES) - inventory.keys()
    if missing:
        raise ValueError('Missing required voices: ' + ', '.join(sorted(missing)))
    rows = []
    for voice in BASELINE_VOICES:
        for kind, texts, rates in (
            ('positive', ('Hey Iris', 'Hey, Iris'), ('-10%', '+10%')),
            ('negative', NEGATIVES, ('+0%',)),
        ):
            for text in texts:
                for rate in rates:
                    rows.append(dict(id=f'baseline-{len(rows):03d}', split='baseline',
                                     kind=kind, wake_positive=kind == 'positive',
                                     voice=voice, text=text, rate=rate, synthetic=True,
                                     voice_metadata={k: inventory[voice][k] for k in ('Gender', 'Locale')}))
    for name, amplitude in (('silence', 0), ('white_quiet', .001),
                            ('white_loud', .04), ('pink', .02)):
        rows.append(dict(id=f'baseline-control-{name}', split='baseline', kind='control',
                         wake_positive=False, synthetic=True, voice=None, text=None,
                         control=name, amplitude=amplitude, seconds=10, seed=731))
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    """Idempotence is allowed, changed protocol/receipts are never overwritten."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing immutable output change: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def verify_generated(case, receipt, wav):
    if receipt['case_sha256'] != object_digest(case):
        raise ValueError('Changed synthesis case binding')
    if digest(wav) != receipt['wav_sha256']:
        raise ValueError('Changed generated audio')


def freeze_baseline(available, *, out=EXPERIMENT):
    """The tracked protocol is written before any provider generation."""
    folder = ROOT / 'artifacts/iris/wake'
    policy = json.loads((folder / 'iris.json').read_text())
    result = dict(schema_version=1, experiment='iris-wake-publication-v1-baseline',
                  hypothesis='The unchanged native graph/gate passes a diverse synthetic smoke panel',
                  base_revision='c0628940400642f204ff98e26908549b1c07e41c',
                  positive_recall_minimum=.9, false_wakes_maximum=0,
                  duplicate_wakes_maximum=0, original_human_positive_n=0, qualified=False,
                  model_sha256=digest(folder / 'iris.onnx'), policy_sha256=digest(folder / 'iris.json'),
                  baseline_policy={k: policy[k] for k in ('threshold', 'release', 'consecutive', 'cooldown_samples')},
                  source_bindings={p: digest(ROOT / p) for p in (
                      'scripts/runtime/mvp_wake.py', 'scripts/models/mvp_wake.py',
                      'scripts/train/iris_wake_helpers.py', 'vcm/frontend.py')},
                  window_samples=25600, step_samples=1280,
                  stream='Reset each clip; 32000 leading and 48000 trailing zero samples; feed 640 samples; actual HeyIris CPU ONNX',
                  negative_exposure='Only original clip duration; padding excluded; not continuous household audio',
                  generation=dict(provider='Microsoft Edge TTS via edge-tts', version='7.3.1',
                                  concurrency=3, maximum_attempts=4, codec='ffmpeg PCM16 mono 16000Hz'),
                  rights='No raw TTS corpus redistribution. Voice metadata is not human demographic evidence.',
                  cases=baseline_cases(available))
    write_once(Path(out) / 'baseline-protocol.json', result)
    return result


async def generate_cases(cases, cache, protocol_sha256, *, synthesizer=None, decoder=None):
    """Only this explicitly invoked asynchronous stage contacts the TTS service."""
    import asyncio
    import array
    import random
    import subprocess
    import wave
    import sys

    cache = Path(cache)
    (cache / 'audio').mkdir(parents=True, exist_ok=True)
    (cache / 'receipts').mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(3)

    async def edge_synth(case, path):
        import edge_tts
        await edge_tts.Communicate(case['text'], case['voice'], rate=case['rate']).save(str(path))

    def ffmpeg_decode(mp3, wav):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp3), '-ac', '1',
                        '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)], check=True,
                       capture_output=True, timeout=45)

    async def one(case):
        if not case['synthetic'] or not case['id'].replace('-', '').replace('_', '').isalnum():
            raise ValueError('Expected closed synthetic case ID')
        wav = cache / 'audio' / (case['id'] + '.wav')
        mp3 = wav.with_suffix('.mp3')
        receipt_path = cache / 'receipts' / (case['id'] + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['protocol_sha256'] != protocol_sha256:
                raise ValueError('Changed generation protocol')
            verify_generated(case, receipt, wav)
            if case.get('voice') and digest(mp3) != receipt['mp3_sha256']:
                raise ValueError('Changed provider audio')
            return receipt
        async with semaphore:
            if case.get('voice'):
                for attempt in range(4):
                    try:
                        await (synthesizer or edge_synth)(case, mp3)
                        (decoder or ffmpeg_decode)(mp3, wav)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2 ** attempt)
            else:
                rng = random.Random(case['seed'])
                values = array.array('h')
                smooth = 0.
                for _ in range(round(case['seconds'] * 16000)):
                    value = rng.gauss(0, case['amplitude'])
                    if case['control'] == 'pink':
                        smooth = .98 * smooth + .02 * value
                        value = smooth * 6
                    values.append(round(max(-.99, min(.99, value)) * 32767))
                if sys.byteorder != 'little':
                    values.byteswap()
                with wave.open(str(wav), 'wb') as f:
                    f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    f.writeframes(values.tobytes())
            with wave.open(str(wav), 'rb') as f:
                if f.getnchannels() != 1 or f.getframerate() != 16000 or f.getsampwidth() != 2:
                    raise ValueError('Invalid synthesized WAV format')
                pcm = f.readframes(f.getnframes())
                samples = f.getnframes()
            receipt = dict(id=case['id'], case_sha256=object_digest(case),
                           protocol_sha256=protocol_sha256, wav_sha256=digest(wav),
                           decoded_pcm_s16le_sha256=hashlib.sha256(pcm).hexdigest(),
                           mp3_sha256=digest(mp3) if case.get('voice') else None,
                           samples=samples, seconds=samples / 16000, sample_rate=16000,
                           synthetic=True, generator='edge-tts==7.3.1' if case.get('voice') else 'seeded software control')
            write_once(receipt_path, receipt)
            print('GENERATED', case['id'], samples, flush=True)
            return receipt

    records = await asyncio.gather(*(one(case) for case in cases))
    write_once(cache / 'generation-complete.json', records)
    return records
