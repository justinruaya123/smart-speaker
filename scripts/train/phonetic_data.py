"""Version two: original speech, preserved legacy test and richer speaker holdouts."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from einops import reduce
from .parameter_study_data import ROOT, dump, digest, read_rows, normalize, fraction, lexicon, phones
from .levels import percentage, spoken_number

DATA = ROOT / 'data/phonetic-v2'
OUT = ROOT / '.cache/phonetic-v2'


def waveform(row):
    x, rate = sf.read(ROOT / row['audio'], dtype='float32')
    if x.ndim == 2:
        x = reduce(x, 'time channel -> time', 'mean')
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        x = resample_poly(x, 16000 // divisor, rate // divisor).astype('float32')
    if not np.isfinite(x).all() or not 1600 <= len(x) <= 320000:
        raise ValueError('Expected original finite 0.1-20 second recording')
    return x


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = DATA / 'manifest.jsonl'
    if manifest.exists():
        audit = json.loads((DATA / 'audit.json').read_text())
        if digest(manifest) != audit['manifest_sha256']:
            raise ValueError('Frozen v2 manifest changed')
        if digest(DATA/'catalogs.json') != audit['catalog_sha256']:
            raise ValueError('Declared study catalog changed')
        frozen = read_rows(manifest)
        for row in frozen:
            if digest(ROOT/row['audio']) != row['audio_sha256']:
                raise ValueError('Original audio changed: '+row['audio'])
        return frozen
    if not (ROOT / 'STT/phonetic-v2/acquisition.json').exists():
        raise ValueError('Complete and verify acquisition first')
    rows = read_rows(ROOT / 'data/parameter-study/manifest.jsonl')
    for r in rows:
        r['legacy_split'] = r['split']
        r['benchmark'] = 'legacy'
    old_splits = {r['speaker']: ('train' if r['split'] == 'enroll' else r['split']) for r in rows}
    old_test = {r['id'] for r in rows if r['split'] == 'test'}
    catalogs = json.loads((ROOT / 'data/parameter-study/catalogs.json').read_text())
    reserved = {item['value'] for item in catalogs['novel']}
    raw = read_rows(ROOT / 'data/asr.jsonl')
    unique = {r['audio']: r for r in reversed(raw)}
    lights = [r for r in unique.values() if r['source'] == 'SnipsSmartLights' and r['audio'].endswith('_close.wav')]
    number_counts = Counter()
    for r in lights:
        try:
            value = percentage(normalize(r['transcript']))
        except ValueError:
            value = None
        if value is not None:
            number_counts[r['speaker_id']] += 1
    eligible = sorted((s for s, n in number_counts.items() if n >= 25 and old_splits.get(s, 'train') == 'train'),
                      key=lambda s: fraction('phonetic-v2-command-holdout/' + s))
    if len(eligible) < 3:
        raise ValueError('Need three independent well-covered numeric speakers')
    holdout = {eligible[0]: 'dev', eligible[1]: 'test'}
    # These supplemental speakers were seen by v1. Only freshly fitted v2
    # models may be evaluated on this new holdout. Legacy test remains fixed.
    for r in rows:
        if r['speaker'] in holdout:
            r['split'] = holdout[r['speaker']]
            r['benchmark'] = 'supplemental_command'
    paths = {r['audio'] for r in rows}
    for r in lights:
        if r['audio'] in paths:
            continue
        speaker = r['speaker_id']
        split = holdout.get(speaker, old_splits.get(speaker))
        if split is None:
            f = fraction('parameter-study/' + speaker)
            split = 'train' if f < .75 else 'dev' if f < .87 else 'test'
        text = normalize(r['transcript'])
        try:
            value = percentage(text)
        except ValueError:
            value = None
        targets = [] if value is None else [dict(role='number', value=str(value), surface=spoken_number(value))]
        rows.append(dict(id=r['id'], audio=r['audio'], text=text, speaker=speaker, split=split,
                         source=r['source'], license=r['license'], kind='natural', targets=targets,
                         command=r['command'], timing_verified=False, benchmark='supplemental_command', legacy_split=None))
        paths.add(r['audio'])
    # Sampling uses identifiers and duration only. No recognition outcomes.
    dictionary = lexicon()
    libri = defaultdict(list)
    omitted = Counter()
    for transcript in sorted((ROOT / 'data/audio/librispeech/LibriSpeech/train-clean-100').rglob('*.trans.txt')):
        for line in transcript.read_text().splitlines():
            identifier, text = line.split(' ', 1)
            path = transcript.parent / (identifier + '.flac')
            relative = path.relative_to(ROOT).as_posix()
            if relative in paths:
                continue
            text = normalize(text)
            if phones(text, dictionary) is None:
                omitted['dictionary_oov'] += 1
                continue
            duration = sf.info(path).duration
            if not .1 <= duration <= 20:
                omitted['duration'] += 1
                continue
            speaker = 'libri/' + identifier.split('-')[0]
            split = old_splits.get(speaker)
            if split is None:
                f = fraction('phonetic-v2-libri/' + speaker)
                split = 'train' if f < .8 else 'dev' if f < .9 else 'test'
            if any(' ' + name + ' ' in ' ' + text + ' ' for name in reserved):
                omitted['reserved_name'] += 1
                continue
            libri[split].append(dict(id='libri-v2/' + identifier, audio=relative, text=text, speaker=speaker,
                                    split=split, source='LibriSpeech-expanded', source_split='train-clean-100', license='CC-BY-4.0',
                                    kind='natural', targets=[], command=None, duration=duration,
                                    timing_verified=False, benchmark='general_speech', legacy_split=None))
    for split, hours in [('train', 30), ('dev', 1), ('test', 1)]:
        seconds = 0
        for r in sorted(libri[split], key=lambda x: fraction('phonetic-v2-sample/' + x['id'])):
            if seconds + r['duration'] > hours * 3600:
                continue
            rows.append(r)
            seconds += r['duration']
    hashes = defaultdict(list)
    for i, r in enumerate(rows):
        duration = r['duration'] if 'duration' in r else sf.info(ROOT / r['audio']).duration
        if not .1 <= duration <= (20 if r['benchmark'] == 'general_speech' else 8):
            omitted['duration'] += 1
            continue
        r['duration'] = duration
        r['audio_sha256'] = digest(ROOT / r['audio'])
        if r['split'] == 'train' and any(' ' + name + ' ' in ' ' + r['text'] + ' ' for name in reserved):
            r['split'] = 'enroll'
        hashes[r['audio_sha256']].append(r)
        if i % 5000 == 0:
            print('v2 audio audit', i, '/', len(rows), flush=True)
    rows = []
    for group in hashes.values():
        if len({(r['text'], r['split']) for r in group}) > 1:
            raise ValueError('Conflicting duplicate recording')
        rows.append(group[0])
    rows.sort(key=lambda r: r['id'])
    speaker_splits = defaultdict(set)
    for r in rows:
        speaker_splits[r['speaker']].add('train' if r['split'] == 'enroll' else r['split'])
    leaks = {s: sorted(v) for s, v in speaker_splits.items() if len(v) > 1}
    if leaks:
        raise ValueError(leaks)
    if old_test != {r['id'] for r in rows if r['legacy_split'] == 'test'}:
        raise ValueError('Legacy test membership changed')
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    dump(DATA / 'catalogs.json', catalogs)
    audit = dict(rows=len(rows), by_split=dict(Counter(r['split'] for r in rows)),
                 hours={s: sum(r['duration'] for r in rows if r['split'] == s) / 3600 for s in ('train', 'enroll', 'dev', 'test')},
                 sources={s: dict(Counter(r['split'] for r in rows if r['source'] == s)) for s in sorted({r['source'] for r in rows})},
                 numeric_commands={s: dict(Counter(t['value'] for r in rows if r['split'] == s and r['kind'] == 'natural' for t in r['targets'] if t['role'] == 'number')) for s in ('train', 'dev', 'test')},
                 supplemental_holdout_speakers=holdout, cross_split_speakers=leaks, omitted=dict(omitted),
                 manifest_sha256=digest(manifest), legacy_manifest_sha256=digest(ROOT / 'data/parameter-study/manifest.jsonl'),
                 legacy_test_preserved=len(old_test), synthetic_audio=0, pretrained_weights=False,
                 librispeech_official_benchmark=False, reserved_names=list(sorted(reserved)),
                 reserved_name_training_overlap=sum(any(' '+name+' ' in ' '+r['text']+' ' for name in reserved) for r in rows if r['split']=='train'),
                 catalog_sha256=digest(DATA/'catalogs.json'))
    dump(DATA / 'audit.json', audit)
    print(json.dumps(audit), flush=True)
    return rows
