"""Audited original public audio and explicitly separate synthetic study data."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from einops import reduce

from vcm.frontend import SPEC, features
from vcm.catalog import dictionary, PHONES
from scripts.runtime.pcv_schema import (INTENTS, DEFAULT_CUSTOM, baseline_cases, extended_cases,
    normalize_text, canonical, command, compile_inventory)

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'experiments/pcv-human-slots-v1'
CACHE = ROOT / '.cache/pcv-human-slots-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf8')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def speaker_split(speaker):
    bucket = int(hashlib.sha256(('pcv-human-slots-v1/' + speaker).encode()).hexdigest()[:12], 16) % 1000
    return 'train' if bucket < 700 else ('dev' if bucket < 800 else ('calibration' if bucket < 900 else 'test'))


def audit_split_rows(rows):
    groups = {key: {} for key in ('speaker', 'pair', 'pcm_sha256')}
    ids = set()
    for row in rows:
        if row['id'] in ids:
            raise ValueError('Duplicate recording ID')
        ids.add(row['id'])
        for key, seen in groups.items():
            value = row.get(key)
            if not value:
                raise ValueError('Missing ' + key)
            if value in seen and seen[value] != row['split']:
                raise ValueError('Cross-split ' + key + ' overlap: ' + value)
            seen[value] = row['split']
    return {'cross_split_speaker_overlap': 0, 'cross_split_pair_overlap': 0, 'cross_split_pcm_overlap': 0,
            'rows': len(rows), 'splits': dict(Counter(r['split'] for r in rows))}


def legacy_command(c, text):
    """Deliberately omit legacy operations not present in PCV-B1."""
    if not c:
        return None
    intent = c['intent']
    direct = {'TIME_QUERY': 'TIME', 'WEATHER_QUERY': 'WEATHER', 'MEDIA_PAUSE': 'PAUSE',
              'MEDIA_STOP': 'STOP', 'MEDIA_NEXT': 'NEXT'}
    if intent in direct:
        return command(direct[intent])
    if intent in ('POWER_ON', 'POWER_OFF') and c.get('entity_id', '').startswith('light.'):
        return command('LIGHT_ON' if intent == 'POWER_ON' else 'LIGHT_OFF', light_id=c['entity_id'])
    if intent == 'BRIGHTNESS':
        return command('BRIGHTNESS', light_id=c.get('entity_id', 'light.all'), percent=c['value'])
    if intent == 'COLOR_SET':
        color = c['color'].replace('_', ' ')
        if color not in {'red','blue','green','yellow','white','orange','purple','pink','cyan','magenta','warm white','cool white'}:
            return None
        return command('COLOR', light_id=c['entity_id'], color=color)
    if intent == 'TIMER_SET':
        return command('TIMER', duration_seconds=c['duration_seconds'])
    if intent == 'ALARM_SET':
        return command('ALARM', time=c['time_local'], timezone='Asia/Manila')
    if intent == 'TEMPERATURE_SET':
        return command('TEMPERATURE', degrees=c['celsius'], unit='C')
    if intent == 'MEDIA_MUTE':
        return command('VOLUME', operation='mute')
    if intent == 'VOLUME_ABSOLUTE':
        return command('VOLUME', operation='set', percent=c['value'])
    if intent == 'VOLUME' and c.get('direction') in ('up', 'down'):
        return command('VOLUME', operation=c['direction'])
    if intent == 'MEDIA_PLAY':
        track = c.get('track_id')
        if track:
            song_id = {'track.bts_dynamite': 'song.dynamite', 'track.bts_butter': 'song.butter'}.get(track, track.replace('track.', 'song.'))
            slots = {'song_id': song_id}
            # Keep title+artist recognition bound to the complete legacy hypothesis.
            if ' by ' in text.lower():
                artist = normalize_text(text.lower().split(' by ', 1)[1])
                artist = re.sub(r' (please|for me)$', '', artist)
                slots['artist_id'] = 'artist.' + artist.replace(' ', '_')
            return command('PLAY_MUSIC', **slots)
        return command('PLAY_MUSIC')
    if intent == 'CALL_START':
        return command('CALL', contact_id=c['contact_id'])
    return None


def make_definitions():
    STUDY.mkdir(parents=True, exist_ok=True)
    rows = baseline_cases() + extended_cases()
    save(STUDY / 'baseline.json', baseline_cases())
    save(STUDY / 'custom-slots.json', DEFAULT_CUSTOM)
    save(STUDY / 'synthesis-prompts.json', rows)
    return rows


def _words():
    ints = dictionary(ROOT / 'data/raw/cmudict/cmudict.dict')
    words = {key: [PHONES[i-1] for i in value] for key, value in ints.items()}
    words.update(json.loads((ROOT / 'experiments/pcv-synthetic-v2/lexicon.json').read_text())['words'])
    return words


def inventory_and_sources():
    initial = make_definitions()
    lookup = {normalize_text(r.get('spoken_text', r['text'])): r['command'] for r in initial}
    catalog_rows = list(initial)
    sources, exclusions = [], []
    # The existing FSC/Speech Commands manifests identify immutable original audio.
    for split in ('train', 'dev', 'calibration', 'test'):
        for row in read_rows(ROOT / '.cache/semifinal-v1/mixed-v2' / f'{split}.jsonl'):
            text = normalize_text(row['text'])
            expected = lookup.get(text, legacy_command(row.get('expected'), text))
            if row['source'].startswith('Speech Commands') and text == 'stop':
                expected = command('STOP')
            if expected is not None:
                catalog_rows.append({'id': 'FSC-T-' + hashlib.sha256(text.encode()).hexdigest()[:14],
                                     'text': text, 'command': expected, 'tier': 'public'})
                lookup[text] = expected
            sources.append({'id': 'public/' + row['id'], 'source': row['source'], 'audio': row['audio'],
                'split': split, 'speaker': row['speaker'], 'pair': row['source'] + '/' + row['id'],
                'text': text, 'expected': expected, 'intent': expected['intent'] if expected else 'OOV',
                'synthetic': False, 'expected_audio_sha256': row['audio_sha256'],
                'annotation': row['annotation'], 'reference_class': 'supported' if expected else 'unsupported_valid_or_word'})
    # Re-map only compatible old hypotheses. User-approved baseline overrides obsolete negatives (e.g. bare Call).
    old_catalog = json.loads((ROOT / '.cache/pcv-synthetic-v2-repair1/prepared/catalog.json').read_text())['entries']
    old_by_id = {r['id']: r for r in old_catalog}
    old_mapping = {}
    for row in old_catalog:
        text = normalize_text(row['text'])
        expected = lookup[text] if text in lookup else legacy_command(row['command'], text)
        old_mapping[text] = expected
        if text not in lookup:
            catalog_rows.append({'id': 'LEGACY-' + row['id'], 'text': text, 'command': expected,
                                 'tier': 'extension' if expected else 'negative', 'legacy_id': row['id']})
    # Compile before source annotation; exact text membership is an annotation aid only.
    words = _words()
    catalog = compile_inventory(catalog_rows, words)
    exact = {r['text']: r['command'] for r in catalog['entries']}
    meta = json.loads((ROOT / 'data/raw/slurp/metadata.json').read_text())
    family = {'weather_query': 'WEATHER', 'play_music': 'PLAY_MUSIC', 'alarm_set': 'ALARM',
        'iot_hue_lighton': 'LIGHT_ON', 'iot_hue_lightoff': 'LIGHT_OFF', 'iot_hue_lightchange': 'COLOR',
        'iot_hue_lightup': 'BRIGHTNESS', 'iot_hue_lightdim': 'BRIGHTNESS',
        'audio_volume_up': 'VOLUME', 'audio_volume_down': 'VOLUME', 'audio_volume_mute': 'VOLUME'}
    original_speakers = {}
    for original in ('train', 'devel', 'test'):
        original_speakers[original] = set()
        for row in read_rows(ROOT / 'data/raw/slurp' / f'{original}.jsonl'):
            info = meta[str(row['slurp_id'])]
            text = normalize_text(info['sentence_normalised'])
            intent = family.get(row['intent'], 'OOV')
            if row['intent'] == 'datetime_query' and 'time' in text.split() and not {'date', 'day', 'tomorrow'} & set(text.split()):
                intent = 'TIME'
            if row['intent'] == 'calendar_set' and 'remind' in text.split():
                intent = 'CREATE_REMINDER'
            for recording in row['recordings']:
                details = info['recordings'].get(recording['file'], {})
                speaker = details.get('usrid')
                original_speakers[original].add(speaker)
                audio = ROOT / 'data/audio/slurp/slurp_real' / recording['file']
                if not audio.is_file():
                    exclusions.append({'id': 'slurp/' + recording['file'], 'reason': 'audio_not_locally_available'})
                    continue
                if not speaker or not str(details.get('status', '')).startswith('correct'):
                    exclusions.append({'id': 'slurp/' + recording['file'], 'reason': 'missing_speaker_or_noncorrect_source_status'})
                    continue
                expected = exact.get(text)
                sources.append({'id': 'slurp/' + recording['file'], 'source': 'SLURP-real', 'audio': str(audio.resolve()),
                    'split': speaker_split('slurp/' + speaker), 'original_split': original, 'speaker': 'slurp/' + speaker,
                    'pair': 'slurp/' + str(details['recid']), 'text': text, 'expected': expected,
                    'intent': expected['intent'] if expected else intent, 'synthetic': False,
                    'annotation': 'published normalized text; exact finite membership for commands; conservative source family for auxiliary intent; not listened',
                    'source_intent': row['intent'], 'reference_class': 'supported' if expected else 'outside_finite_catalog'})
    # Parent Qwen voices retain their original split. No Parler recordings enter this study.
    for row in read_rows(ROOT / '.cache/pcv-synthetic-v2-repair1/qwen-manifest.jsonl'):
        text = normalize_text(old_by_id[row['verbatim_id']]['text'])
        expected = exact.get(text)
        speaker = row.get('speaker_id', row.get('voice_group'))
        if speaker is None:
            speaker = row['recording_id'].split('/')[1]
        sources.append({'id': 'qwen/' + row['recording_id'], 'source': 'Qwen', 'audio': row['audio'],
            'split': row['split'], 'speaker': 'qwen/' + speaker, 'pair': 'qwen/' + row['recording_id'],
            'text': text, 'expected': expected, 'intent': expected['intent'] if expected else 'OOV',
            'synthetic': True, 'expected_audio_sha256': row['audio_sha256'],
            'annotation': 'generated prompt; spoken text not listener verified',
            'reference_class': 'supported' if expected else 'outside_finite_catalog'})
    for row in read_rows(CACHE / 'sapi/manifest.jsonl'):
        text = normalize_text(row.get('spoken_text', row['text']))
        sources.append({**row, 'text': text, 'expected': row['command'], 'intent': row['command']['intent'] if row['command'] else 'OOV',
                       'synthetic': True, 'annotation': 'local SAPI prompt; not independently listened',
                       'reference_class': 'supported' if row['command'] else 'negative'})
    overlap = {f'{a}_{b}': len(original_speakers[a] & original_speakers[b]) for a,b in [('train','devel'),('train','test'),('devel','test')]}
    return sources, catalog, words, exclusions, overlap


def _prepare_one(pair):
    row, words = pair
    try:
        labels = compile_inventory([{'id': row['id'], 'text': row['text'], 'command': None}], words)['entries'][0]['phones']
        audio = Path(row['audio'])
        audio_hash = sha(audio)
        if row.get('expected_audio_sha256') and audio_hash != row['expected_audio_sha256']:
            raise RuntimeError('Source hash mismatch: ' + str(audio))
        pcm, rate = sf.read(audio, dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('invalid_rate_channels_or_nonfinite')
        if not 0.15 <= len(pcm)/rate <= 12:
            raise ValueError('outside_endpoint_duration')
        f = features(pcm)
        if f.shape[1] < len(labels) + sum(a == b for a,b in zip(labels, labels[1:])):
            raise ValueError('ctc_infeasible')
        pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
        path = CACHE / 'prepared/features' / (audio_hash + '.npy')
        if not path.exists():
            with path.open('xb') as stream:
                np.save(stream, f)
        stored = np.load(path, allow_pickle=False)
        if not np.array_equal(stored, f):
            raise RuntimeError('Existing feature cache differs')
        target = -100
        expected = row['expected']
        if expected and expected['intent'] in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'}:
            target = int(expected['slots']['light_id'] != 'light.all')
        return {**row, 'audio_sha256': audio_hash, 'pcm_sha256': pcm_hash, 'features': str(path.resolve()),
                'features_sha256': sha(path), 'frontend_spec_sha256': hashlib.sha256(canonical(SPEC).encode()).hexdigest(),
                'phones': labels, 'frames': f.shape[1], 'duration_seconds': len(pcm)/rate,
                'intent_index': INTENTS.index(row['intent'])+1 if row['intent'] != 'OOV' else 0,
                'target_index': target}, None
    except (ValueError, FileNotFoundError) as exc:
        return None, {'id': row['id'], 'reason': str(exc)}


def prepare():
    destination = CACHE / 'prepared'
    if (destination / 'freeze.json').exists():
        raise FileExistsError('Preparation already frozen; use a new study version')
    (destination / 'features').mkdir(parents=True, exist_ok=True)
    sources, catalog, words, exclusions, original_overlap = inventory_and_sources()
    save(STUDY / 'catalog.json', catalog)
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, (row, error) in enumerate(pool.map(_prepare_one, ((r, words) for r in sources)), 1):
            if error:
                exclusions.append(error)
            else:
                rows.append(row)
            if index % 2000 == 0 or index == len(sources):
                print(json.dumps({'stage': 'prepare', 'processed': index, 'total': len(sources), 'kept': len(rows)}), flush=True)
    # Quarantine every cross-split duplicate; never retain one side based on test quality.
    pcm_splits = defaultdict(set)
    for row in rows:
        pcm_splits[row['pcm_sha256']].add(row['split'])
    bad_hashes = {h for h, splits in pcm_splits.items() if len(splits) > 1}
    for row in rows:
        if row['pcm_sha256'] in bad_hashes:
            exclusions.append({'id': row['id'], 'reason': 'cross_split_pcm_quarantine'})
    rows = [r for r in rows if r['pcm_sha256'] not in bad_hashes]
    # Same-split bit-identical duplicates do not inflate training or test denominators.
    unique = {}
    for row in sorted(rows, key=lambda r: r['id']):
        key = row['pcm_sha256']
        if key in unique:
            if canonical(unique[key]['expected']) != canonical(row['expected']):
                raise ValueError('Conflicting duplicate annotations')
            exclusions.append({'id': row['id'], 'reason': 'same_split_pcm_duplicate'})
        else:
            unique[key] = row
    rows = list(unique.values())
    audit = audit_split_rows(rows)
    stats = Counter((r['source'], r['split'], r['intent']) for r in rows)
    totals = np.zeros(40, dtype='float64')
    squares = np.zeros(40, dtype='float64')
    nframes = 0
    for row in rows:
        if row['split'] == 'train' and not row['synthetic']:
            f = np.load(row['features'], allow_pickle=False).astype('float64')
            totals += reduce(f, 'c t -> c', 'sum')
            squares += reduce(f*f, 'c t -> c', 'sum')
            nframes += row['frames']
    mean = totals/nframes
    std = np.maximum(np.sqrt(np.maximum(0, squares/nframes-mean**2)), 1e-4)
    save(destination / 'normalization.json', {'mean': mean.tolist(), 'std': std.tolist(), 'spec': SPEC, 'fit': 'human_train_only', 'frames': nframes})
    manifests = {}
    for split in sorted({r['split'] for r in rows}):
        path = destination / f'{split}.jsonl'
        with path.open('w', encoding='utf8') as stream:
            for row in rows:
                if row['split'] == split:
                    stream.write(canonical(row) + '\n')
        manifests[split] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    coverage = []
    for baseline in baseline_cases():
        text = normalize_text(baseline.get('spoken_text', baseline['text']))
        coverage.append({'id': baseline['id'], 'text': baseline['text'], 'counts': dict(Counter(
            r['source'] + '/' + r['split'] for r in rows if r['text'] == text and canonical(r['expected']) == canonical(baseline['command'])))})
    save(destination / 'exclusions.json', exclusions)
    audit.update(source_split_intent=[{'source': s, 'split': p, 'intent': i, 'n': n} for (s,p,i), n in sorted(stats.items())],
                 original_slurp_speaker_overlap=original_overlap, exclusion_reasons=dict(Counter(r['reason'] for r in exclusions)),
                 baseline_coverage=coverage, catalog_entries=len(catalog['entries']))
    save(STUDY / 'data-audit.json', audit)
    freeze = {'experiment': 'pcv-human-slots-v1', 'manifests': manifests, 'catalog_sha256': sha(STUDY / 'catalog.json'),
              'normalization_sha256': sha(destination / 'normalization.json'), 'audit_sha256': sha(STUDY / 'data-audit.json'),
              'test_inference_accessed': False, 'sources': {str(p.relative_to(ROOT)): sha(p) for p in
              [ROOT/'data/sources/fluent.source.json', ROOT/'data/sources/slurp.source.json', ROOT/'data/sources/speech_commands.source.json',
               ROOT/'.cache/pcv-synthetic-v2-repair1/qwen-manifest.jsonl', CACHE/'sapi/manifest.jsonl']}}
    save(destination / 'freeze.json', freeze)
    save(STUDY / 'data-freeze.json', freeze)
    print(json.dumps({'stage': 'prepared', 'audit': {k:v for k,v in audit.items() if k not in {'source_split_intent','baseline_coverage','exclusion_reasons'}}}), flush=True)
