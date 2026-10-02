"""Immutable, bounded-memory data preparation for vcm-complete-v1."""
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import re
import time

import numpy as np
from .audio import load_audio, DEFAULT, features
from .family_contract import annotate
from .supervision import repair_reference

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/vcm-complete-v1'
EXP = ROOT / 'experiments/vcm-complete-v1'
CACHE = ROOT / '.cache/vcm-complete-v1'
CUES = ['on', 'off', 'turn', 'switch', 'activate', 'deactivate', 'light', 'lights', 'lamp',
        'brightness', 'brighten', 'dim', 'increase', 'decrease', 'up', 'down', 'louder',
        'quieter', 'volume', 'sound', 'percent', 'play', 'resume', 'continue', 'music',
        'playlist', 'song', 'time', 'clock', 'date', 'today', 'red', 'blue', 'green',
        'white', 'yellow', 'orange', 'pink', 'purple', 'cyan', 'kitchen', 'bedroom',
        'bathroom', 'living', 'office', 'hall', 'garage', 'mute', 'maximum', 'minimum']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True) + '\n')


def finite_reference(row):
    ref = row['annotation'].get('reference')
    if ref and ref.get('status') == 'accepted' and not (set(ref['command']['arguments']) & {'title', 'artist', 'selection_text'}):
        return ref['command']
    return None


def custom_key(row):
    ref = row['annotation'].get('reference')
    if row['command'] != 'MEDIA_PLAY' or not ref or ref.get('status') != 'accepted':
        return None
    args = ref['command']['arguments']
    return json.dumps(args, sort_keys=True) if args else None


def cue_labels(row):
    words = set(re.findall(r"[a-z]+", row.get('transcript', '').lower()))
    return [int(word in words) for word in CUES]


def prepare():
    if (DATA/'frozen.json').exists():
        raise ValueError('This data version is frozen; use a new version to change it')
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((EXP/'protocol.json').read_text())
    for name, expected in protocol['input_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Input changed since protocol: ' + name)
    metadata = json.loads((ROOT/'artifacts/bundle.json').read_text())
    refs = {r['id']:r for r in read_rows(ROOT/'data/references.jsonl')}
    folds = json.loads((ROOT/'data/development-folds.json').read_text())
    human = read_rows(ROOT/'data/vcm.jsonl')
    synthetic = read_rows(ROOT/'data/tts/manifest.jsonl')
    rows, repairs = [], []
    fields = ['id','audio','source','speaker_id','paired_group','transcript','source_intent','command','duration','license']
    for original in human + synthetic:
        row = {k:original[k] for k in fields if k in original}
        row['split'] = original['split']
        row['kind'] = 'synthetic' if original in () or original.get('kind') == 'synthetic' else 'human'
        row['paired_group'] = row.get('paired_group', row['id'])
        if row['kind'] == 'human':
            row['annotation'] = refs[row['id']]
            row['fold'] = folds.get(row['id'])
            training_annotation, repair = repair_reference(row, row['annotation'], metadata['catalog'])
            row['supervision'] = training_annotation if row['split'] == 'train' else row['annotation']
            if repair and row['split'] == 'train': repairs.append(repair)
        else:
            row['annotation'] = annotate(row, metadata['catalog'])
            row['supervision'] = row['annotation']
            row['expected_audio_sha256'] = original['audio_sha256']
            row['voice_id'] = original['voice_id']
            row['label_quality'] = original['label_quality']
        rows.append(row)
    # Read each named recording independently. No archive traversal or waveform pool.
    audio_audit, last = [], time.monotonic()
    for index, row in enumerate(rows):
        path = (ROOT/row['audio']).resolve()
        if not path.is_relative_to((ROOT/'data/audio').resolve()):
            raise ValueError('Unexpected payload path: ' + row['id'])
        sha = digest(path)
        if row.get('expected_audio_sha256',sha) != sha:
            raise ValueError('Existing source audio changed: ' + row['id'])
        pcm, rate = load_audio(path)
        problems = []
        if not np.isfinite(pcm).all(): problems.append('nonfinite_audio')
        if len(pcm) < 400: problems.append('shorter_than_frontend_window')
        if not len(pcm) or float(np.max(np.abs(pcm))) == 0: problems.append('zero_audio')
        row['audio_sha256'] = sha
        row['duration'] = len(pcm)/16000
        row['unusable_reasons'] = problems
        row['filtered_reasons'] = []
        audio_audit.append({'id':row['id'],'sha256':sha,'bytes':path.stat().st_size,
            'samples_16k':len(pcm),'source_rate':rate,'peak':float(np.max(np.abs(pcm))) if len(pcm) else 0,
            'clipped_fraction':float(np.mean(np.abs(pcm) >= .999)) if len(pcm) else 0,'problems':problems})
        if time.monotonic()-last > 20:
            print(f'AUDIT {index+1}/{len(rows)} one recording at a time', flush=True)
            last = time.monotonic()
    human_eval = [r for r in rows if r['kind']=='human' and r['split']!='train']
    reserved_speakers = {r['speaker_id'] for r in human_eval}
    reserved_hashes = {r['audio_sha256'] for r in human_eval}
    groups = defaultdict(list)
    for row in rows:
        if row['split']=='train':groups[row['audio_sha256']].append(row)
    for row in rows:
        if row['split'] != 'train':continue
        if row['speaker_id'] in reserved_speakers:row['unusable_reasons'].append('reserved_human_speaker')
        if row['audio_sha256'] in reserved_hashes:row['unusable_reasons'].append('reserved_human_audio_hash')
        if len({r['command'] for r in groups[row['audio_sha256']]}) > 1:
            row['unusable_reasons'].append('conflicting_duplicate_family')
        if row['duration'] > 8:row['filtered_reasons'].append('exceeds_live_command_limit')
    # Prefer the first sorted recording for the same paired utterance/hash.
    seen_pairs, seen_hashes = set(), set()
    for row in sorted((r for r in rows if r['split']=='train' and not r['unusable_reasons']),key=lambda r:r['id']):
        if row['paired_group'] in seen_pairs:row['filtered_reasons'].append('paired_recording_duplicate')
        if row['audio_sha256'] in seen_hashes:row['filtered_reasons'].append('audio_hash_duplicate')
        seen_pairs.add(row['paired_group']);seen_hashes.add(row['audio_sha256'])
    custom = defaultdict(lambda:defaultdict(list))
    for row in rows:
        key = custom_key(row)
        if row['kind']=='human' and key and not row['unusable_reasons']:
            custom[key][row['split']].append(row)
    catalog, enrollment_ids = [], set()
    for key, splits in sorted(custom.items()):
        available, paired = [], set()
        for row in sorted(splits['train'],key=lambda r:r['id']):
            if row['paired_group'] not in paired:
                available.append(row);paired.add(row['paired_group'])
        if len(available)>=3 and splits['dev'] and splits['test']:
            chosen=available[:3];enrollment_ids.update(r['id'] for r in chosen)
            catalog.append({'id':f'media-{len(catalog):03d}','reference_arguments':json.loads(key),
                'enrollment_ids':[r['id'] for r in chosen],
                'playlist_subset':'playlist' in key.lower(),
                'dev_ids':[r['id'] for r in splits['dev']], 'test_ids':[r['id'] for r in splits['test']]})
        if len(catalog)>=20:break
    for row in rows:
        if row['id'] in enrollment_ids:row['unusable_reasons'].append('reserved_enrollment_recording')
    for split in ['train','dev','test']:
        voices={r['voice_id'] for r in rows if r['kind']=='synthetic' and r['split']==split}
        others={r['voice_id'] for r in rows if r['kind']=='synthetic' and r['split']!=split}
        assert not voices & others
    value_counts = {}
    for kind in ['human','synthetic']:
        for split in ['train','dev','test']:
            selected=[r for r in rows if r['kind']==kind and r['split']==split]
            count=Counter(r['supervision']['labels'].get('amount') for r in selected)
            numeric={k:v for k,v in count.items() if isinstance(k,str) and k.isdigit()}
            value_counts[f'{kind}_{split}']={'counts':numeric,'covered':len(numeric),'missing':sorted(set(map(str,range(101)))-set(numeric),key=int)}
    test=[r for r in rows if r['kind']=='human' and r['split']=='test']
    assert len(test)==4314 and sum(finite_reference(r) is not None for r in test)==1783
    order=list(range(101));random.Random(1701).shuffle(order)
    write_rows(DATA/'manifest.jsonl',rows)
    write_rows(DATA/'audio-audit.jsonl',audio_audit)
    save(DATA/'custom-catalog.json',{'schema_version':1,'catalog':catalog,'scope':'Original human custom-media proxies; playlist subset marked. Three full-utterance enrollments per ID; no actual music binding.'})
    save(DATA/'number-test-plan.json',{'seed':1701,'randomized_values':order,'primary_number_metric':'Exact raw amount head value; report intent/command gates separately.','human_separate':True,'synthetic_voices':'Existing test split; no test voice in fitting.'})
    write_rows(DATA/'label-repairs.jsonl',repairs)
    summary={'rows':len(rows),'splits':dict(Counter(f"{r['kind']}_{r['split']}" for r in rows)),
        'sources':dict(Counter(r['source'] for r in rows)),
        'common_training_exclusions':dict(Counter(x for r in rows if r['split']=='train' for x in r['unusable_reasons'])),
        'filtered_training_exclusions':dict(Counter(x for r in rows if r['split']=='train' for x in r['filtered_reasons'])),
        'numeric_coverage':value_counts,'custom_catalog_size':len(catalog),'playlist_catalog_entries':sum(c['playlist_subset'] for c in catalog),
        'enrollment_recordings':len(enrollment_ids),'test_recordings':len(test),'finite_test_references':1783,
        'test_unreadable_recordings':sum(bool(r['unusable_reasons']) for r in test),'training_label_repairs':len(repairs),
        'payloads_moved_or_deleted':False,'waveform_pool_loaded':False,
        'derived_frontend':asdict(DEFAULT),'source_hashes':protocol['input_hashes']}
    save(DATA/'audit.json',summary)
    files={p.name:digest(p) for p in DATA.iterdir() if p.is_file()}
    save(DATA/'frozen.json',{'files':files,'original_test_unchanged':True,'candidate_fitting_started':False})
    print(json.dumps({k:summary[k] for k in ['splits','common_training_exclusions','filtered_training_exclusions','custom_catalog_size','playlist_catalog_entries','training_label_repairs']},indent=2),flush=True)


def verify_frozen():
    frozen=json.loads((DATA/'frozen.json').read_text())
    for name,sha in frozen['files'].items():
        if digest(DATA/name)!=sha:raise ValueError('Frozen data changed: '+name)
    return frozen


def feature(row):
    """Reuse one content/recipe-keyed frontend cache across the four arms."""
    key=hashlib.sha256((row['audio_sha256']+json.dumps(asdict(DEFAULT),sort_keys=True)).encode()).hexdigest()
    path=CACHE/'features'/key[:2]/(key+'.npy')
    if path.exists():return np.load(path,allow_pickle=False)
    pcm,_=load_audio(ROOT/row['audio'])
    result=features(pcm)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,result,allow_pickle=False)
    return result
