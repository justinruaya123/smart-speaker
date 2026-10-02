"""E18 metadata overlay: immutable source recordings, bounded reads, no transcripts at inference."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import re
import time

from .complete_data import digest, save, read_rows, write_rows
from .family_contract import annotate

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/vcm-prototype-v1'
DATA = ROOT / 'data/vcm-prototype-v1'
CACHE = ROOT / '.cache/vcm-prototype-v1'
PARENT = ROOT / 'data/vcm-complete-v1/manifest.jsonl'


def revise(row, catalog):
    """Frozen, model-independent semantic rules; labels remain programmatic silver."""
    result = copy.deepcopy(row)
    result['annotation'] = copy.deepcopy(row['supervision'] if row['split'] == 'train' else row['annotation'])
    text = re.sub(r'\s+', ' ', row.get('transcript', '').lower()).strip(' .?!')
    reasons = []
    if row['command'] in {'POWER_ON', 'POWER_OFF', 'BRIGHTNESS', 'COLOR_SET'} and 'bed room' in text:
        labeled = {**row, 'transcript': re.sub(r'\bbed room\b', 'bedroom', text)}
        result['annotation'] = {**row['annotation'], **annotate(labeled, catalog)}
        reasons.append('bed_room_target_alias')
    plain_time = re.fullmatch(r"(?:please )?(?:what(?:'s| is) (?:the )?(?:current )?time|what time is it|(?:tell|give) me (?:the )?(?:current )?time)(?: (?:now|right now))?(?: please)?", text)
    plain_date = re.fullmatch(r"(?:please )?(?:what(?:'s| is) (?:the )?(?:current )?date|what date is it|(?:tell|give) me (?:the )?(?:current )?date|what day is it)(?: today)?(?: please)?", text)
    family = 'TIME_QUERY' if plain_time else 'DATE_QUERY' if plain_date else None
    if family and row['command'] in {'UNKNOWN', 'TIME_QUERY', 'DATE_QUERY'} and family != row['command']:
        result['command'] = family
        result['annotation'] = {**row['annotation'], 'family': family, **annotate({**row, 'command': family}, catalog)}
        reasons.append('plain_local_query_semantics')
    qualified = r'\b(tomorrow|yesterday|next|last|following|birthday|christmas|easter|new year|january|february|march|april|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|central|eastern|pacific|gmt|timezone|time zone)\b|\bmay (?:\d|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|twentieth)\b'
    if result['command'] in {'TIME_QUERY', 'DATE_QUERY'} and (re.search(qualified, text) or re.search(r'\b(?:time|date) (?:in|at|for)\b', text)):
        result['command'] = 'UNKNOWN'
        result['annotation'] = {**row['annotation'], 'family': 'UNKNOWN', **annotate({**row, 'command': 'UNKNOWN'}, catalog)}
        reasons.append('qualified_query_has_no_supported_arguments')
    if result['command'] in {'POWER_ON', 'POWER_OFF', 'BRIGHTNESS', 'COLOR_SET'} and re.search(r'\b(porch|den|dining room|disco|media room|garden|backyard|balcony|patio|stairs|upstairs|downstairs|basement|attic)\b', text):
        a = result['annotation']
        a['labels']['target'], a['labels']['guard'] = 'OTHER', 'UNRESOLVED'
        a['reference'] = {'status': 'needs_parameters', 'reason': 'unsupported_named_target'}
        reasons.append('unsupported_location_is_not_main_lights')
    result['supervision'] = copy.deepcopy(result['annotation'])
    return result, reasons


def verify():
    lock = json.loads((DATA / 'frozen.json').read_text(encoding='utf-8'))
    for name, expected in lock['files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Frozen E18 input changed: ' + name)
    return lock


def dataset(splits, revised=True):
    """Keep only requested human partitions in memory; never include test in fitting."""
    overlay = {r['id']: r for r in read_rows(DATA / 'overlay.jsonl')} if revised else {}
    reserved = set(json.loads((DATA / 'audit.json').read_text(encoding='utf-8'))['custom_enrollment_pairs'])
    output = []
    with PARENT.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] != 'human' or row['split'] not in splits:
                continue
            if row['id'] in overlay:
                item = overlay[row['id']]
                row.update(command=item['command'], annotation=item['annotation'], supervision=item['annotation'])
            elif revised:
                row['annotation'] = copy.deepcopy(row['supervision'] if row['split'] == 'train' else row['annotation'])
                row['supervision'] = row['annotation']
            if row['split'] == 'train' and (row['unusable_reasons'] or row['filtered_reasons']):
                continue
            if row['split'] in {'dev', 'calibration'} and row['paired_group'] in reserved:
                continue
            if row['split'] == 'calibration':
                row['fold'] = 'policy_fit'
            output.append(row)
    return output


def prepare():
    if (DATA / 'frozen.json').exists():
        raise ValueError('Data already frozen; declare a new version for changes')
    protocol = json.loads((EXP / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['inputs'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Protocol input changed: ' + name)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_rows(PARENT) if r['kind'] == 'human']
    catalog = json.loads((ROOT / protocol['initial_bundle']).read_text(encoding='utf-8'))['catalog']
    entries = json.loads((ROOT / 'experiments/vcm-complete-v1/custom-catalog-v3.json').read_text(encoding='utf-8'))['catalog']
    enrollment = {i for e in entries for i in e['enrollment_ids']}
    pairs = {r['paired_group'] for r in rows if r['id'] in enrollment}
    # The inherited full audio audit proves readability for the same content hashes.
    # Rehash named payloads one file at a time, without loading a waveform pool.
    def audit(row):
        path = (ROOT / row['audio']).resolve()
        if not path.is_relative_to((ROOT / 'data/audio').resolve()):
            raise ValueError('Unexpected audio path')
        if digest(path) != row['audio_sha256']:
            raise ValueError('Audio changed: ' + row['id'])
        return path.stat().st_size
    last, total_bytes = time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, size in enumerate(pool.map(audit, rows), 1):
            total_bytes += size
            if time.monotonic() - last > 20:
                print(f'PAYLOAD_HASH_AUDIT {count}/{len(rows)}', flush=True)
                last = time.monotonic()
    splits = {s: [r for r in rows if r['split'] == s] for s in ['train', 'dev', 'calibration', 'test']}
    for field in ['speaker_id', 'audio_sha256']:
        train_values = {r[field] for r in splits['train'] if not r['unusable_reasons']}
        other_values = {r[field] for r in rows if r['split'] != 'train'}
        if train_values & other_values:
            raise ValueError('Train/evaluation overlap: ' + field)
    overlay, reasons = [], Counter()
    for row in rows:
        revised, changes = revise(row, catalog)
        if changes:
            overlay.append({'id': row['id'], 'split': row['split'], 'command': revised['command'],
                'annotation': revised['annotation'], 'reasons': changes,
                'previous_command': row['command'], 'previous_annotation': row['annotation']})
            reasons.update(row['split'] + ':' + c for c in changes)
    write_rows(DATA / 'overlay.jsonl', overlay)
    save(DATA / 'audit.json', {'original_human_rows': len(rows), 'payload_bytes_hashed': total_bytes,
        'source_splits': {k: len(v) for k,v in splits.items()}, 'changed_rows': len(overlay),
        'changes_by_split_and_reason': reasons, 'custom_enrollment_pairs': sorted(pairs),
        'speaker_and_hash_train_eval_overlap': 0, 'original_payloads_or_labels_modified': False,
        'annotation_quality': 'programmatic semantic overlay; NOT human audio adjudication',
        'historical_test_previously_examined': True, 'readability_evidence': 'E13 unchanged-content audio audit'})
    locked = [PARENT, DATA / 'overlay.jsonl', DATA / 'audit.json', EXP / 'protocol.json', Path(__file__)]
    save(DATA / 'frozen.json', {'files': {p.relative_to(ROOT).as_posix(): digest(p) for p in locked},
        'test_selection_permitted': False})
    print(json.dumps({'rows': len(rows), 'changed_rows': len(overlay), 'reasons': reasons}), flush=True)


def sample_epoch(rows, cfg, rng, source_balanced):
    if any(r['split'] != 'train' or r['kind'] != 'human' or r['unusable_reasons'] or r['filtered_reasons'] for r in rows):
        raise ValueError('Sampler accepts only eligible human training rows')
    groups, slots = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        groups[row['command']][row['source']].append(i)
        for name in ['operation', 'amount', 'target', 'color']:
            value = row['supervision']['labels'].get(name)
            if value not in {None, 'NONE', 'OTHER'}:
                slots[(name, value)][row['source']].append(i)
    def pick(group):
        values = list(group.values())
        return rng.choice(rng.choice(values)) if source_balanced else rng.choice([i for g in values for i in g])
    chosen = [pick(group) for group in groups.values() for _ in range(cfg['per_family_draws'])]
    keys = sorted(slots)
    chosen.extend(pick(slots[rng.choice(keys)]) for _ in range(cfg['slot_draws']))
    rng.shuffle(chosen)
    return chosen
