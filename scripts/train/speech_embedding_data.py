"""Freeze a new protocol over the exact audited parameter-study human corpus."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from .parameter_study_data import ROOT, read_rows, prepare as original_prepare, dump, digest, lexicon, phones

DATA = ROOT / 'data/speechembedding-v1'
OUT = ROOT / '.cache/speechembedding-v1'
REPORT = ROOT / 'STT/speechembedding'
SPEC = ROOT / 'experiments/speechembedding'
ROLES = ('artist', 'song', 'device', 'number', 'word')


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if (DATA / 'protocol.json').exists():
        protocol = json.loads((DATA / 'protocol.json').read_text())
        for name, expected in protocol['frozen_files'].items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen input changed; create a new study version: ' + name)
        return read_rows(DATA / 'manifest.jsonl'), protocol
    rows = original_prepare()
    (DATA / 'manifest.jsonl').write_bytes((ROOT / 'data/parameter-study/manifest.jsonl').read_bytes())
    catalogs = json.loads((ROOT / 'data/parameter-study/catalogs.json').read_text())
    # Small playlist fixtures use only training frequency; never query-specific lists.
    counts = Counter((t['role'], t['value']) for r in rows if r['split'] == 'train' for t in r['targets'])
    for size in (10, 25):
        catalogs[f'playlist{size}'] = {
            role: sorted(catalogs['fixed'][role], key=lambda v: (-counts[role, v], v))[:size]
            for role in ('artist', 'song')}
    dump(DATA / 'catalogs.json', catalogs)
    speaker_splits = defaultdict(set)
    hashes = defaultdict(set)
    for r in rows:
        split = 'train' if r['split'] == 'enroll' else r['split']
        speaker_splits[r['speaker']].add(split)
        hashes[r['audio_sha256']].add(split)
    assert all(len(v) == 1 for v in speaker_splits.values())
    assert all(len(v) == 1 for v in hashes.values())
    novel = {t['value'] for t in catalogs['novel']}
    assert not any(' ' + name + ' ' in ' ' + r['text'] + ' '
                   for r in rows if r['split'] == 'train' for name in novel)
    preserved_paths = [ROOT / 'pipeline.toml', *sorted((ROOT / 'artifacts').glob('*')),
                       ROOT / '.cache/parameter-study/results.json',
                       ROOT / '.cache/phonetic-v2/results.json']
    dictionary = lexicon()
    protocol = dict(
        version='speechembedding-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        models=['dual', 'dual_kws'], seed=71, epochs=28, patience=9,
        batch_size=64, entity_examples_per_batch=16, learning_rate=.0015,
        weight_decay=.001, temperature=.10, entity_loss_weight=.75, kws_loss_weight=.30,
        initialization='random; no pretrained acoustic or text weights',
        audio='original complete human utterances only; no TTS, splicing, or audio augmentation',
        frontend='shared mono 16 kHz, 80-bin per-utterance CMVN',
        local_windows='ten fixed fractional pools of intact utterance features; timing unverified',
        text='CMUdict pronunciations only; missing entries abstain; no automatic G2P',
        checkpoint_selection='dev: 0.4 artist top1 + 0.4 song top1 + 0.2 isolated-word top1, expanded catalog, local windows',
        confidence_selection='per model/catalog/role: >=95% dev precision, >=10 emissions, <=1% unknown false accept; otherwise disabled',
        test_used_for_selection=False, role_supplied_from_annotations=True,
        full_pipeline_accuracy=False, catalog_integration='offline fixtures; no Spotify service calls',
        historical_test='same already-reported recordings; fresh scoring, not a newly blind test',
        counts=dict(Counter(r['split'] for r in rows)),
        role_counts={s:dict(Counter(t['role'] for r in rows if r['split']==s for t in r['targets']))
                     for s in ('train', 'enroll', 'dev', 'test')},
        sources={s:dict(Counter(r['split'] for r in rows if r['source']==s)) for s in sorted({r['source'] for r in rows})},
        catalog_counts={k:{role:len(v) for role,v in catalogs[k].items()} for k in ('fixed','expanded','playlist10','playlist25')},
        missing_pronunciations={role:[v for v in values if phones(v,dictionary) is None]
                                for role,values in catalogs['expanded'].items() if role in ('artist','song','device')},
        cross_split_speakers=0, cross_split_audio_hashes=0, reserved_name_training_overlap=0,
        frozen_files={str(p.relative_to(ROOT)):digest(p) for p in
                      [DATA/'manifest.jsonl',DATA/'catalogs.json',ROOT/'data/parameter-study/manifest.jsonl',
                       ROOT/'data/parameter-study/catalogs.json',ROOT/'STT/alternatives/sources/cmudict.dict',
                       ROOT/'scripts/runtime/stt_runtime.py']},
        preserved={str(p.relative_to(ROOT)):digest(p) for p in preserved_paths if p.is_file()})
    dump(DATA / 'protocol.json', protocol)
    return rows, protocol


def token_data(rows):
    dictionary = lexicon()
    vocabulary = ['<pad>'] + sorted({p for seq in dictionary.values() for p in seq} | {'|'})
    lookup = {p:i for i,p in enumerate(vocabulary)}
    def encode(text):
        sequence = phones(text, dictionary)
        return [lookup[p] for p in sequence] if sequence else None
    full = {i:encode(r['text']) for i,r in enumerate(rows)}
    full = {i:v for i,v in full.items() if v}
    from .levels import spoken_number
    targets = {}
    for i,r in enumerate(rows):
        values = [(t['role'],t['value'],t['surface']) for t in r['targets']]
        targets[i] = [(role,value,encode(surface)) for role,value,surface in values if encode(surface)]
    return vocabulary, full, targets, encode


def preserved_check(protocol):
    mismatches = [name for name,sha in protocol['preserved'].items() if digest(ROOT/name) != sha]
    if mismatches:
        raise ValueError('Protected input changed: ' + ', '.join(mismatches))
    return {'all_preserved':True, 'files':protocol['preserved']}
