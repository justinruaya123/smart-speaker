"""Build speaker-disjoint human-speech manifests and independent feature caches."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from einops import rearrange, reduce

from common import ROOT, normalize, split_speaker, digest, read_jsonl, write_json, write_jsonl, ONES, extract_number

def slurp():
    base = ROOT / 'data/raw/slurp'
    metadata = json.loads((base / 'metadata.json').read_text())
    records = {}
    skipped = Counter()
    for source_split in ['train', 'devel', 'test']:
        for r in read_jsonl(base / (source_split + '.jsonl')):
            tokens = {t['id']: t['surface'] for t in r['tokens']}
            entities = [{'type': e['type'], 'text': normalize(' '.join(tokens[i] for i in e['span']))} for e in r['entities']]
            filenames = {a['file'] for a in r['recordings']}
            for a in r['recordings']:
                name = a['file']
                # Keep one microphone per original take.
                if not name.endswith('-headset.flac') and name.replace('.flac', '-headset.flac') in filenames:
                    skipped['paired_far_microphone'] += 1
                    continue
                meta = metadata[str(r['slurp_id'])]['recordings'].get(name, {})
                speaker = meta.get('usrid')
                if not speaker:
                    skipped['missing_speaker'] += 1
                    continue
                if a.get('status') != 'correct':
                    skipped['source_status_not_correct'] += 1
                    continue
                text = normalize(r['sentence'])
                if not text:
                    continue
                records[name] = {'id': 'slurp/' + name, 'source': 'SLURP-real',
                    'audio': 'data/audio/slurp/' + name, 'text': text, 'original_text': r['sentence'],
                    'speaker': 'slurp/' + speaker, 'split': split_speaker('slurp/' + speaker),
                    'source_split': source_split, 'intent': r['intent'], 'entities': entities,
                    'group': 'slurp/' + name.replace('-headset', ''), 'kind': 'human_command',
                    'license': 'CC-BY-NC-4.0', 'transcript_source': 'publisher_expected_transcript',
                    'publisher_asr_wer': a.get('wer')}
    out = ROOT / 'data/audio/slurp'
    out.mkdir(parents=True, exist_ok=True)
    shards = list((base / 'mirror').glob('*.parquet'))
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                audio = row.get('audio', {})
                name = Path(audio.get('path', '')).name
                if name in records:
                    dest = out / name
                    if not dest.exists():
                        dest.write_bytes(audio['bytes'])
        print('Extracted SLURP', shard.name, flush=True)
    # Data copied from the existing corpus is read-only at its origin. Fresh
    # downloaded shards take precedence and the receipt records the distinction.
    reused = 0
    rows = []
    for name, row in records.items():
        dest = ROOT / row['audio']
        original = ROOT.parent / 'data/audio/slurp/slurp_real' / name
        if not dest.exists() and original.exists():
            shutil.copyfile(original, dest)
            row['acquisition'] = 'isolated_copy_of_existing_public_audio'
            reused += 1
        elif dest.exists():
            row['acquisition'] = 'fresh_download'
        if dest.exists():
            rows.append(row)
    write_json(ROOT / 'evidence/slurp_preparation.json', {'shards': [p.name for p in shards], 'reused_audio': reused, 'selected': len(rows), 'excluded': dict(skipped)})
    return rows

def supplemental_existing():
    rows = []
    seen_audio = set()
    # Supplementary human speech is copied, never imported as model features or weights.
    for r in read_jsonl(ROOT.parent / 'data/asr.jsonl'):
        if r['source'] not in {'FluentSpeechCommands', 'SnipsSmartLights', 'LibriSpeech'}:
            continue
        if r['source'] == 'SnipsSmartLights' and r.get('distance') != 'close':
            continue
        speaker = r.get('speaker_id')
        if not speaker:
            continue
        original = ROOT.parent / r['audio']
        if not original.exists():
            continue
        if r['audio'] in seen_audio:
            continue
        seen_audio.add(r['audio'])
        dest = ROOT / r['audio']
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(original, dest)
        rows.append({'id': r['id'], 'source': r['source'], 'audio': r['audio'],
                     'text': normalize(r['transcript']), 'original_text': r['transcript'],
                     'speaker': speaker, 'split': split_speaker(speaker), 'intent': r.get('source_intent'),
                     'entities': [], 'group': r.get('paired_group', r['id']),
                     'kind': 'human_command' if r['source'] != 'LibriSpeech' else 'human_read_speech',
                     'license': r.get('license'), 'acquisition': 'isolated_copy_of_existing_public_audio'})
    return rows

def speech_commands():
    archive = ROOT / 'data/raw/speech_commands_v0.02.tar.gz'
    if not archive.exists():
        return []
    out = ROOT / 'data/audio/speech_commands'
    rows = []
    with tarfile.open(archive) as tar:
        for member in tar:
            path = Path(member.name)
            if not member.isfile() or path.suffix != '.wav' or path.parent.name == '_background_noise_':
                continue
            word = path.parent.name
            speaker = 'gsc/' + path.name.split('_nohash_')[0]
            dest = out / word / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(tar.extractfile(member).read())
            rows.append({'id': 'gsc/' + word + '/' + path.name, 'source': 'GoogleSpeechCommandsV2',
                'audio': dest.relative_to(ROOT).as_posix(), 'text': word,
                'speaker': speaker, 'split': split_speaker(speaker), 'group': speaker + '/' + path.name,
                'kind': 'human_word', 'entities': [], 'intent': None, 'license': 'CC-BY-4.0', 'acquisition': 'fresh_download'})
    return rows

def fsdd():
    archive = ROOT / 'data/raw/fsdd/fsdd.zip'
    if not archive.exists():
        return []
    out = ROOT / 'data/audio/fsdd'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if '/recordings/' not in name or not name.endswith('.wav'):
                continue
            filename = Path(name).name
            digit, speaker, index = filename[:-4].split('_')
            dest = out / filename
            if not dest.exists():
                dest.write_bytes(z.read(name))
            # Explicit held-out speakers; avoids FSDD's speaker-overlapping default split.
            split = 'test' if speaker == 'yweweler' else ('dev' if speaker == 'theo' else 'train')
            rows.append({'id': 'fsdd/' + filename, 'source': 'FSDD', 'audio': dest.relative_to(ROOT).as_posix(),
                'text': ONES[int(digit)], 'speaker': 'fsdd/' + speaker, 'split': split,
                'group': 'fsdd/' + filename, 'kind': 'human_word', 'entities': [], 'intent': None,
                'license': 'CC-BY-SA-4.0', 'acquisition': 'fresh_download'})
    return rows

def melbank():
    low, high = 2595 * np.log10(1 + np.array([20, 7600]) / 700)
    edges = 700 * (10 ** (np.linspace(low, high, 82) / 2595) - 1)
    freqs = np.fft.rfftfreq(400, 1 / 16000)
    bank = np.zeros((201, 80), dtype=np.float32)
    for i in range(80):
        bank[:, i] = np.maximum(0, np.minimum((freqs - edges[i]) / (edges[i+1] - edges[i]), (edges[i+2] - freqs) / (edges[i+2] - edges[i+1])))
    return bank

BANK = melbank()
WINDOW = np.hanning(400).astype(np.float32)

def features_from_wave(audio, rate):
    if audio.ndim == 2:
        audio = reduce(audio, 'samples channels -> samples', 'mean')
    if rate != 16000:
        from math import gcd
        d = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // d, rate // d)
    audio = audio.astype(np.float32)
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(audio, (200, 200)), 400)[::160]
    power = np.abs(np.fft.rfft(frames * WINDOW, axis=-1)) ** 2
    mel = np.log(np.maximum(power @ BANK, 1e-7)).astype(np.float32)
    mean = reduce(mel, 'time mel -> 1 mel', 'mean')
    variance = reduce((mel - mean) ** 2, 'time mel -> 1 mel', 'mean')
    return (mel - mean) / np.sqrt(variance + 1e-5)

def cache_one(row):
    dest = ROOT / 'data/features' / (digest(row['id'])[:24] + '.npy')
    if dest.exists():
        shape = np.load(dest, mmap_mode='r').shape
        row['frames'] = shape[0]
    else:
        audio, rate = sf.read(ROOT / row['audio'], dtype='float32')
        duration = len(audio) / rate
        if duration < .12 or duration > 14:
            return None
        features = features_from_wave(audio, rate)
        np.save(dest, features.astype(np.float16))
        row['frames'] = len(features)
    row['duration'] = (row['frames'] - 1) / 100
    row['feature'] = dest.relative_to(ROOT).as_posix()
    if (row['frames'] + 1) // 2 < len(row['text']) + sum(a == b for a,b in zip(row['text'], row['text'][1:])):
        return None
    return row

def main():
    rows = slurp() + supplemental_existing() + speech_commands() + fsdd()
    extra = ROOT / 'data/mswc.jsonl'
    if extra.exists():
        rows += read_jsonl(extra)
    rows = list({r['id']: r for r in rows if r['text']}.values())
    (ROOT / 'data/features').mkdir(parents=True, exist_ok=True)
    good = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(cache_one, rows)):
            if row is not None:
                good.append(row)
            if i % 5000 == 0:
                print(f'Features {i}/{len(rows)}', flush=True)
    speakers = {s: {r['speaker'] for r in good if r['split'] == s} for s in ['train','dev','test']}
    groups = {s: {r['group'] for r in good if r['split'] == s} for s in speakers}
    overlaps = {a+'_'+b: {'speakers': len(speakers[a]&speakers[b]), 'paired_groups': len(groups[a]&groups[b])} for a,b in [('train','dev'),('train','test'),('dev','test')]}
    assert all(v['speakers'] == v['paired_groups'] == 0 for v in overlaps.values()), overlaps
    write_jsonl(ROOT / 'data/manifest.jsonl', good)
    audit = {'rows': len(good), 'excluded_duration_or_ctc_length': len(rows) - len(good),
             'source_split': dict(Counter(r['source']+'/'+r['split'] for r in good)),
             'hours': sum(r['duration'] for r in good)/3600, 'overlap': overlaps,
             'speaker_counts': {s: len(v) for s,v in speakers.items()},
             'human_numeric_coverage': {s: dict(Counter(str(extract_number(r['text'])) for r in good if r['split']==s and extract_number(r['text']) is not None)) for s in speakers}}
    write_json(ROOT / 'evidence/data_audit.json', audit)
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    main()
