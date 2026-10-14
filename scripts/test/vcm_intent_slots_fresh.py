"""Freeze unused-person Fluent speech with exact, unchanged catalog references."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf

from scripts.runtime.pcv_schema import canonical, normalize_text
from scripts.test.speaker_demo_suite import pcm_digest
from scripts.train.vcm_intent_slots_data import rows, sha, write_new

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT.parents[1]
PREFIX = 'fluent_speech_commands_dataset/'


def prepare(archive, output):
    archive, output = Path(archive), Path(output)
    source_path = ROOT/'data/sources/fluent.source.json'
    source = json.loads(source_path.read_text())
    if sha(archive) != source['sha256'] or archive.stat().st_size != source['bytes']:
        raise ValueError('Fluent source archive integrity failure')
    base_path = ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json'
    base = json.loads(base_path.read_text())
    exposed = []
    for binding in base['views'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Changed reserved human/regression views')
        exposed += rows(binding['path'])
    people = {r['speaker_id'] for r in exposed}
    reserved_pcm = {r['pcm_sha256'] for r in exposed}
    reserved_file = {r['sha256'] for r in exposed}
    reserved_groups = {r.get('pair_id') for r in exposed if r.get('pair_id')}
    contract = json.loads((ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json').read_text())
    references = {}
    for r in contract['cases']+contract['catalog_entries']:
        text = normalize_text(r.get('spoken_text', r.get('text')))
        if text in references and canonical(references[text]['command']) != canonical(r['command']):
            raise ValueError('Conflicting preserved text reference')
        references[text] = r
    source_rows, demographics = [], []
    with tarfile.open(archive, 'r:gz') as tar:
        for partition in ('train', 'valid', 'test'):
            name = PREFIX+'data/'+partition+'_data.csv'
            data = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(name), encoding='utf8')))
            for row in data:
                if 'FluentSpeechCommands/'+row['speakerId'] not in people:
                    source_rows.append({**row, 'original_partition': partition})
        demographics = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(
            PREFIX+'data/speaker_demographics.csv'), encoding='utf8')))
    wanted = {PREFIX+r['path']: r for r in source_rows}
    if len(wanted) != len(source_rows):
        raise ValueError('Repeated original Fluent recording identity')
    output.mkdir(parents=True, exist_ok=False)
    payload = MAIN/'data/audio/fluent-unseen-person-20261015-v2'
    payload.mkdir(parents=True, exist_ok=True)
    selected, unmatched, decoded = [], [], set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            original = wanted[member.name]
            data = tar.extractfile(member).read()
            file_hash = hashlib.sha256(data).hexdigest()
            pcm, rate = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
            if rate != 16000 or pcm.ndim != 1 or not 400 <= len(pcm) <= 128000:
                raise ValueError('Fresh original speech violates unchanged audio contract')
            pcm_hash = pcm_digest(pcm, rate)
            group = original['path']
            if file_hash in reserved_file or pcm_hash in reserved_pcm or group in reserved_groups:
                raise ValueError('Fresh evaluation recording contaminates reserved views')
            if pcm_hash in decoded:
                raise ValueError('Duplicate fresh recording PCM')
            decoded.add(pcm_hash)
            target = payload/file_hash[:2]/(file_hash+'.wav')
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != file_hash:
                    raise ValueError('Shared immutable recording changed')
            else:
                with target.open('xb') as stream:
                    stream.write(data)
            text = original['transcription']
            reference = references.get(normalize_text(text))
            row = {'id': 'fresh-fluent-v2/'+file_hash, 'audio': str(target.resolve()),
                   'speaker_id': 'FluentSpeechCommands/'+original['speakerId'],
                   'source': 'FluentSpeechCommands', 'source_revision': source['revision'],
                   'license': source['license'], 'kind': 'human', 'role': 'command',
                   'split': 'fresh_human', 'recording_group': group, 'pair_id': group,
                   'sha256': file_hash, 'pcm_sha256': pcm_hash, 'samples': len(pcm),
                   'sample_rate': rate, 'channels': 1, 'transcript': text,
                   'original_path': original['path'], 'original_partition': original['original_partition'],
                   'original_fluent_labels': {k: original[k] for k in ('action', 'object', 'location')},
                   'fresh_person_against_all_original_views': True,
                   'acoustic_listening_verified': False}
            if reference is None:
                unmatched.append({**row, 'metric_eligible': False,
                                  'reason': 'no exact preserved reference; no guessed OOV label'})
            else:
                selected.append({**row, 'command': reference['command'],
                                 'reference_case': reference['id'], 'metric_eligible': True,
                                 'reference_status': 'supported' if reference['command'] is not None else 'unsupported',
                                 'reference_reason': reference.get('reason')})
    if len(selected)+len(unmatched) != len(source_rows):
        raise ValueError('Missing original fresh recordings')
    for name, values in [('fresh-human.jsonl', selected), ('unmatched.jsonl', unmatched)]:
        with (output/name).open('x', encoding='utf8') as stream:
            for row in sorted(values, key=lambda r:r['id']):
                stream.write(canonical(row)+'\n')
    fresh_people = {r['speaker_id'] for r in selected+unmatched}
    demo = [r for r in demographics if 'FluentSpeechCommands/'+r['speakerId'] in fresh_people]
    write_new(output/'demographics.json', demo)
    report = {'source': {'path': str(archive.resolve()), 'sha256': sha(archive), 'revision': source['revision'],
                         'metadata_sha256': sha(source_path)},
              'script_sha256': sha(__file__), 'contract_hash': contract['contract_hash'],
              'available_fresh_recordings': len(source_rows), 'exact_reference_recordings': len(selected),
              'unmatched_reference_recordings': len(unmatched), 'fresh_declared_people': len(fresh_people),
              'supported': sum(r['command'] is not None for r in selected),
              'negative': sum(r['command'] is None for r in selected),
              'intents': dict(Counter((r['command'] or {}).get('intent','OOV') for r in selected)),
              'file_pcm_person_group_overlap': 0,
              'identity_limits': 'Verified dataset IDs; cross-source physical-person aliases not acoustically adjudicated',
              'independent_listening_performed': False, 'VCM_results_accessed': False,
              'manifest': {'path': str((output/'fresh-human.jsonl').resolve()),
                           'sha256': sha(output/'fresh-human.jsonl')},
              'unmatched_sha256': sha(output/'unmatched.jsonl')}
    write_new(output/'audit.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.archive, args.output)), flush=True)
