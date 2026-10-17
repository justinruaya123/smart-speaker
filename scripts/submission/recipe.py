"""Export portable, privacy-reduced row maps without rewriting frozen manifests."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .common import ROOT, read, save, sha

PREPARED = ROOT/'.cache/speaker-synthetic-completion-v1/prepared-v1'


def jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def export(model_choice='full'):
    full = model_choice == 'full'
    destination = ROOT/('submission/reproduction-full' if full else 'submission/reproduction')
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}
    for receipt in ('data/sources/ai231-primary.source.json', 'data/sources/fil-supplemental.source.json'):
        document = read(ROOT/receipt)
        for record in document['files']:
            sources[record['path']] = dict(repository=document['repository'], revision=document['revision'],
                file=record.get('remote_path') or record['url'].split('/'+document['revision']+'/')[1].split('?')[0],
                sha256=record['sha256'], bytes=record['bytes'], url=record['url'])
    views = {
        'train':PREPARED/'train.jsonl' if full else ROOT/'.cache/speaker-compact-recall-v1/prepared/train.jsonl',
        'dev':PREPARED/'dev.jsonl' if full else ROOT/'.cache/speaker-compact-recall-v1/prepared/dev.jsonl',
        'calibration':PREPARED/'views/human/calibration.jsonl',
        'test':PREPARED/'views/evaluation/test.jsonl',
        'holdout':PREPARED/'views/evaluation/holdout.jsonl',
    }
    all_rows = {}; view_ids = {}; view_hashes = {}; auxiliary = []
    for view,path in views.items():
        rows = jsonl(path)
        if view == 'train' and full:
            rows = [r for r in rows if not r['id'].startswith('qsc1/')]
        view_ids[view] = [r['id'] for r in rows]; view_hashes[view] = sha(path)
        for row in rows:
            if row['id'] in all_rows: continue
            speaker = row.get('speaker',row.get('speaker_id'))
            portable = {k:row[k] for k in ('id','kind','split','source','sha256','pcm_sha256','metric_eligible','reference_reason') if k in row}
            portable.update(speaker='speaker-'+hashlib.sha256(speaker.encode()).hexdigest()[:24],
                expected=row.get('expected',row.get('command')), synthetic=row['kind']=='synthetic',
                training_eligible=view=='train', phones=row.get('phones'),
                intent_index=row.get('intent_index'), target_index=row.get('target_index'),
                features_sha256=row.get('features_sha256'))
            if 'parquet' in row:
                portable['location'] = {**sources[row['parquet']], 'row':row['parquet_row']}
            else:
                portable['location'] = dict(auxiliary_file=row['sha256']+'.wav')
                auxiliary.append(dict(file=row['sha256']+'.wav', local_path=row['audio'], sha256=row['sha256']))
            all_rows[row['id']] = portable
    # Include the six compact-excluded training rows for normalization ancestry.
    original_train = jsonl(PREPARED/'train.jsonl')
    norm = read(PREPARED/'normalization.json')
    norm_ids = set(norm['source_ids'])
    for row in original_train:
        if row['id'] in norm_ids and row['id'] not in all_rows:
            portable = {k:row[k] for k in ('id','kind','split','source','sha256','pcm_sha256','metric_eligible','reference_reason','phones','intent_index','target_index','features_sha256')}
            portable.update(speaker='speaker-'+hashlib.sha256(row['speaker'].encode()).hexdigest()[:24],
                            expected=row['expected'], synthetic=False, training_eligible=False,
                            location={**sources[row['parquet']], 'row':row['parquet_row']})
            all_rows[row['id']] = portable
    view_ids['normalization'] = norm['source_ids']
    packed = destination/'rows.jsonl.gz'
    payload = ''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in all_rows.values()).encode()
    packed.write_bytes(gzip.compress(payload,mtime=0))
    original_protocol = ROOT/('experiments/speaker-final-refinement-v1/protocol.json' if full else 'experiments/speaker-compact-recall-v1/protocol.json')
    metadata = dict(format='smart-speaker-reproduction-v1', original_seed=53, model_choice=model_choice,
        training_experiment='speaker-final-refinement-v1' if full else 'speaker-compact-recall-v1',
        selected_experiment='speaker-final-refinement-v1' if full else 'speaker-pipeline-repair-v1',
        architecture=read(original_protocol)['architecture'], training=read(original_protocol)['training'],
        selected_human_epoch=20 if full else 8, selected_mixed_epoch=32 if full else 28,
        parameters=2066831 if full else 935567,
        reproduction_scope='Selected human-foundation -> mixed-control lineage; the losing 2M completion arm is preserved as historical comparison, not rerun.',
        rows_file=packed.name, rows_sha256=sha(packed), views=view_ids, frozen_original_manifest_sha256=view_hashes,
        normalization=dict(path=f'submission/models/{model_choice}/normalization.json',sha256=sha(ROOT/f'submission/models/{model_choice}/normalization.json')),
        catalog=dict(path=f'submission/models/{model_choice}/catalog.json',sha256=sha(ROOT/f'submission/models/{model_choice}/catalog.json')),
        source_receipts=['data/sources/ai231-primary.source.json','data/sources/fil-supplemental.source.json'],
        privacy='Publisher student/speaker metadata omitted; speaker equality preserved by deterministic pseudonyms.',
        numerical_reproducibility='Seeded CUDA CTC is not bitwise deterministic. Compare tensors, development choice and held-out metrics.',
        protocol_sha256=sha(original_protocol),
        selected_checkpoint_sha256=('ff6a3af63115a8f92a0106b20a17ee57c3e84f263177b89aaedb9cce326fce9b' if full else 'a727e5070053045d6f276a3f58084ef16dd2bcbf9f931eea71db42a512f7d76a'))
    save(destination/'recipe.json', metadata)
    upload_path=ROOT/'.cache/submission/auxiliary-upload.json'
    if upload_path.exists():
        if read(upload_path)!=auxiliary:raise ValueError('Auxiliary upload membership changed')
    else:save(upload_path, auxiliary)
    shared_auxiliary=ROOT/'submission/reproduction/auxiliary-source.json'
    if full and shared_auxiliary.exists():
        (destination/'auxiliary-source.json').write_bytes(shared_auxiliary.read_bytes())
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--export-original',action='store_true',required=True)
    parser.add_argument('--model-choice',choices=('full','compact'),default='full')
    args=parser.parse_args(); result=export(args.model_choice); print('Portable recipe:',args.model_choice,len(result['views']['train']),'train rows')
