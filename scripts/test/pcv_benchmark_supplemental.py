"""Add reserved supplemental voices without treating their train filename as a split."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil

from scripts.test.pcv_benchmark_source import ROOT, HERE, CACHE, REPO, REVISION, sha, fetch, save


def evaluation_split(row):
    split = row.get('voice_split')
    if split not in ('train','test','holdout'):raise ValueError('Missing or unknown supplemental voice split')
    return None if split=='train' else split


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.train.pcv_release_prepare import annotate, annotation_policy
    from scripts.train.pcv_updated_prepare import spoken
    metadata = json.loads((HERE/'publisher-metadata.json').read_text())
    records = [r for r in metadata['siblings'] if r['rfilename'].startswith('supplemental_synth/') and r['rfilename'].endswith('.parquet')]
    base = CACHE/'corpus'
    save(HERE/'supplemental-protocol.json', {'version':'supplemental-evaluation-v1', 'revision':REVISION,
        'payloads':records, 'base_manifest_sha256':sha(base/'manifest.json'),
        'split_rule':'Use voice_split test and holdout only; exclude voice_split train; unknown split is fatal',
        'selection':'Frozen release-v1, no fitting/calibration/catalog changes',
        'prior_primary_run':'pc-single-core preserved; expanded corpus measured in pc-full-single-core',
        'annotation_source_sha256':sha(ROOT/'scripts/train/pcv_release_prepare.py'),
        'source_script_sha256':sha(Path(__file__))})
    identities = defaultdict(set);speakers = defaultdict(set)
    for split in ('train','dev','calibration','test','holdout'):
        for line in (ROOT/'.cache/pcv-release-train-v1/prepared'/(split+'.jsonl')).read_text().splitlines():
            row = json.loads(line);identities[row['pcm_sha256']].add(split)
            speakers[split].add(str(row['source'])+'/'+str(row['speaker_id']))
    definitions = json.loads((base/'definitions.json').read_text())
    phrases = {spoken(d.get('spoken_text',d['text'])):d['id'] for d in definitions}
    policy = annotation_policy();counts = Counter();rows = [];files = [];overlap = Counter()
    for record in records:
        remote = record['rfilename']
        path = fetch(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}', CACHE/'source'/remote,record['lfs']['sha256'])
        files.append({'remote_path':remote,'sha256':sha(path),'bytes':path.stat().st_size})
        print('SUPPLEMENTAL_VERIFIED',remote,flush=True)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                split = evaluation_split(row);counts['source/'+row['voice_split']] += 1
                if split is None:continue
                payload = row.pop('audio')['bytes'];pcm,rate = sf.read(io.BytesIO(payload),dtype='float32')
                if rate!=16000 or pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('Invalid supplemental PCM')
                digest = hashlib.sha256(payload).hexdigest();pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
                relative = 'audio/'+digest+'.wav';audio = base/relative
                if audio.exists():
                    if sha(audio)!=digest:raise ValueError('Audio identity mismatch')
                else:
                    with audio.open('xb') as stream:stream.write(payload)
                annotation = annotate({**row,'published_split':split},policy)
                speaker = str(row.get('source'))+'/'+str(row.get('speaker_id'))
                track = 'supplemental_'+split;index = counts[track];counts[track] += 1
                seen = sorted(identities.get(pcm_hash,set()))
                for prior in seen:overlap[track+'/'+prior] += 1
                rows.append({'id':track+'/'+str(index),'track':track,'audio':relative,'sha256':digest,
                    'pcm_sha256':pcm_hash,'seconds':len(pcm)/rate,'source':row.get('source'),
                    'speaker':speaker,'synthetic':bool(row.get('is_synthetic')),'voice_split':split,
                    'phrase_id':phrases.get(annotation['text']),**annotation,
                    'publisher_intent':row.get('command'),'publisher_slot':row.get('slot_value',''),
                    'publisher_variation':row.get('variation',''),'publisher_transcript':row.get('transcript',''),
                    'prior_pcm_splits':seen,'prior_speaker_splits':sorted(s for s,g in speakers.items() if speaker in g)})
    if any(set(r['prior_pcm_splits']) & {'train','dev','calibration'} for r in rows):
        raise ValueError('Supplemental evaluation overlaps selected model fitting by PCM')
    save(base/'supplemental-manifest.json',rows)
    original = json.loads((base/'manifest.json').read_text())
    combined = original+rows
    save(base/'full-manifest.json',combined)
    groups = defaultdict(list)
    for row in combined:groups[row['pcm_sha256']].append(row['id'])
    save(HERE/'supplemental-audit.json',{'revision':REVISION,'files':files,'counts':dict(counts),
        'eval_n':len(rows),'full_n':len(combined),'prior_pcm_overlap':dict(overlap),
        'prior_speaker_overlap':dict(Counter(r['track']+'/'+s for r in rows for s in r['prior_speaker_splits'])),
        'full_manifest_sha256':sha(base/'full-manifest.json'),'supplemental_manifest_sha256':sha(base/'supplemental-manifest.json'),
        'full_duplicate_pcm_groups':{h:ids for h,ids in groups.items() if len(ids)>1},
        'labels_independently_listened':False,'voice_lineage':'Published voice_split retained; speaker IDs are not independent humans'})
    print('FULL_CORPUS_READY',len(combined),dict(counts),flush=True)


if __name__=='__main__':main()
