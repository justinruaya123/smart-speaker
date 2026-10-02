"""Pin, acquire and prepare evaluation WAVs; never fit or relabel from predictions."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/pcv-full-benchmark-v1'
CACHE = ROOT/'.cache/pcv-full-benchmark-v1'
REPO = 'airimonda/ai231-me2-voice-commands'
REVISION = '6947f13073e57eb6ae67e7e2fc3680700b82aa13'
GITHUB = '17d2dfe98f7dad4da3b5368082ecb47d2a4acd05'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Immutable receipt differs: '+str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)


def fetch(url, path, digest=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Partial payload remains separately named; never publish partial bytes.
        temporary = path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        if digest and sha(temporary) != digest:
            raise ValueError('Download hash mismatch: '+url)
        temporary.replace(path)
    if digest and sha(path) != digest:
        raise ValueError('Existing payload hash mismatch: '+str(path))
    return path


def evaluation_payloads(metadata):
    return [r for r in metadata['siblings'] if r['rfilename'].endswith('.parquet')
            and (Path(r['rfilename']).name.startswith(('test-', 'holdout-')))]


def acquire():
    for path, ignored in [(CACHE/'source/payload.parquet', True), (HERE/'protocol.json', False)]:
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--quiet', str(path)], cwd=ROOT)
        if result.returncode not in (0, 1) or (result.returncode == 0) != ignored:
            raise ValueError('Ignore gate failed: '+str(path))
    metadata_path = fetch(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true', HERE/'publisher-metadata.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata['sha'] != REVISION:
        raise ValueError('Source revision mismatch')
    for remote, local in [('README.md', 'source-card.txt'), ('variations.csv', 'variations.csv')]:
        fetch(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}', HERE/local)
    records = evaluation_payloads(metadata)
    if not records:
        raise ValueError('No evaluation payloads')
    protocol = {'experiment': 'pcv-full-benchmark-v1', 'repository': REPO, 'revision': REVISION,
        'prior_training_revision': '25111444af3adff7588d86ab27c304bb080895cf',
        'external_benchmark_commit': GITHUB, 'payloads': records,
        'model_release': 'experiments/pcv-release-train-v1/runs/release-v1/release.json',
        'release_sha256': sha(ROOT/'experiments/pcv-release-train-v1/runs/release-v1/release.json'),
        'annotation_policy': 'Frozen release-v1 metadata annotation; no prediction-driven corrections',
        'selection': 'Existing development-selected release-v1; no fitting or threshold selection',
        'evidence': 'Previously exposed test repeated on current snapshot; not a new blind test',
        'requirements': ['ME-03', 'ME-04', 'ME-06', 'ME-07', 'ME-08'],
        'ignore_gate_passed': True}
    save(HERE/'protocol.json', protocol)
    def download(row):
        remote = row['rfilename']
        path = fetch(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}',
                     CACHE/'source'/remote, row['lfs']['sha256'])
        if path.stat().st_size != row['size']:
            raise ValueError('Source size mismatch')
        print('VERIFIED', remote, path.stat().st_size, flush=True)
        return {'remote_path': remote, 'path': path.relative_to(ROOT).as_posix(),
                'sha256': sha(path), 'bytes': path.stat().st_size}
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, records))
    save(HERE/'acquisition.json', {'repository': REPO, 'revision': REVISION, 'files': files,
                                 'protocol_sha256': sha(HERE/'protocol.json')})


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.train.pcv_release_prepare import annotate, annotation_policy
    from scripts.train.pcv_updated_prepare import spoken
    from scripts.runtime.pcv_schema import canonical
    release = ROOT/'.cache/pcv-release-train-v1/prepared'
    identities = defaultdict(lambda: defaultdict(set))
    speakers = defaultdict(set)
    bindings = {}
    for split in ('train', 'dev', 'calibration', 'test', 'holdout'):
        path = release/(split+'.jsonl');bindings[path.relative_to(ROOT).as_posix()] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            identities[row['pcm_sha256']][split].add(row['id'])
            speakers[split].add(str(row['source'])+'/'+str(row['speaker_id']))
    definitions = json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    phrases = {spoken(d.get('spoken_text', d['text'])): d['id'] for d in definitions}
    policy = annotation_policy()
    records = json.loads((HERE/'acquisition.json').read_text())['files']
    rows = [];audio_groups = defaultdict(list);counts = Counter();overlap = Counter()
    destination = CACHE/'corpus';destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        parquet = pq.ParquetFile(ROOT/record['path'])
        config = record['remote_path'].split('/')[0]
        split = Path(record['remote_path']).name.split('-')[0]
        track = ('primary_' if config == 'data' else config+'_')+split
        for batch in parquet.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                index = counts[track];counts[track] += 1
                audio = row.pop('audio')['bytes']
                pcm, rate = sf.read(io.BytesIO(audio), dtype='float32')
                if rate != 16000 or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
                    raise ValueError('Invalid audio '+track+'/'+str(index))
                digest = hashlib.sha256(audio).hexdigest()
                pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
                relative = 'audio/'+digest+'.wav'
                path = destination/relative;path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    with path.open('xb') as stream:stream.write(audio)
                elif sha(path) != digest:raise ValueError('Audio identity changed')
                annotation = annotate({**row, 'published_split': split}, policy)
                seen = identities.get(pcm_hash, {})
                speaker = str(row.get('source'))+'/'+str(row.get('speaker_id'))
                item = {'id':track+'/'+str(index), 'track':track, 'audio':relative, 'sha256':digest,
                    'pcm_sha256':pcm_hash, 'seconds':len(pcm)/rate, 'source':row.get('source'),
                    'speaker':speaker, 'synthetic':bool(row.get('is_synthetic')),
                    'phrase_id':phrases.get(annotation['text']), **annotation,
                    'publisher_intent':row.get('command'), 'publisher_slot':row.get('slot_value', ''),
                    'publisher_variation':row.get('variation', ''), 'publisher_transcript':row.get('transcript', ''),
                    'prior_pcm_splits':sorted(seen),
                    'prior_speaker_splits':sorted(s for s, group in speakers.items() if speaker in group)}
                for seen_split in seen:overlap[track+'/'+seen_split] += 1
                rows.append(item);audio_groups[pcm_hash].append(item['id'])
        print('PREPARED', track, counts[track], flush=True)
    save(destination/'manifest.json', rows)
    save(destination/'definitions.json', definitions)
    duplicates = {h:ids for h, ids in audio_groups.items() if len(ids)>1}
    conflicts = []
    lookup = {r['id']:r for r in rows}
    for h, ids in duplicates.items():
        labels = {canonical(lookup[i]['expected']) for i in ids if lookup[i]['reference_status'] != 'ambiguous'}
        if len(labels)>1:conflicts.append({'pcm_sha256':h, 'ids':ids})
    save(HERE/'source-audit.json', {'revision':REVISION, 'rows':len(rows), 'tracks':dict(counts),
        'references':dict(Counter(r['track']+'/'+r['reference_status'] for r in rows)),
        'prior_pcm_overlap':dict(overlap), 'duplicate_pcm_groups':duplicates, 'conflicting_labels':conflicts,
        'prior_speaker_overlap':dict(Counter(r['track']+'/'+s for r in rows for s in r['prior_speaker_splits'])),
        'new_pcm_n':sum(not r['prior_pcm_splits'] for r in rows),
        'labels_independently_listened':False, 'synthetic_voice_lineage_independent':False,
        'prior_manifest_bindings':bindings, 'manifest_sha256':sha(destination/'manifest.json'),
        'annotation_source_sha256':sha(ROOT/'scripts/train/pcv_release_prepare.py'),
        'source_script_sha256':sha(Path(__file__))})
    print('CORPUS_READY', len(rows), dict(counts), flush=True)


if __name__ == '__main__':
    acquire()
    prepare()
