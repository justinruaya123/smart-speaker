"""Pinned acquisition and waveform/provenance audit of the supplied ME2 dataset."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.pcv_public_data import sha, save
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings, verify_parent

REPO = 'airimonda/ai231-me2-voice-commands'
REVISION = 'e3e203b360d5c3205d59f99ad65cd380f9c0f517'
HERE = ROOT / 'experiments/pcv-updated-voice-v1'
CACHE = ROOT / '.cache/pcv-updated-voice-v1'
SOURCE = CACHE / 'source'
PROTOCOL = HERE / 'acquisition-protocol.json'


def get_json(url):
    import requests
    response = requests.get(url, timeout=40)
    response.raise_for_status()
    return response.json()


def freeze():
    import requests
    verify_parent()
    if PROTOCOL.exists():
        verify_bindings(ROOT, json.loads(PROTOCOL.read_text())['bindings'])
        return
    for path, ignored in [(CACHE/'source/data.parquet', True), (HERE/'protocol.json', False)]:
        code = subprocess.run(['git','check-ignore','--no-index','--quiet',str(path)], cwd=ROOT).returncode
        if code not in (0,1) or (code == 0) != ignored:
            raise ValueError('Ignore-first gate failed: '+str(path))
    info = get_json(f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}')
    if info['sha'] != REVISION:
        raise ValueError('Unexpected publisher revision')
    files = get_json(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data?recursive=false&expand=false')
    files = [{k:r[k] for k in ('path','size','lfs')} for r in files if r['type']=='file']
    if len(files) != 10 or any(not r['path'].endswith('.parquet') for r in files):
        raise ValueError('Unexpected source inventory')
    immutable(HERE/'publisher-metadata.json',info)
    for source, name in [('README.md','source-card.txt'),('variations.csv','variations.csv')]:
        response=requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REVISION}/{source}',timeout=40)
        response.raise_for_status()
        path=HERE/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(response.content)
    paths=[Path(__file__),HERE/'publisher-metadata.json',HERE/'source-card.txt',HERE/'variations.csv']
    immutable(PROTOCOL, {'version':'pcv-updated-voice-v1-acquisition', 'repository':REPO,'revision':REVISION,
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),'requirements':['ME-01','ME-02','ME-03','ME-04','ME-06','ME-07','ME-08','UC-01..UC-10'],
        'order':['human recordings first','supplied synthetic recordings second','our synthesis only for measured remaining gaps'],
        'payloads':files,'ignore_gate_passed':True,'test_inference_accessed':False,
        'source_terms':'Preserve each underlying source license; source-card.txt includes class-only sharing limits for group recordings',
        'audit_scope':'All source rows, declared labels, file/PCM identity, speakers, waveform checks; no ASR or model inference',
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


def acquire():
    from huggingface_hub import hf_hub_download
    protocol=json.loads(PROTOCOL.read_text())
    verify_bindings(ROOT,protocol['bindings'])
    SOURCE.mkdir(parents=True,exist_ok=True)
    def one(record):
        path=Path(hf_hub_download(repo_id=REPO,repo_type='dataset',filename=record['path'],revision=REVISION,
                                 local_dir=SOURCE))
        value=sha(path)
        if path.stat().st_size!=record['size'] or value!=record['lfs']['oid']:
            raise ValueError('Publisher file identity failed: '+record['path'])
        print('SOURCE_VERIFIED',record['path'],record['size'],flush=True)
        return {'path':str(path.relative_to(ROOT)),'sha256':value,'bytes':record['size']}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(one,protocol['payloads']))
    verify_bindings(ROOT,protocol['bindings'])
    immutable(HERE/'acquisition-complete.json',{'repository':REPO,'revision':REVISION,'files':results,
        'total_bytes':sum(r['bytes'] for r in results),'protocol_sha256':sha(PROTOCOL)})


def audit():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from einops import reduce
    protocol=json.loads(PROTOCOL.read_text())
    verify_bindings(ROOT,protocol['bindings'])
    acquired=json.loads((HERE/'acquisition-complete.json').read_text())
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in acquired['files']})
    counts=defaultdict(Counter)
    speakers=defaultdict(set)
    qualified_speakers=defaultdict(set)
    file_groups=defaultdict(list)
    pcm_groups=defaultdict(list)
    problems=[]
    output=CACHE/'audited-rows.jsonl'
    total=0;began=time.perf_counter()
    with output.open('x',encoding='utf8') as stream:
        for record in acquired['files']:
            path=ROOT/record['path'];split=path.name.split('-')[0]
            parquet=pq.ParquetFile(path)
            index=0
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    audio=row.pop('audio');identity=f'{split}/{path.stem}/{index:06d}'
                    index+=1
                    item={**row,'id':identity,'published_split':split,'parquet':record['path'],'parquet_row':index-1}
                    payload=audio.get('bytes') if audio else None
                    issue=[]
                    if payload is None:
                        issue.append('missing_embedded_audio')
                    else:
                        audio_hash=hashlib.sha256(payload).hexdigest()
                        item['audio_sha256']=audio_hash
                        file_groups[audio_hash].append(identity)
                        try:
                            info=sf.info(io.BytesIO(payload))
                            pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
                            if rate!=16000 or pcm.ndim!=1 or info.subtype!='PCM_16':
                                issue.append('unexpected_audio_format')
                            if pcm.size==0 or not np.isfinite(pcm).all():
                                issue.append('empty_or_nonfinite_audio')
                            else:
                                pcm_hash=hashlib.sha256(pcm.tobytes()).hexdigest()
                                pcm_groups[pcm_hash].append(identity)
                                duration=len(pcm)/rate
                                active=np.flatnonzero(np.abs(pcm)>.01)
                                item.update(pcm_sha256=pcm_hash,sample_rate=rate,channels=info.channels,
                                    subtype=info.subtype,samples=len(pcm),decoded_duration_seconds=duration,
                                    rms=float(np.sqrt(reduce(pcm*pcm,'samples ->','mean'))),
                                    clipped_fraction=float(reduce((np.abs(pcm)>=.999).astype(np.float32),'samples ->','mean')),
                                    leading_below_001_seconds=float(active[0]/rate) if len(active) else duration,
                                    trailing_below_001_seconds=float((len(pcm)-1-active[-1])/rate) if len(active) else duration)
                                if not .2<=duration<=12: issue.append('outside_current_endpoint_duration')
                                if item['rms']<.001: issue.append('low_rms')
                                if item['clipped_fraction']>.01: issue.append('high_clipped_fraction')
                        except Exception as exc:
                            issue.append('decode_error:'+type(exc).__name__+':'+str(exc))
                    item['waveform_flags']=issue
                    if issue: problems.append({'id':identity,'flags':issue})
                    for key in ('source','command','is_synthetic','bucket','accent_group','transcript_source','whisper_check'):
                        counts[split+'/'+key][str(row.get(key))]+=1
                    counts['rows'][split]+=1
                    speakers[split].add(str(row['speaker_id']))
                    qualified_speakers[split].add(str(row['source'])+'/'+str(row['speaker_id']))
                    stream.write(json.dumps(item,ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n')
                    total+=1
                if total%1000<128:
                    print('AUDIO_AUDIT',total,'seconds',round(time.perf_counter()-began,1),flush=True)
            if index!=parquet.metadata.num_rows:raise ValueError('Parquet row count mismatch')
    split_names=sorted(speakers)
    overlaps={a+'/'+b:sorted(speakers[a]&speakers[b]) for i,a in enumerate(split_names) for b in split_names[i+1:]}
    qualified={a+'/'+b:sorted(qualified_speakers[a]&qualified_speakers[b]) for i,a in enumerate(split_names) for b in split_names[i+1:]}
    duplicates=lambda groups:{h:ids for h,ids in groups.items() if len(ids)>1}
    duplicate_file=duplicates(file_groups);duplicate_pcm=duplicates(pcm_groups)
    cross=lambda groups:{h:ids for h,ids in groups.items() if len({i.split('/')[0] for i in ids})>1}
    immutable(HERE/'audio-audit.json',{'revision':REVISION,'rows':total,'counts':{k:dict(v) for k,v in counts.items()},
        'speaker_counts':{k:len(v) for k,v in speakers.items()},'speaker_overlap':overlaps,
        'source_qualified_speaker_overlap':qualified,'file_duplicate_groups':duplicate_file,'pcm_duplicate_groups':duplicate_pcm,
        'cross_split_file_duplicates':cross(duplicate_file),'cross_split_pcm_duplicates':cross(duplicate_pcm),
        'waveform_flags':problems,'labels_independently_listened':False,'test_inference_accessed':False,
        'manifest':str(output.relative_to(ROOT)),'manifest_sha256':sha(output),'seconds':time.perf_counter()-began,
        'acquisition_sha256':sha(HERE/'acquisition-complete.json'),'audit_source_sha256':sha(Path(__file__))})
    print('AUDIT_COMPLETE',total,'file_dups',len(duplicate_file),'pcm_dups',len(duplicate_pcm),'flags',len(problems),flush=True)


if __name__=='__main__':
    if sys.argv[1:] not in (['freeze'],['acquire'],['audit']):raise SystemExit('Use freeze, acquire or audit')
    globals()[sys.argv[1]]()
