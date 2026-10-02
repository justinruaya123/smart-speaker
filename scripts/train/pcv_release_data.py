"""Refresh the supplied HF source without replacing historical data or splits."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_public_data import sha,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings

HERE=ROOT/'experiments/pcv-source-refresh-v1'
CACHE=ROOT/'.cache/pcv-source-refresh-v1'
OLD=ROOT/'experiments/pcv-updated-voice-v1'
REPO='airimonda/ai231-me2-voice-commands'


def payload_plan(current,previous):
    old={Path(r['path'].replace('\\','/')).name:r for r in previous['files']}
    result=[]
    for row in current['siblings']:
        if not row['rfilename'].endswith('.parquet'):continue
        name=Path(row['rfilename']).name;prior=old.get(name)
        digest=row['lfs']['sha256'];size=row['size']
        reuse=prior['path'] if prior and prior['sha256']==digest and prior['bytes']==size else None
        result.append({'remote_path':row['rfilename'],'sha256':digest,'bytes':size,'reuse':reuse})
    return result


def movement_audit(old,new):
    def groups(rows):
        by_hash=defaultdict(set);speakers=defaultdict(set)
        for r in rows:
            by_hash[r.get('pcm_sha256')].add(r['published_split'])
            speakers[r['published_split']].add(str(r['source'])+'/'+str(r['speaker_id']))
        by_hash.pop(None,None);return by_hash,speakers
    a,sa=groups(old);b,sb=groups(new)
    moves={h:{'old':sorted(a[h]),'new':sorted(b[h])} for h in a.keys()&b.keys() if a[h]!=b[h]}
    forward=sorted(h for h in a.keys()&b.keys() if 'train' in a[h] and b[h]&{'test','holdout'})
    speaker_forward=sorted(sa['train']&(sb['test']|sb['holdout']))
    return {'shared_pcm_n':len(a.keys()&b.keys()),'new_pcm_n':len(b.keys()-a.keys()),
        'removed_pcm_n':len(a.keys()-b.keys()),'split_movements':moves,
        'prior_train_in_new_test_pcm':forward,'prior_train_speakers_in_new_test':speaker_forward,
        'prior_test_speakers_in_new_train':sorted((sa['test']|sa['holdout'])&sb['train']),
        'inherited_model_clean_for_new_test':not forward and not speaker_forward}


def refresh():
    import requests
    from huggingface_hub import hf_hub_download
    HERE.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    protocol_path=HERE/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text());verify_bindings(ROOT,protocol['bindings'])
    else:
        for path,ignored in [(CACHE/'payload.parquet',True),(HERE/'protocol.json',False)]:
            rc=subprocess.run(['git','check-ignore','--no-index','--quiet',str(path)],cwd=ROOT).returncode
            if rc not in (0,1) or (rc==0)!=ignored:raise ValueError('Ignore gate failed')
        response=requests.get(f'https://huggingface.co/api/datasets/{REPO}?blobs=true',timeout=60)
        response.raise_for_status();metadata=response.json()
        immutable(HERE/'publisher-metadata.json',metadata)
        previous=json.loads((OLD/'acquisition-complete.json').read_text())
        records=payload_plan(metadata,previous)
        for remote,local in [('README.md','source-card.txt'),('variations.csv','variations.csv')]:
            response=requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{metadata["sha"]}/{remote}',timeout=60)
            response.raise_for_status()
            with (HERE/local).open('xb') as stream:stream.write(response.content)
        sources=[Path(__file__),ROOT/'scripts/test/test_pcv_release_data.py',
            ROOT/'scripts/train/pcv_updated_data.py',OLD/'acquisition-complete.json',
            HERE/'publisher-metadata.json',HERE/'source-card.txt',HERE/'variations.csv']
        protocol={'experiment':'pcv-source-refresh-v1','repository':REPO,'revision':metadata['sha'],
            'previous_revision':previous['revision'],'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'requirements':['ME-01','ME-02','ME-03','ME-04','ME-06','ME-07','ME-08'],
            'payloads':records,'reuse':'Identical SHA256 and byte size; reference original immutable file, no duplicate payload.',
            'test_policy':'Compare PCM and declared speakers across revisions before any inherited-model use.',
            'bindings':{str(p.relative_to(ROOT)):sha(p) for p in sources},'ignore_gate_passed':True}
        immutable(protocol_path,protocol)
    if not (HERE/'acquisition-complete.json').exists():
        def fetch(record):
            path=ROOT/record['reuse'] if record['reuse'] else Path(hf_hub_download(repo_id=REPO,
                repo_type='dataset',filename=record['remote_path'],revision=protocol['revision'],local_dir=CACHE/'source'))
            if sha(path)!=record['sha256'] or path.stat().st_size!=record['bytes']:raise ValueError('Source identity changed')
            print('REFRESH_VERIFIED',record['remote_path'],'reused' if record['reuse'] else 'downloaded',flush=True)
            return {'path':str(path.relative_to(ROOT)),'sha256':record['sha256'],'bytes':record['bytes'],'reused':bool(record['reuse'])}
        with ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(fetch,protocol['payloads']))
        immutable(HERE/'acquisition-complete.json',{'revision':protocol['revision'],'repository':REPO,'files':files,
            'total_bytes':sum(x['bytes'] for x in files),'protocol_sha256':sha(protocol_path)})
    if not (HERE/'audio-audit.json').exists():
        from scripts.train import pcv_updated_data as base
        base.HERE=HERE;base.CACHE=CACHE;base.REVISION=protocol['revision'];base.PROTOCOL=protocol_path
        base.audit()
    previous_audit=json.loads((OLD/'audio-audit.json').read_text())
    new_audit=json.loads((HERE/'audio-audit.json').read_text())
    verify_bindings(ROOT,{previous_audit['manifest']:previous_audit['manifest_sha256'],new_audit['manifest']:new_audit['manifest_sha256']})
    old=read_rows(ROOT/previous_audit['manifest']);new=read_rows(ROOT/new_audit['manifest'])
    diff=movement_audit(old,new)
    diff.update({'old_manifest_sha256':previous_audit['manifest_sha256'],'new_manifest_sha256':new_audit['manifest_sha256'],
        'old_revision':protocol['previous_revision'],'new_revision':protocol['revision'],
        'prior_model_results_do_not_validate_updated_revision':True})
    immutable(HERE/'revision-difference.json',diff)
    print('REFRESH_COMPLETE',json.dumps({k:v for k,v in diff.items() if k not in ('split_movements','prior_train_in_new_test_pcm')}),flush=True)


if __name__=='__main__':refresh()
