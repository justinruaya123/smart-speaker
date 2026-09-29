"""Fresh waveform replay, exhaustive CSVs, and fail-closed release reporting."""
import csv
import json
from pathlib import Path
from collections import Counter
import time
import numpy as np
import soundfile as sf
from .integrity import save, digest, identity


def write_reports(rows,output,binding):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for r in rows:
        r['exact_correct']=r.get('expected') is not None and identity(r.get('command'))==identity(r['expected']) and r.get('accepted',False)
        r['raw_exact']=r.get('expected') is not None and identity(r.get('candidate'))==identity(r['expected'])
        r['binding']=binding
    subsets={'all_commands':rows,'accepted_commands':[r for r in rows if r['exact_correct']],
             'wrong_acceptances':[r for r in rows if r.get('accepted') and not r['exact_correct']],
             'failed_commands':[r for r in rows if r.get('valid_request') and not r['exact_correct']],
             'negative_results':[r for r in rows if not r.get('valid_request')]}
    columns=sorted(set().union(*(r.keys() for r in rows))) if rows else ['id','expected','candidate','command','accepted','exact_correct','reason','binding']
    for name,records in subsets.items():
        with (output/f'{name}.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns); writer.writeheader()
            for r in records: writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    valid=[r for r in rows if r.get('valid_request')]; negatives=[r for r in rows if not r.get('valid_request')]
    emitted=[r for r in rows if r.get('accepted')]; correct=len(subsets['accepted_commands'])
    per_family={}
    for r in rows:
        family=(r.get('expected') or {}).get('intent','UNSUPPORTED' if r.get('valid_request') else 'NEGATIVE')
        counts=per_family.setdefault(family,Counter())
        counts.update(total=1,raw_correct=int(r['raw_exact']),accepted=int(r.get('accepted',False)),correct_accepted=int(r['exact_correct']))
    summary={'examples':len(rows),'valid_requests':len(valid),'negatives':len(negatives),'correct_accepted':correct,
             'emitted':len(emitted),'accepted_recall':correct/len(valid) if valid else None,
             'accepted_precision':correct/len(emitted) if emitted else None,
             'negative_false_accepts':sum(r.get('accepted',False) for r in negatives),'raw_correct':sum(r['raw_exact'] for r in rows),
             'per_family':per_family,'binding':binding,'evidence':'fresh waveform replay' if rows and all('pcm_sha256' in r for r in rows) else 'software fixture'}
    save(output/'summary.json',summary)
    save(output/'release.json',{'eligible':False,'binding':binding,'summary_sha256':digest(output/'summary.json'),
                              'failed_requirements':['independent_human_confirmation','physical_pi4','all_ten_use_cases','negative_coverage','quality_gates'],
                              'physical_pi4_tested':False,'live_microphone_tested':False})
    return summary


def replay(bundle,manifest):
    from .runtime import Runtime
    from .data import read_manifest
    runtime=Runtime(bundle); rows=[]
    manifest_hash=digest(manifest)
    for index,r in enumerate(read_manifest(manifest)):
        if digest(r['audio'])!=r['audio_sha256']: raise ValueError('Audio hash mismatch')
        pcm,rate=sf.read(r['audio'],dtype='float32')
        if rate!=16000: raise ValueError('Replay requires prepared 16 kHz PCM')
        started=time.perf_counter(); decision=runtime.predict(pcm); elapsed=(time.perf_counter()-started)*1000
        rows.append({**r,**decision,'latency_ms':elapsed,'policy_sha256':digest(Path(bundle)/'policy.json'),
                     'manifest_sha256':manifest_hash,'evidence_type':'original-human-waveform'})
        if (index+1)%100==0: print('Replayed',index+1,flush=True)
    return rows,{**runtime.binding,'recipe':digest(Path(bundle)/'recipe.json')}


def evaluate(bundle,manifest,output):
    output=Path(output)
    if output.exists(): raise FileExistsError('Evaluation output exists; retain the previous receipt')
    rows,binding=replay(bundle,manifest)
    return write_reports(rows,output,binding)
