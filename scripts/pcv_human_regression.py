"""Historical human regression under one fixed catalog, without policy selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
import hashlib
import json
import time
from collections import defaultdict
import numpy as np
from einops import pack
from pcvcm.contract import canonical,file_hash,identity,write_json
from pcvcm.endpoint import EndpointConfig
from pcvcm.runtime import StreamEngine,OnnxEncoder
from scripts.runtime.pcv_deploy.catalog import CatalogContract
from scripts.runtime.pcv_deploy.bundle import STATE_SHAPES


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['expected'] or {}).get('intent','NEGATIVE')].append(row)
    result={}
    for name,values in groups.items():
        result[name]=dict(trials=len(values),baseline_raw_correct=sum(r['baseline']['raw_correct'] for r in values),
            candidate_raw_correct=sum(r['candidate']['raw_correct'] for r in values),
            baseline_faults=sum(bool(r['baseline']['capture_fault']) for r in values),
            candidate_faults=sum(bool(r['candidate']['capture_fault']) for r in values))
    positives=[r for r in rows if r['expected'] is not None]
    return dict(examples=len(rows),positives=len(positives),negatives=len(rows)-len(positives),per_intent=result,
        baseline_raw_correct=sum(r['baseline']['raw_correct'] for r in positives),
        candidate_raw_correct=sum(r['candidate']['raw_correct'] for r in positives),
        accepted_accuracy_measured=False,policy_calibrated=False,human_qualified=False,
        evidence='historical original-human raw-command regression, same catalog and endpoint')


def replay(engine,pcm,expected):
    engine.reset();delivered,_=pack([pcm,np.zeros(12800,dtype=np.float32)],'*')
    started=time.perf_counter()
    for start in range(0,len(delivered),640):
        engine.feed(delivered[start:start+640],start)
        if engine.finalized or engine.endpoint.fault:break
    engine.eof()
    raw=engine.candidate['command']
    return dict(candidate=raw,raw_correct=canonical(raw)==canonical(expected),
        capture_fault=engine.endpoint.fault,endpoint_detected=engine.finalized,
        inference_seconds=time.perf_counter()-started,timing=engine.timing)


def evaluate(candidate,output):
    candidate,output=Path(candidate),Path(output)
    if output.exists():raise FileExistsError('Preserve prior regression evidence')
    protocol_path=ROOT/'experiments/pcv-delivery-v1/human-regression.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    for name,digest in protocol['frozen_inputs'].items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Historical regression inputs changed: '+name)
    manifest=ROOT/protocol['manifest'];baseline=ROOT/protocol['baseline_bundle']
    rows=[json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    if len(rows)!=protocol['examples'] or any(r.get('synthetic') is not False or r.get('split')!='test' for r in rows):
        raise ValueError('Regression requires the frozen original-human test manifest')
    # Source/bundle validation occurs before reading test audio. This never selects
    # a candidate or threshold and does not touch the sealed synthetic test set.
    sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
    from scripts.pcv_study_context import bind_study
    with bind_study(candidate):
        from synthbench import load_runtime
        selected=load_runtime(candidate)
    old=json.loads((baseline/'catalog.json').read_text(encoding='utf-8'))
    catalog=dict(entries=[dict(id=str(i),phones=r['phones'],command=r['command']) for i,r in enumerate(old['commands'])])
    old_norm=json.loads((baseline/'recipe.json').read_text(encoding='utf-8'))['frontend']
    new_norm=json.loads((candidate/'normalization.json').read_text(encoding='utf-8'))
    endpoint=EndpointConfig(**protocol['endpoint'])
    engines={}
    for name,graph,norm in [('baseline',baseline/'model.onnx',old_norm),('candidate',candidate/'model.onnx',new_norm)]:
        binding=identity(dict(model=file_hash(graph),catalog=catalog,mean=norm['mean'],std=norm['std']))
        engines[name]=StreamEngine(CatalogContract(catalog),catalog,OnnxEncoder(graph,STATE_SHAPES,threads=1),norm,endpoint,
            dict(binding=binding,enabled=False,thresholds={}),binding)
    output.mkdir(parents=True)
    write_json(output/'started.json',dict(complete=False,candidate_binding=selected.binding,
        protocol_sha256=file_hash(protocol_path),expected=len(rows),no_selection_or_tuning=True))
    import soundfile as sf
    results=[]
    with (output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for index,row in enumerate(rows):
            if file_hash(row['audio'])!=row['audio_sha256']:raise ValueError('Original audio changed')
            pcm,rate=sf.read(row['audio'],dtype='float32')
            if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all() or hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']:
                raise ValueError('Original PCM changed')
            result={k:row[k] for k in ('id','source','speaker','text','expected','audio','audio_sha256','pcm_sha256')}
            result.update({name:replay(engine,pcm,row['expected']) for name,engine in engines.items()})
            results.append(result);stream.write(canonical(result)+'\n');stream.flush()
            if (index+1)%100==0:print('HUMAN_REGRESSION',index+1,'/',len(rows),flush=True)
    summary=summarize(results);summary.update(complete=len(results)==len(rows),candidate_binding=selected.binding,
        protocol_sha256=file_hash(protocol_path),predictions_sha256=file_hash(output/'predictions.jsonl'),
        ort_provider='CPUExecutionProvider',threads=1,wake_measured=False,physical_pi_measured=False)
    write_json(output/'results.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(evaluate(args.candidate,args.output),indent=2))
