"""Versioned single-item Dia continuation after GPU-memory contention."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_three_engine import HERE,OUTPUT,VOICE,immutable,verify,engine_loader,generated_row
from scripts.train.pcv_public_data import read_rows,sha,save
from scripts.test.pcv_public_qualification import verify_bindings
from scripts.runtime.pcv_schema import canonical


def freeze():
    verify()
    prior=OUTPUT/'dia-generated.jsonl'
    rows=read_rows(prior)
    for row in rows:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['features']:row['features_sha256'],row['native_audio']:row['native_sha256']})
    known={r['phrase_id'] for r in rows}
    for kind,extension in [('audio','.wav'),('native','.wav'),('features','.npy')]:
        if {p.stem for p in (OUTPUT/kind/'dia').glob('*'+extension)}!=known:
            raise ValueError('Orphan Dia output needs separate investigation')
    paths=[Path(__file__),HERE/'protocol.json',prior,OUTPUT/'dia-failed-attempts.jsonl',OUTPUT/'dia-run.log']
    immutable(HERE/'dia-single-protocol.json',{'version':'dia-single-v1','parent_protocol_sha256':sha(HERE/'protocol.json'),
        'reason':'Two-item attempts pushed shared GPU memory to 15738/16376 MiB and stalled progress; only owned Dia processes were stopped',
        'observed_memory_mib':15738,'observed_total_memory_mib':16376,'interrupted_process_ids':[34392,34320],
        'retained_completed':len(rows),'completed_recording_ids':[r['id'] for r in rows],
        'batch_size':1,'dtype':'float32','max_new_tokens':1600,'temperature':1.8,'attempts':20,
        'seed_rule':'SHA256(pcv-b1-three-engine-v1/dia/phrase_id) first 8 hex plus zero-based retry modulo 2**32',
        'runtime':'Collect unused objects and release CUDA allocator cache before each attempt',
        'changed':['Batch size for remaining phrases; per-phrase rather than first-pair-member seed'],
        'unchanged':['All 415 phrases and labels','Voice anchor','Model/codec tensors and precision','Waveform and token-cap gates','VCM models and policies'],
        'test_inference_accessed':False,'other_job_interrupted':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


def generate():
    import numpy as np
    import torch
    verify(assets=True)
    protocol=json.loads((HERE/'dia-single-protocol.json').read_text())
    verify_bindings(ROOT,protocol['bindings'])
    if (HERE/'dia-generation-complete.json').exists():return
    prior=read_rows(OUTPUT/'dia-generated.jsonl')
    path=OUTPUT/'dia-single-generated.jsonl'
    existing=read_rows(path) if path.exists() else []
    for row in existing:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['features']:row['features_sha256'],row['native_audio']:row['native_sha256']})
    seen={r['phrase_id'] for r in prior+existing}
    prompts=json.loads((HERE/'prompts.json').read_text())
    torch.set_num_threads(2)
    began=time.perf_counter()
    with engine_loader('dia') as (generator,captured,_):
        from synthesis import prepare_audio,VOICES
        immutable(OUTPUT/'dia-single-model-identity.json',captured['assets'])
        voice=next(v for v in VOICES if v['id']==VOICE)
        for prompt in prompts:
            if prompt['id'] in seen:continue
            seed=int(hashlib.sha256(('pcv-b1-three-engine-v1/dia/'+prompt['id']).encode()).hexdigest()[:8],16)
            prepared=None;tick=time.perf_counter()
            for attempt in range(20):
                gc.collect();torch.cuda.empty_cache();torch.manual_seed((seed+attempt)%2**32)
                try:
                    waves,rate=generator([prompt.get('spoken_text',prompt['text'])],voice)
                    if len(waves)!=1:raise ValueError('Wrong waveform count')
                    prepared=generated_row('dia',prompt,prepare_audio(waves[0],rate),{
                        'generation_seed':(seed+attempt)%2**32,'attempt':attempt+1,'batch_phrase_ids':[prompt['id']],
                        'generation_metadata':generator.last_metadata[0],
                        'protocol_sha256':sha(HERE/'dia-single-protocol.json'),'source_version':'dia-single-v1'})
                    break
                except ValueError as exc:
                    with (OUTPUT/'dia-single-failed-attempts.jsonl').open('a',encoding='utf8') as stream:
                        stream.write(canonical({'id':prompt['id'],'attempt':attempt+1,'seed':(seed+attempt)%2**32,'error':str(exc)})+'\n')
                    print('DIA_SINGLE_RETRY',prompt['id'],attempt+1,str(exc),flush=True)
            if prepared is None:
                immutable(OUTPUT/'dia-single-failures'/(prompt['id']+'.json'),{'id':prompt['id'],'reason':'twenty waveform attempts exhausted'})
                continue
            row,payloads,f=prepared
            feature=OUTPUT/'features/dia'/(prompt['id']+'.npy')
            for target in [p for p,_ in payloads]+[feature]:
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise FileExistsError('Preserve orphan output: '+str(target))
            for target,payload in payloads:
                with target.open('xb') as stream:stream.write(payload)
            with feature.open('xb') as stream:np.save(stream,f)
            row.update(native_audio=str(payloads[0][0]),native_sha256=sha(payloads[0][0]),
                audio=str(payloads[1][0]),audio_sha256=sha(payloads[1][0]),features=str(feature),features_sha256=sha(feature))
            with path.open('a',encoding='utf8') as stream:stream.write(canonical(row)+'\n');stream.flush()
            existing.append(row);seen.add(prompt['id'])
            save(OUTPUT/'dia-single-status.json',{'complete':len(seen),'planned':415,'elapsed_seconds':time.perf_counter()-began,
                'last_phrase_seconds':time.perf_counter()-tick,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved':torch.cuda.max_memory_reserved()})
            print('DIA_SINGLE',len(seen),'/415','seconds',round(time.perf_counter()-tick,2),flush=True)
    verify(assets=True);verify_bindings(ROOT,protocol['bindings'])
    combined=OUTPUT/'dia-combined.jsonl'
    with combined.open('x',encoding='utf8') as stream:
        for row in prior+existing:stream.write(canonical(row)+'\n')
    immutable(HERE/'dia-generation-complete.json',{'engine':'dia','generated':len(seen),'reused':0,'retained_previous_batch2':len(prior),
        'missing':sorted({p['id'] for p in prompts}-seen),'manifest':str(combined.relative_to(ROOT)),
        'manifest_sha256':sha(combined),'seconds_single_continuation':time.perf_counter()-began,
        'protocol_sha256':sha(HERE/'protocol.json'),'generation_revision_sha256':sha(HERE/'dia-single-protocol.json'),
        'text_listener_verified':False})


if __name__=='__main__':
    if sys.argv[1:]==['freeze']:freeze()
    elif sys.argv[1:]==['generate']:generate()
    else:raise SystemExit('Use freeze or generate')
