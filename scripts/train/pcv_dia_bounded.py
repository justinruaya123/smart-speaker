"""Duration-derived Dia generation bound, preserving completed corpus evidence."""
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_dia_cpu_codec import cpu_codec_loader
from scripts.train.pcv_three_engine import HERE,OUTPUT,immutable,verify,generated_row
from scripts.train.pcv_public_data import sha,read_rows,save
from scripts.test.pcv_public_qualification import verify_bindings
from scripts.runtime.pcv_schema import canonical

WORK=OUTPUT/'dia-bounded-v1'
PROTOCOL=HERE/'dia-bounded-protocol.json'
CAP=math.ceil(12*44100/512)+16


def freeze():
    verify()
    prior=read_rows(OUTPUT/'dia-generated.jsonl')+read_rows(OUTPUT/'dia-cpu-codec-v1/dia-single-generated.jsonl')
    failures=read_rows(OUTPUT/'dia-cpu-codec-v1/dia-single-failed-attempts.jsonl')
    seen={r['phrase_id'] for r in prior}
    attempts={}
    for r in failures:
        if r['id'] not in seen:attempts[r['id']]=max(attempts.get(r['id'],0),r['attempt'])
    for row in prior:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['native_audio']:row['native_sha256'],row['features']:row['features_sha256']})
    for kind in ('audio','native'):
        if {p.stem for p in (OUTPUT/kind/'dia').glob('*.wav')}!=seen:raise ValueError('Orphan Dia audio')
    immutable(HERE/'dia-before-bounded.json',prior)
    paths=[Path(__file__),HERE/'dia-before-bounded.json',HERE/'dia-cpu-codec-protocol.json',HERE/'dia-cpu-output-protocol.json',
        OUTPUT/'dia-cpu-run.log',OUTPUT/'dia-cpu-codec-v1/dia-single-failed-attempts.jsonl',
        ROOT/'scripts/train/pcv_dia_cpu_codec.py',ROOT/'experiments/pcv-synthetic-v1/synthesis.py',
        ROOT/'.cache/pcv-synthetic-v1/models/dac_44khz/config.json']
    immutable(PROTOCOL,{'version':'dia-bounded-v1','purpose':'Reject generations that cannot satisfy the existing 12-second maximum sooner',
        'retained':len(prior),'resume_attempt_counts':attempts,'interrupted_owned_process_ids':[12240,3948],
        'max_new_tokens':CAP,'prior_max_new_tokens':1600,'max_audio_seconds':12,'sample_rate':44100,'codec_hop':512,
        'maximum_valid_audio_frames':math.floor(12*44100/512),'delay_and_stop_allowance':16,
        'derivation':'ceil(12 * 44100 / 512) + 16 = 1050; waveform duration remains checked independently',
        'same_valid_output_requirement':'Two predetermined development probes must produce identical native and derived PCM at 1600 and 1050 tokens',
        'unchanged':['FP32 acoustic and codec weights','CUDA acoustic model and CPU codec','Single-item seeds','Twenty total attempts per phrase','All 415 texts and expected labels','All waveform/CTC gates'],
        'test_inference_accessed':False,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths}})


@contextmanager
def loader():
    with cpu_codec_loader('dia') as (generator,captured,module):
        original=inspect.getclosurevars(generator).nonlocals['original']
        model=inspect.getclosurevars(original).nonlocals['model']
        generate=model.generate
        settings={'cap':CAP}
        def bounded_generate(*args,**kwargs):
            kwargs['max_new_tokens']=settings['cap']
            return generate(*args,**kwargs)
        model.generate=bounded_generate
        try:yield generator,captured,settings
        finally:model.generate=generate


def check():
    import torch
    verify(assets=True)
    p=json.loads(PROTOCOL.read_text());verify_bindings(ROOT,p['bindings'])
    torch.set_num_threads(2)
    rows=[]
    with loader() as (generator,_,settings):
        from synthesis import VOICES,prepare_audio
        voice=next(v for v in VOICES if v['id']=='dev-au-m')
        for i,text in enumerate(['The system is ready.','Please set the brightness to sixty percent.']):
            results=[]
            for cap in (1600,CAP):
                settings['cap']=cap;gc.collect();torch.cuda.empty_cache();torch.manual_seed(91237+i*10)
                began=time.perf_counter();waves,rate=generator([text],voice);value=prepare_audio(waves[0],rate)
                results.append({'cap':cap,'native_sha256':hashlib.sha256(value[0]).hexdigest(),
                    'derived_sha256':hashlib.sha256(value[1]).hexdigest(),'pcm_sha256':value[4],
                    'duration_seconds':value[3]['duration_seconds'],'seconds':time.perf_counter()-began})
            same=all(results[0][key]==results[1][key] for key in ('native_sha256','derived_sha256','pcm_sha256'))
            rows.append({'text':text,'same':same,'conditions':results})
            print('DIA_BOUND_DEVELOPMENT',i+1,'identical',same,flush=True)
    immutable(HERE/'dia-bounded-preflight.json',{'pass':all(r['same'] for r in rows),'rows':rows,
        'protocol_sha256':sha(PROTOCOL),'scope':'Two development voice generations; no VCM inference; not proof for every random sample'})


def generate():
    import numpy as np
    import torch
    verify(assets=True)
    p=json.loads(PROTOCOL.read_text());verify_bindings(ROOT,p['bindings'])
    proof=json.loads((HERE/'dia-bounded-preflight.json').read_text())
    if not proof['pass'] or proof['protocol_sha256']!=sha(PROTOCOL):raise ValueError('Duration-bound preflight failed')
    if (HERE/'dia-generation-complete.json').exists():return
    prior=json.loads((HERE/'dia-before-bounded.json').read_text())
    WORK.mkdir(parents=True,exist_ok=True);path=WORK/'generated.jsonl'
    existing=read_rows(path) if path.exists() else []
    for row in existing:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['native_audio']:row['native_sha256'],row['features']:row['features_sha256']})
    seen={r['phrase_id'] for r in prior+existing}
    prompts=json.loads((HERE/'prompts.json').read_text())
    torch.set_num_threads(2);began=time.perf_counter()
    with loader() as (generator,assets,_):
        from synthesis import VOICES,prepare_audio
        immutable(WORK/'model-identity.json',assets['assets'])
        voice=next(v for v in VOICES if v['id']=='test-gb-m')
        for prompt in prompts:
            if prompt['id'] in seen:continue
            seed=int(hashlib.sha256(('pcv-b1-three-engine-v1/dia/'+prompt['id']).encode()).hexdigest()[:8],16)
            prepared=None;tick=time.perf_counter()
            for attempt in range(p['resume_attempt_counts'].get(prompt['id'],0),20):
                gc.collect();torch.cuda.empty_cache();torch.manual_seed((seed+attempt)%2**32)
                try:
                    waves,rate=generator([prompt.get('spoken_text',prompt['text'])],voice)
                    if len(waves)!=1:raise ValueError('Wrong waveform count')
                    prepared=generated_row('dia',prompt,prepare_audio(waves[0],rate),{
                        'generation_seed':(seed+attempt)%2**32,'attempt':attempt+1,'batch_phrase_ids':[prompt['id']],
                        'generation_metadata':{**generator.last_metadata[0],'max_new_tokens':CAP,'generation_revision_sha256':sha(PROTOCOL)},
                        'protocol_sha256':sha(PROTOCOL),'source_version':'dia-bounded-v1'})
                    break
                except ValueError as exc:
                    with (WORK/'failed-attempts.jsonl').open('a',encoding='utf8') as stream:
                        stream.write(canonical({'id':prompt['id'],'attempt':attempt+1,'seed':(seed+attempt)%2**32,'error':str(exc)})+'\n')
                    print('DIA_BOUND_RETRY',prompt['id'],attempt+1,str(exc),flush=True)
            if prepared is None:
                immutable(WORK/'failures'/(prompt['id']+'.json'),{'id':prompt['id'],'reason':'Twenty total attempts exhausted'})
                continue
            row,payloads,f=prepared;feature=WORK/'features'/(prompt['id']+'.npy')
            for target in [t for t,_ in payloads]+[feature]:
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():raise FileExistsError('Preserve orphan output: '+str(target))
            for target,payload in payloads:
                with target.open('xb') as stream:stream.write(payload)
            with feature.open('xb') as stream:np.save(stream,f)
            row.update(native_audio=str(payloads[0][0]),native_sha256=sha(payloads[0][0]),audio=str(payloads[1][0]),
                audio_sha256=sha(payloads[1][0]),features=str(feature),features_sha256=sha(feature))
            with path.open('a',encoding='utf8') as stream:stream.write(canonical(row)+'\n');stream.flush()
            existing.append(row);seen.add(prompt['id'])
            save(WORK/'status.json',{'complete':len(seen),'planned':415,'elapsed_seconds':time.perf_counter()-began,
                'last_phrase_seconds':time.perf_counter()-tick,'peak_cuda_reserved':torch.cuda.max_memory_reserved()})
            print('DIA_BOUND',len(seen),'/415','seconds',round(time.perf_counter()-tick,2),flush=True)
    verify(assets=True);verify_bindings(ROOT,p['bindings'])
    combined=WORK/'test-rows.jsonl'
    with combined.open('x',encoding='utf8') as stream:
        for row in prior+existing:stream.write(canonical(row)+'\n')
    immutable(HERE/'dia-generation-complete.json',{'engine':'dia','generated':len(seen),'reused':0,'retained_previous':len(prior),
        'missing':sorted({r['id'] for r in prompts}-seen),'manifest':str(combined.relative_to(ROOT)),'manifest_sha256':sha(combined),
        'seconds_bounded_continuation':time.perf_counter()-began,'protocol_sha256':sha(HERE/'protocol.json'),
        'generation_revision_sha256':sha(PROTOCOL),'text_listener_verified':False})


if __name__=='__main__':
    if sys.argv[1:] not in (['freeze'],['check'],['generate']):raise SystemExit('Use freeze, check or generate')
    globals()[sys.argv[1]]()
