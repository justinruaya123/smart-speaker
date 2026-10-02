"""Frozen test-only synthesis and evaluation; never changes trained candidates."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

from scripts.train.pcv_public_data import ROOT, CACHE, STUDY, sha, save, read_rows, legacy_command
from scripts.train.pcv_calibration import CHILD, WORK
from scripts.runtime.pcv_schema import INTENTS, canonical, normalize_text
from scripts.test.pcv_public_qualification import verify_parent, verify_bindings
from vcm.frontend import features

HERE = ROOT/'experiments/pcv-b1-three-engine-v1'
OUTPUT = ROOT/'.cache/pcv-b1-three-engine-v1'
ENGINES = ('qwen', 'dia', 'parler')
VOICE = 'test-gb-m'


def immutable(path, value):
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    if path.exists():
        if path.read_text(encoding='utf8') != payload:
            raise FileExistsError('Preserve immutable receipt: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(payload)


def exact_reuse(prompt, row):
    return (row['split']=='test' and row['speaker']=='qwen/'+VOICE
        and normalize_text(prompt.get('spoken_text',prompt['text']))==normalize_text(row['text'])
        and canonical(prompt['command'])==canonical(row['expected']))


def audit_records(prompts, engines, rows, non_test):
    cells = {e+'/'+p['id']:(e,p) for e in engines for p in prompts}
    seen, pcm = set(), set()
    blocked_pcm = {r['pcm_sha256'] for r in non_test}
    blocked_speakers = {r['speaker'] for r in non_test}
    for row in rows:
        key = row['engine']+'/'+row['phrase_id']
        if key not in cells or key in seen or row['id']!=key:
            raise ValueError('Unexpected or duplicate test cell')
        e,p = cells[key]
        if (normalize_text(row['text'])!=normalize_text(p.get('spoken_text',p['text']))
            or canonical(row['expected'])!=canonical(p['command'])):
            raise ValueError('Test label differs from frozen prompt')
        if row['split']!='test' or row['speaker']!=e+'/'+VOICE:
            raise ValueError('Unreserved test condition')
        if row['pcm_sha256'] in pcm:
            raise ValueError('Within-test duplicate PCM')
        if row['pcm_sha256'] in blocked_pcm or row['speaker'] in blocked_speakers:
            raise ValueError('Test/non-test overlap')
        seen.add(key)
        pcm.add(row['pcm_sha256'])
    return {'planned':len(cells),'present':len(seen),'complete':len(seen)==len(cells),
        'missing':sorted(set(cells)-seen),'per_engine':dict(Counter(r['engine'] for r in rows)),
        'cross_split_pcm_overlap':0,'cross_split_speaker_overlap':0,'within_test_pcm_duplicates':0}


def freeze():
    if (HERE/'protocol.json').exists():
        return verify()
    verify_parent()
    prompts = json.loads((CHILD/'calibration-prompts.json').read_text())
    assert len(prompts)==415 and sum(p['command'] is not None for p in prompts)==401
    parent = read_rows(CACHE/'prepared/test.jsonl')
    by_audio = {r['audio_sha256']:r for r in parent if r['speaker']=='qwen/'+VOICE}
    old = read_rows(ROOT/'.cache/pcv-synthetic-v2-repair1/qwen-manifest.jsonl')
    verbs = {r['id']:r for r in read_rows(ROOT/'experiments/pcv-synthetic-v2/verbatims.jsonl')}
    reuse = []
    for prompt in prompts:
        for candidate in old:
            if candidate['voice_group']!=VOICE or candidate['split']!='test':
                continue
            v = verbs[candidate['verbatim_id']]
            r = by_audio.get(candidate['audio_sha256'])
            if (r is None or not exact_reuse(prompt,r)
                or canonical(legacy_command(v.get('expected_command'),v['text']))!=canonical(prompt['command'])):
                continue
            reuse.append({**r, 'id':'qwen/'+prompt['id'], 'phrase_id':prompt['id'], 'engine':'qwen',
                'source':'qwen', 'reused_parent_id':r['id'], 'reused_original_recording':candidate['recording_id'],
                'native_audio':candidate['native_audio'], 'native_sha256':candidate['native_sha256'],
                'generation_metadata':candidate.get('generation_metadata',{}), 'prompt_listener_verified':False})
            break
    immutable(HERE/'prompts.json',prompts)
    immutable(HERE/'reused-qwen.json',reuse)
    paths = [Path(__file__), ROOT/'scripts/test/test_pcv_three_engine.py', HERE/'run.py',
        CHILD/'comparison.json', CHILD/'protocol.json', CHILD/'catalog.json', CHILD/'custom-slots.json',
        STUDY/'baseline.json', STUDY/'selection-before-test.json',
        CHILD/'calibration-prompts.json', HERE/'prompts.json', HERE/'reused-qwen.json',
        ROOT/'scripts/test/pcv_public_qualification.py',ROOT/'scripts/test/pcv_public_eval.py',
        ROOT/'scripts/pcv_parallel.py', ROOT/'vcm/frontend.py',
        ROOT/'experiments/pcv-synthetic-v1/synthesis.py', ROOT/'experiments/pcv-synthetic-v1/study.py',
        ROOT/'experiments/pcv-synthetic-v3/generation.py',ROOT/'scripts/pcv_parler_weights.py',
        ROOT/'.cache/pcv-synthetic-v1/anchors'/f'{VOICE}.wav',
        ROOT/'.cache/pcv-synthetic-v1/anchors'/f'{VOICE}.json',
        ROOT/'.cache/pcv-synthetic-v2-repair1/qwen-manifest.jsonl',
        ROOT/'experiments/pcv-synthetic-v2/verbatims.jsonl',CACHE/'prepared/test.jsonl']
    for arm in ('human','mixed'):
        paths.extend(p for p in (WORK/'evaluation'/arm/'bundle').iterdir() if p.is_file())
        paths.append(WORK/'evaluation'/arm/'selection-before-test.json')
    folders = [ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-Base',
        *[ROOT/'.cache/pcv-synthetic-v1/models'/name for name in ('Dia-1.6B-0626','dac_44khz','parler-tts-large-v1')]]
    assets = [p for folder in folders for p in folder.rglob('*') if p.is_file()
        and '.cache' not in p.relative_to(folder).parts and p.suffix in {'.safetensors','.json','.txt','.model'}]
    print('FREEZING_ASSETS',len(assets),flush=True)
    result = {'version':'pcv-b1-three-engine-v1','engines':list(ENGINES),'third_engine':'Parler working assumption stated to user',
        'purpose':'All 415 phrase cells on three synthetic engines; unchanged trained models and policies',
        'requirements':['ME-03','ME-04','ME-06','ME-07','ME-08','UC-01..UC-10'],
        'owner':'PCV schema/public-speech task','baseline':'pcv-human-slots-v1-qualification1',
        'allowed_outputs':['experiments/pcv-b1-three-engine-v1','.cache/pcv-b1-three-engine-v1'],
        'voice':VOICE,'per_engine':415,'planned':1245,'positives_per_engine':401,'negatives_per_engine':14,
        'baseline_per_engine':93,'reuse_qwen':len(reuse),'training_or_threshold_changes':False,
        'generation':{'qwen':{'batch_size':2,'attempts':3,'dtype':'bfloat16','max_new_tokens':256,'temperature':.75,'top_p':.95},
            'dia':{'batch_size':2,'attempts':20,'dtype':'float32','max_new_tokens':1600,'temperature':1.8,'reject_token_cap':True},
            'parler':{'batch_size':4,'attempts':3,'dtype':'bfloat16','max_new_tokens':1400,'temperature':.9,'verify_all_tensors':True}},
        'seed_rule':'SHA256(pcv-b1-three-engine-v1/engine/first_phrase_id) first 8 hex plus zero-based retry, modulo 2**32',
        'generation_filter':'Existing native/derived waveform gates, <=12 seconds and sufficient finite CTC frames only; never model correctness',
        'failure_rule':'Keep failed batch receipt and missing cells after maximum attempts; no fallback substitution',
        'inference':'Complete-utterance frozen features; actual FP32 CPU ONNX, 1 thread, 12 exact decoder workers',
        'selection':'Fixed development winner mixed; no test-selected thresholds/weights',
        'text_listener_verified':False,'accent_verified':False,'synthetic_conditions_are_human_speakers':False,
        'installed_artifacts_changed':False,'test_inference_accessed':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'tts_assets':{str(p.relative_to(ROOT)):sha(p) for p in assets}}
    immutable(HERE/'protocol.json',result)
    return result


def verify(assets=False):
    p = json.loads((HERE/'protocol.json').read_text())
    verify_bindings(ROOT,p['bindings'])
    if assets:
        verify_bindings(ROOT,p['tts_assets'])
    verify_parent()
    return p


@contextmanager
def engine_loader(engine):
    sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v1'))
    if engine=='parler':
        spec = importlib.util.spec_from_file_location('pcv_verified_parler',ROOT/'experiments/pcv-synthetic-v3/generation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with module.verified_loader() as (generator,captured):
            yield generator,captured,module
    else:
        from synthesis import load_engine
        generator,assets = load_engine(engine)
        yield generator,{'assets':assets},None


def generated_row(engine, prompt, value, metadata):
    from synthesis import audit_waveform
    native,derived,native_metrics,metrics,pcm_hash = value
    catalog = json.loads((CHILD/'catalog.json').read_text())
    entry = next(e for e in catalog['entries'] if e['text']==normalize_text(prompt.get('spoken_text',prompt['text']))
        and canonical(e['command'])==canonical(prompt['command']))
    pcm,rate = sf.read(io.BytesIO(derived),dtype='float32')
    assert rate==16000
    audit_waveform(pcm,rate)
    f = features(pcm)
    phones = entry['phones']
    if len(pcm)>12*16000 or f.shape[1]<len(phones)+sum(a==b for a,b in zip(phones,phones[1:])):
        raise ValueError('Generated utterance violates endpoint/finite CTC capacity')
    c = prompt['command']
    row = {'id':engine+'/'+prompt['id'],'phrase_id':prompt['id'],'engine':engine,'source':engine,
        'speaker':engine+'/'+VOICE,'split':'test','synthetic':True,'text':entry['text'],'expected':c,
        'pcm_sha256':pcm_hash,'phones':phones,'frames':f.shape[1],'native_metrics':native_metrics,**metrics,
        'intent_index':INTENTS.index(c['intent'])+1 if c else 0,
        'target_index':int(c['slots']['light_id']!='light.all') if c and 'light_id' in c['slots'] else -100,
        'prompt_listener_verified':False,'accent_verified':False,**metadata}
    return row,[(OUTPUT/'native'/engine/(prompt['id']+'.wav'),native),
                (OUTPUT/'audio'/engine/(prompt['id']+'.wav'),derived)],f


def generate(engine):
    import torch
    protocol = verify(assets=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    prompts = json.loads((HERE/'prompts.json').read_text())
    reused = json.loads((HERE/'reused-qwen.json').read_text()) if engine=='qwen' else []
    path = OUTPUT/(engine+'-generated.jsonl')
    existing = read_rows(path) if path.exists() else []
    for r in existing+reused:
        verify_bindings(ROOT,{r[k]:r[k+'_sha256'] for k in ('audio','features')})
        verify_bindings(ROOT,{r['native_audio']:r['native_sha256']})
    if (HERE/(engine+'-generation-complete.json')).exists():
        return
    all_ids = {r['phrase_id'] for r in existing}
    reuse_ids = {r['phrase_id'] for r in reused}
    planned = [p for p in prompts if p['id'] not in reuse_ids]
    recipe = protocol['generation'][engine]
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Declared generator requires local CUDA')
    begun = time.perf_counter()
    with engine_loader(engine) as (generator,captured,parler):
        from synthesis import prepare_audio, VOICES
        voice = next(v for v in VOICES if v['id']==VOICE)
        if parler:
            immutable(OUTPUT/'parler-weights-before.json',captured['verification'])
        else:
            immutable(OUTPUT/(engine+'-model-identity.json'),captured['assets'])
        first = True
        for offset in range(0,len(planned),recipe['batch_size']):
            batch = planned[offset:offset+recipe['batch_size']]
            if all(p['id'] in all_ids for p in batch):
                continue
            seed = int(hashlib.sha256(('pcv-b1-three-engine-v1/'+engine+'/'+batch[0]['id']).encode()).hexdigest()[:8],16)
            prepared = None
            tick = time.perf_counter()
            for attempt in range(recipe['attempts']):
                torch.manual_seed((seed+attempt)%2**32)
                try:
                    waves,rate = generator([p.get('spoken_text',p['text']) for p in batch],voice)
                    if len(waves)!=len(batch):
                        raise ValueError('Generator returned wrong batch size')
                    values = [prepare_audio(w,rate) for w in waves]
                    gm = getattr(generator,'last_metadata',[{} for _ in batch])
                    prepared = [generated_row(engine,p,value,{'generation_seed':(seed+attempt)%2**32,
                        'attempt':attempt+1,'batch_phrase_ids':[b['id'] for b in batch],
                        'generation_metadata':meta,'protocol_sha256':sha(HERE/'protocol.json')})
                        for p,value,meta in zip(batch,values,gm,strict=True)]
                    break
                except ValueError as exc:
                    with (OUTPUT/(engine+'-failed-attempts.jsonl')).open('a',encoding='utf8') as stream:
                        stream.write(canonical({'ids':[p['id'] for p in batch],'attempt':attempt+1,
                            'seed':(seed+attempt)%2**32,'error':str(exc)})+'\n')
                    print('GENERATION_RETRY',engine,offset,attempt+1,str(exc),flush=True)
            if parler and first:
                checked = parler.repair_and_verify(captured['model'],parler.MODEL)
                immutable(OUTPUT/'parler-weights-after-first.json',checked)
                if checked['changed_before'] or not checked['verified_after']:
                    raise ValueError('Parler weights drifted during generation')
            first = False
            if prepared is None:
                immutable(OUTPUT/'failed-batches'/engine/(batch[0]['id']+'.json'),{'ids':[p['id'] for p in batch],'reason':'waveform_attempts_exhausted'})
                continue
            for row,payloads,f in prepared:
                if row['phrase_id'] in all_ids:
                    continue
                feature_path = OUTPUT/'features'/engine/(row['phrase_id']+'.npy')
                for target in [p for p,_ in payloads]+[feature_path]:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():
                        raise FileExistsError('Preserve orphan generation file: '+str(target))
                for target,payload in payloads:
                    with target.open('xb') as stream:
                        stream.write(payload)
                with feature_path.open('xb') as stream:
                    np.save(stream,f)
                row.update(native_audio=str(payloads[0][0]),native_sha256=sha(payloads[0][0]),
                    audio=str(payloads[1][0]),audio_sha256=sha(payloads[1][0]),
                    features=str(feature_path),features_sha256=sha(feature_path))
                with path.open('a',encoding='utf8') as stream:
                    stream.write(canonical(row)+'\n');stream.flush()
                existing.append(row)
                all_ids.add(row['phrase_id'])
            save(OUTPUT/(engine+'-status.json'),{'engine':engine,'complete':len(existing)+len(reused),
                'planned':415,'batch_seconds':time.perf_counter()-tick,'elapsed_seconds':time.perf_counter()-begun})
            print('GENERATED',engine,len(existing)+len(reused),'/415','seconds',round(time.perf_counter()-tick,2),flush=True)
        if parler:
            checked = parler.repair_and_verify(captured['model'],parler.MODEL)
            immutable(OUTPUT/'parler-weights-final.json',checked)
            if checked['changed_before'] or not checked['verified_after']:
                raise ValueError('Parler weights drifted during generation')
    verify(assets=True)
    immutable(HERE/(engine+'-generation-complete.json'),{'engine':engine,'generated':len(existing),'reused':len(reused),
        'missing':sorted({p['id'] for p in prompts}-all_ids-reuse_ids),'manifest':str(path.relative_to(ROOT)),
        'manifest_sha256':sha(path) if path.exists() else None,'seconds':time.perf_counter()-begun,
        'protocol_sha256':sha(HERE/'protocol.json'),'text_listener_verified':False})


def assemble():
    verify(assets=True)
    prompts = json.loads((HERE/'prompts.json').read_text())
    rows = json.loads((HERE/'reused-qwen.json').read_text())
    for engine in ENGINES:
        receipt = json.loads((HERE/(engine+'-generation-complete.json')).read_text())
        if receipt['manifest_sha256']:
            verify_bindings(ROOT,{receipt['manifest']:receipt['manifest_sha256']})
            rows += read_rows(ROOT/receipt['manifest'])
    rows.sort(key=lambda r:(ENGINES.index(r['engine']),r['phrase_id']))
    non_test = [r for split in ('train','dev','calibration') for r in read_rows(CACHE/'prepared'/f'{split}.jsonl')]
    non_test += read_rows(WORK/'generated.jsonl')
    audit = audit_records(prompts,ENGINES,rows,non_test)
    for row in rows:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['native_audio']:row['native_sha256'],row['features']:row['features_sha256']})
        pcm,rate = sf.read(row['audio'],dtype='float32')
        if rate!=16000 or pcm.ndim!=1 or hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']:
            raise ValueError('Retained PCM identity failed')
        f = np.load(row['features'],allow_pickle=False)
        if f.shape!=(40,row['frames']) or not np.isfinite(f).all():
            raise ValueError('Invalid retained features')
    immutable(OUTPUT/'test.json',rows)
    immutable(HERE/'data-audit.json',{**audit,'manifest':str((OUTPUT/'test.json').relative_to(ROOT)),
        'sha256':sha(OUTPUT/'test.json'),'protocol_sha256':sha(HERE/'protocol.json'),
        'reused_qwen':len(json.loads((HERE/'reused-qwen.json').read_text())),
        'non_test_overlap_records_checked':len(non_test),'text_listener_verified':False})
    print('TEST_AUDIT',json.dumps(audit),flush=True)


def evaluate():
    import onnxruntime as ort
    from scripts.test.pcv_public_eval import score
    from scripts.test.pcv_public_qualification import accepted_candidate,decision,precise_metrics
    verify()
    audit = json.loads((HERE/'data-audit.json').read_text())
    verify_bindings(ROOT,{audit['manifest']:audit['sha256']})
    rows = json.loads((ROOT/audit['manifest']).read_text())
    prompts = {p['id']:p for p in json.loads((HERE/'prompts.json').read_text())}
    for row in rows:
        verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['features']:row['features_sha256']})
    immutable(HERE/'before-inference.json',{'protocol_sha256':sha(HERE/'protocol.json'),
        'data_audit_sha256':sha(HERE/'data-audit.json'),'rows':len(rows),'test_inference_accessed':False})
    for arm in ('human','mixed'):
        if (HERE/('results-'+arm+'.json')).exists():
            raise FileExistsError('Preserve completed evaluation')
        bundle = WORK/'evaluation'/arm/'bundle'
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle/'model.onnx'),options,providers=['CPUExecutionProvider'])
        norm = json.loads((bundle/'normalization.json').read_text())
        entries = json.loads((bundle/'catalog.json').read_text())['entries']
        policy = json.loads((bundle/'policy.json').read_text())
        predicted = score(session,rows,norm,entries,workers=12)
        for r in predicted:
            r['policy_candidate'] = accepted_candidate(r,policy)
            r['output'] = decision(r,policy)
            r['accepted_command'] = r['output']['command']
        immutable(OUTPUT/('predictions-'+arm+'.json'),predicted)
        result = {'arm':arm,'development_selected':arm=='mixed','test_used_for_selection':False,
            'qualified':False,'policy_enabled':policy['enabled'],'data_audit':audit,'engines':{},
            'predictions_sha256':sha(OUTPUT/('predictions-'+arm+'.json')),'cells':[]}
        for engine in ENGINES:
            subset = [r for r in predicted if r['source']==engine]
            baseline = [r for r in subset if r['phrase_id'].startswith('B1-')]
            result['engines'][engine] = {'raw':precise_metrics(subset),'baseline_raw':precise_metrics(baseline),
                'diagnostic_policy_point':precise_metrics(subset,'policy_candidate'),
                'actual_outputs':precise_metrics(subset,'accepted_command'),
                'missing':[k for k in audit['missing'] if k.startswith(engine+'/')],
                'required_baseline':93,'required_positive':401,'required_negative':14,'slots':{}}
            for r in subset:
                if not r['expected']:
                    continue
                for role,value in r['expected']['slots'].items():
                    counts = result['engines'][engine]['slots'].setdefault(r['expected']['intent']+'/'+role,
                        {'n':0,'raw_correct':0,'accepted_correct':0})
                    counts['n'] += 1
                    for field,key in [('candidate','raw_correct'),('accepted_command','accepted_correct')]:
                        actual = r[field]
                        counts[key] += int(actual is not None and actual['intent']==r['expected']['intent']
                            and canonical(actual['slots'].get(role))==canonical(value))
            by_phrase = {r['phrase_id']:r for r in subset}
            for pid,prompt in prompts.items():
                row = by_phrase.get(pid)
                result['cells'].append({'engine':engine,'phrase_id':pid,'text':prompt.get('spoken_text',prompt['text']),
                    'expected':prompt['command'],'present':row is not None,
                    'raw_exact':row is not None and canonical(row['candidate'])==canonical(prompt['command']),
                    'candidate':row['candidate'] if row else None,
                    'accepted_command':row['accepted_command'] if row else None})
        immutable(HERE/('results-'+arm+'.json'),result)
        print('THREE_ENGINE_RESULT',arm,json.dumps({e:r['baseline_raw']['overall'] for e,r in result['engines'].items()}),flush=True)
    verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','generate','assemble','evaluate'])
    parser.add_argument('--engine',choices=ENGINES)
    args = parser.parse_args()
    if args.stage=='generate':
        if args.engine is None:
            parser.error('--engine is required')
        generate(args.engine)
    else:
        globals()[args.stage]()


if __name__=='__main__':
    main()
