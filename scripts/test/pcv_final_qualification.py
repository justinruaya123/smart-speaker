"""Common-catalog qualification of source-only PCV candidates."""
import json
from pathlib import Path
import sys
import shutil
import time
import hashlib
import io

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from einops import rearrange
from scripts.runtime.pcv_schema import compile_inventory,normalize_text,canonical
from scripts.train.pcv_public_data import sha,save,read_rows,_words
from scripts.train.pcv_three_engine import immutable
from scripts.train.pcv_teacher_data import PARENT,PARENT_CACHE
from scripts.train.pcv_updated_training import development_subset
from scripts.test.pcv_public_eval import score,normalized
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings,accepted_candidate,decision
from scripts.test.pcv_updated_eval import enrich,calibrate,measured,phrase_table
from scripts.train.pcv_context_training import score_context
from scripts.pcv_parallel import ParallelScorer

HERE=ROOT/'experiments/pcv-final-qualification-v1'
CACHE=ROOT/'.cache/pcv-final-qualification-v1'
SMALL=ROOT/'experiments/pcv-updated-context-v1'
WIDE=ROOT/'experiments/pcv-updated-context-int8-v1'


def complete_catalog(parent,prompts,words):
    def signature(row):
        return (normalize_text(row.get('spoken_text',row['text'])),canonical(row['command']))
    old={signature(r) for r in parent['entries']}
    missing=[r for r in prompts if signature(r) not in old]
    result=compile_inventory(parent['entries']+missing,words)
    new={signature(r) for r in result['entries']}
    if result['entries'][:len(parent['entries'])]!=parent['entries']:
        raise ValueError('Final catalog unexpectedly changed an existing candidate')
    return result,{'parent_entries':len(parent['entries']),'entries':len(result['entries']),
        'approved_phrases':len(prompts),'added_ids':[r['id'] for r in missing],
        'missing_after':sum(signature(r) not in new for r in prompts),
        'evidence':'finite definition coverage only; missing recordings remain untested',
        'acoustic_inventory_changed':False}


def quantization_fidelity(reference,quantized):
    return bool(quantized['human']['exact']>=reference['human']['exact']-1 and
                quantized['synthetic']['exact']>=reference['synthetic']['exact']-1 and
                quantized['human']['negative_errors']<=reference['human']['negative_errors']+1)


def strata(records):
    return {name:precise_metrics([r for r in records if r['synthetic']==(name=='synthetic')])['overall']
            for name in ('human','synthetic')}


def prepare():
    for directory in (SMALL,WIDE):
        protocol=json.loads((directory/'protocol.json').read_text());verify_bindings(ROOT,protocol['bindings'])
    data=json.loads((PARENT/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in data['manifests'].values()})
    parent=json.loads((PARENT/'catalog.json').read_text())
    prompts=json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    catalog,receipt=complete_catalog(parent,prompts,_words())
    immutable(HERE/'catalog.json',catalog);immutable(HERE/'catalog-coverage.json',receipt)
    paths=[Path(__file__),ROOT/'scripts/test/test_pcv_final_catalog.py',HERE/'catalog.json',
        PARENT/'data-freeze.json',PARENT/'catalog.json',PARENT/'custom-slots.json',
        SMALL/'protocol.json',WIDE/'protocol.json',ROOT/'scripts/models/pcv_contextual.py',
        ROOT/'scripts/train/pcv_context_training.py',ROOT/'scripts/train/pcv_context_int8.py',
        ROOT/'scripts/test/pcv_updated_eval.py',ROOT/'scripts/test/pcv_public_eval.py',
        ROOT/'scripts/test/pcv_public_qualification.py',ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json',
        ROOT/'scripts/runtime/pcv_schema.py',ROOT/'scripts/train/pcv_public_data.py',
        ROOT/'scripts/pcv_parallel.py',ROOT/'experiments/phrase-contract-v1/pcvcm/decoder.py',
        ROOT/'vcm/frontend.py',ROOT/'experiments/pcv-human-slots-v1/output.schema.json',
        PARENT_CACHE/'prepared/normalization.json',ROOT/'data/raw/cmudict/cmudict.dict',
        ROOT/'experiments/pcv-synthetic-v2/lexicon.json',
        ROOT/'scripts/train/pcv_updated_training.py',ROOT/'scripts/train/pcv_three_engine.py',
        SMALL/'selection-before-calibration.json',WIDE/'selection-before-calibration.json',
        ROOT/'experiments/pcv-updated-voice-v3/results-continued.json',
        ROOT/'.cache/pcv-updated-voice-v3/evaluation/continued/bundle/recipe.json']
    immutable(HERE/'protocol.json',{'experiment':'pcv-final-qualification-v1',
        'requirements':['ME-03','ME-04','ME-06','ME-07','ME-08'],'source_recordings':'pinned supplied HF only',
        'catalog_change':'Add the already approved call mother alias; preserve every parent candidate and the 40-output acoustic inventory.',
        'selection':'Same development subset and new common catalog for all candidates; maximize mean human/synthetic exact recall, then fewer human negative errors; ties retain the earlier candidate in parent, context80, context128 order.',
        'compression_gate':'At most one lost human exact, one lost synthetic exact, one extra human negative versus same selected FP32 checkpoint; graph below two MiB.',
        'calibration':'Unchanged precision .99, recall .90, human FAR .01 and full positive-intent human coverage gate. Missing coverage stays visible.',
        'test':'All original test/holdout rows retained; ambiguous references separate. No test-based model or threshold selection.',
        '415_cases':'All definitions present; report zero-denominator cells as untested, never passing.',
        'software_oracle':'Ideal acoustic cue logits exercise the finite contract only; never count them as recordings or model accuracy.',
        'endpoint':'Contextual candidates run once on complete-utterance features, not causal chunks. Pi and live-microphone measurements remain pending.',
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths},'installed_artifact_hashes':protocol['installed_artifact_hashes'],
        'installation_authorized':False})
    if not (HERE/'oracle.json').exists():
        oracle=compile_inventory(prompts,_words())['entries'];results=[]
        def logits():
            for row in oracle:
                x=np.full((2*len(row['phones'])+1,40),-12.,dtype='float32');x[:,0]=12.
                for i,p in enumerate(row['phones']):x[2*i+1,0]=-12.;x[2*i+1,p]=12.
                yield x
        with ParallelScorer(catalog['entries'],workers=4) as scorer:
            for row,item in zip(oracle,scorer.map(logits()),strict=True):
                results.append({'id':row['id'],'pass':canonical(item['prediction']['command'])==canonical(row['command'])})
        immutable(HERE/'oracle.json',{'n':len(results),'correct':sum(r['pass'] for r in results),'rows':results,
            'evidence':'ideal cue software oracle; no audio and no neural-model accuracy claim'})
    return catalog['entries']


def session(path):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    began=time.perf_counter();value=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    return value,(time.perf_counter()-began)*1000


def run_score(candidate,rows,arrays,norm,entries):
    engine,_=session(ROOT/candidate['graph'])
    records=(score_context(engine,rows,arrays,entries) if candidate['mode']=='endpoint_context'
             else score(engine,rows,norm,entries,workers=4,arrays=arrays))
    return enrich(records,rows)


def select_candidate(entries):
    import torch
    from scripts.models.pcv_contextual import ContextualVCM,export_onnx
    from scripts.train.pcv_context_int8 import export_quantized
    torch.set_num_threads(2)
    if (HERE/'development-selection.json').exists():
        selected=json.loads((HERE/'development-selection.json').read_text())
        verify_bindings(ROOT,{r['graph']:r['graph_sha256'] for r in selected['candidates']})
        return selected
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text())
    dev=development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    verify_bindings(ROOT,{r['features']:r['features_sha256'] for r in dev})
    arrays={r['id']:normalized(r,norm) for r in dev}
    candidates=[];parent_graph=ROOT/'.cache/pcv-updated-voice-v3/evaluation/continued/bundle/model.onnx'
    parent_report=json.loads((ROOT/'experiments/pcv-updated-voice-v3/results-continued.json').read_text())
    parent_recipe=json.loads((parent_graph.parent/'recipe.json').read_text())
    if sha(parent_graph)!=parent_report['policy']['binding']['model_sha256']:raise ValueError('Parent graph changed')
    candidates.append({'name':'parent_v3','mode':'causal_full_utterance','graph':str(parent_graph.relative_to(ROOT)),
        'graph_sha256':sha(parent_graph),'export':parent_report['export'],'selection':parent_report['selection'],
        'model_config':parent_recipe['model_config'],'eligible':True})
    for name,directory in (('context80',SMALL),('context128',WIDE)):
        selected=json.loads((directory/'selection-before-calibration.json').read_text())['selected']
        verify_bindings(ROOT,{selected['checkpoint']:selected['checkpoint_sha256']})
        payload=torch.load(ROOT/selected['checkpoint'],map_location='cpu',weights_only=True)
        model=ContextualVCM(**payload['model_config']);model.load_state_dict(payload['weights'])
        output=CACHE/'exports'/name
        if name=='context80':
            graph=output/'model.onnx';export=export_onnx(model,graph)
            if not export['pass']:raise ValueError('Context80 export failed')
            candidate={'name':name,'mode':'endpoint_context','graph':str(graph.relative_to(ROOT)),
                'graph_sha256':sha(graph),'export':export,'selection':selected,'eligible':True}
        else:
            export=export_quantized(model,output);graph=output/'model.int8.onnx'
            float_candidate={'mode':'endpoint_context','graph':str((output/'model.fp32.onnx').relative_to(ROOT))}
            fp_records=run_score(float_candidate,dev,arrays,norm,entries);fp_metrics=strata(fp_records)
            save(output/'development-fp32.json',fp_records)
            candidate={'name':name,'mode':'endpoint_context','graph':str(graph.relative_to(ROOT)),
                'graph_sha256':sha(graph),'export':export,'selection':selected,'eligible':bool(export['size_pass']),
                'fp32_development':fp_metrics}
        candidate['model_config']=payload['model_config']
        candidates.append(candidate)
    for candidate in candidates:
        records=run_score(candidate,dev,arrays,norm,entries);candidate['development']=strata(records)
        save(CACHE/(candidate['name']+'-development.json'),records)
        if candidate['name']=='context128':
            candidate['compression_fidelity_pass']=quantization_fidelity(candidate['fp32_development'],candidate['development'])
            candidate['eligible']=candidate['eligible'] and candidate['compression_fidelity_pass']
        print('CANDIDATE_DEVELOPMENT',candidate['name'],json.dumps(candidate['development']),flush=True)
    def key(index):
        c=candidates[index];m=c['development']
        return ((m['human']['recall']+m['synthetic']['recall'])/2,-m['human']['negative_errors'],-index)
    winner=max((i for i,c in enumerate(candidates) if c['eligible']),key=key)
    result={'winner':candidates[winner]['name'],'candidates':candidates,'catalog_sha256':sha(HERE/'catalog.json'),
        'development_ids':[r['id'] for r in dev],'test_inference_accessed':False,'calibration_inference_accessed':False}
    immutable(HERE/'development-selection.json',result);return result


def role_metrics(records,key):
    eligible=[r for r in records if r['reference_status'] in ('supported','unsupported')]
    roles=sorted({(r['expected']['intent'],slot) for r in eligible if r['expected'] for slot in r['expected']['slots']})
    result={}
    for intent,slot in roles:
        refs=[r for r in eligible if r['expected'] and r['expected']['intent']==intent and slot in r['expected']['slots']]
        preds=[r for r in eligible if r[key] and r[key]['intent']==intent and slot in r[key]['slots']]
        correct=sum(bool(r[key] and r[key]['intent']==intent and r[key]['slots'].get(slot)==r['expected']['slots'][slot]) for r in refs)
        result[intent+'.'+slot]={'reference_n':len(refs),'predicted_n':len(preds),'correct':correct,
            'recall':correct/len(refs) if refs else None,'precision':correct/len(preds) if preds else None}
    return result


def endpoint_check(engine,rows,norm,entries):
    import soundfile as sf
    import pyarrow.parquet as pq
    from vcm.frontend import features
    from pcvcm.decoder import PrefixCTC,rank_commands
    selected=[]
    for synthetic in (False,True):
        selected.extend(sorted([r for r in rows if r['synthetic']==synthetic and r['expected']],key=lambda r:r['id'])[:6])
    payloads={}
    for path in {r['parquet'] for r in selected}:
        wanted={r['parquet_row']:r for r in selected if r['parquet']==path};index=0
        for batch in pq.ParquetFile(ROOT/path).iter_batches(batch_size=128,columns=['audio']):
            for value in batch.to_pylist():
                if index in wanted:payloads[wanted[index]['id']]=value['audio']['bytes']
                index+=1
    decoder=PrefixCTC([r['phones'] for r in entries]);results=[];timings=[]
    mean=np.asarray(norm['mean'],dtype='float32');std=np.asarray(norm['std'],dtype='float32')
    def infer(x):
        feed={'features':rearrange(x,'c t -> 1 c t')}
        feed.update({p.name:np.zeros(p.shape,dtype='float32') for p in engine.get_inputs()[1:]})
        return engine.run(None,feed)
    for row in selected:
        payload=payloads[row['id']]
        if hashlib.sha256(payload).hexdigest()!=row['audio_sha256']:raise ValueError('Endpoint WAV identity changed')
        pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
        if rate!=16000 or hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']:raise ValueError('Endpoint PCM identity changed')
        fresh=features(pcm,mean,std);cached=normalized(row,norm)
        a=infer(fresh);b=infer(cached);feature_error=float(np.max(np.abs(fresh-cached)))
        logit_error=max(float(np.max(np.abs(x-y))) for x,y in zip(a[:3],b[:3]))
        command_a=rank_commands(decoder.score(a[0][0]),entries)['command']
        command_b=rank_commands(decoder.score(b[0][0]),entries)['command']
        passed=feature_error<2e-5 and logit_error<2e-5 and canonical(command_a)==canonical(command_b)
        results.append({'id':row['id'],'frontend_max_error':feature_error,'logit_max_error':logit_error,
            'same_command':canonical(command_a)==canonical(command_b),'pass':passed})
        for repetition in range(4):
            began=time.perf_counter();p,_=sf.read(io.BytesIO(payload),dtype='float32')
            outputs=infer(features(p,mean,std));rank_commands(decoder.score(outputs[0][0]),entries)
            if repetition:timings.append((time.perf_counter()-began)*1000)
    return {'pass':all(r['pass'] for r in results),'n':len(results),'rows':results,
        'mode':'complete-utterance waveform frontend and one full ONNX invocation; no chunk or streaming claim',
        'desktop_timing':{'n':len(timings),'p50_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),
            'warmup':'one unmeasured repeat per recording; session already loaded',
            'includes':'in-memory WAV decode, frontend, CPU ONNX with one thread, direct single-process finite scoring',
            'excludes':'disk, utterance duration, wake, VAD, 400ms endpoint wait, device execution'},'physical_pi':False}


def qualify():
    entries=prepare();selection=select_candidate(entries)
    protocol=json.loads((HERE/'protocol.json').read_text());verify_bindings(ROOT,protocol['bindings'])
    winner=next(c for c in selection['candidates'] if c['name']==selection['winner'])
    directory=CACHE/'selected-evaluation'
    if directory.exists():raise FileExistsError('Preserve prior qualification output')
    bundle=directory/'bundle';bundle.mkdir(parents=True)
    shutil.copyfile(ROOT/winner['graph'],bundle/'model.onnx')
    engine,load_ms=session(bundle/'model.onnx');norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text())
    data=json.loads((PARENT/'data-freeze.json').read_text())
    calibration=[r for r in read_rows(PARENT_CACHE/'prepared/calibration.jsonl') if r['reference_status'] in ('supported','unsupported')]
    test=read_rows(PARENT_CACHE/'prepared/test.jsonl');holdout=read_rows(PARENT_CACHE/'prepared/holdout.jsonl')
    arrays={}
    for row in calibration+test+holdout:
        if sha(row['features'])!=row['features_sha256']:raise ValueError('Qualification feature identity changed')
        arrays[row['id']]=normalized(row,norm)
    cal=run_score(winner,calibration,arrays,norm,entries);save(directory/'calibration-predictions.json',cal)
    policy=calibrate(cal)
    binding={'model_sha256':sha(bundle/'model.onnx'),'catalog_sha256':sha(HERE/'catalog.json'),
        'normalization_sha256':sha(PARENT_CACHE/'prepared/normalization.json'),
        'custom_slots_sha256':sha(PARENT/'custom-slots.json'),
        'schema_sha256':sha(ROOT/'experiments/pcv-human-slots-v1/output.schema.json')}
    policy['binding']=binding;save(bundle/'policy.json',policy)
    for p in (HERE/'catalog.json',PARENT/'custom-slots.json',PARENT_CACHE/'prepared/normalization.json',
              ROOT/'experiments/pcv-human-slots-v1/output.schema.json'):
        shutil.copyfile(p,bundle/p.name)
    immutable(HERE/'selection-before-test.json',{'winner':winner,'binding':binding,
        'development_selection_sha256':sha(HERE/'development-selection.json'),
        'policy_sha256':sha(bundle/'policy.json'),'calibration_predictions_sha256':sha(directory/'calibration-predictions.json'),
        'test_manifest':data['manifests']['test'],'holdout_manifest':data['manifests']['holdout'],
        'test_inference_accessed':False,'prior_parent_test_exposure':'declared repeated benchmark, not a new blind corpus'})
    report={'winner':winner,'policy':policy,'binding':binding,'session_load_ms':load_ms,
            'source_only_downstream_audio':True,'offline_pretrained_teacher':winner['name']!='parent_v3'}
    for name,rows in (('test',test),('holdout',holdout)):
        predictions=run_score(winner,rows,arrays,norm,entries)
        for r in predictions:
            r['policy_candidate']=accepted_candidate(r,policy);r['output']=decision(r,policy)
            r['accepted_command']=r['output']['command']
        save(directory/(name+'-predictions.json'),predictions)
        report[name]={key:measured(predictions,key) for key in ('candidate','policy_candidate','accepted_command')}
        report[name]['roles']={origin:{key:role_metrics([r for r in predictions if r['synthetic']==(origin=='synthetic')],key)
            for key in ('candidate','policy_candidate','accepted_command')} for origin in ('human','synthetic')}
        tables={origin:{key:phrase_table([r for r in predictions if r['synthetic']==(origin=='synthetic')],key)
            for key in ('candidate','policy_candidate','accepted_command')} for origin in ('human','synthetic')}
        immutable(HERE/('phrases-'+name+'.json'),tables)
        report[name]['phrase_evidence']={origin:{key:{'covered':sum(r['n']>0 for r in table),
            'perfect_cells':sum(r['pass'] for r in table),'baseline_covered':sum(r['n']>0 for r in table if r['tier']=='baseline'),
            'baseline_perfect_cells':sum(r['pass'] for r in table if r['tier']=='baseline'),
            'baseline_n':sum(r['n'] for r in table if r['tier']=='baseline'),
            'baseline_correct':sum(r['correct'] for r in table if r['tier']=='baseline')}
            for key,table in modes.items()} for origin,modes in tables.items()}
    report['endpoint']=endpoint_check(engine,test,norm,entries)
    report.update({'installed':False,'physical_pi':False,'new_external_audio':0,'test_tuned':False})
    recipe={'schema_version':3,'model_config':winner['model_config'],
        'state_shapes':[p.shape for p in engine.get_inputs()[1:]],
        'heads':['ctc','intent','target'],'binding':binding,'policy_enabled':policy['enabled'],
        'input':'float32 [1,40,input_frames]','output':'raw [1,output_frames,classes]',
        'feature_hop_ms':10,'output_hop_ms':20 if winner['mode']=='endpoint_context' else 10,
        'inference_mode':'complete_utterance','teacher_shipped':False,'installed':False,
        'status':'qualified_candidate' if policy['enabled'] and report['endpoint']['pass'] else 'unqualified_research_candidate'}
    save(bundle/'recipe.json',recipe)
    immutable(HERE/'results.json',report)
    verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    print('QUALIFICATION_COMPLETE',json.dumps({'winner':winner['name'],'policy_enabled':policy['enabled'],
        'human_raw':report['test']['candidate']['human']['overall'],
        'synthetic_raw':report['test']['candidate']['synthetic']['overall'],'endpoint_pass':report['endpoint']['pass']}),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    else:qualify()
