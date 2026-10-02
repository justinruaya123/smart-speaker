"""Locked source-only evaluation; distinguish ambiguity and absent evidence."""
from collections import Counter
import io
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from einops import pack, reduce, rearrange
from scripts.runtime.pcv_schema import INTENTS,baseline_cases,extended_cases,canonical
from scripts.train.pcv_updated_data import HERE,CACHE
from scripts.train.pcv_updated_prepare import spoken
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import (verify_bindings,precise_metrics,accepted_candidate,decision)
from scripts.test.pcv_public_eval import score,normalized
from scripts.models.pcv_residual import ResidualVCM,export_onnx
from scripts.pcv_parallel import ParallelScorer


def phrase_table(records,key):
    table=[]
    # Reuse the user-approved phrase contract, never its generated recordings.
    definitions=json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    for definition in definitions:
        text=spoken(definition['text'])
        matching=[r for r in records if r.get('reference_status')!='ambiguous' and r.get('text')==text]
        correct=sum(canonical(r[key])==canonical(definition['command']) for r in matching)
        table.append({'id':definition['id'],'text':definition['text'],'tier':definition['tier'],
            'expected':definition['command'],'n':len(matching),'correct':correct,
            'accuracy':correct/len(matching) if matching else None,'pass':bool(matching) and correct==len(matching)})
    return table


def measured(records,key):
    eligible=[r for r in records if r.get('reference_status') in ('supported','unsupported')]
    return {'all':precise_metrics(eligible,key),
        'human':precise_metrics([r for r in eligible if not r['synthetic']],key),
        'synthetic':precise_metrics([r for r in eligible if r['synthetic']],key),
        'per_source':{s:precise_metrics([r for r in eligible if r['source']==s],key) for s in sorted({r['source'] for r in eligible})},
        'unresolved_reference_n':sum(r.get('reference_status')=='ambiguous' for r in records)}


def gates(metrics,require_negatives=True):
    return bool(metrics['positive_n'] and (metrics['precision'] or 0)>=.99 and (metrics['recall'] or 0)>=.90
        and ((metrics['negative_n'] and metrics['false_accept_rate']<=.01) if require_negatives else True))


def calibrate(records):
    records=[r for r in records if r['reference_status'] in ('supported','unsupported')]
    values=[r['score'] for r in records if r['candidate'] and r['score'] is not None]
    thresholds=sorted(set([-100.]+[float(v) for v in np.quantile(values,np.linspace(0,1,11))])) if values else [0.]
    missing={origin:sorted(set(INTENTS)-{r['expected']['intent'] for r in records if r['expected'] and r['synthetic']==(origin=='synthetic')}) for origin in ('human','synthetic')}
    best=None;best_key=None;feasible=[]
    correct=np.asarray([r['expected'] is not None and canonical(r['expected'])==canonical(r['candidate']) for r in records])
    positive=np.asarray([r['expected'] is not None for r in records]);human=np.asarray([not r['synthetic'] for r in records])
    for threshold in thresholds:
        for margin in (0.,.002,.005,.01,.02,.05,.1):
            for probability in (0.,.5,.7,.9):
                policy={'min_score':threshold,'min_margin':margin,'min_intent_probability':probability}
                accepted=np.asarray([accepted_candidate(r,policy) is not None for r in records]);strata={}
                for origin,mask in [('human',human),('synthetic',~human)]:
                    p=int(np.sum(positive&mask));n=int(np.sum(~positive&mask));a=int(np.sum(accepted&mask))
                    c=int(np.sum(correct&accepted&mask));f=int(np.sum(accepted&~positive&mask))
                    strata[origin]={'positive_n':p,'negative_n':n,'accepted':a,'exact':c,'negative_errors':f,
                        'precision':c/a if a else None,'recall':c/p if p else None,'false_accept_rate':f/n if n else None}
                # Absence of synthetic negatives is explicitly unmeasured. Human
                # negatives are required; missing human intent calibration blocks
                # a global 18-intent release, regardless of pooled synthetic scores.
                safe=bool((strata['human']['precision'] or 0)>=.99 and strata['human']['negative_n'] and strata['human']['false_accept_rate']<=.01
                          and (strata['synthetic']['precision'] or 0)>=.99)
                recalls=[m['recall'] or 0 for m in strata.values()]
                key=(int(safe),min(recalls),sum(recalls),sum((m['precision'] or 0) for m in strata.values()))
                item={**policy,'calibration_strata':strata}
                if best_key is None or key>best_key:best=item;best_key=key
                if not missing['human'] and not missing['synthetic'] and gates(strata['human']) and gates(strata['synthetic'],False):feasible.append((key,item))
    chosen=max(feasible,key=lambda x:x[0])[1] if feasible else best
    return {**chosen,'enabled':bool(feasible),'missing_positive_intents':missing,
        'reason':'calibration_gates_passed' if feasible else 'calibration_quality_or_coverage_gates_failed',
        'synthetic_negative_rejection':'unmeasured: no supplied synthetic negative recordings',
        'grid_size':len(thresholds)*7*4,'test_used':False}


def enrich(records,rows):
    source={r['id']:r for r in rows}
    for r in records:
        s=source[r['id']]
        for key in ('reference_status','reason','accent_group','published_split','transcript_source','exclusion_reason'):
            r[key]=s.get(key)
    return records


def endpoint_checks(session,rows,norm,entries):
    """One complete-waveform frontend; compare full vs cached ONNX computation."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from vcm.frontend import features
    selected=[]
    for synthetic in (False,True):
        available=[r for r in rows if r['synthetic']==synthetic and r['expected']]
        selected.extend(sorted(available,key=lambda r:r['id'])[:6])
    payloads={};groups={r['parquet'] for r in selected}
    for path in groups:
        wanted={r['parquet_row']:r for r in selected if r['parquet']==path};offset=0
        for batch in pq.ParquetFile(ROOT/path).iter_batches(batch_size=128,columns=['audio']):
            for value in batch.to_pylist():
                if offset in wanted:payloads[wanted[offset]['id']]=value['audio']['bytes']
                offset+=1
    results=[];latencies=[]
    with ParallelScorer(entries,workers=1) as scorer:
        for row in selected:
            began=time.perf_counter();pcm,rate=sf.read(io.BytesIO(payloads[row['id']]),dtype='float32')
            if rate!=16000:raise ValueError('Unexpected rate')
            x=features(pcm,np.asarray(norm['mean'],dtype='float32'),np.asarray(norm['std'],dtype='float32'))
            cached=normalized(row,norm)
            error=float(np.max(np.abs(x-cached)))
            feed={'features':rearrange(x,'c t -> 1 c t')}
            feed.update({p.name:np.zeros(p.shape,dtype='float32') for p in session.get_inputs()[1:]})
            full=session.run(None,feed)
            output=list(scorer.map([full[0][0]]))[0]['prediction']
            latencies.append((time.perf_counter()-began)*1000)
            parts=[[],[],[]];states={p.name:np.zeros(p.shape,dtype='float32') for p in session.get_inputs()[1:]}
            offset=0;index=0;chunks=(1,3,4,8,17,41)
            while offset<x.shape[1]:
                end=min(x.shape[1],offset+chunks[index%len(chunks)])
                out=session.run(None,{'features':rearrange(x[:,offset:end],'c t -> 1 c t'),**states})
                for k in range(3):parts[k].append(out[k])
                states={p.name:v for p,v in zip(session.get_inputs()[1:],out[3:],strict=True)}
                offset=end;index+=1
            joined=[pack(part,'b * c')[0] for part in parts]
            state_error=max(float(np.max(np.abs(a-b))) for a,b in zip(full[:3],joined))
            other=list(scorer.map([joined[0][0]]))[0]['prediction']
            results.append({'id':row['id'],'synthetic':row['synthetic'],'frontend_max_error':error,
                'cached_logits_max_error':state_error,'same_command':canonical(output['command'])==canonical(other['command']),
                'pass':error<2e-5 and state_error<2e-5 and canonical(output['command'])==canonical(other['command'])})
    return {'pass':all(r['pass'] for r in results),'rows':results,'n':len(results),
        'path':'complete-waveform endpoint frontend, then stateful acoustic chunks; no streaming-waveform parity claim',
        'desktop_timing':{'n':len(latencies),'p50_ms':float(np.median(latencies)),'p95_ms':float(np.quantile(latencies,.95)),
            'includes':'in-memory WAV decode, frontend, ONNX, single worker finite scorer; first-call warmup included',
            'excludes':'disk access, utterance duration, wake, VAD, endpoint wait, device execution'},'physical_pi':False}


def evaluate(version=None):
    import torch
    import onnxruntime as ort
    torch.set_num_threads(2)
    training=json.loads((HERE/'training-protocol.json').read_text());verify_bindings(ROOT,training['bindings'])
    verify_bindings(ROOT,training['installed_artifact_hashes'])
    data=json.loads((HERE/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in data['manifests'].values()})
    selected=json.loads((HERE/'selection-before-calibration.json').read_text())
    destination=HERE;work=CACHE
    extra_sources=[]
    if version is not None:
        if version!='pcv-updated-voice-v3':raise ValueError('Unregistered evaluation version')
        destination=ROOT/'experiments'/version;work=ROOT/'.cache'/version
        followup=json.loads((destination/'protocol.json').read_text());verify_bindings(ROOT,followup['bindings'])
        chosen=json.loads((destination/'selection-before-calibration.json').read_text())
        selected={'arms':{'continued':chosen},'winner':'continued'}
        extra_sources=[destination/'protocol.json',destination/'selection-before-calibration.json']
    sources=[Path(__file__),ROOT/'scripts/test/test_pcv_updated_eval.py',HERE/'training-protocol.json',HERE/'selection-before-calibration.json',
             ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json',ROOT/'experiments/pcv-human-slots-v1/output.schema.json',*extra_sources]
    immutable(destination/'evaluation-protocol.json',{'bindings':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'annotation_policy':'all original test rows retained; ambiguous references separated from exact-command metrics',
        'all_415_phrases':'per-cell denominators; absent recordings never count as passing',
        'calibration':'frozen threshold grid and strict named-light target gate; 18 human intents required for global enablement',
        'test_inference_accessed':False,'holdout_inference_accessed':False})
    norm=json.loads((CACHE/'prepared/normalization.json').read_text());entries=json.loads((HERE/'catalog.json').read_text())['entries']
    calibration=[r for r in read_rows(CACHE/'prepared/calibration.jsonl') if r['reference_status'] in ('supported','unsupported')]
    test=read_rows(CACHE/'prepared/test.jsonl');holdout=read_rows(CACHE/'prepared/holdout.jsonl')
    for row in calibration+test+holdout:
        if sha(row['features'])!=row['features_sha256']:raise ValueError('Changed feature '+row['id'])
    summaries={}
    for arm,selection in selected['arms'].items():
        directory=work/'evaluation'/arm
        if directory.exists():raise FileExistsError('Preserve completed/partial evaluation '+str(directory))
        directory.mkdir(parents=True);bundle=directory/'bundle';bundle.mkdir()
        verify_bindings(ROOT,{selection['checkpoint']:selection['checkpoint_sha256']})
        payload=torch.load(ROOT/selection['checkpoint'],map_location='cpu',weights_only=True)
        model=ResidualVCM(**payload['model_config']);model.load_state_dict(payload['weights'])
        export=export_onnx(model,bundle/'model.onnx');save(directory/'export.json',export)
        if not export['pass']:raise ValueError('Trained ONNX export failed')
        opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(bundle/'model.onnx'),opts,providers=['CPUExecutionProvider'])
        cal=enrich(score(session,calibration,norm,entries,workers=4),calibration)
        save(directory/'calibration-predictions.json',cal)
        policy=calibrate(cal)
        binding={'model_sha256':sha(bundle/'model.onnx'),'catalog_sha256':sha(HERE/'catalog.json'),
            'normalization_sha256':sha(CACHE/'prepared/normalization.json'),'custom_slots_sha256':sha(HERE/'custom-slots.json'),
            'schema_sha256':sha(ROOT/'experiments/pcv-human-slots-v1/output.schema.json')}
        policy['binding']=binding;save(bundle/'policy.json',policy)
        for file in (HERE/'catalog.json',HERE/'custom-slots.json',CACHE/'prepared/normalization.json',ROOT/'experiments/pcv-human-slots-v1/output.schema.json'):
            shutil.copyfile(file,bundle/file.name)
        immutable(directory/'selection-before-test.json',{'selection':selection,'binding':binding,
            'policy_sha256':sha(bundle/'policy.json'),'test_manifest':data['manifests']['test'],
            'holdout_manifest':data['manifests']['holdout'],'calibration_predictions_sha256':sha(directory/'calibration-predictions.json'),
            'test_inference_accessed':False,'holdout_inference_accessed':False})
        report={'arm':arm,'selection':selection,'export':export,'policy':policy}
        for name,rows in [('test',test),('holdout',holdout)]:
            predictions=enrich(score(session,rows,norm,entries,workers=4),rows)
            for r in predictions:
                r['policy_candidate']=accepted_candidate(r,policy);r['output']=decision(r,policy)
                r['accepted_command']=r['output']['command']
            save(directory/(name+'-predictions.json'),predictions)
            report[name]={key:measured(predictions,key) for key in ('candidate','policy_candidate','accepted_command')}
            tables={}
            for origin in ('human','synthetic'):
                subset=[r for r in predictions if r['synthetic']==(origin=='synthetic')]
                tables[origin]={key:phrase_table(subset,key) for key in ('candidate','policy_candidate','accepted_command')}
            immutable(destination/f'phrases-{arm}-{name}.json',tables)
            report[name]['phrase_evidence']={origin:{key:{'covered':sum(r['n']>0 for r in table),
                'perfect_cells':sum(r['pass'] for r in table),'baseline_covered':sum(r['n']>0 for r in table if r['tier']=='baseline'),
                'baseline_perfect_cells':sum(r['pass'] for r in table if r['tier']=='baseline'),
                'baseline_n':sum(r['n'] for r in table if r['tier']=='baseline'),
                'baseline_correct':sum(r['correct'] for r in table if r['tier']=='baseline')} for key,table in modes.items()} for origin,modes in tables.items()}
        report['endpoint']=endpoint_checks(session,test,norm,entries)
        report['source_only']=True;report['new_external_audio']=0;report['distillation_used']=False
        report['physical_pi']=False;report['installed']=False;report['test_tuned']=False
        immutable(destination/f'results-{arm}.json',report)
        save(bundle/'recipe.json',{'schema_version':3,'model_config':model.config,'state_shapes':model.state_shapes,
            'heads':['ctc','intent','target'],'binding':binding,'policy_enabled':policy['enabled'],
            'status':'research_candidate','source_revision':training['source_revision'],'installed':False})
        summaries[arm]={'human_test_raw':report['test']['candidate']['human']['overall'],
            'synthetic_test_raw':report['test']['candidate']['synthetic']['overall'],
            'human_test_policy':report['test']['policy_candidate']['human']['overall'],
            'synthetic_test_policy':report['test']['policy_candidate']['synthetic']['overall'],
            'policy_enabled':policy['enabled'],'endpoint_pass':report['endpoint']['pass'],'export':export,
            'phrase_evidence':report['test']['phrase_evidence']}
        print('ARM_COMPLETE',arm,json.dumps(summaries[arm]),flush=True)
    verify_bindings(ROOT,training['installed_artifact_hashes'])
    immutable(destination/'comparison.json',{'development_selected_arm':selected['winner'],'arms':summaries,
        'installed_artifacts_unchanged':True,'test_used_for_selection':False,'external_fallback_used':False})


if __name__=='__main__':evaluate(sys.argv[1] if len(sys.argv)>1 else None)
