"""Locked refreshed-source ONNX qualification and machine-bound release summary."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.train.pcv_release_training import HERE,CACHE,SOURCE,valid_run_id
from scripts.test.pcv_public_qualification import verify_bindings,accepted_candidate,decision
from scripts.test.pcv_updated_eval import calibrate,enrich,measured,phrase_table,gates
from scripts.test.pcv_final_qualification import session,strata,role_metrics,endpoint_check,quantization_fidelity
from scripts.train.pcv_context_training import score_context
from scripts.test.pcv_public_eval import normalized
from scripts.train.pcv_updated_training import development_subset
from scripts.train.pcv_updated_prepare import spoken
from scripts.runtime.pcv_schema import INTENTS,canonical,output as typed_output
from vcm.catalog import PHONES


def release_quality(enabled,human,synthetic,endpoint):
    return bool(enabled and gates(human) and gates(synthetic,False) and endpoint)


def typed_phrase_table(records):
    """Actual acceptance must match schema and rejection reason, not just null."""
    definitions=json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    table=[]
    for definition in definitions:
        expected=typed_output(definition['command'],None if definition['command'] is not None else definition.get('reason','OOV'))
        text=spoken(definition['text'])
        matches=[r for r in records if r.get('reference_status')!='ambiguous' and r.get('text')==text]
        correct=sum(canonical(r['output'])==canonical(expected) for r in matches)
        table.append({'id':definition['id'],'text':definition['text'],'tier':definition['tier'],
            'expected':definition['command'],'expected_output':expected,'comparison':'exact_schema_output',
            'n':len(matches),'correct':correct,'accuracy':correct/len(matches) if matches else None,
            'pass':bool(matches) and correct==len(matches)})
    return table


def phrase_tables(records):
    result={}
    for origin in ('human','synthetic'):
        subset=[r for r in records if r['synthetic']==(origin=='synthetic')]
        modes={key:[{**r,'comparison':'command_only'} for r in phrase_table(subset,key)] for key in ('candidate','policy_candidate')}
        modes['accepted_command']=typed_phrase_table(subset);result[origin]=modes
    return result


def bound(paths):
    return {str(Path(p).relative_to(ROOT)):sha(p) for p in paths}


def show_metric(metric):
    n=metric['positive_n'];correct=metric['exact']
    return f'{correct}/{n} ({100*correct/n:.2f}%)' if n else 'Untested (0 recordings)'


def phrase_evidence(tables):
    return {origin:{key:{'covered':sum(r['n']>0 for r in table),
        'perfect_cells':sum(r['pass'] for r in table),'baseline_covered':sum(r['n']>0 for r in table if r['tier']=='baseline'),
        'baseline_perfect_cells':sum(r['pass'] for r in table if r['tier']=='baseline'),
        'baseline_n':sum(r['n'] for r in table if r['tier']=='baseline'),
        'baseline_correct':sum(r['correct'] for r in table if r['tier']=='baseline')}
        for key,table in modes.items()} for origin,modes in tables.items()}


def score(graph,rows,norm,entries):
    verify_bindings(ROOT,{r['features']:r['features_sha256'] for r in rows})
    arrays={r['id']:normalized(r,norm) for r in rows};engine,_=session(graph)
    return enrich(score_context(engine,rows,arrays,entries),rows)


def catalog_oracle(entries):
    import numpy as np
    from scripts.pcv_parallel import ParallelScorer
    from scripts.runtime.pcv_schema import compile_inventory,canonical
    from scripts.train.pcv_public_data import _words
    prompts=json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    cases=compile_inventory(prompts,_words())['entries']
    def logits():
        for row in cases:
            x=np.full((2*len(row['phones'])+1,40),-12.,dtype='float32');x[:,0]=12.
            for i,p in enumerate(row['phones']):x[2*i+1,0]=-12.;x[2*i+1,p]=12.
            yield x
    with ParallelScorer(entries,workers=4) as scorer:
        results=[{'id':r['id'],'pass':canonical(r['command'])==canonical(s['prediction']['command'])}
                 for r,s in zip(cases,scorer.map(logits()),strict=True)]
    return {'n':len(results),'correct':sum(r['pass'] for r in results),'rows':results,
        'evidence':'Ideal acoustic cues, finite command-only software check; rejection reasons are not inferred from these cues. Not recorded speech or model accuracy.'}


def qualify(run_id='release-v1'):
    import torch
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.train.pcv_context_int8 import export_quantized
    valid_run_id(run_id);directory=HERE/'runs'/run_id;output=CACHE/'runs'/run_id/'evaluation'
    release_path=directory/'release.json'
    selection_path=directory/'selection-before-calibration.json';selected=json.loads(selection_path.read_text())
    verify_bindings(ROOT,selected['bindings']);protocol=json.loads((directory/'protocol.json').read_text())
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    data=json.loads((HERE/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in data['manifests'].values()})
    if release_path.exists():
        release=json.loads(release_path.read_text());verify_bindings(ROOT,release['bindings'])
        # Reuse locked test receipts without reevaluating for selection. Publisher
        # workflow subsequently executes an independent extracted-package replay.
        print('VERIFIED_LOCKED_RELEASE',run_id,release['qualified'],flush=True);return release
    evaluation_files=[Path(__file__),ROOT/'scripts/test/test_pcv_release_qualification.py',
        ROOT/'scripts/test/pcv_final_qualification.py',ROOT/'scripts/test/pcv_updated_eval.py',
        ROOT/'scripts/test/pcv_public_qualification.py',selection_path,directory/'protocol.json',
        HERE/'data-freeze.json',HERE/'catalog.json',HERE/'custom-slots.json',
        ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json',ROOT/'experiments/pcv-human-slots-v1/output.schema.json']
    immutable(directory/'evaluation-protocol.json',{'bindings':bound(evaluation_files),
        'selection':'Checkpoint fixed by development; INT8 compared to same FP32 checkpoint only. No test selection.',
        'gates':protocol['quality_gates'],'quantization_gate':protocol['quantization_gate'],
        'holdout_role':'Published test determines release quality; separately reserved holdout is descriptive replay evidence and never selects a model or threshold.',
        'test_accessed':False,'all415':'Each approved phrase has a denominator; zero is untested.'})
    output.mkdir(parents=True,exist_ok=True)
    entries=json.loads((HERE/'catalog.json').read_text())['entries']
    oracle=catalog_oracle(entries);immutable(directory/'oracle.json',oracle)
    if oracle['correct']!=415 or oracle['n']!=415:raise ValueError('Finite schema oracle failed')
    norm=json.loads((CACHE/'prepared/normalization.json').read_text())
    export_path=directory/'export-development.json'
    if export_path.exists():
        export=json.loads(export_path.read_text());verify_bindings(ROOT,export['bindings'])
    else:
        torch.set_num_threads(2)
        payload=torch.load(ROOT/selected['checkpoint'],map_location='cpu',weights_only=True)
        model=PooledContextVCM(**payload['model_config']);model.load_state_dict(payload['weights'])
        exported=export_quantized(model,output/'export');dev=development_subset(read_rows(CACHE/'prepared/dev.jsonl'))
        comparisons={}
        for kind in ('fp32','int8'):
            predictions=score(output/'export'/f'model.{kind}.onnx',dev,norm,entries)
            save(output/f'development-{kind}.json',predictions);comparisons[kind]=strata(predictions)
        fidelity=quantization_fidelity(comparisons['fp32'],comparisons['int8'])
        export={'export':exported,'development':comparisons,'fidelity_pass':fidelity,
            'resource_pass':exported['int8_bytes']<protocol['maximum_graph_bytes'],
            'bindings':bound([output/'export/model.int8.onnx',output/'export/model.fp32.onnx']),
            'test_inference_accessed':False}
        immutable(export_path,export)
    if not export['fidelity_pass'] or not export['resource_pass']:raise ValueError('INT8 development fidelity/resource gate failed; no test accessed')
    bundle=output/'bundle';bundle.mkdir(exist_ok=True)
    graph=output/'export/model.int8.onnx';shutil.copyfile(graph,bundle/'model.onnx')
    schema=ROOT/'experiments/pcv-human-slots-v1/output.schema.json'
    for source in (HERE/'catalog.json',HERE/'custom-slots.json',CACHE/'prepared/normalization.json',schema):
        shutil.copyfile(source,bundle/source.name)
    binding={'model_sha256':sha(bundle/'model.onnx'),'catalog_sha256':sha(bundle/'catalog.json'),
        'normalization_sha256':sha(bundle/'normalization.json'),'custom_slots_sha256':sha(bundle/'custom-slots.json'),
        'schema_sha256':sha(bundle/'output.schema.json')}
    calibration=[r for r in read_rows(CACHE/'prepared/calibration.jsonl') if r['reference_status'] in ('supported','unsupported')]
    cal=score(graph,calibration,norm,entries);save(output/'calibration-predictions.json',cal)
    policy=calibrate(cal);policy['binding']=binding;save(bundle/'policy.json',policy)
    immutable(directory/'selection-before-test.json',{'model_binding':binding,'policy_sha256':sha(bundle/'policy.json'),
        'selection_sha256':sha(selection_path),'export_development_sha256':sha(export_path),
        'calibration_predictions_sha256':sha(output/'calibration-predictions.json'),
        'test_manifest':data['manifests']['test'],'holdout_manifest':data['manifests']['holdout'],
        'test_inference_accessed':False,'test_used_for_selection':False})
    report={'model_binding':binding,'policy_sha256':sha(bundle/'policy.json'),'policy':policy,
        'test_used_for_selection':False,'source_revision':protocol['source_revision'],
        'teacher_used':selected['winner']=='distilled','teacher_shipped':False,'export':export,
        'source_only_downstream_audio':True,'installed':False,'physical_pi':False,'oracle':{'n':415,'correct':415}}
    test_rows=None;stored_predictions={};stored_tables={}
    for name in ('test','holdout'):
        rows=read_rows(CACHE/'prepared'/f'{name}.jsonl')
        if name=='test':test_rows=rows
        predictions=score(graph,rows,norm,entries)
        for row in predictions:
            row['policy_candidate']=accepted_candidate(row,policy);row['output']=decision(row,policy)
            row['accepted_command']=row['output']['command']
        save(output/f'{name}-predictions.json',predictions)
        report[name]={key:measured(predictions,key) for key in ('candidate','policy_candidate','accepted_command')}
        report[name]['roles']={origin:{key:role_metrics([r for r in predictions if r['synthetic']==(origin=='synthetic')],key)
            for key in ('candidate','policy_candidate','accepted_command')} for origin in ('human','synthetic')}
        tables=phrase_tables(predictions)
        stored_predictions[name]=predictions;stored_tables[name]=tables
        report[name]['phrase_evidence']=phrase_evidence(tables)
    engine,load_ms=session(bundle/'model.onnx');endpoint=endpoint_check(engine,test_rows,norm,entries)
    report['endpoint']=endpoint;report['session_load_ms']=load_ms
    metrics=report['test']['policy_candidate']
    qualified=release_quality(policy['enabled'],metrics['human']['overall'],metrics['synthetic']['overall'],endpoint['pass'])
    if policy['enabled'] and not qualified:
        # Thresholds remain unchanged. A failed held-out quality gate can only
        # disable execution, never change thresholds to improve the test score.
        report['locked_calibration_policy']=dict(policy)
        policy={**policy,'enabled':False,'reason':'held_out_quality_or_endpoint_gate_failed'}
        save(bundle/'policy.json',policy);report['policy']=policy;report['policy_sha256']=sha(bundle/'policy.json')
        report['acceptance_disabled_after_test']=True
        for name,predictions in stored_predictions.items():
            for row in predictions:
                row['output']=decision(row,policy);row['accepted_command']=row['output']['command']
            save(output/f'{name}-predictions.json',predictions)
            report[name]['accepted_command']=measured(predictions,'accepted_command')
            for origin in ('human','synthetic'):
                subset=[r for r in predictions if r['synthetic']==(origin=='synthetic')]
                report[name]['roles'][origin]['accepted_command']=role_metrics(subset,'accepted_command')
                stored_tables[name][origin]['accepted_command']=typed_phrase_table(subset)
            report[name]['phrase_evidence']=phrase_evidence(stored_tables[name])
    for name,tables in stored_tables.items():immutable(directory/f'phrases-{name}.json',tables)
    recipe={'schema_version':3,'model_config':protocol['architecture'],'architecture_class':'PooledContextVCM',
        'intent_labels':['OOV',*INTENTS],'acoustic_labels':['<blank>',*PHONES],
        'state_shapes':[],'heads':['ctc','intent','target'],'binding':binding,'policy_enabled':policy['enabled'],
        'input':'float32 [1,40,input_frames]','output':'raw [1,ceil(input_frames/2),classes]',
        'feature_hop_ms':10,'output_hop_ms':20,'inference_mode':'complete_utterance',
        'teacher_shipped':False,'installed':False,'status':'qualified_candidate' if qualified else 'unqualified_research_candidate'}
    save(bundle/'recipe.json',recipe);report['qualified']=qualified
    if not endpoint['pass']:raise ValueError('Endpoint parity failed')
    immutable(directory/'results.json',report)
    if qualified:
        immutable(directory/'qualification.json',{'qualified':True,'test_used_for_selection':False,
            'model_binding':binding,'policy_sha256':sha(bundle/'policy.json'),
            'results_path':'results.json','results_sha256':sha(directory/'results.json')})
    raw=report['test']['candidate'];baseline=report['test']['phrase_evidence']['human']['candidate']
    source_audit=json.loads((SOURCE/'audio-audit.json').read_text())
    original=read_rows(ROOT/source_audit['manifest'])
    hours=sum(r['samples'] for r in original)/16000/3600
    human_ids={(r['source'],str(r['speaker_id'])) for r in original if not r['is_synthetic']}
    synthetic_ids={(r['source'],str(r['speaker_id'])) for r in original if r['is_synthetic']}
    slot_roles={key for r in entries if r['command'] for key in r['command']['slots']}
    summary={'title':'ME2 - Voice Controlled Smart Device',
        'subtitle':'Local wake word, VAD and finite command recognition; no cloud round-trip',
        'author':'ME2 project','date':'2 October 2026','status':'Qualified for Pi validation' if qualified else 'Diagnostic candidate - acceptance gates failed',
        'architecture':['Mic 16 kHz / 40-bin log-mel','Endpoint encoder: 6 blocks / 192 channels','Intent + target heads','Finite command scorer','Validated JSON frontend','Raspberry Pi 4'],
        'model':{'name':'Pooled context VCM / '+selected['winner'],'parameters':f"{export['export']['fp32']['parameters']/1e6:.3f} M",
            'weights':f"{(bundle/'model.onnx').stat().st_size/1e6:.3f} MB INT8",'frontend':'40 mel; 25 ms window / 10 ms hop'},
        'dataset':{'source':'https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands',
            'sourceShort':'airimonda/ai231-me2-voice-commands','revision':protocol['source_revision'],
            'hours':f'{hours:.2f} h','utterances':f'{len(original):,} recordings',
            'speakers':f'{len(human_ids):,} human IDs / {len(synthetic_ids)} TTS IDs',
            'intents':'18 + OOV','slots':f'{len(slot_roles)} typed roles',
            'access':'Public HF snapshot; no DOI verified','license':'Source-dependent; see bundled source card'},
        'training':{'compute':protocol['hardware']['gpu']+' (local; no A100)',
            'objective':'CTC + intent/target; matched hard / teacher arms','optimizer':'AdamW / cosine schedule',
            'steps':str((protocol['training']['human_epochs']+protocol['training']['mixed_epochs']*len(selected['arms']))*protocol['training']['steps_per_epoch'])+' student updates across completed arms',
            'loss':'See training histories','seeds':'1102 / 1202 student; 1112 / 1212 teacher'},
        'metrics':{'human':'Raw candidate '+show_metric(raw['human']['overall']),
            'synthetic':'Raw candidate '+show_metric(raw['synthetic']['overall']),
            'humanAccepted':show_metric(report['test']['accepted_command']['human']['overall']),
            'syntheticAccepted':show_metric(report['test']['accepted_command']['synthetic']['overall']),
            'baseline':f"Raw {baseline['baseline_correct']}/{baseline['baseline_n']}; {baseline['baseline_covered']}/93 phrases",
            'keyword':'Pi pending','intent':'Pi pending','falseAccept':'Pi pending',
            'latencyP95':'Pi pending','rtf':'Pi pending','runtime':'onnxruntime CPU','threads':'1'},
        'release':{'repository':'github.com/justinruaya123/smart-speaker (public status not verified)',
            'codeLicense':'Not verified; no MIT claim','modelArtifact':'Bundled models/default/model.onnx',
            'modelLicense':'Not assigned; teacher/source notices retained',
            'reproduction':'python release.py [--skip-train]','logs':'Frozen manifests, protocols and local run histories',
            'piScript':'pi_test.py writes REPORT.md + metrics.json; Pi measurement pending',
            'heldout':'Published test/holdout retained; fresh weights after split changes',
            'baseline':'Matched hard-label versus distilled compact student'},
        'sources':['Pinned HF source card','Frozen training + qualification receipts','Desktop measurements are not physical Pi evidence'],
        'qualified':qualified,'model_binding':binding,'physical_pi_measured':False,'raw_results':str((directory/'results.json').relative_to(ROOT))}
    immutable(directory/'summary.json',summary)
    files=[selection_path,directory/'protocol.json',directory/'results.json',directory/'summary.json',
        SOURCE/'acquisition-complete.json',SOURCE/'source-card.txt',CACHE/'prepared/test.jsonl',
        *[p for p in bundle.iterdir() if p.is_file()]]
    release={'bundle':str(bundle.relative_to(ROOT)),'summary':str((directory/'summary.json').relative_to(ROOT)),
        'qualified':qualified,'sample_manifest':str((CACHE/'prepared/test.jsonl').relative_to(ROOT)),
        'source_receipt':str((SOURCE/'acquisition-complete.json').relative_to(ROOT)),
        'source_card':str((SOURCE/'source-card.txt').relative_to(ROOT)),
        'qualification_receipt':str((directory/'qualification.json').relative_to(ROOT)) if qualified else None,
        'bindings':bound(files)}
    immutable(release_path,release);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    print('RELEASE_QUALIFICATION',json.dumps({'qualified':qualified,'human_raw':raw['human']['overall'],
        'synthetic_raw':raw['synthetic']['overall'],'endpoint_pass':endpoint['pass']}),flush=True)
    return release


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',default='release-v1')
    qualify(parser.parse_args().run_id)
