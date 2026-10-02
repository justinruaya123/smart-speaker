"""Locked replay for the matched HF-only students, retaining all earlier gates."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import onnxruntime as ort
from scripts.train.pcv_teacher_data import HERE,CACHE,PARENT,PARENT_CACHE,REVISION
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.models.pcv_residual import ResidualVCM,export_onnx
from scripts.test.pcv_public_eval import score
from scripts.test.pcv_public_qualification import verify_bindings,accepted_candidate,decision
from scripts.test.pcv_updated_eval import enrich,calibrate,measured,phrase_table,endpoint_checks


def choose_with_parent(selected, parent):
    """The already-selected parent remains the zero-update control on a tie."""
    candidates={**selected['arms'],'parent_v3':parent}
    def key(name):
        candidate=candidates[name];metrics=candidate['development']
        return ((metrics['human']['recall']+metrics['synthetic']['recall'])/2,
                -metrics['human']['negative_errors'],0 if name=='parent_v3' else -candidate['epoch'])
    return max(candidates,key=key)


def evaluate():
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'student-protocol.json').read_text());verify_bindings(ROOT,protocol['bindings'])
    verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    selected=json.loads((HERE/'student-selection-before-calibration.json').read_text())
    winner=choose_with_parent(selected,protocol['initial_student'])
    if winner=='parent_v3':
        immutable(HERE/'student-nonpromotion.json',{'decision':'retain parent; neither new student improves the frozen development criterion',
            'development_selected_candidate':winner,'new_arms':selected['arms'],
            'parent':protocol['initial_student'],'student_selection_sha256':sha(HERE/'student-selection-before-calibration.json'),
            'selection_source_sha256':sha(Path(__file__)),'new_calibration_inference':False,
            'new_test_inference':False,'installed':False,
            'reason':'Do not spend held-out test access on a development-rejected candidate.'})
        print('STUDENT_NOT_PROMOTED',winner,flush=True)
        return
    data=json.loads((PARENT/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in data['manifests'].values()})
    files=[Path(__file__),ROOT/'scripts/test/test_pcv_distill_selection.py',ROOT/'scripts/test/pcv_updated_eval.py',
        HERE/'student-protocol.json',HERE/'student-selection-before-calibration.json',
        ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json',
        ROOT/'experiments/pcv-human-slots-v1/output.schema.json']
    immutable(HERE/'student-evaluation-protocol.json',{'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},
        'comparison':'Matched hard-label and distilled arms; no test selection.',
        'development_selected_candidate_including_parent':winner,
        'reference_policy':'Preserve original test/holdout membership; ambiguous references separate; no missing phrase counts as passing.',
        'calibration':'Unchanged parent threshold grid, 99% precision/90% recall/1% human FAR and full human intent coverage for global enablement.',
        'endpoint':'Complete-utterance frontend. Retain original chunk numerical check; never relabel failed tolerances as passes.',
        'test_inference_accessed':False})
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text())
    entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    calibration=[r for r in read_rows(PARENT_CACHE/'prepared/calibration.jsonl') if r['reference_status'] in ('supported','unsupported')]
    test=read_rows(PARENT_CACHE/'prepared/test.jsonl');holdout=read_rows(PARENT_CACHE/'prepared/holdout.jsonl')
    for row in calibration+test+holdout:
        if sha(row['features'])!=row['features_sha256']:raise ValueError('Evaluation feature changed')
    summary={}
    for arm,selection in selected['arms'].items():
        directory=CACHE/'evaluation'/arm
        if directory.exists():raise FileExistsError('Preserve evaluation: '+str(directory))
        bundle=directory/'bundle';bundle.mkdir(parents=True)
        verify_bindings(ROOT,{selection['checkpoint']:selection['checkpoint_sha256']})
        payload=torch.load(ROOT/selection['checkpoint'],map_location='cpu',weights_only=True)
        model=ResidualVCM(**payload['model_config']);model.load_state_dict(payload['weights'])
        export=export_onnx(model,bundle/'model.onnx');save(directory/'export.json',export)
        if not export['pass']:raise ValueError('Student ONNX export failed; preserve failed receipt')
        opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(bundle/'model.onnx'),opts,providers=['CPUExecutionProvider'])
        cal=enrich(score(session,calibration,norm,entries,workers=4),calibration)
        save(directory/'calibration-predictions.json',cal);policy=calibrate(cal)
        binding={'model_sha256':sha(bundle/'model.onnx'),'catalog_sha256':sha(PARENT/'catalog.json'),
            'normalization_sha256':sha(PARENT_CACHE/'prepared/normalization.json'),
            'custom_slots_sha256':sha(PARENT/'custom-slots.json'),
            'schema_sha256':sha(ROOT/'experiments/pcv-human-slots-v1/output.schema.json')}
        policy['binding']=binding;save(bundle/'policy.json',policy)
        for path in (PARENT/'catalog.json',PARENT/'custom-slots.json',PARENT_CACHE/'prepared/normalization.json',
                     ROOT/'experiments/pcv-human-slots-v1/output.schema.json'):
            shutil.copyfile(path,bundle/path.name)
        immutable(directory/'selection-before-test.json',{'selection':selection,'binding':binding,
            'policy_sha256':sha(bundle/'policy.json'),'calibration_predictions_sha256':sha(directory/'calibration-predictions.json'),
            'test_manifest':data['manifests']['test'],'holdout_manifest':data['manifests']['holdout'],
            'test_inference_accessed':False})
        report={'arm':arm,'selection':selection,'export':export,'policy':policy}
        for name,rows in (('test',test),('holdout',holdout)):
            predictions=enrich(score(session,rows,norm,entries,workers=4),rows)
            for r in predictions:
                r['policy_candidate']=accepted_candidate(r,policy);r['output']=decision(r,policy)
                r['accepted_command']=r['output']['command']
            save(directory/(name+'-predictions.json'),predictions)
            report[name]={key:measured(predictions,key) for key in ('candidate','policy_candidate','accepted_command')}
            tables={origin:{key:phrase_table([r for r in predictions if r['synthetic']==(origin=='synthetic')],key)
                for key in ('candidate','policy_candidate','accepted_command')} for origin in ('human','synthetic')}
            immutable(HERE/f'phrases-{arm}-{name}.json',tables)
            report[name]['phrase_evidence']={origin:{key:{
                'covered':sum(r['n']>0 for r in table),'perfect_cells':sum(r['pass'] for r in table),
                'baseline_covered':sum(r['n']>0 for r in table if r['tier']=='baseline'),
                'baseline_perfect_cells':sum(r['pass'] for r in table if r['tier']=='baseline'),
                'baseline_n':sum(r['n'] for r in table if r['tier']=='baseline'),
                'baseline_correct':sum(r['correct'] for r in table if r['tier']=='baseline')}
                for key,table in modes.items()} for origin,modes in tables.items()}
        report['endpoint']=endpoint_checks(session,test,norm,entries)
        report.update({'source_only_downstream_audio':True,'external_pretraining':arm=='distilled',
            'distillation_used':arm=='distilled','new_external_audio':0,'installed':False,'physical_pi':False,'test_tuned':False})
        immutable(HERE/f'results-{arm}.json',report)
        save(bundle/'recipe.json',{'schema_version':3,'model_config':model.config,'state_shapes':model.state_shapes,
            'heads':['ctc','intent','target'],'binding':binding,'policy_enabled':policy['enabled'],
            'status':'research_candidate','source_revision':REVISION,'installed':False,'teacher_shipped':False})
        summary[arm]={'human_test_raw':report['test']['candidate']['human']['overall'],
            'synthetic_test_raw':report['test']['candidate']['synthetic']['overall'],
            'human_test_policy':report['test']['policy_candidate']['human']['overall'],
            'synthetic_test_policy':report['test']['policy_candidate']['synthetic']['overall'],
            'policy_enabled':policy['enabled'],'export':export,'endpoint_pass':report['endpoint']['pass'],
            'phrase_evidence':report['test']['phrase_evidence']}
        print('STUDENT_ARM_COMPLETE',arm,json.dumps(summary[arm]),flush=True)
    verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    immutable(HERE/'student-comparison.json',{'development_selected_arm':winner,
        'training_arm_winner':selected['winner'],'parent_retained':winner=='parent_v3',
        'arms':summary,'test_used_for_selection':False,'external_recordings_used':False,'installed_artifacts_unchanged':True})


if __name__=='__main__':evaluate()
