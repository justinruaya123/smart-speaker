"""Export, lock, evaluate and package the final unchanged VCM through existing runtime."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch

from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.runtime.pcv_final_runtime import BINDINGS, Runtime, verify_bundle
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new
from scripts.test.pcv_public_qualification import accepted_candidate, precise_metrics
from scripts.train import speaker_final_train as training
from scripts.train import speaker_synthetic_data as generation

ROOT, HERE, CACHE, PREPARED = training.ROOT, training.HERE, training.CACHE, training.PREPARED


def quantization_gate(floating, quantized, graph_bytes):
    losses={kind:floating[kind]['raw_exact']-quantized[kind]['raw_exact'] for kind in ('human','synthetic')}
    extra=quantized['human']['raw_negative_errors']-floating['human']['raw_negative_errors']
    return {'pass':all(value<=1 for value in losses.values()) and extra<=1 and graph_bytes<4*1024*1024,
            'exact_losses':losses,'extra_human_negative_errors':extra,'graph_bytes':graph_bytes,
            'maximum_graph_bytes':4*1024*1024,'physical_pi_measured':False}


def calibrate_human(records):
    if any(r['synthetic'] is not False for r in records):
        raise ValueError('Only original human calibration can determine human gates')
    missing=sorted(set(INTENTS)-{r['expected']['intent'] for r in records if r['expected']})
    values=[r['score'] for r in records if r['candidate'] is not None and r['score'] is not None]
    thresholds=sorted(set([-100.]+[float(v) for v in np.quantile(values,np.linspace(0,1,11))])) if values else [0.]
    best=None; best_key=None
    for threshold in thresholds:
        for margin in (0.,.002,.005,.01,.02,.05,.1):
            for probability in (0.,.5,.7,.9):
                policy=dict(min_score=threshold,min_margin=margin,min_intent_probability=probability)
                rows=[dict(r,accepted_command=accepted_candidate(r,policy)) for r in records]
                metric=precise_metrics(rows,'accepted_command')['overall']
                precision=metric['precision'] or 0.; recall=metric['recall'] or 0.
                far=metric['false_accept_rate'] if metric['false_accept_rate'] is not None else 1.
                safe=bool(metric['exact'] and precision>=.99 and far<=.01)
                key=(int(safe),recall,precision,-far)
                if best_key is None or key>best_key:
                    best_key=key;best={**policy,'calibration_metrics':metric}
    metric=best['calibration_metrics']
    enabled=bool(not missing and metric['positive_n'] and metric['negative_n'] and
                 (metric['precision'] or 0)>=.99 and (metric['recall'] or 0)>=.90 and
                 metric['false_accept_rate'] is not None and metric['false_accept_rate']<=.01)
    return {**best,'enabled':enabled,'missing_positive_intents':missing,
            'reason':'human_calibration_gates_passed' if enabled else 'human_calibration_gates_failed',
            'calibration_rows':len(records),'calibration_kind':'original_human_only',
            'grid_size':len(thresholds)*7*4,'test_used':False}


def partition_evaluation(rows):
    eligible=[];excluded=[]
    for row in rows:
        if row['metric_eligible'] is not True:
            excluded.append(row);continue
        status=row['reference_status']
        if status not in {'supported','unsupported'} or ((row['command'] is None)!=(status=='unsupported')):
            raise ValueError('Invalid typed evaluation reference; ambiguity cannot become UNKNOWN')
        eligible.append(row)
    return eligible,excluded


def sample_rows(rows):
    return [dict(row,source='SLURP',synthetic=False,expected=row['command'],speaker=row['speaker_id'],
                 text=row['publisher_metadata']['transcript'],file=row['publisher_metadata']['file'],audio_sha256=row['sha256'])
            for row in rows if row['kind']=='human' and row['split'] in {'test','holdout'}
            and row['metric_eligible'] is True and row['reference_status'] in {'supported','unsupported'}
            and row['publisher_metadata'].get('source')=='SLURP']


def frozen_evaluation_views(prepared):
    prepared=Path(prepared).resolve();receipt=read_json(prepared/'preparation.json');views={}
    for name in ('human/calibration','evaluation/test','evaluation/holdout'):
        expected=(prepared/'views'/f'{name}.jsonl').resolve()
        frozen=receipt['manifests'][name]
        if Path(frozen['path']).resolve()!=expected:
            raise ValueError('Prepared evaluation path identity mismatch: '+name)
        generation._verify({str(expected):frozen['sha256']})
        views[name]=expected
    return views


def verify_release_inputs(receipt):
    generation._verify(receipt['bindings'])
    protocol=read_json(HERE/'release-protocol-before-evaluation.json')
    generation._verify(protocol['bindings'])


def complete_pi_archive(base,output):
    """Keep the old publisher intact and include its current CLI dependency."""
    with zipfile.ZipFile(base) as archive:
        entries={name:archive.read(name) for name in archive.namelist()}
    manifest=json.loads(entries.pop('MANIFEST.json'))
    if set(entries)!=set(manifest['files']) or any(hashlib.sha256(data).hexdigest()!=manifest['files'][name]
                                                  for name,data in entries.items()):
        raise ValueError('Base Pi archive identity mismatch')
    entries['scripts/runtime/speaker_config.py']=(ROOT/'scripts/runtime/speaker_config.py').read_bytes()
    entries['config.toml']=b'''[inference]
engine = "onnxruntime"
mode = "qualified"
threads = 1
intent_threshold = 0.95
slot_threshold = 0.90

[actions]
enabled = false
'''
    manifest['files']={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    manifest['pi_runtime']='Existing CPU ONNX endpoint with current configuration resolver; actions disabled'
    entries['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,10,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,data)
    return dict(path=str(output),sha256=file_digest(output),bytes=output.stat().st_size,
                base_package_sha256=file_digest(base),members=len(entries))


class ONNXModel:
    def __init__(self,path):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def eval(self):return self

    def __call__(self,features,lengths):
        return (*[torch.from_numpy(v.transpose(0,2,1)) for v in self.session.run(None,{'features':features.numpy()})],[])


def external_inputs(rows,norm,name):
    """Preserve every eligible reference, including capture faults as failures."""
    from vcm.frontend import features
    target=CACHE/'evaluation-features'/name;target.mkdir(parents=True,exist_ok=False)
    mean=np.asarray(norm['mean'],dtype=np.float32)[:,None];std=np.asarray(norm['std'],dtype=np.float32)[:,None]

    def prepare(row):
        pcm=read_audio(row,ROOT)
        valid=400<=len(pcm)<=240000 and float(np.max(np.abs(pcm),initial=0))>=1e-6
        result={**row,'expected':row['command'],'synthetic':row['kind']=='synthetic',
                'speaker':row['speaker_id'],'text':row.get('publisher_metadata',{}).get('transcript',''),
                'capture_fault':not valid}
        if not valid:return result,None
        raw=features(pcm);path=target/(row['pcm_sha256']+'.npy')
        with path.open('xb') as stream:np.save(stream,raw,allow_pickle=False)
        result.update(features=str(path),features_sha256=file_digest(path))
        return result,(raw-mean)/std
    prepared=[];arrays={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row,values in pool.map(prepare,rows):
            prepared.append(row)
            if values is not None:arrays[row['id']]=values
    write_new(target/'manifest.jsonl',prepared,jsonl=True)
    return prepared,arrays


def score_external(model,rows,arrays,entries,scorer):
    valid=[r for r in rows if not r.get('capture_fault',False)]
    records=training.score_model(model,valid,arrays,entries,scorer,device='cpu',batch_size=1)
    by_id={r['id']:r for r in records}
    for row in rows:
        if row.get('capture_fault',False):
            by_id[row['id']]=dict(id=row['id'],expected=row['expected'],candidate=None,application_command=None,
                synthetic=row['synthetic'],speaker=row['speaker'],source=row['source'],capture_fault=True,
                score=None,margin=None,intent_index=0,intent_probability=0.,top_intent_probability=0.,target_named_probability=0.,
                application={'accepted':False,'command':None,'reason':'capture_fault'})
    return [by_id[r['id']] for r in rows]


def evaluate():
    selection=read_json(HERE/'selection-before-calibration.json');cfg=training.freeze()
    generation._verify(selection['bindings']);generation._verify(cfg['installed_metadata'])
    output=CACHE/'export'
    if output.exists():raise FileExistsError('Preserve previous final export/evaluation evidence')
    schema=ROOT/'experiments/pcv-human-slots-v1/output.schema.json'
    custom=ROOT/'experiments/pcv-release-train-v1/custom-slots.json'
    files=[HERE/'selection-before-calibration.json',HERE/'protocol.json',Path(__file__),schema,custom,
           ROOT/'scripts/train/pcv_context_int8.py',ROOT/'scripts/runtime/pcv_final_runtime.py',
           ROOT/'scripts/pcv_final_delivery.py',ROOT/'scripts/test/pcv_public_qualification.py',
           ROOT/'scripts/runtime/speaker_config.py',ROOT/'scripts/runtime/pcv_pi_benchmark.py',
           ROOT/'data/sources/speaker-final-pi-assets.source.json',
           ROOT/'experiments/phrase-contract-v1/configs/runtime-cp312-aarch64/dependencies.json']
    views=frozen_evaluation_views(PREPARED)
    files+=list(views.values())
    release_protocol=dict(experiment_id=cfg['experiment_id'],selected_before_calibration=True,
        application=training.APPLICATION,calibration='original human only; no synthetic replacement',
        published_test='one locked regression replay after selection; previously examined corpus',
        augmentation_at_evaluation=False,maximum_int8_graph_bytes=cfg['maximum_int8_graph_bytes'],
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in files})
    write_new(HERE/'release-protocol-before-evaluation.json',release_protocol)
    rows,dev,norm,entries=training.load_inputs();arrays=training.load_arrays(dev,norm)
    payload=torch.load(ROOT/selection['selected']['checkpoint'],map_location='cpu',weights_only=True)
    if payload['model_config']!=training.ARCHITECTURE:raise ValueError('Selected architecture changed')
    model=training.PooledContextVCM(**training.ARCHITECTURE);model.load_state_dict(payload['weights']);model.eval()
    torch.set_num_threads(1)
    from scripts.train.pcv_context_int8 import export_quantized
    exported=export_quantized(model,output)
    from scripts.pcv_parallel import ParallelScorer
    with ParallelScorer(entries,workers=6,return_scores=True) as scorer:
        development={};predictions={}
        for kind in ('fp32','int8'):
            engine=ONNXModel(output/f'model.{kind}.onnx')
            records=training.score_model(engine,dev,arrays,entries,scorer,device='cpu',batch_size=1)
            predictions[kind]=records;development[kind]=training.summarize(records)
            write_new(CACHE/f'development-{kind}.json',records)
        fidelity=quantization_gate(development['fp32'],development['int8'],exported['int8_bytes'])
        export_receipt=dict(export=exported,development=development,fidelity=fidelity,
            exact_candidate_agreement=sum(canonical(a['candidate'])==canonical(b['candidate']) for a,b in zip(predictions['fp32'],predictions['int8'],strict=True)),
            development_rows=len(dev),note='Historical exporter two-MiB flags are not the declared four-MiB full-model gate')
        write_new(HERE/'export-development.json',export_receipt)
        if not fidelity['pass']:raise ValueError('Declared INT8 development/resource gate failed; no calibration/test read')
        bundle=CACHE/'bundle';bundle.mkdir(exist_ok=False)
        for source,name in ((output/'model.int8.onnx','model.onnx'),(PREPARED/'catalog.json','catalog.json'),
                            (PREPARED/'normalization.json','normalization.json'),(schema,'output.schema.json'),(custom,'custom-slots.json')):
            shutil.copyfile(source,bundle/name)
        binding={key:file_digest(bundle/name) for key,name in BINDINGS.items()}
        cal_source=read_jsonl(views['human/calibration']);cal_rows,cal_excluded=partition_evaluation(cal_source)
        cal_rows,cal_arrays=external_inputs(cal_rows,norm,'calibration')
        cal=score_external(ONNXModel(bundle/'model.onnx'),cal_rows,cal_arrays,entries,scorer)
        write_new(CACHE/'calibration-predictions.json',cal)
        policy=calibrate_human(cal);policy['binding']=binding;write_new(bundle/'policy.json',policy)
        from vcm.catalog import PHONES
        recipe=dict(schema_version=3,model_config=training.ARCHITECTURE,architecture_class='PooledContextVCM',
            intent_labels=['OOV',*INTENTS],acoustic_labels=['<blank>',*PHONES],state_shapes=[],heads=['ctc','intent','target'],
            binding=binding,policy_enabled=policy['enabled'],input='float32 [1,40,input_frames]',
            output='raw [1,ceil(input_frames/2),classes]',feature_hop_ms=10,output_hop_ms=20,
            inference_mode='complete_utterance',teacher_shipped=False,installed=False,
            status='qualified_candidate' if policy['enabled'] else 'unqualified_research_candidate')
        write_new(bundle/'recipe.json',recipe);verify_bundle(bundle,diagnostic=not policy['enabled'])
        write_new(HERE/'selection-before-test.json',dict(model_binding=binding,policy_sha256=file_digest(bundle/'policy.json'),
            selection_sha256=file_digest(HERE/'selection-before-calibration.json'),
            export_sha256=file_digest(HERE/'export-development.json'),test_used_for_selection=False,
            protocol_sha256=file_digest(HERE/'release-protocol-before-evaluation.json')))
        results={}
        engine=ONNXModel(bundle/'model.onnx')
        for name in ('test','holdout'):
            source=read_jsonl(views['evaluation/'+name]);eligible,excluded=partition_evaluation(source)
            prepared,values=external_inputs(eligible,norm,name)
            scored=score_external(engine,prepared,values,entries,scorer)
            for record in scored:
                record['qualified_command']=accepted_candidate(record,policy) if policy['enabled'] else None
            write_new(CACHE/f'{name}-predictions.json',scored)
            write_new(CACHE/f'{name}-excluded.jsonl',excluded,jsonl=True)
            results[name]=dict(published_rows=len(source),metric_eligible_rows=len(scored),
                excluded_rows=len(excluded),excluded_references='Preserved ambiguous or publisher-ineligible references; never converted to UNKNOWN',
                application=training.summarize(scored),
                per_intent={kind:precise_metrics((r for r in scored if r['synthetic']==(kind=='synthetic')),'application_command')['per_intent']
                            for kind in ('human','synthetic')},
                qualified={kind:precise_metrics((r for r in scored if r['synthetic']==(kind=='synthetic')),'qualified_command')['overall']
                           for kind in ('human','synthetic')},
                capture_fault_rows=sum(bool(r.get('capture_fault')) for r in scored))
    wake=CACHE/'wake';shutil.copytree(wake,bundle/'wake')
    generation._verify(release_protocol['bindings']);generation._verify(cfg['installed_metadata'])
    receipt=dict(experiment_id=cfg['experiment_id'],status='final_refinement_evaluated',winner=selection['winner'],
        qualified=policy['enabled'],model_binding=binding,policy=policy,parameters=cfg['parameters'],
        graph_bytes=(bundle/'model.onnx').stat().st_size,catalog_entries=len(entries),
        export_development='experiments/speaker-final-refinement-v1/export-development.json',
        test=results['test'],holdout=results['holdout'],physical_pi_measured=False,live_microphone_measured=False,
        installed=False,published_test_used_for_selection=False,
        limitations=['Single seed; augmentation common to both arms, no isolated augmentation-effect estimate',
                     'Synthetic intended-text labels and voice identities lack independent listening validation',
                     'Human calibration missing-class failures remain; demo confidence scores are not calibrated accuracy probabilities',
                     'Published evaluation was previously examined; no new blind unseen-human claim',
                     'Wake/VAD is not trained speaker ownership or overlap recovery'],
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in (
            HERE/'selection-before-test.json',HERE/'release-protocol-before-evaluation.json',
            bundle/'model.onnx',bundle/'recipe.json',bundle/'policy.json')})
    write_new(HERE/'results.json',receipt);return receipt


def package_pi():
    receipt=read_json(HERE/'results.json');verify_release_inputs(receipt)
    source=sample_rows(read_jsonl(PREPARED/'views/evaluation/test.jsonl'))
    sample_dir=CACHE/'pi-sample-source';sample_dir.mkdir(exist_ok=False)
    manifest=sample_dir/'manifest.jsonl';write_new(manifest,source,jsonl=True)
    card=ROOT/'experiments/pcv-source-refresh-v1/source-card.txt'
    protocol=dict(bindings={str(card.relative_to(ROOT)):file_digest(card)},
                  selection='Original CC BY SLURP test rows only; package smoke, not new qualification')
    write_new(sample_dir/'protocol.json',protocol)
    ledger=read_json(ROOT/'data/sources/ai231-primary.source.json')
    allowed={str(p['path']).replace('\\','/'):p['sha256'] for p in ledger['files']}
    files=[]
    for name in sorted({r['parquet'] for r in source}):
        digest=file_digest(ROOT/name)
        if allowed.get(name)!=digest:raise ValueError('Sample shard is not the pinned installed source')
        files.append(dict(path=name,sha256=digest))
    acquisition=dict(repository='airimonda/ai231-me2-voice-commands',revision='25111444af3adff7588d86ab27c304bb080895cf',
                     protocol_sha256=file_digest(sample_dir/'protocol.json'),files=files)
    write_new(sample_dir/'acquisition.json',acquisition)
    from scripts.pcv_final_delivery import package
    base=CACHE/'pi-base.zip'
    result=package(CACHE/'bundle',base,diagnostic=not receipt['qualified'],
        sample_manifest=manifest,source_receipt=sample_dir/'acquisition.json',source_card=card,
        extra_files={'docs/FINAL-REFINEMENT.md':ROOT/'docs/EXPERIMENT-speaker-final-refinement-v1.md',
                     'evidence/final-results.json':HERE/'results.json','evidence/training-protocol.json':HERE/'protocol.json',
                     'evidence/references.json':HERE/'references.json'})
    archive=ROOT/'dist/speaker-final-refinement-v1/speaker-pi4-final.zip'
    result.update(complete_pi_archive(base,archive))
    write_new(archive.with_suffix('.receipt.json'),result)
    write_new(HERE/'package.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('evaluate','package'))
    args=parser.parse_args();result=evaluate() if args.stage=='evaluate' else package_pi()
    print(json.dumps({'stage':args.stage,'status':'complete','qualified':result.get('qualified',False)},indent=2))


if __name__=='__main__':main()
