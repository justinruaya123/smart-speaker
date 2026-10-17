"""Export the bounded compact fit, freeze human recall thresholds, then replay once."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import zipfile

import torch

from scripts.runtime.pcv_final_runtime import BINDINGS, verify_bundle
from scripts.runtime.pcv_schema import INTENTS, canonical, normalize_text
from scripts.runtime.speaker_compact import apply_policy, load_bound_policy
from scripts.test.pcv_public_qualification import accepted_candidate, precise_metrics
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.train import speaker_compact as compact
from scripts.train import speaker_synthetic_data as generation
from scripts.train.speaker_compact_policy import fit_policy

ROOT,HERE,CACHE,PREPARED=compact.ROOT,compact.HERE,compact.CACHE,compact.PREPARED


def shared_release():
    spec=importlib.util.spec_from_file_location('scripts.train._compact_release_base',ROOT/'scripts/train/speaker_final_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE;module.CACHE=CACHE;module.PREPARED=PREPARED;module.training=compact.shared_trainer()
    return module


def quantization_gate(floating,quantized,graph_bytes):
    losses={kind:floating[kind]['raw_exact']-quantized[kind]['raw_exact'] for kind in ('human','synthetic')}
    extra=quantized['human']['raw_negative_errors']-floating['human']['raw_negative_errors']
    return {
        'pass':all(v<=1 for v in losses.values()) and extra<=1 and 0<graph_bytes<2*1024*1024,
        'exact_losses':losses,'extra_human_negative_errors':extra,'graph_bytes':graph_bytes,
        'maximum_graph_bytes':2*1024*1024,'physical_pi_measured':False}


def verify_result_bindings(receipt):
    generation._verify(receipt['bindings'])
    generation._verify(read_json(HERE/'release-protocol-before-evaluation.json')['bindings'])


def complete_archive(base,output,bundle):
    """Close the old publisher's runtime allowlist without changing its evidence."""
    with zipfile.ZipFile(base) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe or duplicate base archive members')
        entries={name:archive.read(name) for name in names}
    manifest=json.loads(entries.pop('MANIFEST.json'))
    if set(entries)!=set(manifest['files']) or any(hashlib.sha256(data).hexdigest()!=manifest['files'][name]
                                                 for name,data in entries.items()):
        raise ValueError('Base Pi archive identity mismatch')
    for name in ('speaker_compact.py','speaker_compact_benchmark.py','speaker_config.py'):
        entries['scripts/runtime/'+name]=(ROOT/'scripts/runtime'/name).read_bytes()
    old=b'scripts.runtime.pcv_pi_benchmark'
    if entries['pi_test.py'].count(old)!=1:raise ValueError('Unexpected Pi bootstrap benchmark import')
    entries['pi_test.py']=entries['pi_test.py'].replace(old,b'scripts.runtime.speaker_compact_benchmark')
    policy=(Path(bundle)/'recall-policy.json').read_bytes()
    entries['models/default/recall-policy.json']=policy
    deployment=json.loads(entries['models/default/deployment.json'])
    deployment['files']['recall-policy.json']=hashlib.sha256(policy).hexdigest()
    entries['models/default/deployment.json']=(json.dumps(deployment,indent=2)+'\n').encode()
    entries['config.toml']=b'''[inference]
engine = "onnxruntime"
mode = "demo"
threads = 1
intent_threshold = 0.95
slot_threshold = 0.90

[actions]
enabled = false
'''
    entries['HANDOFF.md']=b'''# Compact Pi4 endpoint

Run `python3 pi_test.py --verify-only` first. The bootstrap supports offline
ARM64 CPython 3.12 setup using the included locked wheels. See its `--help`.
Run `python3 pi_test.py --output results/demo --mode demo` for bundled replay,
or `--mode qualified` to preserve the strict qualified-policy behavior.
Default mode is demo. Intent thresholds come from the graph-bound
models/default/recall-policy.json, selected on original human dev/calibration.
The .95 config value documents the fixed baseline; it does not override this
versioned per-intent policy. Slots/targets retain .90 and actions are disabled.
Demo acceptance does not grant qualification. Replay is a small smoke subset;
physical Pi, microphone, wake, owner and overlap performance remain unmeasured.
See docs/COMPACT-RECALL.md and evidence/ for methods, limits and full results.
'''
    manifest['files']={name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    manifest['pi_runtime']='Compact CPU ONNX endpoint; explicit demo/qualified modes; actions disabled'
    manifest.update(decision_mode='demo',demo_acceptance_enabled=True,
                    qualified_acceptance_enabled=manifest.get('acceptance_enabled',False))
    entries['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,10,16,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;archive.writestr(info,data)
    return dict(path=str(output),sha256=file_digest(output),bytes=output.stat().st_size,
                base_package_sha256=file_digest(base),members=len(entries),decision_mode='demo',
                demo_acceptance_enabled=True,qualified_acceptance_enabled=manifest['qualified_acceptance_enabled'])


def grouped_metrics(rows,key):
    return {kind:precise_metrics((r for r in rows if r['synthetic']==(kind=='synthetic')),key)
            for kind in ('human','synthetic')}


def wording_results(records,source_rows,definitions):
    """Exact publisher-wording evidence; missing recordings remain explicit."""
    source={r['id']:r for r in source_rows};predictions={r['id']:r for r in records}
    if len(source)!=len(source_rows) or len(predictions)!=len(records):
        raise ValueError('Duplicate wording-analysis identities')
    for record in records:
        if record['id'] not in source or canonical(record['expected'])!=canonical(source[record['id']]['command']):
            raise ValueError('Wording reference identity mismatch')
    by_text={}
    for row in source_rows:
        text=normalize_text(row.get('publisher_metadata',{}).get('transcript',''))
        by_text.setdefault(text,[]).append(row)
    result=[]
    for definition in definitions:
        text=definition.get('spoken_text',definition.get('text',''))
        matches=by_text.get(normalize_text(text),[])
        item=dict(id=definition['id'],verbatim=text,expected=definition['command'])
        for kind in ('human','synthetic'):
            rows=[r for r in matches if r['kind']==kind]
            valid=[r for r in rows if canonical(r['command'])==canonical(definition['command'])]
            scored=[predictions[r['id']] for r in valid if r['id'] in predictions]
            cell=dict(n=len(scored),untested=not scored,declared_speakers=len({r['speaker'] for r in scored}),
                      reference_conflicts=len(rows)-len(valid),unscored_matching_references=len(valid)-len(scored))
            for name,key in (('raw','candidate'),('fixed_demo','application_command'),
                             ('relaxed_demo','relaxed_command'),('qualified','qualified_command')):
                accepted=sum(r[key] is not None for r in scored)
                correct=sum(r[key] is not None and canonical(r[key])==canonical(r['expected']) for r in scored)
                cell[name]=dict(correct=correct,wrong=accepted-correct,rejected=len(scored)-accepted)
            item[kind]=cell
        result.append(item)
    return result


def add_decisions(records,recall,qualified):
    for record in records:
        event=dict(output={'accepted':False,'reason':'capture_fault' if record.get('capture_fault') else 'unqualified_model'},
                   diagnostic_candidate=record)
        record['relaxed_application']=apply_policy(event,recall)
        record['relaxed_command']=record['relaxed_application']['command']
        record['qualified_command']=accepted_candidate(record,qualified) if qualified['enabled'] and not record.get('capture_fault') else None


def evaluate():
    began=time.perf_counter();base=shared_release();training=base.training;cfg=compact.freeze()
    selection=read_json(HERE/'selection-before-calibration.json')
    generation._verify(selection['bindings']);generation._verify(cfg['installed_metadata'])
    output=CACHE/'export'
    if output.exists():raise FileExistsError('Preserve previous compact export/evaluation evidence')
    schema=ROOT/'experiments/pcv-human-slots-v1/output.schema.json'
    custom=ROOT/'experiments/pcv-release-train-v1/custom-slots.json'
    optionb=ROOT/'experiments/pcv-release-train-v1/baseline.json'
    views=base.frozen_evaluation_views(PREPARED)
    files=[HERE/'selection-before-calibration.json',HERE/'protocol.json',HERE/'references.json',Path(__file__),schema,custom,optionb,
        ROOT/'scripts/train/speaker_final_release.py',ROOT/'scripts/train/speaker_compact_policy.py',
        ROOT/'scripts/train/pcv_context_int8.py',ROOT/'scripts/runtime/pcv_final_runtime.py',
        ROOT/'scripts/runtime/speaker_compact.py',ROOT/'scripts/runtime/speaker_compact_benchmark.py',
        ROOT/'scripts/runtime/speaker_confidence.py',ROOT/'scripts/runtime/pcv_pi_benchmark.py',
        ROOT/'scripts/runtime/speaker_config.py',ROOT/'scripts/pcv_final_delivery.py',
        ROOT/'scripts/test/pcv_public_qualification.py',ROOT/'data/sources/speaker-final-pi-assets.source.json',
        ROOT/'experiments/phrase-contract-v1/configs/runtime-cp312-aarch64/dependencies.json',*views.values()]
    release_protocol=dict(experiment_id=cfg['experiment_id'],selected_before_calibration=True,
        fixed_application=training.APPLICATION,recall_policy=cfg['recall_policy'],
        calibration='Original human only; no synthetic replacement; policy frozen before test',
        published_test='One locked regression replay; previously examined corpus',
        augmentation_at_evaluation=False,maximum_int8_graph_bytes=2*1024*1024,
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in files})
    write_new(HERE/'release-protocol-before-evaluation.json',release_protocol)
    _,dev,norm,entries=training.load_inputs();arrays=training.load_arrays(dev,norm)
    payload=torch.load(ROOT/selection['selected']['checkpoint'],map_location='cpu',weights_only=True)
    if payload['model_config']!=compact.ARCHITECTURE:raise ValueError('Selected architecture changed')
    model=training.PooledContextVCM(**compact.ARCHITECTURE)
    if compact.check_parameters(model)!=cfg['parameters']:raise ValueError('Selected parameter count changed')
    model.load_state_dict(payload['weights']);model.eval();torch.set_num_threads(1)
    from scripts.train.pcv_context_int8 import export_quantized
    exported=export_quantized(model,output)
    from scripts.pcv_parallel import ParallelScorer
    with ParallelScorer(entries,workers=6,return_scores=True) as scorer:
        development={};predictions={}
        for kind in ('fp32','int8'):
            engine=base.ONNXModel(output/f'model.{kind}.onnx')
            records=training.score_model(engine,dev,arrays,entries,scorer,device='cpu',batch_size=1)
            predictions[kind]=records;development[kind]=training.summarize(records)
            write_new(CACHE/f'development-{kind}.json',records)
        fidelity=quantization_gate(development['fp32'],development['int8'],exported['int8_bytes'])
        write_new(HERE/'export-development.json',dict(export=exported,development=development,fidelity=fidelity,
            exact_candidate_agreement=sum(canonical(a['candidate'])==canonical(b['candidate'])
                for a,b in zip(predictions['fp32'],predictions['int8'],strict=True)),development_rows=len(dev)))
        if not fidelity['pass']:raise ValueError('Compact INT8 development/resource gate failed; no calibration/test read')
        bundle=CACHE/'bundle';bundle.mkdir(exist_ok=False)
        for source,name in ((output/'model.int8.onnx','model.onnx'),(PREPARED/'catalog.json','catalog.json'),
                (PREPARED/'normalization.json','normalization.json'),(schema,'output.schema.json'),(custom,'custom-slots.json')):
            shutil.copyfile(source,bundle/name)
        binding={key:file_digest(bundle/name) for key,name in BINDINGS.items()}
        cal_source=read_jsonl(views['human/calibration']);cal_rows,cal_excluded=base.partition_evaluation(cal_source)
        cal_rows,cal_arrays=base.external_inputs(cal_rows,norm,'calibration')
        cal=base.score_external(base.ONNXModel(bundle/'model.onnx'),cal_rows,cal_arrays,entries,scorer)
        write_new(CACHE/'calibration-predictions.json',cal)
        write_new(CACHE/'calibration-excluded.jsonl',cal_excluded,jsonl=True)
        policy=base.calibrate_human(cal);policy['binding']=binding;write_new(bundle/'policy.json',policy)
        recall=fit_policy([r for r in predictions['int8'] if r['synthetic'] is False],cal,binding)
        write_new(bundle/'recall-policy.json',recall)
        from vcm.catalog import PHONES
        recipe=dict(schema_version=3,model_config=compact.ARCHITECTURE,architecture_class='PooledContextVCM',
            parameters=cfg['parameters'],maximum_parameters=compact.MAX_PARAMETERS,
            intent_labels=['OOV',*INTENTS],acoustic_labels=['<blank>',*PHONES],state_shapes=[],heads=['ctc','intent','target'],
            binding=binding,policy_enabled=policy['enabled'],recall_policy_sha256=file_digest(bundle/'recall-policy.json'),
            input='float32 [1,40,input_frames]',output='raw [1,ceil(input_frames/2),classes]',
            feature_hop_ms=10,output_hop_ms=20,inference_mode='complete_utterance',teacher_shipped=False,installed=False,
            status='qualified_candidate' if policy['enabled'] else 'unqualified_demo_candidate')
        write_new(bundle/'recipe.json',recipe);verify_bundle(bundle,diagnostic=True);load_bound_policy(bundle,recipe)
        write_new(HERE/'selection-before-test.json',dict(model_binding=binding,policy_sha256=file_digest(bundle/'policy.json'),
            recall_policy_sha256=file_digest(bundle/'recall-policy.json'),test_used_for_selection=False,
            selection_sha256=file_digest(HERE/'selection-before-calibration.json'),
            export_sha256=file_digest(HERE/'export-development.json'),protocol_sha256=file_digest(HERE/'release-protocol-before-evaluation.json')))
        write_new(HERE/'recall-policy.json',recall)
        results={};engine=base.ONNXModel(bundle/'model.onnx')
        for name in ('test','holdout'):
            source=read_jsonl(views['evaluation/'+name]);eligible,excluded=base.partition_evaluation(source)
            prepared,values=base.external_inputs(eligible,norm,name)
            scored=base.score_external(engine,prepared,values,entries,scorer);add_decisions(scored,recall,policy)
            write_new(CACHE/f'{name}-predictions.json',scored)
            write_new(CACHE/f'{name}-excluded.jsonl',excluded,jsonl=True)
            results[name]=dict(published_rows=len(source),metric_eligible_rows=len(scored),excluded_rows=len(excluded),
                excluded_references='Preserved ambiguous/publisher-ineligible rows, never UNKNOWN',
                raw=grouped_metrics(scored,'candidate'),fixed_demo=grouped_metrics(scored,'application_command'),
                relaxed_demo=grouped_metrics(scored,'relaxed_command'),qualified=grouped_metrics(scored,'qualified_command'),
                optionb_wordings=wording_results(scored,source,read_json(optionb)),
                capture_fault_rows=sum(bool(r.get('capture_fault')) for r in scored))
    shutil.copytree(ROOT/'.cache/speaker-final-refinement-v1/wake',bundle/'wake')
    generation._verify(release_protocol['bindings']);generation._verify(cfg['installed_metadata'])
    bound=[HERE/'selection-before-test.json',HERE/'release-protocol-before-evaluation.json',
        bundle/'model.onnx',bundle/'recipe.json',bundle/'policy.json',bundle/'recall-policy.json',
        CACHE/'development-fp32.json',CACHE/'development-int8.json',CACHE/'calibration-predictions.json',
        CACHE/'test-predictions.json',CACHE/'holdout-predictions.json',
        CACHE/'calibration-excluded.jsonl',CACHE/'test-excluded.jsonl',CACHE/'holdout-excluded.jsonl']
    receipt=dict(experiment_id=cfg['experiment_id'],status='compact_recall_evaluated',qualified=policy['enabled'],
        model_binding=binding,policy=policy,recall_policy=recall,parameters=cfg['parameters'],
        graph_bytes=(bundle/'model.onnx').stat().st_size,catalog_entries=len(entries),
        test=results['test'],holdout=results['holdout'],evaluation_seconds=time.perf_counter()-began,
        physical_pi_measured=False,live_microphone_measured=False,installed=False,published_test_used_for_selection=False,
        limitations=['One seed; width and six supervision exclusions change together; no causal attribution',
            'Synthetic intended-text labels/voices lack independent listening validation',
            'Human dev/calibration constraints cannot guarantee unseen-human precision or UNKNOWN behavior',
            'Missing human calibration classes retain .95 and cannot become qualified through synthesis',
            'Previously examined published regression, no new blind unseen-human claim',
            'Wake/VAD does not establish ownership or overlap recovery'],
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in bound})
    write_new(HERE/'results.json',receipt);return receipt


def package_pi():
    receipt=read_json(HERE/'results.json');verify_result_bindings(receipt);base=shared_release()
    source=base.sample_rows(read_jsonl(PREPARED/'views/evaluation/test.jsonl'))
    sample_dir=CACHE/'pi-sample-source';sample_dir.mkdir(exist_ok=False)
    manifest=sample_dir/'manifest.jsonl';write_new(manifest,source,jsonl=True)
    card=ROOT/'experiments/pcv-source-refresh-v1/source-card.txt'
    write_new(sample_dir/'protocol.json',dict(bindings={str(card.relative_to(ROOT)):file_digest(card)},
        selection='Original licensed SLURP test rows; bounded package smoke, not new qualification'))
    ledger=read_json(ROOT/'data/sources/ai231-primary.source.json')
    allowed={str(p['path']).replace('\\','/'):p['sha256'] for p in ledger['files']};files=[]
    for name in sorted({r['parquet'] for r in source}):
        digest=file_digest(ROOT/name)
        if allowed.get(name)!=digest:raise ValueError('Sample shard differs from pinned source')
        files.append(dict(path=name,sha256=digest))
    write_new(sample_dir/'acquisition.json',dict(repository='airimonda/ai231-me2-voice-commands',
        revision='25111444af3adff7588d86ab27c304bb080895cf',
        protocol_sha256=file_digest(sample_dir/'protocol.json'),files=files))
    from scripts.pcv_final_delivery import package
    result=package(CACHE/'bundle',CACHE/'pi-base.zip',diagnostic=True,
        sample_manifest=manifest,source_receipt=sample_dir/'acquisition.json',source_card=card,
        extra_files={'docs/COMPACT-RECALL.md':ROOT/'docs/EXPERIMENT-speaker-compact-recall-v1.md',
            'evidence/results.json':HERE/'results.json','evidence/training-protocol.json':HERE/'protocol.json',
            'evidence/recall-policy.json':HERE/'recall-policy.json','evidence/references.json':HERE/'references.json'})
    archive=ROOT/'dist/speaker-compact-recall-v1/speaker-pi4-compact.zip'
    result.update(complete_archive(CACHE/'pi-base.zip',archive,CACHE/'bundle'))
    write_new(archive.with_suffix('.receipt.json'),result);write_new(HERE/'package.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['evaluate','package'])
    args=parser.parse_args();result=evaluate() if args.stage=='evaluate' else package_pi()
    print(json.dumps(dict(stage=args.stage,status='complete',qualified=result.get('qualified',False))))


if __name__=='__main__':main()
