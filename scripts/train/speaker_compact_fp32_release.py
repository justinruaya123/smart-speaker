"""Preserved-weight FP32 fallback after the compact INT8 development stop."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import time

from scripts.runtime.pcv_final_runtime import BINDINGS, verify_bundle
from scripts.runtime.pcv_schema import INTENTS
from scripts.runtime.speaker_compact import load_bound_policy
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.train import speaker_compact as compact
from scripts.train import speaker_synthetic_data as generation
from scripts.train.speaker_compact_policy import fit_policy

ROOT=compact.ROOT
HERE=compact.HERE/'fp32-fallback'
CACHE=compact.CACHE/'fp32-fallback'


def fallback_gate(export_receipt,graph_bytes,parameters):
    if export_receipt['fidelity']['pass'] is not False:
        raise ValueError('FP32 fallback requires a preserved failed INT8 attempt')
    if export_receipt['export']['fp32_numerical_parity_pass'] is not True:
        raise ValueError('FP32 numerical export fidelity must pass')
    if not 0<graph_bytes<4*1024*1024 or not 0<parameters<=compact.MAX_PARAMETERS:
        raise ValueError('FP32 fallback resource/parameter ceiling exceeded')
    return dict(precision='fp32',graph_bytes=graph_bytes,parameters=parameters,
        maximum_graph_bytes=4*1024*1024,maximum_parameters=compact.MAX_PARAMETERS,
        int8_fidelity_passed=False,retrained=False,architecture_changed=False,
        reason='Retain numerically verified full-precision graph; do not loosen the failed INT8 guard')


def release_context():
    spec=importlib.util.spec_from_file_location('scripts.train._compact_fp32_base',Path(__file__).with_name('speaker_compact_release.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE;module.CACHE=CACHE
    return module


def evaluate():
    began=time.perf_counter();release=release_context();base=release.shared_release()
    cfg=compact.freeze();selection=read_json(compact.HERE/'selection-before-calibration.json')
    generation._verify(selection['bindings']);generation._verify(cfg['installed_metadata'])
    exported=read_json(compact.HERE/'export-development.json')
    graph=compact.CACHE/'export/model.fp32.onnx'
    frozen=exported['export']['files']['model.fp32.onnx']
    if file_digest(graph)!=frozen['sha256']:raise ValueError('Selected FP32 graph identity mismatch')
    gate=fallback_gate(exported,graph.stat().st_size,cfg['parameters'])
    old_protocol=read_json(compact.HERE/'release-protocol-before-evaluation.json')
    generation._verify(old_protocol['bindings'])
    if HERE.exists() or CACHE.exists():raise FileExistsError('Preserve prior fallback evaluation evidence')
    HERE.mkdir(parents=True);CACHE.mkdir(parents=True)
    stop=dict(status='int8_development_gate_failed',fidelity=exported['fidelity'],
        calibration_read=False,test_read=False,holdout_read=False,
        next_stage='Separately frozen FP32 fallback; same weights and parameter ceiling; larger graph allowance')
    write_new(compact.HERE/'export-stop.json',stop)
    fallback=copy.deepcopy(cfg);fallback.update(experiment_id='speaker-compact-recall-fp32-v1',
        parent_experiment=cfg['experiment_id'],training_repeated=False,selected_export=gate,
        original_int8_graph_limit=cfg['maximum_int8_graph_bytes'])
    fallback.pop('maximum_int8_graph_bytes')
    write_new(HERE/'protocol.json',fallback)
    shutil.copyfile(compact.HERE/'selection-before-calibration.json',HERE/'selection-before-calibration.json')
    shutil.copyfile(compact.HERE/'references.json',HERE/'references.json')
    shutil.copyfile(compact.CACHE/'development-fp32.json',CACHE/'development-fp32.json')
    views=base.frozen_evaluation_views(compact.PREPARED)
    files=[Path(__file__),HERE/'protocol.json',compact.HERE/'export-development.json',
        compact.HERE/'release-protocol-before-evaluation.json',compact.HERE/'export-stop.json',
        HERE/'selection-before-calibration.json',graph,CACHE/'development-fp32.json',*views.values()]
    protocol=dict(experiment_id=fallback['experiment_id'],export_selection=gate,
        new_training=False,new_export_search=False,published_test_used=False,
        recall_policy=cfg['recall_policy'],strict_qualification_unchanged=True,
        bindings={**old_protocol['bindings'],**{str(p.relative_to(ROOT)):file_digest(p) for p in files}})
    write_new(HERE/'release-protocol-before-evaluation.json',protocol)
    _,_,norm,entries=base.training.load_inputs()
    schema=ROOT/'experiments/pcv-human-slots-v1/output.schema.json'
    custom=ROOT/'experiments/pcv-release-train-v1/custom-slots.json'
    bundle=CACHE/'bundle';bundle.mkdir()
    for source,name in ((graph,'model.onnx'),(compact.PREPARED/'catalog.json','catalog.json'),
            (compact.PREPARED/'normalization.json','normalization.json'),(schema,'output.schema.json'),(custom,'custom-slots.json')):
        shutil.copyfile(source,bundle/name)
    binding={key:file_digest(bundle/name) for key,name in BINDINGS.items()}
    from scripts.pcv_parallel import ParallelScorer
    with ParallelScorer(entries,workers=6,return_scores=True) as scorer:
        engine=base.ONNXModel(bundle/'model.onnx')
        source=read_jsonl(views['human/calibration']);cal_rows,cal_excluded=base.partition_evaluation(source)
        cal_rows,values=base.external_inputs(cal_rows,norm,'calibration')
        cal=base.score_external(engine,cal_rows,values,entries,scorer)
        write_new(CACHE/'calibration-predictions.json',cal)
        write_new(CACHE/'calibration-excluded.jsonl',cal_excluded,jsonl=True)
        policy=base.calibrate_human(cal);policy['binding']=binding;write_new(bundle/'policy.json',policy)
        dev=read_json(CACHE/'development-fp32.json')
        recall=fit_policy([r for r in dev if r['synthetic'] is False],cal,binding)
        write_new(bundle/'recall-policy.json',recall);write_new(HERE/'recall-policy.json',recall)
        from vcm.catalog import PHONES
        recipe=dict(schema_version=3,model_config=compact.ARCHITECTURE,architecture_class='PooledContextVCM',
            parameters=cfg['parameters'],maximum_parameters=compact.MAX_PARAMETERS,precision='fp32',
            intent_labels=['OOV',*INTENTS],acoustic_labels=['<blank>',*PHONES],state_shapes=[],heads=['ctc','intent','target'],
            binding=binding,policy_enabled=policy['enabled'],recall_policy_sha256=file_digest(bundle/'recall-policy.json'),
            input='float32 [1,40,input_frames]',output='raw [1,ceil(input_frames/2),classes]',
            feature_hop_ms=10,output_hop_ms=20,inference_mode='complete_utterance',teacher_shipped=False,
            installed=False,status='unqualified_demo_candidate')
        write_new(bundle/'recipe.json',recipe);verify_bundle(bundle,diagnostic=True);load_bound_policy(bundle,recipe)
        write_new(HERE/'selection-before-test.json',dict(model_binding=binding,
            selected_export=gate,policy_sha256=file_digest(bundle/'policy.json'),
            recall_policy_sha256=file_digest(bundle/'recall-policy.json'),
            protocol_sha256=file_digest(HERE/'release-protocol-before-evaluation.json'),test_used_for_selection=False))
        results={}
        for name in ('test','holdout'):
            source=read_jsonl(views['evaluation/'+name]);eligible,excluded=base.partition_evaluation(source)
            prepared,values=base.external_inputs(eligible,norm,name)
            scored=base.score_external(engine,prepared,values,entries,scorer);release.add_decisions(scored,recall,policy)
            write_new(CACHE/f'{name}-predictions.json',scored);write_new(CACHE/f'{name}-excluded.jsonl',excluded,jsonl=True)
            results[name]=dict(published_rows=len(source),metric_eligible_rows=len(scored),excluded_rows=len(excluded),
                excluded_references='Preserved ambiguous/publisher-ineligible rows, never UNKNOWN',
                raw=release.grouped_metrics(scored,'candidate'),fixed_demo=release.grouped_metrics(scored,'application_command'),
                relaxed_demo=release.grouped_metrics(scored,'relaxed_command'),qualified=release.grouped_metrics(scored,'qualified_command'),
                optionb_wordings=release.wording_results(scored,source,read_json(ROOT/'experiments/pcv-release-train-v1/baseline.json')),
                capture_fault_rows=sum(bool(r.get('capture_fault')) for r in scored))
    shutil.copytree(ROOT/'.cache/speaker-final-refinement-v1/wake',bundle/'wake')
    generation._verify(protocol['bindings']);generation._verify(cfg['installed_metadata'])
    paths=[HERE/'selection-before-test.json',HERE/'release-protocol-before-evaluation.json',
        bundle/'model.onnx',bundle/'recipe.json',bundle/'policy.json',bundle/'recall-policy.json',
        CACHE/'development-fp32.json',CACHE/'calibration-predictions.json',CACHE/'test-predictions.json',CACHE/'holdout-predictions.json',
        CACHE/'calibration-excluded.jsonl',CACHE/'test-excluded.jsonl',CACHE/'holdout-excluded.jsonl']
    receipt=dict(experiment_id=fallback['experiment_id'],status='compact_recall_fp32_fallback_evaluated',
        precision='fp32',int8_fidelity_passed=False,export_selection=gate,model_binding=binding,
        policy=policy,recall_policy=recall,qualified=policy['enabled'],parameters=cfg['parameters'],
        graph_bytes=graph.stat().st_size,catalog_entries=len(entries),test=results['test'],holdout=results['holdout'],
        evaluation_seconds=time.perf_counter()-began,physical_pi_measured=False,live_microphone_measured=False,
        installed=False,published_test_used_for_selection=False,
        limitations=['Original INT8 development gate failed and remains failed; FP32 graph uses a separately declared four-MiB allowance',
            'One seed, previously examined regression data; no isolated architecture effect or unseen-human guarantee',
            'Threshold constraints on small original human fitting views do not guarantee unchanged errors on test',
            'Missing human calibration classes, listening validation, owner/overlap and physical Pi evidence remain unresolved',
            'Fallback wrapper added after the one independent review; guard tests and extracted execution validate it without a second review'],
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in paths})
    write_new(HERE/'results.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['evaluate','package'])
    args=p.parse_args();result=evaluate() if args.stage=='evaluate' else release_context().package_pi()
    print(json.dumps(dict(stage=args.stage,status='complete',qualified=result.get('qualified',False))))


if __name__=='__main__':main()
