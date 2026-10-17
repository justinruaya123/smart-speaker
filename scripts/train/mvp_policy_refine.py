"""Paired command-only calibration ablation with unchanged V2 neural weights."""
import argparse
import json
from pathlib import Path
import shutil
from scripts.runtime.mvp_static import Runtime,decide,digest,fit_policy
from scripts.train.mvp_static_train import ROOT,read_rows,write
from scripts.train import mvp_static_release as release

HERE=ROOT/'experiments/vcm-mvp-hey-iris-v3'
CACHE=ROOT/'.cache/vcm-mvp-hey-iris-v3'
RUN='policy-20261017'

def inputs():
    cfg=json.loads((HERE/'protocol.json').read_text())
    parent=ROOT/'experiments/vcm-mvp-hey-iris-v2/protocol.json'
    if digest(parent)!=cfg['parent_protocol_sha256']:raise ValueError('Changed parent protocol')
    base=json.loads(parent.read_text())
    for path,expected in {**base['inputs'],**base['benchmark_bindings']}.items():
        if digest(ROOT/path)!=expected:raise ValueError('Changed calibration input: '+path)
    return cfg,base

def counts(records,policy,catalog):
    measured=[]
    for row in records:
        decision=decide(row['logits'],catalog,policy) if row['logits'] is not None else {'command':None}
        measured.append({**row,'accepted_index':decision['class_index'] if decision['command'] else None})
    return {s:release.summarize([r for r in measured if r['split']==s],catalog) for s in ('dev','calibration')}

def fit(name):
    cfg,base=inputs();parent=ROOT/cfg['bundles'][name]['path']
    if digest(parent/'bundle.json')!=cfg['bundles'][name]['bundle_sha256']:raise ValueError('Changed parent bundle')
    if digest(parent/'development-predictions.jsonl')!=cfg['bundles'][name]['calibration_predictions_sha256']:raise ValueError('Changed fitting predictions')
    Runtime(parent,engine='ncnn')
    folder=CACHE/'runs'/RUN/name;folder.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((parent/'catalog.json').read_text());original=json.loads((parent/'policy.json').read_text())
    records=read_rows(parent/'development-predictions.jsonl')
    fitting=[r for r in records if r['expected_index'] is not None and r['logits'] is not None]
    policies={'original_v2':original}
    for arm,group in [('observed_intent','intent'),('observed_class_min2','class')]:
        policy=fit_policy(fitting,original['trained_classes'],catalog=catalog,risk=cfg['known_error_limits'],
            threshold_strategy='observed',grouping=group,min_correct=cfg['minimum_class_correct_positives'])
        policy.update(enabled=True,qualified=False,qualification_reason='Empirical calibration; no human/Pi/owner qualification',
            threshold_strategy='observed',grouping=group)
        policies[arm]=policy
    summaries={arm:counts(records,policy,catalog) for arm,policy in policies.items()}
    ranked=[]
    for index,(arm,summary) in enumerate(summaries.items()):
        human=[summary[s]['strata']['human'] for s in ('dev','calibration')]
        synthetic=[summary[s]['strata']['synthetic'] for s in ('dev','calibration')]
        key=(sum(r['correct_accepts'] for r in human),sum(r['correct_accepts'] for r in synthetic),-index)
        ranked.append((key,arm))
        print('ARM',name,arm,json.dumps({s:summary[s]['strata'] for s in summary}),flush=True)
    key,selected=max(ranked)
    for arm,policy in policies.items():write(folder/(arm+'-policy.json'),policy)
    write(folder/'calibration-ablation.json',dict(protocol_sha256=digest(HERE/'protocol.json'),
        selected=selected,selection_key=list(key),summaries=summaries,test_used_for_selection=False))
    target=folder/'export';target.mkdir()
    for file in ('model.onnx','catalog.json','normalization.json'):
        shutil.copyfile(parent/file,target/file)
    shutil.copytree(parent/'ncnn',target/'ncnn',ignore=shutil.ignore_patterns('*.pt','*.py','*.onnx','*.pnnx.*','*.debug.*'))
    write(target/'policy.json',policies[selected])
    meta=json.loads((parent/'bundle.json').read_text())
    meta['files']['policy.json']=digest(target/'policy.json');meta['calibration_experiment']='vcm-mvp-hey-iris-v3'
    write(target/'bundle.json',meta)
    shutil.copyfile(parent.parent/'best.pt',folder/'best.pt')
    write(folder/'selection-before-test.json',dict(bundle_path='export',protocol_sha256=digest(HERE/'protocol.json'),
        catalog_sha256=digest(target/'catalog.json'),bundle_sha256=digest(target/'bundle.json'),
        policy_sha256=digest(target/'policy.json'),checkpoint_sha256=digest(folder/'best.pt'),
        model_sha256=digest(target/'model.onnx'),test_used_for_selection=False,physical_pi_measured=False,
        selected_arm=selected,parent_selection_sha256=digest(parent.parent/'selection-before-test.json')))
    write(folder/'engine-selection-before-test.json',dict(source_lock_sha256=digest(folder/'selection-before-test.json'),
        files={str(p.relative_to(target)):digest(p) for p in (target/'ncnn').iterdir() if p.name in
            ('engine.json','model.ncnn.param','model.ncnn.bin','position-frequency.json')},
        parity=json.loads((target/'ncnn/engine.json').read_text())['parity']))
    view=CACHE/'views'
    if not view.exists():view.symlink_to(ROOT/'.cache/vcm-mvp-hey-iris-v2/views',target_is_directory=True)
    print('LOCKED',name,selected,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit','personal','official','evaluate'])
    parser.add_argument('--model',choices=['compact','full'],required=True)
    args=parser.parse_args()
    if args.stage=='fit':return fit(args.model)
    cfg,base=inputs();release.HERE,release.CACHE=HERE,CACHE
    return {'personal':release.personal,'official':release.official,'evaluate':release.evaluate}[args.stage](
        args.model,RUN,'ncnn',cfg=base)

if __name__=='__main__':main()
