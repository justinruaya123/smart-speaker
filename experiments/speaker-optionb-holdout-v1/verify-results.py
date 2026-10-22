"""Verify the completed immutable holdout/data audit from the repository root."""
import argparse,ast,csv,json,math,re,sys
from collections import Counter
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.runtime.pcv_schema import INTENTS,baseline_cases,canonical
from scripts.train.speaker_human_data import read_rows,file_sha256,split_leakage,write_json
from scripts.test.pcv_public_qualification import precise_metrics
from scripts.train.pcv_release_prepare import REJECTED_PROPOSALS
D=ROOT/'experiments/speaker-optionb-holdout-v1'
R=ROOT/'.cache/speaker-optionb-holdout-v1/replay-01'
H=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
protocol=json.loads((D/'selection-before-holdout.json').read_text())
for path,digest in protocol['bindings'].items():assert file_sha256(ROOT/path)==digest,path
assert len(protocol['bindings'])==78
assert file_sha256(D/'replay-driver.py')==protocol['driver_sha256']
receipt=json.loads((R/'receipt.json').read_text());assert receipt['protocol_sha256']==file_sha256(D/'selection-before-holdout.json')
old=json.loads((D/'audit-inputs.json').read_text())
archive=ROOT/'.cache/speaker-optionb-holdout-v1/audit-01/implementation'
for path,digest in old['bindings'].items():
    candidate=archive/Path(path).name if path in ('scripts/test/speaker_optionb_holdout.py','scripts/test/test_speaker_optionb_holdout.py') else ROOT/path
    assert file_sha256(candidate)==digest,path
for path,digest in json.loads((D/'frozen-label-recheck-inputs.json').read_text())['bindings'].items():assert file_sha256(ROOT/path)==digest,path
# Only the explicit annotation-policy adapter changed after the first audit.
def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
old_functions=functions(archive/'speaker_optionb_holdout.py');current_functions=functions(ROOT/'scripts/test/speaker_optionb_holdout.py')
for name in old_functions:
    if name!='annotation_review':assert old_functions[name]==current_functions[name],name
out=json.loads((D/'holdout-results.json').read_text());counts={}
original_ids=None
for model,info in receipt['outputs'].items():
    originals=read_rows(ROOT/info['original']['path']);stress=read_rows(ROOT/info['stress']['path'])
    for kind,records in (('original',originals),('stress',stress)):
        spec=info[kind];assert file_sha256(ROOT/spec['path'])==spec['sha256'];assert len(records)==spec['rows'];assert len({r['id'] for r in records})==len(records)
    assert len(originals)==196 and len(stress)==282
    if original_ids is None:original_ids={r['id'] for r in originals}
    else:assert original_ids=={r['id'] for r in originals}
    by_id={r['id']:r for r in originals}
    human=[r for r in originals if r['kind']=='human'];synthetic=[r for r in originals if r['kind']=='synthetic']
    assert len(human)==96 and len(synthetic)==100
    assert sum(r['metric_eligible'] for r in human)==94
    assert all(r['synthetic'] is False for r in human) and all(r['synthetic'] is True for r in synthetic)
    assert sum(not r['metric_eligible'] for r in human)==2
    for population,members in (('human',human),('synthetic',synthetic)):
        eligible=[r for r in members if r['metric_eligible']]
        for decision,key in (('raw','candidate'),('fixed','application_command'),('qualified','qualified_command')):
            actual=precise_metrics(eligible,key);assert actual==out['models'][model][population][decision]
            assert sum(x['reference_n'] for x in actual['per_intent'].values())==actual['overall']['positive_n']
            assert sum(x['complete_correct'] for x in actual['per_intent'].values())==actual['overall']['exact']
        assert out['models'][model][population]['qualified']['overall']['accepted']==0
    human_metric=out['models'][model]['human']['comprehensive']
    assert human_metric['overall']['positive_n']==83 and human_metric['overall']['negative_n']==11 and human_metric['overall']['ambiguous_n']==2
    assert sum(x['tp'] for k,x in human_metric['per_intent'].items() if k!='UNKNOWN')==human_metric['overall']['accepted_correct']
    assert sum(x['fp'] for k,x in human_metric['per_intent'].items() if k!='UNKNOWN')==human_metric['overall']['wrong_positive_accepts']+human_metric['overall']['unknown_false_accepts']
    for key,matrix in human_metric['confusion'].items():
        expected=human_metric['overall']['intent_head_scored_rows'] if key=='intent_head' else 94
        assert sum(sum(c.values()) for c in matrix.values())==expected
    eligible_ids={r['id'] for r in human if r['metric_eligible']}
    assert Counter(r['parent_id'] for r in stress)==Counter({rid:3 for rid in eligible_ids})
    for row in stress:
        parent=by_id[row['parent_id']]
        assert row['kind']=='constructed_silence_stress' and row['synthetic'] is None and row['training_eligible'] is False
        assert row['expected']==parent['expected'] and row['parent_pcm_sha256']==parent['pcm_sha256']
        assert row['audio'] is None and row['sha256'] is None and row['features'] is None and row['features_sha256'] is None
        assert row['padding_samples']==protocol['stress']['variants'][row['variant']]
    for variant in protocol['stress']['variants']:
        subset=[r for r in stress if r['variant']==variant];assert len(subset)==94
        assert precise_metrics(subset,'candidate')==out['models'][model]['stress'][variant]['raw']
        assert precise_metrics(subset,'application_command')==out['models'][model]['stress'][variant]['fixed']
    counts[model]=dict(original=len(originals),stress=len(stress),human_positive=83,human_negative=11,ambiguous=2,synthetic_positive=100)
for spec in out['outputs'].values():
    for identity in spec.values():assert file_sha256(ROOT/identity['path'])==identity['sha256']
wording=json.loads((D/'optionb-wording-data.json').read_text())['cases'];definitions=baseline_cases()
assert len(wording)==len(definitions)==93
for a,b in zip(wording,definitions,strict=True):assert a['id']==b['id'] and a['command']==b['command']
with (D/'optionb-wording-data.csv').open(newline='',encoding='utf-8') as stream:csv_wording=list(csv.DictReader(stream))
assert len(csv_wording)==930 and len({(r['id'],r['stratum']) for r in csv_wording})==930
with (D/'holdout-per-intent.csv').open(newline='',encoding='utf-8') as stream:metric_rows=list(csv.DictReader(stream))
assert len(metric_rows)==54 and len({(r['view'],r['intent']) for r in metric_rows})==54
for row in metric_rows:
    tp,fp,fn=map(int,(row['tp'],row['fp'],row['fn']));assert int(row['reference_n'])==tp+fn and int(row['predicted_n'])==tp+fp
    denom=2*tp+fp+fn
    assert (not row['f1']) if not denom else math.isclose(float(row['f1']),2*tp/denom,abs_tol=1e-12)
plan=json.loads((D/'human-keyword-collection-plan.json').read_text());assert len(plan['cases'])==93 and plan['recordings_created']==0
assert all(x['new_human_recordings']==0 and x['keyword_phone_targets_created'] is False for x in plan['cases'])
labels=json.loads((D/'frozen-label-recheck.json').read_text())
assert labels['rows']==3416 and labels['semantic_disagreements']==[] and labels['labels_changed']==0 and labels['new_train_admissions']==0
approval=json.loads((ROOT/'experiments/pcv-release-train-v1/annotation-approval.json').read_text())
proposal=json.loads((ROOT/'experiments/pcv-release-train-v1/training-label-review.json').read_text())
projected={r['text']:r['expected'] for r in proposal['supported_correction_evidence'] if r['text'] not in REJECTED_PROPOSALS}
assert projected==approval['supported'] and proposal['text_to_ambiguous_corrections']==approval['ambiguous']
assert len(projected)==110
loader=json.loads((D/'verified-human-loader.json').read_text());assert loader['rows']==1889 and loader['full_ctc_rows']==1863 and loader['semantic_only_rows']==26 and loader['target_rows']==373
human_views=[r for s in ('train','dev','calibration','test','holdout') for r in read_rows(H/(s+'.jsonl'))]
assert not split_leakage(human_views)
assert set(json.loads((H/'optimizer-smoke/normalization-human.json').read_text())['source_ids'])=={r['id'] for r in human_views if r['split']=='train'}
report=ROOT/'docs/EXPERIMENT-speaker-optionb-holdout-v1.md';body=report.read_text(encoding='utf-8')
for link in re.findall(r'\]\(([^)#]+)(?:#[^)]*)?\)',body):assert (report.parent/link).resolve().is_file(),link
assert '108 focused tests pass' in body
log=ROOT/'.cache/speaker-optionb-holdout-v1/verification-01/focused-tests.log'
assert '108 passed' in log.read_text(encoding='utf-8')
files={p.relative_to(ROOT).as_posix():file_sha256(p) for p in D.iterdir() if p.is_file() and p.name!='verification.json'}
files[report.relative_to(ROOT).as_posix()]=file_sha256(report)
files['scripts/test/speaker_optionb_holdout.py']=file_sha256(ROOT/'scripts/test/speaker_optionb_holdout.py')
files['scripts/test/test_speaker_optionb_holdout.py']=file_sha256(ROOT/'scripts/test/test_speaker_optionb_holdout.py')
result=dict(format='optionb-holdout-final-verification-v1',pass_=True,protected_inputs=len(protocol['bindings']),
    counts=counts,csv_rows=dict(optionb_wording=930,positive_intent_metrics=54),optionb_cases=93,
    focused_tests=108,test_log_path=log.relative_to(ROOT).as_posix(),test_log_sha256=file_sha256(log),
    verified_frozen_label_semantic_disagreements=0,earlier_base_rule_evidence_preserved=True,
    artifacts=files,training=False,model_or_policy_changed=False,independent_listening=False,blind_holdout=False)
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-receipt',action='store_true');args=parser.parse_args()
if args.write_receipt:write_json(D/'verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='artifacts'},indent=2))
