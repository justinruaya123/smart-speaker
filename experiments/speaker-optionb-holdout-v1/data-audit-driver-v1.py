"""Execute post-replay dataset/label/loader diagnosis without model adaptation."""
import ast,copy,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.runtime.pcv_schema import baseline_cases,canonical,INTENTS
from scripts.train.speaker_human_data import read_rows,write_json,file_sha256,human_supervision
from scripts.test.speaker_optionb_holdout import wording_audit,annotation_review,load_admitted_features,cue_match
from scripts.test.pcv_public_qualification import precise_metrics
D=ROOT/'experiments/speaker-optionb-holdout-v1'
R=ROOT/'.cache/speaker-optionb-holdout-v1/replay-01'
A=ROOT/'.cache/speaker-optionb-holdout-v1/audit-01';A.mkdir(parents=True,exist_ok=False)
H=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
P=ROOT/'.cache/submission/reproductions/frozen-replay-compact-01/.cache/reproduction/prepared'
S=ROOT/'.cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source'
protocol=json.loads((D/'selection-before-holdout.json').read_text())
for path,digest in protocol['bindings'].items():assert file_sha256(ROOT/path)==digest,path
full=read_rows(R/'full-original.jsonl');compact=read_rows(R/'compact-v7-original.jsonl')
parents={r['id']:r for r in full}
rows=[r for split in ('train','dev','calibration','test') for r in read_rows(H/(split+'.jsonl'))]+full
assert len({r['id'] for r in rows})==len(rows)
groups=defaultdict(list)
for row in rows:groups[(row['location']['file'],row['location']['sha256'])].append(row)
metadata={};bound_shards={}
for (name,digest),members in groups.items():
    assert all(r['location']['repository']=='airimonda/ai231-me2-voice-commands' and r['location']['revision']=='25111444af3adff7588d86ab27c304bb080895cf' for r in members)
    path=(S/name).resolve();assert path.is_relative_to(S.resolve()) and file_sha256(path)==digest
    q=pq.ParquetFile(path);items=q.read(columns=[c for c in q.schema_arrow.names if c not in ('audio','whisper_transcript')]).to_pylist()
    bound_shards[path.relative_to(ROOT).as_posix()]=digest
    for row in members:
        index=row['location']['row'];assert type(index) is int and index>=0
        assert row['id'].endswith('/'+Path(name).stem+f'/{index:06d}')
        meta=items[index];assert bool(meta['is_synthetic']) is row['synthetic']
        metadata[row['id']]=meta
        row.update(publisher_transcript=meta['transcript'],publisher_variation=meta['variation'],
            publisher_source=meta['source'],transcript_source=meta['transcript_source'])
paths=[ROOT/'scripts/test/speaker_optionb_holdout.py',ROOT/'scripts/test/test_speaker_optionb_holdout.py',
    H/'optimizer-smoke/normalization-human.json',D/'selection-before-holdout.json',D/'holdout-results.json']
write_json(D/'audit-inputs.json',dict(format='optionb-data-audit-inputs-v1',post_holdout=True,
    purpose='Diagnose data; cannot select a new holdout model/policy or change frozen references',
    bindings={**bound_shards,**{p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths}},
    driver_sha256=file_sha256(__file__),training=False,independent_listening=False))
definitions=baseline_cases();wording=wording_audit(rows,definitions,full)
assert len(wording)==93
write_json(D/'optionb-wording-data.json',dict(cases=wording,scope='Admitted human fitting/evaluation views plus all 196 holdout rows. Synthetic training/test views are unprovided, not assumed empty.',
    case_count=len(wording),independent_listening=False,source_records=len(rows)))
with (D/'optionb-wording-data.csv').open('x',encoding='utf-8',newline='') as stream:
    fields=['id','verbatim','command','stratum',*wording[0]['strata']['human/train']]
    w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
    for case in wording:
        for stratum,values in case['strata'].items():w.writerow(dict(id=case['id'],verbatim=case['verbatim'],command=canonical(case['command']),stratum=stratum,**values))
reviews=[annotation_review(r,metadata[r['id']]) for r in rows]
review_flags=Counter(flag for r in reviews for flag in r['flags'])
with (A/'label-rechecks.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
    for row in reviews:stream.write(canonical(row)+'\n')
train=[r for r in rows if r['split']=='train'];reserved=[r for r in rows if r['split']!='train' and r['kind']=='human']
norm=json.loads((H/'optimizer-smoke/normalization-human.json').read_text())
arrays,loaded=load_admitted_features(train,norm,reserved=reserved)
write_json(D/'verified-human-loader.json',loaded)
# Exercise the old correction and basename-loading paths on read-only/fixture inputs.
from scripts.train.speaker_targeted_train import apply_corrections_and_filter,load_label_corrections,TargetedTrainingPools
legacy=apply_corrections_and_filter([r for r in read_rows(P/'train.jsonl') if r['kind']=='human'],load_label_corrections(),is_train=True)
wrong_targets=[r for r in legacy if r['target_index']!=human_supervision(r)['target_index']]
wrong_intents=[r for r in legacy if r['intent_index']!=human_supervision(r)['intent_index']]
probe=copy.deepcopy(next(r for r in rows if r['split']=='dev'));probe['training_eligible']=True
unsafe_pool=TargetedTrainingPools([probe]);accepted_wrong_split=bool(unsafe_pool.groups)
fixture=A/'basename-fixture';(fixture/'true').mkdir(parents=True);(fixture/'fallback').mkdir()
np.save(fixture/'true/same.npy',np.ones((40,11),dtype=np.float32));np.save(fixture/'fallback/same.npy',np.full((40,11),9,dtype=np.float32))
parsed=ast.parse((ROOT/'scripts/train/speaker_targeted_train.py').read_text())
trainer=next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=='train_targeted')
loader=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='_load_one')
env=dict(Path=Path,np=np,features_dir=fixture/'fallback',mean=np.zeros((40,1),dtype=np.float32),std=np.ones((40,1),dtype=np.float32))
exec(compile(ast.Module(body=[loader],type_ignores=[]),'existing_nested_loader_fixture','exec'),env)
_,legacy_values=env['_load_one'](('fixture',str(fixture/'true/same.npy')))
assert np.all(legacy_values==9)
compat=dict(legacy_human_rows=len(legacy),legacy_target_mismatches=len(wrong_targets),legacy_intent_mismatches=len(wrong_intents),
    target_mismatches_by_intent=dict(Counter(r['expected']['intent'] for r in wrong_targets)),
    old_pool_accepts_contrived_dev_row_if_eligibility_is_flipped=accepted_wrong_split,
    basename_fixture=dict(requested_mean=1.,actually_loaded_mean=float(legacy_values.mean()),hash_checked=False),
    interpretation='Existing implementation probes/constructed software fixtures, not proof of actual split contamination or wrong-file loading in the frozen full model. Historical code is unchanged.')
write_json(D/'loader-label-findings.json',compat)
# Distinguish correct batch padding from real silence seen by the frontend/model.
import torch
from scripts.models.pcv_release_pooling import PooledContextVCM
from vcm.frontend import features
old_threads=torch.get_num_threads();torch.set_num_threads(2);torch.manual_seed(53)
model=PooledContextVCM().eval()
short=torch.randn(1,40,37);batch=torch.zeros(2,40,91);batch[0,:,:37]=short[0];batch[1]=torch.randn(40,91)
with torch.inference_mode():
    alone=model(short,torch.tensor([37]));padded=model(batch,torch.tensor([37,91]))
errors=[float(torch.max(torch.abs(a[0,:,:19]-b[0,:,:19]))) for a,b in zip(alone[:3],padded[:3])]
assert max(errors)<1e-5
model_norm=json.loads((ROOT/'submission/models/full/normalization.json').read_text())
silence=features(np.zeros(16000,dtype=np.float32),model_norm['mean'],model_norm['std'])
preprocess=dict(batch_padding_fixture=dict(max_errors_by_head=errors,pass_=max(errors)<1e-5,weights='random-init software fixture, not a recognition improvement'),
    actual_silence_normalized_mean=float(silence.mean()),actual_silence_normalized_min=float(silence.min()),
    actual_silence_normalized_max=float(silence.max()),training_mask_fill=0.,
    distinction='Batch padding is masked and uses normalized zero; actual PCM silence becomes nonzero floor features and shifts absolute positions. No preprocessing change was applied.',
    original_augmentation='SpecAugment only, no waveform prefix/suffix jitter/noise/reverb in the frozen full-model protocol')
write_json(D/'preprocessing-findings.json',preprocess)
del model;torch.set_num_threads(old_threads)
# Text metamorphics are rule checks, never acoustic tests or newly recorded data.
cue_plan=json.loads((D/'optionb-cue-plan.json').read_text())
benign=[];reject=[]
for case,definition in zip(cue_plan['cases'],definitions,strict=True):
    assert case['id']==definition['id']
    benign.extend(cue_match(t,definition) for t in case['benign_text_probe'])
    reject.extend(not cue_match(t,definition) for t in case['reject_context_probe'])
assert all(benign) and all(reject)
metamorphic=dict(benign_text_passed=sum(benign),benign_text_n=len(benign),reject_context_passed=sum(reject),reject_context_n=len(reject),
    acoustic_word_injection_tested=False,new_human_recordings=0,
    limitation='No generated audio or transcripts reach runtime; these only validate annotation/collection cue contrasts.')
write_json(D/'cue-rule-tests.json',metamorphic)
# Aggregate wording and observed natural courtesy/wording subsets without oracle restriction.
coverage={}
for split in ('train','dev','calibration','test','holdout'):
    key='human/'+split
    coverage[key]=dict(exact_cases=sum(c['strata'][key]['exact_n']>0 for c in wording),
        cue_nonexact_cases=sum(c['strata'][key]['cue_nonexact_n']>0 for c in wording),
        publisher_bucket_cases=sum(c['strata'][key]['publisher_bucket_n']>0 for c in wording),total_cases=len(wording))
word_groups=defaultdict(list)
by_command=defaultdict(list)
for definition in definitions:by_command[canonical(definition['command'])].append(definition)
for row in full:
    if row['kind']!='human' or not row['metric_eligible'] or row['expected'] is None:continue
    cases=by_command[canonical(row['expected'])];text=metadata[row['id']]['transcript']
    from scripts.train.pcv_updated_prepare import spoken
    exact=any(spoken(text)==spoken(d['spoken_text']) for d in cases)
    cue=any(cue_match(text,d) for d in cases)
    group='exact_optionb' if exact else 'courtesy_function_word_variant' if cue else 'other_wording_or_extension'
    word_groups[group].append(row)
word_metrics={g:dict(n=len(v),raw=precise_metrics(v,'candidate')['overall'],fixed=precise_metrics(v,'application_command')['overall']) for g,v in word_groups.items()}
# Explicitly attribute observed padding changes; no crop/threshold/model is chosen.
paired={}
for model in ('full','compact-v7'):
    original={r['id']:r for r in read_rows(R/(model+'-original.jsonl'))}
    stress=read_rows(R/(model+'-silence-stress.jsonl'));per_variant={}
    for variant in protocol['stress']['variants']:
        members=[r for r in stress if r['variant']==variant and r['expected'] is not None]
        lost=gained=0;reasons=Counter();intent_probability_deltas=[]
        for row in members:
            parent=original[row['parent_id']];before=parent['application_command']==parent['expected'];after=row['application_command']==row['expected']
            if before and not after:lost+=1;reasons[row['application_reason']]+=1
            if after and not before:gained+=1
            if parent['candidate']==row['candidate'] and row['candidate'] is not None:
                intent_probability_deltas.append(row['diagnostic_candidate']['intent_probability']-parent['diagnostic_candidate']['intent_probability'])
        per_variant[variant]=dict(lost_correct_accepts=lost,gained_correct_accepts=gained,lost_reasons=dict(reasons),
            same_candidate_confidence_change_mean=float(np.mean(intent_probability_deltas)) if intent_probability_deltas else None)
    paired[model]=per_variant
write_json(D/'paired-padding-findings.json',paired)
result=dict(format='optionb-data-label-loader-audit-v1',wording_coverage=coverage,loader=loaded,label_flag_counts=dict(review_flags),
    label_flags_by_split={s:dict(Counter(flag for r in reviews if r['split']==s for flag in r['flags'])) for s in ('train','dev','calibration','test','holdout')},
    labels_changed=0,independently_listened=0,new_admissions=0,holdout_original_wording_groups=word_metrics,
    text_metamorphics=metamorphic,loader_findings=compat,preprocessing=preprocess,
    recheck_records=len(reviews),post_holdout_diagnostic=True,model_or_policy_changed=False)
write_json(D/'data-audit-results.json',result)
for path,digest in protocol['bindings'].items():assert file_sha256(ROOT/path)==digest,path
for path,digest in json.loads((D/'audit-inputs.json').read_text())['bindings'].items():assert file_sha256(ROOT/path)==digest,path
print(json.dumps(dict(coverage=coverage,labels=review_flags,loader=loaded,compatibility=compat,
    preprocessing=preprocess,metamorphic=metamorphic,natural_wording_groups=word_metrics),indent=2))
