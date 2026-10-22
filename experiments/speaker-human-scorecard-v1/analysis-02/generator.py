"""Build a complete read-only human-data/model scorecard from pinned inputs."""
import csv, json, sys, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
from scripts.train.speaker_human_data import LABELS,SPLITS,read_rows,write_json,file_sha256,split_leakage
from scripts.runtime.pcv_schema import INTENTS,baseline_cases,canonical
from scripts.train.pcv_updated_prepare import spoken
from scripts.test.speaker_human_scorecard import dataset_profile,prediction_metrics,speaker_bootstrap
E=ROOT/'experiments/speaker-human-scorecard-v1'
D=E/'analysis-02'
D.mkdir(exist_ok=False)
R=ROOT/'.cache/speaker-human-scorecard-v1/replay-01'
H=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
A=ROOT/'.cache/speaker-human-first-v1/audit-02'
protocol=json.loads((E/'protocol.json').read_text())
for name,digest in protocol['bindings'].items():assert file_sha256(ROOT/name)==digest,name
replay=json.loads((R/'receipt.json').read_text())
rows=[r for s in SPLITS for r in read_rows(H/(s+'.jsonl'))]
ids={r['id'] for r in rows}
assert len(ids)==len(rows) and not split_leakage(rows)
train=[r for r in rows if r['split']=='train']
dev=[r for r in rows if r['split']=='dev']
audio=[r for p in (A/'audio-audit.jsonl',H/'audio-audit.jsonl') for r in read_rows(p) if r['id'] in ids]
assert len({r['id'] for r in audio})==len(audio)
assert {r['id'] for r in audio}=={r['id'] for r in train+dev}
audio_by_id={r['id']:r for r in audio}
normalization_path=H/'optimizer-smoke/normalization-human.json'
norm=json.loads(normalization_path.read_text())
mean,std=np.asarray(norm['mean']),np.asarray(norm['std'])
feature_groups=defaultdict(list); feature_integrity=[]
for row in train+dev:
    assert file_sha256(row['audio'])==row['sha256'],row['id']
    assert file_sha256(row['features'])==row['features_sha256'],row['id']
    values=np.load(row['features'],allow_pickle=False,mmap_mode='r')
    samples=round(audio_by_id[row['id']]['duration_seconds']*16000)
    assert values.dtype==np.float32 and values.shape==(40,max(0,1+(samples-400)//160))
    assert np.isfinite(values).all()
    feature_groups[(row['split'],row.get('publisher_source','unknown'))].append(
        (values.astype(np.float64).mean(axis=1)-mean)/std)
    feature_integrity.append(dict(id=row['id'],split=row['split'],frames=values.shape[1],features_sha256=row['features_sha256']))
feature_shifts=[]
for (split,source),vectors in sorted(feature_groups.items()):
    z=np.mean(vectors,axis=0)
    feature_shifts.append(dict(split=split,source=source,n=len(vectors),mean_abs_bin_shift_z=float(np.mean(np.abs(z))),
        max_abs_bin_shift_z=float(np.max(np.abs(z))),mean_bin_shift_z=z.tolist()))
catalog=json.loads((ROOT/'submission/models/compact/catalog.json').read_text())['entries']
baseline=baseline_cases()
data=dataset_profile(rows,audio,baseline,catalog)
queue=read_rows(H/'review-queue.jsonl')
treatments=json.loads((ROOT/'experiments/speaker-human-recovery-v1/intent-fixes.json').read_text())['per_intent']
for label,entry in data['per_intent'].items():
    entry['training_review_rows']=sum(r['review_intent']==label and r['split']=='train' for r in queue)
    entry['review_rows']=sum(r['review_intent']==label for r in queue)
    entry['targeted_followup']=treatments[label]['followup']
    entry['priority']=treatments[label]['priority']
    entry['exact_baseline_wording']={}
    for split in ('train','dev'):
        subset=[r for r in rows if r['split']==split and (r['expected'] or {}).get('intent','UNKNOWN')==label]
        cases=[b for b in baseline if (b['command'] or {}).get('intent','UNKNOWN')==label]
        matched={b['id']:[r for r in subset if r.get('text')==spoken(b.get('spoken_text',b['text'])) and r['expected']==b['command']] for b in cases}
        entry['exact_baseline_wording'][split]=dict(covered=sum(bool(v) for v in matched.values()),total=len(cases),
            case_rows={k:len(v) for k,v in matched.items()})
    t=entry['splits']['train'];dv=entry['splits']['dev'];cal=entry['splits']['calibration']
    barriers=[]
    if not t['rows']:barriers.append('no_human_training_positive')
    if not dv['rows']:barriers.append('no_human_development_positive')
    if not cal['rows']:barriers.append('no_human_calibration_positive')
    if t['speakers']<5:barriers.append('fewer_than_five_declared_training_speakers')
    if 0<dv['rows']<10:barriers.append('development_denominator_below_ten')
    if (t['largest_speaker_fraction'] or 0)>.5:barriers.append('one_speaker_supplies_majority_of_training_rows')
    if entry['coverage']['baseline_commands']['train']['covered']<entry['coverage']['baseline_commands']['train']['total']:barriers.append('missing_baseline_typed_training_cells')
    if entry['coverage']['catalog_commands']['train']['covered']<entry['coverage']['catalog_commands']['train']['total']:barriers.append('missing_approved_catalog_typed_training_cells')
    if t['rows'] and not t['independently_reviewed_rows']:barriers.append('no_independently_listened_training_annotations')
    entry['evidence_barriers']=barriers
predictions={}; performance={}
for model,spec in replay['models'].items():
    path=ROOT/spec['path'];assert file_sha256(path)==spec['sha256']
    items=read_rows(path)
    assert len(items)==len(dev) and {r['id'] for r in items}=={r['id'] for r in dev}
    assert all(r['model_sha256']==spec['model_sha256'] for r in items)
    predictions[model]=items
    performance[model]=prediction_metrics(items)
compact=predictions['compact-v7-frozen']
relaxed=[{**r,'application_command':r['relaxed_application_command'],'application_reason':r['relaxed_application_reason']} for r in compact]
performance['compact-v7-existing-relaxed']=prediction_metrics(relaxed)
predictions['compact-v7-existing-relaxed']=relaxed
for model,metrics in performance.items():
    metrics['speaker_cluster_intervals']=speaker_bootstrap(predictions[model])
    for label in INTENTS:
        metrics['per_intent'][label]['speaker_cluster_intervals']=speaker_bootstrap(predictions[model],label=label)
    times=np.asarray([r['replay_wall_ms'] for r in predictions[model]])
    metrics['desktop_replay_timing_ms']=dict(n=len(times),median=float(np.median(times)),p95=float(np.percentile(times,95)),maximum=float(times.max()),
        scope='One pass on CPU, 2 ONNX threads/1 BLAS thread; includes frontend/decoder/policy, excludes file IO, live speech and endpoint waiting; not Pi timing.')
    metrics['qualified_accepts']=sum(r.get('qualified_command') is not None for r in predictions[model])
full_by_id={r['id']:r for r in predictions['full-shipping-frozen']}
paired=Counter()
for r in compact:
    if r['expected'] is None:continue
    a=full_by_id[r['id']]['application_command']==r['expected']
    b=r['application_command']==r['expected']
    paired['both_correct' if a and b else 'full_only_correct' if a else 'compact_only_correct' if b else 'neither_correct']+=1
assert sum(paired.values())==performance['full-shipping-frozen']['overall']['positive_n']
model_meta=dict(replay['models'])
full_recipe_path=ROOT/'submission/reproduction-full/recipe.json'
full_recipe=json.loads(full_recipe_path.read_text())
model_meta['full-shipping-frozen']={**model_meta['full-shipping-frozen'],'parameters':full_recipe['parameters'],
    'parameter_count_source':'submission/reproduction-full/recipe.json','parameter_count_source_sha256':file_sha256(full_recipe_path)}
result=dict(format='human-training-and-development-scorecard-v1',date='2026-10-22',analysis_revision=2,
    prior_analysis=dict(path=(E/'scorecard.json').relative_to(ROOT).as_posix(),sha256=file_sha256(E/'scorecard.json'),
        changes='Exclude rejection sentinels from typed-command counts; add head-only per-class F1 and explicit missing-head denominators. Same recordings, predictions and complete-command outcomes.'),
    protocol_sha256=file_sha256(E/'protocol.json'),replay_receipt_sha256=file_sha256(R/'receipt.json'),
    data=data,performance=performance,model_metadata=model_meta,paired_fixed_policy_outcomes=dict(paired),
    current_compact_bundle_attempt=replay['current_compact_bundle_attempt'],feature_shifts=feature_shifts,
    feature_integrity=dict(checked_rows=len(feature_integrity),splits=['train','dev'],failures=0,
        normalization_path=normalization_path.relative_to(ROOT).as_posix(),normalization_sha256=file_sha256(normalization_path),
        interpretation='Source mean offsets are diagnostics relative to the 1889-human candidate normalization, not proof of a causal preprocessing defect.'),
    review=dict(total=len(queue),by_split=dict(Counter(r['split'] for r in queue))),
    evaluation_limits=dict(human_only_new_fit=False,new_data_accuracy_measured=False,blind_test=False,
        physical_pi_measured=False,live_microphone_measured=False,peak_memory_measured=False,
        source_ablation_run=False,qualified_best_source=None,
        note='Do not infer causal source quality from development-source differences or select per-intent winning models to fabricate a hybrid. Existing models have synthetic-training exposure.'),
    implementation_bindings={str(p.relative_to(ROOT)):file_sha256(p) for p in
        [ROOT/'scripts/test/speaker_human_scorecard.py',ROOT/'scripts/test/test_speaker_human_scorecard.py']},
    generator_sha256=file_sha256(__file__))
write_json(D/'scorecard.json',result)

def csv_new(name,values):
    values=list(values)
    assert values,name
    fields=list(dict.fromkeys(key for value in values for key in value))
    with (D/name).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(values)
    return len(values)

data_rows=[]; source_rows=[]; coverage_rows=[]; collection_rows=[]
for label,entry in data['per_intent'].items():
    t=entry['splits']['train'];dv=entry['splits']['dev'];cal=entry['splits']['calibration'];c=entry['coverage']
    data_rows.append(dict(intent=label,train_rows=t['rows'],train_speakers=t['speakers'],speaker_effective_count=t['speaker_effective_count'],
        largest_speaker_fraction=t['largest_speaker_fraction'],full_ctc_rows=t['full_ctc_rows'],semantic_only_rows=t['semantic_only_rows'],
        unique_texts=t['unique_texts'],dev_rows=dv['rows'],dev_speakers=dv['speakers'],calibration_rows=cal['rows'],calibration_speakers=cal['speakers'],
        source_n=len(entry['training_sources']),source_depth_order='; '.join(s['source'] for s in entry['training_sources']),
        train_baseline_covered=c['baseline_commands']['train']['covered'],baseline_total=c['baseline_commands']['train']['total'],
        train_catalog_covered=c['catalog_commands']['train']['covered'],catalog_total=c['catalog_commands']['train']['total'],
        baseline_wordings_covered=entry['exact_baseline_wording']['train']['covered'],baseline_wordings_total=entry['exact_baseline_wording']['train']['total'],
        train_audio_seconds=t['duration']['total_seconds'],median_duration=t['duration']['median'],p95_duration=t['duration']['p95'],
        train_asr_transcript_rows=t['transcript_origins'].get('whisper',0),independently_reviewed=t['independently_reviewed_rows'],
        internal_pause_flags=t['flags'].get('internal_pause_endpoint_risk',0),training_review_rows=entry['training_review_rows'],
        barriers='; '.join(entry['evidence_barriers']),targeted_followup=entry['targeted_followup']))
    for source in entry['training_sources']:
        source_rows.append(dict(intent=label,source=source['source'],rows=source['rows'],speakers=source['speakers'],
            speaker_effective_count=source['speaker_effective_count'],largest_speaker_fraction=source['largest_speaker_fraction'],
            typed_commands=source['typed_commands'],unique_texts=source['unique_texts'],full_ctc_rows=source['full_ctc_rows'],
            semantic_only_rows=source['semantic_only_rows'],transcript_origins=canonical(source['transcript_origins']),
            audio_seconds=source['duration']['total_seconds'],qualified_best_source=False))
    for tier,key in (('baseline','collection_queue'),('catalog','catalog_collection_queue')):
        for item in c[key]:collection_rows.append(dict(intent=label,tier=tier,split=item['split'],command=canonical(item['command']),
            current_speakers=item['current_speakers'],target_speakers=item['target_speakers'],speaker_cell_deficit=item['minimum_additional_speakers']))
    for tier,definitions in (('baseline',baseline),('catalog',catalog)):
        commands={canonical(x['command']):x['command'] for x in definitions if x['command'] is not None and x['command']['intent']==label}
        for split in SPLITS:
            present=c['splits'][split]['commands']
            for signature,command in commands.items():
                cell=present.get(signature,dict(rows=0,speakers=0))
                coverage_rows.append(dict(intent=label,tier=tier,split=split,command=signature,rows=cell['rows'],speakers=cell['speakers']))
perf_rows=[];perf_source=[];slot_rows=[];confusion_rows=[]
for model,metrics in performance.items():
    for label,item in metrics['per_intent'].items():
        perf_rows.append(dict(model=model,intent=label,**{k:v for k,v in item.items() if not isinstance(v,(list,dict))},
            recall_wilson95=canonical(item.get('accepted_recall_wilson95')),
            precision_wilson95=canonical(item.get('accepted_precision_wilson95')),
            false_accept_wilson95=canonical(item.get('false_accept_wilson95')),
            speaker_bootstrap=canonical(item.get('speaker_cluster_intervals'))))
        for slot,values in item.get('slots',{}).items():slot_rows.append(dict(model=model,intent=label,slot=slot,**values))
    for source,values in metrics['per_source'].items():
        for label,cell in values['per_intent'].items():
            if cell['reference_n'] or cell.get('accepted_predictions'):
                perf_source.append(dict(model=model,source=source,intent=label,**{k:v for k,v in cell.items() if not isinstance(v,(list,dict))}))
    for kind,matrix in metrics['confusion'].items():
        for truth,counts in matrix.items():
            for prediction,count in counts.items():confusion_rows.append(dict(model=model,kind=kind,reference=truth,prediction=prediction,count=count))
counts={name:csv_new(name,items) for name,items in [
    ('data-per-intent.csv',data_rows),('sources-per-intent.csv',source_rows),('command-coverage.csv',coverage_rows),
    ('collection-priority.csv',collection_rows),('performance-per-intent.csv',perf_rows),('performance-per-source.csv',perf_source),
    ('slot-performance.csv',slot_rows),('confusion.csv',confusion_rows),
    ('source-feature-shift.csv',[{k:v for k,v in x.items() if k!='mean_bin_shift_z'} for x in feature_shifts]) ]}
assert len(data_rows)==len(LABELS)
assert len(perf_rows)==len(performance)*len(LABELS)
assert sum(x['train_rows'] for x in data_rows)==len(train)
assert sum(x['training_review_rows'] for x in data_rows)==sum(r['split']=='train' for r in queue)
for name,digest in protocol['bindings'].items():assert file_sha256(ROOT/name)==digest,name
write_json(D/'generation-receipt.json',dict(format='human-scorecard-artifact-receipt-v1',rows=counts,
    input_row_counts={s:sum(r['split']==s for r in rows) for s in SPLITS},feature_integrity_rows=len(feature_integrity),
    files={p.name:file_sha256(p) for p in D.iterdir() if p.is_file()}))
print(json.dumps(dict(csv_rows=counts,totals={s:{k:data['totals'][s][k] for k in ('rows','speakers','full_ctc_rows','semantic_only_rows','transcript_origins')} for s in SPLITS},
    performance={model:{k:metrics['overall'][k] for k in ('positive_n','negative_n','raw_correct','accepted_correct','accepted_precision','accepted_recall','accepted_f1','unknown_false_accepts')} for model,metrics in performance.items()},
    paired_fixed_policy=dict(paired)),indent=2))
