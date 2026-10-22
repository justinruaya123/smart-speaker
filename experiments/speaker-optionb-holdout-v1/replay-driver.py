"""Freeze unchanged policies, then replay original holdout and labelled silence probes."""
import importlib.metadata, json, re, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from threadpoolctl import threadpool_limits
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.runtime.pcv_schema import INTENTS,baseline_cases,canonical
from scripts.runtime.pcv_final_runtime import Runtime,verify_bundle,BINDINGS
from scripts.runtime.speaker_compact import CompactRuntime
from scripts.runtime.speaker_confidence import application_decision
from scripts.train.pcv_updated_prepare import spoken
from scripts.train.speaker_human_data import read_rows,write_json,file_sha256,split_leakage,audio_diagnostics
from scripts.test.speaker_human_scorecard import prediction_metrics,wilson_interval,speaker_bootstrap
from scripts.test.pcv_public_qualification import precise_metrics
from scripts.test.speaker_demo_suite import pcm_digest
D=ROOT/'experiments/speaker-optionb-holdout-v1'
R=ROOT/'.cache/speaker-optionb-holdout-v1/replay-01'
P=ROOT/'.cache/submission/reproductions/frozen-replay-compact-01/.cache/reproduction/prepared'
H=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
SOURCE=ROOT/'.cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source'
old=json.loads((ROOT/'experiments/speaker-human-scorecard-v1/protocol.json').read_text())
for path,digest in old['bindings'].items():assert file_sha256(ROOT/path)==digest,path
rows=read_rows(P/'holdout.jsonl');human=read_rows(H/'holdout.jsonl')
assert len(rows)==196 and len({r['id'] for r in rows})==196
assert all(r['split']=='holdout' and r['training_eligible'] is False for r in rows)
human_by_id={r['id']:r for r in human}
assert len(human_by_id)==94
for row in rows:
    if row['id'] in human_by_id:
        peer=human_by_id[row['id']]
        assert all(row[k]==peer[k] for k in ('expected','pcm_sha256','sha256','speaker','synthetic','metric_eligible'))
        row.update({k:v for k,v in peer.items() if k not in row})
assert {r['id'] for r in rows if r['synthetic'] is False and r['metric_eligible']}==set(human_by_id)
all_human=[r for s in ('train','dev','calibration','test','holdout') for r in read_rows(H/(s+'.jsonl'))]
assert not split_leakage(all_human)
definitions=baseline_cases()
assert definitions==json.loads((ROOT/'experiments/pcv-release-train-v1/baseline.json').read_text())
shards={}
for row in rows:
    loc=row['location'];key=loc['file']
    assert loc['repository']=='airimonda/ai231-me2-voice-commands'
    assert loc['revision']=='25111444af3adff7588d86ab27c304bb080895cf'
    assert re.fullmatch(r'data/holdout-\d{5}-of-\d{5}\.parquet',key)
    assert type(loc['row']) is int and loc['row']>=0
    path=SOURCE/key
    if key not in shards:
        assert file_sha256(path)==loc['sha256']
        q=pq.ParquetFile(path)
        shards[key]=q.read(columns=[c for c in q.schema_arrow.names if c not in ('audio','whisper_transcript')]).to_pylist()
    meta=shards[key][loc['row']]
    assert row['id'].endswith('/'+Path(key).stem+f"/{loc['row']:06d}")
    assert type(meta['is_synthetic']) in (int,bool) and bool(meta['is_synthetic']) is row['synthetic']
    row.update(publisher_metadata=meta,publisher_source=meta['source'],text=spoken(meta['transcript']),
        publisher_transcript=meta['transcript'],transcript_source=meta['transcript_source'],
        publisher_variation=meta['variation'],listening_status='not_independently_reviewed')
paths={ROOT/p for p in old['bindings']}
paths.update(P/(s+'.jsonl') for s in ('train','dev','calibration','test','holdout'))
paths.update(H/(s+'.jsonl') for s in ('train','dev','calibration','test','holdout'))
paths.update(SOURCE/k for k in shards)
paths.update(ROOT/p for p in ('scripts/test/speaker_human_scorecard.py','scripts/test/test_speaker_human_scorecard.py',
    'scripts/train/speaker_final_train.py','scripts/train/speaker_targeted_train.py','scripts/train/pcv_updated_prepare.py',
    'scripts/models/pcv_contextual.py','scripts/models/pcv_release_pooling.py','experiments/pcv-release-train-v1/baseline.json',
    'submission/reproduction-full/recipe.json','experiments/speaker-human-scorecard-v1/protocol.json'))
for folder in (ROOT/'submission/models/full',ROOT/'experiments/speaker-targeted-v7'):
    verify_bundle(folder,diagnostic=True)
    paths.update(folder/name for name in (*BINDINGS.values(),'recipe.json','policy.json') )
D.mkdir(exist_ok=False);R.mkdir(parents=True,exist_ok=False)
stopwords={'the','an','some','my','me','to','for','at','it','is','what','what\'s'}
cues=[]
for definition in definitions:
    tokens=spoken(definition['spoken_text']).split()
    excluded=stopwords|({'a'} if definition['command']['intent']!='ALARM' else set())
    cues.append(dict(id=definition['id'],verbatim=definition['text'],spoken=definition['spoken_text'],expected=definition['command'],
        candidate_cue_tokens=[token for token in tokens if token not in excluded],required_slots=definition['command']['slots'],
        benign_text_probe=['please '+definition['spoken_text'],'could you '+definition['spoken_text']+' for me',definition['spoken_text']+' please'],
        reject_context_probe=['do not '+definition['spoken_text'],'the phrase '+definition['spoken_text']+' is an example not a request'],
        evidence='Offline annotation/collection plan only; no acoustic labels, examples, runtime parser or acceptance override created'))
write_json(D/'optionb-cue-plan.json',dict(cases=cues,case_count=len(cues),new_human_recordings=0,
    contract='Preserve full extension catalog and all typed slots; negation, changed values/targets and extra actions are not harmless padding.'))
protocol=dict(format='speaker-optionb-holdout-v1',date='2026-10-22',
    authorization='User requested holdout testing, OptionB keyword/wording resilience and dataset/label/loading/pretraining audit',
    requirements=['ME-01','ME-02','ME-03','ME-04','ME-08'],
    baseline_git_head=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip(),
    dirty_baseline=True,bindings={p.relative_to(ROOT).as_posix():file_sha256(p) for p in sorted(paths)},
    cue_plan_sha256=file_sha256(D/'optionb-cue-plan.json'),driver_sha256=file_sha256(__file__),
    primary_model='submission/models/full',comparison_model='experiments/speaker-targeted-v7',
    policy=dict(mode='demo',intent_threshold=.95,slot_threshold=.90),
    secondary_policy='Existing bound v7 relaxed policy only; no new threshold fitting',
    holdout_rows=len(rows),strata=dict(human_positive=83,human_true_negative=11,human_ambiguous=2,synthetic_positive=100),
    stress=dict(variants={'silence_prefix_1s':[16000,0],'silence_suffix_1s':[0,16000],'silence_both_1s':[16000,16000]},
        scope='All 94 eligible original-human holdout parents, including negatives; constructed deterministic waveform tests, not independent human recordings',
        no_clipping_or_trimming=True,overlong='Preserve as capture fault, never remove from the denominator'),
    evaluation='Already-examined published holdout regression; not blind generalization or a newly trained human-only model',
    thresholds_or_model_selected_on_holdout=False,training=False,synthetic_training_injection=False,installation=False,
    action_providers=False,live_microphone=False,physical_pi=False,
    allowed_paths=['scripts/test/speaker_optionb_holdout.py','scripts/test/test_speaker_optionb_holdout.py',
        'experiments/speaker-optionb-holdout-v1/','.cache/speaker-optionb-holdout-v1/','docs/EXPERIMENT-speaker-optionb-holdout-v1.md','EXPERIMENTS.md'])
write_json(D/'selection-before-holdout.json',protocol)
write_json(R/'metadata-before-replay.json',dict(rows=len(rows),shards=list(shards),audio_column_read=False,
    original_human=len(human),declared_speakers=len({r['speaker'] for r in human}),
    original_human_positive_speakers=len({r['speaker'] for r in human if r['expected'] is not None})))
summary={};outputs={};diagnostics=[]

def evaluate_record(runtime,row,pcm,rate,variant='original'):
    started=time.perf_counter();event=runtime.recognize(pcm,rate)
    app=application_decision(event,protocol['policy']);diag=event.get('diagnostic_candidate') or {}
    record={**row,'variant':variant,'candidate':diag.get('candidate'),'application_command':app['command'],
        'application_reason':app['reason'],'diagnostic_candidate':diag,'model_sha256':event['model_sha256'],
        'qualified_command':(event.get('qualified_output') or event['output']).get('command'),
        'timing':event['timing'],'wall_ms':1000*(time.perf_counter()-started)}
    if hasattr(runtime,'recall_policy'):
        record.update(relaxed_application_command=event['application']['command'],relaxed_application_reason=event['application']['reason'])
    return record

def measure(records):
    result={}
    for kind in ('human','synthetic'):
        members=[r for r in records if r['kind']==kind]
        eligible=[r for r in members if r['metric_eligible']]
        result[kind]=dict(total=len(members),ambiguous=len(members)-len(eligible),
            raw=precise_metrics(eligible,'candidate'),fixed=precise_metrics(eligible,'application_command'),
            qualified=precise_metrics(eligible,'qualified_command'))
        if kind=='human':
            result[kind]['comprehensive']=prediction_metrics(members)
            result[kind]['speaker_cluster']=speaker_bootstrap(members)
    return result
with threadpool_limits(limits=1,user_api='blas'):
    for model,folder in (('full',ROOT/'submission/models/full'),('compact-v7',ROOT/'experiments/speaker-targeted-v7')):
        runtime=Runtime(folder,diagnostic=True,threads=2) if model=='full' else CompactRuntime(folder,threads=2,mode='demo')
        original=[];stress=[]
        dest=R/(model+'-original.jsonl');stress_dest=R/(model+'-silence-stress.jsonl')
        with dest.open('x',encoding='utf-8',newline='\n') as stream,stress_dest.open('x',encoding='utf-8',newline='\n') as ss:
            for i,row in enumerate(rows,1):
                assert file_sha256(row['audio'])==row['sha256']
                pcm,rate=sf.read(row['audio'],dtype='float32')
                assert pcm_digest(pcm,rate)==row['pcm_sha256']
                result=evaluate_record(runtime,row,pcm,rate);original.append(result);stream.write(canonical(result)+'\n')
                if model=='full':diagnostics.append(dict(id=row['id'],**audio_diagnostics(pcm,rate)))
                if row['synthetic'] is False and row['metric_eligible']:
                    for variant,(before,after) in protocol['stress']['variants'].items():
                        modified=np.pad(pcm,(before,after))
                        derived={**row,'id':row['id']+'#'+variant,'parent_id':row['id'],'kind':'constructed_silence_stress','synthetic':None,
                            'parent_kind':'human','parent_audio_sha256':row['sha256'],'parent_pcm_sha256':row['pcm_sha256'],
                            'audio':None,'sha256':None,'pcm_sha256':pcm_digest(modified,rate),
                            'features':None,'features_sha256':None,
                            'evidence_kind':'artificial_silence_padding_not_new_human_data','training_eligible':False,
                            'padding_samples':[before,after]}
                        trial=evaluate_record(runtime,derived,modified,rate,variant);stress.append(trial);ss.write(canonical(trial)+'\n')
                if i%32==0:stream.flush();ss.flush();print(model,i,'/',len(rows),flush=True)
        assert len(original)==196 and len(stress)==94*len(protocol['stress']['variants'])
        original_by_id={r['id']:r for r in original}
        summary[model]=measure(original)
        summary[model]['stress']={v:dict(raw=precise_metrics([r for r in stress if r['variant']==v],'candidate'),
            fixed=precise_metrics([r for r in stress if r['variant']==v],'application_command'),
            paired_raw_changes=sum(r['candidate']!=original_by_id[r['parent_id']]['candidate'] for r in stress if r['variant']==v),
            paired_acceptance_changes=sum(r['application_command']!=original_by_id[r['parent_id']]['application_command'] for r in stress if r['variant']==v))
            for v in protocol['stress']['variants']}
        if model=='compact-v7':
            relaxed=[{**r,'application_command':r['relaxed_application_command'],'application_reason':r['relaxed_application_reason']} for r in original]
            summary[model]['existing_relaxed']=measure(relaxed)
        outputs[model]=dict(original=dict(path=dest.relative_to(ROOT).as_posix(),sha256=file_sha256(dest),rows=len(original)),
            stress=dict(path=stress_dest.relative_to(ROOT).as_posix(),sha256=file_sha256(stress_dest),rows=len(stress)))
        del runtime
with (R/'audio-audit.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
    for row in diagnostics:stream.write(canonical(row)+'\n')
for path,digest in protocol['bindings'].items():assert file_sha256(ROOT/path)==digest,path
write_json(D/'holdout-results.json',dict(protocol_sha256=file_sha256(D/'selection-before-holdout.json'),models=summary,
    outputs=outputs,blind=False,new_training=False,independent_listening=False,qualified=False,
    note='Original-human, synthetic, ambiguous and constructed stress evidence are separate. No holdout-driven changes.'))
write_json(R/'receipt.json',dict(protocol_sha256=file_sha256(D/'selection-before-holdout.json'),driver_sha256=file_sha256(__file__),
    outputs=outputs,environment={p:importlib.metadata.version(p) for p in ('numpy','onnxruntime','soundfile','threadpoolctl','pyarrow')},
    cpu_threads=2,blas_threads=1,protected_inputs_verified=len(protocol['bindings']),training=False,actions=False))
print(json.dumps({m:{kind:v[kind]['fixed']['overall'] for kind in ('human','synthetic')} for m,v in summary.items()},indent=2))
