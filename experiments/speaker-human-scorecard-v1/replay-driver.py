"""Frozen-policy human development replay; no fitting, tuning or actions."""
import importlib.metadata, json, sys, time, subprocess, hashlib
from pathlib import Path
import soundfile as sf
from threadpoolctl import threadpool_limits
ROOT=Path('D:/School/AI231/.worktrees/me2/smart-speaker-fresh')
sys.path.insert(0,str(ROOT))
from scripts.train.speaker_human_data import file_sha256,write_json,read_rows
from scripts.runtime.pcv_final_runtime import Runtime,verify_bundle,BINDINGS
from scripts.runtime.speaker_compact import CompactRuntime
from scripts.runtime.speaker_confidence import application_decision
from scripts.runtime.pcv_schema import canonical
from scripts.test.speaker_demo_suite import pcm_digest
R=ROOT/'.cache/speaker-human-scorecard-v1/replay-01'
D=ROOT/'experiments/speaker-human-scorecard-v1'
P=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
old=json.loads((ROOT/'experiments/speaker-human-recovery-v1/protocol.json').read_text())
for name,digest in old['bindings'].items():
    assert file_sha256(ROOT/name)==digest,name
paths=[ROOT/name for name in old['bindings']]
paths += [P/(s+'.jsonl') for s in ('train','dev','calibration','test','holdout')]
paths += [P/'recovery.json',P/'audio-audit.jsonl',ROOT/'.cache/speaker-human-first-v1/audit-02/audio-audit.jsonl',
    ROOT/'experiments/speaker-human-recovery-v1/results.json',ROOT/'experiments/speaker-human-recovery-v1/intent-fixes.json',
    ROOT/'scripts/runtime/pcv_final_runtime.py',ROOT/'scripts/runtime/speaker_compact.py',ROOT/'scripts/runtime/speaker_confidence.py']
v7=ROOT/'experiments/speaker-targeted-v7'
paths += [v7/name for name in (*BINDINGS.values(),'recipe.json','policy.json','recall-policy.json')]
paths=sorted(set(paths))
protocol=dict(format='speaker-human-scorecard-v1',date='2026-10-22',
    user_request='Continue and report best available training data per intent with comprehensive metrics',
    requirements=['ME-01','ME-02','ME-03','ME-04','ME-08'],
    baseline_git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),
    baseline_dirty=True,
    bindings={p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths},
    driver_sha256=file_sha256(__file__),
    allowed_paths=['scripts/test/speaker_human_scorecard.py','scripts/test/test_speaker_human_scorecard.py',
        'experiments/speaker-human-scorecard-v1/','docs/EXPERIMENT-speaker-human-scorecard-v1.md','EXPERIMENTS.md','docs/DATASETS.md'],
    models=['full-shipping-frozen','compact-v7-frozen'],
    comparison_scope='Descriptive unchanged-model development diagnostics, not a source ablation, model selection, new human-only fit or blind qualification.',
    human_rows='All 256 admitted original-human development references, including 63 finite-scope negatives; no test/holdout audio or calibration fitting.',
    primary_policy=dict(mode='demo',intent_threshold=.95,slot_threshold=.90),
    compact_existing_relaxed_policy='Secondary labelled view only; no threshold changes and no comparison against full fixed as if matched.',
    independent_best_source_claim=False,source_ranking='Descriptive data depth/diversity/provenance only; no source is proven best without a controlled ablation.',
    uncertainty=dict(wilson_confidence=.95,speaker_cluster_percentile_bootstrap=2000,seed=53,
        caveat='Wilson intervals are recording-level descriptive; cluster bootstrap is conditional on observed speakers and may degenerate for very sparse strata.'),
    acoustic_training=False,synthetic_injection=False,installation=False,physical_actions=False,
    preserve_published_splits=True,threshold_tuning=False,test_audio_decoded=False)
write_json(D/'protocol.json',protocol)
R.mkdir(parents=True,exist_ok=False)
rows=read_rows(P/'dev.jsonl')
assert len({r['id'] for r in rows})==len(rows)
assert all(r['split']=='dev' and r['kind']=='human' and r['synthetic'] is False and r['metric_eligible'] for r in rows)
attempts=[]
try:
    verify_bundle(ROOT/'submission/models/compact',diagnostic=True)
    attempts.append(dict(path='submission/models/compact',status='verified'))
except ValueError as error:
    attempts.append(dict(path='submission/models/compact',status='blocked',error=str(error)))
results={}
with threadpool_limits(limits=1,user_api='blas'):
    for name,folder in (('full-shipping-frozen',ROOT/'submission/models/full'),('compact-v7-frozen',v7)):
        verify_bundle(folder,diagnostic=True)
        runtime=CompactRuntime(folder,threads=2,mode='demo') if name.startswith('compact') else Runtime(folder,diagnostic=True,threads=2)
        dest=R/(name+'-dev.jsonl')
        with dest.open('x',encoding='utf-8',newline='\n') as stream:
            total=0;started=time.perf_counter()
            for i,row in enumerate(rows,1):
                assert file_sha256(row['audio'])==row['sha256']
                pcm,rate=sf.read(row['audio'],dtype='float32')
                assert pcm_digest(pcm,rate)==row['pcm_sha256']
                begin=time.perf_counter()
                event=runtime.recognize(pcm,rate)
                app=application_decision(event,protocol['primary_policy'])
                elapsed=1000*(time.perf_counter()-begin)
                diagnostic=event.get('diagnostic_candidate') or {}
                record={**row,'candidate':diagnostic.get('candidate'),
                    'application_command':app['command'],'application_reason':app['reason'],
                    'diagnostic_candidate':diagnostic,'fixed_policy':protocol['primary_policy'],
                    'qualified_command':(event.get('qualified_output') or event['output']).get('command'),
                    'timing':event['timing'],'replay_wall_ms':elapsed,
                    'audio_seconds':len(pcm)/rate,'model_sha256':event['model_sha256']}
                if name.startswith('compact'):
                    record.update(relaxed_application_command=event['application']['command'],
                                  relaxed_application_reason=event['application']['reason'])
                stream.write(canonical(record)+'\n');total+=1
                if i%32==0:
                    stream.flush();print(name,i,'/',len(rows),flush=True)
        assert total==len(rows)
        recipe=runtime.recipe
        results[name]=dict(rows=total,path=dest.relative_to(ROOT).as_posix(),sha256=file_sha256(dest),
            folder=folder.relative_to(ROOT).as_posix(),model_sha256=recipe['binding']['model_sha256'],
            recipe_sha256=file_sha256(folder/'recipe.json'),policy_sha256=file_sha256(folder/'policy.json'),
            parameters=recipe.get('parameters',recipe.get('parameter_count')),
            elapsed_seconds=time.perf_counter()-started,policy_enabled=runtime.policy['enabled'],
            initialization_lineage='Existing human-plus-synthetic trained checkpoint, not a human-only candidate',
            onnx_bytes=(folder/'model.onnx').stat().st_size)
        del runtime
for path,digest in protocol['bindings'].items():
    assert file_sha256(ROOT/path)==digest,path
receipt=dict(format='human-scorecard-frozen-development-replay-v1',protocol_sha256=file_sha256(D/'protocol.json'),
    rows_per_model=len(rows),models=results,current_compact_bundle_attempt=attempts,
    environment={p:importlib.metadata.version(p) for p in ('numpy','onnxruntime','soundfile','threadpoolctl')},
    cpu_threads=2,blas_threads=1,model_selection=False,threshold_tuning=False,training=False,
    calibration_fitting=False,test_or_holdout_audio_opened=False,actions=False)
write_json(R/'receipt.json',receipt)
print(json.dumps(receipt,indent=2))
