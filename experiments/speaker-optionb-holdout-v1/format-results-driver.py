"""Publish compact existing results and a per-verbatim human collection queue."""
import csv,json,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.runtime.pcv_schema import INTENTS,canonical
from scripts.train.speaker_human_data import write_json
from scripts.test.speaker_human_scorecard import wilson_interval
D=ROOT/'experiments/speaker-optionb-holdout-v1'
r=json.loads((D/'holdout-results.json').read_text());w=json.loads((D/'optionb-wording-data.json').read_text())['cases']
cues=json.loads((D/'optionb-cue-plan.json').read_text())['cases'];assert len(w)==len(cues)==93
views={'full-fixed':r['models']['full']['human'],'compact-v7-fixed':r['models']['compact-v7']['human'],
       'compact-v7-existing-relaxed':r['models']['compact-v7']['existing_relaxed']['human']}
rows=[]
for name,view in views.items():
    for intent in INTENTS:
        c=view['fixed']['per_intent'][intent];raw=view['raw']['per_intent'][intent];n=c['reference_n'];p=c['predicted_n'];tp=c['complete_correct']
        rows.append(dict(view=name,intent=intent,reference_n=n,predicted_n=p,tp=tp,fp=p-tp,fn=n-tp,raw_correct=raw['complete_correct'],
            precision=c['complete_precision'],recall=c['complete_recall'],f1=2*tp/(n+p) if n+p else None,
            recall_wilson95=canonical(wilson_interval(tp,n))))
with (D/'holdout-per-intent.csv').open('x',encoding='utf-8',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
plan=[]
for case,cue in zip(w,cues,strict=True):
    assert case['id']==cue['id'];strata=case['strata'];train=strata['human/train']
    plan.append(dict(id=case['id'],verbatim=case['verbatim'],command=case['command'],candidate_cue_tokens=cue['candidate_cue_tokens'],
        train_exact=train['exact_n'],train_cue_nonexact=train['cue_nonexact_n'],train_typed_speakers=train['typed_speakers'],
        dev_exact=strata['human/dev']['exact_n'],calibration_exact=strata['human/calibration']['exact_n'],
        priority='missing_human_typed_training_cell' if not train['typed_reference_n'] else 'diversify_speakers_and_wording',
        requested_evidence=['original connected verbatim','same-command minor wording/courtesy variant',
            'bounded leading/trailing-context variation','negated/quoted/changed-value and extra-action contrast'],
        rule='Same required slots and one intended command; unrelated content changing meaning is not benign padding',
        keyword_phone_targets_created=False,new_human_recordings=0))
write_json(D/'human-keyword-collection-plan.json',dict(cases=plan,case_count=len(plan),recordings_created=0,
    interpretation='Coverage-driven collection prompts, not a manufactured acoustic dataset or new independent speakers. Holdout examples never enter training.'))
pct=lambda value:'N/A' if value is None else f'{100*value:.2f}%'
print('MODEL | RAW | ACCEPTED | P | R | F1 | UNKNOWN')
for name,view in views.items():
    o=view['fixed']['overall'];raw=view['raw']['overall'];tp=o['exact'];f1=2*tp/(o['positive_n']+o['accepted'])
    print(name,f"{raw['exact']}/{o['positive_n']}",f"{tp}/{o['positive_n']}",pct(o['precision']),pct(o['recall']),pct(f1),f"{o['negative_errors']}/{o['negative_n']}",sep=' | ')
print('INTENT | N | RAW FULL | ACCEPTED FULL | ACCEPTED PREFIX | ACCEPTED COMPACT FIXED | ACCEPTED COMPACT RELAXED')
for intent in INTENTS:
    m=r['models'];print(intent,m['full']['human']['fixed']['per_intent'][intent]['reference_n'],
        m['full']['human']['raw']['per_intent'][intent]['complete_correct'],m['full']['human']['fixed']['per_intent'][intent]['complete_correct'],
        m['full']['stress']['silence_prefix_1s']['fixed']['per_intent'][intent]['complete_correct'],
        m['compact-v7']['human']['fixed']['per_intent'][intent]['complete_correct'],
        m['compact-v7']['existing_relaxed']['human']['fixed']['per_intent'][intent]['complete_correct'],sep=' | ')
print('FULL ZERO-NEGATIVE 95% WILSON',wilson_interval(0,11))
print('FULL CORRECT REJECTED',r['models']['full']['human']['comprehensive']['overall']['raw_correct_rejected'])
print('HUMAN POSITIVE SPEAKERS',r['models']['full']['human']['speaker_cluster']['speaker_n'])
print('CSV ROWS',len(rows),'COLLECTION CASES',len(plan))
