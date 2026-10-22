"""Recheck the actual frozen annotation policy; preserve older-rule diagnostics."""
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from scripts.train.speaker_human_data import file_sha256,read_rows,write_json
from scripts.test.speaker_optionb_holdout import annotation_review
from scripts.runtime.pcv_schema import canonical
from scripts.train.pcv_release_prepare import PROPOSAL_SHA256
D=ROOT/'experiments/speaker-optionb-holdout-v1'
A=ROOT/'.cache/speaker-optionb-holdout-v1/audit-02';A.mkdir(exist_ok=False)
H=ROOT/'.cache/speaker-human-recovery-v1/prepared-01'
S=ROOT/'.cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source'
freeze=ROOT/'experiments/pcv-release-train-v1/data-freeze.json'
approval=ROOT/'experiments/pcv-release-train-v1/annotation-approval.json'
review=ROOT/'experiments/pcv-release-train-v1/training-label-review.json'
frozen=json.loads(freeze.read_text())
key=next(k for k in frozen['bindings'] if k.replace('\\','/')=='experiments/pcv-release-train-v1/annotation-approval.json')
assert file_sha256(approval)==frozen['bindings'][key]
policy=json.loads(approval.read_text())
assert policy['proposal_sha256']==PROPOSAL_SHA256
paths=[freeze,approval,review,ROOT/'scripts/train/pcv_release_prepare.py',ROOT/'scripts/test/speaker_optionb_holdout.py',
       ROOT/'scripts/test/test_speaker_optionb_holdout.py',D/'selection-before-holdout.json',D/'data-audit-results.json']
write_json(D/'frozen-label-recheck-inputs.json',dict(format='optionb-frozen-label-audit-v2',
    reference_rule=policy['version'],frozen_approval_verified_against_data_freeze=True,
    source_proposal_expected=PROPOSAL_SHA256,source_proposal_actual=file_sha256(review),
    regeneration_guard='Default annotation_policy() fails the current proposal checksum; no guard is disabled. The archived approval is explicitly verified and passed to the existing replay API.',
    bindings={p.relative_to(ROOT).as_posix():file_sha256(p) for p in paths},driver_sha256=file_sha256(__file__),
    explanation='The first audit compared older base rules only. Its 258 disagreements are not 258 new labeling errors; all first-pass evidence remains preserved.'))
rows=[r for split in ('train','dev','calibration','test') for r in read_rows(H/(split+'.jsonl'))]
rows+=read_rows(ROOT/'.cache/speaker-optionb-holdout-v1/replay-01/full-original.jsonl')
groups=defaultdict(list)
for row in rows:groups[(row['location']['file'],row['location']['sha256'])].append(row)
results=[]
for (name,digest),members in groups.items():
    path=S/name;assert file_sha256(path)==digest
    q=pq.ParquetFile(path);metadata=q.read(columns=[c for c in q.schema_arrow.names if c not in ('audio','whisper_transcript')]).to_pylist()
    for row in members:
        meta=metadata[row['location']['row']]
        results.append(annotation_review(row,meta,policy=policy))
with (A/'label-rechecks.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
    for row in results:stream.write(canonical(row)+'\n')
flags=Counter(flag for r in results for flag in r['flags'])
per_split={s:dict(Counter(flag for r in results if r['split']==s for flag in r['flags'])) for s in ('train','dev','calibration','test','holdout')}
disagreements=[r for r in results if 'semantic_reference_disagreement' in r['flags']]
write_json(D/'frozen-label-recheck.json',dict(format='optionb-frozen-label-audit-v2',rows=len(results),
    annotation_rule_version=policy['version'],flag_counts=dict(flags),flags_by_split=per_split,
    semantic_disagreements=disagreements,labels_changed=0,new_train_admissions=0,independent_listening=False,
    old_rule_audit='data-audit-results.json is preserved as an older-base-rule comparison, not the authoritative frozen-label check',
    proposal_regeneration_blocked=file_sha256(review)!=PROPOSAL_SHA256,
    frozen_approval_sha256=file_sha256(approval)))
for path,digest in json.loads((D/'frozen-label-recheck-inputs.json').read_text())['bindings'].items():assert file_sha256(ROOT/path)==digest,path
print(json.dumps(dict(rows=len(results),flags=flags,by_split=per_split,semantic_disagreement_n=len(disagreements),
    proposal_regeneration_blocked=file_sha256(review)!=PROPOSAL_SHA256,disagreements=disagreements[:10]),indent=2))
