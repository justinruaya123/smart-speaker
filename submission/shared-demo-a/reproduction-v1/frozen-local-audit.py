"""Independent read-only primitive audit after the actual fresh-main fit."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import subprocess
import sys

import numpy as np
import onnx
import torch

TASK=Path('D:/School/AI231/smart-speaker/.cache/hf-main-reproduction-v1')
ROOT=TASK/'reproduce'
OUT=ROOT/'.cache/fresh-main-02'
REFERENCE=ROOT/'submission/shared-demo-a/reproduction-reference.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
reference=load(REFERENCE)
source=load(OUT/'source-acquisition.json')
acquisition=load(OUT/'dataset-acquisition.json')
assert source['git_branch']=='main'
assert source['git_revision']=='64c3ea19bb0a0f0bd52866ad04f6c8b41b5a92e4'
assert acquisition['resolved_revision']=='13d388c31985be7f42af56b46570281728ecc054'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
lock=load(ROOT/'submission/shared-demo-a/dataset.lock.json')
data=OUT/'dataset/shared-demo-a-v1'
assert sha(data/'manifest.json')==lock['manifest_sha256']==acquisition['manifest_sha256']
manifest=load(data/'manifest.json')
for name,record in manifest['files'].items():
    assert (data/name).stat().st_size==record['bytes'] and sha(data/name)==record['sha256'],name
upstream_payloads=0
for record in manifest['upstream_shards']:
    path=OUT/'hf-cache'/('datasets--'+record['repository'].replace('/','--'))/'snapshots'/record['revision']/record['file']
    assert sha(path)==record['sha256'],record['file']
    upstream_payloads+=1
views={}
for name,record in manifest['views'].items():
    with gzip.open(data/(name+'.jsonl.gz'),'rt',encoding='utf-8') as stream:
        views[name]=[json.loads(line) for line in stream if line.strip()]
    assert len(views[name])==record['rows']
fitrows=views['train']+views['original-dev']+views['new-dev']
arrays=torch.load(OUT/'train-dev-features.pt',weights_only=False,map_location='cpu')
assert set(arrays)=={r['id'] for r in fitrows} and len(arrays)==28090
for row in fitrows:
    array=np.asarray(arrays[row['id']],dtype='<f4')
    assert list(array.shape)==row['feature_shape']
    assert hashlib.sha256(f'f32le:{array.shape}\0'.encode()+array.tobytes()).hexdigest()==row['feature_sha256']
del arrays
assert load(OUT/'normalization.json')==load(data/'normalization.json')
assert load(OUT/'normalization.json')==load(ROOT/'submission/models/shared-demo-a/normalization.json')
normalizer_bytes_identical=sha(OUT/'normalization.json')==sha(ROOT/'submission/models/shared-demo-a/normalization.json')
original=torch.load(ROOT/'submission/checkpoints/shared-demo-a-selected.pt',weights_only=False,map_location='cpu')
best=torch.load(OUT/'heads_s53/best.pt',weights_only=False,map_location='cpu')
last=torch.load(OUT/'heads_s53/last.pt',weights_only=False,map_location='cpu')
assert set(original['state_dict'])==set(best['state_dict'])
for name,value in original['state_dict'].items():
    assert torch.equal(value,best['state_dict'][name]),name
assert best['epoch']==original['epoch']==38
assert best['dev']==original['dev']
assert best['sample_counts']==original['sample_counts']
curve=[json.loads(l) for l in (OUT/'heads_s53/training-curve.jsonl').read_text().splitlines()]
assert [{k:v for k,v in e.items() if k!='elapsed_seconds'} for e in curve]==reference['curve']
result=load(OUT/'heads_s53/training-result.json')
assert len(curve)==48 and last['epoch']==48 and result['optimizer_steps']==4608
assert result['draws']==147456 and sum(result['sample_counts'].values())==147456
assert result['sequence_sha256']==reference['sequence_sha256']
assert result['source_draws']==reference['source_draws']
actual=load(OUT/'reproduced-predictions.json')
expected={r['id']:r for r in reference['predictions']}
assert len(actual)==len(expected)==642 and len({r['id'] for r in actual})==642
for row in actual:
    assert all(row[k]==expected[row['id']][k] for k in ('id','class_index','predicted_class','sha256','pcm_sha256'))
lookup={r['id']:r for r in actual}
summary={}
for cohort,rows in [('exact_synthetic',[r for r in views['test'] if r.get('form')=='clean']),
    ('context_synthetic',[r for r in views['test'] if r['class_index']>0 and r.get('form')!='clean']),
    ('human_supported',[r for r in views['official'] if r['kind']=='human' and r['class_index']>0]),
    ('synthetic_test',views['test']),('official',views['official'])]:
    positive=[r for r in rows if r['class_index']>0]
    negative=[r for r in rows if r['class_index']==0]
    summary[cohort]=dict(rows=len(rows),supported=len(positive),
        correct=sum(lookup[r['id']]['predicted_class']==r['class_index'] for r in positive),
        rejected=sum(lookup[r['id']]['predicted_class']==0 for r in positive),
        wrong_accepted=sum(lookup[r['id']]['predicted_class'] not in (0,r['class_index']) for r in positive),
        negative_rows=len(negative),negative_false_accepts=sum(lookup[r['id']]['predicted_class']>0 for r in negative))
assert summary['exact_synthetic']['correct']==203 and summary['exact_synthetic']['supported']==204
assert summary['human_supported']['correct']==71 and summary['human_supported']['supported']==77
assert summary['context_synthetic']['correct']==98 and summary['context_synthetic']['supported']==204
graph=onnx.load(str(OUT/'reproduced.onnx')); onnx.checker.check_model(graph)
parameters=sum(int(np.prod(t.dims)) for t in graph.graph.initializer)
assert parameters==175653
receipt=dict(passed=True,git_revision=source['git_revision'],dataset_revision=acquisition['resolved_revision'],
    fresh_environment=load(TASK/'training-environment.json'),declared_dataset_files_readback=len(manifest['files'])+1,
    upstream_shards_independently_verified=upstream_payloads,
    treated_rows=sum(len(v) for v in views.values()),feature_arrays_independently_verified=len(fitrows),
    normalization_human_rows=1889,selected_tensors_identical=len(best['state_dict']),parameters=parameters,
    selected_epoch=38,completed_epochs=48,optimizer_updates=4608,draws=147456,
    original_checkpoint_sha256=sha(ROOT/'submission/checkpoints/shared-demo-a-selected.pt'),
    reproduced_checkpoint_sha256=sha(OUT/'heads_s53/best.pt'),
    checkpoint_bytes_identical=sha(ROOT/'submission/checkpoints/shared-demo-a-selected.pt')==sha(OUT/'heads_s53/best.pt'),
    checkpoint_semantics_identical=all(original[k]==best[k] for k in original if k!='state_dict'),
    graph_sha256=sha(OUT/'reproduced.onnx'),
    graph_bytes_identical=sha(OUT/'reproduced.onnx')==sha(ROOT/'submission/models/shared-demo-a/model.onnx'),
    normalizer_values_identical=True,normalizer_bytes_identical=normalizer_bytes_identical,
    normalizer_diff_only_CRLF=(ROOT/'submission/models/shared-demo-a/normalization.json').read_bytes().replace(b'\r\n',b'\n')==(OUT/'normalization.json').read_bytes(),
    all_642_original_decisions_identical=True,summary=summary,
    bindings={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [REFERENCE,OUT/'source-acquisition.json',
        OUT/'dataset-acquisition.json',OUT/'preparation.json',OUT/'heads_s53/training-result.json',
        OUT/'heads_s53/training-curve.jsonl',OUT/'training-comparison.json',OUT/'selection-before-test.json',
        OUT/'reproduced-predictions.json',OUT/'evaluation.json',OUT/'reproduced.onnx']},
    verifier_sha256=sha(__file__),refit_for_audit=False,retuned=False,installed=False,qualified=False,actions_enabled=False)
with (TASK/'independent-reproduction-audit.json').open('x',encoding='utf-8') as stream:
    json.dump(receipt,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('bindings','fresh_environment')},indent=2))
