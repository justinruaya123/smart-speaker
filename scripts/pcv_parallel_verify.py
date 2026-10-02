"""Validate host execution acceleration on development only, without refitting."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import pcv_resume as recovery
from scripts.pcv_parallel import ParallelScorer


def main():
    import numpy as np
    import torch
    from einops import rearrange
    from pcvcm.data import verified_features
    from pcvcm.decoder import PrefixCTC,rank_commands
    from pcvcm.training import prediction_metrics
    from vcm.model import CausalVCM
    recovery.validate_recovery()
    output=ROOT/'.cache/pcv-delivery-v1/parallel-eval-v2'
    output.mkdir(exist_ok=False)
    run=recovery.RUN;prepared=recovery.CACHE/'prepared'
    checkpoint=run/'seed-53/epoch-016.pt';reference_path=run/'seed-53/dev-016.json'
    freeze=json.loads((prepared/'freeze.json').read_text())
    for name,sha in freeze['files'].items():
        if recovery.digest(prepared/name)!=sha:raise ValueError('Prepared metadata changed: '+name)
    history=json.loads((run/'history.json').read_text())
    completed=next(r for r in history if r['seed']==53 and r['epoch']==16)
    if (recovery.digest(checkpoint)!=completed['checkpoint_sha256']
            or recovery.digest(reference_path)!=completed['predictions_sha256']):
        raise ValueError('Completed checkpoint or prediction identity changed')
    source_paths=[Path(__file__).resolve(),ROOT/'scripts/pcv_parallel.py',ROOT/'experiments/phrase-contract-v1/pcvcm/decoder.py',ROOT/'experiments/phrase-contract-v1/pcvcm/training.py']
    source_hashes={str(p.relative_to(ROOT)):recovery.digest(p) for p in source_paths}
    input_hashes={str(p.relative_to(ROOT)):recovery.digest(p) for p in
        [checkpoint,reference_path,prepared/'freeze.json',run/'catalog.json',run/'normalization.json',
         *[prepared/name for name in freeze['files']]]}
    recovery.atomic_json(output/'before.json',dict(source_hashes=source_hashes,input_hashes=input_hashes))
    references=json.loads(reference_path.read_text());before={r['recording_id']:r for r in references}
    rows=recovery.repair.read_rows(prepared/'dev.jsonl')
    if len(rows)!=5636 or set(before)!={r['recording_id'] for r in rows}:raise ValueError('Development scope changed')
    if any(r['split']!='dev' for r in rows):raise ValueError('Development only')
    entries=json.loads((run/'catalog.json').read_text())['entries']
    norm=json.loads((run/'normalization.json').read_text())
    mean=rearrange(np.array(norm['mean'],np.float32),'c -> c 1')
    std=rearrange(np.array(norm['std'],np.float32),'c -> c 1')
    arrays={r['recording_id']:(verified_features(r,prepared)-mean)/std for r in rows}
    model=CausalVCM().to('cuda');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights']);model.eval()
    torch.set_num_threads(4)
    def posterior(row):
        with torch.no_grad():
            x=torch.from_numpy(rearrange(arrays[row['recording_id']],'c t -> 1 c t')).to('cuda')
            y,_=model(x)
            return rearrange(y,'1 c t -> t c').cpu().numpy()
    representatives={}
    for row in rows:representatives.setdefault((row['engine'],row['phase'],row['use_case']),row)
    selected=list(representatives.values());logits=[posterior(row) for row in selected]
    decoder=PrefixCTC([e['phones'] for e in entries]);t=time.perf_counter()
    expected=[decoder.score(x).copy() for x in logits];serial=time.perf_counter()-t
    timings={'serial':serial};scores_equal={}
    for workers in (4,8):
        t=time.perf_counter()
        with ParallelScorer(entries,workers=workers,return_scores=True) as scorer:
            results=list(scorer.map(logits))
        timings[str(workers)]=time.perf_counter()-t
        equal=all(np.array_equal(a,b['scores']) and rank_commands(a,entries)==b['prediction'] for a,b in zip(expected,results))
        scores_equal[str(workers)]=equal
        print('SCORE_PARITY',workers,len(logits),equal,'seconds',timings[str(workers)],flush=True)
        if not equal:raise ValueError('Exact score/ranking parity failed')
    workers=min((4,8),key=lambda n:timings[str(n)])
    if serial/timings[str(workers)]<1.5:raise ValueError('Insufficient demonstrated speedup')
    sampled={r['recording_id']:score for r,score in zip(selected,expected)}
    records=[];errors=[];began=time.perf_counter();score_comparisons=0
    with (output/'rows.jsonl').open('x',encoding='utf-8') as stream,ParallelScorer(entries,workers=workers,return_scores=True) as scorer:
        for row,item in zip(rows,scorer.map(posterior(row) for row in rows),strict=True):
            ref=before[row['recording_id']];prediction=item['prediction']
            actual={**ref,'candidate':prediction['command'],'raw_exact':prediction['command']==ref['expected']}
            exact=actual==ref;score_match=None
            if row['recording_id'] in sampled:
                score_match=np.array_equal(item['scores'],sampled[row['recording_id']]);score_comparisons+=1
            if not exact or score_match is False:errors.append(row['recording_id'])
            stream.write(json.dumps(dict(recording_id=row['recording_id'],typed_output_equal=exact,scores_equal=score_match))+'\n')
            records.append(actual)
            if len(records)%100==0:stream.flush();print('FULL_DEV_PARITY',len(records),'/',len(rows),'elapsed',round(time.perf_counter()-began,2),flush=True)
    for name,sha in {**source_hashes,**input_hashes}.items():
        if recovery.digest(ROOT/name)!=sha:raise ValueError('Validation source/input changed during evaluation: '+name)
    receipt=dict(complete=len(records)==len(rows),recordings=len(records),workers=workers,errors=errors,
        score_comparisons=score_comparisons,score_parity=scores_equal,timing_seconds=timings,
        measured_speedup=serial/timings[str(workers)],full_dev_seconds=time.perf_counter()-began,
        checkpoint_sha256=recovery.digest(checkpoint),reference_predictions_sha256=recovery.digest(reference_path),
        prepared_freeze_sha256=recovery.digest(prepared/'freeze.json'),rows_sha256=recovery.digest(output/'rows.jsonl'),
        protocol_sha256=recovery.digest(recovery.repair.HERE/'protocol.json'),
        source_hashes=source_hashes,input_hashes=input_hashes,
        metrics=prediction_metrics(records),reference_metrics=prediction_metrics(references),
        sealed_test_accessed=False,training_performed=False,acceptance_qualified=False)
    recovery.atomic_json(output/'results.json',receipt)
    if errors:raise ValueError('Full development output parity failed')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
