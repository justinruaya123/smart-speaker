"""Host-only parallel execution of the unchanged exact finite decoder.

GPU inference stays in the parent. Spawned workers own independent CPU decoder
state; bounded, ordered submission preserves recording order and memory bounds.
No model, catalog, score arithmetic, threshold or selection rule is changed.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
_decoder=None
_entries=None


def _initialize(entries):
    global _decoder,_entries
    from pcvcm.decoder import PrefixCTC
    _entries=entries
    _decoder=PrefixCTC([entry['phones'] for entry in entries])


def _score(logits,return_scores):
    from pcvcm.decoder import rank_commands
    scores=_decoder.score(logits)
    result=dict(prediction=rank_commands(scores,_entries))
    if return_scores:result['scores']=scores
    return result


class ParallelScorer:
    def __init__(self,entries,workers=8,max_pending=None,return_scores=False,budget_check=None):
        if type(workers) is not int or not 1<=workers<=32:raise ValueError('Worker count must be 1..32')
        if max_pending is None:max_pending=2*workers
        if type(max_pending) is not int or not workers<=max_pending<=4*workers:
            raise ValueError('Pending bound must be workers..4*workers')
        self.entries=entries;self.workers=workers;self.max_pending=max_pending
        self.return_scores=return_scores;self.budget_check=budget_check;self.pool=None

    def __enter__(self):
        if self.pool is not None:raise RuntimeError('Scorer already active')
        self.pool=ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'),initializer=_initialize,initargs=(self.entries,))
        return self

    def __exit__(self,*args):
        self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None

    def map(self,logits):
        if self.pool is None:raise RuntimeError('Use scorer as a context manager')
        source=iter(logits);pending=deque();exhausted=False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending)<self.max_pending:
                    try:value=next(source)
                    except StopIteration:exhausted=True;break
                    pending.append(self.pool.submit(_score,value,self.return_scores))
                if not pending:break
                future=pending.popleft()
                while True:
                    if self.budget_check is not None:self.budget_check()
                    try:result=future.result(timeout=1.);break
                    except TimeoutError:continue
                yield result
        finally:
            for future in pending:future.cancel()


def development(model,rows,arrays,entries,device,budget_check=None,*,workers=8):
    import torch
    from einops import rearrange
    from pcvcm.contract import canonical
    from pcvcm.training import prediction_metrics
    records=[];model.eval();began=time.perf_counter()
    def posteriors():
        with torch.no_grad():
            for row in rows:
                if budget_check is not None:budget_check()
                x=torch.from_numpy(rearrange(arrays[row['recording_id']],'c t -> 1 c t')).to(device)
                logits,_=model(x)
                yield rearrange(logits,'1 c t -> t c').cpu().numpy()
    with ParallelScorer(entries,workers=workers,budget_check=budget_check) as scorer:
        for row,result in zip(rows,scorer.map(posteriors()),strict=True):
            prediction=result['prediction']
            records.append(dict(recording_id=row['recording_id'],phrase_id=row.get('phrase_id'),
                use_case=row['use_case'],reference_class=row['reference_class'],expected=row.get('expected_command'),
                candidate=prediction['command'],raw_exact=canonical(prediction['command'])==canonical(row.get('expected_command'))))
            if len(records)%100==0 or len(records)==len(rows):
                print('PARALLEL_DEVELOPMENT',len(records),'/',len(rows),'workers',workers,
                      'elapsed_seconds',round(time.perf_counter()-began,2),flush=True)
    return prediction_metrics(records),records
