"""Repeatable numerical/cost preflight. Random tensors are NOT speech evaluation."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from scripts.test.pcv_strict_audit import ROOT,BASE,REFERENCES,source_bytes,load_reference,expand_schema,cp_upper
from scripts.models.pcv_strict_candidates import StrictTinyVCM


def run(repeats:int=20) -> dict:
    if repeats<3:raise ValueError('Use at least three timing repetitions')
    old=os.sched_getaffinity(0) if hasattr(os,'sched_getaffinity') else None
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if old is not None:os.sched_setaffinity(0,{min(old)})
    try:
        affinity=sorted(os.sched_getaffinity(0)) if old is not None else None
        for key in REFERENCES:source_bytes(key)
        torch.manual_seed(1102)
        Model=load_reference('model')['PooledContextVCM']
        models={'pooled_context_192':Model(),
                'pooled_context_80':Model(width=80,feedforward=160),
                'strict_ds_tcn_96':StrictTinyVCM(kind='tcn'),
                'strict_crnn_96':StrictTinyVCM(kind='crnn')}
        cases=[]
        with torch.inference_mode():
            for name,m in models.items():
                m.eval()
                for frames in [100,200,400,800]:
                    x=torch.randn(1,40,frames)
                    for _ in range(5):m(x)
                    times=[]
                    for _ in range(repeats):
                        start=time.perf_counter_ns();m(x);times.append((time.perf_counter_ns()-start)/1e6)
                    cases.append(dict(name=name,feature_frames=frames,parameters=sum(p.numel() for p in m.parameters()),
                        median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),repeats=repeats))
        PrefixCTC=load_reference('decoder')['PrefixCTC']
        rng=np.random.default_rng(1102)
        paths=[rng.integers(1,40,size=int(rng.integers(12,35))).tolist() for _ in range(1031)]
        x=rng.normal(size=(100,40));decoder_cases=[];full=None
        for count in [1031,415,93]:
            d=PrefixCTC(paths[:count]);times=[]
            scores=d.score(x)
            if full is None:full=scores.copy()
            for _ in range(7):
                start=time.perf_counter_ns();d.score(x);times.append((time.perf_counter_ns()-start)/1e6)
            decoder_cases.append(dict(entries=count,trie_nodes=d.node_count,frames=100,
                max_score_error_vs_full=float(np.max(np.abs(scores-full[:count]))),
                median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95))))
        x=np.zeros((10,40));x[2,1]=4;x[6,2]=4
        tail=np.full((100,40),-10.);tail[:,0]=10.
        a=float(PrefixCTC([[1,2]]).score(x)[0]);b=float(PrefixCTC([[1,2]]).score(np.concatenate([x,tail]))[0])
        rows=expand_schema(ROOT/'experiments/pcv-strict-final-v1/option_b.csv')
        return dict(experiment='pcv-strict-final-v1',base_commit=BASE,
            evidence_class='Numerical source tests and untrained PyTorch CPU cost probes; NO fresh speech accuracy or ONNX/Pi timing.',
            environment=dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,
                numpy=np.__version__,cpu_affinity=affinity,torch_threads=torch.get_num_threads(),
                onnx_available=bool(importlib.util.find_spec('onnx')),onnxruntime_available=bool(importlib.util.find_spec('onnxruntime'))),
            source_blob_hashes={k:v[1] for k,v in REFERENCES.items()},
            schema=dict(phrases=len(rows),intents=len({r['intent'] for r in rows}),semantics=len({r['semantic_key'] for r in rows})),
            acoustic_model_cost=cases,decoder_cost=decoder_cases,
            decoder_topology='Seeded random phone paths: mechanism/cost fixture, NOT the release catalog.',
            score_duration_probe=dict(original_frames=10,with_blank_tail_frames=110,original_score=a,with_blank_tail_score=b),
            negative_evidence=dict(upper95_zero_of_250=cp_upper(0,250),upper95_zero_of_299=cp_upper(0,299),upper95_eight_of_250=cp_upper(8,250)),
            exclusions=['No trained checkpoint was mounted','No fresh HF audio download','No waveform replay','No training','No ONNX Runtime available','No physical Raspberry Pi'])
    finally:
        if old is not None:os.sched_setaffinity(0,old)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=20)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior evidence; select a new output path')
    result=run(args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
