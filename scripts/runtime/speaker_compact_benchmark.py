"""Compact endpoint benchmark with separate fixed, relaxed and qualified counts."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from scripts.runtime.pcv_pi_benchmark import metrics
from scripts.runtime.speaker_compact import CompactRuntime


def decision_metrics(trials):
    rows = []
    for trial in trials:
        event = trial['event']
        rows.append(dict(expected=trial['expected'],
            fixed=(event.get('fixed_application') or {}).get('command'),
            relaxed=(event.get('application') or {}).get('command'),
            qualified=(event.get('qualified_output') or {}).get('command')))
    return {name:metrics(rows, key) for name,key in
            (('fixed_demo','fixed'),('relaxed_demo','relaxed'),('qualified','qualified'))}


def write_report(folder, result, runtime):
    trials = [json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()] if (folder/'trials.jsonl').exists() else []
    strata = {'replay':[r for r in trials if r['mode']=='replay'],
              'guided_live':[r for r in trials if r['mode']=='guided_live' and r.get('capture_completed',True)]}
    result.update(decision_mode=runtime.mode, qualification_enabled=runtime.policy['enabled'],
        recall_policy_sha256=runtime.recipe['recall_policy_sha256'],
        recall_intent_thresholds=runtime.recall_policy['intent_thresholds'],
        decision_comparison=decision_metrics(strata[result['selected_metric_mode']]),
        decision_strata={name:decision_metrics(rows) for name,rows in strata.items()})
    (folder/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Compact VCM device report','',
        f"Decision mode: {runtime.mode}; strict qualification enabled: {runtime.policy['enabled']}.",
        f"Physical Pi measured: {result['hardware']['physical_pi_measured']}; live microphone measured: {result['live_microphone_measured']}.",
        'Demo decisions use the frozen human-fitted intent thresholds and .90 slot/target checks. Qualified decisions retain the strict calibration gate. Actions are disabled.',
        'Replay and completed live captures are reported separately. Incomplete capture attempts remain in live_attempts in metrics.json; missing recordings remain untested.',
        'The bundled recordings are a previously observed smoke subset, not new qualification. Guided expected labels describe requested speech, without automatic wording verification.','']
    for name,comparison in result['decision_strata'].items():
        lines.extend([f'## {name}','','| Policy | Correct / positive | Accepted | UNKNOWN accepts / negatives |','|---|---:|---:|---:|'])
        for policy,item in comparison.items():
            lines.append(f"| {policy} | {item['exact']}/{item['positive_n']} | {item['accepted']} | {item['negative_errors']}/{item['negative_n']} |")
        timing=result['strata'][name]['latency_ms']
        lines.extend(['',f"Inference ms p50/p95: {timing['p50']} / {timing['p95']} (n={timing['n']}).",''])
    lines.extend([f"Peak RSS MiB: {result['peak_rss_mb']}; VCM load ms: {result['session_load_ms']}.",
        'Inference latency includes frontend, ONNX, finite scoring and recall policy, including the first inference. It excludes file reading, spoken duration, wake listening and endpoint silence. Wake/VAD load timings are separate in metrics.json.',
        'All per-intent, phrase, attempted-capture and resource denominators are in metrics.json. No threshold is fitted on this device replay.'])
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run(args):
    # Isolate the frozen benchmark globals; do not mutate another caller's module.
    path=Path(__file__).with_name('pcv_pi_benchmark.py')
    spec=importlib.util.spec_from_file_location('scripts.runtime._compact_benchmark_base',path)
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    holder={}
    def runtime_factory(*a,**kw):
        runtime=CompactRuntime(*a,**kw,mode=args.mode);holder['runtime']=runtime;return runtime
    base.Runtime=runtime_factory
    base.write_report=lambda folder,result:write_report(folder,result,holder['runtime'])
    return base.run(args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--engine',choices=['onnxruntime']);p.add_argument('--mode',choices=['demo','qualified'])
    p.add_argument('--guided',action='store_true');p.add_argument('--mic',type=int)
    p.add_argument('--case',action='append');p.add_argument('--seconds',type=float);p.add_argument('--no-wake',action='store_true')
    args=p.parse_args()
    if args.guided and args.mic is None:p.error('--guided requires an explicitly selected --mic index')
    from scripts.runtime.speaker_config import resolve_config
    cfg,_=resolve_config(args.root,overrides={'engine':args.engine},profile='benchmark')
    if cfg['inference']['engine']!='onnxruntime':p.error('Compact archive supports only ONNX Runtime; select --engine onnxruntime')
    args.engine=cfg['inference']['engine'];args.mode=args.mode or cfg['inference']['mode']
    run(args)


if __name__=='__main__':main()
