"""Replay saved predictions through the pinned external benchmark's actual scorer."""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/pcv-full-benchmark-v1'
CACHE = ROOT/'.cache/pcv-full-benchmark-v1'


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='pc-full-single-core')
    args = parser.parse_args()
    if not args.run or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.run):
        raise ValueError('Invalid run ID')
    output = CACHE/('external-'+args.run)
    output.mkdir(exist_ok=False)
    destinations = [HERE/(args.run+'-external-results.json'), HERE/(args.run+'-threshold-results.json')]
    if any(p.exists() for p in destinations):raise FileExistsError('Scoring receipts are immutable')
    receipt = json.loads((HERE/'external-source.json').read_text())
    for name,digest in receipt['files'].items():
        if hashlib.sha256((CACHE/'external'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('External scoring source mismatch')
    sys.path.insert(0,str(CACHE/'external'))
    from vcmbench.schema import SLOTTED, INTENTS, build_alias_table, normalize_prediction
    from vcmbench.metrics import classification_report, command_label, intent_label, slot_report
    from vcmbench.slots import slot_distance
    from scripts.runtime.pcv_benchmark_adapter import class_prediction
    from scripts.runtime.pcv_schema import baseline_cases
    from scripts.runtime.pcv_full_benchmark import summarize
    from scripts.runtime.pcv_final_runtime import decision
    # Check every baseline typed output with the actual external normalization.
    import csv
    variations = list(csv.DictReader((CACHE/'external/vcmbench/variations.csv').open()))
    baselines = baseline_cases()
    aliases = build_alias_table()
    adapter_checks = []
    for case, variation in zip(baselines, variations):
        # Ordering differs: join on exact phrase, not class index.
        variation = next(v for v in variations if v['phrase'].casefold()==case['text'].casefold())
        intent,slot = class_prediction(case['command'])
        intent,slot,known = normalize_prediction(intent,slot,aliases)
        exact = intent==variation['label'] and (intent not in SLOTTED or slot_distance(intent,variation['value'],slot)['exact'])
        if not exact or not known:raise ValueError('Baseline adapter mismatch: '+case['id'])
        adapter_checks.append(case['id'])
    rows = [json.loads(line) for line in (CACHE/args.run/'trials.jsonl').read_text().splitlines()]
    definitions = json.loads((CACHE/'corpus/definitions.json').read_text())
    policy = json.loads((ROOT/'.cache/pcv-release-train-v1/runs/release-v1/evaluation/bundle/policy.json').read_text())
    for row in rows:
        candidate = row['event']['diagnostic_candidate']
        row['threshold_candidate'] = decision(candidate,{**policy,'enabled':True})['command'] if candidate else None
    result = {'external_repository':receipt['repository'],'external_commit':receipt['revision'],
              'input_trials_sha256':hashlib.sha256((CACHE/args.run/'trials.jsonl').read_bytes()).hexdigest(),
              'run':args.run,
              'adapter_baselines_passed':len(adapter_checks),'adapter_baselines_total':len(baselines),
              'mode':'Offline scoring compatibility, not live speaker-to-Pi testing',
              'reference':'Publisher labels for official class scoring; frozen typed references remain separate',
              'threshold_candidate':'Hypothetical enabled frozen thresholds only; authoritative policy remains disabled',
              'tracks':{}}
    for track in sorted({r['track'] for r in rows}):
        subset = [r for r in rows if r['track']==track]
        scores = {}
        for mode in ('accepted','candidate','threshold_candidate'):
            ti=[];pi=[];tc=[];pc=[];distances=[]
            for row in subset:
                target = row['publisher_intent'];intent,slot = class_prediction(row[mode])
                if intent not in INTENTS and intent!='OUT_OF_SCOPE':intent='OTHER:'+intent
                ti.append(intent_label(target));pi.append(intent_label(intent))
                distance = slot_distance(target,row['publisher_slot'],slot) if target in SLOTTED else None
                exact = intent==target and (distance is None or distance['exact'])
                tc.append(row['publisher_variation'] if target!='OUT_OF_SCOPE' else 'REJECT')
                pc.append(command_label(intent,slot,row['publisher_variation'] if exact else None,exact))
                if target==intent and distance is not None:distances.append({'intent':target,'slot':distance})
            scores[mode] = {'intent':classification_report(ti,pi),'command':classification_report(tc,pc),
                            'conditional_slots':slot_report(distances)}
        result['tracks'][track] = scores
    with destinations[0].open('x') as stream:stream.write(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    from scripts.runtime.pcv_full_benchmark import typed_metrics
    diagnostics = {track:typed_metrics([r for r in rows if r['track']==track],'threshold_candidate')
                   for track in sorted({r['track'] for r in rows})}
    with destinations[1].open('x') as stream:stream.write(json.dumps(diagnostics,indent=2)+'\n')
    for name in ('accepted','candidate'):
        with (output/('failed-'+name+'.jsonl')).open('x') as stream:
            for row in rows:
                if row['reference_status']!='ambiguous' and row[name]!=row['expected']:
                    stream.write(json.dumps(row)+'\n')
    print(json.dumps({'adapter_passed':len(adapter_checks),'tracks':{
        t:{k:{'accuracy':v['command']['accuracy'],'n':v['command']['n'],'false_accepts':v['command']['false_accepts']}
           for k,v in scores.items()} for t,scores in result['tracks'].items()}},indent=2))


if __name__ == '__main__':main()
