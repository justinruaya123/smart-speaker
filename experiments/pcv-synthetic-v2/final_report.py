"""Add explicit parameter/search/wake evidence to the shared command report."""
from collections import defaultdict
import json
from pathlib import Path
import revision as rev
from revision import CACHE,HERE,read_rows,write_json,canonical,file_hash


def parameter_metrics(rows):
    groups=defaultdict(list)
    for row in rows:
        expected=row['expected_command']
        if expected is None:continue
        for key,value in expected.items():
            if key=='intent':continue
            raw=row.get('candidate') or {};actual=row['output'].get('command') or {}
            groups[key].append((raw.get('intent')==expected['intent'] and canonical(raw.get(key))==canonical(value),
                row['output']['accepted'] and actual.get('intent')==expected['intent'] and canonical(actual.get(key))==canonical(value)))
    return {k:dict(trials=len(v),raw_correct=sum(x[0] for x in v),accepted_correct=sum(x[1] for x in v)) for k,v in groups.items()}


def report(output):
    import reporting
    result=reporting.report(output);output=Path(output)
    rows=read_rows(CACHE/'final-test/predictions.jsonl')
    command=json.loads((CACHE/'final-test/results.json').read_text())
    wake=json.loads((CACHE/'wake-final-test/results.json').read_text())
    if not wake['complete'] or wake['binding']!=command['binding']:raise ValueError('Wake and command evidence must bind the same candidate')
    if file_hash(CACHE/'wake-final-test/predictions.jsonl')!=wake['predictions_sha256']:raise ValueError('Wake predictions changed')
    write_json(output/'parameter-metrics.json',parameter_metrics(rows))
    suite=read_rows(output/'all-verbatims-with-outputs.jsonl')
    demo=[]
    for uc in range(1,11):
        candidates=[r for r in suite if r['use_case']==f'UC-{uc:02}' and r['expected_command'] is not None]
        for phase in (1,2):
            chosen=sorted([r for r in candidates if r['phase']==phase],key=lambda r:(not r['id'].startswith('V2-'),r['id']))[:3]
            demo.extend(chosen)
    # Explicitly show endpoints, representative interior numbers and every K-pop title.
    for r in suite:
        c=r['expected_command'] or {}
        if (c.get('intent')=='BRIGHTNESS' and c.get('value') in (0,1,37,73,99,100)) or (
                c.get('intent')=='MEDIA_PLAY' and r['id'].startswith('V2-')):
            if not any(d['id']==r['id'] for d in demo):demo.append(r)
    write_json(output/'demo-user-requirements.json',demo)
    final=dict(command_gates_passed=command['synthetic_gates_passed'],wake_gates_passed=wake['wake_gates_passed'],
        synthetic_qualified=command['synthetic_gates_passed'] and wake['wake_gates_passed'],production_ready=False,
        human_qualified=False,physical_pi_measured=False,binding=command['binding'],
        command_results_sha256=file_hash(CACHE/'final-test/results.json'),wake_results_sha256=file_hash(CACHE/'wake-final-test/results.json'),
        third_engine_exhaustive=False,command_dia_clips=600,continuous_wake_generation_clips=288)
    write_json(output/'finalization.json',final)
    return {**result,**final}


def write_verbatim_guide():
    rows=read_rows(HERE/'verbatims.jsonl')
    lines=['# PCV synthetic v2: verbatims and expected typed commands','',
        'Expectations only; no measured acceptance is implied. Say "Hey Jarvis", pause for command_ready, then the verbatim. ',
        'Qwen and Parler cover every listed phrase; Dia covers a frozen balanced subset.','',
        '| ID | Phase | Use case | Verbatim | Expected JSON (null = reject) |','| --- | --- | --- | --- | --- |']
    lines += [f'| {r["id"]} | {r["phase"]} | {r["use_case"]} | {r["text"]} | `{canonical(r["expected_command"])}` |' for r in rows]
    path=rev.ROOT/'docs/PCV-SYNTHETIC-V2-VERBATIMS.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return str(path)
