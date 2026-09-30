"""Produce command-by-command observed outputs without turning expectations into passes."""
from collections import defaultdict
import json
from pathlib import Path
from study import CACHE,HERE,PCV,canonical,file_hash,read_rows,write_json,write_rows


def assemble_outputs(verbatims, predictions):
    grouped=defaultdict(list)
    seen=set()
    for row in predictions:
        if row['recording_id'] in seen:raise ValueError('Duplicate prediction')
        seen.add(row['recording_id']);grouped[row['verbatim_id']].append(row)
    refs={r['id']:r for r in verbatims}
    if set(grouped)-set(refs):raise ValueError('Prediction has no frozen verbatim')
    result=[]
    for ref in verbatims:
        actual=[]
        for row in grouped[ref['id']]:
            if canonical(row['expected_command'])!=canonical(ref['expected_command']):
                raise ValueError('Prediction expectation differs from frozen verbatim')
            actual.append(dict(recording_id=row['recording_id'],engine=row['engine'],
                requested_accent=row['accent_requested'],audio=row.get('audio'),
                actual_output=row['output'],raw_candidate=row['candidate'],
                exact_correct=(row['output']['accepted'] and row['output']['command']==ref['expected_command'])
                    if ref['expected_command'] is not None else
                    (not row['output']['accepted'] and row['endpoint_detected'] and not row['capture_fault']),
                capture_fault=row['capture_fault'],endpoint_detected=row['endpoint_detected']))
        result.append({**ref,'status':'MEASURED' if actual else 'NOT_RUN','observations':actual})
    return result


def report(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Reports are immutable')
    final=CACHE/'final-test'
    results=json.loads((final/'results.json').read_text())
    predictions=read_rows(final/'predictions.jsonl')
    if file_hash(final/'predictions.jsonl')!=results['predictions_sha256']:
        raise ValueError('Prediction receipt changed')
    refs=read_rows(HERE/'verbatims.jsonl')
    plan={r['recording_id']:r for r in read_rows(HERE/'generation-plan.jsonl')}
    for row in predictions:row['audio']=str(CACHE/plan[row['recording_id']]['audio'])
    suite=assemble_outputs(refs,predictions)
    demo_ids=set(json.loads((PCV/'reports/demo-suite/commands.json').read_text())['demo_test_ids'])
    output.mkdir(parents=True)
    write_rows(output/'all-verbatims-with-outputs.jsonl',suite)
    write_json(output/'demo-outputs.json',[r for r in suite if r['parent_id'] in demo_ids])
    failed=[r for r in suite if any(not x['exact_correct'] for x in r['observations'])]
    write_json(output/'failed-verbatims.json',failed)
    write_json(output/'metrics.json',results)
    lines=['# Synthetic ONNX Runtime benchmark','',
        f"Synthetic gates passed: **{results['synthetic_gates_passed']}**. Human/Pi qualification: **not measured**.",'',
        'Every output below is from retained waveform replay through CPU ONNX Runtime. The requested accents and synthesis text have not been independently verified by a listener.','',
        '| Phase | Positive exact accepts | Accepted precision | Negative false accepts | Invalid trials |',
        '| --- | ---: | ---: | ---: | ---: |']
    for phase,m in results['phase'].items():
        precision='unavailable' if m['accepted_precision'] is None else f"{100*m['accepted_precision']:.2f}%"
        lines.append(f"| {phase} | {m['correct_accepted']}/{m['positives']} | {precision} | {m['negative_false_accepts']}/{m['negatives']} | {m['invalid_trials']} |")
    lines+=['','## Every verbatim and actual output','',
        'The raw candidate is diagnostic only. `actual_output` is the accepted command or explicit rejection.','']
    for row in suite:
        lines += [f"### {row['id']}",'',f"Say: **{row['text']}**",'',
                  f"Expected: `{canonical(row['expected_command'])}`",'']
        for observed in row['observations']:
            label=f"{observed['engine']} / {observed['requested_accent']}"
            lines.append(f"- [{label}]({Path(observed['audio']).as_posix()}) → `{canonical(observed['actual_output'])}`")
        lines.append('')
    (output/'benchmark.md').write_text('\n'.join(lines),encoding='utf-8')
    return dict(report=str(output/'benchmark.md'),complete=results['complete'],
        synthetic_gates_passed=results['synthetic_gates_passed'],failed_verbatims=len(failed),
        production_ready=False,human_qualified=False)
