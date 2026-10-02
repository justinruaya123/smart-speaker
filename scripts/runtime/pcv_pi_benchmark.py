"""Replay and user-operated Pi measurements with explicit missing denominators."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from scripts.runtime.pcv_final_runtime import Runtime,read_wav,read_json,sha,capture
from scripts.runtime.pcv_schema import INTENTS,canonical,output


def metrics(rows,key):
    positives=[r for r in rows if r['expected'] is not None]
    negatives=[r for r in rows if r['expected'] is None]
    exact=sum(r[key] is not None and canonical(r[key])==canonical(r['expected']) for r in positives)
    accepted=sum(r[key] is not None for r in rows)
    fa=sum(r[key] is not None for r in negatives)
    slots={}
    for r in positives:
        for name,expected in r['expected']['slots'].items():
            item=slots.setdefault(name,{'reference_n':0,'correct':0});item['reference_n']+=1
            actual=r[key]
            item['correct']+=int(actual is not None and actual['intent']==r['expected']['intent'] and
                canonical(actual['slots'].get(name))==canonical(expected))
    return {'n':len(rows),'positive_n':len(positives),'negative_n':len(negatives),'exact':exact,
            'accepted':accepted,'recall':exact/len(positives) if positives else None,
            'precision':exact/accepted if accepted else None,'negative_errors':fa,
            'false_accept_rate':fa/len(negatives) if negatives else None,'slots':slots}


def typed_metrics(rows):
    typed=[r for r in rows if r.get('expected_output') is not None]
    return {'n':len(typed),'exact':sum(canonical(r['event'].get('output'))==canonical(r['expected_output']) for r in typed),
            'reason_unannotated_n':len(rows)-len(typed)}


def summarize(rows,definitions):
    result={'accepted':metrics(rows,'accepted'),'diagnostic':metrics(rows,'candidate'),
            'typed_output':typed_metrics(rows)}
    result['per_intent']={}
    for intent in INTENTS:
        references=[r for r in rows if r['expected'] and r['expected']['intent']==intent]
        result['per_intent'][intent]={}
        for key in ('accepted','candidate'):
            item=metrics(references,key)
            predicted=sum(r[key] is not None and r[key]['intent']==intent for r in rows)
            item.update({'predicted_n':predicted,'accepted':predicted,
                         'precision':item['exact']/predicted if predicted else None})
            result['per_intent'][intent][key]=item
    result['phrases']={}
    for definition in definitions:
        trials=[r for r in rows if r.get('phrase_id')==definition['id']]
        result['phrases'][definition['id']]={'trials':len(trials),'typed_output':typed_metrics(trials),
            **{key:metrics(trials,key) for key in ('accepted','candidate')}}
    result['untested_phrase_count']=sum(v['trials']==0 for v in result['phrases'].values())
    result['defined_phrase_count']=len(definitions)
    def quantiles(values):
        return {'n':len(values),'p50':float(np.percentile(values,50)) if values else None,
                'p95':float(np.percentile(values,95)) if values else None}
    result['latency_ms']=quantiles([r['event']['timing']['total_ms'] for r in rows if 'timing' in r['event']])
    result['rtf']=quantiles([r['event']['timing']['rtf'] for r in rows if r['event'].get('timing',{}).get('rtf') is not None])
    return result


def summarize_session(rows,definitions,*,guided):
    """Keep live acoustic, live-attempt and prerecorded evidence disjoint."""
    replay=[r for r in rows if r['mode']=='replay']
    attempts=[r for r in rows if r['mode']=='guided_live']
    completed=[r for r in attempts if r.get('capture_completed',True)]
    strata={'replay':summarize(replay,definitions),'guided_live':summarize(completed,definitions)}
    selected='guided_live' if guided else 'replay'
    result={**strata[selected],'selected_metric_mode':selected,'strata':strata}
    # An initiated positive trial without a completed capture remains a miss in
    # this end-to-end denominator. It cannot create acoustic phrase coverage.
    attempt_metrics=metrics(attempts,'accepted')
    typed=[r for r in attempts if r.get('expected_output') is not None]
    result['live_attempts']={**attempt_metrics,'completed_n':len(completed),
        'incomplete_capture_n':len(attempts)-len(completed),
        'no_completed_command_n':sum(r['event'].get('capture_status')=='no_completed_command' for r in attempts),
        'capture_fault_n':sum(r['event'].get('output',{}).get('reason')=='capture_fault' for r in attempts),
        'typed_output_n':len(typed),
        'typed_output_exact':sum(r.get('capture_completed',True) and r['event'].get('output',{}).get('reason')!='capture_fault'
            and canonical(r['event'].get('output'))==canonical(r['expected_output']) for r in typed),
        'per_phrase':{d['id']:{'attempts':sum(r.get('phrase_id')==d['id'] for r in attempts),
                              **metrics([r for r in attempts if r.get('phrase_id')==d['id']],'accepted')} for d in definitions}}
    result['live_microphone_attempted']=bool(attempts)
    result['live_microphone_measured']=bool(completed)
    return result


def hardware():
    model='unavailable'
    for name in ('/sys/firmware/devicetree/base/model','/proc/device-tree/model'):
        if Path(name).exists(): model=Path(name).read_bytes().decode(errors='replace').strip('\x00');break
    return {'os':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),
            'python':platform.python_version(),'cpu_count':os.cpu_count(),'device_model':model,
            'physical_pi_measured':'Raspberry Pi' in model}


def peak_rss_mb():
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/(1024*1024 if sys.platform=='darwin' else 1024)
    except ImportError: return None


def write_report(folder,result):
    (folder/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# PCV device test report','',f"Measured UTC: {result['utc']}",
           f"Device: {result['hardware']['device_model']} / {result['hardware']['os']}",
           f"Physical Raspberry Pi measurement: {result['hardware']['physical_pi_measured']}",
           f"Mode: {result['mode']}; acceptance policy enabled: {result['policy_enabled']}",'',
           f"Headline metric mode: **{result['selected_metric_mode']}**. Guided headlines contain completed live captures only; replay cannot increase live phrase coverage.",'',
           '## Session resources','', '| Metric | Result |','|---|---|',
           f"| VCM session load ms | {result['session_load_ms']} |",
           f"| Wake/VAD session loads ms | {result['wake']['session_load_ms']} |",
           f"| Peak process RSS MiB | {result['peak_rss_mb']} |",'',
           'Inference latency includes normalization/frontend, ONNX heads, finite scoring and validation. It excludes disk read, utterance duration, wake listening and the 400 ms endpoint silence. RTF divides inference wall time by supplied command duration. The first inference is included; no warmup samples are discarded.',
           'Diagnostic scores are not accepted commands. A disabled policy rejects all speech; perfect negative rejection in that mode is not model qualification. Unrecorded definitions are untested.',
           'Bundled replay is a bounded previously observed smoke subset. It does not replace held-out qualification. Guided expected labels describe requested speech; actual human wording is not verified automatically.', '',
           '## Guided live attempts, including capture failures','']
    attempt=result['live_attempts']
    lines.extend([f"Initiated: {attempt['n']}; completed captures: {attempt['completed_n']}; incomplete captures: {attempt['incomplete_capture_n']}; no completed command: {attempt['no_completed_command_n']}; capture faults: {attempt['capture_fault_n']}.",
        f"End-to-end accepted exact commands: **{attempt['exact']}/{attempt['positive_n']} positive attempts**. A wake or capture failure remains a miss here.",
        f"Full expected output (including rejection reason): {attempt['typed_output_exact']}/{attempt['typed_output_n']} annotated attempts.",
        'Failed captures do not create recorded acoustic coverage. Nonacceptance on a capture fault is not a successful typed negative. Skipped and unstarted prompts remain untested.',''])
    for key,title in [('guided_live','Completed live captures'),('replay','Prerecorded replay — separate evidence')]:
        data=result['strata'][key];a,d=data['accepted'],data['diagnostic']
        lines.extend([f'## {title}','','| Metric | Result |','|---|---|',
            f"| Accepted exact commands | {a['exact']}/{a['positive_n']} |",
            f"| Accepted negative examples (safety count) | {a['negative_errors']}/{a['negative_n']} |",
            f"| Full typed output exact | {data['typed_output']['exact']}/{data['typed_output']['n']} |",
            f"| Diagnostic candidate exact commands | {d['exact']}/{d['positive_n']} |",
            f"| Missing phrase recordings in this mode | {data['untested_phrase_count']}/{data['defined_phrase_count']} |",
            f"| Inference latency ms p50 / p95 (n={data['latency_ms']['n']}) | {data['latency_ms']['p50']} / {data['latency_ms']['p95']} |",
            f"| RTF p50 / p95 | {data['rtf']['p50']} / {data['rtf']['p95']} |",''])
        lines.extend(['Recorded phrases in this mode (full typed output includes rejection reason):','',
            '| Phrase ID | Recorded trials | Typed exact / annotated | Reason unannotated |',
            '|---|---|---|---|'])
        for phrase,item in data['phrases'].items():
            if item['trials']:
                typed=item['typed_output']
                lines.append(f"| {phrase} | {item['trials']} | {typed['exact']}/{typed['n']} | {typed['reason_unannotated_n']} |")
        lines.extend(['',f"All {data['defined_phrase_count']} phrase definitions and their denominators are in metrics.json; zero recordings remain untested.",''])
    lines.extend([f"## Per-intent complete commands — {result['selected_metric_mode']}",'',
                  '| Intent | Accepted exact / positive | Candidate exact / positive |','|---|---|---|'])
    for intent,values in result['per_intent'].items():
        x,y=values['accepted'],values['candidate']
        lines.append(f"| {intent} | {x['exact']}/{x['positive_n']} | {y['exact']}/{y['positive_n']} |")
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    root=Path(args.root).resolve();folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=False)
    definitions=read_json(root/'verbatims/definitions.json')
    runtime=Runtime(getattr(args,'bundle',None) or root/'models/default',diagnostic=True,threads=args.threads,engine=getattr(args,'engine','onnxruntime'))
    seconds=args.seconds if args.seconds is not None else (5 if args.no_wake else 25)
    rows=[];untested=[];wake_loads=[];cpu_start=time.process_time();wall_start=time.perf_counter()
    def append(reference,event,mode,*,completed=True):
        candidate=event.get('diagnostic_candidate') or {}
        row={'id':reference['id'],'phrase_id':reference.get('phrase_id'),
             'expected':reference.get('expected',reference.get('command')),
             'expected_output':reference.get('expected_output'),
             'accepted':event.get('output',{}).get('command'),'candidate':candidate.get('candidate'),
             'event':event,'mode':mode,'capture_completed':completed}
        rows.append(row)
        with (folder/'trials.jsonl').open('a',encoding='utf-8') as stream: stream.write(canonical(row)+'\n')
        print(canonical(event),flush=True)
    manifest=Path(args.manifest) if args.manifest else root/'samples/manifest.json'
    for reference in read_json(manifest):
        path=manifest.parent/reference['audio']
        if not path.exists(): untested.append({'id':reference['id'],'reason':'missing_audio','mode':'replay'});continue
        if sha(path)!=reference['sha256']: raise ValueError('Sample identity mismatch: '+reference['id'])
        append(reference,runtime.recognize(read_wav(path)),'replay')
    if args.guided:
        chosen=[d for d in definitions if not args.case or d['id'] in args.case]
        for definition in chosen:
            print(f"{definition['id']}: say Hey Jarvis, wait for command_ready, then: {definition.get('spoken_text',definition.get('text'))}",file=sys.stderr)
            answer=input('Enter to record; s to skip; q to stop: ').strip().lower()
            if answer=='q':break
            if answer=='s':untested.append({'id':definition['id'],'reason':'user_skipped','mode':'guided_live'});continue
            captured=[]
            for event in capture(runtime,args.mic,seconds=seconds,wake=not args.no_wake):
                print(canonical(event),flush=True)
                if event.get('stage')=='wake' and event.get('status')=='ready':wake_loads.append(event['session_load_ms'])
                if event.get('event')=='command':captured.append(event)
            ref={**definition,'phrase_id':definition['id']}
            if captured:append(ref,captured[0],'guided_live',completed='audio_seconds' in captured[0])
            else:append(ref,{'output':output(None,'capture_fault'),'diagnostic_candidate':None,
                             'capture_status':'no_completed_command'},'guided_live',completed=False)
    result=summarize_session(rows,definitions,guided=args.guided)
    wall=time.perf_counter()-wall_start;cpu=time.process_time()-cpu_start
    result.update({'utc':datetime.now(timezone.utc).isoformat(),'hardware':hardware(),
        'mode':'replay_and_guided_live' if args.guided else 'recorded_replay','untested_records':untested,
        'requested_live_case_ids':[d['id'] for d in definitions if args.guided and (not args.case or d['id'] in args.case)],
        'policy_enabled':runtime.policy['enabled'],'engine':runtime.engine,'model_binding':runtime.recipe['binding'],
        'runtime_sha256':sha(Path(__file__).with_name('pcv_final_runtime.py')),
        'model_bytes':sum(p.stat().st_size for p in (runtime.root/'ncnn').glob('model.ncnn.*')) if runtime.engine=='ncnn' else (runtime.root/'model.onnx').stat().st_size,
        'all_onnx_bytes':sum(p.stat().st_size for p in runtime.root.rglob('*.onnx')),
        'session_load_ms':runtime.load_ms,'threads':args.threads,'providers':runtime.session.get_providers(),
        'wake':{'enabled':args.guided and not args.no_wake,'phrase':'hey jarvis','threshold':.5,
                'release_ms':320,'endpoint_silence_ms':400,'replay_wake_tested':False,'capture_seconds':seconds,
                'session_load_ms':wake_loads},
        'cpu_seconds':cpu,'wall_seconds':wall,'cpu_percent_one_core':100*cpu/wall if wall else None,
        'peak_rss_mb':peak_rss_mb(),
        'qualification_changed':False})
    write_report(folder,result)
    print(f"Wrote {folder/'metrics.json'} and {folder/'REPORT.md'}",file=sys.stderr)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--engine',choices=['ncnn','onnxruntime'])
    p.add_argument('--guided',action='store_true');p.add_argument('--mic',type=int)
    p.add_argument('--case',action='append');p.add_argument('--seconds',type=float)
    p.add_argument('--no-wake',action='store_true');args=p.parse_args()
    if args.guided and args.mic is None:p.error('--guided requires an explicitly selected --mic index')
    from scripts.runtime.speaker_config import resolve_config
    cfg,_=resolve_config(args.root,overrides={'engine':args.engine},profile='benchmark');args.engine=cfg['inference']['engine']
    run(args)


if __name__=='__main__':main()
