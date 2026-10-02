"""Verify portable CPU deployment against the frozen study on development audio."""
from pathlib import Path
import os
import sys

# Apply CPU limits before importing NumPy, ONNX Runtime or any study modules.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v1'))
import hashlib
import json
import math
import time
from pcvcm.contract import canonical,file_hash,write_json

TOLERANCE=2e-5
TIMING_KEYS={'frontend_ms','encoder_ms','decoder_ms','endpoint_ms','gate_ms'}


def numerical_comparison(a,b,keys):
    """Finite diagnostics may differ within tolerance; availability must agree."""
    maximum=0.;equal=True
    for key in keys:
        if key not in a or key not in b:
            equal=False;continue
        x,y=a[key],b[key]
        if x is None and y is None:continue
        if (type(x) not in (int,float) or type(y) not in (int,float)
                or not math.isfinite(x) or not math.isfinite(y)):
            equal=False;continue
        error=abs(x-y);maximum=max(maximum,error)
        equal &= error<=TOLERANCE
    return dict(equal=bool(equal),max_error=maximum)


def compare_command(source,portable):
    a,b=source['diagnostic'],portable['diagnostic']
    numeric=numerical_comparison(a,b,('score','margin'))
    ignored=TIMING_KEYS|{'score','margin'}
    exact=all(canonical(source[k])==canonical(portable[k])
        for k in ('output','candidate','endpoint_detected','capture_fault'))
    exact &= canonical({k:v for k,v in a.items() if k not in ignored})==canonical(
        {k:v for k,v in b.items() if k not in ignored})
    return dict(equal=bool(exact and numeric['equal']),max_error=numeric['max_error'])


def compare_wake(source,portable):
    equal=len(source)==len(portable);maximum=0.
    for a,b in zip(source,portable):
        numeric=numerical_comparison(a,b,('score',)) if 'score' in a or 'score' in b else dict(equal=True,max_error=0.)
        maximum=max(maximum,numeric['max_error'])
        equal &= numeric['equal'] and canonical({k:v for k,v in a.items() if k!='score'})==canonical(
            {k:v for k,v in b.items() if k!='score'})
    return dict(equal=bool(equal),max_error=maximum)


def validate_development_rows(rows,expected):
    ids=[r['recording_id'] for r in rows]
    if (not ids or any(r.get('split')!='dev' for r in rows) or len(ids)!=len(set(ids))
            or set(ids)!=set(expected)):
        raise ValueError('Requires complete unique development coverage; sealed test replay is forbidden')


def verify_equivalence_inputs(source,portable):
    source=Path(source);manifest=json.loads((source/'synthetic-bundle.json').read_text(encoding='utf-8'))
    if (portable['manifest'].get('source_binding')!=manifest['binding'] or
            portable['manifest'].get('source_manifest_sha256')!=file_hash(source/'synthetic-bundle.json')):
        raise ValueError('Portable bundle belongs to another study candidate')
    for name in ('model.onnx','catalog.json','normalization.json','endpoint.json'):
        if file_hash(source/name)!=file_hash(portable['root']/name):raise ValueError('Different '+name)
    policy=json.loads((source/'policy.json').read_text(encoding='utf-8'))
    if any(canonical(policy.get(k))!=canonical(portable['policy'].get(k)) for k in ('enabled','thresholds')):
        raise ValueError('Deployment changed the measured policy')
    from scripts.runtime.pcv_deploy.bundle import WAKE_FILES
    for name in WAKE_FILES:
        if file_hash(ROOT/'artifacts'/name)!=file_hash(portable['root']/'wake'/name):
            raise ValueError('Deployment changed wake asset '+name)


def compare_recording(source_gate,portable_gate,row,*,mode):
    import numpy as np
    import soundfile as sf
    path=Path(row['audio'])
    if file_hash(path)!=row['audio_sha256']:raise ValueError('Changed audio file')
    pcm,rate=sf.read(path,dtype='float32')
    if (rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all()
            or hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']):
        raise ValueError('Changed audio PCM')
    if row.get('split')!='dev':raise ValueError('Only development audio may be replayed for portable parity')
    if mode=='command':
        from synthbench import replay
        source=replay(source_gate.command,row);portable=replay(portable_gate.command,row)
        comparison=compare_command(source,portable)
    elif mode=='wake':
        from wake_gate import replay_pcm
        results=[]
        for gate in (source_gate,portable_gate):
            began=time.perf_counter();events=replay_pcm(gate,pcm)
            results.append(dict(events=events,eof=gate.eof(),inference_seconds=time.perf_counter()-began))
        source,portable=results;comparison=compare_wake(source['events'],portable['events'])
        comparison['equal'] &= canonical(source['eof'])==canonical(portable['eof'])
    else:raise ValueError('Unknown comparison mode')
    if file_hash(path)!=row['audio_sha256']:raise ValueError('Audio changed during replay')
    return dict(recording_id=row['recording_id'],mode=mode,audio=str(path.resolve()),
        audio_sha256=row['audio_sha256'],pcm_sha256=row['pcm_sha256'],
        source=source,portable=portable,**comparison)


def development_inputs():
    """Only frozen dev assignments; no threshold/model selection and no test replay."""
    import revision as rev
    from synthbench import load_split
    from wake_benchmark import audit_wake
    commands=load_split('dev');wake=audit_wake('dev')
    plan=rev.read_rows(rev.HERE/'generation-plan.jsonl')
    wake_plan=rev.read_rows(rev.HERE/'wake-generation-plan.jsonl')
    validate_development_rows(commands,{r['recording_id'] for r in plan if r['split']=='dev'})
    validate_development_rows(wake,{r['recording_id'] for r in wake_plan if r['split']=='dev'})
    by_id={r['recording_id']:r for r in commands}
    for row in list(wake):
        if row['case_type']=='wake_command':
            key=f'{row["engine"]}/{row["voice_group"]}/{row["command_verbatim_id"]}'
            wake.append({**by_id[key],'recording_id':'no-wake/'+key})
    for row in [*commands,*wake]:row['audio']=str((rev.CACHE/row['audio']).resolve())
    return commands,wake


def expected_recordings():
    definitions=ROOT/'experiments/pcv-synthetic-v2'
    def read(name):return [json.loads(line) for line in (definitions/name).read_text(encoding='utf-8').splitlines()]
    commands={r['recording_id'] for r in read('generation-plan.jsonl') if r['split']=='dev'}
    wake=[r for r in read('wake-generation-plan.jsonl') if r['split']=='dev']
    references={r['id']:r for r in read('wake-verbatims.jsonl')}
    wake_ids={r['recording_id'] for r in wake}
    for row in wake:
        ref=references[row['verbatim_id']]
        if ref['case_type']=='wake_command':
            key=f'{row["engine"]}/{row["voice_group"]}/{ref["command_verbatim_id"]}'
            if key not in commands:raise ValueError('Missing no-wake development assignment')
            wake_ids.add('no-wake/'+key)
    return dict(command=commands,wake=wake_ids)


def runtime_source_hashes():
    paths=[Path(__file__),ROOT/'scripts/pcv_study_context.py',*list((ROOT/'scripts/runtime/pcv_deploy').glob('*.py'))]
    return {str(p.resolve()):file_hash(p) for p in paths}


def verify_receipt(path,portable):
    """Fail publication on incomplete parity, substituted scores or adapter drift."""
    if path is None:raise ValueError('Enabled publication requires a complete portable parity receipt')
    path=Path(path);report=json.loads(path.read_text(encoding='utf-8'))
    if (report.get('complete') is not True or report.get('parity_passed') is not True
            or report.get('split')!='dev' or report.get('test_audio_replayed') is not False
            or report.get('policy_changed') is not False or report.get('tolerance')!=TOLERANCE
            or report.get('source_binding')!=portable['manifest'].get('source_binding')
            or report.get('deployment_binding')!=portable['binding']
            or report.get('deployment_manifest_sha256')!=file_hash(portable['root']/'deployment.json')):
        raise ValueError('Incomplete, failed or differently bound portable parity receipt')
    if report.get('source_files')!=runtime_source_hashes():raise ValueError('Portable parity source changed')
    rows_path=path.parent/'rows.jsonl'
    if file_hash(rows_path)!=report.get('rows_sha256'):raise ValueError('Portable parity rows changed')
    expected=expected_recordings();seen={mode:set() for mode in expected};maximum=0.
    with rows_path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);mode=row['mode'];key=row['recording_id']
            if mode not in expected or key not in expected[mode] or key in seen[mode]:
                raise ValueError('Portable parity coverage changed')
            seen[mode].add(key)
            if mode=='command':comparison=compare_command(row['source'],row['portable'])
            else:
                comparison=compare_wake(row['source']['events'],row['portable']['events'])
                comparison['equal'] &= canonical(row['source']['eof'])==canonical(row['portable']['eof'])
            if (row.get('equal') is not True or not comparison['equal']
                    or row.get('max_error')!=comparison['max_error']):
                raise ValueError('Portable parity mismatch in saved outputs')
            maximum=max(maximum,comparison['max_error'])
    counts={mode:len(ids) for mode,ids in expected.items()}
    if (seen!=expected or any(n==0 for n in counts.values()) or report.get('expected')!=counts
            or report.get('observed')!=counts or report.get('mismatches')!=0 or report.get('max_error')!=maximum):
        raise ValueError('Incomplete portable parity coverage or summary')
    return report


def evaluate(source,portable,output,*,diagnostic=False):
    source,portable,output=Path(source),Path(portable),Path(output)
    if output.exists():raise FileExistsError('Parity receipts are immutable; preserve prior failures')
    from scripts.runtime.pcv_deploy.bundle import load_bundle
    from scripts.pcv_study_context import bind_study
    with bind_study(source):
        from wake_gate import from_bundle
        reference=from_bundle(source,ROOT/'artifacts')
        deployed=load_bundle(portable,allow_diagnostic=diagnostic)
        verify_equivalence_inputs(source,deployed)
        commands,wake=development_inputs()
        counts=dict(command=len(commands),wake=len(wake))
        output.mkdir(parents=True)
        source_files=runtime_source_hashes()
        started=dict(complete=False,source_binding=reference.command.binding,deployment_binding=deployed['binding'],
            source_manifest_sha256=file_hash(source/'synthetic-bundle.json'),
            deployment_manifest_sha256=file_hash(portable/'deployment.json'),source_files=source_files,
            split='dev',expected=counts,tolerance=TOLERANCE,physical_pi_measured=False,
            test_audio_replayed=False,accuracy_qualification=False,policy_changed=False)
        write_json(output/'started.json',started)
        observed=dict(command=0,wake=0);mismatches=0;maximum=0.
        try:
            with (output/'rows.jsonl').open('x',encoding='utf-8') as stream:
                for mode,rows in (('command',commands),('wake',wake)):
                    for row in rows:
                        result=compare_recording(reference,deployed['gate'],row,mode=mode)
                        stream.write(canonical(result)+'\n');stream.flush()
                        observed[mode]+=1;mismatches+=int(not result['equal']);maximum=max(maximum,result['max_error'])
                        if observed[mode]%100==0:print('PORTABLE_PARITY',mode,observed[mode],'/',counts[mode],flush=True)
            result={**started,'complete':observed==counts,'observed':observed,'mismatches':mismatches,
                'max_error':maximum,'parity_passed':observed==counts and mismatches==0,
                'rows_sha256':file_hash(output/'rows.jsonl')}
            write_json(output/'results.json',result)
            return result
        except Exception as error:
            write_json(output/'failure.json',dict(error=repr(error),complete=False,observed=observed,mismatches=mismatches))
            raise


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--portable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostic',action='store_true')
    args=parser.parse_args()
    result=evaluate(args.source,args.portable,args.output,diagnostic=args.diagnostic)
    print(json.dumps(result,indent=2));return 0 if result['parity_passed'] else 1


if __name__=='__main__':sys.exit(main())
