"""Run the authorized long study, stopping on failed integrity or software gates."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from study import CACHE,HERE,ROOT,file_hash,read_rows,write_json,validate_frozen_study


def verify_stage(receipt):
    for name,digest in receipt.get('outputs',{}).items():
        if file_hash(Path(name))!=digest:raise ValueError('Completed stage output changed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True,type=Path)
    args=parser.parse_args();job=args.job.resolve()
    if not job.is_relative_to(CACHE):raise ValueError('Job must stay in the study cache')
    job.mkdir(parents=True,exist_ok=False)
    sources={str(p.relative_to(ROOT)):file_hash(p) for p in HERE.glob('*.py')}
    write_json(job/'started.json',dict(pid=os.getpid(),time=datetime.now(timezone.utc).isoformat(),
        source_hashes=sources,protocol_sha256=file_hash(HERE/'protocol.json'),python=sys.executable))
    state=dict(status='RUNNING',pid=os.getpid(),steps=[])

    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=job/'status.pending'
        write_json(temporary,state);temporary.replace(job/'status.json')

    def check():
        validate_frozen_study()
        for name,digest in sources.items():
            if file_hash(ROOT/name)!=digest:raise ValueError('Executing study source changed during the run')

    def launch(name,arguments,parler=False):
        check()
        env=os.environ.copy()
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        if parler:env['PYTHONPATH']=str(CACHE/'parler-site')+os.pathsep+env.get('PYTHONPATH','')
        log=(job/f'{name}.log').open('x',encoding='utf-8')
        command=[sys.executable,'-u',*map(str,arguments)]
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,pid=process.pid,command=command,status='RUNNING',started=time.time(),log=str(job/f'{name}.log'))
        state['steps'].append(row);save()
        return process,log,row

    def collect(item):
        process,log,row=item
        code=process.wait();log.close()
        row.update(exit_code=code,status='SUCCEEDED' if code==0 else 'FAILED',seconds=time.time()-row['started'])
        save()
        return code

    def stage(name,args):
        code=collect(launch(name,[HERE/'run.py',*args]))
        if code:raise RuntimeError(f'{name} failed with exit {code}; preserved evidence in {job}')

    save()
    try:
        # These two BF16 models fitted together during the measured generation pilot.
        # Dia FP32 and all recognition measurements run alone afterward.
        workers=[launch(e,[HERE/'synthesis.py',e],parler=e=='parler') for e in ('qwen','parler')]
        codes=[collect(worker) for worker in workers]
        if any(codes):raise RuntimeError('Qwen/Parler generation incomplete; inspect retained logs')
        code=collect(launch('dia',[HERE/'synthesis.py','dia']))
        if code:raise RuntimeError('Dia generation incomplete; inspect retained logs')
        stage('prepare',['prepare'])
        stage('training',['train','--output',CACHE/'training'])
        stage('export',['export','--run',CACHE/'training','--output',CACHE/'bundle-uncalibrated'])
        stage('waveform-parity',['parity','--run',CACHE/'training','--bundle',CACHE/'bundle-uncalibrated',
            '--output',CACHE/'waveform-parity'])
        stage('calibration',['calibrate','--bundle',CACHE/'bundle-uncalibrated','--output',CACHE/'calibration',
            '--parity',CACHE/'waveform-parity/results.json'])
        stage('final-test',['evaluate','--bundle',CACHE/'calibration/bundle','--split','test','--output',CACHE/'final-test'])
        stage('report',['report','--output',CACHE/'report'])
        results=json.loads((CACHE/'final-test/results.json').read_text())
        state.update(status='COMPLETE',synthetic_gates_passed=results['synthetic_gates_passed'],
            production_ready=False,report=str(CACHE/'report/benchmark.md'))
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc),production_ready=False)
        raise
    finally:save()


if __name__=='__main__':main()
