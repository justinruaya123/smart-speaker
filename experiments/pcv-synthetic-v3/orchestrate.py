"""One v3 continuation owner; waits for the existing generator without touching it."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_pcv_v3_stage_cli',HERE/'run.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
wf=cli.wf
PLAN=HERE/'execution-plan.json'


def freeze_execution():
    # Freeze adapter code before launching a waiting controller. Data hashes are
    # added to the separate study protocol only after generation finishes.
    wf.generation_identity()
    sources=[HERE/'workflow.py',HERE/'run.py',Path(__file__),*HERE.glob('tests/*.py'),
        wf.ROOT/'scripts/pcv_parallel.py',wf.ROOT/'scripts/pcv_atomic_io.py']
    plan=dict(version='pcv-synthetic-v3-execution',generation_protocol_sha256=wf.file_hash(HERE/'generation-protocol.json'),
        sources={str(p.relative_to(wf.ROOT)):wf.file_hash(p) for p in sources},
        generation_job=str(wf.CACHE/'jobs/verified-parler-run-20261001'),stages=list(cli.STAGES),
        no_generator_restart=True,original_model_and_gates_unchanged=True,automatic_test_retry=False,
        budget_source=str(wf.PARENT/'compute-budget.json'),budget_sha256=wf.file_hash(wf.PARENT/'compute-budget.json'))
    wf.immutable_json(PLAN,plan);return plan


def validate_execution():
    p=json.loads(PLAN.read_text());wf.verify_inputs(p['sources'])
    if p['generation_protocol_sha256']!=wf.file_hash(HERE/'generation-protocol.json') or p['budget_sha256']!=wf.file_hash(p['budget_source']):
        raise ValueError('Generation contract or inherited budget changed')
    return p


def generation_state(plan):
    import psutil
    job=Path(plan['generation_job']);state=json.loads((job/'status.json').read_text())
    if state['protocol_sha256']!=plan['generation_protocol_sha256']:raise ValueError('Wrong generator identity')
    if state['status']=='FAILED':raise RuntimeError('Generation failed; existing evidence preserved')
    if state['status']=='SUCCEEDED':
        wf.require_generation_complete();return 'ready'
    if state['status']!='RUNNING':raise ValueError('Unknown generation status')
    try:
        process=psutil.Process(state['pid']);command=process.cmdline()
    except psutil.NoSuchProcess:raise RuntimeError('Generator disappeared; preserve interrupted evidence')
    joined=' '.join(command).replace('\\','/')
    if 'pcv-synthetic-v3/generation.py' not in joined or 'verified-parler-run-20261001' not in joined:
        raise RuntimeError('Generator PID identity differs')
    return 'waiting'


def run(job):
    plan=validate_execution();job=Path(job).resolve()
    if not job.is_relative_to((wf.CACHE/'jobs').resolve()):raise ValueError('Job escapes v3 jobs')
    job.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',stage='waiting-generation',pid=os.getpid(),steps=[],job=str(job),
        execution_plan_sha256=wf.file_hash(PLAN))
    with (wf.CACHE/'pipeline-started.json').open('x',encoding='utf-8') as f:
        json.dump(state,f);f.flush();os.fsync(f.fileno())
    def save():state['updated_utc']=datetime.now(timezone.utc).isoformat();wf.atomic_json(job/'status.json',state)
    save()
    try:
        while generation_state(plan)!='ready':
            time.sleep(30);validate_execution();save()
        for name in plan['stages']:
            validate_execution()
            if wf.file_hash(PLAN)!=state['execution_plan_sha256']:raise ValueError('Execution plan changed')
            command=[sys.executable,'-u',str(HERE/'run.py'),name]
            env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            with (job/f'{name}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=wf.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                row=dict(name=name,pid=process.pid,command=command,started=time.time(),log=str(job/f'{name}.log'),status='RUNNING')
                state['steps'].append(row);state['stage']=name;save();code=process.wait()
            row.update(exit_code=code,seconds=time.time()-row['started'],status='SUCCEEDED' if code==0 else 'FAILED');save()
            if code:raise RuntimeError(f'{name} failed; evidence preserved')
        final=json.loads((wf.CACHE/'report/finalization.json').read_text())
        state.update(status='COMPLETE',synthetic_qualified=final['synthetic_qualified'],production_ready=False)
    except BaseException as exc:state.update(status='FAILED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--job',type=Path);a=p.parse_args()
    if a.freeze:print(json.dumps(freeze_execution(),indent=2))
    elif a.job:run(a.job)
    else:p.error('--freeze or --job required')
