"""User-authorized checkpoint handoff to parallel development execution.

The frozen trainer is reused with a process-local development function override.
Its optimizer, RNG restoration, corpus, schedule and selection remain unchanged.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
from functools import partial
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import pcv_resume as recovery
from scripts import pcv_parallel

CACHE=recovery.CACHE
RUN=recovery.RUN
PLAN=ROOT/'experiments/pcv-delivery-v1/parallel-execution-20261001.json'
ARCHIVE=CACHE/'recoveries/parallel-20261001'
JOB=CACHE/'jobs/parallel-run-20261001'
VALIDATION=ROOT/'.cache/pcv-delivery-v1/parallel-eval-v2/results.json'
OLD_JOB=recovery.JOB/'status.json'


def handoff_position(run):
    run=Path(run);history=json.loads((run/'history.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    checkpoints=sorted((run/'seed-53').glob('epoch-*.pt'))
    if not checkpoints or (run/'seed-71').exists():raise ValueError('Expected first-seed checkpoint handoff')
    epoch=int(checkpoints[-1].stem.split('-')[-1])
    if (epoch not in protocol['selection_epochs'] or
        [(r['seed'],r['epoch']) for r in history]!=[(53,n) for n in range(1,epoch)] or
        (run/f'seed-53/dev-{epoch:03}.json').exists() or list(run.rglob('*.pending'))):
        raise ValueError('Wait for a completed checkpoint with pending development and no partial write')
    return 53,epoch


def require_parity(receipt):
    if (receipt.get('complete') is not True or receipt.get('recordings')!=5636 or receipt.get('errors')!=[]
        or receipt.get('score_comparisons',0)<60 or receipt.get('workers') not in (4,8)
        or receipt.get('score_parity')!={'4':True,'8':True}
        or receipt.get('measured_speedup',0)<1.5 or receipt.get('sealed_test_accessed') is not False):
        raise ValueError('Complete exact development parity and measured acceleration required')


def require_identity(expected,actual):
    if any(expected[k]!=actual[k] for k in ('pid','created','command')):
        raise ValueError('Process identity changed; do not suspend or terminate it')


def process_identity(process):
    return dict(pid=process.pid,created=process.create_time(),command=process.cmdline(),cwd=process.cwd(),ppid=process.ppid())


def authenticate_job(state,controller,launcher,worker):
    step=state['steps'][-1];first=state['steps'][0]['started']
    for process,stage in ((controller,'orchestrate'),(launcher,'train'),(worker,'train')):
        command=process['command'];cwd=Path(process['cwd']).resolve()
        if (len(command)<3 or command[-1]!=stage or cwd!=ROOT.resolve()
                or (cwd/command[-2]).resolve()!=Path(recovery.__file__).resolve()):
            raise ValueError('Process belongs to another checkout or program')
    if (controller['pid']!=state['pid'] or launcher['pid']!=step['pid']
            or launcher['ppid']!=controller['pid'] or worker['ppid']!=launcher['pid']
            or not first-30<=controller['created']<=first+1
            or not step['started']-5<=launcher['created']<=step['started']+1
            or not launcher['created']<=worker['created']<=step['started']+10):
        raise ValueError('Process creation times/tree do not match the saved job')


def verify_validation():
    receipt=json.loads(VALIDATION.read_text());require_parity(receipt)
    for name,sha in {**receipt['source_hashes'],**receipt['input_hashes']}.items():
        if recovery.digest(ROOT/name)!=sha:raise ValueError('Validated source/input changed: '+name)
    if recovery.digest(VALIDATION.parent/'rows.jsonl')!=receipt['rows_sha256']:
        raise ValueError('Parity comparison evidence changed')
    return receipt


@contextmanager
def execution_override(*,workers,validator,plan,archive):
    from pcvcm import training
    changes=[(training,'development',training.development),(recovery,'validate_recovery',recovery.validate_recovery),
             (recovery,'PLAN',recovery.PLAN),(recovery,'RECOVERY',recovery.RECOVERY)]
    try:
        training.development=partial(pcv_parallel.development,workers=workers)
        recovery.validate_recovery=validator;recovery.PLAN=plan;recovery.RECOVERY=archive
        yield
    finally:
        for module,name,value in reversed(changes):setattr(module,name,value)


def handoff():
    import psutil
    import torch
    recovery.validate_recovery();validation=verify_validation()
    if PLAN.exists() or ARCHIVE.exists() or JOB.exists():raise FileExistsError('Handoff already exists; preserve it')
    state=json.loads(OLD_JOB.read_text())
    if state['status']!='RUNNING' or state['steps'][-1]['name']!='train':raise ValueError('Original job no longer training')
    controller=psutil.Process(state['pid']);controller_id=process_identity(controller)
    if controller.cmdline()[-1]!='orchestrate' or Path(controller.cmdline()[-2]).name!='pcv_resume.py':
        raise ValueError('Unexpected original controller')
    launcher=psutil.Process(state['steps'][-1]['pid'])
    leaves=[p for p in launcher.children() if p.cmdline()[-1:]==['train']
            and len(p.cmdline())>=3 and Path(p.cmdline()[-2]).name=='pcv_resume.py']
    if len(leaves)!=1:raise ValueError('Cannot uniquely identify original training worker')
    worker=leaves[0];worker_id=process_identity(worker);launcher_id=process_identity(launcher)
    authenticate_job(state,controller_id,launcher_id,worker_id)
    paused=False;controller_stopped=False
    try:
        deadline=time.monotonic()+60
        while True:
            if time.monotonic()>deadline:raise TimeoutError('No safe pending-development boundary within 60 seconds')
            try:position=handoff_position(RUN)
            except ValueError:time.sleep(.2);continue
            require_identity(worker_id,process_identity(worker))
            authenticate_job(state,process_identity(controller),process_identity(launcher),process_identity(worker))
            worker.suspend();paused=True
            try:
                if position!=handoff_position(RUN) or (CACHE/'compute-budget.pending').exists():raise ValueError('Boundary moved')
                break
            except ValueError:worker.resume();paused=False;time.sleep(.2)
        recovery.validate_recovery();verify_validation()
        seed,epoch=position;checkpoint=RUN/f'seed-{seed}/epoch-{epoch:03}.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        protocol=json.loads((RUN/'protocol.json').read_text());history=json.loads((RUN/'history.json').read_text())
        if (saved['seed'],saved['epoch'])!=position or saved['protocol_sha256']!=recovery.repair.identity(protocol):
            raise ValueError('Checkpoint identity changed')
        from vcm.model import CausalVCM
        probe=CausalVCM();probe.load_state_dict(saved['weights'],strict=True)
        if len(saved['optimizer']['state'])!=len(list(probe.parameters())):raise ValueError('Incomplete optimizer state')
        recovery.pending_record(history,saved,checkpoint,recovery.digest(checkpoint),protocol['selection_epochs'])
        budget=json.loads((CACHE/'compute-budget.json').read_text())
        if [r['label'] for r in budget['epochs']]!=[f'53/{n}' for n in range(1,epoch+1)]:raise ValueError('Budget epoch mismatch')
        for row in history:
            if recovery.digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Old checkpoint changed')
            if 'predictions' in row and recovery.digest(row['predictions'])!=row['predictions_sha256']:
                raise ValueError('Old development output changed')
        for path in (RUN/'selection.json',CACHE/'bundle-uncalibrated',CACHE/'test-access',CACHE/'wake-test-access'):
            if path.exists():raise ValueError('Cannot repeat selected/exported/sealed stages')
        ARCHIVE.mkdir(parents=True);snapshots={}
        for name,path in [('history-before.json',RUN/'history.json'),('budget-before.json',CACHE/'compute-budget.json'),('job-before.json',OLD_JOB)]:
            shutil.copyfile(path,ARCHIVE/name);snapshots[name]=recovery.digest(ARCHIVE/name)
        sources=[Path(__file__).resolve(),Path(pcv_parallel.__file__),Path(recovery.__file__)]
        plan=dict(version='pcv-parallel-execution-20261001',time_utc=datetime.now(timezone.utc).isoformat(),
            authorization='User said Do it after diagnosis and recommendation to parallelize independent development recordings',
            sources={str(p.relative_to(ROOT)):recovery.digest(p) for p in sources},
            validation_sha256=recovery.digest(VALIDATION),workers=validation['workers'],
            prior_recovery_plan_sha256=recovery.digest(recovery.PLAN),
            frozen_study_protocol_sha256=recovery.digest(recovery.repair.HERE/'protocol.json'),
            prepared_freeze_sha256=recovery.digest(CACHE/'prepared/freeze.json'),
            original_training_files={str(p.relative_to(ROOT)):recovery.digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='history.json'},
            snapshots=snapshots,original_history_sha256=recovery.digest(RUN/'history.json'),
            original_budget_sha256=recovery.digest(CACHE/'compute-budget.json'),original_lock_sha256=recovery.digest(CACHE/'compute-budget.lock'),
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=recovery.digest(checkpoint),seed=seed,epoch=epoch,
            retained_budget=budget,pending_development=True,missing_training_loss='Unavailable interrupted epoch mean loss is null; no optimization repeated',
            stopped_processes=[controller_id,worker_id],schedule_unchanged=True,selection_unchanged=True,sealed_test_accessed=False)
        recovery.repair.immutable_json(PLAN,plan)
        require_identity(controller_id,process_identity(controller));controller.terminate();controller.wait(timeout=10);controller_stopped=True
        require_identity(worker_id,process_identity(worker));worker.terminate();worker.wait(timeout=10);paused=False
        recovery.repair.immutable_json(ARCHIVE/'handoff.json',dict(controller_stopped=True,worker_stopped=True,
            plan_sha256=recovery.digest(PLAN),old_status_sha256=recovery.digest(OLD_JOB)))
        return plan
    finally:
        if paused and worker.is_running():
            # Before a completed handoff, leave original work capable of progressing.
            worker.resume()
            if controller_stopped:print('Original worker resumed without controller; inspect preserved handoff before any retry',flush=True)


def validate(initial=False):
    recovery.repair.validate();plan=json.loads(PLAN.read_text());verify_validation()
    for name,sha in {**plan['sources'],**plan['original_training_files']}.items():
        if recovery.digest(ROOT/name)!=sha:raise ValueError('Frozen handoff source/evidence changed: '+name)
    if recovery.digest(VALIDATION)!=plan['validation_sha256']:raise ValueError('Validation receipt changed')
    for name,sha in plan['snapshots'].items():
        if recovery.digest(ARCHIVE/name)!=sha:raise ValueError('Handoff snapshot changed')
    if recovery.digest(CACHE/'prepared/freeze.json')!=plan['prepared_freeze_sha256']:raise ValueError('Prepared freeze changed')
    if recovery.digest(recovery.repair.HERE/'protocol.json')!=plan['frozen_study_protocol_sha256']:raise ValueError('Study protocol changed')
    if initial:
        for path,key in [(RUN/'history.json','original_history_sha256'),(CACHE/'compute-budget.json','original_budget_sha256'),(CACHE/'compute-budget.lock','original_lock_sha256')]:
            if recovery.digest(path)!=plan[key]:raise ValueError('Handoff state changed before resume')
    return plan


def no_duplicate():
    import psutil
    ignored={os.getpid(),*[p.pid for p in psutil.Process().parents()]}
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            command=p.info['cmdline'] or []
            if p.pid in ignored or not (p.info['name'] or '').lower().startswith('python') or len(command)<2:continue
            if command[-1] in ('train','orchestrate') and Path(command[-2]).name in ('pcv_resume.py','pcv_parallel_run.py'):
                raise RuntimeError(f'Existing controller/worker {p.pid}; no duplicate launch')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue


def orchestrate():
    no_duplicate();validate(initial=True)
    with (ARCHIVE/'controller-started.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
    JOB.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',pid=os.getpid(),steps=[],execution_plan_sha256=recovery.digest(PLAN))
    def save():state['updated_utc']=datetime.now(timezone.utc).isoformat();recovery.atomic_json(JOB/'status.json',state)
    save()
    try:
        for stage in ('train','export','parity','calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report'):
            validate()
            script=Path(__file__).resolve() if stage=='train' else recovery.repair.HERE/'run.py'
            command=[sys.executable,'-u',str(script),stage]
            with (JOB/f'{stage}.log').open('x',encoding='utf-8') as log:
                p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row=dict(name=stage,pid=p.pid,command=command,status='RUNNING',started=time.time(),log=str(JOB/f'{stage}.log'))
                state['steps'].append(row);save();code=p.wait()
            row.update(exit_code=code,seconds=time.time()-row['started'],status='SUCCEEDED' if code==0 else 'FAILED');save()
            if code:raise RuntimeError(stage+' failed; execution receipts retained')
        final=json.loads((CACHE/'report/finalization.json').read_text());state.update(status='COMPLETE',synthetic_qualified=final['synthetic_qualified'],production_ready=False)
    except BaseException as exc:state.update(status='FAILED',error=repr(exc));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['handoff','train','orchestrate','status']);stage=parser.parse_args().stage
    if stage=='handoff':result=handoff()
    elif stage=='train':
        plan=validate(initial=True)
        with recovery.repair.bind(),execution_override(workers=plan['workers'],validator=validate,plan=PLAN,archive=ARCHIVE):result=recovery.train()
    elif stage=='orchestrate':orchestrate();return
    else:result=json.loads((JOB/'status.json').read_text())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
