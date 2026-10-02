"""Resume the preserved parallel job after Windows atomic-file locking failure."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import pcv_parallel_run as parent
from scripts import pcv_atomic_io as atomic

recovery=parent.recovery
CACHE=parent.CACHE
RUN=parent.RUN
PLAN=ROOT/'experiments/pcv-delivery-v1/parallel-io-repair-20261001.json'
ARCHIVE=CACHE/'recoveries/parallel-io-repair-20261001'
JOB=CACHE/'jobs/parallel-io-repair-run-20261001'
FAILED=parent.JOB/'status.json'


def absent_lock(lock,archive,expected):
    if expected is not None or Path(lock).exists():raise ValueError('Expected cleanly released failed-job budget lock')


def freeze():
    import torch
    parent.no_duplicate();parent.validate()
    if PLAN.exists() or ARCHIVE.exists() or JOB.exists():raise FileExistsError('Repair already frozen')
    failed=json.loads(FAILED.read_text())
    if failed['status']!='FAILED' or failed['steps'][-1]['name']!='train':raise ValueError('Expected retained training failure')
    failure_log=Path(failed['steps'][-1]['log'])
    if 'PermissionError: [WinError 5]' not in failure_log.read_text():raise ValueError('Unexpected failure; diagnose separately')
    absent_lock(CACHE/'compute-budget.lock',None,None)
    if (CACHE/'compute-budget.pending').exists():raise ValueError('Preserve pending write before separate repair')
    seed,epoch=parent.handoff_position(RUN);checkpoint=RUN/f'seed-{seed}/epoch-{epoch:03}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    protocol=json.loads((RUN/'protocol.json').read_text());history=json.loads((RUN/'history.json').read_text())
    if saved['protocol_sha256']!=recovery.repair.identity(protocol) or (saved['seed'],saved['epoch'])!=(seed,epoch):raise ValueError('Checkpoint changed')
    recovery.pending_record(history,saved,checkpoint,recovery.digest(checkpoint),protocol['selection_epochs'])
    budget=json.loads((CACHE/'compute-budget.json').read_text())
    if [r['label'] for r in budget['epochs']]!=[f'53/{n}' for n in range(1,epoch+1)]:raise ValueError('Budget epochs changed')
    ARCHIVE.mkdir(parents=True);snapshots={}
    for name,path in [('history-before.json',RUN/'history.json'),('budget-before.json',CACHE/'compute-budget.json'),('failed-status.json',FAILED)]:
        shutil.copyfile(path,ARCHIVE/name);snapshots[name]=recovery.digest(ARCHIVE/name)
    plan=dict(version='pcv-parallel-io-repair-20261001',time_utc=datetime.now(timezone.utc).isoformat(),
        reason='Bounded retry of Windows sharing/access conflicts during atomic budget writes; failed job and gates retained',
        parent_plan_sha256=recovery.digest(parent.PLAN),failed_status_sha256=recovery.digest(FAILED),failure_log_sha256=recovery.digest(failure_log),
        sources={str(p.relative_to(ROOT)):recovery.digest(p) for p in [Path(__file__).resolve(),Path(atomic.__file__).resolve()]},
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=recovery.digest(checkpoint),seed=seed,epoch=epoch,
        workers=json.loads(parent.PLAN.read_text())['workers'],snapshots=snapshots,
        original_history_sha256=recovery.digest(RUN/'history.json'),original_budget_sha256=recovery.digest(CACHE/'compute-budget.json'),
        original_lock_sha256=None,clean_failed_exit_released_lock=True,retained_budget=budget,
        optimizer_and_rng_restored=True,no_optimization_repeated=True,acceptance_rules_unchanged=True,
        io_retry=dict(windows_errors=[5,32,33],maximum_attempts=20,delay_seconds=.05,permanent_failure_raises=True),sealed_test_accessed=False)
    recovery.repair.immutable_json(PLAN,plan);return plan


def validate(initial=False):
    parent.validate();plan=json.loads(PLAN.read_text())
    if recovery.digest(parent.PLAN)!=plan['parent_plan_sha256'] or recovery.digest(FAILED)!=plan['failed_status_sha256']:
        raise ValueError('Preserved parallel parent changed')
    failed=json.loads(FAILED.read_text())
    if recovery.digest(failed['steps'][-1]['log'])!=plan['failure_log_sha256']:raise ValueError('Failure log changed')
    for name,sha in plan['sources'].items():
        if recovery.digest(ROOT/name)!=sha:raise ValueError('Frozen IO-repair source changed')
    for name,sha in plan['snapshots'].items():
        if recovery.digest(ARCHIVE/name)!=sha:raise ValueError('IO-repair snapshot changed')
    if initial:
        absent_lock(CACHE/'compute-budget.lock',None,None)
        for path,key in [(RUN/'history.json','original_history_sha256'),(CACHE/'compute-budget.json','original_budget_sha256')]:
            if recovery.digest(path)!=plan[key]:raise ValueError('Failed state changed before retry')
    return plan


def train():
    plan=validate(initial=True);old_lock=recovery.preserve_stale_lock
    try:
        recovery.preserve_stale_lock=absent_lock
        with recovery.repair.bind(),parent.execution_override(workers=plan['workers'],validator=validate,plan=PLAN,archive=ARCHIVE),atomic.reliable_budget_io(recovery):
            return recovery.train()
    finally:recovery.preserve_stale_lock=old_lock


def orchestrate():
    parent.no_duplicate();validate(initial=True)
    with (ARCHIVE/'controller-started.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
    JOB.mkdir(parents=True,exist_ok=False);state=dict(status='RUNNING',pid=os.getpid(),steps=[],execution_plan_sha256=recovery.digest(PLAN))
    def save():state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic.atomic_json(JOB/'status.json',state)
    save()
    try:
        for stage in ('train','export','parity','calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report'):
            validate();script=Path(__file__).resolve() if stage=='train' else recovery.repair.HERE/'run.py'
            command=[sys.executable,'-u',str(script),stage]
            with (JOB/f'{stage}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row=dict(name=stage,pid=process.pid,command=command,status='RUNNING',started=time.time(),log=str(JOB/f'{stage}.log'))
                state['steps'].append(row);save();code=process.wait()
            row.update(exit_code=code,seconds=time.time()-row['started'],status='SUCCEEDED' if code==0 else 'FAILED');save()
            if code:raise RuntimeError(stage+' failed; repair evidence preserved')
        final=json.loads((CACHE/'report/finalization.json').read_text());state.update(status='COMPLETE',synthetic_qualified=final['synthetic_qualified'],production_ready=False)
    except BaseException as exc:state.update(status='FAILED',error=repr(exc));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','orchestrate','status']);stage=parser.parse_args().stage
    if stage=='freeze':result=freeze()
    elif stage=='train':result=train()
    elif stage=='orchestrate':orchestrate();return
    else:result=json.loads((JOB/'status.json').read_text())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
