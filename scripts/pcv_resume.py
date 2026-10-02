"""Checkpoint recovery after the user-reported PC shutdown; frozen studies stay intact.

This host-only recovery retains the existing optimizer, random state, inputs,
schedule, selection rule and compute budget. It never restarts synthesis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2-repair1'))
import workflow as repair

CACHE = repair.CACHE
RUN = CACHE/'training'
RECOVERY = CACHE/'recoveries/shutdown-20261001'
PLAN = ROOT/'experiments/pcv-delivery-v1/recovery-shutdown-20261001.json'
JOB = CACHE/'jobs/resumed-run-20261001'
OLD_JOB = CACHE/'jobs/repair-run-20261001/status.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.pending')
    if temporary.exists(): raise FileExistsError('Preserve orphan receipt: '+str(temporary))
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(path, state):
    import torch
    path = Path(path); temporary = path.with_suffix('.pt.pending')
    if path.exists() or temporary.exists(): raise FileExistsError('Checkpoint or interrupted save already exists')
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('xb') as stream:
        torch.save(state,stream); stream.flush(); os.fsync(stream.fileno())
    temporary.rename(path)


def restore_state(model, optimizer, rng, saved, device):
    import torch
    model.load_state_dict(saved['weights'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    rng.bit_generator.state = saved['numpy_rng']
    torch.set_rng_state(saved['torch_rng'].cpu())
    if str(device).startswith('cuda'):
        if len(saved['cuda_rng']) != torch.cuda.device_count(): raise ValueError('CUDA random-state device count changed')
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])


def pending_record(history, saved, checkpoint, checkpoint_sha256, selection_epochs):
    seed, epoch = saved['seed'], saved['epoch']
    observed = [(r['seed'],r['epoch']) for r in history]
    if (len(set(observed)) != len(observed) or epoch not in selection_epochs
            or observed != [(seed,n) for n in range(1,epoch)]):
        raise ValueError('Recovery requires one pending evaluation after contiguous saved epochs')
    return dict(seed=seed,epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha256,
        loss=None,loss_unavailable_due_to_interruption=True,optimization_complete_before_shutdown=True)


def preserve_stale_lock(lock, archive, expected_sha256):
    lock,archive = Path(lock).resolve(),Path(archive).resolve()
    if not archive.is_relative_to(lock.parent): raise ValueError('Recovery archive escapes the study cache')
    if archive.exists(): raise FileExistsError('Original lock already archived')
    if digest(lock) != expected_sha256: raise ValueError('Lock changed since shutdown audit')
    lock.rename(archive)


def require_no_other_worker():
    import psutil
    ignored = {os.getpid(),*[p.pid for p in psutil.Process().parents()]}
    active=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if process.pid in ignored or not (process.info['name'] or '').lower().startswith('python'): continue
            command=process.info['cmdline'] or []
            if any((str(repair.HERE).lower() in value.lower() and value.lower().endswith('.py'))
                    or Path(value).name=='pcv_resume.py' for value in command): active.append(process.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied): continue
    if active: raise RuntimeError(f'Existing study worker(s): {active}; no duplicate recovery')


def freeze_recovery():
    import psutil
    import torch
    from vcm.model import CausalVCM
    require_no_other_worker(); repair.validate()
    if PLAN.exists() or RECOVERY.exists(): raise FileExistsError('Recovery already frozen; preserve its receipt')
    old=json.loads(OLD_JOB.read_text()); protocol=json.loads((RUN/'protocol.json').read_text())
    history=json.loads((RUN/'history.json').read_text()); checkpoint=RUN/'seed-53/epoch-012.pt'
    if old['steps'][-1]['name']!='train' or old['steps'][-1]['started']>=psutil.boot_time():
        raise ValueError('Expected a pre-boot interrupted training stage')
    for path in (RUN/'selection.json',CACHE/'bundle-uncalibrated',CACHE/'test-access',CACHE/'wake-test-access',
            repair.rev.CACHE/'test-access',repair.rev.CACHE/'wake-test-access'):
        if path.exists(): raise ValueError('Recovery cannot repeat selection/export or sealed evaluation: '+str(path))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['protocol_sha256']!=repair.identity(protocol) or (saved['seed'],saved['epoch'])!=(53,12):
        raise ValueError('Checkpoint recipe or recovery position changed')
    probe=CausalVCM();probe.load_state_dict(saved['weights'],strict=True)
    if any(not torch.isfinite(value).all() for value in saved['weights'].values()): raise ValueError('Nonfinite checkpoint')
    if len(saved['optimizer']['state'])!=len(list(probe.parameters())): raise ValueError('Optimizer checkpoint incomplete')
    pending_record(history,saved,checkpoint,digest(checkpoint),protocol['selection_epochs'])
    for row in history:
        if digest(row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Recorded checkpoint changed')
        if 'predictions' in row and digest(row['predictions'])!=row['predictions_sha256']:
            raise ValueError('Completed development result changed')
    budget=json.loads((CACHE/'compute-budget.json').read_text())
    if [x['label'] for x in budget['epochs']] != [f'53/{n}' for n in range(1,13)]: raise ValueError('Budget history mismatch')
    if budget['limits']!={'epochs':200,'seconds':86400}: raise ValueError('Budget limits changed')
    RECOVERY.mkdir(parents=True)
    snapshots={}
    for name,path in [('history-before.json',RUN/'history.json'),('budget-before.json',CACHE/'compute-budget.json'),
            ('job-before.json',OLD_JOB)]:
        destination=RECOVERY/name;shutil.copyfile(path,destination);snapshots[name]=digest(destination)
    immutable={str(p.relative_to(ROOT)):digest(p) for p in RUN.rglob('*') if p.is_file() and p.name!='history.json'}
    plan=dict(version='pcv-shutdown-recovery-20261001',time_utc=datetime.now(timezone.utc).isoformat(),
        reason='User reported PC shutdown and explicitly requested continuation',boot_time=psutil.boot_time(),
        source_sha256=digest(__file__),frozen_study_protocol_sha256=digest(repair.HERE/'protocol.json'),
        training_protocol_sha256=digest(RUN/'protocol.json'),prepared_freeze_sha256=digest(CACHE/'prepared/freeze.json'),
        original_training_files=immutable,snapshots=snapshots,original_history_sha256=digest(RUN/'history.json'),
        original_budget_sha256=digest(CACHE/'compute-budget.json'),original_lock_sha256=digest(CACHE/'compute-budget.lock'),
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest(checkpoint),seed=53,epoch=12,
        pending_development=True,retained_budget=budget,optimizer_and_rng_restored=True,
        missing_training_loss='Epoch 12 training loss was not saved before interrupted development; report null, never invent it',
        schedule_unchanged=True,selection_unchanged=True,test_accessed=False)
    repair.immutable_json(PLAN,plan)
    return plan


def validate_recovery(initial=False):
    repair.validate(); plan=json.loads(PLAN.read_text())
    if digest(__file__)!=plan['source_sha256']: raise ValueError('Frozen recovery implementation changed')
    if digest(repair.HERE/'protocol.json')!=plan['frozen_study_protocol_sha256']:
        raise ValueError('Frozen study protocol changed')
    if digest(CACHE/'prepared/freeze.json')!=plan['prepared_freeze_sha256']: raise ValueError('Prepared freeze changed')
    for name,sha in plan['original_training_files'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Preserved training evidence changed: '+name)
    for name,sha in plan['snapshots'].items():
        if digest(RECOVERY/name)!=sha: raise ValueError('Shutdown snapshot changed')
    if initial:
        for path,key in [(RUN/'history.json','original_history_sha256'),(CACHE/'compute-budget.json','original_budget_sha256'),
                (CACHE/'compute-budget.lock','original_lock_sha256')]:
            if digest(path)!=plan[key]: raise ValueError('Interrupted state changed before recovery')
    return plan


def train():
    import numpy as np
    import torch
    from einops import rearrange
    from vcm.model import CausalVCM
    from pcvcm.data import batch_arrays,verified_features
    from pcvcm.compute import ComputeBudget
    from pcvcm.training import development,sampling_weights,select_checkpoint
    plan=validate_recovery(initial=True)
    if not torch.cuda.is_available(): raise RuntimeError('Declared training CUDA device unavailable')
    prepared=CACHE/'prepared';freeze=json.loads((prepared/'freeze.json').read_text())
    for name,sha in freeze['files'].items():
        if digest(prepared/name)!=sha: raise ValueError('Prepared metadata changed')
    protocol=json.loads((RUN/'protocol.json').read_text());history=json.loads((RUN/'history.json').read_text())
    train_rows=repair.read_rows(prepared/'train.jsonl');dev=repair.read_rows(prepared/'dev.jsonl')
    import fitting
    fitting.require_training_split(train_rows+dev)
    norm=json.loads((RUN/'normalization.json').read_text())
    mean=rearrange(np.array(norm['mean'],dtype=np.float32),'c -> c 1')
    std=rearrange(np.array(norm['std'],dtype=np.float32),'c -> c 1')
    arrays={r['recording_id']:(verified_features(r,prepared)-mean)/std for r in train_rows+dev}
    catalog=json.loads((RUN/'catalog.json').read_text());baseline=json.loads((RUN/'baseline-development.json').read_text())
    original=ROOT/'.cache/semifinal-v1/longer/seed-71/selected.pt'
    if digest(original)!=protocol['baseline_sha256']: raise ValueError('Original initialization changed')
    initial=torch.load(original,map_location='cpu',weights_only=True)['weights']
    saved=torch.load(ROOT/plan['checkpoint'],map_location='cpu',weights_only=True)
    pending=pending_record(history,saved,ROOT/plan['checkpoint'],plan['checkpoint_sha256'],protocol['selection_epochs'])
    candidates=[dict(row) for row in history if 'predictions' in row]
    weights=sampling_weights(train_rows,.8,1.);torch.set_num_threads(4)
    preserve_stale_lock(CACHE/'compute-budget.lock',RECOVERY/'compute-budget.lock.original',plan['original_lock_sha256'])
    recovery_binding=dict(plan_sha256=digest(PLAN),source_sha256=digest(__file__),checkpoint_sha256=plan['checkpoint_sha256'])
    def evaluate_record(model,record,budget):
        scores,predictions=development(model,dev,arrays,catalog['entries'],'cuda',budget.check)
        path=RUN/f'seed-{record["seed"]}'/f'dev-{record["epoch"]:03}.json'
        if path.exists(): raise FileExistsError('Do not overwrite completed development evidence')
        atomic_json(path,predictions)
        record.update(scores,predictions=str(path),predictions_sha256=digest(path));candidates.append(dict(record))
    with ComputeBudget(CACHE/'compute-budget.json',RUN) as budget:
        for seed in protocol['training_seeds']:
            if seed==plan['seed']:
                model=CausalVCM().to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                rng=np.random.default_rng(seed);restore_state(model,optimizer,rng,saved,'cuda')
                print('RESUMED_CHECKPOINT',seed,plan['epoch'],'pending development; no optimization repeated',flush=True)
                evaluate_record(model,pending,budget);history.append(pending)
                atomic_json(RUN/'history.json',history);print(json.dumps(pending),flush=True)
                first=plan['epoch']+1
            else:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                model=CausalVCM().to('cuda');model.load_state_dict(initial)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                (RUN/f'seed-{seed}').mkdir();first=1
            for epoch in range(first,41):
                budget.start_epoch(f'{seed}/{epoch}');model.train();losses=[]
                order=rng.choice(len(train_rows),size=len(train_rows),p=weights,replace=True)
                for offset in range(0,len(order),64):
                    budget.check();batch=[train_rows[int(i)] for i in order[offset:offset+64]]
                    x,lengths=batch_arrays([arrays[row['recording_id']] for row in batch])
                    logits,_=model(torch.from_numpy(x).to('cuda'))
                    targets=torch.tensor([phone for row in batch for phone in row['phones']],device='cuda')
                    loss=torch.nn.functional.ctc_loss(rearrange(logits.log_softmax(1),'b c t -> t b c'),targets,
                        torch.tensor(lengths),torch.tensor([len(row['phones']) for row in batch]),zero_infinity=False)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite CTC loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                checkpoint=RUN/f'seed-{seed}'/f'epoch-{epoch:03}.pt'
                save_checkpoint(checkpoint,dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),seed=seed,epoch=epoch,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=rng.bit_generator.state,
                    protocol_sha256=repair.identity(protocol),recovery=recovery_binding))
                record=dict(seed=seed,epoch=epoch,loss=float(np.mean(losses)),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
                if epoch in protocol['selection_epochs']:evaluate_record(model,record,budget)
                history.append(record);atomic_json(RUN/'history.json',history);print(json.dumps(record),flush=True)
    if len(history)!=80 or len(candidates)!=20: raise ValueError('Recovery did not complete the frozen selection schedule')
    selection=dict(winner=select_checkpoint(candidates),candidates=candidates,test_accessed=False,calibration_accessed=False,
        synthetic_only=True,protocol_sha256=digest(RUN/'protocol.json'),recovery=recovery_binding)
    if (RUN/'selection.json').exists(): raise FileExistsError('Selected model already exists')
    atomic_json(RUN/'selection.json',selection)
    selected=json.loads(Path(selection['winner']['predictions']).read_text());before={row['recording_id']:row for row in baseline['predictions']}
    atomic_json(RUN/'development-improvement.json',dict(baseline=baseline['metrics'],
        selected={key:selection['winner'][key] for key in baseline['metrics']},
        gained=[row['recording_id'] for row in selected if row['raw_exact'] and not before[row['recording_id']]['raw_exact']],
        lost=[row['recording_id'] for row in selected if not row['raw_exact'] and before[row['recording_id']]['raw_exact']],
        baseline_predictions_sha256=digest(RUN/'baseline-development.json'),selected_predictions_sha256=selection['winner']['predictions_sha256'],
        recovery=recovery_binding))
    return selection


def orchestrate():
    require_no_other_worker();validate_recovery(initial=True)
    # Exclusive per-recovery controller receipt, including across another reboot.
    with (RECOVERY/'controller-started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),job=str(JOB)),stream);stream.flush();os.fsync(stream.fileno())
    JOB.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',pid=os.getpid(),steps=[],recovery_plan_sha256=digest(PLAN),
        frozen_study_protocol_sha256=digest(repair.HERE/'protocol.json'),interrupted_job=str(OLD_JOB))
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(JOB/'status.json',state)
    save()
    try:
        for stage in ('train','export','parity','calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report'):
            validate_recovery()
            command=[sys.executable,'-u',str(Path(__file__).resolve() if stage=='train' else repair.HERE/'run.py'),stage]
            with (JOB/f'{stage}.log').open('x',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                row=dict(name=stage,pid=process.pid,command=command,status='RUNNING',started=time.time(),log=str(JOB/f'{stage}.log'))
                state['steps'].append(row);save();code=process.wait()
            row.update(exit_code=code,seconds=time.time()-row['started'],status='SUCCEEDED' if code==0 else 'FAILED');save()
            if code:raise RuntimeError(stage+' failed; recovery evidence preserved')
        final=json.loads((CACHE/'report/finalization.json').read_text())
        state.update(status='COMPLETE',synthetic_qualified=final['synthetic_qualified'],production_ready=False)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));raise
    finally:save()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','train','orchestrate','status'])
    stage=parser.parse_args().stage
    if stage=='freeze':result=freeze_recovery()
    elif stage=='train':
        with repair.bind():result=train()
    elif stage=='orchestrate':orchestrate();return
    else:result=json.loads((JOB/'status.json').read_text())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
