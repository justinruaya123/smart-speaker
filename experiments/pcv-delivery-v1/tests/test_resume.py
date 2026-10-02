import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts import pcv_resume as recovery


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_restore_reproduces_next_optimizer_step_and_random_draws(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(53);rng=np.random.default_rng(53)
    model=torch.nn.Linear(3,2).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    def step(m,o,r):
        x=torch.from_numpy(r.normal(size=(4,3)).astype(np.float32)).to(device)
        y=torch.randn(4,2,device=device);loss=(m(x)-y).square().mean()
        o.zero_grad(set_to_none=True);loss.backward();o.step()
        return float(loss.detach())
    step(model,optimizer,rng)
    saved=copy.deepcopy(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],numpy_rng=rng.bit_generator.state))
    expected_loss=step(model,optimizer,rng);expected=copy.deepcopy(model.state_dict())
    resumed=torch.nn.Linear(3,2).to(device);opt=torch.optim.AdamW(resumed.parameters(),lr=.001,weight_decay=.0001)
    resumed_rng=np.random.default_rng(999)
    recovery.restore_state(resumed,opt,resumed_rng,saved,device)
    assert step(resumed,opt,resumed_rng)==expected_loss
    assert all(torch.equal(value,resumed.state_dict()[key]) for key,value in expected.items())


def test_recovery_history_allows_only_missing_latest_evaluation():
    history=[dict(seed=53,epoch=n,checkpoint=f'{n}.pt',checkpoint_sha256=str(n)) for n in range(1,12)]
    result=recovery.pending_record(history,dict(seed=53,epoch=12),Path('12.pt'),'12',[4,8,12,16,20,24,28,32,36,40])
    assert result['loss'] is None and result['loss_unavailable_due_to_interruption'] is True
    assert result['epoch']==12 and len(history)==11
    with pytest.raises(ValueError):recovery.pending_record(history,dict(seed=53,epoch=14),Path('14.pt'),'14',[4,8,12])
    with pytest.raises(ValueError):recovery.pending_record(history,dict(seed=71,epoch=12),Path('12.pt'),'12',[4,8,12])


def test_stale_budget_lock_is_preserved_and_budget_is_not_reset(tmp_path):
    budget=tmp_path/'compute-budget.json';before={'seconds':8899.8,'epochs':[{'label':'53/12'}],'limits':{'epochs':200,'seconds':86400}}
    budget.write_text(json.dumps(before));lock=budget.with_suffix('.lock');lock.write_text('old run')
    target=tmp_path/'recovery';target.mkdir()
    recovery.preserve_stale_lock(lock,target/'original.lock',recovery.digest(lock))
    assert not lock.exists() and (target/'original.lock').read_text()=='old run'
    assert json.loads(budget.read_text())==before
    with pytest.raises(FileExistsError):
        lock.write_text('new active run');recovery.preserve_stale_lock(lock,target/'original.lock',recovery.digest(lock))
    assert lock.read_text()=='new active run'


def test_atomic_checkpoint_never_overwrites_an_existing_epoch(tmp_path):
    path=tmp_path/'epoch.pt';recovery.save_checkpoint(path,{'x':torch.tensor([1.])})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):recovery.save_checkpoint(path,{'x':torch.tensor([2.])})
    assert path.read_bytes()==before and torch.load(path,weights_only=True)['x'].item()==1.


@pytest.fixture
def deterministic_ctc_fixture(monkeypatch):
    # The installed CUDA CTC backward explicitly has no deterministic implementation.
    # Use CPU CTC only in this artificial fixture so exact long-run equality tests
    # recovery mechanics rather than GPU atomic-reduction ordering. The model and
    # optimizer still run on CUDA; the actual study's CUDA CTC recipe is unchanged.
    original=torch.nn.functional.ctc_loss
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    def cpu_ctc(logits,targets,input_lengths,target_lengths,**kwargs):
        return original(logits.cpu(),targets.cpu(),input_lengths.cpu(),target_lengths.cpu(),**kwargs)
    monkeypatch.setattr(torch.nn.functional,'ctc_loss',cpu_ctc)
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Recovery fixture requires the declared CUDA device')
def test_complete_recovery_matches_uninterrupted_frozen_training_on_artificial_features(tmp_path,monkeypatch,deterministic_ctc_fixture):
    import importlib.util
    import shutil
    spec=importlib.util.spec_from_file_location('recovery_control_fixture',
        ROOT/'experiments/pcv-synthetic-v1/tests/test_stage_integration.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with recovery.repair.bind(),monkeypatch.context() as scoped:
        fixture.test_training_continuation_writes_baseline_selection_and_improvement_on_artificial_features(tmp_path,scoped)
        control=tmp_path/'training';run=tmp_path/'recovered-training';run.mkdir()
        for name in ('protocol.json','normalization.json','catalog.json','baseline-development.json'):
            shutil.copyfile(control/name,run/name)
        history=json.loads((control/'history.json').read_text())[:11]
        (run/'history.json').write_text(json.dumps(history))
        budget=json.loads((tmp_path/'compute-budget.json').read_text());budget['epochs']=budget['epochs'][:12]
        budget['seconds']=10.;(tmp_path/'compute-budget.json').write_text(json.dumps(budget))
        lock=tmp_path/'compute-budget.lock';lock.write_text(str(control))
        archive=tmp_path/'recovery';archive.mkdir()
        checkpoint=control/'seed-53/epoch-012.pt'
        plan=dict(checkpoint=str(checkpoint),checkpoint_sha256=recovery.digest(checkpoint),seed=53,epoch=12,
            original_lock_sha256=recovery.digest(lock))
        plan_path=archive/'plan.json';plan_path.write_text(json.dumps(plan))
        for key,value in [('CACHE',tmp_path),('RUN',run),('RECOVERY',archive),('PLAN',plan_path)]:scoped.setattr(recovery,key,value)
        scoped.setattr(recovery,'validate_recovery',lambda **kwargs:plan)
        result=recovery.train()
        assert len(result['candidates'])==20
        assert result['winner']['seed']==json.loads((control/'selection.json').read_text())['winner']['seed']
        assert result['winner']['epoch']==json.loads((control/'selection.json').read_text())['winner']['epoch']
        for seed in (53,71):
            expected=torch.load(control/f'seed-{seed}/epoch-040.pt',map_location='cpu',weights_only=True)
            actual=torch.load(run/f'seed-{seed}/epoch-040.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(value,actual['weights'][key]) for key,value in expected['weights'].items())
            assert actual['numpy_rng']==expected['numpy_rng']
        resumed_history=json.loads((run/'history.json').read_text())
        assert len(resumed_history)==80 and resumed_history[11]['loss'] is None
        assert json.loads((tmp_path/'compute-budget.json').read_text())['epochs'][:12]==budget['epochs']
        assert len(json.loads((tmp_path/'compute-budget.json').read_text())['epochs'])==80
        assert (archive/'compute-budget.lock.original').read_text()==str(control)
