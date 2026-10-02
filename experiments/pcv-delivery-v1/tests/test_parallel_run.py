import importlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def module():
    try:return importlib.import_module('scripts.pcv_parallel_run')
    except ModuleNotFoundError:pytest.fail('Parallel handoff controller is not implemented')


def test_handoff_accepts_only_pending_development_after_complete_checkpoint(tmp_path):
    api=module();run=tmp_path/'training';seed=run/'seed-53';seed.mkdir(parents=True)
    import json
    (run/'protocol.json').write_text(json.dumps({'selection_epochs':[4,8,12,16,20,24,28,32,36,40]}))
    (run/'history.json').write_text(json.dumps([dict(seed=53,epoch=e) for e in range(1,20)]))
    (seed/'epoch-020.pt').write_bytes(b'intact saved checkpoint')
    assert api.handoff_position(run)==(53,20)
    (seed/'dev-020.json.pending').write_text('partial')
    with pytest.raises(ValueError):api.handoff_position(run)
    (seed/'dev-020.json.pending').unlink()
    (seed/'epoch-021.pt').write_bytes(b'optimization moved on')
    with pytest.raises(ValueError):api.handoff_position(run)


def test_parallel_launch_requires_complete_exact_development_parity():
    api=module()
    good=dict(complete=True,recordings=5636,workers=8,errors=[],score_comparisons=60,
              score_parity={'4':True,'8':True},measured_speedup=3.,sealed_test_accessed=False)
    api.require_parity(good)
    for key,value in [('complete',False),('recordings',5635),('errors',['a']),('score_comparisons',59),
                      ('score_parity',{'8':False}),('sealed_test_accessed',True)]:
        with pytest.raises(ValueError):api.require_parity({**good,key:value})


def test_process_guard_rejects_recycled_pid_or_different_command():
    api=module();expected=dict(pid=1,created=2.,command=['python','-u','scripts/pcv_resume.py','train'])
    api.require_identity(expected,dict(expected))
    for key,value in [('pid',8),('created',3.),('command',['python','unrelated.py'])]:
        with pytest.raises(ValueError):api.require_identity(expected,{**expected,key:value})


def test_saved_job_authentication_rejects_other_checkout_reused_pid_and_wrong_tree():
    api=module()
    assert hasattr(api,'authenticate_job'),'Saved job process authentication is missing'
    state=dict(pid=1,steps=[dict(pid=2,started=100.)])
    controller=dict(pid=1,created=99.,command=['python','scripts/pcv_resume.py','orchestrate'],cwd=str(ROOT),ppid=99)
    launcher=dict(pid=2,created=99.9,command=['python',str(ROOT/'scripts/pcv_resume.py'),'train'],cwd=str(ROOT),ppid=1)
    worker=dict(pid=3,created=100.,command=launcher['command'],cwd=str(ROOT),ppid=2)
    api.authenticate_job(state,controller,launcher,worker)
    for key,value in [('cwd',str(ROOT.parent/'unrelated')),('created',200.),('pid',7)]:
        with pytest.raises(ValueError):api.authenticate_job(state,{**controller,key:value},launcher,worker)
    with pytest.raises(ValueError):api.authenticate_job(state,controller,{**launcher,'pid':4},worker)
    with pytest.raises(ValueError):api.authenticate_job(state,controller,launcher,{**worker,'ppid':8})


def test_execution_override_restores_frozen_modules_on_error(monkeypatch,tmp_path):
    api=module();from pcvcm import training
    original=training.development;validate=api.recovery.validate_recovery;plan=api.recovery.PLAN
    with pytest.raises(RuntimeError,match='fixture abort'):
        with api.execution_override(workers=4,validator=lambda **kwargs:{},plan=tmp_path/'plan.json',archive=tmp_path):
            assert training.development is not original
            assert api.recovery.PLAN==tmp_path/'plan.json'
            raise RuntimeError('fixture abort')
    assert training.development is original and api.recovery.validate_recovery is validate and api.recovery.PLAN==plan


def test_parallel_continuation_matches_controlled_uninterrupted_training(monkeypatch,tmp_path):
    api=module()
    import importlib.util
    import torch
    if not torch.cuda.is_available():pytest.skip('Declared CUDA device unavailable')
    spec=importlib.util.spec_from_file_location('recovery_control_checks',Path(__file__).with_name('test_resume.py'))
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    # Bind/load the frozen reference trainer before installing the process-local
    # adapter, so the control still uses its original sequential development call.
    with api.recovery.repair.bind():pass
    deterministic=checks.deterministic_ctc_fixture.__wrapped__(monkeypatch);next(deterministic)
    try:
        with api.execution_override(workers=2,validator=lambda **kwargs:{},plan=tmp_path/'unused.json',archive=tmp_path):
            checks.test_complete_recovery_matches_uninterrupted_frozen_training_on_artificial_features(tmp_path,monkeypatch,None)
    finally:deterministic.close()
