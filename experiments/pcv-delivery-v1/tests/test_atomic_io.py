import importlib
from pathlib import Path
import pytest
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def module():
    try:return importlib.import_module('scripts.pcv_atomic_io')
    except ModuleNotFoundError:pytest.fail('Bounded Windows replace retry is missing')


def test_transient_windows_lock_retries_same_payload(tmp_path,monkeypatch):
    api=module();source=tmp_path/'pending';target=tmp_path/'budget.json';source.write_text('new');target.write_text('old')
    original=api.os.replace;calls=[]
    def busy(a,b):
        calls.append((str(a),str(b)))
        if len(calls)<3:
            error=PermissionError('sharing conflict');error.winerror=5;raise error
        return original(a,b)
    monkeypatch.setattr(api.os,'replace',busy);monkeypatch.setattr(api.time,'sleep',lambda _:None)
    api.replace_with_retry(source,target)
    assert len(calls)==3 and target.read_text()=='new' and not source.exists()


def test_permanent_error_remains_bounded_and_preserves_pending(tmp_path,monkeypatch):
    api=module();source=tmp_path/'pending';target=tmp_path/'budget.json';source.write_text('new');target.write_text('old');calls=[]
    def busy(*args):
        calls.append(1);error=PermissionError('locked');error.winerror=32;raise error
    monkeypatch.setattr(api.os,'replace',busy);monkeypatch.setattr(api.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):api.replace_with_retry(source,target,attempts=3)
    assert len(calls)==3 and source.read_text()=='new' and target.read_text()=='old'


def test_unrelated_filesystem_error_is_not_retried(tmp_path,monkeypatch):
    api=module();calls=[]
    def failed(*args):calls.append(1);raise OSError('disk failure')
    monkeypatch.setattr(api.os,'replace',failed)
    with pytest.raises(OSError):api.replace_with_retry(tmp_path/'a',tmp_path/'b')
    assert len(calls)==1


def test_actual_windows_reader_lock_is_recovered(tmp_path):
    api=module()
    if sys.platform!='win32':pytest.skip('Windows share-delete semantics')
    import ctypes,threading,time
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE;kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    source=tmp_path/'pending';target=tmp_path/'budget.json';source.write_text('new');target.write_text('old')
    handle=kernel.CreateFileW(str(target),0x80000000,1,None,3,0,None)
    assert handle!=wintypes.HANDLE(-1).value
    worker=threading.Thread(target=lambda:(time.sleep(.15),kernel.CloseHandle(handle)));worker.start()
    try:api.replace_with_retry(source,target)
    finally:worker.join()
    assert target.read_text()=='new'


def test_reliable_budget_preserves_limits_and_prior_charges(tmp_path):
    api=module()
    from scripts import pcv_resume as recovery
    from pcvcm import compute
    import json
    budget=tmp_path/'budget.json';budget.write_text(json.dumps(dict(limits={'epochs':200,'seconds':86400},seconds=17006.78,epochs=[{'label':'53/24'}])))
    original=compute.ComputeBudget
    with api.reliable_budget_io(recovery):
        with compute.ComputeBudget(budget,'fixture') as b:
            b.check();assert b.state['seconds']>=17006.78
        exhausted=tmp_path/'exhausted.json'
        exhausted.write_text(json.dumps(dict(limits={'epochs':200,'seconds':86400},seconds=86400,epochs=[])))
        with pytest.raises(RuntimeError,match='exhausted'):
            with compute.ComputeBudget(exhausted,'fixture'):pass
    observed=json.loads(budget.read_text())
    assert observed['epochs']==[{'label':'53/24'}] and observed['limits']=={'epochs':200,'seconds':86400}
    assert observed['seconds']>=17006.78 and compute.ComputeBudget is original


def test_io_repaired_parallel_continuation_preserves_exact_fixture(monkeypatch,tmp_path):
    api=module()
    from scripts import pcv_parallel_repair as repair
    assert repair.absent_lock(tmp_path/'missing.lock',None,None) is None
    (tmp_path/'active.lock').write_text('active')
    with pytest.raises(ValueError):repair.absent_lock(tmp_path/'active.lock',None,None)
    import importlib.util
    spec=importlib.util.spec_from_file_location('parallel_fixture_for_io',Path(__file__).with_name('test_parallel_run.py'))
    checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
    (tmp_path/'continuation').mkdir()
    with api.reliable_budget_io(repair.recovery):
        checks.test_parallel_continuation_matches_controlled_uninterrupted_training(monkeypatch,tmp_path/'continuation')
