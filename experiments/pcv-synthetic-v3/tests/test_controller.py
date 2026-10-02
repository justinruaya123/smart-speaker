import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest


def module():
    spec=importlib.util.spec_from_file_location('v3_controller_fixture',Path(__file__).resolve().parents[1]/'orchestrate.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('status',['FAILED','RUNNING','SUCCEEDED'])
def test_generation_state_never_restarts_generator(tmp_path,monkeypatch,status):
    m=module();job=tmp_path/'gen';job.mkdir()
    (job/'status.json').write_text(json.dumps(dict(status=status,protocol_sha256='same',pid=88)))
    monkeypatch.setattr(m.wf,'require_generation_complete',lambda:({},job))
    class Process:
        def __init__(self,pid):assert pid==88
        def cmdline(self):return ['python','experiments/pcv-synthetic-v3/generation.py','verified-parler-run-20261001']
    monkeypatch.setitem(sys.modules,'psutil',types.SimpleNamespace(Process=Process,NoSuchProcess=LookupError))
    plan=dict(generation_job=str(job),generation_protocol_sha256='same')
    if status=='FAILED':
        with pytest.raises(RuntimeError,match='Generation failed'):m.generation_state(plan)
    else:assert m.generation_state(plan)==('ready' if status=='SUCCEEDED' else 'waiting')


def test_waiting_generator_pid_reuse_rejected(tmp_path,monkeypatch):
    m=module();(tmp_path/'status.json').write_text(json.dumps(dict(status='RUNNING',protocol_sha256='same',pid=88)))
    monkeypatch.setitem(sys.modules,'psutil',types.SimpleNamespace(Process=lambda _:types.SimpleNamespace(cmdline=lambda:['unrelated']),NoSuchProcess=LookupError))
    with pytest.raises(RuntimeError,match='identity'):m.generation_state(dict(generation_job=str(tmp_path),generation_protocol_sha256='same'))


@pytest.mark.parametrize('failing',[None,'export'])
def test_controller_stops_on_failure_and_preserves_exclusive_owner(tmp_path,monkeypatch,failing):
    m=module();cache=tmp_path/'cache';cache.mkdir();planfile=tmp_path/'plan.json';planfile.write_text('{}')
    m.PLAN=planfile;m.wf.CACHE=cache
    plan=dict(stages=['freeze','audit','train','export','parity','evaluate','report'])
    monkeypatch.setattr(m,'validate_execution',lambda:plan)
    monkeypatch.setattr(m,'generation_state',lambda _:'ready')
    final=cache/'report';final.mkdir();(final/'finalization.json').write_text(json.dumps({'synthetic_qualified':False}))
    launched=[]
    class Process:
        def __init__(self,command,**kwargs):self.name=command[-1];self.pid=7;launched.append(self.name)
        def wait(self):return 1 if self.name==failing else 0
    monkeypatch.setattr(m.subprocess,'Popen',Process)
    job=cache/'jobs'/'fixture'
    if failing:
        with pytest.raises(RuntimeError,match='export failed'):m.run(job)
        assert launched==['freeze','audit','train','export']
    else:m.run(job);assert launched==plan['stages']
    result=json.loads((job/'status.json').read_text())
    assert result['status']==('FAILED' if failing else 'COMPLETE')
    if not failing:assert result['synthetic_qualified'] is False and result['production_ready'] is False
    with pytest.raises(FileExistsError):m.run(cache/'jobs'/'duplicate')
