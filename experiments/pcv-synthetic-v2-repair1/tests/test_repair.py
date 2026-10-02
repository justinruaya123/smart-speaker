import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v1'));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
spec=importlib.util.spec_from_file_location('repair_workflow',Path(__file__).resolve().parents[1]/'workflow.py')
repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)


def test_exact_ctc_capacity_reproduces_short_parler_failure():
    assert repair.ctc_capacity(4087,list(range(1,31)))==dict(frames=24,required=30,feasible=False)
    assert repair.ctc_capacity(400,[1,1])['feasible'] is False
    assert repair.ctc_capacity(720,[1,1])==dict(frames=3,required=3,feasible=True)


def original(split='train'):
    return dict(recording_id='parler/train-gb-f/one',verbatim_id='one',split=split,engine='parler',
        voice_group='train-gb-f',audio='old.wav',native_audio='old-native.wav',audio_sha256='old',
        pcm_sha256='old-pcm',seed=71,batch_seed=71,generation_attempt=1)


def test_only_named_training_record_can_be_replaced_and_all_other_rows_survive():
    from pcvcm.contract import identity
    old=original();dev=dict(old,recording_id='parler/dev/two',split='dev')
    plan=dict(records=[dict(recording_id=old['recording_id'],parent_record_sha256=identity(old))])
    new={**old,'audio':'new.wav','native_audio':'new-native.wav','audio_sha256':'new',
        'pcm_sha256':'new-pcm','repair_parent_record_sha256':identity(old),'source_version':'v2-repair1'}
    result=repair.merge_records([old,dev],[new],plan)
    assert result==[new,dev] and result[1] is dev
    for mutation in ({'split':'test'},{'voice_group':'other'},{'verbatim_id':'other'},{'seed':72}):
        with pytest.raises(ValueError):repair.merge_records([old,dev],[dict(new,**mutation)],plan)
    with pytest.raises(ValueError):repair.merge_records([old,dev],[new,new],plan)
    with pytest.raises(ValueError):repair.merge_records([old,dev],[],plan)


@pytest.mark.parametrize('split',['dev','calibration','test'])
def test_repair_cannot_replace_held_out_audio(split):
    from pcvcm.contract import identity
    old=original(split);plan=dict(records=[dict(recording_id=old['recording_id'],parent_record_sha256=identity(old))])
    new={**old,'repair_parent_record_sha256':identity(old),'source_version':'v2-repair1'}
    with pytest.raises(ValueError,match='training'):repair.merge_records([old],[new],plan)


def test_preflight_rejects_short_audio_before_writing_a_recording(tmp_path):
    import synthesis
    # A signal can pass waveform energy checks yet be impossible for its CTC target.
    wave=np.random.default_rng(4).normal(0,.05,4087).astype(np.float32)
    prepared=synthesis.prepare_audio(wave,16000)
    with pytest.raises(ValueError,match='CTC'):repair.require_feasible(prepared,list(range(1,31)))
    assert list(tmp_path.iterdir())==[]


def test_child_context_restores_parent_globals_even_after_failure():
    import fitting,synthbench,wake_benchmark,revision
    before=(fitting.CACHE,synthbench.load_split,wake_benchmark.audit_wake,revision.CACHE)
    with pytest.raises(RuntimeError,match='fixture'):
        with repair.bind():
            assert fitting.CACHE==repair.CACHE and synthbench.load_split is repair.load_split
            assert revision.CACHE==before[3]
            raise RuntimeError('fixture')
    assert (fitting.CACHE,synthbench.load_split,wake_benchmark.audit_wake,revision.CACHE)==before


def test_every_unrepaired_row_is_preserved_including_all_held_out_splits():
    from pcvcm.contract import identity
    old=original()
    parents=[old]+[dict(original(s),recording_id=f'parler/{s}/{i}')
        for s in ('train','dev','calibration','test') for i in range(3)]
    plan=dict(records=[dict(recording_id=old['recording_id'],parent_record_sha256=identity(old))])
    new={**old,'repair_parent_record_sha256':identity(old),'source_version':'v2-repair1'}
    result=repair.merge_records(parents,[new],plan)
    assert len(result)==len(parents) and all(a is b for a,b in zip(result[1:],parents[1:]))
    with pytest.raises(ValueError,match='identity'):
        repair.merge_records([dict(old,seed=72),*parents[1:]],[new],plan)


@pytest.mark.parametrize('kind',['command','wake'])
def test_identical_parent_test_cannot_be_read_after_child_reservation(tmp_path,monkeypatch,kind):
    sys.path.insert(0,str(repair.HERE))
    spec=importlib.util.spec_from_file_location('repair_cli',repair.HERE/'run.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(cli,'repair',repair)
    parent=tmp_path/'parent';parent.mkdir();here=tmp_path/'definitions';here.mkdir()
    monkeypatch.setattr(repair.rev,'CACHE',parent)
    monkeypatch.setattr(repair,'PARENT',here);monkeypatch.setattr(repair,'HERE',here)
    for name in ('protocol.json','generation-plan.jsonl','wake-generation-plan.jsonl','wake-verbatims.jsonl'):
        (here/name).write_text('{}')
    for e in repair.ENGINES:
        for prefix in ('','wake-'):(parent/f'{prefix}{e}-manifest.jsonl').write_text('{}')
    path=cli.reserve_parent_test(kind,tmp_path/'bundle',tmp_path/'output')
    assert json.loads(path.read_text())['state']=='RESERVED_PRESERVE_ON_FAILURE'
    if kind=='command':
        from pcvcm.contract import identity,file_hash
        # This is the unchanged parent evaluator's exact reservation identity.
        key=identity(dict(plan=file_hash(here/'generation-plan.jsonl'),
            manifests={e:file_hash(parent/f'{e}-manifest.jsonl') for e in repair.ENGINES}))
        assert path==parent/'test-access'/f'{key}.json'
    with pytest.raises(FileExistsError):cli.reserve_parent_test(kind,tmp_path/'bundle2',tmp_path/'output2')


def test_child_executes_shared_export_parity_calibration_report_with_artificial_audio(tmp_path,monkeypatch):
    path=repair.rev.V1/'tests/test_stage_integration.py'
    spec=importlib.util.spec_from_file_location('child_stage_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with repair.bind(),monkeypatch.context() as scoped:
        fixture.test_export_replay_parity_calibration_and_report_preserve_failed_acceptance(tmp_path,scoped)
