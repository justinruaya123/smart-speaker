import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[3]


def module():
    spec=importlib.util.spec_from_file_location('parler_corpus_v3',ROOT/'experiments/pcv-synthetic-v3/generation.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def rows():
    batch=['parler/train-us-m/a','parler/train-us-m/b']
    return [dict(recording_id=k,verbatim_id=k[-1],engine='parler',voice_group='train-us-m',split='train',
                 accent_requested='American English',synthetic=True,seed=10+i,batch_ids=batch,batch_size=4,
                 generation_attempt=3,batch_seed=12) for i,k in enumerate(batch)]


def test_original_batch_membership_and_first_seed_are_preserved():
    m=module();rr=rows();batches=m.original_batches(list(reversed(rr)))
    assert [r['recording_id'] for r in batches[0]]==rr[0]['batch_ids']
    assert m.retry_seeds(batches[0])==[10,11,12]


@pytest.mark.parametrize('change',['duplicate','missing','reordered','voice','engine'])
def test_rejects_inconsistent_parent_batches(change):
    m=module();rr=rows()
    if change=='duplicate':rr.append(dict(rr[0]))
    elif change=='missing':rr.pop()
    elif change=='reordered':rr[1]['batch_ids']=list(reversed(rr[1]['batch_ids']))
    elif change=='voice':rr[1]['voice_group']='train-gb-f'
    else:rr[1]['engine']='qwen'
    with pytest.raises(ValueError):m.original_batches(rr)


def test_capacity_accounts_for_ctc_repeated_phones():
    m=module()
    assert m.ctc_capacity(400+160*3,[1,1,2])=={'frames':4,'required':4,'feasible':True}
    assert not m.ctc_capacity(400+160*2,[1,1,2])['feasible']


def test_replacement_does_not_inherit_invalid_generation_claims(tmp_path):
    m=module();old=rows()[0]
    old.update(source_record_sha256='old',voice_anchor_sha256='misleading',text_verified_by_listener=True)
    row=m.planned_row(old,tmp_path)
    assert row['parent_record_sha256']==m.identity(old)
    assert 'source_record_sha256' not in row and 'text_verified_by_listener' not in row
    assert row['voice_anchor_sha256'] is None and row['source_version']=='v3'
    assert Path(row['audio']).is_relative_to(tmp_path)
    assert all(row[k]==old[k] for k in m.ASSIGNMENT)


def test_immutable_receipt_rejects_changes(tmp_path):
    m=module();p=tmp_path/'receipt.json'
    m.immutable_json(p,{'x':1});m.immutable_json(p,{'x':1})
    with pytest.raises(FileExistsError):m.immutable_json(p,{'x':2})
    assert json.loads(p.read_text())=={'x':1}


def test_parent_manifest_must_match_prior_audit_and_wake_lineage(tmp_path):
    m=module();m.ROOT=tmp_path;m.PARENT=tmp_path/'parent';m.PARENT.mkdir()
    experiment=tmp_path/'experiments/pcv-synthetic-v2-repair1';experiment.mkdir(parents=True)
    audit={};wake={}
    for engine in ('qwen','parler','dia'):
        for prefix in ('','wake-'):
            name=f'{prefix}{engine}-manifest.jsonl';p=m.PARENT/name;p.write_text(engine+prefix)
            if prefix:wake[name]=m.digest(p)
            else:audit[engine]=m.digest(p)
    (m.PARENT/'assembly-audit.json').write_text(json.dumps({'manifests':audit}))
    (experiment/'repair-plan.json').write_text(json.dumps({'parent_manifests':wake}))
    m.validate_parent_manifests()
    p=m.PARENT/'parler-manifest.jsonl';original=p.read_text();p.write_text('altered')
    with pytest.raises(ValueError,match='command'):m.validate_parent_manifests()
    p.write_text(original);(m.PARENT/'wake-parler-manifest.jsonl').write_text('altered')
    with pytest.raises(ValueError,match='wake'):m.validate_parent_manifests()


def test_full_corpus_reuses_non_parler_rows_exactly():
    m=module();parent=rows()+[dict(recording_id='qwen/x',engine='qwen',audio='unchanged')]
    new=[m.planned_row(r,Path('C:/fixture')) for r in parent if r['engine']=='parler']
    output=m.merge_manifest(parent,new)
    assert output[-1]==parent[-1] and output[:2]==new
    with pytest.raises(ValueError):m.merge_manifest(parent,new[:1])
    new[0]['split']='test'
    with pytest.raises(ValueError):m.merge_manifest(parent,new)


@pytest.mark.parametrize('mode',['success','too_short','weight_drift','final_weight_drift','final_validation','protocol_changed','mixed_retry'])
def test_generation_preflight_and_terminal_receipts(tmp_path,monkeypatch,mode):
    """Exercise the real loop with PCM output and fixture-only generator weights."""
    from contextlib import contextmanager
    import io
    import sys
    import types
    import numpy as np
    import soundfile as sf
    import torch
    m=module();cache=tmp_path/'cache';parent=tmp_path/'parent';v2=tmp_path/'v2'
    for folder in (cache,parent,v2):folder.mkdir()
    m.CACHE=cache;m.PARENT=parent;m.V2=v2;m.HERE=v2
    old=rows()
    if mode=='mixed_retry':old[1]['verbatim_id']='WAKE-B'
    refs=[{'id':r['verbatim_id'],'text':r['verbatim_id']} for r in old]
    (v2/'verbatims.jsonl').write_text('\n'.join(json.dumps(r) for r in refs if not r['id'].startswith('WAKE-')))
    (v2/'wake-verbatims.jsonl').write_text('\n'.join(json.dumps(r) for r in refs if r['id'].startswith('WAKE-')))
    (v2/'generation-protocol.json').write_text('{}')
    for name in m.MANIFESTS:
        values=old if name=='parler-manifest.jsonl' else []
        (parent/name).write_text('\n'.join(json.dumps(r) for r in values))
    monkeypatch.setattr(m,'activate_study',lambda:None)
    validations=[];verifications=[]
    def validate():
        validations.append(True)
        if len(validations)==2:
            if mode=='final_validation':raise ValueError('Final integrity check failed')
            if mode=='protocol_changed':(v2/'generation-protocol.json').write_text('{"changed":true}')
        return dict(records=2,tts_assets_sha256='fixture')
    def verify(*_):
        verifications.append(True)
        return dict(changed_before=['bad'] if mode=='weight_drift' or (mode=='final_weight_drift' and len(verifications)==2) else [])
    monkeypatch.setattr(m,'validate',validate)
    monkeypatch.setattr(m,'repair_and_verify',verify)
    monkeypatch.setitem(sys.modules,'revision',types.SimpleNamespace(VOICES=[dict(id='train-us-m',parler='Jon')],
        compile_catalog=lambda commands:dict(entries=[dict(id=r['id'],phones=[1,1,2]) for r in commands])))
    calls=[];saved=[];seeds=[]
    def fake(texts,voice):
        calls.append(texts);seeds.append(torch.initial_seed());fake.last_metadata=[{} for _ in texts]
        short=mode=='too_short' or (mode=='mixed_retry' and len(calls)==1)
        return [np.zeros(400 if short else 1600,dtype=np.float32) for _ in texts],16000
    @contextmanager
    def loader():yield fake,dict(model=None,verification={'fixture':True})
    monkeypatch.setattr(m,'verified_loader',loader)
    def prepare(wave,rate):
        stream=io.BytesIO();sf.write(stream,wave,rate,format='WAV',subtype='PCM_16')
        return (stream.getvalue(),stream.getvalue())
    def persist(root,row,wave,rate,prepared):saved.append(row);return dict(row)
    monkeypatch.setitem(sys.modules,'synthesis',types.SimpleNamespace(prepare_audio=prepare,persist_audio=persist))
    job=cache/'jobs'/'fixture'
    original_threads=torch.get_num_threads()
    try:
        if mode in ('success','mixed_retry'):m.generate(job)
        else:
            with pytest.raises(ValueError):m.generate(job)
    finally:torch.set_num_threads(original_threads)
    state=json.loads((job/'status.json').read_text())
    if mode in ('success','mixed_retry'):
        assert state['status']=='SUCCEEDED' and state['generated']==2 and len(saved)==2
        assert json.loads((cache/'generation-complete.json').read_text())['training_started'] is False
        if mode=='mixed_retry':
            assert seeds==[10,11] and calls==[['a','WAKE-B'],['a','WAKE-B']]
            assert all(r['batch_seed']==11 and r['generation_attempt']==2 for r in m.read_rows(cache/'replacements.jsonl'))
    elif mode in ('final_weight_drift','final_validation','protocol_changed'):
        assert state['status']=='FAILED' and len(saved)==2
        assert (cache/'replacements.jsonl').exists() and not (cache/'generation-complete.json').exists()
        assert not (cache/'parler-manifest.jsonl').exists()
    else:
        assert state['status']=='FAILED' and not saved and not (cache/'generation-complete.json').exists()
        assert len(calls)==(3 if mode=='too_short' else 1)
    with pytest.raises(FileExistsError):m.generate(cache/'jobs'/'duplicate')
    assert json.loads((job/'status.json').read_text())==state


@pytest.mark.parametrize('inherited',[False,True])
def test_loader_restores_class_descriptor_on_failure(monkeypatch,inherited):
    import sys,types
    m=module();monkeypatch.setattr(m,'activate_study',lambda:None)
    class Base:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):raise RuntimeError('fixture load failure')
    class Derived(Base):pass
    cls=Derived if inherited else Base
    before=dict(cls.__dict__)
    monkeypatch.setitem(sys.modules,'parler_tts',types.SimpleNamespace(ParlerTTSForConditionalGeneration=cls))
    monkeypatch.setitem(sys.modules,'synthesis',types.SimpleNamespace(load_engine=lambda _:cls.from_pretrained()))
    with pytest.raises(RuntimeError,match='fixture load failure'):
        with m.verified_loader():pass
    assert dict(cls.__dict__)==before
