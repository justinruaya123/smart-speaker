import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def module():
    spec=importlib.util.spec_from_file_location('v3_workflow_fixture',Path(__file__).resolve().parents[1]/'workflow.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_budget_copies_charged_epochs_exactly_once(tmp_path):
    m=module();source=tmp_path/'prior.json';target=tmp_path/'new.json'
    state=dict(limits={'epochs':200,'seconds':86400},seconds=22475.141,epochs=[{'run':'old','label':str(i)} for i in range(80)])
    source.write_text(json.dumps(state));before=source.read_bytes()
    m.carry_budget(source,target)
    assert target.read_bytes()==before and source.read_bytes()==before
    with pytest.raises(FileExistsError):m.carry_budget(source,target)
    state['limits']['epochs']=201;source.write_text(json.dumps(state))
    with pytest.raises(ValueError):m.carry_budget(source,tmp_path/'invalid.json')


@pytest.mark.parametrize('case',['missing','reset','shortened','lower_seconds','changed_prefix','valid','appended'])
def test_inherited_budget_cannot_disappear_or_reset(tmp_path,monkeypatch,case):
    m=module();parent=tmp_path/'parent';child=tmp_path/'child';parent.mkdir();child.mkdir()
    monkeypatch.setattr(m,'PARENT',parent);monkeypatch.setattr(m,'CACHE',child)
    original=dict(limits={'epochs':200,'seconds':86400},seconds=22475.141,epochs=[dict(run='old',label='1')])
    source=parent/'compute-budget.json';source.write_text(json.dumps(original))
    inherited=dict(path=str(source.relative_to(tmp_path)),sha256=m.file_hash(source))
    (child/'budget-inheritance.json').write_text(json.dumps(dict(source=inherited,inherited_state=original,copied_before_training=True)))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    state=json.loads(json.dumps(original))
    if case=='reset':state.update(seconds=0,epochs=[])
    if case=='shortened':state['epochs']=[]
    if case=='lower_seconds':state['seconds']=22475
    if case=='changed_prefix':state['epochs'][0]['run']='different'
    if case=='appended':state['seconds']+=1;state['epochs'].append(dict(run='new',label='1'))
    if case!='missing':(child/'compute-budget.json').write_text(json.dumps(state))
    if case in ('valid','appended'):
        m.validate_budget(inherited)
        if case=='valid':m.validate_budget(inherited,initial=True)
        else:
            with pytest.raises(ValueError,match='initial'):m.validate_budget(inherited,initial=True)
    else:
        with pytest.raises(ValueError):m.validate_budget(inherited)


def test_v3_budget_never_loses_prior_charge_to_clock_cancellation(tmp_path,monkeypatch):
    m=module();import fitting
    clock=[30000.003];monkeypatch.setattr(m,'monotonic',lambda:clock[0],raising=False)
    source=dict(limits={'epochs':200,'seconds':86400},seconds=17006.78,epochs=[dict(run='prior',label='1')])
    path=tmp_path/'budget.json';path.write_text(json.dumps(source))
    from pcvcm import compute
    monkeypatch.setattr(compute.time,'monotonic',lambda:clock[0])
    with m.bind():
        with fitting.ComputeBudget(path,'fixture') as budget:
            budget.check();assert budget.state['seconds']==source['seconds']
            clock[0]+=.25;budget.check();assert budget.state['seconds']==source['seconds']+.25
    assert json.loads(path.read_text())['seconds']>=source['seconds']+.25


def test_v3_budget_constructor_cannot_create_a_fresh_zero_budget(tmp_path):
    m=module();import fitting
    with m.bind(),pytest.raises(ValueError,match='missing'):
        with fitting.ComputeBudget(tmp_path/'missing.json','fixture'):pass
    assert not (tmp_path/'missing.json').exists()


@pytest.mark.parametrize('kind',['command','wake'])
def test_test_reservation_consumes_both_ancestors_once(tmp_path,kind):
    m=module();parents=[]
    for label in ('parent','child'):
        cache=tmp_path/label;cache.mkdir();here=cache/'defs';here.mkdir()
        for name in ('generation-plan.jsonl','wake-generation-plan.jsonl','wake-verbatims.jsonl'):(here/name).write_text(label)
        for name in m.gen.MANIFESTS:(cache/name).write_text(label+name)
        parents.append((cache,here))
    paths=m.reserve_ancestors(kind,tmp_path/'bundle',tmp_path/'out',parents=parents,protocol_sha256='fixture')
    assert len(paths)==2 and all(p.exists() for p in paths)
    before=[p.read_bytes() for p in paths]
    with pytest.raises(FileExistsError):m.reserve_ancestors(kind,tmp_path/'other',tmp_path/'out2',parents=parents,protocol_sha256='fixture')
    assert [p.read_bytes() for p in paths]==before
    assert all(json.loads(p.read_text())['state']=='RESERVED_PRESERVE_ON_FAILURE' for p in paths)


def test_audio_audit_checks_native_derived_and_ctc(tmp_path):
    import numpy as np
    import soundfile as sf
    m=module();rng=np.random.default_rng(12)
    pcm=rng.normal(0,.03,3200).astype(np.float32)
    native=tmp_path/'native.wav';audio=tmp_path/'audio.wav'
    sf.write(native,pcm,16000,subtype='PCM_16');sf.write(audio,pcm,16000,subtype='PCM_16')
    decoded,rate=sf.read(audio,dtype='float32')
    row=dict(audio=str(audio),native_audio=str(native),audio_sha256=m.file_hash(audio),native_sha256=m.file_hash(native),
        pcm_sha256=m.pcm_hash(decoded),sample_rate=16000,channels=1,native_sample_rate=16000)
    result=m.audit_audio(row,[tmp_path],phones=[1,1,2])
    assert result['ctc_capacity']['feasible'] and result['frames']==len(decoded)
    with pytest.raises(ValueError,match='CTC'):m.audit_audio(row,[tmp_path],phones=[1]*30)
    with pytest.raises(ValueError,match='escape'):m.audit_audio(row,[tmp_path/'other'])
    sf.write(audio,pcm*.5,rate,subtype='PCM_16')
    with pytest.raises(ValueError,match='identity'):m.audit_audio(row,[tmp_path])


def test_cross_command_wake_duplicates_and_split_voice_overlap_fail():
    m=module();rows=[dict(recording_id='a',pcm_sha256='x',voice_group='voice1',split='train',engine='parler'),
                    dict(recording_id='b',pcm_sha256='y',voice_group='voice2',split='dev',engine='parler')]
    assert m.audit_disjoint(rows)['unique_pcm']==2
    with pytest.raises(ValueError,match='Duplicate'):m.audit_disjoint([rows[0],dict(rows[1],pcm_sha256='x')])
    with pytest.raises(ValueError,match='voice'):m.audit_disjoint([rows[0],dict(rows[1],voice_group='voice1')])


def test_binding_restores_globals_and_keeps_parent_context():
    m=module();import revision,fitting,synthbench,wake_benchmark
    before=(fitting.CACHE,fitting.development,fitting.ComputeBudget,synthbench.load_split,wake_benchmark.audit_wake,revision.CACHE)
    with pytest.raises(RuntimeError):
        with m.bind():
            assert fitting.CACHE==m.CACHE and synthbench.load_split is m.load_split
            assert wake_benchmark.audit_wake is m.audit_wake and revision.CACHE==before[-1]
            assert fitting.development is m.parallel.development
            raise RuntimeError('fixture')
    assert (fitting.CACHE,fitting.development,fitting.ComputeBudget,synthbench.load_split,wake_benchmark.audit_wake,revision.CACHE)==before


def test_shared_cpu_onnx_stages_in_v3_context_preserve_failed_acceptance(tmp_path,monkeypatch):
    m=module();definitions=tmp_path/'definitions';definitions.mkdir()
    for source in m.PARENT_HERE.iterdir():
        if source.is_file() and source.suffix in ('.json','.jsonl'):(definitions/source.name).write_bytes(source.read_bytes())
    protocol=json.loads((definitions/'protocol.json').read_text());protocol['version']='pcv-synthetic-v3'
    (definitions/'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(m,'HERE',definitions);monkeypatch.setattr(m,'CACHE',tmp_path)
    monkeypatch.setattr(m,'validate',lambda:protocol)
    path=m.ROOT/'experiments/pcv-synthetic-v1/tests/test_stage_integration.py'
    spec=importlib.util.spec_from_file_location('v3_onnx_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with m.bind(),monkeypatch.context() as scoped:
        fixture.test_export_replay_parity_calibration_and_report_preserve_failed_acceptance(tmp_path,scoped)
    assert json.loads((tmp_path/'bundle/protocol.json').read_text())['version']=='pcv-synthetic-v3'


def test_complete_audio_audit_preserves_reuse_and_rejects_tampering(tmp_path,monkeypatch):
    """Six original/derived PCM fixtures exercise the actual combined auditor."""
    import numpy as np
    import soundfile as sf
    m=module();cache=tmp_path/'new';parent=tmp_path/'parent';here=tmp_path/'definitions';old_audio=tmp_path/'old-audio'
    for folder in (cache,parent,here,old_audio):folder.mkdir()
    monkeypatch.setattr(m,'CACHE',cache);monkeypatch.setattr(m,'PARENT',parent);monkeypatch.setattr(m,'HERE',here)
    monkeypatch.setattr(m.gen,'CACHE',cache);monkeypatch.setattr(m.rev,'OLD_CACHE',old_audio)
    monkeypatch.setattr(m.rev,'CACHE',tmp_path/'old-v2')
    monkeypatch.setattr(m,'validate',lambda:{'expected_clips':3})
    job=cache/'jobs'/'fixture';job.mkdir(parents=True);(job/'weight-verification.json').write_text('{}')
    monkeypatch.setattr(m,'require_generation_complete',lambda:({'generated':2},job))
    refs=[dict(id='C',text='fixture command',expected_command=None),dict(id='WAKE',text='fixture wake',expected_command=None)]
    (here/'verbatims.jsonl').write_text(json.dumps(refs[0])+'\n');(here/'wake-verbatims.jsonl').write_text(json.dumps(refs[1])+'\n')
    (here/'wake-generation-plan.jsonl').write_text('{}\n'*3)
    (here/'generation-protocol.json').write_text(json.dumps({'tts_assets_sha256':'assets'}));(here/'generation.py').write_text('fixture')
    (cache/'generation-complete.json').write_text('{}')
    monkeypatch.setattr(m.rev,'compile_catalog',lambda _:dict(entries=[dict(id='C',phones=[1,1,2])]))
    (old_audio/'anchors').mkdir();(old_audio/'anchors/dev-au-m.wav').write_bytes(b'anchor')
    anchor=m.file_hash(old_audio/'anchors/dev-au-m.wav')
    def write_rows(path,rr):path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
    def audio(row,base,seed):
        rng=np.random.default_rng(seed);wave=rng.normal(0,.04,8000).astype(np.float32)
        path=base/'audio'/f'{seed}.wav';native=base/'native'/f'{seed}.wav'
        path.parent.mkdir(exist_ok=True);native.parent.mkdir(exist_ok=True)
        sf.write(path,wave,16000,subtype='PCM_16');sf.write(native,wave,16000,subtype='PCM_16')
        pcm,_=sf.read(path,dtype='float32')
        return dict(row,audio=str(path),native_audio=str(native),audio_sha256=m.file_hash(path),native_sha256=m.file_hash(native),
            pcm_sha256=m.pcm_hash(pcm),sample_rate=16000,native_sample_rate=16000,channels=1)
    original={};new={};parler=[];replacements=[]
    for index,name in enumerate(m.gen.MANIFESTS):
        engine=name.removeprefix('wake-').removesuffix('-manifest.jsonl');vid='WAKE' if name.startswith('wake-') else 'C'
        row=dict(recording_id=f'{engine}/dev-au-m/{vid}',verbatim_id=vid,engine=engine,voice_group='dev-au-m',split='dev',
            accent_requested='Australian English',seed=index,synthetic=True,voice_anchor_sha256=None if engine=='parler' else anchor)
        row=audio(row,old_audio,index)
        original[name]=[row]
        if engine=='parler':parler.append(row)
        else:new[name]=[row]
    batch=[r['recording_id'] for r in parler]
    for old in parler:old.update(batch_ids=batch,batch_size=4)
    for old in parler:
        row=m.gen.planned_row(old,cache);seed=100+old['seed'];generated=audio(row,cache,seed)
        # Use the declared per-ID paths rather than the helper's seed filenames.
        for key in ('audio','native_audio'):
            target=Path(row[key]);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(generated[key]).read_bytes());generated[key]=str(target)
        generated.update(batch_ids=batch,batch_size=4,batch_seed=parler[0]['seed'],generation_attempt=1,
            generation_protocol_sha256=m.file_hash(here/'generation-protocol.json'),generation_source_sha256=m.file_hash(here/'generation.py'),
            verbatim_sha256=m.gen.identity(next(r for r in refs if r['id']==old['verbatim_id'])),
            weight_verification_path=str(job/'weight-verification.json'),weight_verification_sha256=m.file_hash(job/'weight-verification.json'),
            tts_assets_sha256='assets',voice_conditioning=dict(type='named_voice_description',name='Gary'),
            ctc_capacity=m.gen.ctc_capacity(8000,[1,1,2]) if old['verbatim_id']=='C' else None)
        replacements.append(generated)
        name=('wake-' if old['verbatim_id']=='WAKE' else '')+'parler-manifest.jsonl';new[name]=[generated]
    monkeypatch.setattr(m.gen,'parent_rows',lambda:parler)
    for name in m.gen.MANIFESTS:write_rows(parent/name,original[name]);write_rows(cache/name,new[name])
    write_rows(cache/'replacements.jsonl',replacements)
    result=m.audit_all_splits()
    assert result['recordings']==3 and result['wake_recordings']==3 and result['combined_unique_pcm']==6
    broken=[dict(replacements[0],batch_seed=999),replacements[1]]
    write_rows(cache/'replacements.jsonl',broken);write_rows(cache/'parler-manifest.jsonl',[broken[0]])
    with pytest.raises(ValueError,match='provenance'):m.audit_all_splits()
    write_rows(cache/'replacements.jsonl',replacements);write_rows(cache/'parler-manifest.jsonl',[replacements[0]])
    changed=dict(new['qwen-manifest.jsonl'][0],seed=999);write_rows(cache/'qwen-manifest.jsonl',[changed])
    with pytest.raises(ValueError,match='Merged'):m.audit_all_splits()
