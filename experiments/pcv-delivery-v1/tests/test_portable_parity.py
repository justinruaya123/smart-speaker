import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from test_bundle import fixture_bundle

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts import pcv_portable_parity as parity


def command_result():
    command=dict(intent='BRIGHTNESS',entity_id='light.kitchen',value=0)
    return dict(output=dict(schema_version=2,accepted=True,command=command,reason='accepted'),
        candidate=command,endpoint_detected=True,capture_fault=None,
        diagnostic=dict(command=command,score=-.1,margin=.4,phrase_id='zero',use_case='UC-04',
            endpoint_detected=True,end_sample=32000,capture_fault=None,encoder_ms=2.))


def test_comparison_ignores_compute_time_but_catches_typed_decisions_and_endpoints():
    original=command_result();other=copy.deepcopy(original)
    other['diagnostic'].update(encoder_ms=40.,score=-.100001)
    assert parity.compare_command(original,other)['equal']
    for change in ('type','decision','endpoint','score','missing_score','nonfinite'):
        altered=copy.deepcopy(other)
        if change=='type':altered['output']['command']['value']=False
        elif change=='decision':altered['output']['accepted']=False
        elif change=='endpoint':altered['diagnostic']['end_sample']=32001
        elif change=='score':altered['diagnostic']['score']=-.2
        elif change=='missing_score':del altered['diagnostic']['score']
        else:altered['diagnostic']['score']=float('nan')
        assert not parity.compare_command(original,altered)['equal'],change


def test_wake_comparison_preserves_event_order_count_and_scores():
    a=[dict(stage='wake',status='detected',wake_phrase='hey jarvis',score=.7),
       dict(stage='wake',status='command_ready'),dict(stage='command',accepted=False,command=None,reason='unknown')]
    b=copy.deepcopy(a);b[0]['score']+=.000001
    assert parity.compare_wake(a,b)['equal']
    assert not parity.compare_wake(a,b[::-1])['equal']
    assert not parity.compare_wake(a,b[:-1])['equal']
    b[0]['score']=.8
    assert not parity.compare_wake(a,b)['equal']


def test_development_selection_rejects_test_rows_duplicates_and_missing_coverage():
    rows=[dict(recording_id='qwen/dev/a',split='dev'),dict(recording_id='parler/dev/a',split='dev')]
    expected={r['recording_id'] for r in rows}
    parity.validate_development_rows(rows,expected)
    for changed in (rows[:1],rows+[rows[0]],[rows[0],dict(rows[1],split='test')]):
        with pytest.raises(ValueError):parity.validate_development_rows(changed,expected)


def test_portable_identity_rejects_different_graph_and_inherited_thresholds(fixture_bundle,tmp_path):
    from pcvcm.contract import file_hash,write_json
    from scripts.runtime.pcv_deploy.bundle import verify_bundle
    source=tmp_path/'source';source.mkdir()
    for name in ('model.onnx','catalog.json','normalization.json','endpoint.json'):
        (source/name).write_bytes((fixture_bundle/name).read_bytes())
    write_json(source/'synthetic-bundle.json',dict(binding='source'))
    write_json(source/'policy.json',dict(binding='source',enabled=False,thresholds={}))
    manifest=json.loads((fixture_bundle/'deployment.json').read_text());manifest.update(
        source_binding='source',source_manifest_sha256=file_hash(source/'synthetic-bundle.json'))
    write_json(fixture_bundle/'deployment.json',manifest)
    portable=verify_bundle(fixture_bundle,allow_diagnostic=True)
    parity.verify_equivalence_inputs(source,portable)
    (source/'model.onnx').write_bytes(b'other')
    with pytest.raises(ValueError,match='model.onnx'):parity.verify_equivalence_inputs(source,portable)
    (source/'model.onnx').write_bytes((fixture_bundle/'model.onnx').read_bytes())
    portable['policy']['thresholds']={'global':[-99.,0.]}
    with pytest.raises(ValueError,match='policy'):parity.verify_equivalence_inputs(source,portable)


def test_real_cpu_waveform_comparison_and_changed_audio_rejection(fixture_bundle,tmp_path):
    import soundfile as sf
    from pcvcm.runtime import StreamEngine,OnnxEncoder
    from pcvcm.contract import file_hash
    from scripts.runtime.pcv_deploy.bundle import load_bundle,STATE_SHAPES
    from scripts.runtime.pcv_deploy.catalog import CatalogContract
    from scripts.runtime.wake_models import HeyJarvis,SileroProbability
    from wake_gate import WakeGate
    portable=load_bundle(fixture_bundle,allow_diagnostic=True)
    reference=StreamEngine(CatalogContract(portable['catalog']),portable['catalog'],
        OnnxEncoder(fixture_bundle/'model.onnx',STATE_SHAPES,threads=2),portable['normalization'],
        portable['endpoint'],portable['policy'],portable['binding'])
    reference_gate=WakeGate(reference,HeyJarvis(ROOT/'artifacts'),SileroProbability(ROOT/'artifacts'))
    path=tmp_path/'noise.wav';sf.write(path,np.random.default_rng(3).normal(0,.03,16000),16000,subtype='PCM_16')
    pcm,_=sf.read(path,dtype='float32')
    row=dict(audio=str(path),audio_sha256=file_hash(path),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
        recording_id='fixture',verbatim_id='fixture',phase=1,use_case='UC-03',engine='fixture',
        accent_requested='none',voice_group='fixture',split='dev',expected_command=None,critical_zero_accept=False)
    result=parity.compare_recording(reference_gate,portable['gate'],row,mode='command')
    assert result['equal'] and result['source']['candidate']==dict(intent='POWER_ON',entity_id='light.kitchen')
    assert result['source']['output']['reason']=='policy_disabled'
    assert result['portable']['diagnostic']['end_sample']==result['source']['diagnostic']['end_sample']
    result=parity.compare_recording(reference_gate,portable['gate'],row,mode='wake')
    assert result['equal'] and result['source']['events']==[]
    row['audio_sha256']='changed'
    with pytest.raises(ValueError,match='audio'):parity.compare_recording(reference_gate,portable['gate'],row,mode='command')


def test_receipt_verification_checks_full_coverage_mismatches_and_runtime_identity(tmp_path,monkeypatch):
    from pcvcm.contract import canonical,file_hash,write_json
    root=tmp_path/'repo';definitions=root/'experiments/pcv-synthetic-v2';definitions.mkdir(parents=True)
    command=dict(recording_id='qwen/dev/a',verbatim_id='a',engine='qwen',voice_group='dev',split='dev')
    wake=dict(recording_id='qwen/dev/wake-a',verbatim_id='wake-a',engine='qwen',voice_group='dev',split='dev')
    for name,data in [('generation-plan.jsonl',[command]),('wake-generation-plan.jsonl',[wake]),
            ('wake-verbatims.jsonl',[dict(id='wake-a',case_type='wake_command',command_verbatim_id='a')])]:
        (definitions/name).write_text(''.join(canonical(r)+'\n' for r in data),encoding='utf-8')
    runtime=root/'scripts/runtime/pcv_deploy';runtime.mkdir(parents=True)
    (runtime/'bundle.py').write_text('# fixture runtime\n')
    selector=root/'scripts/pcv_study_context.py';selector.write_text('# fixture selector\n')
    monkeypatch.setattr(parity,'ROOT',root)
    output=tmp_path/'receipt';output.mkdir();manifest=tmp_path/'deployment.json';write_json(manifest,{'binding':'portable'})
    portable=dict(root=tmp_path,binding='portable',manifest={'source_binding':'source'})
    rows=[dict(recording_id=command['recording_id'],mode='command',equal=True,max_error=0.,
        source=command_result(),portable=command_result())]
    for key in ('qwen/dev/wake-a','no-wake/qwen/dev/a'):
        rows.append(dict(recording_id=key,mode='wake',equal=True,max_error=0.,
            source=dict(events=[],eof={'reason':'wake_required'}),portable=dict(events=[],eof={'reason':'wake_required'})))
    report=dict(complete=True,parity_passed=True,split='dev',test_audio_replayed=False,policy_changed=False,
        source_binding='source',deployment_binding='portable',deployment_manifest_sha256=file_hash(manifest),
        source_files={str(p):file_hash(p) for p in (Path(parity.__file__),selector,runtime/'bundle.py')},
        expected={'command':1,'wake':2},observed={'command':1,'wake':2},mismatches=0,max_error=0.,tolerance=2e-5)
    def save():
        (output/'rows.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
        report['rows_sha256']=file_hash(output/'rows.jsonl');write_json(output/'results.json',report)
    save()
    parity.verify_receipt(output/'results.json',portable)
    rows[0]['portable']['output']['accepted']=False;save()
    with pytest.raises(ValueError,match='mismatch'):parity.verify_receipt(output/'results.json',portable)
    rows[0]['portable']=copy.deepcopy(rows[0]['source']);rows.pop();save()
    with pytest.raises(ValueError,match='coverage'):parity.verify_receipt(output/'results.json',portable)
    (runtime/'bundle.py').write_text('# changed\n')
    with pytest.raises(ValueError,match='source'):parity.verify_receipt(output/'results.json',portable)
