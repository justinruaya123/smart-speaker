"""Final export must preserve quality gates, ambiguous rows and bundle identity."""
import importlib

import pytest


def adapter():
    name='scripts.train.speaker_final_release'
    assert importlib.util.find_spec(name) is not None, 'final export adapter is missing'
    return importlib.import_module(name)


def metrics(human=10,synthetic=10,negative=0):
    return {'human':{'raw_exact':human,'raw_negative_errors':negative},
            'synthetic':{'raw_exact':synthetic,'raw_negative_errors':0}}


def test_quantization_allows_only_declared_development_losses():
    m=adapter()
    assert m.quantization_gate(metrics(),metrics(9,9,1),3_200_000)['pass'] is True
    assert m.quantization_gate(metrics(),metrics(8,10,0),3_200_000)['pass'] is False
    assert m.quantization_gate(metrics(),metrics(10,10,2),3_200_000)['pass'] is False
    assert m.quantization_gate(metrics(),metrics(),4*1024*1024)['pass'] is False


def calibration_row(command,synthetic=False):
    from scripts.runtime.pcv_schema import INTENTS
    return dict(expected=command,candidate=command,synthetic=synthetic,score=-1. if command else None,
                margin=.1,intent_index=INTENTS.index(command['intent'])+1 if command else 0,
                intent_probability=.999,top_intent_probability=.999,target_named_probability=0.)


def test_missing_human_calibration_classes_cannot_enable_acceptance():
    m=adapter()
    rows=[calibration_row({'intent':'MESSAGE','slots':{}}),calibration_row(None)]
    policy=m.calibrate_human(rows)
    assert policy['enabled'] is False
    assert 'CALL' in policy['missing_positive_intents']
    assert 'MESSAGE' not in policy['missing_positive_intents']


def test_synthetic_calibration_cannot_fill_human_gates():
    with pytest.raises(ValueError,match='human'):
        adapter().calibrate_human([calibration_row({'intent':'MESSAGE','slots':{}},synthetic=True)])


def test_complete_perfect_human_calibration_can_enable_existing_gates():
    from scripts.runtime.pcv_schema import baseline_cases,INTENTS
    commands={r['command']['intent']:r['command'] for r in baseline_cases() if r['command']}
    assert set(commands)==set(INTENTS)
    rows=[calibration_row(command) for command in commands.values()]+[calibration_row(None)]
    assert adapter().calibrate_human(rows)['enabled'] is True


def test_ambiguous_published_rows_remain_outside_typed_metrics():
    rows=[dict(id='positive',metric_eligible=True,reference_status='supported',command={'intent':'MESSAGE','slots':{}}),
          dict(id='unknown',metric_eligible=True,reference_status='unsupported',command=None),
          dict(id='ambiguous',metric_eligible=False,reference_status='ambiguous',command=None)]
    eligible,excluded=adapter().partition_evaluation(rows)
    assert [r['id'] for r in eligible]==['positive','unknown']
    assert excluded==[rows[2]] and excluded[0]['reference_status']=='ambiguous'


def test_mislabelled_reference_cannot_become_a_negative():
    with pytest.raises(ValueError,match='reference'):
        adapter().partition_evaluation([dict(id='wrong',metric_eligible=True,reference_status='supported',command=None)])


def test_public_smoke_samples_keep_only_actual_slurp_origin():
    common=dict(kind='human',split='test',published_split='test',reference_status='supported',metric_eligible=True,
                command={'intent':'MESSAGE','slots':{}},sha256='a'*64,parquet='original.parquet',parquet_row=1,
                speaker_id='speaker',id='source-row')
    rows=[dict(common,publisher_metadata=dict(source='real_voice',transcript='Message',file='group.wav')),
          dict(common,id='slurp-row',publisher_metadata=dict(source='SLURP',transcript='Message',file='slurp.wav'))]
    result=adapter().sample_rows(rows)
    assert len(result)==1 and result[0]['source']=='SLURP'
    assert result[0]['file']=='slurp.wav' and result[0]['audio_sha256']==common['sha256']


def test_pi_archive_includes_current_config_resolver_and_cpu_default(tmp_path):
    import hashlib,json,zipfile
    m=adapter();original=tmp_path/'base.zip';payload=b'unchanged model bytes'
    with zipfile.ZipFile(original,'x') as archive:
        archive.writestr('models/default/model.onnx',payload)
        archive.writestr('MANIFEST.json',json.dumps({'format':'pcv-endpoint-pi-v1',
            'files':{'models/default/model.onnx':hashlib.sha256(payload).hexdigest()}}))
    before=original.read_bytes();output=tmp_path/'pi.zip'
    m.complete_pi_archive(original,output)
    assert original.read_bytes()==before
    with zipfile.ZipFile(output) as archive:
        manifest=json.loads(archive.read('MANIFEST.json'))
        assert archive.read('models/default/model.onnx')==payload
        assert 'scripts/runtime/speaker_config.py' in manifest['files']
        assert b'engine = "onnxruntime"' in archive.read('config.toml')
        assert b'enabled = false' in archive.read('config.toml')
        assert set(archive.namelist())==set(manifest['files'])|{'MANIFEST.json'}
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest


@pytest.mark.parametrize('changed',['human/calibration','evaluation/test','evaluation/holdout'])
def test_modified_prepared_evaluation_copy_is_rejected_before_lock(tmp_path,changed):
    import json
    from scripts.test.speaker_demo_suite import file_digest
    m=adapter();manifests={}
    for name in ('human/calibration','evaluation/test','evaluation/holdout'):
        path=tmp_path/'views'/f'{name}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{"id":"original"}\n')
        manifests[name]={'path':str(path),'sha256':file_digest(path)}
    (tmp_path/'preparation.json').write_text(json.dumps({'manifests':manifests}))
    assert len(m.frozen_evaluation_views(tmp_path))==3
    (tmp_path/'views'/f'{changed}.jsonl').write_text('{"id":"changed"}\n')
    with pytest.raises(ValueError,match='identity mismatch'):
        m.frozen_evaluation_views(tmp_path)


def test_package_rechecks_evaluation_bound_runtime_sources(tmp_path,monkeypatch):
    import json
    from scripts.test.speaker_demo_suite import file_digest
    m=adapter();source=tmp_path/'runtime.py';source.write_text('original runtime')
    protocol=tmp_path/'release-protocol-before-evaluation.json'
    protocol.write_text(json.dumps({'bindings':{str(source):file_digest(source)}}))
    receipt={'bindings':{str(protocol):file_digest(protocol)}}
    monkeypatch.setattr(m,'HERE',tmp_path)
    m.verify_release_inputs(receipt)
    source.write_text('changed after evaluation')
    with pytest.raises(ValueError,match='identity mismatch'):
        m.verify_release_inputs(receipt)
