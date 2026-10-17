import hashlib
import importlib
import json
import zipfile

import pytest


def module():
    return importlib.import_module('scripts.train.speaker_compact_release')


def metrics(human=10, synthetic=10, negative=0):
    return {'human':{'raw_exact':human,'raw_negative_errors':negative},
            'synthetic':{'raw_exact':synthetic,'raw_negative_errors':0}}


def test_compact_graph_cannot_use_parent_four_mib_allowance():
    assert module().quantization_gate(metrics(),metrics(),1550000)['pass'] is True
    assert module().quantization_gate(metrics(),metrics(),2100000)['pass'] is False
    assert module().quantization_gate(metrics(),metrics(8),1550000)['pass'] is False


def test_new_archive_keeps_model_and_closes_demo_runtime_dependencies(tmp_path):
    base=tmp_path/'base.zip'; output=tmp_path/'compact.zip'; bundle=tmp_path/'bundle';bundle.mkdir()
    payloads={'models/default/model.onnx':b'unchanged selected graph',
              'pi_test.py':b'from scripts.runtime.pcv_pi_benchmark import main\n',
              'models/default/deployment.json':json.dumps({'files':{}}).encode()}
    (bundle/'recall-policy.json').write_text('{"policy":"bound by actual recipe"}')
    with zipfile.ZipFile(base,'x') as archive:
        for name,data in payloads.items():archive.writestr(name,data)
        archive.writestr('MANIFEST.json',json.dumps({'format':'pcv-endpoint-pi-v1','files':{
            name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}}))
    before=base.read_bytes();module().complete_archive(base,output,bundle)
    assert base.read_bytes()==before
    with zipfile.ZipFile(output) as archive:
        assert archive.read('models/default/model.onnx')==payloads['models/default/model.onnx']
        assert b'speaker_compact_benchmark' in archive.read('pi_test.py')
        for name in ('scripts/runtime/speaker_compact.py','scripts/runtime/speaker_compact_benchmark.py',
                     'scripts/runtime/speaker_config.py','models/default/recall-policy.json'):
            assert name in archive.namelist()
        assert b'mode = "demo"' in archive.read('config.toml')
        assert b'enabled = false' in archive.read('config.toml')
        manifest=json.loads(archive.read('MANIFEST.json'))
        assert manifest['demo_acceptance_enabled'] is True
        assert manifest['qualified_acceptance_enabled'] is False
        assert set(archive.namelist())==set(manifest['files'])|{'MANIFEST.json'}
        for name,digest in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest


def test_release_rechecks_nested_evaluation_sources(tmp_path,monkeypatch):
    m=module();source=tmp_path/'runtime.py';source.write_text('evaluated source')
    protocol=tmp_path/'release-protocol-before-evaluation.json'
    protocol.write_text(json.dumps({'bindings':{str(source):hashlib.sha256(source.read_bytes()).hexdigest()}}))
    receipt={'bindings':{str(protocol):hashlib.sha256(protocol.read_bytes()).hexdigest()}}
    monkeypatch.setattr(m,'HERE',tmp_path)
    m.verify_result_bindings(receipt)
    source.write_text('changed runtime')
    with pytest.raises(ValueError,match='identity'):
        m.verify_result_bindings(receipt)


def test_demo_benchmark_keeps_qualified_metrics_separate():
    m=importlib.import_module('scripts.runtime.speaker_compact_benchmark')
    time={'intent':'TIME','slots':{}}
    trials=[dict(expected=time,event=dict(fixed_application={'command':None},application={'command':time},
                qualified_output={'command':None})),
            dict(expected=None,event=dict(fixed_application={'command':None},application={'command':None},
                qualified_output={'command':None}))]
    result=m.decision_metrics(trials)
    assert result['relaxed_demo']['exact']==1 and result['relaxed_demo']['negative_errors']==0
    assert result['fixed_demo']['exact']==0 and result['qualified']['accepted']==0


def test_wording_results_join_exact_text_and_preserve_untested_and_conflicts():
    play={'intent':'PLAY_MUSIC','slots':{}};time={'intent':'TIME','slots':{}}
    definitions=[dict(id='play',spoken_text='Play music!',command=play),
                 dict(id='time',spoken_text='What time is it?',command=time)]
    source=[dict(id='a',kind='human',publisher_metadata={'transcript':'play music'},command=play),
            dict(id='b',kind='human',publisher_metadata={'transcript':'PLAY MUSIC.'},command=time),
            dict(id='c',kind='synthetic',publisher_metadata={'transcript':'What time is it'},command=time)]
    rows=[dict(id='a',expected=play,synthetic=False,speaker='one',candidate=play,
               application_command=None,relaxed_command=play,qualified_command=None),
          dict(id='c',expected=time,synthetic=True,speaker='voice',candidate=play,
               application_command=play,relaxed_command=play,qualified_command=None)]
    result=module().wording_results(rows,source,definitions)
    assert len(result)==2
    assert result[0]['human']['n']==1 and result[0]['human']['reference_conflicts']==1
    assert result[0]['human']['raw']['correct']==1
    assert result[0]['human']['fixed_demo']['rejected']==1
    assert result[0]['human']['relaxed_demo']['correct']==1
    assert result[0]['human']['qualified']['rejected']==1
    assert result[1]['human']['n']==0 and result[1]['human']['untested'] is True
    assert result[1]['synthetic']['relaxed_demo']['wrong']==1


@pytest.mark.parametrize('explicit',[True,False])
def test_compact_cli_rejects_unavailable_ncnn(tmp_path,monkeypatch,explicit):
    import sys
    from scripts.runtime import speaker_compact_benchmark as benchmark
    from scripts.runtime import speaker_config
    args=['benchmark','--root',str(tmp_path),'--output',str(tmp_path/'result')]
    if explicit:args+=['--engine','ncnn']
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.setattr(speaker_config,'resolve_config',lambda *a,**k:({'inference':{'engine':'ncnn','mode':'demo'}},{}))
    monkeypatch.setattr(benchmark,'run',lambda args:pytest.fail('Unavailable engine reached execution'))
    with pytest.raises(SystemExit) as error:benchmark.main()
    assert error.value.code==2
