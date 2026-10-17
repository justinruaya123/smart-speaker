"""Actual tiny CPU graphs exercise accounting; fixtures do not qualify speech."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def graph(path, inputs, outputs, nodes):
    from onnx import helper
    model = helper.make_model(helper.make_graph(nodes, 'resource_software_fixture', inputs, outputs),
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 9
    import onnx
    onnx.save(model, path)


@pytest.fixture
def components(tmp_path):
    pytest.importorskip('onnxruntime'); pytest.importorskip('onnx')
    from onnx import helper, TensorProto as T
    from scripts.runtime.audio import DEFAULT
    from scripts.runtime.iris_owner import PCM_NORMALIZATION
    from scripts.test.test_iris_pi_benchmark import tiny_bundle
    bundle = tiny_bundle(tmp_path/'vcm')
    write(bundle/'ncnn/position-frequency.json', [1.])
    for name in ('model.ncnn.param', 'model.ncnn.bin'):
        (bundle/'ncnn'/name).write_bytes(b'ORT-only software inventory fixture')
    write(bundle/'ncnn/engine.json', {'onnx_sha256':sha(bundle/'model.onnx'), 'parity':{'passed':True},
        'files':{name:sha(bundle/'ncnn'/name) for name in
                 ('model.ncnn.param', 'model.ncnn.bin', 'position-frequency.json')}})
    metadata = json.loads((bundle/'bundle.json').read_text()); metadata['parameters']=11
    write(bundle/'bundle.json', metadata)
    wake=tmp_path/'wake'; wake.mkdir()
    constant=lambda name,values,shape:helper.make_node('Constant', [], [name],
        value=helper.make_tensor(name, T.FLOAT, shape, values))
    graph(wake/'iris.onnx', [helper.make_tensor_value_info('features',T.FLOAT,[1,40,'time'])],
          [helper.make_tensor_value_info('wake',T.FLOAT,[1,2])], [constant('wake',[0.,8.],[1,2])])
    for name in ('iris.ncnn.param','iris.ncnn.bin'):(wake/name).write_bytes(b'ORT-only inventory fixture')
    write(wake/'iris.json', {'files':{name:sha(wake/name) for name in
        ('iris.onnx','iris.ncnn.param','iris.ncnn.bin')},'normalization':{'mean':[0.]*40,'std':[1.]*40},
        'threshold':.9,'release':.3,'consecutive':2,'cooldown_samples':16000,'parameters':13})
    owner=tmp_path/'owner';owner.mkdir()
    graph(owner/'owner.onnx', [helper.make_tensor_value_info('features',T.FLOAT,[1,40,'time']),
        helper.make_tensor_value_info('lengths',T.INT64,[1])],
        [helper.make_tensor_value_info('embedding',T.FLOAT,[1,128]),helper.make_tensor_value_info('overlap',T.FLOAT,[1])],
        [constant('embedding',[1.]+[0.]*127,[1,128]),constant('overlap',[.01],[1])])
    write(owner/'owner.json', {'schema_version':1,'frontend':asdict(DEFAULT),
        'pcm_normalization':PCM_NORMALIZATION,'calibration_kind':'short_utterance_proxy+synthetic_wake_proxy',
        'model':{'path':'owner.onnx','sha256':sha(owner/'owner.onnx'),'runtime_parameters':17},
        'files':{'owner.onnx':sha(owner/'owner.onnx')},'normalization':{'mean':[0.]*40,'std':[1.]*40},
        'policy':{'threshold':.5,'overlap_threshold':.5}})
    vad=tmp_path/'vad';vad.mkdir()
    graph(vad/'silero-vad.onnx', [helper.make_tensor_value_info('input',T.FLOAT,[1,576]),
        helper.make_tensor_value_info('state',T.FLOAT,[2,1,128]),helper.make_tensor_value_info('sr',T.INT64,[])],
        [helper.make_tensor_value_info('probability',T.FLOAT,[1]),helper.make_tensor_value_info('state_out',T.FLOAT,[2,1,128])],
        [constant('probability',[.8],[1]),helper.make_node('Identity',['state'],['state_out'])])
    write(vad/'wake-models.json', {'files':{'silero-vad.onnx':{'bytes':(vad/'silero-vad.onnx').stat().st_size,
        'sha256':sha(vad/'silero-vad.onnx'),'license':'MIT'}}})
    personal=[];generated=[]
    for kind,target in [('command',personal),('wake',generated)]:
        for i in range(8):
            path=tmp_path/'audio'/f'{kind}-{i}.wav';path.parent.mkdir(exist_ok=True)
            seconds=2.1 if kind=='command' else 1.0
            sf.write(path,np.sin(np.arange(round(16000*seconds))*(.1+i*.001))*.04,16000,subtype='PCM_16')
            target.append({'id':f'{kind}-{i}','audio':str(path.relative_to(tmp_path)),'sha256':sha(path),
                'expected':{'intent':'TIMER','slots':{'duration_seconds':123}} if kind=='command' else None,
                'role':kind,'wake_positive':kind=='wake','split':'voice-test' if kind=='wake' else 'test',
                'synthetic':True,'source':'Qwen3-original-designed-resource-software-fixture'})
    manifest=tmp_path/'.cache/iris-release-v1/inputs/base-views/personal-test.jsonl'
    generated_path=tmp_path/'.cache/iris-release-v1/diversity/generated.jsonl'
    for path,rows in [(manifest,personal),(generated_path,generated)]:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    protocol=tmp_path/'experiments/iris-release-v1/protocol.json'
    write(protocol,{'input_bindings':{str(manifest.relative_to(tmp_path)):sha(manifest)}})
    write(protocol.with_name('generation-complete.json'),{'manifest_sha256':sha(generated_path),'protocol_sha256':sha(protocol)})
    write(tmp_path/'data/sources/qwen-synthetic-completion.source.json',{'license':'Apache-2.0'})
    return dict(root=tmp_path,bundle=bundle,wake=wake,owner=owner,vad_folder=vad,output=tmp_path/'result.json')


def test_real_cpu_components_include_every_owner_window_and_prefix_without_embeddings(components):
    from scripts.test.iris_resources import profile
    result=profile(**components,engine='onnxruntime')
    assert result['actions_executed'] is False and result['physical_pi_measured'] is False
    assert result['measurements']['command_n']==result['measurements']['wake_n']==8
    trials=result['measurements']['commands']
    assert all(row['preview_calls']==4 and len(row['preview_candidates'])==4 for row in trials)
    assert all(row['owner_continuity_calls']>=3 for row in trials)
    assert all(row['compute_budget']['wall_ms']>=row['timing']['vcm_final']['wall_ms'] for row in trials)
    for row,wake in zip(trials,result['measurements']['wakes']):
        for key in ('wall_ms','cpu_ms'):
            expected=sum(stage[key] for name,stage in row['timing'].items() if name not in
                         ('vcm_frontend','vcm_engine_decision_and_wrapper'))
            expected+=sum(preview['timing'][key] for preview in row['preview_candidates'])
            assert row['compute_budget'][key]==pytest.approx(expected)
            assert row['complete_compute_budget'][key]==pytest.approx(expected+wake['active'][key])
        assert row['preview_candidate_changes']==0
        assert all(preview['raw_candidate']['intent']=='TIMER' for preview in row['preview_candidates'])
    assert result['measurements']['component_latency_ms']['vcm_preview']['wall_ms']['n']==32
    assert result['memory']['resident_sessions']=={'vcm':1,'wake':1,'owner':1,'silero':2}
    assert result['parameters']['known_own_runtime_total']==41
    assert result['parameters']['silero']['trainable_parameters'] is None
    assert result['sizes']['graph_bytes']==sum(row['bytes'] for row in result['sizes']['files'] if row['kind']=='graph')
    saved=json.loads(components['output'].read_text())
    assert saved==result
    assert '"embedding":' not in json.dumps(saved)


def test_frozen_receipt_verifies_bindings_and_does_not_load_or_remeasure(components,monkeypatch):
    from scripts.test import iris_resources as module
    original=module.profile(**components,engine='onnxruntime')
    frozen=components['output'].read_bytes()
    monkeypatch.setattr(module,'load_components',lambda *args,**kwargs:pytest.fail('A frozen receipt cannot remeasure'))
    assert module.profile(**components,engine='onnxruntime')==original
    assert components['output'].read_bytes()==frozen
    with pytest.raises(ValueError,match='binding|changed|preserv'):
        module.profile(**components,engine='onnxruntime',threads=2)
    (components['wake']/'iris.json').write_text((components['wake']/'iris.json').read_text()+' ')
    with pytest.raises(ValueError,match='binding|changed|preserv'):
        module.profile(**components,engine='onnxruntime')
    assert components['output'].read_bytes()==frozen


def test_missing_complete_models_or_generation_never_writes_partial_receipt(tmp_path):
    from scripts.test.iris_resources import profile
    output=tmp_path/'pipeline-resources.json'
    with pytest.raises((ValueError,FileNotFoundError)):
        profile(root=tmp_path,bundle=tmp_path/'vcm',wake=tmp_path/'wake',owner=tmp_path/'owner',
                vad_folder=tmp_path/'vad',output=output)
    assert not output.exists()


def test_unapproved_audio_cannot_enter_a_resource_receipt(components):
    from scripts.test.iris_resources import profile
    source=components['root']/'data/sources/qwen-synthetic-completion.source.json'
    write(source,{'license':'unknown'})
    with pytest.raises(ValueError,match='license|synthetic'):
        profile(**components,engine='onnxruntime')
    assert not components['output'].exists()


def test_cli_defaults_and_independent_output_override(tmp_path,monkeypatch,capsys):
    from scripts.test import iris_resources as module
    calls=[]
    monkeypatch.setattr(module,'profile',lambda **kwargs:calls.append(kwargs) or {'schema_version':1})
    module.main(['--root',str(tmp_path),'--output',str(tmp_path/'independent.json')])
    assert calls[0]['bundle']==tmp_path/'.cache/iris-release-v1/diversity/release-export'
    assert calls[0]['wake']==tmp_path/'.cache/iris-release-v1/wake/export'
    assert calls[0]['owner']==tmp_path/'.cache/iris-release-v1/owner/export'
    assert calls[0]['vad_folder']==tmp_path/'.cache/iris-release-v1/inputs/vad'
    assert calls[0]['output']==tmp_path/'independent.json'
    assert calls[0]['threads']==1 and calls[0]['engine']=='ncnn'
    assert json.loads(capsys.readouterr().out)['reused'] is False


def test_shared_cache_mount_preserves_protocol_model_and_frozen_binding_paths(components):
    from scripts.test import iris_resources as module
    root=components['root']; cache=root/'.cache'
    shared=root.with_name(root.name+'-shared-cache')
    cache.rename(shared)
    cache.symlink_to(shared,target_is_directory=True)
    # profile() resolves these model folders before inventory() receives them.
    for key in ('bundle','wake','owner','vad_folder'):
        target=cache/'models'/key
        target.parent.mkdir(parents=True,exist_ok=True)
        components[key].rename(target)
        components[key]=target
    personal=cache/'iris-release-v1/inputs/base-views/personal-test.jsonl'
    assert module.relative(personal.resolve(),root)=='.cache/iris-release-v1/inputs/base-views/personal-test.jsonl'
    sibling=shared.with_name(shared.name+'-sibling')/'unrelated.onnx'
    assert module.relative(sibling,root)==str(sibling)
    result=module.profile(**components,engine='onnxruntime')
    assert all(row['path'].startswith('.cache/models/') for row in result['sizes']['files'])
    assert all(str(shared) not in name for name in result['bindings']['files'])
    assert result['source_context']['module_root']==str(module.ROOT)
    assert result['source_context']['input_root']==str(root)
    frozen=components['output'].read_bytes()
    assert module.profile(**components,engine='onnxruntime')==result
    assert components['output'].read_bytes()==frozen


def test_input_root_override_does_not_claim_unexecuted_shadow_source(tmp_path):
    from scripts.test import iris_resources as module
    shadow=tmp_path/'scripts/runtime/mvp_static.py'
    shadow.parent.mkdir(parents=True);shadow.write_text('raise RuntimeError("not executed")')
    config=tmp_path/'config.toml';config.write_text('[inference]\nthreads=1\n')
    paths=module.source_bindings(tmp_path)
    assert paths['scripts/runtime/mvp_static.py']==module.ROOT/'scripts/runtime/mvp_static.py'
    assert paths['configuration:config.toml']==config
