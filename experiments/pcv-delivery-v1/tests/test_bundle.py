import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
sys.path.insert(0,str(ROOT/'experiments/pcv-synthetic-v2'))
from pcvcm.contract import file_hash,identity,write_json
from scripts.runtime.pcv_deploy.bundle import ASSETS,WAKE_FILES,verify_bundle,load_bundle


@pytest.fixture
def fixture_bundle(tmp_path):
    from vcm.model import CausalVCM,export_model
    root=tmp_path/'bundle';root.mkdir();(root/'wake').mkdir()
    model=CausalVCM()
    import torch
    torch.save({'weights':model.state_dict()},tmp_path/'fixture.pt')
    export_model(model,root/'model.onnx')
    for name in WAKE_FILES:shutil.copyfile(ROOT/'artifacts'/name,root/'wake'/name)
    write_json(root/'catalog.json',dict(entries=[dict(id='fixture',text='fixture',phones=[1,2,3],
        use_case='UC-03',command={'intent':'POWER_ON','entity_id':'light.kitchen'})]))
    write_json(root/'normalization.json',dict(mean=[0]*40,std=[1]*40))
    from dataclasses import asdict
    from pcvcm.endpoint import EndpointConfig
    write_json(root/'endpoint.json',asdict(EndpointConfig()))
    write_json(root/'settings.json',dict(sample_rate=16000,threads=1,wake_threshold=.5,release_ms=320,
        release_timeout_seconds=3.,command_timeout_seconds=5.))
    files={name:file_hash(root/name) for name in ASSETS};binding=identity(files)
    write_json(root/'policy.json',dict(binding=binding,enabled=False,thresholds={}))
    write_json(root/'deployment.json',dict(schema_version=1,files=files,binding=binding,
        policy_sha256=file_hash(root/'policy.json'),status='DIAGNOSTIC_UNQUALIFIED'))
    return root


def test_bundle_loads_cpu_only_without_study_or_corpus_and_cannot_accept(fixture_bundle):
    snapshot=load_bundle(fixture_bundle,allow_diagnostic=True)
    assert snapshot['gate'].command.encoder.session.get_providers()==['CPUExecutionProvider']
    gate=snapshot['gate']
    for session in (gate.command.encoder.session,gate.wake.mel,gate.wake.embedding,gate.wake.head,gate.voice.model):
        options=session.get_session_options()
        assert options.intra_op_num_threads==options.inter_op_num_threads==1
        assert options.get_session_config_entry('session.intra_op.allow_spinning')=='0'
        assert options.get_session_config_entry('session.inter_op.allow_spinning')=='0'
    events=[]
    for i in range(80):events+=snapshot['gate'].feed(np.zeros(512,dtype=np.float32),i*512)
    assert not any(e.get('accepted') for e in events)
    with pytest.raises(ValueError,match='Diagnostic'):verify_bundle(fixture_bundle)


def test_catalog_mutation_and_stale_policy_fail_closed(fixture_bundle):
    path=fixture_bundle/'catalog.json';original=path.read_bytes();path.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='identity'):verify_bundle(fixture_bundle,allow_diagnostic=True)
    path.write_bytes(original)
    policy=json.loads((fixture_bundle/'policy.json').read_text());policy['binding']='stale'
    write_json(fixture_bundle/'policy.json',policy)
    manifest=json.loads((fixture_bundle/'deployment.json').read_text());manifest['policy_sha256']=file_hash(fixture_bundle/'policy.json')
    write_json(fixture_bundle/'deployment.json',manifest)
    with pytest.raises(ValueError,match='Stale policy'):verify_bundle(fixture_bundle,allow_diagnostic=True)


def test_renaming_diagnostic_as_validated_does_not_bypass_measured_evidence(fixture_bundle):
    policy=json.loads((fixture_bundle/'policy.json').read_text());policy['enabled']=True
    write_json(fixture_bundle/'policy.json',policy)
    manifest=json.loads((fixture_bundle/'deployment.json').read_text())
    manifest.update(status='SYNTHETIC_VALIDATED',policy_sha256=file_hash(fixture_bundle/'policy.json'))
    write_json(fixture_bundle/'deployment.json',manifest)
    with pytest.raises(ValueError,match='Missing bound'):verify_bundle(fixture_bundle)


def qualify_fixture(root,*,inherited_wake_source=False):
    """Fabricated metadata exercises integrity checks, never speech accuracy."""
    from scripts.runtime.pcv_deploy.bundle import EVIDENCE
    version='pcv-synthetic-v2-repair1' if inherited_wake_source else 'pcv-synthetic-v2'
    protocol=json.loads((ROOT/'experiments'/version/'protocol.json').read_text())
    write_json(root/'source-protocol.json',protocol)
    write_json(root/'source-identity.json',{} if inherited_wake_source else
        {'experiments\\pcv-synthetic-v2\\wake_gate.py':file_hash(ROOT/'experiments/pcv-synthetic-v2/wake_gate.py')})
    source_files={n:file_hash(root/n) for n in ('model.onnx','catalog.json','normalization.json','endpoint.json')}
    source_files['protocol.json']=file_hash(root/'source-protocol.json')
    source_files['source-identity.json']=file_hash(root/'source-identity.json');binding=identity(source_files)
    source_policy=dict(binding=binding,enabled=True,thresholds={'global':[-4.,.5]})
    write_json(root/'source-policy.json',source_policy)
    write_json(root/'calibration-selection.json',dict(policy=source_policy,test_accessed=False))
    write_json(root/'source-manifest.json',dict(files=source_files,binding=binding,policy_sha256=file_hash(root/'source-policy.json')))
    write_json(root/'command-results.json',dict(binding=binding,complete=True,synthetic_gates_passed=True,predictions_sha256='fixture'))
    write_json(root/'wake-results.json',dict(binding=binding,complete=True,wake_gates_passed=True))
    write_json(root/'demo-results.json',dict(binding=binding,demo_gates_passed=True,predictions_sha256='fixture'))
    write_json(root/'source-finalization.json',dict(binding=binding,synthetic_qualified=True,
        command_results_sha256=file_hash(root/'command-results.json'),wake_results_sha256=file_hash(root/'wake-results.json')))
    policy=json.loads((root/'policy.json').read_text());policy.update(enabled=True,thresholds=source_policy['thresholds'])
    write_json(root/'policy.json',policy)
    manifest=json.loads((root/'deployment.json').read_text());manifest.update(status='SYNTHETIC_VALIDATED',
        source_binding=binding,policy_sha256=file_hash(root/'policy.json'),evidence_files={n:file_hash(root/n) for n in EVIDENCE})
    write_json(root/'deployment.json',manifest)
    verify_bundle(root)


def refresh_metadata(root):
    from scripts.runtime.pcv_deploy.bundle import EVIDENCE
    manifest=json.loads((root/'deployment.json').read_text());manifest['files']={n:file_hash(root/n) for n in ASSETS}
    manifest['binding']=identity(manifest['files'])
    policy=json.loads((root/'policy.json').read_text());policy['binding']=manifest['binding'];write_json(root/'policy.json',policy)
    manifest['policy_sha256']=file_hash(root/'policy.json');manifest['evidence_files']={n:file_hash(root/n) for n in EVIDENCE}
    write_json(root/'deployment.json',manifest)


def test_wake_substitution_cannot_inherit_passing_evidence(fixture_bundle):
    qualify_fixture(fixture_bundle)
    path=fixture_bundle/'wake/wake-hey-jarvis.onnx';path.write_bytes(path.read_bytes()+b' ')
    refresh_metadata(fixture_bundle)
    with pytest.raises(ValueError,match='Wake assets'):verify_bundle(fixture_bundle)


def test_policy_substitution_cannot_inherit_passing_evidence(fixture_bundle):
    qualify_fixture(fixture_bundle)
    for name in ('policy.json','source-policy.json'):
        policy=json.loads((fixture_bundle/name).read_text());policy['thresholds']['global'][0]=-100.
        write_json(fixture_bundle/name,policy)
    source=json.loads((fixture_bundle/'source-manifest.json').read_text());source['policy_sha256']=file_hash(fixture_bundle/'source-policy.json')
    write_json(fixture_bundle/'source-manifest.json',source);refresh_metadata(fixture_bundle)
    with pytest.raises(ValueError,match='pre-test selection'):verify_bundle(fixture_bundle)


def test_enabled_reload_rejects_a_bundle_measured_with_other_runtime(fixture_bundle):
    qualify_fixture(fixture_bundle)
    protocol=json.loads((fixture_bundle/'source-protocol.json').read_text())
    protocol['shared_sources']['vcm\\frontend.py']='another-runtime'
    write_json(fixture_bundle/'source-protocol.json',protocol)
    source=json.loads((fixture_bundle/'source-manifest.json').read_text())
    source['files']['protocol.json']=file_hash(fixture_bundle/'source-protocol.json')
    source['binding']=identity(source['files'])
    for name in ('source-policy.json','command-results.json','wake-results.json','demo-results.json','source-finalization.json'):
        value=json.loads((fixture_bundle/name).read_text());value['binding']=source['binding'];write_json(fixture_bundle/name,value)
    source['policy_sha256']=file_hash(fixture_bundle/'source-policy.json');write_json(fixture_bundle/'source-manifest.json',source)
    write_json(fixture_bundle/'calibration-selection.json',dict(test_accessed=False,policy=json.loads((fixture_bundle/'source-policy.json').read_text())))
    final=json.loads((fixture_bundle/'source-finalization.json').read_text())
    final.update(command_results_sha256=file_hash(fixture_bundle/'command-results.json'),wake_results_sha256=file_hash(fixture_bundle/'wake-results.json'))
    write_json(fixture_bundle/'source-finalization.json',final);refresh_metadata(fixture_bundle)
    verify_bundle(fixture_bundle)
    with pytest.raises(ValueError,match='Executing recognition source'):load_bundle(fixture_bundle)
