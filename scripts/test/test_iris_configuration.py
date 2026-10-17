import json
import pytest

def test_candidate_deployment_cannot_enable_actions_through_config_or_environment(tmp_path):
    from scripts.runtime.speaker_config import resolve_config
    (tmp_path/'deployment.json').write_text('{"actions_enabled": false}')
    (tmp_path/'config.toml').write_text('[actions]\nenabled=true\n')
    cfg,env=resolve_config(tmp_path,environ={'ACTIONS_ENABLED':'true'})
    assert cfg['actions']['enabled'] is False
    assert env['ACTIONS_ENABLED']=='false'



def test_configuration_defaults_and_precedence(tmp_path):
    from scripts.runtime.speaker_config import resolve_config
    (tmp_path/'config.toml').write_text('[inference]\nintent_threshold=0.96\nslot_threshold=0.91\n[server]\nport=3010\n')
    (tmp_path/'.env').write_text('INTENT_THRESHOLD=0.97\nHA_TOKEN="private-token"\nPORT=3011\n')
    cfg, env = resolve_config(tmp_path, environ={'INTENT_THRESHOLD':'0.98'}, overrides={'port':3012})
    assert cfg['inference']['intent_threshold']==.98
    assert cfg['inference']['slot_threshold']==.91
    assert cfg['actions']['enabled'] is True
    assert cfg['server']['port']==3012
    assert env['HA_TOKEN']=='private-token'
    assert 'private-token' not in json.dumps(cfg)
    defaults,_=resolve_config(tmp_path/'fresh',environ={})
    assert defaults['inference']['intent_threshold']==.95
    assert defaults['inference']['slot_threshold']==.90



@pytest.mark.parametrize('text',['[inference]\nslot_threshold=90','[actions]\nenabled="false"','[audio]\nmicrophone=-2','[inference]\nunknown=3'])
def test_invalid_config_fails_with_field(tmp_path,text):
    from scripts.runtime.speaker_config import resolve_config
    (tmp_path/'config.toml').write_text(text)
    with pytest.raises(ValueError):resolve_config(tmp_path,environ={})



def test_profile_inventory_no_ui_or_benchmark_leaks():
    from scripts.speaker_publish import application_files
    inference=application_files('inference','skip')
    benchmark=application_files('benchmark','only')
    full=application_files('full','skip')
    assert 'benchmark.py' not in inference and 'scripts/runtime/pcv_full_benchmark.py' not in inference
    assert not any(n.startswith('SpeakerUI/') for n in benchmark)
    assert 'benchmark.py' in full and 'scripts/runtime/iris_pi_benchmark.py' in benchmark
    for files in (inference,benchmark,full):
        assert '.env.example' in files and 'config.example.toml' in files
        assert '.env' not in files and 'config.toml' not in files



def test_service_transport_uses_bind_address_tls_and_browser_origin():
    from types import SimpleNamespace
    from scripts.runtime.speaker_start import service_connection
    args=SimpleNamespace(port=3443,settings={'server':{'host':'::','origin':'https://speaker.example'}},runtime_env={'TLS_CERT_PATH':'cert.pem'})
    assert service_connection(args)=={'url':'https://[::1]:3443','origin':'https://speaker.example','tls_cert':'cert.pem'}
    args.settings['server']['host']='192.168.1.8';args.runtime_env={}
    assert service_connection(args)['url']=='http://192.168.1.8:3443'



def test_environment_can_override_bindings_without_exposing_secrets(tmp_path):
    from scripts.runtime.speaker_config import resolve_config
    cfg,env=resolve_config(tmp_path,environ={'SMART_SPEAKER_BINDINGS':'{"lights":{"light.desk":"light.real"}}','HA_TOKEN':'private'})
    assert cfg['bindings']['lights']=={'light.desk':'light.real'}
    assert 'private' not in str(cfg)
    assert env['HA_TOKEN']=='private'



def test_launcher_does_not_serve_package_from_unrelated_release(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.runtime import speaker_start as launch
    import hashlib
    folder=tmp_path/'experiments/iris-release-v1';folder.mkdir(parents=True)
    release=folder/'reproduction-release.json';release.write_text('{"bundle":"current"}')
    stale=tmp_path/'stale.zip';stale.write_bytes(b'old but hash-verified')
    (folder/'reproduction-run-z-stale.json').write_text(json.dumps({'status':'complete',
        'release':'experiments/iris-release-v1/other-release.json','release_sha256':'0'*64,
        'package':{'path':'stale.zip','sha256':hashlib.sha256(stale.read_bytes()).hexdigest()}}))
    seen=[]
    monkeypatch.setattr(launch,'extract_local',lambda root,path:seen.append(path))
    def rebuild(*args,**kwargs):raise RuntimeError('correctly requested current release rebuild')
    monkeypatch.setattr(launch,'run',rebuild)
    with pytest.raises(RuntimeError,match='current release rebuild'):
        launch.local_package(tmp_path,SimpleNamespace())
    assert seen==[]
