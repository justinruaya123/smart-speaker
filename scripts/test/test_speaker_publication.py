"""Shipping profiles, editable configuration and independent application gates."""
import json
import math
from pathlib import Path
import pytest


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


def test_slot_competition_and_alias_invariance():
    from scripts.runtime.speaker_confidence import slot_confidences
    a={'intent':'TIMER','slots':{'duration_seconds':10}}
    b={'intent':'TIMER','slots':{'duration_seconds':30}}
    entries=[{'command':a},{'command':b}]
    scores=[-1,-1-math.log(9)/10]
    assert slot_confidences(scores,entries,a,10)['duration_seconds']==pytest.approx(.9)
    assert slot_confidences(scores+[scores[0]],entries+[entries[0]],a,10)==slot_confidences(scores,entries,a,10)
    assert slot_confidences([-1,-1],entries,a,10)['duration_seconds']==pytest.approx(.5)


def test_demo_requires_intent_and_every_slot_preserves_frozen_output():
    from scripts.runtime.speaker_confidence import application_decision
    cfg={'mode':'demo','intent_threshold':.95,'slot_threshold':.9}
    command={'intent':'BRIGHTNESS','slots':{'light_id':'light.kitchen','percent':50}}
    record={'candidate':command,'intent_probability':.95,'intent_index':16,'score':-1,'margin':.5,'target_named_probability':.99,'slot_confidence':{'light_id':.9,'percent':.9}}
    event={'output':{'accepted':False,'command':None,'reason':'unqualified_model'},'diagnostic_candidate':record}
    assert application_decision(event,cfg)['accepted'] is True
    assert event['output']['accepted'] is False
    record['slot_confidence']['percent']=.899
    assert application_decision(event,cfg)['reason']=='slot_confidence'
    record['slot_confidence']['percent']=.99;record['intent_probability']=.949
    assert application_decision(event,cfg)['reason']=='intent_confidence'
    record['intent_probability']=.99;record['candidate']['slots']['percent']=101
    assert application_decision(event,cfg)['accepted'] is False


def test_qualified_and_capture_fault_cannot_be_promoted():
    from scripts.runtime.speaker_confidence import application_decision
    e={'output':{'accepted':False,'command':None,'reason':'capture_fault'},'diagnostic_candidate':None}
    assert not application_decision(e,{'mode':'demo','intent_threshold':.95,'slot_threshold':.9})['accepted']
    e['output']['reason']='unqualified_model'
    assert not application_decision(e,{'mode':'qualified','intent_threshold':.95,'slot_threshold':.9})['accepted']


def test_profile_inventory_no_ui_or_benchmark_leaks():
    from scripts.speaker_publish import application_files
    inference=application_files('inference','skip')
    benchmark=application_files('benchmark','only')
    full=application_files('full','skip')
    assert 'benchmark.py' not in inference and 'scripts/runtime/pcv_full_benchmark.py' not in inference
    assert not any(n.startswith('SpeakerUI/') for n in benchmark)
    assert 'benchmark.py' in full and 'guided_test.py' in benchmark
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
