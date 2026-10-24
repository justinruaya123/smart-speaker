"""Public inference always selects epoch44 and keeps replay/check/CLI silent."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]


def test_actual_public_check_loads_selected_model_and_companions_without_setup():
    result = subprocess.run([sys.executable,'-B',str(ROOT/'inference.py'),'--check','--no-setup'],
                            cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    event = json.loads(result.stdout)
    assert event['parameters'] == 258912 and event['selected_epoch'] == 44
    assert event['actions_executed'] is False and event['owner_gating_enabled'] is True


def test_wav_is_silent_nonactuating_even_with_inherited_playback_flags(tmp_path):
    path = tmp_path/'silent.wav'
    sf.write(path,np.zeros(16000),16000,subtype='PCM_16')
    env = {**os.environ,'IRIS_FEEDBACK_ENABLED':'true','ACTIONS_ENABLED':'true','IRIS_SPEAKER_INDEX':'0'}
    result = subprocess.run([sys.executable,'-B',str(ROOT/'inference.py'),'--wav',str(path),'--no-setup'],
                            cwd=ROOT,capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines()]
    final = next(e for e in events if e['event']=='command')
    assert final['model_sha256'] == 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'
    assert final['source'] == 'wav' and final['actions_enabled'] is False
    assert final['output']['reason'] == 'capture_fault'
    assert not any(e.get('event') == 'feedback' and e.get('played') for e in events)


def test_ui_defaults_actions_off_and_uses_combined_gateway_environment(tmp_path):
    from scripts.iris_inference import make_parser, launch_environment, resolve_settings
    args = make_parser(tmp_path).parse_args(['--mode','ui','--speaker','7','--no-browser','--no-setup'])
    settings, env = resolve_settings(tmp_path,args,environ={})
    assert settings['actions']['enabled'] is False
    result = launch_environment(tmp_path,Path(sys.executable),args,settings,env,token='test-token')
    assert result['SMART_SPEAKER_BUNDLE'] == str(tmp_path/'submission/models/optionb-tiny')
    assert result['INFERENCE_ENGINE'] == 'onnxruntime' and result['SMART_SPEAKER_THREADS'] == '1'
    assert result['IRIS_FEEDBACK_ENABLED'] == 'true' and result['IRIS_SPEAKER_INDEX'] == '7'
    assert result['IRIS_PUBLICATION'] == 'true' and result['SERVE_UI'] == 'true'
    assert result['RECOGNITION_PROFILE'] == 'owner-protected'
    assert result['ACTIONS_ENABLED'] == 'false'


def test_private_explicit_actions_are_honored_but_no_audio_overrides(tmp_path):
    from scripts.iris_inference import make_parser, resolve_settings, launch_environment
    (tmp_path/'config.toml').write_text('[actions]\nenabled=true\n',encoding='utf8')
    args = make_parser(tmp_path).parse_args(['--mode','ui','--no-audio'])
    settings, env = resolve_settings(tmp_path,args,environ={})
    assert settings['actions']['enabled'] is False
    result = launch_environment(tmp_path,Path(sys.executable),args,settings,env,token='test')
    assert result['IRIS_FEEDBACK_ENABLED'] == 'false' and result['ACTIONS_ENABLED'] == 'false'
    args.no_audio = False
    assert resolve_settings(tmp_path,args,environ={})[0]['actions']['enabled'] is True


def test_readiness_cannot_approve_the_legacy_model():
    from scripts.iris_inference import ready_matches
    good = dict(instance='token',inference='onnx',runtime={'loaded':True,'parameters':258912,
        'model_sha256':'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce','selected_epoch':44})
    assert ready_matches(good,'token') is True
    assert ready_matches({**good,'runtime':{**good['runtime'],'parameters':969928}},'token') is False
    assert ready_matches({**good,'runtime':{**good['runtime'],'model_sha256':'bad'}},'token') is False
    assert ready_matches(good,'other-instance') is False


def test_wake_root_is_an_explicit_companion_override(tmp_path):
    from scripts.iris_inference import make_parser, launch_environment, resolve_settings
    args = make_parser(tmp_path).parse_args(['--mode','ui','--wake-root',str(tmp_path/'wake')])
    settings, env = resolve_settings(tmp_path,args,environ={})
    output = launch_environment(tmp_path,Path(sys.executable),args,settings,env,token='test')
    assert output['IRIS_WAKE_ROOT'] == str((tmp_path/'wake').resolve())


def test_public_inference_cannot_report_check_success_for_legacy_bundle():
    result=subprocess.run([sys.executable,'-B',str(ROOT/'inference.py'),'--check','--no-setup',
        '--bundle',str(ROOT/'artifacts/iris')],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert 'epoch44' in result.stderr and 'start.py' in result.stderr
    assert not result.stdout
