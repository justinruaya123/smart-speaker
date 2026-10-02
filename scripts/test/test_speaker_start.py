"""Launcher contracts: no accidental training/server, unsafe updates never apply."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_start_help_is_available_without_dependencies():
    result = subprocess.run([sys.executable, '-S', str(ROOT/'start.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--benchmark-only' in result.stdout
    assert '--skip-train' in result.stdout
    assert '--wav' in result.stdout


@pytest.mark.parametrize('flags',[['--benchmark-only'],['--benchmarks','only']])
def test_benchmark_only_does_not_plan_training_ui_or_server(flags):
    from scripts.runtime.speaker_start import make_parser, resolve_mode
    args = make_parser(False).parse_args(flags)
    assert resolve_mode(args, deployed=False, interactive=False) == ('benchmark', 'only')


def test_publish_only_is_training_free():
    from scripts.runtime.speaker_start import make_parser, resolve_mode
    args = make_parser(False).parse_args(['--skip-train', '--benchmarks', 'skip'])
    assert resolve_mode(args, deployed=False, interactive=False) == ('publish', 'skip')


def test_receipt_paths_from_windows_are_portable(tmp_path):
    from scripts.pcv_release import resolved
    assert resolved(tmp_path,'cache\\models\\selected.onnx')==(tmp_path/'cache/models/selected.onnx').resolve()


def archive(path, files, **metadata):
    manifest = {'format':'smart-speaker-v1', 'files':{n:hashlib.sha256(b).hexdigest() for n,b in files.items()}, **metadata}
    with zipfile.ZipFile(path,'w') as z:
        for n,b in files.items(): z.writestr(n,b)
        z.writestr('MANIFEST.json',json.dumps(manifest))
    return path


@pytest.mark.parametrize('name',['../escape','/absolute','C:/outside','a\\b','a/../b','CON.txt','a:stream','file.'])
def test_archive_rejects_unsafe_names(tmp_path,name):
    from scripts.runtime.speaker_package import inspect_archive
    with pytest.raises(ValueError): inspect_archive(archive(tmp_path/'bad.zip',{name:b'no'}))
    assert not (tmp_path.parent/'escape').exists()


def test_archive_rejects_corruption_and_case_collisions(tmp_path):
    from scripts.runtime.speaker_package import inspect_archive
    with pytest.raises(ValueError): inspect_archive(archive(tmp_path/'bad.zip',{'Model':b'a','model':b'b'}))
    p=archive(tmp_path/'corrupt.zip',{'model.onnx':b'a'})
    with pytest.warns(UserWarning,match='Duplicate name'):
        with zipfile.ZipFile(p,'a') as z:z.writestr('model.onnx',b'b')
    with pytest.raises(ValueError):inspect_archive(p)


def test_update_keeps_previous_pointer_when_load_check_fails(tmp_path):
    from scripts.runtime.speaker_package import apply_update
    files={'models/default/model.onnx':b'candidate'}
    p=archive(tmp_path/'new.zip',files,target='pi',runtime_id='runtime-a',startup_action='install-model',bundle_id='candidate')
    (tmp_path/'state').mkdir()
    previous={'bundle':'models/default','applied_zip':None}
    (tmp_path/'state/install.json').write_text(json.dumps(previous))
    def fail(_):raise ValueError('bad graph')
    with pytest.raises(ValueError):apply_update(tmp_path,p,{'runtime_id':'runtime-a'},fail)
    assert json.loads((tmp_path/'state/install.json').read_text()) == previous


def test_update_is_complete_idempotent_and_does_not_touch_ui(tmp_path):
    from scripts.runtime.speaker_package import apply_update
    files={'models/default/model.onnx':b'candidate','models/default/catalog.json':b'{}','SpeakerUI/dist/index.html':b'new ui'}
    p=archive(tmp_path/'new.zip',files,target='pi',runtime_id='runtime-a',startup_action='install-model',bundle_id='candidate')
    (tmp_path/'SpeakerUI/dist').mkdir(parents=True)
    page=tmp_path/'SpeakerUI/dist/index.html';page.write_bytes(b'old ui')
    applied=apply_update(tmp_path,p,{'runtime_id':'runtime-a'},lambda folder: (folder/'catalog.json').read_bytes())
    assert (tmp_path/applied['bundle']/'model.onnx').read_bytes()==b'candidate'
    assert page.read_bytes()==b'old ui'
    assert apply_update(tmp_path,p,{'runtime_id':'runtime-a'},lambda _:pytest.fail('loaded twice'))==applied


def test_update_rejects_incompatible_runtime_before_load(tmp_path):
    from scripts.runtime.speaker_package import apply_update
    p=archive(tmp_path/'new.zip',{'models/default/model.onnx':b'x'},target='pi',runtime_id='different',startup_action='install-model')
    with pytest.raises(ValueError,match='contract changed'):
        apply_update(tmp_path,p,{'runtime_id':'current'},lambda _:pytest.fail('loaded incompatible model'))
    assert not (tmp_path/'state/install.json').exists()


def test_single_instance_lock_releases_after_close(tmp_path):
    from scripts.runtime.speaker_start import InstallLock
    first=InstallLock(tmp_path);second=InstallLock(tmp_path)
    try:
        assert first.acquire();assert not second.acquire()
        first.close();assert second.acquire()
    finally:first.close();second.close()


def test_export_assets_have_no_trainers_private_env_or_native_pc_modules():
    from scripts.speaker_publish import application_files
    files=application_files('standalone','only')
    assert not any('SpeakerUI/' in n or '/train/' in n or '.venv/' in n or 'node_modules/' in n for n in files)
    metadata=json.loads(files['deployment.json'])
    assert metadata['default_mode']=='benchmark' and metadata['actions_enabled'] is False
    assert 'uv.lock' in files and 'start.py' in files


def test_release_only_reuses_selection_without_rendering_or_legacy_packaging(tmp_path):
    from scripts.pcv_release import run_workflow
    from scripts.test.test_pcv_release_workflow import fixture,ExternalStages
    fixture(tmp_path);stages=ExternalStages(tmp_path)
    result=run_workflow(root=tmp_path,skip_train=True,release_only=True,runner=stages,
        packager=lambda *a,**k:pytest.fail('Legacy package path must not run'))
    assert result['status']=='diagnostic_complete'
    assert [s['name'] for s in result['stages']]==['verify-frozen-selection','software-tests','qualification-or-locked-replay','verify-release']

@pytest.mark.parametrize('flags',[['--wav','command.wav'],['--list-mics']])
def test_benchmark_profile_still_supports_explicit_inference(flags):
    from scripts.runtime.speaker_start import make_parser,resolve_mode
    args=make_parser(True).parse_args(flags)
    args.benchmarks='only'
    assert resolve_mode(args,deployed=True,interactive=False)==('infer','skip')
