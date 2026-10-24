"""Per-use epoch44 bootstrap; no training, devices or real installs in fixtures."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('arguments,extras', [
    (['--check'], ()),
    (['--wav', __file__], ()),
    (['--list-devices'], ('audio',)),
])
def test_inference_selects_only_its_epoch44_dependencies(monkeypatch, arguments, extras):
    from scripts import iris_inference as entry
    from scripts.runtime import speaker_tools
    calls = []
    def setup(root, *, extras=()):
        calls.append((root, extras))
        return Path(sys.executable)
    monkeypatch.setitem(sys.modules, 'scripts.runtime.epoch44_setup', SimpleNamespace(sync_epoch44=setup))
    monkeypatch.setattr(speaker_tools, 'sync_python', lambda *a, **k: pytest.fail(
        'ONNX-only epoch44 must not install the legacy ncnn/OpenCV runtime'))
    monkeypatch.setattr(entry, 'resolve_settings', lambda *a, **k: ({'actions': {'enabled': False}}, {}))
    monkeypatch.setattr(subprocess, 'call', lambda *a, **k: 17)
    assert entry.main(arguments, root=ROOT) == 17
    assert calls == [(ROOT, extras)]


@pytest.mark.parametrize('arguments', [
    ['--list-devices', '--wav', __file__],
    ['--wav', __file__, '--list-devices', '--no-setup'],
])
def test_inference_rejects_mixed_device_listing_and_wav(monkeypatch, capsys, arguments):
    from scripts import iris_inference as entry
    calls = []
    def setup(root, *, extras=()):
        calls.append(('setup', extras))
        return Path(sys.executable)
    monkeypatch.setitem(sys.modules, 'scripts.runtime.epoch44_setup', SimpleNamespace(sync_epoch44=setup))
    monkeypatch.setattr(entry, 'resolve_settings', lambda *a, **k: ({'actions': {'enabled': False}}, {}))
    monkeypatch.setattr(subprocess, 'call', lambda command, **kwargs: calls.append(command) or 17)
    with pytest.raises(SystemExit) as stopped:
        entry.main(arguments, root=ROOT)
    assert stopped.value.code == 2
    assert '--list-devices cannot be combined with --wav' in capsys.readouterr().err
    assert calls == [], 'Conflicting operations must be rejected before setup or worker launch'


def test_minimal_setup_reuses_a_superset_without_touching_legacy_lock(tmp_path, monkeypatch):
    from scripts.runtime import epoch44_setup as setup
    project = tmp_path/'scripts/runtime/epoch44'
    project.mkdir(parents=True)
    (project/'pyproject.toml').write_bytes(b'fixture-project')
    (project/'uv.lock').write_bytes(b'fixture-lock')
    legacy = tmp_path/'scripts/runtime/uv.lock'
    legacy.write_bytes(b'legacy-must-stay-frozen')
    executable = tmp_path/'.venv'/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    calls = []
    def execute(command, *, cwd, env):
        calls.append((command, cwd, env))
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b'software-install-spy')
    monkeypatch.setattr(setup, 'find_uv', lambda root: 'uv-spy')
    monkeypatch.setattr(setup, 'run', execute)
    monkeypatch.setenv('VIRTUAL_ENV', 'unrelated-environment')
    assert setup.sync_epoch44(tmp_path, extras=('benchmark',)) == executable
    assert setup.sync_epoch44(tmp_path) == executable
    assert len(calls) == 1
    assert '--no-build' in calls[0][0] and '--inexact' in calls[0][0]
    assert calls[0][0][calls[0][0].index('--project')+1] == project
    assert calls[0][0][-2:] == ['--extra', 'benchmark']
    assert calls[0][2]['UV_PROJECT_ENVIRONMENT'] == str(tmp_path/'.venv')
    assert 'VIRTUAL_ENV' not in calls[0][2]
    assert setup.sync_epoch44(tmp_path, extras=('audio',)) == executable
    assert setup.sync_epoch44(tmp_path, extras=('benchmark',)) == executable
    assert len(calls) == 2
    assert legacy.read_bytes() == b'legacy-must-stay-frozen'


def test_removed_environment_does_not_reuse_stale_feature_claims(tmp_path, monkeypatch):
    from scripts.runtime import epoch44_setup as setup
    import json
    project = tmp_path/'scripts/runtime/epoch44'
    project.mkdir(parents=True)
    (project/'pyproject.toml').write_bytes(b'fixture-project')
    (project/'uv.lock').write_bytes(b'fixture-lock')
    executable = setup.python_path(tmp_path)
    def execute(*args, **kwargs):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b'software-install-spy')
    monkeypatch.setattr(setup, 'find_uv', lambda root: 'uv-spy')
    monkeypatch.setattr(setup, 'run', execute)
    setup.sync_epoch44(tmp_path, extras=('benchmark',))
    executable.unlink()
    setup.sync_epoch44(tmp_path, extras=('audio',))
    saved = json.loads((tmp_path/'.runtime/python-epoch44.json').read_bytes())
    assert saved['extras'] == ['audio']


def test_pending_runtime_install_does_not_reuse_ready_stamp(tmp_path, monkeypatch):
    import json
    from scripts.runtime import epoch44_setup as setup
    project = tmp_path/'scripts/runtime/epoch44'
    project.mkdir(parents=True)
    (project/'pyproject.toml').write_bytes(b'fixture-project')
    (project/'uv.lock').write_bytes(b'fixture-lock')
    executable = setup.python_path(tmp_path)
    ready = tmp_path/'.runtime/python-epoch44.json'
    pending = tmp_path/'.runtime/python-epoch44-pending.json'
    calls = []
    install_fails = False
    def execute(command, **kwargs):
        calls.append(command)
        assert pending.is_file(), 'Runtime ownership must precede package mutation'
        assert not ready.exists(), 'An installation in progress must not claim dependencies are ready'
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes(b'software-install-spy')
        if install_fails:
            raise RuntimeError('partial-runtime-install-failure')
    monkeypatch.setattr(setup, 'find_uv', lambda root: 'uv-stop-spy')
    monkeypatch.setattr(setup, 'run', execute)
    setup.sync_epoch44(tmp_path, extras=('benchmark',))
    assert ready.is_file() and not pending.exists()
    pending.write_text('{}\n')
    install_fails = True
    with pytest.raises(RuntimeError, match='partial-runtime-install-failure'):
        setup.sync_epoch44(tmp_path, extras=('benchmark',))
    assert len(calls) == 2, 'Pending installation must invalidate a matching ready stamp'
    assert pending.is_file() and not ready.exists()
    install_fails = False
    setup.sync_epoch44(tmp_path)
    assert json.loads(ready.read_bytes())['extras'] == []
    assert not pending.exists()
    setup.sync_epoch44(tmp_path, extras=('benchmark',))
    assert len(calls) == 4, 'A partial install must not preserve unverified feature claims'


@pytest.mark.parametrize('install_fails', [False, True])
def test_training_prepares_a_managed_runtime_but_invalidates_its_cache(tmp_path, monkeypatch, install_fails):
    import json
    from scripts import iris_release as release
    from scripts.runtime import speaker_tools
    executable = speaker_tools.python_path(tmp_path)
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'software-existing-runtime-spy')
    marker = tmp_path/'.runtime/python-epoch44.json'
    marker.parent.mkdir()
    marker.write_text('{}')
    environment = tmp_path/release.HERE/'reproduction-environment.json'
    environment.parent.mkdir(parents=True)
    environment.write_text(json.dumps(dict(bootstrap_profiles=dict(cpu=['fixture==1']), packages=dict(torch='1.0+cpu'))))
    (tmp_path/'requirements.lock').write_text('# software bootstrap fixture\n')
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if install_fails:
            raise RuntimeError('original-install-failure')
    monkeypatch.setattr(speaker_tools, 'find_uv', lambda root: 'uv-stop-spy')
    if install_fails:
        with pytest.raises(RuntimeError, match='original-install-failure'):
            release.ensure_source_environment(tmp_path, runner=execute)
    else:
        assert release.ensure_source_environment(tmp_path, runner=execute) == executable
        assert calls, 'A managed inference environment still needs the research dependencies'
        receipt = json.loads((tmp_path/release.CACHE/'environment/bootstrap.json').read_bytes())
        assert receipt['existing_environment_modified'] is True
    assert calls and all(command[1] == 'pip' for command in calls)
    assert '--require-hashes' in calls[0]
    assert not marker.exists()
    assert (tmp_path/'.runtime/python-training-pending.json').exists() is install_fails
    if install_fails:
        failed_command = calls[0]
        install_fails = False
        assert release.ensure_source_environment(tmp_path, runner=execute) == executable
        assert len(calls) == 4 and calls[1] == failed_command
        assert not marker.exists() and not (tmp_path/'.runtime/python-training-pending.json').exists()
        receipt = json.loads((tmp_path/release.CACHE/'environment/bootstrap.json').read_bytes())
        assert receipt['existing_environment_modified'] is True


def test_training_prepares_interpreter_left_by_failed_runtime_install(tmp_path, monkeypatch):
    import json
    from scripts import iris_release as release
    from scripts.runtime import epoch44_setup as setup, speaker_tools
    project = tmp_path/'scripts/runtime/epoch44'
    project.mkdir(parents=True)
    (project/'pyproject.toml').write_bytes(b'fixture-project')
    (project/'uv.lock').write_bytes(b'fixture-lock')
    executable = setup.python_path(tmp_path)
    runtime_pending = tmp_path/'.runtime/python-epoch44-pending.json'
    training_pending = tmp_path/'.runtime/python-training-pending.json'
    ready = tmp_path/'.runtime/python-epoch44.json'
    observed = []
    def failed_sync(*args, **kwargs):
        observed.append(runtime_pending.is_file())
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b'interpreter-from-failed-install-spy')
        raise RuntimeError('original-runtime-install-failure')
    monkeypatch.setattr(setup, 'find_uv', lambda root: 'uv-stop-spy')
    monkeypatch.setattr(setup, 'run', failed_sync)
    with pytest.raises(RuntimeError, match='original-runtime-install-failure'):
        setup.sync_epoch44(tmp_path)
    assert executable.is_file() and not ready.exists()
    environment = tmp_path/release.HERE/'reproduction-environment.json'
    environment.parent.mkdir(parents=True)
    environment.write_text(json.dumps(dict(bootstrap_profiles=dict(cpu=['fixture==1']), packages=dict(torch='1.0+cpu'))))
    (tmp_path/'requirements.lock').write_text('# software bootstrap fixture\n')
    calls = []
    def research_install(command, **kwargs):
        assert training_pending.is_file(), 'Research must claim ownership before mutating packages'
        calls.append(command)
    monkeypatch.setattr(speaker_tools, 'find_uv', lambda root: 'uv-stop-spy')
    assert release.ensure_source_environment(tmp_path, runner=research_install) == executable
    assert len(calls) == 3, 'A failed runtime install must not be mistaken for unmanaged research'
    assert observed == [True], 'Runtime ownership must precede the uv mutation'
    assert all(command[1:3] == ['pip', 'install'] for command in calls)
    assert '--require-hashes' in calls[0] and '--no-deps' in calls[-1]
    assert not ready.exists() and not runtime_pending.exists() and not training_pending.exists()
