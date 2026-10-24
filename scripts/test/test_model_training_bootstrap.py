"""Training bootstrap remains pinned without any deployment source directory."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '94c0021d7e22c9aa719246ac97105d7140a9f2a2'


def test_training_environment_is_included_in_new_model_source_closure():
    from scripts.repository_boundary import model_source_files
    assert model_source_files(['scripts/runtime/pyproject.toml', 'scripts/runtime/uv.lock']) == [
        'scripts/train/pyproject.toml', 'scripts/train/uv.lock']


@pytest.mark.parametrize('name', ['pyproject.toml', 'uv.lock'])
def test_training_environment_preserves_original_lock_bytes(name):
    target = ROOT / 'scripts/train' / name
    original = subprocess.check_output(['git', 'show', BASELINE + ':scripts/runtime/' + name], cwd=ROOT)
    assert target.is_file(), 'Training must own its pinned verification environment'
    assert target.read_bytes() == original


@pytest.mark.parametrize('source', [True, False])
def test_bootstrap_uses_only_its_own_project_without_deployment(tmp_path, monkeypatch, source):
    from scripts import training_tools as tools
    project = tmp_path / 'scripts/train' if source else tmp_path
    project.mkdir(parents=True, exist_ok=True)
    (project / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.0.0"\n')
    (project / 'uv.lock').write_text('fixture lock\n')
    calls = []
    monkeypatch.setattr(tools, 'find_uv', lambda root: 'fixture-uv')
    monkeypatch.setattr(tools, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    result = tools.sync_python(tmp_path, source=source)
    assert len(calls) == 1
    command, options = calls[0]
    assert command[command.index('--project') + 1] == project
    assert '--locked' in command and '--no-build' in command and '--no-dev' in command
    assert ('--inexact' in command) is source
    assert options['cwd'] == tmp_path
    assert options['env']['UV_PROJECT_ENVIRONMENT'] == str(tmp_path / '.venv')
    assert options['env']['UV_CACHE_DIR'] == str(tmp_path / '.cache/uv')
    assert result == tmp_path / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    assert not (tmp_path / 'scripts/runtime').exists()
