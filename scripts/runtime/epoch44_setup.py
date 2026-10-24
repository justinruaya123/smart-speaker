"""Small epoch44 environments; legacy/research locks remain untouched."""
import hashlib
import json
import os
from pathlib import Path

from scripts.runtime.speaker_tools import find_uv, python_path, run

EXTRAS = frozenset(('audio', 'benchmark'))


def sync_epoch44(root, *, extras=()):
    root = Path(root).resolve()
    if not isinstance(extras, (tuple, list, set, frozenset)) or any(x not in EXTRAS for x in extras):
        raise ValueError('Epoch44 extras must contain only audio and/or benchmark')
    requested = set(extras)
    project = root/'scripts/runtime/epoch44'
    if not all((project/name).is_file() for name in ('pyproject.toml', 'uv.lock')):
        raise RuntimeError('Missing minimal epoch44 pyproject.toml / uv.lock; update the main checkout')
    identity = hashlib.sha256((project/'uv.lock').read_bytes() + (project/'pyproject.toml').read_bytes()).hexdigest()
    executable = python_path(root)
    stamp = root/'.runtime/python-epoch44.json'
    pending = root/'.runtime/python-epoch44-pending.json'
    previous = set()
    try:
        saved = json.loads(stamp.read_bytes())
        features = saved.get('extras')
        if (not pending.is_file() and saved.get('lock_sha256') == identity
                and isinstance(features, list) and all(x in EXTRAS for x in features)):
            previous = set(features)
            if executable.is_file() and requested <= previous:
                return executable
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    if not executable.is_file():
        previous.clear()
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)
    env['UV_PROJECT_ENVIRONMENT'] = str(root/'.venv')
    env['UV_CACHE_DIR'] = str(root/'.cache/uv')
    command = [find_uv(root), 'sync', '--locked', '--no-dev', '--no-build', '--inexact',
               '--project', project, '--python', executable if executable.is_file() else '3.12']
    for extra in sorted(requested):
        command += ['--extra', extra]
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_text('{}\n', encoding='utf8')
    stamp.unlink(missing_ok=True)
    run(command, cwd=root, env=env)
    if not executable.is_file():
        raise RuntimeError('Epoch44 dependency setup did not produce its local interpreter')
    stamp.write_text(json.dumps(dict(lock_sha256=identity, extras=sorted(previous | requested))), encoding='utf8')
    pending.unlink()
    return executable
