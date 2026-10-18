"""Standard-library bootstrap and integrity utilities shared by the three scripts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
UV_VERSION = '0.9.7'


def limit_math_threads():
    """Call before numerical imports in standalone inference worker processes."""
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'


def sha(path):
    # This runs before uv installs managed Python, including on Pi OS Python 3.9.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run_id():
    import uuid
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:6]


def fetch(url, path, expected=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected is None or sha(path) == expected:
            return path
        raise ValueError('Cached file checksum mismatch: ' + str(path))
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, stream)
        except BaseException:
            # Windows cannot unlink an open NamedTemporaryFile. Close first so
            # cleanup does not mask the original download exception.
            stream.close()
            temporary.unlink(missing_ok=True); raise
    try:
        if expected and sha(temporary) != expected:
            raise ValueError('Downloaded checksum mismatch: ' + url)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def ensure_uv(root=ROOT):
    available = shutil.which('uv')
    if available:
        return available
    binary = Path(root)/'.runtime/uv'/('uv.exe' if os.name == 'nt' else 'uv')
    if binary.exists():
        return str(binary)
    # A uv wheel contains a standalone binary; pip and a compiler are unnecessary.
    machine = platform.machine().lower()
    arch = {'amd64':'x86_64', 'x86_64':'x86_64', 'aarch64':'aarch64', 'arm64':'aarch64'}.get(machine)
    tags = {'Linux': 'manylinux_2_17_' + str(arch),
            'Darwin': 'macosx_11_0_arm64' if arch == 'aarch64' else 'macosx_10_12_x86_64',
            'Windows': 'win_arm64' if arch == 'aarch64' else 'win_amd64'}
    if not arch or platform.system() not in tags:
        raise RuntimeError('Use a supported 64-bit OS (Raspberry Pi OS 64-bit on Pi).')
    with urllib.request.urlopen(f'https://pypi.org/pypi/uv/{UV_VERSION}/json', timeout=30) as response:
        metadata = json.load(response)
    match = next((r for r in metadata['urls'] if tags[platform.system()] in r['filename']), None)
    if match is None:
        raise RuntimeError('No uv binary for this OS; install uv from https://docs.astral.sh/uv/.')
    wheel = fetch(match['url'], Path(root)/'.cache/tools'/match['filename'], match['digests']['sha256'])
    with zipfile.ZipFile(wheel) as archive:
        entry = next(n for n in archive.namelist() if n.endswith('/' + binary.name) and '/scripts/' in n)
        binary.parent.mkdir(parents=True, exist_ok=True)
        with binary.open('xb') as stream:
            stream.write(archive.read(entry))
    binary.chmod(0o755)
    return str(binary)


def python_path(root=ROOT):
    return Path(root)/'.venv'/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def bootstrap(*, group=None, root=ROOT):
    root = Path(root); python = python_path(root)
    project = root if (root/'pyproject.toml').exists() else root/'submission/environment'
    env = os.environ.copy(); env.pop('VIRTUAL_ENV', None)
    env['UV_PROJECT_ENVIRONMENT'] = str(root/'.venv')
    env['UV_CACHE_DIR'] = str(root/'.cache/uv')
    env['UV_PYTHON_INSTALL_DIR'] = str(root/'.runtime/python')
    env['UV_PYTHON_BIN_DIR'] = str(root/'.runtime/bin')
    command = [ensure_uv(root), 'sync', '--locked', '--inexact', '--project', str(project),
               '--python', str(python) if python.exists() else '3.13']
    if group:
        command += ['--group', group]
    print('Setting up locked ' + (group or 'inference') + ' environment…', flush=True)
    subprocess.run(command, check=True, env=env)
    return python


def reexec(script, argv, *, group=None):
    python = bootstrap(group=group)
    return subprocess.call([str(python), str(ROOT/script), *argv, '--no-setup'], cwd=ROOT)


def runtime_entries(*, model_choice='full'):
    """One stdlib-only allowlist for the shared inference/benchmark ZIP closure."""
    names = ['vcm/__init__.py', 'vcm/frontend.py', 'vcm/catalog.py',
             'scripts/runtime/pcv_final_runtime.py', 'scripts/runtime/pcv_schema.py',
             'scripts/runtime/speaker_confidence.py', 'scripts/runtime/speaker_compact.py',
             'scripts/runtime/wake_models.py', 'scripts/submission/__init__.py',
             'scripts/submission/common.py', 'scripts/submission/model.py']
    entries = {n:(ROOT/n).read_bytes() for n in names}
    for name in ('scripts/__init__.py', 'scripts/runtime/__init__.py'):
        entries[name] = (ROOT/name).read_bytes() if (ROOT/name).exists() else b''
    pcvcm = ROOT/'pcvcm'
    if not pcvcm.is_dir(): pcvcm = ROOT/'experiments/phrase-contract-v1/pcvcm'
    for name in ('__init__.py', 'contract.py', 'decoder.py'):
        entries['pcvcm/'+name] = (pcvcm/name).read_bytes()
    wake_gate = ROOT/'wake_gate.py'
    if not wake_gate.exists(): wake_gate = ROOT/'experiments/pcv-synthetic-v2/wake_gate.py'
    entries['wake_gate.py'] = wake_gate.read_bytes()
    environment = ROOT/'submission/environment'
    if not environment.is_dir(): environment = ROOT
    for name in ('pyproject.toml', 'uv.lock'):
        entries[name] = (environment/name).read_bytes()
    if model_choice not in ('full','compact'):
        raise ValueError('Model choice must be full or compact')
    model_root = ROOT/'submission/models'/model_choice
    if not (model_root/'model.onnx').is_file():
        raise FileNotFoundError('Selected model is absent: ' + str(model_root))
    for file in model_root.rglob('*'):
        if file.is_file(): entries[file.relative_to(ROOT).as_posix()] = file.read_bytes()
    for file in (ROOT/'submission/licenses').glob('*'):
        if file.is_file(): entries[file.relative_to(ROOT).as_posix()] = file.read_bytes()
    if (ROOT/'LICENSE').exists(): entries['LICENSE'] = (ROOT/'LICENSE').read_bytes()
    return entries


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name and path.parts and path.as_posix() == name and name != '.'
                and not path.is_absolute() and '..' not in path.parts
                and '\\' not in name and ':' not in name)


def write_archive(path, entries, **metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Preserve existing archive: ' + str(path))
    files = {name:hashlib.sha256(data).hexdigest() for name,data in entries.items()}
    if not all(safe_name(n) for n in files):
        raise ValueError('Unsafe archive path')
    manifest = json.dumps({'format':'smart-speaker-submission-v1', 'files':files, **metadata}, indent=2).encode()
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in entries.items():
            archive.writestr(name, data)
        archive.writestr('SUBMISSION-MANIFEST.json', manifest)
    return {'path':str(path), 'sha256':sha(path), 'bytes':path.stat().st_size, 'files':len(files)}


def extract_verified(path, destination):
    destination = Path(destination)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read('SUBMISSION-MANIFEST.json'))
        if len(names) != len(set(names)) or set(names) != {*manifest['files'], 'SUBMISSION-MANIFEST.json'}:
            raise ValueError('Archive inventory mismatch')
        for name in names:
            if not safe_name(name):
                raise ValueError('Unsafe archive member: ' + name)
        for name,expected in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Archive checksum mismatch: ' + name)
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)
    return manifest
