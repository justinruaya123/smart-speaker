"""Local, pinned tool bootstrap. Never modify the user's PATH or global Node."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

NODE_VERSION = '24.13.0'
PNPM_VERSION = '12.5.1'
UV_VERSION = '0.12.5'
# Official PyPI wheel URLs and SHA-256 from https://pypi.org/pypi/uv/0.12.5/json.
# Keep these beside the packaged bootstrap; no shell installer or pip is needed.
UV_WHEELS = {
    ('Linux', 'x64'): (
        'https://files.pythonhosted.org/packages/93/22/dacc9a0bc8604187a1ba954a3aef8329e4104eb0af772d2c3c634893bd9b/uv-0.12.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        '3e195ccf1ed60c8bb24a6447ce306441a4181d54b602407e09bc56e963911c15'),
    ('Linux', 'arm64'): (
        'https://files.pythonhosted.org/packages/7d/13/07585043c10e648820bf826474dac46864ce6691da5dc52fee43c5c7523a/uv-0.12.5-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl',
        '2d65b7b3bc3fd28678f62aa7fb5d90f106ad9782c1354af60b6cecdf9ea9ecd9'),
    ('Darwin', 'x64'): (
        'https://files.pythonhosted.org/packages/bd/ec/d76387b388fa21620088b89b9c67f2596a707add585104e0cb5e8abf55f2/uv-0.12.5-py3-none-macosx_10_12_x86_64.whl',
        '1a06c8bc4d43b5f6c1e3f2ae3d0f6455b07515f762516f95e52e6c0cbccedf15'),
    ('Darwin', 'arm64'): (
        'https://files.pythonhosted.org/packages/6d/bc/81ab953b7261ae6be40874b1f283a10873871e02eb353d354614dd8da96b/uv-0.12.5-py3-none-macosx_11_0_arm64.whl',
        'd87156bc174d94fae890bb7a261e2867140abb9fe1e9de81a5295e582fb9d0f5'),
    ('Windows', 'x64'): (
        'https://files.pythonhosted.org/packages/ce/f6/a9af9311c7f5640ca2bfcfdedb7aca37fa6d1d9f5c981fb50c5be02b7477/uv-0.12.5-py3-none-win_amd64.whl',
        '455c3e57602e2141e66e2f0bf685898c9c5e5a70377d14c9a71554a3baf3ddbf'),
    ('Windows', 'arm64'): (
        'https://files.pythonhosted.org/packages/bc/fb/e1266399f755f97a0783de379f2fed6dae0a2a240db32fe5a2eb976fec8a/uv-0.12.5-py3-none-win_arm64.whl',
        'bea86f27a027e0e3af908db4bdd4f1ceef3ca2bd47673b5ccca7f550e325b1b4'),
}


def run(command, **kwargs):
    print('>', ' '.join(map(str,command)), flush=True)
    return subprocess.run(list(map(str,command)), check=True, **kwargs)


def find_uv(root):
    """Use installed uv, or extract its checksum-pinned PyPI wheel locally."""
    available = shutil.which('uv')
    if available:
        return available
    root = Path(root)
    directory = root / '.runtime' / 'tools' / 'uv'
    system = platform.system()
    binary = 'uv.exe' if system == 'Windows' else 'uv'
    executable = directory / binary
    if executable.is_file():
        return str(executable)
    arch = {'amd64': 'x64', 'x86_64': 'x64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(platform.machine().lower())
    if (system, arch) not in UV_WHEELS:
        raise RuntimeError('Supported hosts: Windows, macOS, Linux; x64/ARM64 (Pi requires 64-bit OS)')
    url, expected = UV_WHEELS[(system, arch)]
    print('Installing uv ' + UV_VERSION + ' locally from its pinned PyPI wheel (first setup only)...', flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='uv-bootstrap-', dir=directory) as temporary:
        archive = Path(temporary) / 'uv.whl'
        with urllib.request.urlopen(url, timeout=90) as response, archive.open('wb') as stream:
            shutil.copyfileobj(response, stream)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError('uv wheel checksum mismatch')
        member = 'uv-' + UV_VERSION + '.data/scripts/' + binary
        staged = Path(temporary) / binary
        with zipfile.ZipFile(archive) as wheel:
            if wheel.namelist().count(member) != 1 or wheel.getinfo(member).is_dir():
                raise RuntimeError('uv wheel does not contain the expected binary: ' + member)
            with wheel.open(member) as source, staged.open('xb') as stream:
                shutil.copyfileobj(source, stream)
        if not staged.stat().st_size:
            raise RuntimeError('uv wheel contains an empty binary')
        staged.chmod(0o755)
        # Preserve a local tool another setup process finished during download.
        if executable.is_file():
            return str(executable)
        os.replace(staged, executable)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('uv installation did not produce ' + str(executable))
    return str(executable)


def python_path(root):
    return Path(root)/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')


def sync_python(root, *, source=False):
    root=Path(root); project=root/'scripts/runtime' if source else root
    if not all((project/n).exists() for n in ('pyproject.toml','uv.lock')):raise RuntimeError('Missing runtime pyproject.toml / uv.lock')
    env=os.environ.copy();env.pop('VIRTUAL_ENV',None)
    env['UV_PROJECT_ENVIRONMENT']=str(root/'.venv');env['UV_CACHE_DIR']=str(root/'.cache/uv')
    executable=python_path(root)
    stamp=root/'.runtime/python-lock.json'
    expected=hashlib.sha256((project/'uv.lock').read_bytes()+(project/'pyproject.toml').read_bytes()).hexdigest()
    if executable.exists() and stamp.exists() and stamp.read_text()==expected:return executable
    # --inexact preserves the research packages in the shared root environment.
    command=[find_uv(root),'sync','--locked','--no-dev','--no-build','--project',project,'--python',executable if executable.exists() else '3.12']
    if source:command.append('--inexact')
    run(command,cwd=root,env=env)
    stamp.parent.mkdir(parents=True,exist_ok=True);stamp.write_text(expected)
    return executable


def node_tools(root):
    root=Path(root);tools=root/'.runtime/tools';tools.mkdir(parents=True,exist_ok=True)
    system={'Windows':'win','Linux':'linux','Darwin':'darwin'}.get(platform.system())
    arch={'amd64':'x64','x86_64':'x64','aarch64':'arm64','arm64':'arm64'}.get(platform.machine().lower())
    if not system or not arch:raise RuntimeError('Supported hosts: Windows, macOS, Linux; x64/ARM64 (Pi requires 64-bit OS)')
    name=f'node-v{NODE_VERSION}-{system}-{arch}'
    directory=tools/name;binary=directory/('node.exe' if system=='win' else 'bin/node')
    if not binary.exists():
        suffix='.zip' if system=='win' else '.tar.gz'
        filename=name+suffix;base=f'https://nodejs.org/dist/v{NODE_VERSION}/'
        with urllib.request.urlopen(base+'SHASUMS256.txt',timeout=60) as response:
            checks=dict((line.split()[1].lstrip('*'),line.split()[0]) for line in response.read().decode().splitlines())
        with tempfile.TemporaryDirectory(dir=tools) as temporary:
            archive=Path(temporary)/filename
            urllib.request.urlretrieve(base+filename,archive)
            if hashlib.sha256(archive.read_bytes()).hexdigest()!=checks[filename]:raise RuntimeError('Node checksum mismatch')
            if system=='win':
                with zipfile.ZipFile(archive) as z:z.extractall(tools)
            else:
                with tarfile.open(archive) as z:z.extractall(tools,filter='data')
    npm=directory/('node_modules/npm/bin/npm-cli.js' if system=='win' else 'lib/node_modules/npm/bin/npm-cli.js')
    pnpm_root=tools/f'pnpm-{PNPM_VERSION}'
    pnpm=pnpm_root/'node_modules/pnpm/bin/pnpm.mjs'
    env=os.environ.copy();env['PATH']=str(binary.parent)+os.pathsep+env.get('PATH','')
    if not pnpm.exists():run([binary,npm,'install','--prefix',pnpm_root,'--no-audit','--no-fund',f'pnpm@{PNPM_VERSION}'],env=env)
    env['PATH']=str(pnpm_root/'node_modules/.bin')+os.pathsep+env['PATH']
    return binary,pnpm,env


def prepare_ui(root, *, build):
    root=Path(root);ui=root/'SpeakerUI';node,pnpm,env=node_tools(root)
    stamp=ui/'node_modules/.smart-speaker-lock'
    lock=hashlib.sha256((ui/'pnpm-lock.yaml').read_bytes()+str(build).encode()).hexdigest()
    if not stamp.exists() or stamp.read_text()!=lock:
        command=[node,pnpm,'install','--frozen-lockfile']
        if not build:command.append('--prod')
        run(command,cwd=ui,env=env);stamp.write_text(lock)
    if build:
        for task in ('check','test','build'):run([node,pnpm,'run',task],cwd=ui,env=env)
    if not (ui/'dist/index.html').exists() or not (ui/'dist-server/server/main.js').exists():raise RuntimeError('Missing compiled SpeakerUI. Publish again on the PC.')
    return node,env
