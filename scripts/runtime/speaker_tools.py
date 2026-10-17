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


def run(command, **kwargs):
    print('>', ' '.join(map(str,command)), flush=True)
    return subprocess.run(list(map(str,command)), check=True, **kwargs)


def find_uv(root):
    # Reuse the existing official installer, but keep all tools in ignored .runtime.
    from scripts.runtime.launcher import find_uv as existing
    location=Path(root)/'.runtime'; location.mkdir(parents=True,exist_ok=True)
    return existing(location)


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
