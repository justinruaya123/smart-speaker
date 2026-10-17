"""Application assets for the single Iris package writer."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def application_files(target,benchmarks):
    target={'pi':'full','standalone':'benchmark'}.get(target,target)
    if target not in ('inference','benchmark','full') or benchmarks not in ('include','skip','only'):raise ValueError('Invalid publication mode')
    entries={}
    for name in ('start.py','scripts/runtime/speaker_start.py',
                 'scripts/runtime/speaker_tools.py','scripts/runtime/speaker_package.py','scripts/runtime/speaker_worker.py',
                 'scripts/runtime/speaker_config.py'):
        entries[name]=(ROOT/name).read_bytes()
    if target!='inference':
        for name in ('benchmark.py','scripts/runtime/iris_pi_benchmark.py'):
            entries[name]=(ROOT/name).read_bytes()
        entries['benchmark_pi.sh']=b'#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec .venv/bin/python benchmark.py "$@"\n'
    for name in ('pyproject.toml','uv.lock'):entries[name]=(ROOT/'scripts/runtime'/name).read_bytes()
    entries['start.sh']=b'#!/bin/sh\nset -eu\ncd -- "$(dirname -- "$0")"\nexec python3 start.py "$@"\n'
    entries['start.cmd']=b'@echo off\r\ncd /d "%~dp0"\r\nwhere python >nul 2>nul\r\nif errorlevel 1 (py -3 start.py %*) else (python start.py %*)\r\n'
    if target!='benchmark':
        for name in ('package.json','pnpm-lock.yaml','pnpm-workspace.yaml','UPSTREAM.json','scripts/start.mjs'):
            entries['SpeakerUI/'+name]=(ROOT/'SpeakerUI'/name).read_bytes()
        for folder in ('dist','dist-server'):
            base=ROOT/'SpeakerUI'/folder
            if not base.is_dir():raise ValueError('Build SpeakerUI on the PC before publishing')
            for file in base.rglob('*'):
                if file.is_file():entries['SpeakerUI/'+file.relative_to(ROOT/'SpeakerUI').as_posix()]=file.read_bytes()
        for file in (ROOT/'SpeakerUI').glob('*LICENSE*'):entries['licenses/SpeakerUI-'+file.name]=file.read_bytes()
    entries['licenses/ncnn-LICENSE.txt']=(ROOT/'scripts/runtime/ncnn-LICENSE.txt').read_bytes()
    entries['.env.example']=(ROOT/'scripts/runtime/speaker.env.example').read_bytes()
    template=(ROOT/'scripts/runtime/config.example.toml').read_text(encoding='utf8')
    if target=='benchmark':template=template.replace('mode = "demo"','mode = "qualified"').replace('enabled = true','enabled = false')
    entries['config.example.toml']=template.encode()
    entries['deployment.json']=(json.dumps({'target':target,'benchmarks':'only' if target=='benchmark' else 'skip','runtime_api_version':2,
        'actions_enabled':target!='benchmark','default_mode':'benchmark' if target=='benchmark' else 'serve'},indent=2)+'\n').encode()
    entries['START-HERE.txt']=b"Iris local smart speaker. See docs/HANDOFF-iris-rpi-v1.md.\n"
    return entries


def runtime_identity(entries):
    bound={n:hashlib.sha256(b).hexdigest() for n,b in entries.items()
           if n.startswith(('scripts/','pcvcm/','vcm/','SpeakerUI/')) or n in ('start.py','wake_gate.py','pyproject.toml','uv.lock')}
    return hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
