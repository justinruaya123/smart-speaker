"""Application assets for the canonical PCV publisher, not a second model exporter."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def application_files(target,benchmarks):
    target={'pi':'full','standalone':'benchmark'}.get(target,target)
    if target not in ('inference','benchmark','full') or benchmarks not in ('include','skip','only'):raise ValueError('Invalid publication mode')
    entries={}
    for name in ('start.py','scripts/runtime/launcher.py','scripts/runtime/speaker_start.py',
                 'scripts/runtime/speaker_tools.py','scripts/runtime/speaker_package.py','scripts/runtime/speaker_worker.py',
                 'scripts/runtime/speaker_config.py','scripts/runtime/speaker_confidence.py','scripts/runtime/speaker_engine.py'):
        entries[name]=(ROOT/name).read_bytes()
    if target!='inference':
        for name in ('pcv_full_benchmark.py','pcv_benchmark_adapter.py','pcv_pi_benchmark.py'):
            entries['scripts/runtime/'+name]=(ROOT/'scripts/runtime'/name).read_bytes()
        entries['benchmark.py']=b'''"""Run the full pinned corpus on one logical CPU (PC or Raspberry Pi)."""
from scripts.runtime.pcv_full_benchmark import main
if __name__ == '__main__': main()
'''
        entries['guided_test.py']=b'''"""Record guided microphone requirement evidence; no device actions."""
from scripts.runtime.pcv_pi_benchmark import main
if __name__ == '__main__': main()
'''
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
    entries['START-HERE.txt']=f'''Smart speaker - {target}

Run python3 start.py (Windows: python start.py), Python 3.11+.
First setup requires internet; runtime tools and dependencies are local thereafter.
On Debian/Pi OS use --system-packages for PortAudio/libsndfile if missing.
config.toml is created on first start. Copy .env.example to .env for secrets.
Settings apply after restart. CLI > environment > .env > config.toml > defaults.
UI profiles use live microphone demo execution at 95% intent / 90% every slot.
Model qualification remains separately reported. Configure providers and bindings
before operating devices. Sample fixtures and uploaded WAVs do not execute actions.

python3 start.py --list-mics
python3 start.py --mode serve --mic INDEX --benchmarks skip  (UI profiles)
python3 start.py --wav command.wav                         (all profiles)

Benchmark/full only:
python3 start.py --benchmark-only
.venv/bin/python benchmark.py --output results/recorded-01
.venv/bin/python guided_test.py --guided --mic INDEX --output results/live-01
Recorded reports: metrics.json, REPORT.md, phrases.csv, slots.csv,
trials.jsonl and timing.jsonl. Guided reports: metrics.json, REPORT.md and trials.jsonl.
Recorded replay and live microphone results remain distinct.

Model updates: start.py --update path/to/new.zip. Complete new runtime/UI releases
require a fresh extraction. Preserve your private .env and config.toml separately.
'''.encode()
    return entries


def runtime_identity(entries):
    bound={n:hashlib.sha256(b).hexdigest() for n,b in entries.items()
           if n.startswith(('scripts/','pcvcm/','vcm/','SpeakerUI/')) or n in ('start.py','wake_gate.py','pyproject.toml','uv.lock')}
    return hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
