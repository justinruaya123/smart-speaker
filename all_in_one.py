"""Rebuild retained baseline A on desktop CUDA, then replay the published WAV."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    if os.name != 'nt':
        raise SystemExit('The exact training recipe requires Windows x64 with an RTX 3080-compatible CUDA host.')
    uv = shutil.which('uv')
    if not uv:
        raise SystemExit('Install uv and put it on PATH, then run python all_in_one.py again.')
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    env = cache / 'desktop-a-venv'
    py = env / 'Scripts/python.exe'
    run = cache / datetime.now(timezone.utc).strftime('desktop-a-%Y%m%dT%H%M%S%fZ')
    print('Source:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), '\nRun:', run, flush=True)
    def call(*args):
        subprocess.run(args, cwd=ROOT, check=True)
    if not py.is_file():
        call(uv, 'venv', '--python', '3.13.14', env)
    call(py, '-c', "import sys; sys.exit(0 if sys.version_info[:3] == (3,13,14) else 'Expected CPython 3.13.14')")
    call(uv, 'pip', 'install', '--python', py, '--index-strategy', 'unsafe-best-match', '-r', ROOT / 'submission/shared-demo-a/requirements-training.txt')
    call(py, '-c', "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'CUDA with BF16 is required')")
    call(py, '-B', ROOT / 'scripts/shared_demo_a.py', 'all', '--output', run)
    # The verified frozen runtime requires the model and its four companions together.
    bundle = run / 'cli-bundle'
    bundle.mkdir()
    retained = ROOT / 'submission/models/shared-demo-a'
    for name in ('recipe.json', 'commands.json', 'normalization.json', 'policy.json'):
        shutil.copyfile(retained / name, bundle / name)
    shutil.copyfile(run / 'reproduced.onnx', bundle / 'model.onnx')
    # inference.py enforces the original model hash before loading this fresh export.
    stdout = subprocess.check_output([py, '-B', ROOT / 'inference.py', '--bundle', bundle], cwd=ROOT, text=True)
    print(stdout.strip(), flush=True)
    output = json.loads(stdout)['output']
    expected = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 50}}
    if output.get('accepted') is not True or output.get('command') != expected:
        raise SystemExit('Rebuilt model did not reproduce the published brightness command.')
    print('Rebuild and CLI replay passed:', run)


if __name__ == '__main__':
    main()
