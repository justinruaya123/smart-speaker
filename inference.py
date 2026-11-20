"""Offline desktop baseline-A WAV recognition. No UI, microphone or device actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'submission/shared-demo-a'
REFERENCE = ROOT / 'submission/models/shared-demo-a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, default=ROOT / 'docs/site/bmo/data/sample.wav', help='mono 16 kHz WAV; defaults to the published brightness sample')
    parser.add_argument('--bundle', type=Path, default=REFERENCE, help='baseline-A export and frozen companions')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    snapshot = json.loads((FROZEN / 'snapshot.json').read_bytes())
    for name in ('model.onnx', 'recipe.json', 'commands.json', 'normalization.json', 'policy.json'):
        expected = snapshot['bindings']['submission/models/shared-demo-a/' + name]['sha256']
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise SystemExit('Baseline A identity mismatch: ' + name)
    py = ROOT / '.cache/desktop-a-venv/Scripts/python.exe'
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    # Run inside the historical import closure, not the newer producer namespace.
    return subprocess.call([str(py) if py.is_file() else sys.executable, '-E', '-s', '-B', '-m',
                            'scripts.runtime.demo_slots_v1', '--bundle', str(bundle),
                            '--wav', str(args.wav.resolve())], cwd=FROZEN / 'source', env=env)


if __name__ == '__main__':
    raise SystemExit(main())
