"""Copy named immutable historical receipts; never copy audio/private caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = '3c05b3beb46f06c3108089e0d8c79474670c6cb7'
INPUTS = {
    'pc-original.json': 'experiments/speaker-optionb-release-v1/pc-benchmark-results.json',
    'pc-repeat-v2.json': 'experiments/speaker-optionb-release-v1/pc-benchmark-rerun-v2.json',
    'training-curve.json': 'experiments/speaker-optionb-tiny-v1/training-curve.json',
    'training-summary.json': 'experiments/speaker-optionb-tiny-v1/training-summary.json',
    'source-ledger.json': 'experiments/speaker-optionb-tiny-v1/references-ledger.json',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    head = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != SOURCE_REVISION:
        raise ValueError('historical source must be the exact pinned revision')
    destination = ROOT / 'presentation' / 'evidence'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'source_revision': head, 'files': {}}
    for name, relative in INPUTS.items():
        original = args.source / relative
        payload = original.read_bytes()
        committed = subprocess.check_output(['git', '-C', str(args.source), 'show', f'{head}:{relative}'])
        if payload != committed:
            raise ValueError(f'dirty historical receipt: {relative}')
        shutil.copyfile(original, destination / name)
        manifest['files'][name] = {'source_path': relative, 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
