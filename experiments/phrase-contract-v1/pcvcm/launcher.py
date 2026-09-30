"""Compact archive entry point; no training, transcript, or external actions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .contract import Contract, file_hash
from .paths import below


def verify_archive_files(root):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if file_hash(below(root, name)) != expected:
            raise ValueError(f'Package integrity mismatch: {name}')
    return manifest


def main(root):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--wav', type=Path)
    modes.add_argument('--mic', type=int)
    modes.add_argument('--list-mics', action='store_true')
    parser.add_argument('--threads', type=int, choices=[1, 2, 4], default=1)
    args = parser.parse_args()
    manifest = verify_archive_files(root)
    from .runtime import Runtime
    runtime = Runtime.from_bundle(Path(root) / 'bundle', Contract(Path(root) / 'spec'), args.threads)
    if args.check:
        print(json.dumps({'integrity': 'verified', 'binding': runtime.binding,
            'package_status': manifest['status'], 'acceptance_enabled': runtime.policy['enabled']}))
    elif args.list_mics:
        import sounddevice as sd
        print(sd.query_devices())
    elif args.mic is not None:
        from .capture import microphone
        input('Press Enter to arm one utterance; endpoint stops capture: ')
        print(json.dumps(microphone(runtime, args.mic)))
    else:
        import soundfile as sf
        pcm, rate = sf.read(args.wav, dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Expected finite mono 16 kHz original PCM')
        for start in range(0, len(pcm), 640):
            runtime.feed(pcm[start:start + 640], start)
            if runtime.finalized or runtime.endpoint.fault:
                break
        print(json.dumps(runtime.eof()))
