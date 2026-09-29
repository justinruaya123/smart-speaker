#!/usr/bin/env python3
"""PC-v1 commands. Every output is immutable; qualification is never implied."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from pcvcm.contract import Contract, compile_catalog, write_json
from pcvcm.endpoint import EndpointConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=ROOT / 'spec')
    parser.add_argument('--lexicon', type=Path, default=ROOT / 'configs/lexicon.json')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    commands.add_parser('init-spec')
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--output', type=Path, required=True)
    plan = commands.add_parser('recording-plan')
    plan.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--audio-root', type=Path, required=True)
    prepare.add_argument('--confirmation-commitment', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    train = commands.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--endpoint-config', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    for name in ('calibrate', 'evaluate'):
        sub = commands.add_parser(name)
        sub.add_argument('--bundle', type=Path, required=True)
        sub.add_argument('--manifest', type=Path, required=True)
        sub.add_argument('--audio-root', type=Path, required=True)
        sub.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            sub.add_argument('--split', choices=['dev', 'calibration', 'confirmation'],
                             required=True)
            sub.add_argument('--confirmation-commitment', type=Path)
    package = commands.add_parser('package-diagnostic')
    package.add_argument('--bundle', type=Path, required=True)
    package.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init-spec':
        from pcvcm.initialize import initialize
        print(json.dumps(initialize(ROOT), indent=2))
        return
    contract = Contract(args.spec)
    if args.command == 'check':
        catalog = compile_catalog(contract, args.lexicon)
        result = {'phrases': len(contract.positive), 'commands': len(contract.allowed),
                  'rejections': len(contract.negative),
                  'compiled_entries': len(catalog['entries']),
                  'contract_sha256': contract.sha256, 'qualified': False}
    elif args.command == 'smoke':
        from pcvcm.training import training_smoke
        if args.output.exists():
            raise FileExistsError(args.output)
        result = training_smoke()
        write_json(args.output, result)
    elif args.command == 'recording-plan':
        from pcvcm.evidence import recording_plan
        if args.output.exists():
            raise FileExistsError(args.output)
        rows = recording_plan(contract)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        result = {'slots': len(rows), 'recordings_obtained': 0}
    elif args.command == 'prepare':
        from pcvcm.data import prepare
        result = prepare(args.manifest, args.audio_root, args.output, contract,
                         args.lexicon, args.confirmation_commitment)
    elif args.command == 'train':
        from pcvcm.training import fit
        result = fit(args.config, contract, args.lexicon, args.output)
    elif args.command == 'export':
        from pcvcm.bundle import export_selected
        endpoint = EndpointConfig(**json.loads(args.endpoint_config.read_text()))
        result = export_selected(args.run, args.output, contract, args.lexicon, endpoint)
    elif args.command == 'calibrate':
        from pcvcm.pipeline import calibrate_bundle
        result = calibrate_bundle(args.bundle, args.manifest, args.audio_root,
                                  args.output, contract)
    elif args.command == 'evaluate':
        from pcvcm.pipeline import evaluate
        result = evaluate(args.bundle, args.manifest, args.audio_root, args.output,
                          contract, args.split, args.confirmation_commitment)
    else:
        from pcvcm.pipeline import package_diagnostic
        result = package_diagnostic(args.bundle, args.output, contract)
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == '__main__':
    main()
