"""Explicit frozen-artifact verification, evaluation preparation and source fitting.

Default verification never installs, trains or silently chooses legacy Iris.
The original CPU epoch44 exact-fit comparison remains authoritative and may fail.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from scripts.optionb_epoch44 import ROOT, SNAPSHOT, checked_path, digest, verify_frozen, verify_source_checkout


def run_relative(value):
    pure = PurePosixPath(value)
    if (not value or pure.is_absolute() or '..' in pure.parts or '\\' in value or ':' in value
            or pure.as_posix() != value or len(pure.parts) < 3 or pure.parts[0] != '.cache'):
        raise ValueError('Choose a new run under .cache/ inside the pinned source checkout')
    return value


def snapshot_rows(root):
    verify_frozen(root)
    snapshot = json.loads(checked_path(root, SNAPSHOT).read_bytes())
    rows = {}
    for split in ('test', 'holdout'):
        record = snapshot['views'][split]
        path = checked_path(root, record['path'])
        if digest(path) != record['sha256']:
            raise ValueError('Frozen evaluation metadata identity changed')
        with gzip.open(path, 'rt', encoding='utf8') as stream:
            values = [json.loads(line) for line in stream if line.strip()]
        if len(values) != record['rows'] or any(row['split'] != split for row in values):
            raise ValueError('Frozen evaluation membership changed')
        rows[split] = values
    return rows


def portable_row(row):
    """Keep unresolved references unscored, never treat them as known UNKNOWNs."""
    from scripts.runtime.pcv_schema import validate_command
    expected = row['expected']
    if expected is not None:
        validate_command(expected)
    if row.get('command') != expected:
        raise ValueError('Conflicting original typed references')
    eligible = row['metric_eligible']
    if type(eligible) is not bool:
        raise ValueError('Original metric_eligible must be boolean')
    return dict(id=row['id'], audio=row['audio'], sha256=row['sha256'], pcm_sha256=row['pcm_sha256'],
        expected=expected, metric_eligible=eligible, split=row['split'],
        reference_status='explicit_typed_reference' if eligible else 'ambiguous',
        provenance={'kind':row['kind'],'source':row['source'],'reference_reason':row['reference_reason']})


def export_evaluation(source, run, output=None):
    """Read back exact prepared rows + original WAV/PCM; write only new manifests."""
    from scripts.runtime.iris_pi_benchmark import read_pcm
    source = Path(source).absolute()
    run_relative(run)
    prepared = checked_path(source, run)
    destination = Path(output).absolute() if output is not None else prepared/'portable'
    if destination.exists():
        raise FileExistsError('Preserve existing portable evaluation manifests: ' + str(destination))
    if destination == source or destination.is_relative_to(source/'submission'):
        raise ValueError('Portable evaluation output cannot overwrite frozen publication assets')
    receipt = json.loads((prepared/'prepare-test-holdout.json').read_bytes())
    if receipt.get('wav_and_pcm_verified') is not True or receipt.get('actions_enabled') is not False:
        raise ValueError('Missing exact original preparation evidence')
    frozen = snapshot_rows(source)
    collected = {}
    for split, expected in frozen.items():
        path = prepared/'prepared'/f'{split}.jsonl'
        relative = path.relative_to(source).as_posix()
        if digest(path) != receipt['bindings'][relative]:
            raise ValueError('Prepared evaluation view identity changed')
        actual = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
        canonical = [{**row,'audio':f"{run}/audio/{row['sha256']}.wav"} for row in expected]
        if actual != canonical:
            raise ValueError('Prepared references differ from original frozen membership/order')
        for row in actual:
            audio = checked_path(source,row['audio'])
            pcm = read_pcm(audio,row['sha256'])
            # Original snapshot pcm_digest binds dtype, sample rate and length too.
            header = f'pcm:f32le:16000:{len(pcm)}\0'.encode()
            if hashlib.sha256(header + pcm.astype('<f4').tobytes()).hexdigest() != row['pcm_sha256']:
                raise ValueError('Original decoded PCM identity changed: ' + row['id'])
        collected[split] = actual
    holdout = [portable_row(row) for row in collected['holdout']]
    human = [portable_row(row) for split in ('test','holdout') for row in collected[split] if row['kind']=='human']
    if len(holdout) != 196 or len(human) != 1146:
        raise ValueError('Frozen evaluation population changed')
    outputs = {'holdout-196.jsonl':holdout,'human-test-holdout.jsonl':human}
    destination.mkdir(parents=True,exist_ok=False)
    bindings = {}
    for name, rows in outputs.items():
        with (destination/name).open('x',encoding='utf8',newline='\n') as stream:
            stream.write(''.join(json.dumps(row,sort_keys=True,allow_nan=False)+'\n' for row in rows))
        bindings[name] = {'sha256':digest(destination/name),'rows':len(rows),
                          'ambiguous_n':sum(not row['metric_eligible'] for row in rows)}
    report = dict(evidence_kind='published_artifact_evaluation_inputs',source_root=str(source),
        source_revision='3c05b3beb46f06c3108089e0d8c79474670c6cb7',
        snapshot_sha256=digest(checked_path(source,SNAPSHOT)), manifests=bindings,
        optimizer_updates=0,actions_enabled=False,original_CPU_model_retrained=False)
    with (destination/'manifest-receipt.json').open('x',encoding='utf8') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-choice',choices=['epoch44'],default='epoch44')
    parser.add_argument('operation',nargs='?',choices=['verify','verify-gpu','prepare-evaluation','export-evaluation','fit'],default='verify')
    parser.add_argument('--load',action='store_true',help='Verify and load frozen CPU ONNX; no fit or audio')
    parser.add_argument('--source-checkout',type=Path,help='Clean exact original source revision')
    parser.add_argument('--source-python',type=Path,help='Existing pinned source interpreter; never installed or synchronized here')
    parser.add_argument('--fit',type=Path,help='Completed GPU fit to verify; does not authorize fitting')
    parser.add_argument('--output',default='.cache/iris-publication-evaluation/run-01',help='New run path relative to pinned source')
    parser.add_argument('--export-output',type=Path,help='New portable manifest directory; default SOURCE/RUN/portable')
    parser.add_argument('--execute-fit',action='store_true',help='Explicitly authorize the original full fit, whose exact CPU comparison may fail')
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    args = parser.parse_args(argv)
    from scripts import optionb_epoch44
    if args.operation in ('verify','verify-gpu'):
        if args.execute_fit or args.device != 'cpu':
            parser.error('Fitting requires the explicit fit operation and --execute-fit; no implicit legacy fit')
        command = [args.operation]
        if args.load: command.append('--load')
        if args.source_checkout: command += ['--source-checkout',str(args.source_checkout)]
        if args.fit: command += ['--fit',str(args.fit)]
        return optionb_epoch44.main(command,root=root)
    if args.source_checkout is None:
        parser.error('This operation requires --source-checkout at the pinned original revision')
    if args.load or args.fit:
        parser.error('--load/--fit are verification options only')
    if args.operation == 'fit' and not args.execute_fit:
        parser.error('Full fitting requires fit --execute-fit; exact CPU epoch44 retraining is not verified')
    if args.operation != 'fit' and (args.execute_fit or args.device != 'cpu'):
        parser.error('Evaluation input preparation never trains')
    source = args.source_checkout.absolute()
    try:
        run_relative(args.output)
        verify_source_checkout(source)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    if args.operation != 'export-evaluation':
        if args.source_python is None or not args.source_python.is_file():
            parser.error('Use --source-python pointing to the existing pinned original interpreter')
        command = [str(args.source_python.resolve()),'-B','-m']
        if args.operation == 'prepare-evaluation':
            command += ['scripts.submission.optionb_reproduce','prepare','--splits','test','holdout','--output',args.output]
        else:
            command += ['scripts.submission.optionb_fit','--output',args.output,'--device',args.device]
        # Run original modules only from their own source root, without main's imports.
        env = os.environ.copy(); env.pop('PYTHONPATH',None)
        code = subprocess.call(command,cwd=source,env=env)
        if code or args.operation == 'fit':
            return code  # Never mask the original exact-comparison failure.
    report = export_evaluation(source,args.output,args.export_output)
    print(json.dumps(report,indent=2,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
