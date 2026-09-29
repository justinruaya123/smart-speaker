"""Strict allowlist distribution; a diagnostic ZIP is not a qualified release."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile
from .integrity import load, digest, save

RUNTIME_FILES=('__init__.py','integrity.py','frontend.py','decoder.py','catalog.py','acceptance.py','runtime.py','tui.py','system.py')
BUNDLE_FILES=('model.onnx','recipe.json','catalog.json','policy.json')


def safe_members(entries):
    for name in entries:
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name: raise ValueError('Unsafe archive path')


def check_release(report,binding):
    if not report.get('eligible') or report.get('failed_requirements'):
        raise ValueError('No qualified release: use --diagnostic explicitly')
    if report.get('binding')!=binding: raise ValueError('Release identity mismatch')
    required=('physical_pi4_tested','live_microphone_tested','independent_human_confirmation','all_ten_use_cases_passed','calibration_passed')
    if not all(report.get(k) is True for k in required): raise ValueError('Missing qualified evidence')


def publish(bundle,output,report=None,diagnostic=False):
    from .runtime import Runtime
    bundle=Path(bundle); output=Path(output)
    if output.exists(): raise FileExistsError('Refusing to overwrite a previous package')
    runtime=Runtime(bundle)
    if diagnostic and load(bundle/'policy.json').get('enabled') is not False:
        raise ValueError('Diagnostic publication requires an explicitly disabled acceptance policy')
    binding={**runtime.binding,'recipe':digest(bundle/'recipe.json')}
    if not diagnostic:
        if report is None: raise ValueError('A qualified report or --diagnostic is required')
        receipt=load(report); check_release(receipt,binding)
        summary=Path(report).parent/'summary.json'
        if digest(summary)!=receipt['summary_sha256']: raise ValueError('Release summary changed')
        metrics=load(summary)
        if (metrics.get('accepted_recall') or 0)<.9 or (metrics.get('accepted_precision') or 0)<.99 or not metrics.get('negatives'):
            raise ValueError('Release quality gates failed')
        if metrics['negative_false_accepts']/metrics['negatives']>.01: raise ValueError('Negative false accepts failed')
    root=Path(__file__).resolve().parents[1]
    entries={f'vcm/{name}':(root/'vcm'/name).read_bytes() for name in RUNTIME_FILES}
    entries.update({f'models/selected/{name}':(bundle/name).read_bytes() for name in BUNDLE_FILES})
    entries['benchmark.py']=(root/'benchmark.py').read_bytes()
    entries['requirements.txt']=(root/'requirements.lock').read_bytes()
    entries['README.txt']=(
        'DIAGNOSTIC - NOT QUALIFIED\n' if diagnostic else 'QUALIFIED RELEASE\n').encode()+(
        '64-bit Raspberry Pi OS, CPython 3.12. Install OS libsndfile and PortAudio for microphone input.\n'
        'python3 -m venv .venv\n.venv/bin/python -m pip install --require-hashes --only-binary=:all: -r requirements.txt\n'
        '.venv/bin/python -m vcm.tui --bundle models/selected --wav /path/to/mono.wav\n'
        '.venv/bin/python -m vcm.tui --bundle models/selected --mic DEVICE_INDEX\n'
        'Setup needs network unless dependencies are preinstalled. Recognition makes no network calls.\n'
        'Training corpus and PyTorch are excluded. No device actions. Physical Pi/microphone verification is separate.\n'
        'FSC-derived academic research artifact; see MODEL_CARD.md for restrictions and coverage.\n').encode()
    for name in ('MODEL_CARD.md','DATASET.md'):
        if (root/name).is_file(): entries[name]=(root/name).read_bytes()
    entries['CMUDICT-LICENSE.txt']=(root/'data/sources/cmudict-LICENSE.txt').read_bytes()
    manifest={'diagnostic':diagnostic,'qualified':not diagnostic,'binding':binding,'files':{k:hashlib.sha256(v).hexdigest() for k,v in entries.items()}}
    entries['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    safe_members(entries); output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,9,29,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None: raise ValueError('ZIP CRC failure')
    receipt={'zip_sha256':digest(output),'bytes':output.stat().st_size,**manifest}
    save(output.with_suffix('.receipt.json'),receipt); return receipt
