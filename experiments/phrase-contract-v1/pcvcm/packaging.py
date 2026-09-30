"""Deterministic allowlist publisher with no boolean qualification override."""
import hashlib
import json
from pathlib import Path
import zipfile
from .bundle import ASSETS, RUNTIME_CODE, verify_bundle
from .contract import file_hash, identity
from .dependencies import verify_lock, REQUIRED


def runtime_entries(bundle, contract, dependency_lock, wheels):
    bundle, dependency_lock = Path(bundle), Path(dependency_lock)
    content = verify_bundle(bundle, contract)
    dependencies = verify_lock(dependency_lock, wheels)
    if not REQUIRED <= set(dependencies['wheels']):
        raise ValueError('Incomplete runtime dependency closure')
    import vcm.frontend
    root = Path(__file__).parent
    entries = {f'pcvcm/{name}': (root / name).read_bytes() for name in RUNTIME_CODE}
    entries['pcvcm/__init__.py'] = (root / '__init__.py').read_bytes()
    entries['vcm/__init__.py'] = b'"""PC-v1 frontend only."""\n'
    entries['vcm/frontend.py'] = Path(vcm.frontend.__file__).read_bytes()
    entries['pcv1.py'] = ("from pathlib import Path\nimport sys\n"
        "ROOT = Path(__file__).resolve().parent\nsys.path.insert(0, str(ROOT))\n"
        "from pcvcm.launcher import main\nmain(ROOT)\n").encode()
    for name in (*ASSETS, 'policy.json', 'recipe.json'):
        entries[f'bundle/{name}'] = (bundle / name).read_bytes()
    for name in contract.binding:
        entries[f'spec/{name}'] = (contract.root / name).read_bytes()
    lexicon = root.parent / 'configs/lexicon.json'
    if file_hash(lexicon) != content['catalog']['lexicon_sha256']:
        raise ValueError('Pronunciation provenance differs from package')
    entries['lexicon.json'] = lexicon.read_bytes()
    entries['CMUDICT-LICENSE.txt'] = (root.parent / 'configs/CMUDICT-LICENSE.txt').read_bytes()
    for name in ('requirements.txt', 'THIRD_PARTY_NOTICES.txt', 'dependencies.json'):
        entries[name] = (dependency_lock / name).read_bytes()
    for record in dependencies['wheels'].values():
        entries['wheels/' + record['filename']] = (Path(wheels) / record['filename']).read_bytes()
    return entries


def runtime_content_hash(bundle, contract, dependency_lock, wheels):
    return identity({n: hashlib.sha256(b).hexdigest() for n, b in
                     runtime_entries(bundle, contract, dependency_lock, wheels).items()})


def stage_candidate(bundle, output, contract, dependency_lock, wheels):
    """Local clean-install test input, never a publication or qualification."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    entries = runtime_entries(bundle, contract, dependency_lock, wheels)
    content = verify_bundle(bundle, contract)
    hashes = {n: hashlib.sha256(b).hexdigest() for n, b in entries.items()}
    for name, data in entries.items():
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    receipt = {'status': 'CANDIDATE_UNQUALIFIED', 'qualified': False, 'binding': content['binding'],
               'content_sha256': identity(hashes), 'files': hashes}
    (output / 'MANIFEST.json').write_text(json.dumps(receipt, sort_keys=True) + '\n')
    return receipt


def publish(bundle, output, contract, dependency_lock, wheels, *, diagnostic=False, evidence=None):
    output, bundle, dependency_lock = Path(output), Path(bundle), Path(dependency_lock)
    if output.exists():
        raise FileExistsError('Packages are immutable')
    content = verify_bundle(bundle, contract)
    # Freeze the payload before qualification; verify its identity again below.
    entries = runtime_entries(bundle, contract, dependency_lock, wheels)
    payload_hash = identity({n: hashlib.sha256(b).hexdigest() for n, b in entries.items()})
    if diagnostic:
        if content['policy'].get('enabled') is not False:
            raise ValueError('Diagnostic package requires disabled acceptance')
        release = {'qualified': False, 'status': 'DIAGNOSTIC_UNQUALIFIED'}
    else:
        if evidence is None:
            raise ValueError('Missing bound qualification evidence')
        from .release import verify_release
        release = verify_release(bundle, evidence, contract, dependency_lock, wheels)
        if payload_hash != runtime_content_hash(bundle, contract, dependency_lock, wheels):
            raise ValueError('Runtime assets changed during qualification')
    entries['release.json'] = (json.dumps(release, sort_keys=True, allow_nan=False) + '\n').encode()
    entries['README.txt'] = (release['status'] + '\n'
        '64-bit Linux ARM64, glibc >=2.28, CPython 3.12.\n'
        'Requires OS libportaudio2 and libsndfile1 (record installed versions on target).\n'
        'python3.12 -m venv .venv\n'
        '.venv/bin/python -m pip install --no-index --find-links wheels --only-binary=:all: --require-hashes -r requirements.txt\n'
        '.venv/bin/python pcv1.py --check\n'
        '.venv/bin/python pcv1.py --wav /path/to/mono-16k.wav\n'
        '.venv/bin/python pcv1.py --list-mics\n'
        '.venv/bin/python pcv1.py --mic DEVICE_INDEX --threads 1\n'
        'Recognition is local; no action executor. WAV EOF never authorizes acceptance.\n'
        'Diagnostic bundles do not establish human accuracy or Pi qualification.\n').encode()
    manifest = {'status': release['status'], 'qualified': release['qualified'],
        'content_sha256': payload_hash,
        'binding': content['binding'], 'files': {n: hashlib.sha256(b).hexdigest() for n, b in entries.items()}}
    entries['MANIFEST.json'] = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 30, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failed')
    return {'path': str(output), 'sha256': file_hash(output), 'qualified': release['qualified'],
            'status': release['status'], 'binding': content['binding']}
