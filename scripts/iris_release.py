"""Hash-bound Iris reproduction, software validation and final package workflow.

The default workflow continues our pinned initializer. --skip-train verifies the
selected exports before rebuilding and evaluating. Neither path operates a Pi or
microphone. Source inputs and their access requirements are explicit receipts.
"""
from datetime import datetime, timezone
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'iris-release-v1'
HERE = 'experiments/' + EXPERIMENT
CACHE = '.cache/' + EXPERIMENT
DEFAULT_INPUTS = HERE + '/reproduction-inputs.json'
DEFAULT_RELEASE = HERE + '/reproduction-release.json'
DEFAULT_TESTS = tuple('scripts/test/' + name + '.py' for name in (
    'test_iris_models', 'test_iris_wake_models', 'test_iris_archive',
    'test_iris_configuration', 'test_iris_session', 'test_iris_worker',
    'test_iris_wake_release', 'test_iris_pi_benchmark', 'test_iris_diversity',
    'test_iris_owner', 'test_iris_resources', 'test_iris_metrics', 'test_iris_release',
    'test_iris_reproduce', 'test_iris_publish_privacy', 'test_iris_bootstrap'))
SHA256 = re.compile(r'[0-9a-f]{64}')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def relative_path(root, value):
    """Keep receipts portable; the configured shared .cache symlink is allowed."""
    value = str(value)
    pure = PurePosixPath(value)
    if (not value or pure.is_absolute() or '..' in pure.parts or ':' in value or '\\' in value
            or value.startswith('./') or value.endswith('/')):
        raise ValueError('Unsafe relative path: ' + value)
    return Path(root) / Path(*pure.parts)


def name_under(root, path):
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    try:
        name = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError('Input must use a repository-relative path: ' + str(path)) from error
    relative_path(root, name)
    return name


def verify_bindings(root, bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Nonempty file bindings are required')
    for name, expected in bindings.items():
        path = relative_path(root, name)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('Invalid SHA256: ' + name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('File identity mismatch: ' + name)
    return dict(bindings)


@lru_cache(maxsize=8)
def _audio_mapping(root, receipt_path, modified_ns, size):
    return read_audio_map(Path(root), read(receipt_path))


def read_audio_map(root, receipt):
    if receipt.get('audio_map_path'):
        path = relative_path(root, receipt['audio_map_path'])
        if not path.is_file() or sha(path) != receipt.get('audio_map_sha256'):
            raise ValueError('Portable audio map identity mismatch')
        return read(path)
    return receipt.get('audio_map', {})


def resolve_audio(row, root=ROOT):
    """Resolve immutable historical rows through the declared portable payload map."""
    root = Path(root)
    original = row.get('audio')
    if not isinstance(original, str):
        raise ValueError('Recording requires a relative audio path')
    relative_path(root, original)
    receipt = relative_path(root, os.environ.get('IRIS_REPRODUCTION_INPUTS', DEFAULT_INPUTS))
    mapping = {}
    if receipt.is_file():
        stat = receipt.stat()
        mapping = _audio_mapping(str(root), str(receipt), stat.st_mtime_ns, stat.st_size)
    record = mapping.get(original)
    expected = row.get('sha256')
    if record:
        if record.get('sha256') != expected:
            raise ValueError('Portable audio map identity disagrees with frozen row: ' + original)
        path = relative_path(root, record['path'])
    else:
        path = relative_path(root, original)
    if not expected or not path.is_file() or sha(path) != expected:
        raise ValueError('Missing or changed recording identity: ' + original
                         + '; run the pinned reproduction input preflight')
    return path


def prepare_inputs(root, receipt_path, *, download=True, downloader=None, from_scratch=False):
    """Restore missing pinned inputs; never overwrite a changed local input."""
    root = Path(root)
    receipt_path = Path(receipt_path)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    if not receipt_path.is_file():
        raise ValueError('Missing reproduction input receipt: ' + name_under(root, receipt_path)
                         + '; obtain the published release receipt or reconstruct its declared canonical sources')
    receipt = read(receipt_path)
    if receipt.get('schema_version') != 1 or receipt.get('experiment') != EXPERIMENT:
        raise ValueError('Unsupported reproduction input receipt')
    files = receipt.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Reproduction inputs must declare required files')
    omitted = set()
    if from_scratch:
        included = [item for item in files if not (item.get('role') in ('vcm_initializer', 'vcm_fallback',
                 'vcm_completed_run', 'vcm_candidate', 'vcm_selected')
                 or Path(item['path']).name in ('compact-initializer.pt', 'full-initializer.pt')
                 or '/fallback-bundles/' in item['path'])]
        omitted = {item['path'] for item in files} - {item['path'] for item in included}
        files = included
    seen = set()
    for item in files:
        name, expected = item['path'], item['sha256']
        relative_path(root, name)
        if name in seen or not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('Duplicate input path or invalid SHA256: ' + name)
        seen.add(name)
    def inspect_input(item):
        name, expected = item['path'], item['sha256']
        path = relative_path(root, name)
        if path.is_file():
            if sha(path) != expected or ('bytes' in item and path.stat().st_size != item['bytes']):
                raise ValueError('Input identity mismatch: ' + name)
            return None
        return item
    with ThreadPoolExecutor(max_workers=8) as pool:
        missing = [item for item in pool.map(inspect_input, files) if item is not None]
    for item in missing:
        name = item['path']
        remote = item.get('remote')
        if not remote or not download:
            details = json.dumps(receipt.get('data_requirements', []), sort_keys=True)
            raise ValueError('Missing input ' + name + '; source/access requirements: ' + details)
        if not re.fullmatch(r'[0-9a-f]{40}', remote.get('revision', '')):
            raise ValueError('Remote input revision must be an immutable commit SHA: ' + name)
        if remote.get('repo_type') not in ('model', 'dataset') or not remote.get('repo_id'):
            raise ValueError('Remote input must declare a model/dataset repository: ' + name)
        relative_path(root, remote['filename'])
        if not PurePosixPath(name).is_relative_to(PurePosixPath(CACHE + '/inputs')):
            raise ValueError('Hydration destination must be inside the relative Iris inputs cache: ' + name)
    if missing and downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    def restore_input(item):
        name, expected, remote = item['path'], item['sha256'], item['remote']
        path = relative_path(root, name)
        downloaded = Path(downloader(repo_id=remote['repo_id'], repo_type=remote['repo_type'],
                                     revision=remote['revision'], filename=remote['filename']))
        if sha(downloaded) != expected or ('bytes' in item and downloaded.stat().st_size != item['bytes']):
            raise ValueError('Downloaded input identity mismatch: ' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        shutil.copyfile(downloaded, temporary)
        if path.exists():
            temporary.unlink()
            raise ValueError('Input changed during hydration: ' + name)
        os.replace(temporary, path)
        return name
    with ThreadPoolExecutor(max_workers=8) as pool:
        restored = list(pool.map(restore_input, missing))
    for alias in receipt.get('cache_aliases', []):
        if alias['source'] in omitted:
            continue
        destination = relative_path(root, alias['path'])
        source = relative_path(root, alias['source'])
        if not PurePosixPath(alias['path']).is_relative_to(PurePosixPath(CACHE)):
            raise ValueError('Declared aliases may populate only the Iris release cache')
        if not source.is_file() or sha(source) != alias['sha256']:
            raise ValueError('Cache alias source identity mismatch: ' + alias['source'])
        if destination.exists():
            if not destination.is_file() or sha(destination) != alias['sha256']:
                raise ValueError('Cache alias identity mismatch: ' + alias['path'])
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    audio = hydrate_audio(root, receipt, download=download, downloader=downloader)
    return {'receipt': name_under(root, receipt_path), 'receipt_sha256': sha(receipt_path),
            'verified_files': len(files), 'restored': restored,
            'restored_audio': audio['restored'], 'verified_audio': audio['verified'],
            'data_requirements': receipt.get('data_requirements', [])}


def hydrate_audio(root, receipt, *, download=True, downloader=None):
    """Restore exact embedded WAV bytes from pinned lawful upstream shards."""
    groups, verified = {}, 0
    def inspect_audio(pair):
        original, item = pair
        relative_path(root, original)
        path = relative_path(root, item['path'])
        expected = item.get('sha256', '')
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('Invalid mapped audio SHA256: ' + original)
        if path.is_file():
            if sha(path) != expected:
                raise ValueError('Mapped recording identity mismatch: ' + original)
            return original, item, True
        return original, item, False
    with ThreadPoolExecutor(max_workers=8) as pool:
        inspected = list(pool.map(inspect_audio, read_audio_map(root, receipt).items()))
    for original, item, present in inspected:
        if present:
            verified += 1
        elif item.get('parquet') is not None:
            groups.setdefault(item['parquet'], []).append(item)
        else:
            raise ValueError('Missing mapped recording: ' + original + '; source/access requirements: '
                             + json.dumps(receipt.get('data_requirements', []), sort_keys=True))
    sources = {r['id']: r for r in receipt.get('parquet_sources', [])}
    restored = 0
    for identifier, recordings in groups.items():
        if identifier not in sources:
            raise ValueError('Missing pinned parquet source receipt: ' + identifier)
        source = sources[identifier]
        local = relative_path(root, source['path'])
        # Existing shared shards can be reused read-only after an exact hash check.
        for candidate in source.get('local_candidates', []):
            possible = relative_path(root, candidate)
            if possible.is_file() and sha(possible) == source['sha256']:
                local = possible
                break
        if not local.is_file():
            transient = Path(root) / CACHE / 'source-hydration' / (hashlib.sha256(identifier.encode()).hexdigest() + '.json')
            save(transient, {'schema_version': 1, 'experiment': EXPERIMENT, 'files': [source],
                             'data_requirements': receipt.get('data_requirements', [])})
            prepare_inputs(root, transient, download=download, downloader=downloader)
        if sha(local) != source['sha256'] or local.stat().st_size != source['bytes']:
            raise ValueError('Pinned parquet source identity mismatch: ' + identifier)
        from pyarrow.parquet import ParquetFile
        needed = {}
        for item in recordings:
            row = item.get('parquet_row')
            if type(row) is not int or row < 0:
                raise ValueError('Invalid pinned parquet row: ' + identifier)
            needed.setdefault(row, []).append(item)
        offset = 0
        for batch in ParquetFile(local).iter_batches(batch_size=128, columns=['audio']):
            if not needed:
                break
            end = offset + batch.num_rows
            indices = [index for index in needed if offset <= index < end]
            if indices:
                data = batch.to_pylist()
                for index in indices:
                    embedded = (data[index - offset].get('audio') or {}).get('bytes')
                    if not isinstance(embedded, bytes):
                        raise ValueError('Pinned row lacks embedded original audio: ' + identifier)
                    digest = hashlib.sha256(embedded).hexdigest()
                    for item in needed.pop(index):
                        if digest != item['sha256']:
                            raise ValueError('Embedded recording identity mismatch: ' + item['path'])
                        pure = PurePosixPath(item['path'])
                        if not (pure.is_relative_to(PurePosixPath('data/audio')) or
                                pure.is_relative_to(PurePosixPath(CACHE + '/inputs/audio'))):
                            raise ValueError('Audio hydration must target the declared immutable audio pool')
                        target = relative_path(root, item['path'])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            if sha(target) != digest:
                                raise ValueError('Recording changed during hydration: ' + item['path'])
                        else:
                            with target.open('xb') as stream:
                                stream.write(embedded)
                            restored += 1
            offset = end
        if needed:
            raise ValueError('Pinned parquet row is absent: ' + identifier)
    return {'restored': restored, 'verified': verified + restored}


def source_version(root, source_files):
    def git(*args):
        result = subprocess.run(['git', *args], cwd=root, capture_output=True, check=False)
        return result.stdout if result.returncode == 0 else b''
    return {'commit': git('rev-parse', 'HEAD').decode().strip() or None,
            'tree': git('rev-parse', 'HEAD^{tree}').decode().strip() or None,
            'source_diff_sha256': hashlib.sha256(git('diff', '--binary', '--', *source_files)).hexdigest()}


def source_closure(root, seeds):
    """Discover local Python imports; explicit recipe/config inputs remain listed."""
    root = Path(root)
    pending = [name_under(root, p) for p in seeds]
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = relative_path(root, name)
        if not path.is_file():
            raise ValueError('Missing canonical source: ' + name)
        found.add(name)
        if path.suffix != '.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or '').split('.') if node.module else []
                if node.level:
                    package = list(PurePosixPath(name).parent.parts)
                    base = package[:len(package) - node.level + 1] + base
                prefix = '.'.join(base)
                modules = [prefix, *(prefix + '.' + alias.name for alias in node.names)]
            for module in modules:
                if not module or module.startswith('.'):
                    continue
                relative = module.replace('.', '/')
                for candidate in (relative + '.py', relative + '/__init__.py'):
                    if (root / candidate).is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)


def _owner_resolver_contract(source):
    """Fingerprint only the owner-consumed resolver and its module bindings."""
    required = {'sha', 'read', 'relative_path', 'read_audio_map', '_audio_mapping', 'resolve_audio',
                'ROOT', 'EXPERIMENT', 'HERE', 'CACHE', 'DEFAULT_INPUTS', 'SHA256',
                'Path', 'PurePosixPath', 'os', 'json', 'hashlib', 'lru_cache', 're'}
    found = {}
    def add(name, node):
        if name in required:
            if name in found:
                raise ValueError('Owner resolver source binding was rebound: ' + name)
            found[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    def inspect(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                add(node.name, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    add(alias.asname or alias.name.split('.')[0], node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            add(name.id, node)
            else:
                for field in ('body', 'orelse', 'finalbody'):
                    value = getattr(node, field, [])
                    if isinstance(value, list):
                        inspect(value)
                for handler in getattr(node, 'handlers', []):
                    inspect(handler.body)
    inspect(ast.parse(source).body)
    if set(found) != required:
        raise ValueError('Owner resolver source contract is incomplete')
    return found


def verify_completed_owner(root, input_receipt):
    """Reuse a published completed owner fit without rebinding or raw Libri I/O.

    The original fitting receipt stays immutable. An exact released copy of its
    driver proves that later packaging edits leave the consumed resolver intact.
    """
    root, input_receipt = Path(root), Path(input_receipt)
    cache = root / CACHE / 'owner'
    inputs = read(input_receipt)
    inventory = {item['path']: item for item in inputs.get('files', [])}
    aliases = {item['path']: item for item in inputs.get('cache_aliases', [])}
    def pinned(path):
        name = name_under(root, path)
        alias = aliases.get(name)
        source = alias['source'] if alias else name
        item = inventory.get(source)
        if item is None or (alias and alias['sha256'] != item['sha256']):
            raise ValueError('Completed owner input lacks a pinned inventory identity: ' + name)
        verify_bindings(root, {source: item['sha256'], name: item['sha256']})
        return item['sha256']
    for name in ('selected.pt', 'selection-before-test.json', 'preparation.json', 'normalization.json',
                 'manifest.jsonl', 'synthetic-proxies.jsonl'):
        pinned(cache / name)
    prepared = read(cache / 'preparation.json')
    selected = read(cache / 'selection-before-test.json')
    if selected.get('selection_completed_before_test') is not True:
        raise ValueError('Owner checkpoint has no completed pre-test selection')
    for filename, key in (('selected.pt', 'checkpoint_sha256'), ('manifest.jsonl', 'manifest_sha256'),
                          ('normalization.json', 'normalization_sha256')):
        if sha(cache / filename) != selected.get(key):
            raise ValueError('Completed owner selection identity changed: ' + filename)
    execution_path = root / HERE / 'owner-execution.json'
    if sha(execution_path) != selected.get('execution_sha256'):
        raise ValueError('Completed owner execution identity changed')
    pointer = read(execution_path)['implementation_before_fitting']
    implementation_path = relative_path(root, pointer['path'])
    if sha(implementation_path) != pointer['sha256']:
        raise ValueError('Original owner implementation receipt identity changed')
    implementation = read(implementation_path)
    binding = implementation['binding']
    if implementation['fingerprint'] != pointer['fingerprint']:
        raise ValueError('Owner implementation fingerprint changed')
    verify_bindings(root, {name: expected for name, expected in binding['source_files'].items()
                           if name != 'scripts/iris_release.py'})
    for filename, key in (('manifest.jsonl', 'manifest_sha256'), ('synthetic-proxies.jsonl', 'synthetic_proxy_manifest_sha256'),
                          ('normalization.json', 'normalization_sha256')):
        if sha(cache / filename) != binding.get(key):
            raise ValueError('Frozen owner preparation identity changed: ' + filename)
    protocol_path = root / HERE / 'owner-protocol.json'
    if (sha(protocol_path) != binding['protocol_sha256'] or selected['protocol_sha256'] != binding['protocol_sha256']
            or prepared['protocol_sha256'] != binding['protocol_sha256']):
        raise ValueError('Completed owner protocol identity changed')
    verify_bindings(root, {item['path']: item['sha256'] for item in binding['base_views']})
    for path, key in ((CACHE + '/inputs/owner-reservations.json', 'reservation_sha256'),
                      (HERE + '/owner-base-view-equivalence.json', 'base_view_equivalence_sha256')):
        if binding.get(key) is not None:
            verify_bindings(root, {path: binding[key]})
    prefit_path = root / CACHE / 'inputs/provenance/owner-prefit-driver.py'
    pinned(prefit_path)
    if sha(prefit_path) != binding['source_files']['scripts/iris_release.py']:
        raise ValueError('Owner prefitting resolver source identity changed')
    before = _owner_resolver_contract(prefit_path.read_text(encoding='utf-8'))
    if _owner_resolver_contract((root / 'scripts/iris_release.py').read_text(encoding='utf-8')) != before:
        raise ValueError('Owner resolver source changed after fitting')
    from scripts.train.iris_owner import _verify_prepared
    protocol, _ = _verify_prepared(root, cache, prepared)
    normalization = read(cache / 'normalization.json')
    if normalization.get('manifest_sha256') != prepared['manifest_sha256'] or normalization.get('frontend') != protocol['frontend']:
        raise ValueError('Owner normalization source identity changed')
    return {'status': 'verified_completed_owner_preparation', 'checkpoint_sha256': selected['checkpoint_sha256'],
            'selection_sha256': sha(cache / 'selection-before-test.json'), 'implementation_before_fitting': pointer,
            'resolver_contract': before, 'raw_librispeech_hydrated': False, 'original_receipts_rewritten': False}


def _diversity_fit_contract(source):
    """Preserve fitting/selection AST except the exact duplicate-key repair."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'evaluate']
    duplicate = ast.parse("dict(**chosen, all_runs=selections, rule=cfg['selection'], test_used_for_selection=False)", mode='eval').body
    duplicate_ast = ast.dump(duplicate, include_attributes=False)
    for function in tree.body:
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name == 'select':
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and ast.dump(node, include_attributes=False) == duplicate_ast:
                    node.keywords.pop()
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def verify_completed_diversity(root, input_receipt, model, arm, seed):
    """Verify a completed fit with the declared replay/serialization repairs."""
    root = Path(root)
    inputs = read(input_receipt)
    inventory = {item['path']: item for item in inputs.get('files', [])}
    aliases = {item['path']: item for item in inputs.get('cache_aliases', [])}
    folder = root / CACHE / 'diversity/runs' / (model + '-' + arm + '-s' + str(seed))
    def pinned(path):
        name = name_under(root, path)
        alias = aliases.get(name)
        source = alias['source'] if alias else name
        item = inventory.get(source)
        if item is None or (alias and alias['sha256'] != item['sha256']):
            raise ValueError('Completed diversity input lacks a pinned inventory identity: ' + name)
        verify_bindings(root, {source: item['sha256'], name: item['sha256']})
        return item['sha256']
    for name in ('best.pt', 'last.pt', 'selection.json', 'run-inputs.json', 'history.json'):
        pinned(folder / name)
    selected, binding = read(folder / 'selection.json'), read(folder / 'run-inputs.json')
    for key, value in (('model', model), ('arm', arm), ('seed', seed)):
        if binding.get(key) != value or selected.get(key) != value:
            raise ValueError('Completed diversity run identity changed: ' + key)
    if selected.get('test_used_for_selection') is not False or selected.get('checkpoint') != name_under(root, folder / 'best.pt'):
        raise ValueError('Completed diversity pre-test selection identity changed')
    if sha(folder / 'best.pt') != selected.get('checkpoint_sha256'):
        raise ValueError('Completed diversity checkpoint identity changed')
    paths = {'protocol_sha256': HERE + '/protocol.json', 'data_audit_sha256': CACHE + '/diversity/data-audit.json',
             'initializer_sha256': CACHE + '/inputs/' + model + '-initializer.pt',
             'model_source_sha256': 'scripts/models/mvp_static.py',
             'contrastive_source_sha256': 'scripts/models/speaker_demo.py',
             'frontend_sha256': 'vcm/frontend.py', 'policy_source_sha256': 'scripts/runtime/mvp_static.py'}
    verify_bindings(root, {name: binding.get(key) for key, name in paths.items()})
    prefit = root / CACHE / 'inputs/provenance/diversity-prefit-trainer.py'
    pinned(prefit)
    if sha(prefit) != binding.get('trainer_sha256'):
        raise ValueError('Prefitting diversity trainer source identity changed')
    contract = _diversity_fit_contract(prefit.read_text(encoding='utf-8'))
    if _diversity_fit_contract((root / 'scripts/train/iris_diversity.py').read_text(encoding='utf-8')) != contract:
        raise ValueError('Completed diversity fit source changed outside the declared repairs')
    return {'status': 'verified_completed_diversity_fit', 'model': model, 'arm': arm, 'seed': seed,
            'checkpoint_sha256': selected['checkpoint_sha256'], 'selection_sha256': sha(folder / 'selection.json'),
            'fit_source_contract_sha256': contract, 'original_receipts_rewritten': False}


def ensure_source_environment(root=ROOT, *, device='cpu', runner=subprocess.run):
    """Bootstrap one root environment only when it does not already exist.

    An active research environment is never synchronized or changed by this
    helper. Recorded training pins are separate from the portable runtime lock.
    """
    root = Path(root)
    python = root / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
    if python.is_file():
        return python
    began = time.perf_counter()
    from scripts.runtime.speaker_tools import find_uv
    environment = read(root / HERE / 'reproduction-environment.json')
    profile = 'cuda' if str(device).startswith('cuda') else 'cpu'
    pins = environment.get('bootstrap_profiles', {}).get(profile)
    if not pins:
        raise ValueError('Missing recorded training environment bootstrap profile: ' + profile)
    uv = find_uv(root)
    runner([str(uv), 'venv', str(root / '.venv'), '--python', sys.executable], cwd=root, check=True)
    if not python.is_file():
        raise ValueError('Root environment bootstrap did not create its interpreter')
    # Verify the repository's hash-pinned runtime prerequisites first, then
    # apply the explicitly recorded research environment in that same .venv.
    runner([str(uv), 'pip', 'install', '--python', str(python), '--require-hashes',
            '-r', str(root / 'requirements.lock')], cwd=root, check=True)
    index = 'https://download.pytorch.org/whl/cu128' if profile == 'cuda' else 'https://download.pytorch.org/whl/cpu'
    local = '+cu128' if profile == 'cuda' else '+cpu'
    torch_version = environment['packages']['torch'].split('+', 1)[0]
    runner([str(uv), 'pip', 'install', '--python', str(python), '--index-url', index,
            'torch==' + torch_version + local, 'torchaudio==' + torch_version + local], cwd=root, check=True)
    path = root / CACHE / 'environment' / ('training-' + profile + '.txt')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(pins) + '\n', encoding='utf-8')
    runner([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-r', str(path)], cwd=root, check=True)
    plots = root / 'experiments/iris-reproduction-v1/plot-requirements.txt'
    if plots.is_file():
        runner([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-r', str(plots)], cwd=root, check=True)
    save(root / CACHE / 'environment/bootstrap.json', {'profile': profile,
         'environment_receipt_sha256': sha(root / HERE / 'reproduction-environment.json'),
         'runtime_lock_sha256': sha(root / 'requirements.lock'), 'training_requirements_sha256': sha(path),
         'elapsed_seconds': time.perf_counter() - began,
         'python': name_under(root, python), 'existing_environment_modified': False,
         'wheel_hash_lock': 'portable runtime only; research versions are pinned in recorded environment metadata'})
    return python


def _capture_selected_cuda_rng(torch, device):
    return torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None


def _restore_selected_cuda_rng(torch, device, state):
    if str(device).startswith('cuda') and state is not None:
        torch.cuda.set_rng_state(state, device)


def fit_random_initializer(*, root, folder, train_rows, dev_rows, norm, classes,
                           width, seed=53, epochs=48, steps=96, batch=32,
                           learning_rate=3e-4, device='cpu'):
    """Canonical atomic-command CE fit, separate from the historical lineage.

    Whole original utterances retain their labels/splits. No historical weights
    are read. Every saved optimizer boundary includes RNG and selection state.
    """
    import random
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.models.mvp_static import StaticCommandNet
    from vcm.frontend import features
    root, folder = Path(root), Path(folder)
    if epochs < 1 or steps < 1 or batch < 2 or batch % 2:
        raise ValueError('Canonical initializer needs positive epochs/steps and an even batch')
    if width not in (128, 192) or not 2 <= classes <= 4096:
        raise ValueError('Unsupported finite initializer architecture')
    train_rows = [r for r in train_rows if r.get('class_index') is not None]
    dev_rows = [r for r in dev_rows if r.get('class_index') is not None]
    if not train_rows or not dev_rows:
        raise ValueError('Frozen train and development recordings are required')
    config = {'classes': classes, 'width': width}
    identity = {'architecture': config, 'seed': seed, 'epochs': epochs, 'steps_per_epoch': steps,
                'batch': batch, 'learning_rate': learning_rate, 'weight_decay': .001,
                'label_smoothing': .02, 'clip': 5., 'initialization': 'random',
                'train_rows_sha256': hashlib.sha256(json.dumps(train_rows, sort_keys=True).encode()).hexdigest(),
                'dev_rows_sha256': hashlib.sha256(json.dumps(dev_rows, sort_keys=True).encode()).hexdigest(),
                'normalization_sha256': hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest(),
                'model_source_sha256': sha(Path(sys.modules[StaticCommandNet.__module__].__file__)),
                'trainer_source_sha256': sha(Path(__file__))}
    binding = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    selection_path = folder / 'selection.json'
    if selection_path.is_file():
        selected = read(selection_path)
        if selected['identity_sha256'] != binding:
            raise ValueError('Canonical initializer identity changed')
        if sha(relative_path(root, selected['checkpoint'])) != selected['checkpoint_sha256']:
            raise ValueError('Canonical initializer checkpoint identity changed')
        return selected
    save(folder / 'protocol.json', identity)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.random.default_generator.manual_seed(seed)
    torch.set_num_threads(1)
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('Requested CUDA device is unavailable; use --device cpu')
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    net = StaticCommandNet(**config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    latest = folder / 'latest.pt'
    start_epoch, start_step, best, history, resume_losses = 1, 0, None, [], []
    if latest.is_file():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        if saved['identity_sha256'] != binding:
            raise ValueError('Canonical initializer resume identity changed')
        net.load_state_dict(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(saved['scheduler'])
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        _restore_selected_cuda_rng(torch, device, saved.get('cuda_rng'))
        start_epoch, start_step = saved['epoch'], saved['next_step']
        best, history = saved['best'], saved['history']
        resume_losses = saved.get('epoch_losses', []) if start_step else []
    arrays = {}
    feature_dir = folder.parent / 'features'
    feature_dir.mkdir(parents=True, exist_ok=True)
    for row in [*train_rows, *dev_rows]:
        identifier = row['id']
        if identifier in arrays:
            continue
        audio = resolve_audio(row, root=root)
        key = hashlib.sha256((row['sha256'] + identity['normalization_sha256'] + 'vcm-logmel40-v1').encode()).hexdigest()
        cached = feature_dir / (key + '.npy')
        if cached.is_file():
            acoustic = np.load(cached, allow_pickle=False)
        else:
            pcm, rate = sf.read(audio, dtype='float32')
            if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
                raise ValueError('Canonical initializer requires original mono 16 kHz PCM')
            acoustic = features(pcm, norm['mean'], norm['std'])
            with cached.open('wb') as stream:
                np.save(stream, acoustic, allow_pickle=False)
        if acoustic.shape[0] != 40 or not acoustic.shape[1] or not np.isfinite(acoustic).all():
            raise ValueError('Invalid canonical initializer features')
        arrays[identifier] = acoustic
    strata = {}
    for synthetic in (False, True):
        members = [r for r in train_rows if bool(r.get('synthetic')) == synthetic]
        if not members:
            raise ValueError('Canonical source balance needs human and synthetic training recordings')
        groups = {}
        for row in members:
            groups.setdefault(row['class_index'], {}).setdefault(row['speaker'], []).append(row)
        strata[synthetic] = groups

    def sample():
        result = []
        for synthetic in (False, True):
            groups = strata[synthetic]
            positive = sorted(k for k in groups if k != 0)
            for _ in range(batch // 2):
                category = 0 if 0 in groups and (rng.random() < .2 or not positive) else int(rng.choice(positive))
                speakers = sorted(groups[category])
                speaker = speakers[int(rng.integers(len(speakers)))]
                examples = groups[category][speaker]
                result.append(examples[int(rng.integers(len(examples)))])
        return result

    def tensor(rows):
        lengths = [arrays[r['id']].shape[1] for r in rows]
        acoustic = np.zeros((len(rows), 40, max(lengths)), dtype='float32')
        for index, row in enumerate(rows):
            acoustic[index, :, :lengths[index]] = arrays[row['id']]
        return (torch.from_numpy(acoustic).to(device),
                torch.tensor(lengths, dtype=torch.long, device=device),
                torch.tensor([r['class_index'] for r in rows], dtype=torch.long, device=device))

    def atomic_save(path, value):
        temporary = path.with_name(path.name + '.tmp')
        torch.save(value, temporary)
        os.replace(temporary, path)

    def state(epoch, next_step):
        return {'config': config, 'weights': {k: v.detach().cpu() for k, v in net.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state,
                'python_rng': random.getstate(),
                'cuda_rng': _capture_selected_cuda_rng(torch, device),
                'epoch': epoch, 'next_step': next_step, 'best': best, 'history': history,
                'epoch_losses': losses,
                'identity_sha256': binding, 'initialization': 'random'}

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        losses = list(resume_losses) if epoch == start_epoch and start_step else []
        for step in range(start_step if epoch == start_epoch else 0, steps):
            chosen = sample()
            optimizer.zero_grad(set_to_none=True)
            value = 0.
            for offset in range(0, batch, 8):
                part = chosen[offset:offset + 8]
                x, lengths, labels = tensor(part)
                loss = torch.nn.functional.cross_entropy(net(x, lengths), labels, label_smoothing=.02)
                (loss * len(part) / batch).backward()
                value += float(loss.detach()) * len(part) / batch
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            losses.append(value)
            if (step + 1) % 16 == 0:
                atomic_save(latest, state(epoch, step + 1))
        net.eval()
        people, human, synthetic = {}, 0, 0
        with torch.no_grad():
            for offset in range(0, len(dev_rows), 16):
                part = dev_rows[offset:offset + 16]
                x, lengths, labels = tensor(part)
                predictions = net(x, lengths).argmax(1).cpu().tolist()
                for row, prediction in zip(part, predictions):
                    correct = prediction == row['class_index']
                    if row.get('synthetic'):
                        synthetic += int(correct)
                    else:
                        people.setdefault(row['speaker'], []).append(correct)
                        human += int(correct)
        key = [float(np.mean([np.mean(v) for v in people.values()])) if people else 0., human, synthetic]
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)) if losses else None, 'development_key': key})
        scheduler.step()
        if best is None or key > best['key']:
            best = {'epoch': epoch, 'key': key}
            atomic_save(folder / 'best.pt', state(epoch + 1, 0))
        atomic_save(latest, state(epoch + 1, 0))
        print(json.dumps({'canonical_initializer': width, **history[-1]}), flush=True)
    if best is None:
        raise ValueError('Canonical initializer completed without a development selection')
    selected = {'checkpoint': name_under(root, folder / 'best.pt'), 'checkpoint_sha256': sha(folder / 'best.pt'),
                'identity_sha256': binding, 'config': config, 'parameters': sum(p.numel() for p in net.parameters()),
                'best': best, 'initialization': 'random', 'test_used_for_selection': False,
                'source_balance': 'half original-human, half declared-synthetic; UNKNOWN probability .2 per stratum'}
    save(selection_path, selected)
    return selected


def run_from_scratch(*, root=ROOT, run_id, device='cpu'):
    """A separately versioned random study, never a replacement for V2 weights."""
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.models.mvp_static import StaticCommandNet
    from scripts.runtime.mvp_static import Runtime, fit_policy, decide
    from scripts.train.iris_export import export_onnx, export_ncnn
    from scripts.test.iris_metrics import summarize
    from vcm.frontend import features
    root = Path(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise ValueError('Invalid random-study version')
    inputs = root / CACHE / 'inputs'
    directory = root / CACHE / 'from-scratch' / run_id
    protocol_path = root / HERE / ('reproduction-from-scratch-' + run_id + '-protocol.json')
    result_path = root / HERE / ('reproduction-from-scratch-' + run_id + '.json')
    def rows(split):
        return [json.loads(line) for line in (inputs / 'base-views' / (split + '.jsonl')).read_text().splitlines() if line]
    source_inputs = [inputs / 'base-views' / (s + '.jsonl') for s in ('train', 'dev', 'calibration', 'test', 'holdout', 'personal-test')]
    source_inputs += [inputs / 'catalog-parent.json', inputs / 'normalization.json']
    sources = read(relative_path(root, os.environ.get('IRIS_REPRODUCTION_INPUTS', DEFAULT_INPUTS)))['source_files']
    protocol = {'schema_version': 1, 'experiment': 'iris-random-' + run_id,
                'hypothesis': 'Our finite atomic encoder can be reconstructed from random weights on the frozen base corpus',
                'requirements': ['ME-01', 'ME-02', 'ME-03', 'ME-04', 'ME-06', 'ME-07', 'ME-08'],
                'initialization': 'random; independent reproducibility alternative, not historical V2 weight reproduction',
                'training': {'epochs': 48, 'steps_per_epoch': 96, 'batch': 32, 'microbatch': 8,
                             'optimizer': 'AdamW', 'learning_rate': 3e-4, 'weight_decay': .001,
                             'seed': 53, 'source_balance': 'half human/half synthetic; UNKNOWN probability .2'},
                'development_selection': 'macro human-speaker raw exact, human count, synthetic count, fewer parameters',
                'test_used_for_selection': False, 'physical_pi_measured': False,
                'bindings': {name_under(root, p): sha(p) for p in source_inputs},
                'source_bindings': {name: sha(relative_path(root, name)) for name in sources}}
    if protocol_path.exists():
        if read(protocol_path) != protocol:
            raise ValueError('Random-study protocol identity changed; use a new run ID')
    else:
        save(protocol_path, protocol)
    verify_bindings(root, protocol['bindings'])
    verify_bindings(root, protocol['source_bindings'])
    if result_path.exists():
        result = read(result_path)
        verify_bindings(root, result['bindings'])
        return result
    train_rows, dev_rows = rows('train'), rows('dev')
    norm = read(inputs / 'normalization.json')
    catalog = read(inputs / 'catalog-parent.json')
    fits = {}
    for label, width in (('compact', 128), ('full', 192)):
        fits[label] = fit_random_initializer(root=root,
            folder=inputs / 'rebuilt-initializers' / run_id / label, train_rows=train_rows,
            dev_rows=dev_rows, norm=norm, classes=len(catalog), width=width, device=device)
    selected_name = max(fits, key=lambda k: (*fits[k]['best']['key'], -fits[k]['parameters']))
    chosen = fits[selected_name]
    saved = torch.load(relative_path(root, chosen['checkpoint']), map_location='cpu', weights_only=True)
    net = StaticCommandNet(**saved['config'])
    net.load_state_dict(saved['weights'])
    net.eval()
    torch.set_num_threads(1)
    export = directory / 'export'
    export.mkdir(parents=True, exist_ok=True)
    if not (export / 'model.onnx').exists():
        export_onnx(net, export / 'model.onnx')
    if not (export / 'ncnn/engine.json').exists():
        export_ncnn(net, export / 'model.onnx', export / 'ncnn')
    for name in ('catalog.json', 'normalization.json'):
        source = inputs / ('catalog-parent.json' if name == 'catalog.json' else name)
        destination = export / name
        if destination.exists() and sha(destination) != sha(source):
            raise ValueError('Random-study export input identity changed')
        if not destination.exists():
            shutil.copyfile(source, destination)
    def predict(records):
        result = []
        with torch.no_grad():
            for row in records:
                pcm, rate = sf.read(resolve_audio(row, root=root), dtype='float32')
                if rate != 16000 or pcm.ndim != 1:
                    raise ValueError('Unexpected original recording format')
                acoustic = features(pcm, norm['mean'], norm['std'])
                logits = net(torch.from_numpy(acoustic[None]), torch.tensor([acoustic.shape[1]])).numpy()[0]
                result.append({**{k: row.get(k) for k in ('id', 'speaker', 'synthetic', 'split', 'source')},
                               'expected_index': row['class_index'], 'logits': logits.tolist()})
        return result
    fit_rows = [r for split in ('dev', 'calibration') for r in rows(split) if r['class_index'] is not None]
    trained = sorted({r['class_index'] for r in train_rows if r['class_index'] is not None})
    policy = fit_policy(predict(fit_rows), trained, catalog=catalog, risk={'human': .03, 'synthetic': .01},
                        threshold_strategy='observed', grouping='class', min_correct=2)
    policy.update(enabled=True, qualified=False,
                  qualification_reason='Independent random reconstruction study; strict/Pi/household qualification unestablished')
    if (export / 'policy.json').exists() and read(export / 'policy.json') != policy:
        raise ValueError('Preserve frozen random-study policy')
    save(export / 'policy.json', policy)
    save(export / 'bundle.json', {'format': 'mvp-static-v1', 'parameters': chosen['parameters'],
                                 'files': {n: sha(export / n) for n in ('model.onnx', 'catalog.json', 'normalization.json', 'policy.json')},
                                 'qualified': False, 'inference_mode': 'complete_utterance'})
    # Both the policy and all selected graph bytes are frozen before test replay.
    selection_path = directory / 'selection-before-test.json'
    lock = {'experiment': protocol['experiment'], 'selected': selected_name, 'fits': fits,
            'protocol_sha256': sha(protocol_path), 'test_used_for_selection': False,
            'bindings': {name_under(root, p): sha(p) for p in export.rglob('*') if p.is_file() and p.suffix != '.pt'}}
    if selection_path.exists() and read(selection_path) != lock:
        raise ValueError('Random-study pre-test selection identity changed')
    save(selection_path, lock)
    runtime = Runtime(export, engine='onnxruntime')
    lookup = {json.dumps(item['command'], sort_keys=True): item['command_id'] for item in catalog}
    results = {}
    for split in ('test', 'holdout', 'personal-test'):
        records = []
        for row in rows(split):
            pcm, rate = sf.read(resolve_audio(row, root=root), dtype='float32')
            event = runtime.recognize(pcm, rate)
            diagnostic = event.get('diagnostic_candidate') or {}
            records.append({**{k: row.get(k) for k in ('id', 'speaker', 'synthetic', 'split', 'source', 'accent_group')},
                            'expected_index': row['class_index'], 'raw_index': diagnostic.get('class_index'),
                            'accepted_index': lookup.get(json.dumps(event['output']['command'], sort_keys=True))
                             if event['output']['accepted'] else None})
        results[split] = summarize(records, catalog)
    verify_bindings(root, protocol['bindings'])
    verify_bindings(root, protocol['source_bindings'])
    verify_bindings(root, lock['bindings'])
    result = {'schema_version': 1, 'experiment': protocol['experiment'], 'status': 'complete',
              'bundle': name_under(root, export), 'selection': name_under(root, selection_path),
              'protocol': name_under(root, protocol_path), 'selected': selected_name, 'fits': fits,
              'results': results, 'qualified': False, 'physical_pi_measured': False,
              'historical_v2_reproduced': False, 'test_used_for_selection': False,
              'bindings': {**lock['bindings'], name_under(root, protocol_path): sha(protocol_path),
                           name_under(root, selection_path): sha(selection_path)}}
    save(result_path, result)
    return result


def export_files(folder, metadata):
    folder = Path(folder)
    if not (folder / metadata).is_file():
        raise ValueError('Missing selected export metadata: ' + str(folder / metadata))
    document = read(folder / metadata)
    result = [folder / metadata]
    files = document.get('files', {})
    names = files if isinstance(files, dict) else files
    for name in names:
        path = relative_path(folder, name)
        if not path.is_file():
            raise ValueError('Missing selected export member: ' + str(path))
        if isinstance(files, dict):
            expected = files[name]
            expected = expected.get('sha256') if isinstance(expected, dict) else expected
            if isinstance(expected, str) and SHA256.fullmatch(expected) and sha(path) != expected:
                raise ValueError('Selected export identity mismatch: ' + str(path))
        result.append(path)
    for name in ('ncnn/engine.json', 'ncnn/model.ncnn.param', 'ncnn/model.ncnn.bin',
                 'ncnn/position-frequency.json', 'owner.onnx'):
        if (folder / name).is_file():
            result.append(folder / name)
    return sorted(set(result))


def freeze_release(*, root=ROOT, bundle=None, wake_folder=None, owner_folder=None,
                   input_receipt=None, selection=None, source_files=None, output=None, protocol=None,
                   vad_folder=None, study_kind='continued', evaluation_receipts=None):
    """Bind the full selected lineage without copying a training cache into Git."""
    root = Path(root)
    bundle = Path(bundle or root / CACHE / 'diversity/release-export')
    wake_folder = Path(wake_folder or root / CACHE / 'wake/export')
    owner_folder = Path(owner_folder or root / CACHE / 'owner/export')
    vad_folder = Path(vad_folder or root / CACHE / 'inputs/vad')
    input_receipt = Path(input_receipt or root / DEFAULT_INPUTS)
    selection = Path(selection or root / HERE / 'selection-before-test.json')
    protocol = Path(protocol or root / HERE / 'protocol.json')
    output = Path(output or root / DEFAULT_RELEASE)
    files = export_files(bundle, 'bundle.json') + export_files(wake_folder, 'iris.json')
    files += export_files(owner_folder, 'owner.json')
    if (vad_folder / 'silero-vad.onnx').is_file():
        files.append(vad_folder / 'silero-vad.onnx')
    else:
        raise ValueError('Missing pinned Silero VAD input')
    if (vad_folder / 'wake-models.json').is_file():
        files.append(vad_folder / 'wake-models.json')
    for name in ('artifacts/wake-models.json', 'artifacts/silero-LICENSE.txt'):
        if not (root / name).is_file():
            raise ValueError('Missing pinned VAD source/license metadata: ' + name)
        files.append(root / name)
    input_document = read(input_receipt)
    source_files = list(source_files or input_document.get('source_files', []))
    publication_proof = input_document.get('input_publication_receipt')
    if publication_proof:
        proof = relative_path(root, publication_proof['path'])
        if not proof.is_file() or sha(proof) != publication_proof.get('sha256'):
            raise ValueError('Pinned input publication receipt identity changed')
        if publication_proof['path'] not in source_files:
            source_files.append(publication_proof['path'])
    source_files = [p if Path(p).is_absolute() else root / p for p in source_files]
    if not source_files:
        raise ValueError('Release must bind the canonical training/runtime source closure')
    for path in (input_receipt, selection, protocol, *source_files):
        if not path.is_file():
            raise ValueError('Missing release lineage input: ' + str(path))
        files.append(path)
    evidence = [name_under(root, selection), name_under(root, protocol)]
    default_evidence = ['wake-results.json', 'owner-results.json', 'owner-export.json', 'wake-selection-before-test.json',
                        'wake-protocol.json', 'owner-protocol.json', 'owner-base-view-equivalence.json',
                        'owner-execution.json', 'owner-selection-before-test.json', 'owner-graph-before-test.json',
                        'owner-runtime-checks.json', 'owner-wake-duration-diagnostic.json',
                        'software-verification.json', 'pipeline-resources.json']
    if study_kind == 'continued':
        default_evidence += ['results.json', 'recognition-results.json', 'release-choice.json',
                             'width-selection-before-repeat.json', 'decision-parity.json']
    for name in default_evidence:
        path = root / HERE / name
        if path.is_file():
            files.append(path)
            evidence.append(name_under(root, path))
    for name in evaluation_receipts or []:
        path = relative_path(root, name)
        if not path.is_file():
            raise ValueError('Missing selected-study evaluation evidence: ' + name)
        files.append(path)
        evidence.append(name)
    for name in ('requirements.lock', 'pipeline.toml', 'scripts/runtime/pyproject.toml',
                 'scripts/runtime/uv.lock', 'SpeakerUI/pnpm-lock.yaml', 'SpeakerUI/package.json'):
        if (root / name).is_file():
            files.append(root / name)
    bindings = {name_under(root, p): sha(p) for p in sorted(set(files))}
    sources = {name_under(root, p): sha(p) for p in source_files}
    receipt = {'schema_version': 1, 'experiment': EXPERIMENT, 'study_kind': study_kind,
               'bundle': name_under(root, bundle), 'wake_folder': name_under(root, wake_folder),
               'owner_folder': name_under(root, owner_folder),
               'vad_folder': name_under(root, vad_folder),
               'input_receipt': name_under(root, input_receipt), 'selection': name_under(root, selection),
               'protocol': name_under(root, protocol),
               'source_version': source_version(root, list(sources)), 'source_bindings': sources,
               'bindings': bindings, 'evaluation_receipts': evidence,
               'parameters': read(bundle / 'bundle.json').get('parameters'),
               'qualified': False, 'physical_pi_measured': False, 'live_microphone_measured': False,
               'actions_capability': True, 'recorded_audio_executes_actions': False,
               'profile': 'owner-protected'}
    if output.exists():
        previous = read(output)
        comparable = lambda r: {k: v for k, v in r.items() if k != 'source_version'}
        if comparable(previous) != comparable(receipt):
            raise ValueError('Existing release identity differs; preserve it and use a new release receipt')
        receipt = previous
    else:
        save(output, receipt)
    validate_release(root, output)
    return receipt


def verify_present_release(root, path):
    """Reject changed files before hydrating only declared missing cache assets."""
    path = Path(path)
    if not path.is_absolute():
        path = Path(root) / path
    release = read(path)
    if release.get('schema_version') != 1 or release.get('experiment') != EXPERIMENT:
        raise ValueError('Unsupported release receipt')
    verify_bindings(root, release.get('source_bindings'))
    bindings = release.get('bindings')
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Nonempty release bindings are required')
    for name, expected in bindings.items():
        file = relative_path(root, name)
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError('Invalid release SHA256: ' + name)
        if file.is_file():
            if sha(file) != expected:
                raise ValueError('File identity mismatch: ' + name)
        elif not PurePosixPath(name).is_relative_to(PurePosixPath(CACHE)):
            raise ValueError('Missing source/contract identity: ' + name)
    return release


def validate_release(root, path):
    release = verify_present_release(root, path)
    verify_bindings(root, release['bindings'])
    for role, metadata in (('bundle', 'bundle.json'), ('wake_folder', 'iris.json'), ('owner_folder', 'owner.json')):
        for file in export_files(relative_path(root, release[role]), metadata):
            if name_under(root, file) not in release['bindings']:
                raise ValueError('Selected export member is not bound: ' + str(file))
    for role in ('input_receipt', 'selection'):
        if release.get(role) not in release['bindings']:
            raise ValueError('Release must bind ' + role)
    if type(release.get('qualified')) is not bool:
        raise ValueError('Release qualified must be boolean')
    return release


def verify_package(path, destination=None):
    """Check exact inventory and hashes before extracting any archive member."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        manifest = json.loads(archive.read('MANIFEST.json'))
        declared = manifest.get('files', {})
        if not declared or set(names) != set(declared) | {'MANIFEST.json'}:
            raise ValueError('Archive inventory differs from its manifest')
        for name in names:
            relative_path(Path('.'), name)
            pure = PurePosixPath(name)
            if (pure.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors')
                    or pure.name == '.env' or (pure.name.startswith('.env.') and pure.name != '.env.example')
                    or '__pycache__' in pure.parts or 'node_modules' in pure.parts):
                raise ValueError('Private settings or training payload cannot ship: ' + name)
            if (archive.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not allowed: ' + name)
            if name != 'MANIFEST.json' and hashlib.sha256(archive.read(name)).hexdigest() != declared[name]:
                raise ValueError('Archive file identity mismatch: ' + name)
        if archive.testzip() is not None:
            raise ValueError('Archive CRC failed')
        if destination is not None:
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=False)
            archive.extractall(destination)
    return {'passed': True, 'sha256': sha(path), 'bytes': path.stat().st_size,
            'members': len(names), 'manifest': manifest}


def run_workflow(*, root=ROOT, skip_train=False, publish=False, from_scratch=False,
                 preflight_only=False, download=True, input_receipt=None, release_path=None,
                 run_id=None, python=None, device='cpu', runner=subprocess.run,
                 packager=None, publisher=None, test_modules=None, ui_command=None):
    root = Path(root).absolute()
    python = str(python or root / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python'))
    input_receipt = Path(input_receipt or root / DEFAULT_INPUTS)
    explicit_release = release_path
    release_path = Path(release_path or root / DEFAULT_RELEASE)
    token = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:6]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', token):
        raise ValueError('Run ID must be a bounded simple directory name')
    if skip_train and from_scratch:
        raise ValueError('--from-scratch and --skip-train are mutually exclusive')
    if from_scratch and explicit_release is None:
        release_path = root / HERE / ('reproduction-release-from-scratch-' + token + '.json')
    receipt_path = root / HERE / ('reproduction-run-' + token + '.json')
    if receipt_path.exists():
        raise FileExistsError('Preserve existing workflow receipt: ' + str(receipt_path))
    record = {'schema_version': 1, 'experiment': EXPERIMENT, 'run_id': token, 'status': 'running',
              'training_mode': 'verified_selected_exports' if skip_train else ('from_scratch' if from_scratch else 'pinned_continuation'),
              'physical_pi_measured': False, 'live_microphone_measured': False, 'stages': [],
              'receipt': name_under(root, receipt_path)}
    logs = root / CACHE / 'workflow' / token

    def persist():
        save(receipt_path, record)

    def stage(name, command=None, action=None, cwd=None, environment=None):
        item = {'name': name, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()}
        record['stages'].append(item)
        persist()
        try:
            if command:
                item['command'] = [str(x) for x in command]
                logs.mkdir(parents=True, exist_ok=True)
                log = logs / (name + '.log')
                item['log'] = name_under(root, log)
                persist()
                with log.open('w', encoding='utf-8') as stream:
                    runner(item['command'], cwd=str(cwd or root), check=True,
                           stdout=stream, stderr=subprocess.STDOUT,
                           env={**os.environ, **(environment or {}), 'PYTHONHASHSEED': '53',
                                'IRIS_REPRODUCTION_INPUTS': name_under(root, input_receipt)})
            value = action() if action else None
            if value is not None:
                item['result'] = value
            item['status'] = 'passed'
            return value
        except Exception as error:
            item['status'] = 'failed'
            item['error'] = str(error)
            raise
        finally:
            item['finished_utc'] = datetime.now(timezone.utc).isoformat()
            persist()

    def module(name, stage_name, *args):
        return [python, '-m', 'scripts.train.iris_' + name, stage_name, *map(str, args)]

    def fit_diversity(model, arm, seed):
        name = 'diversity-' + model + '-' + arm + '-' + str(seed)
        if (not read(input_receipt).get('force_vcm_fitting') and
                (root / CACHE / 'diversity/runs' / (model + '-' + arm + '-s' + str(seed)) / 'selection.json').is_file()):
            return stage(name, action=lambda: verify_completed_diversity(root, input_receipt, model, arm, seed))
        return stage(name, module('diversity', 'train', '--model', model, '--arm', arm,
                                 '--seed', seed, '--device', device))

    try:
        if skip_train:
            def verify_current_release():
                release = verify_present_release(root, release_path)
                expected = {'bundle': CACHE + '/diversity/release-export', 'wake_folder': CACHE + '/wake/export',
                            'owner_folder': CACHE + '/owner/export', 'vad_folder': CACHE + '/inputs/vad',
                            'selection': HERE + '/selection-before-test.json', 'protocol': HERE + '/protocol.json'}
                if release.get('study_kind') != 'continued' or any(release.get(key) != value for key, value in expected.items()):
                    raise ValueError('Skip-train requires the canonical Iris release components and selection')
                return release
            stage('verify-present-lineage', action=verify_current_release)
        record['inputs'] = stage('preflight-inputs', action=lambda: prepare_inputs(root, input_receipt, download=download,
                                                                                from_scratch=from_scratch))
        if skip_train:
            stage('verify-selected-lineage', action=lambda: validate_release(root, release_path))
        if not Path(python).is_file():
            raise ValueError('Use the configured root .venv interpreter: ' + python)
        if preflight_only:
            record['status'] = 'preflight_complete'
            persist()
            return record
        if not skip_train and not from_scratch:
            for name in ('freeze', 'generate', 'prepare'):
                stage('diversity-' + name, module('diversity', name))
        if not skip_train:
            for component in ('wake', 'owner'):
                if component == 'owner' and (root / CACHE / 'owner/selection-before-test.json').is_file():
                    stage('owner-prepare', action=lambda: verify_completed_owner(root, input_receipt))
                else:
                    stage(component + '-prepare', module(component, 'prepare'))
        if from_scratch:
            stage('canonical-random-study', [python, '-m', 'scripts.iris_release', '--fit-from-scratch',
                                            '--run-id', token, '--device', device])
            random_study = read(root / HERE / ('reproduction-from-scratch-' + token + '.json'))
            record['canonical_random_study'] = random_study
        if not skip_train and not from_scratch:
            for model in ('compact', 'full'):
                for arm in ('ce', 'supcon'):
                    fit_diversity(model, arm, 53)
            stage('diversity-select-width', module('diversity', 'select'))
            repeat_path = root / HERE / 'width-selection-before-repeat.json'
            repeat = read(repeat_path)
            if repeat.get('model') not in ('compact', 'full') or repeat.get('seed') != 67 or repeat.get('arms') != ['ce', 'supcon']:
                raise ValueError('Invalid development-only width selection receipt')
            record['repeat_selection_sha256'] = sha(repeat_path)
            for arm in ('ce', 'supcon'):
                fit_diversity(repeat['model'], arm, 67)
            for name in ('select', 'export'):
                stage('diversity-' + name, module('diversity', name))
        if not skip_train:
            for component in ('wake', 'owner'):
                names = ('train', 'select', 'export') if component == 'wake' else ('train', 'export')
                for name in names:
                    extra = ('--device', device) if name == 'train' else ()
                    stage(component + '-' + name, module(component, name, *extra))
        for component in (('wake', 'owner') if from_scratch else ('diversity', 'wake', 'owner')):
            stage(component + '-evaluate', module(component, 'evaluate'))
        if not from_scratch:
            parity_command = [python, '-m', 'scripts.test.iris_decision_parity']
            if skip_train:
                parity_receipt = name_under(root, logs / 'decision-parity.json')
                parity_command.extend(['--output', parity_receipt])
            stage('decision-parity', parity_command)
            if skip_train:
                record['decision_parity'] = {'path': parity_receipt,
                    'sha256': sha(relative_path(root, parity_receipt)),
                    'historical_release_parity_unchanged': True}
        resource_command = [python, '-m', 'scripts.test.iris_resources']
        if from_scratch:
            resource_receipt = CACHE + '/from-scratch/' + token + '/pipeline-resources.json'
            resource_command.extend(['--bundle', random_study['bundle'], '--output', resource_receipt])
        elif skip_train:
            resource_receipt = name_under(root, logs / 'pipeline-resources.json')
            resource_command.extend(['--output', resource_receipt])
        profiler_environment = {key: os.environ.get(key) or '1' for key in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
        stage('pipeline-resources', resource_command, environment=profiler_environment)
        if skip_train or from_scratch:
            record['resource_measurement'] = {'path': resource_receipt,
                'sha256': sha(relative_path(root, resource_receipt)),
                'historical_release_resources_unchanged': True, 'physical_pi_measured': False}
        if not from_scratch:
            stage('report-acoustic-evidence', [python, '-m', 'scripts.iris_report'])
        options = {} if not from_scratch else {
            'bundle': relative_path(root, random_study['bundle']),
            'selection': relative_path(root, random_study['selection']),
            'protocol': relative_path(root, random_study['protocol']), 'study_kind': 'random',
            'evaluation_receipts': [HERE + '/reproduction-from-scratch-' + token + '.json', resource_receipt]}
        if not skip_train:
            before_software = stage('snapshot-before-software', action=lambda: freeze_release(
                root=root, input_receipt=input_receipt, output=logs / 'before-software-release.json', **options))
        else:
            before_software = stage('verify-exported-lineage', action=lambda: validate_release(root, release_path))
        # These reports alone may change when the current software checks become
        # available. All source, export and acoustic evidence remains immutable.
        derived = {'MODEL_CARD.md', 'docs/EXPERIMENT-iris-release-v1.md', 'EXPERIMENTS.md',
                   HERE + '/software-verification.json'}
        immutable = {name: expected for name, expected in before_software['bindings'].items() if name not in derived}
        ui = root / 'SpeakerUI'
        ui_environment = None
        if ui_command is None:
            from scripts.runtime.speaker_tools import node_tools
            toolchain = []
            def bootstrap_ui():
                node, pnpm, environment = node_tools(root)
                toolchain.extend(([str(node), str(pnpm)], environment))
                return {'node': name_under(root, node), 'pnpm': name_under(root, pnpm)}
            stage('ui-toolchain', action=bootstrap_ui)
            ui_command, ui_environment = toolchain
        for name, args in (('install', ('install', '--frozen-lockfile')), ('test', ('test',)),
                           ('check', ('check',)), ('build', ('build',))):
            stage('ui-' + name, [*ui_command, *args], cwd=ui, environment=ui_environment)
        stage('python-tests', [python, '-m', 'pytest', *(test_modules or DEFAULT_TESTS), '-q'])
        stage('verify-after-build', action=lambda: verify_bindings(root, immutable))
        software = {'schema_version': 1, 'passed': True,
                    'checks': ['python-tests', 'ui-install', 'ui-test', 'ui-check', 'ui-build'],
                    'test_modules': list(test_modules or DEFAULT_TESTS),
                    'physical_pi_measured': False, 'live_microphone_measured': False}
        record['software_verification'] = software
        if not skip_train and not from_scratch:
            software_path = root / HERE / 'software-verification.json'
            if software_path.exists() and read(software_path) != software:
                raise ValueError('Existing software acceptance evidence differs; preserve it and use a new study version')
            save(software_path, software)
        if not from_scratch:
            stage('report-final-evidence', [python, '-m', 'scripts.iris_report'])
        stage('verify-after-report', action=lambda: verify_bindings(root, immutable))
        if not skip_train:
            stage('freeze-selected-release', action=lambda: freeze_release(root=root, input_receipt=input_receipt,
                                                                          output=release_path, **options))
        release = stage('verify-final-lineage', action=lambda: validate_release(root, release_path))
        release_sha = sha(release_path)
        if packager is None:
            from scripts.mvp_delivery import package_release
            packager = package_release
        output = root / 'dist' / EXPERIMENT / token / 'iris-pi.zip'
        output.parent.mkdir(parents=True, exist_ok=True)
        extras = {'evidence/reproduction-release.json': release_path.read_bytes(),
                  'evidence/reproduction-inputs.json': input_receipt.read_bytes()}
        for name, member in (('MODEL_CARD.md', 'MODEL_CARD.md'), ('docs/DATASETS.md', 'docs/DATASETS.md'),
                             ('docs/HANDOFF-iris-rpi-v1.md', 'docs/HANDOFF-iris-rpi-v1.md'),
                             ('docs/EXPERIMENT-iris-release-v1.md', 'docs/EXPERIMENT-iris-release-v1.md')):
            if name in release['bindings'] and not (name == 'MODEL_CARD.md' and from_scratch):
                extras[member] = relative_path(root, name).read_bytes()
        for name in release.get('evaluation_receipts', []):
            extras['evidence/' + Path(name).name] = relative_path(root, name).read_bytes()
        for role, member in (('decision_parity', 'evidence/current-host-decision-parity.json'),
                             ('resource_measurement', 'evidence/current-host-pipeline-resources.json')):
            if role in record:
                extras[member] = relative_path(root, record[role]['path']).read_bytes()
        package = stage('package', action=lambda: packager(relative_path(root, release['bundle']), output,
                         wake_folder=relative_path(root, release['wake_folder']),
                         owner_folder=relative_path(root, release['owner_folder']),
                         vad_folder=relative_path(root, release['vad_folder']), extra_files=extras))
        record['package'] = {**package, 'path': name_under(root, output), 'sha256': sha(output), 'bytes': output.stat().st_size}
        verification = stage('verify-package', action=lambda: verify_package(output))
        record['package_verification'] = {k: v for k, v in verification.items() if k != 'manifest'}
        record['release'] = name_under(root, release_path)
        record['release_sha256'] = release_sha
        if publish:
            if publisher is None:
                from scripts.iris_publish import publish_workflow
                publisher = publish_workflow
            record['publication'] = stage('publish', action=lambda: publisher(root=root, workflow=record))
        record['status'] = 'complete'
        persist()
        return record
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--from-scratch', action='store_true')
    parser.add_argument('--fit-from-scratch', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--input-receipt', type=Path)
    parser.add_argument('--release-receipt', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    recipe_sources = ROOT / 'experiments/iris-reproduction-v1/recipe-sources.json'
    if recipe_sources.is_file() and not args.fit_from_scratch:
        from scripts.iris_reproduce import hydrate_recipe_sources
        hydrate_recipe_sources(ROOT, read(recipe_sources), download=not args.no_download)
    if args.fit_from_scratch:
        if not args.run_id:
            parser.error('--fit-from-scratch needs --run-id')
        result = run_from_scratch(run_id=args.run_id, device=args.device)
        print(json.dumps({'status': result['status'], 'bundle': result['bundle']}), flush=True)
        return result
    result = run_workflow(skip_train=args.skip_train, publish=args.publish, from_scratch=args.from_scratch,
                          preflight_only=args.preflight_only, download=not args.no_download,
                          input_receipt=args.input_receipt, release_path=args.release_receipt,
                          run_id=args.run_id, device=args.device)
    print(json.dumps({k: result[k] for k in ('status', 'receipt', 'training_mode')}), flush=True)


if __name__ == '__main__':
    main()
