"""Git-owned recipe; isolated continuation fits and selected-artifact comparison.

The source checkout stays unchanged. Each run uses its own ignored workspace and
data/model cache, sharing only this clone's bootstrapped Python environment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
import uuid
import zipfile

from scripts.iris_release import ROOT, read, relative_path, save, sha, verify_bindings

HERE = 'experiments/iris-reproduction-v1'
CACHE = '.cache/iris-reproduction-v1'


def training_inputs(base):
    """Keep project initialization, never hydrate a completed candidate fit."""
    result = deepcopy(base)
    omitted = {item['path'] for item in base['files'] if (
        item.get('role') in ('vcm_completed_run', 'vcm_candidate', 'vcm_selected')
        or '/completed-stages/diversity/official-results/' in item['path']
        or '/completed-stages/diversity/fallback-official-results/' in item['path'])}
    result['files'] = [item for item in result['files'] if item['path'] not in omitted]
    result['cache_aliases'] = [item for item in result.get('cache_aliases', []) if item['source'] not in omitted]
    result['force_vcm_fitting'] = True
    result['reproduction'] = {'experiment': 'iris-reproduction-v1',
        'completed_vcm_inputs_omitted': len(omitted), 'recipe_repository': 'https://github.com/justinruaya123/smart-speaker',
        'training': 'actual matched continuation fits; verified project initializers and frozen fallback retained'}
    return result


def hydrate_recipe_sources(root, mapping, *, fetch=None, download=True):
    """Recipe code comes from this Git checkout or hash-pinned upstream GitHub."""
    root = Path(root)
    def get(url):
        with urlopen(url, timeout=90) as response:
            return response.read()
    fetch = fetch or get
    def restore(item):
        name = item['input_path']
        if not name.startswith('.cache/iris-release-v1/inputs/'):
            raise ValueError('Recipe destination must be in the inputs cache')
        target = relative_path(root, name)
        if target.exists():
            if not target.is_file() or sha(target) != item['sha256'] or target.stat().st_size != item['bytes']:
                raise ValueError('Recipe cache identity mismatch: ' + name)
            return name
        kind = item['source_kind']
        if kind == 'git':
            source = relative_path(root, item['source_path'])
            verify_bindings(root, {item['source_path']: item['sha256']})
            payload = source.read_bytes()
        elif kind == 'upstream_github':
            if not download:
                raise ValueError('Missing upstream recipe source with --no-download: ' + name)
            upstream = item['upstream']
            revision, url = upstream.get('revision', ''), upstream.get('raw_url', '')
            if not re.fullmatch('[0-9a-f]{40}', revision) or not re.fullmatch(
                    r'https://raw\.githubusercontent\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/' + revision + r'/[^?#]+', url):
                raise ValueError('Recipe source must use immutable upstream GitHub bytes')
            payload = fetch(url)
        else:
            raise ValueError('Unknown recipe source kind: ' + kind)
        import hashlib
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256']:
            raise ValueError('Recipe source identity mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
        return name
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(restore, mapping['files']))


def compare_bundle(reference, archive):
    """Compare the actual deployment layout, including generated VAD metadata."""
    import hashlib
    reference = Path(reference)
    result = {}
    with zipfile.ZipFile(archive) as package:
        for name, item in read(reference / 'integrity.json')['files'].items():
            verify_bindings(reference, {name: item['sha256']})
            actual_sha = hashlib.sha256(package.read('models/default/' + name)).hexdigest()
            result[name] = {'expected_sha256': item['sha256'], 'actual_sha256': actual_sha,
                            'matches': item['sha256'] == actual_sha}
    return result


def preserve_derived_results(workspace, recovery, names):
    """Only the new run's copies of explicitly named generated results change."""
    workspace, recovery = Path(workspace), Path(recovery)
    bindings = {}
    for name in names:
        source = relative_path(workspace, name)
        if source.is_file():
            target = relative_path(recovery, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            bindings[name] = sha(source)
            if sha(target) != bindings[name]:
                raise ValueError('Published result recovery identity mismatch: ' + name)
            source.unlink()
    save(recovery / 'recovery.json', bindings)
    return bindings


def snapshot_sources(root, workspace):
    """Copy committed source bytes; never copy ignored caches or local config."""
    root, workspace = Path(root), Path(workspace)
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--'], cwd=root, check=True,
                   stdout=subprocess.DEVNULL)
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    bindings = {}
    for name in filter(None, files):
        source = relative_path(root, name)
        if source.is_symlink() or not source.is_file():
            raise ValueError('Tracked source must be a regular file: ' + name)
        target = relative_path(workspace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        bindings[name] = sha(source)
    if not (workspace / 'reproduce.py').is_file():
        raise ValueError('Commit the reproduction entry point before snapshotting')
    return bindings


def run(*, root=ROOT, run_id=None, device='cpu', skip_train=False, preflight_only=False,
        download=True, publish=False):
    root = Path(root).absolute()
    if (root / '.cache').is_symlink():
        raise ValueError('Use a fresh Git clone with its own .cache directory for reproduction')
    token = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}', token):
        raise ValueError('Run ID must be a bounded simple directory name')
    protocol_path = root / HERE / 'protocol.json'
    protocol = read(protocol_path)
    verify_bindings(root, {protocol['base_inputs']: protocol['base_inputs_sha256']})
    verify_bindings(root, {'artifacts/iris/model.onnx': protocol['reference_vcm_onnx_sha256']})
    directory = root / CACHE / token
    if directory.exists():
        raise FileExistsError('Preserve the existing reproduction run; choose a new --run-id')
    directory.mkdir(parents=True)
    workspace = directory / 'work'
    began = time.perf_counter()
    record = {'schema_version': 1, 'experiment': 'iris-reproduction-v1', 'run_id': token,
              'status': 'running', 'device': device, 'host': {'platform': platform.platform(),
                  'python': platform.python_version()}, 'skip_train': skip_train,
              'physical_pi_measured': False, 'original_human_wake_pairs_measured': False,
              'protocol_sha256': sha(protocol_path), 'workspace': workspace.relative_to(root).as_posix()}
    bootstrap = root / '.cache/iris-release-v1/environment/bootstrap.json'
    if bootstrap.is_file():
        record['environment_bootstrap'] = {**read(bootstrap), 'receipt_sha256': sha(bootstrap)}
    output = directory / 'reproduction.json'
    save(output, record)
    try:
        record['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        record['source_tree'] = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=root, text=True).strip()
        bindings = snapshot_sources(root, workspace)
        save(directory / 'source-hashes.json', bindings)
        record['source_hashes_sha256'] = sha(directory / 'source-hashes.json')
        # One environment for this clone. No links to any data or model caches.
        (workspace / '.venv').symlink_to(root / '.venv', target_is_directory=True)
        base = read(workspace / protocol['base_inputs'])
        mapping = read(workspace / protocol['recipe_sources'])
        inventory = {item['path']: item for item in base['files']}
        for item in mapping['files']:
            original = inventory[item['input_path']]
            if (original['sha256'], original['bytes']) != (item['sha256'], item['bytes']):
                raise ValueError('Recipe mapping disagrees with frozen input receipt')
        hydrate_recipe_sources(workspace, mapping, download=download)
        inputs = deepcopy(base) if skip_train else training_inputs(base)
        inputs['lineage'] = {**inputs.get('lineage', {}),
            'canonical_source_snapshot': 'GitHub checkout',
            'canonical_source_repository': 'https://github.com/justinruaya123/smart-speaker',
            'canonical_source_revision': record['source_commit']}
        for item in inputs['files']:
            if item['path'] in {entry['input_path'] for entry in mapping['files']}:
                item.pop('remote', None)
                item.pop('publication', None)
                item['recipe_source'] = 'GitHub; see ' + protocol['recipe_sources']
        extra_sources = ['reproduce.py', 'scripts/iris_reproduce.py', HERE + '/protocol.json',
                         HERE + '/recipe-sources.json', 'scripts/test/test_iris_reproduce.py']
        inputs['source_files'] = sorted(set(inputs['source_files']) | set(extra_sources))
        if not skip_train:
            # This evidence is created only after the current software checks.
            # It must not be required by the snapshot taken before those checks.
            inputs['source_files'] = [name for name in inputs['source_files']
                if name != 'experiments/iris-release-v1/software-verification.json']
        input_path = workspace / '.cache/iris-release-v1/reproduction-github-inputs.json'
        save(input_path, inputs)
        record['input_receipt_sha256'] = sha(input_path)
        record['completed_vcm_inputs_excluded'] = inputs.get('reproduction', {}).get('completed_vcm_inputs_omitted', 0)
        if not skip_train:
            record['published_results_preserved'] = preserve_derived_results(
                workspace, directory / 'published-reference', protocol['derived_files'])
        command = [str(root / '.venv/bin/python'), '-u', '-m', 'scripts.iris_release',
                   '--device', device, '--run-id', token,
                   '--input-receipt', input_path.relative_to(workspace).as_posix()]
        if os.name == 'nt':
            command[0] = str(root / '.venv/Scripts/python.exe')
        for flag, enabled in (('--skip-train', skip_train), ('--preflight-only', preflight_only),
                              ('--no-download', not download), ('--publish', publish)):
            if enabled:
                command.append(flag)
        record['command'] = command
        save(output, record)
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'HF_HUB_CACHE': str(directory / 'hub'),
                       'HF_XET_CACHE': str(directory / 'xet'), 'HF_DATASETS_CACHE': str(directory / 'datasets'),
                       'TORCH_HOME': str(directory / 'torch'),
                       'IRIS_REPRODUCTION_INPUTS': input_path.relative_to(workspace).as_posix()}
        subprocess.run(command, cwd=workspace, env=environment, check=True)
        workflow = read(workspace / 'experiments/iris-release-v1' / ('reproduction-run-' + token + '.json'))
        record['workflow'] = workflow
        if workflow['status'] == 'complete':
            release = read(workspace / workflow['release'])
            reference = root / 'artifacts/iris'
            comparison = compare_bundle(reference, workspace / workflow['package']['path'])
            record['inference_comparison'] = comparison
            record['matches_shipped_vcm'] = all(item['matches'] for name, item in comparison.items()
                if not name.startswith(('wake/', 'owner/')))
            record['matches_complete_inference_bundle'] = all(item['matches'] for item in comparison.values())
            record['promotion'] = read(workspace / 'experiments/iris-release-v1/release-choice.json')
            record['training_runs'] = [read(path) for path in sorted((workspace / '.cache/iris-release-v1/diversity/runs').glob('*/selection.json'))]
            record['package'] = (workspace / workflow['package']['path']).relative_to(root).as_posix()
        record['status'] = workflow['status']
        verify_bindings(root, bindings)
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - began
        record['bootstrap_plus_run_seconds'] = (record['elapsed_seconds']
            + record.get('environment_bootstrap', {}).get('elapsed_seconds', 0))
        save(output, record)
        print(json.dumps({'status': record['status'], 'receipt': output.relative_to(root).as_posix()}), flush=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--run-id')
    parser.add_argument('--skip-train', action='store_true', help='Verify selected artifacts; explicitly no new VCM fits')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--publish', action='store_true', help='Publish verified artifacts privately; Git remains canonical source')
    args = parser.parse_args(argv)
    run(device=args.device, run_id=args.run_id, skip_train=args.skip_train,
        preflight_only=args.preflight_only, download=not args.no_download, publish=args.publish)
    return 0
