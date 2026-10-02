"""Bound, inspectable release workflow; physical Pi and microphone runs stay manual.

No stage falls back to another model. --skip-train requires the frozen checkpoint,
protocol and source receipt, reruns software validation, and republishes only that
bound release. Normal runs let immutable stage implementations resume unchanged
completed work. Each invocation writes a unique workflow receipt and stage logs.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import uuid
import zipfile
from scripts.pcv_atomic_io import replace_with_retry

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS = tuple('scripts/test/' + x + '.py' for x in (
    'test_pcv_release_data', 'test_pcv_release_prepare', 'test_pcv_release_training',
    'test_pcv_release_pooling', 'test_pcv_release_qualification',
    'test_pcv_final_delivery', 'test_pcv_release_workflow'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolved(root, value):
    path = Path(str(value).replace('\\','/'))
    return (path if path.is_absolute() else root / path).resolve()


def verify_bindings(root, receipt):
    bindings = receipt.get('bindings')
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Receipt must declare nonempty file bindings')
    checked = {}
    for name, expected in bindings.items():
        file = resolved(root, name)
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid SHA256 binding: ' + str(name))
        if not file.is_file() or sha(file) != expected:
            raise ValueError('File identity mismatch: ' + str(file))
        checked[file] = expected
    return checked


def validate_selection(root, path):
    receipt = read(path)
    checked = verify_bindings(root, receipt)
    for role in ('checkpoint', 'protocol', 'source_receipt'):
        if not receipt.get(role) or resolved(root, receipt[role]) not in checked:
            raise ValueError('Selection must bind ' + role)
    return receipt


def validate_preparation(root, path):
    receipt=read(path)
    verify_bindings(root,receipt)
    verify_bindings(root,{'bindings':{r['path']:r['sha256'] for r in receipt['manifests'].values()}})
    return receipt


def validate_release(root, path, selection_path):
    release = read(path)
    if type(release.get('qualified')) is not bool:
        raise ValueError('Release qualified must be a boolean')
    checked = verify_bindings(root, release)
    if selection_path.resolve() not in checked:
        raise ValueError('Release must bind selection-before-calibration receipt')
    for role in ('summary', 'sample_manifest', 'source_receipt', 'source_card'):
        if not release.get(role) or resolved(root, release[role]) not in checked:
            raise ValueError('Release must bind ' + role)
    results = read(resolved(root, release['summary'])).get('raw_results')
    if results is not None:
        if not isinstance(results, str) or not results or resolved(root, results) not in checked:
            raise ValueError('Detailed model results must be bound by the release receipt')
    bundle = resolved(root, release['bundle'])
    if not bundle.is_dir() or not any(p.is_relative_to(bundle) for p in checked):
        raise ValueError('Release must bind its bundle files')
    selected = validate_selection(root, selection_path)
    if resolved(root, selected['source_receipt']) != resolved(root, release['source_receipt']):
        raise ValueError('Release source differs from selected training source')
    return release


def verify_extract_package(package, destination):
    """Verify every member before extracting into a new private replay directory."""
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate package member')
        manifest = json.loads(archive.read('MANIFEST.json'))
        declared = manifest['files']
        if set(names) != set(declared) | {'MANIFEST.json'}:
            raise ValueError('Package member inventory differs from manifest')
        for name in names:
            member = PurePosixPath(name)
            target = destination / name
            if member.is_absolute() or '..' in member.parts or ':' in name or '\\' in name or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe package member: ' + name)
            if member.suffix.lower() in ('.pt', '.pth', '.ckpt', '.safetensors'):
                raise ValueError('Training checkpoint cannot ship: ' + name)
            if name != 'MANIFEST.json' and hashlib.sha256(archive.read(name)).hexdigest() != declared[name]:
                raise ValueError('Package identity mismatch: ' + name)
        archive.extractall(destination)
    return manifest


def run_workflow(*, root=ROOT, run_id='release-v1', skip_train=False, python=None,
                 node=None, test_modules=None, runner=subprocess.run, packager=None, release_only=False):
    root = Path(root).resolve()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', run_id):
        raise ValueError('Run ID must be a bounded simple directory name')
    python = str(python or root / ('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python'))
    node = str(node or os.environ.get('RUNTIME_NODE') or Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:6]
    ledger = root / 'experiments/pcv-release-train-v1/workflow' / token
    ledger.mkdir(parents=True, exist_ok=False)
    logs = root / '.cache/pcv-release-v1/workflow' / token
    logs.mkdir(parents=True, exist_ok=False)
    receipt_path = ledger / 'workflow.json'
    run = root / 'experiments/pcv-release-train-v1/runs' / run_id
    selection_path = run / 'selection-before-calibration.json'
    release_path = run / 'release.json'
    existed = selection_path.exists()
    record = {'schema_version': 1, 'run_id': run_id, 'invocation': token,
              'status': 'running', 'skip_train': skip_train, 'physical_pi_measured': False,
              'training_mode': 'skipped_bound_selection' if skip_train else ('resume_or_verify' if existed else 'train'),
              'stages': [], 'receipt': str(receipt_path)}

    def save():
        temporary = receipt_path.with_suffix('.pending')
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        replace_with_retry(temporary, receipt_path)

    def stage(name, command=None, action=None):
        item = {'name': name, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat()}
        record['stages'].append(item)
        if command: item['command'] = [str(x) for x in command]
        save()
        print('RELEASE_STAGE', name, flush=True)
        try:
            if command:
                log = logs / (name + '.log')
                item['log'] = str(log)
                with log.open('w', encoding='utf-8') as stream:
                    outcome = runner(item['command'], cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
                if outcome.returncode:
                    raise subprocess.CalledProcessError(outcome.returncode, command)
                item['returncode'] = outcome.returncode
                result = None
            else:
                result = action()
            item['status'] = 'passed'
            return result
        except Exception as error:
            item.update(status='failed', error=f'{type(error).__name__}: {error}')
            if isinstance(error, subprocess.CalledProcessError): item['returncode'] = error.returncode
            raise
        finally:
            item['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()

    save()
    try:
        if skip_train:
            stage('verify-frozen-selection', action=lambda: validate_selection(root, selection_path))
            record['source_refresh'] = 'Skipped: preserve the training-bound revision; no current API data rebound'
        else:
            if existed: stage('verify-resume-selection', action=lambda: validate_selection(root, selection_path))
            stage('source-refresh', [python, '-m', 'scripts.train.pcv_release_data'])
            preparation=root/'experiments/pcv-release-train-v1/data-freeze.json'
            if preparation.exists():
                stage('verify-prepared-source', action=lambda:validate_preparation(root,preparation))
            else:
                stage('prepare', [python, '-m', 'scripts.train.pcv_release_prepare'])
            stage('train-or-resume', [python, '-m', 'scripts.train.pcv_release_training', '--run-id', run_id])
            stage('verify-trained-selection', action=lambda: validate_selection(root, selection_path))
        selection_hash = sha(selection_path)
        modules = list(DEFAULT_TESTS if test_modules is None else test_modules)
        if not modules: raise ValueError('Software test list cannot be empty')
        stage('software-tests', [python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
              '--basetemp', str(logs / 'pytest'), *modules])
        stage('qualification-or-locked-replay', [python, '-m', 'scripts.test.pcv_release_qualification', '--run-id', run_id])
        if sha(selection_path) != selection_hash: raise ValueError('Qualification changed the frozen selection identity')
        release = stage('verify-release', action=lambda: validate_release(root, release_path, selection_path))
        record.update(qualified=release['qualified'], source_receipt=str(resolved(root, release['source_receipt'])),
                      selection_sha256=selection_hash, release_sha256=sha(release_path))
        if release_only:
            record.update(status='qualified_complete' if release['qualified'] else 'diagnostic_complete',release=str(release_path))
            save()
            return record
        def locked_release():
            if sha(selection_path) != selection_hash or sha(release_path) != record['release_sha256']:
                raise ValueError('Frozen release identity changed during publication')
            return validate_release(root, release_path, selection_path)
        output = root / 'dist/pcv-release-v1' / token
        output.mkdir(parents=True, exist_ok=False)
        summary = resolved(root, release['summary'])
        slide_name = 'me2-release' if release['qualified'] else 'me2-diagnostic'
        stage('deck', [node, str(root / 'scripts/report/pcv_release_slides.mjs'), '--summary', str(summary),
                      '--output', str(output), '--name', slide_name])
        deck = output / (slide_name + '.pptx')
        if not deck.is_file(): raise ValueError('Deck stage did not create its expected output')
        pdf = output / (slide_name + '.pdf')
        if not pdf.is_file(): raise ValueError('Deck stage did not create its PDF companion')
        selected = validate_selection(root, selection_path)
        reproduction = ledger / 'reproduction.json'
        reproduction.write_text(json.dumps({'run_id': run_id, 'training_command': [python, 'release.py', '--run-id', run_id],
            'validation_command': [python, 'release.py', '--run-id', run_id, '--skip-train'],
            'selection': selected, 'selection_sha256': selection_hash,
            'replay_source': 'Bound source receipt and original test split; never substitute a newer revision',
            'weights_policy': 'Only selected ONNX runtime graphs ship; fitting checkpoints stay outside the Pi ZIP',
            'physical_pi_measured': False}, indent=2) + '\n', encoding='utf-8')
        if packager is None:
            from scripts.pcv_delivery import package as packager
        zip_path = output / ('pcv-release.zip' if release['qualified'] else 'pcv-diagnostic.zip')
        extras = {f'slides/{deck.name}': deck, f'slides/{pdf.name}': pdf, 'evidence/release-summary.json': summary,
                  'evidence/reproduction.json': reproduction,
                  'evidence/selection-before-calibration.json': selection_path,
                  'evidence/training-protocol.json': resolved(root, selected['protocol'])}
        results = read(summary).get('raw_results')
        if results is not None:
            extras['evidence/model-results.json'] = resolved(root, results)
        qualification = resolved(root, release.get('qualification_receipt', str(release_path))) if release['qualified'] else None
        stage('verify-prepackage-bindings', action=locked_release)
        stage('package', action=lambda: packager(resolved(root, release['bundle']), zip_path,
            diagnostic=not release['qualified'], sample_manifest=resolved(root, release['sample_manifest']),
            source_receipt=resolved(root, release['source_receipt']), source_card=resolved(root, release['source_card']),
            qualification_receipt=qualification, extra_files=extras))
        extracted = logs / 'package-smoke'
        stage('package-integrity', action=lambda: verify_extract_package(zip_path, extracted))
        stage('packaged-verify', [python, str(extracted / 'pi_test.py'), '--verify-only'])
        stage('packaged-smoke-replay', [python, str(extracted / 'pi_test.py'), '--output', str(logs / 'replay')])
        stage('verify-final-bindings', action=locked_release)
        record.update(status='qualified_complete' if release['qualified'] else 'diagnostic_complete',
                      package=str(zip_path), package_sha256=sha(zip_path), deck=str(deck), deck_sha256=sha(deck),
                      pdf=str(pdf), pdf_sha256=sha(pdf),
                      reproduction=str(reproduction), replay_directory=str(logs / 'replay'))
        save()
        return record
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        print('RELEASE_FAILED_RECEIPT', receipt_path, file=sys.stderr)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-id', default='release-v1', help='Immutable fitting identity (default: release-v1)')
    parser.add_argument('--skip-train', action='store_true', help='Verify and reuse the bound selection; still test, validate, render, package and replay')
    parser.add_argument('--python', help='Configured root interpreter; defaults to .venv/Scripts/python.exe')
    parser.add_argument('--node', help='Bundled Node executable for editable slide generation')
    parser.add_argument('--test-module', action='append', help='Explicit scoped pytest path; repeat to replace the default module list')
    args = parser.parse_args(argv)
    result = run_workflow(run_id=args.run_id, skip_train=args.skip_train, python=args.python,
                          node=args.node, test_modules=args.test_module)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
