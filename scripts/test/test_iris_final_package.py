"""Final source-layout package boundaries; no fitting, capture or providers."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]

import pytest


def package_module():
    assert importlib.util.find_spec('scripts.iris_final_package') is not None, 'Final safe package builder is missing'
    from scripts import iris_final_package
    return iris_final_package


def test_package_scope_rejects_private_or_unapproved_payloads():
    package = package_module()
    package.validate_entries({'inference.py': b'pass\n', 'submission/feedback/intent.wav': b'feedback'}, target='inference')
    for name in (
        '../escape', '.env', 'config.toml', '.cache/finished-fit/selected.pt',
        '.venv/Scripts/python.exe', 'data/audio/person.wav', 'SpeakerUI/node_modules/private.js',
        'artifacts/iris/model.onnx', 'submission/checkpoints/optionb-tiny-selected.pt',
        'experiments/iris-openwakeword-publication-v1/embedding_model.onnx',
        'submission/feedback/recording.wav', 'scripts/train.py', 'SpeakerUI/dist/.env',
    ):
        with pytest.raises(ValueError):
            package.validate_entries({name: b'PRIVATE'}, target='inference')


def test_build_uses_single_writer_and_records_source_closure(tmp_path, monkeypatch):
    package = package_module()
    from scripts import archive_io
    calls = []
    original = archive_io.write_archive

    def observed(entries, output, metadata):
        calls.append((set(entries), metadata))
        return original(entries, output, metadata)

    monkeypatch.setattr(archive_io, 'write_archive', observed)
    artifact = package.build_archive(ROOT, tmp_path/'benchmark.zip', target='benchmark', allow_dirty=True)
    assert len(calls) == 1
    with zipfile.ZipFile(artifact['path']) as archive:
        names = set(archive.namelist())
        required = {
            'inference.py', 'benchmark.py', 'scripts/runtime/pyproject.toml', 'scripts/runtime/uv.lock',
            'scripts/runtime/speaker_worker.py', 'vcm/frontend.py', 'SOURCE-CLOSURE.json',
            'deployment.json', 'config.example.toml', 'RESEARCH-STATUS.json',
            'scripts/runtime/iris_openwakeword.py', 'artifacts/silero-LICENSE.txt',
            'experiments/iris-openwakeword-publication-v1/findings.json',
            'experiments/iris-openwakeword-publication-v1/LICENSE-openWakeWord-Apache-2.0.txt',
            'artifacts/iris/wake/iris.ncnn.bin', 'artifacts/iris/wake/iris.ncnn.param',
            'artifacts/iris/wake/silero-vad.onnx', 'artifacts/iris/owner/owner.onnx',
            'experiments/iris-publication-v1/desktop-product-20261024.json',
        }
        assert required <= names
        assert set(package.MODEL_FILES) <= names
        assert set(package.FEEDBACK_FILES) <= names
        assert not any(n.startswith('SpeakerUI/') for n in names)
        deployment = json.loads(archive.read('deployment.json'))
        assert deployment['actions_enabled'] is False
        assert deployment['qualified'] is False
        assert deployment['entrypoint'] == 'benchmark.py'
        assert deployment['wake_backend'] == 'native_experimental_failed_gate'
        status = json.loads(archive.read('RESEARCH-STATUS.json'))
        assert status['openwakeword']['installed'] is False
        assert status['openwakeword']['decision'] == 'DO_NOT_PROMOTE'
        closure = json.loads(archive.read('SOURCE-CLOSURE.json'))
        assert closure['source_revision']
        assert closure['source_is_clean'] == package.git_state(ROOT)['source_is_clean']
        assert closure['verification_only'] is (not closure['source_is_clean'])
        assert closure['files']['inference.py'] == hashlib.sha256(archive.read('inference.py')).hexdigest()
        assert json.loads(archive.read('MANIFEST.json'))['actions_enabled'] is False
        assert 'enabled = false' in archive.read('config.example.toml').decode()
    before = Path(artifact['path']).read_bytes()
    with pytest.raises(FileExistsError):
        package.build_archive(ROOT, tmp_path/'benchmark.zip', target='benchmark', allow_dirty=True)
    assert Path(artifact['path']).read_bytes() == before


def test_inference_requires_compiled_ui_and_binds_build_source(tmp_path):
    package = package_module()
    entries = package.assemble_files(ROOT, target='inference', allow_dirty=True)
    assert 'SpeakerUI/dist/index.html' in entries
    assert 'SpeakerUI/dist-server/server/main.js' in entries
    assert 'SpeakerUI/server/inference/iris-publication.test.ts' in entries
    assert entries['SpeakerUI/pnpm-lock.yaml'] == (ROOT/'SpeakerUI/pnpm-lock.yaml').read_bytes()
    assert 'SpeakerUI/mockups/ambient-v6/licenses/AURA-NOTICE.txt' in entries
    closure = json.loads(entries['SOURCE-CLOSURE.json'])
    assert closure['ui_build']['lock_sha256'] == hashlib.sha256(entries['SpeakerUI/pnpm-lock.yaml']).hexdigest()
    assert closure['ui_build']['source_sha256']
    assert closure['ui_build']['compiled_sha256']


def test_benchmark_evidence_refuses_raw_payload_and_missing_receipt(tmp_path):
    package = package_module()
    with pytest.raises((ValueError, FileNotFoundError)):
        package.benchmark_evidence(tmp_path)
    (tmp_path/'verification.json').write_text(json.dumps({
        'output_files': {'../person.wav': {'sha256': '0'*64, 'bytes': 3}},
    }))
    with pytest.raises(ValueError, match='evidence'):
        package.benchmark_evidence(tmp_path)


def test_cli_refuses_existing_output_and_dirty_publication(tmp_path):
    package = package_module()
    assert callable(getattr(package, 'main', None)), 'Final rebuild CLI is missing'
    with pytest.raises(SystemExit) as existing:
        package.main(['--output-dir', str(tmp_path), '--benchmark-evidence', str(tmp_path)])
    assert existing.value.code == 2
    # A clean checkout is a valid future publication input; this case characterizes
    # the current dirty verification tree without requiring unrelated edits.
    if not package.git_state(ROOT)['source_is_clean']:
        with pytest.raises(ValueError, match='Commit final source'):
            package.assemble_files(ROOT, target='benchmark')


def test_cli_freezes_clean_source_before_nonignored_in_repo_output(tmp_path, monkeypatch):
    package = package_module()
    from scripts import archive_io
    from scripts.runtime import iris_epoch44

    # Bound the source closure, keeping the real frozen-model and Git checks.
    monkeypatch.setattr(package, 'RUNTIME_FILES', ('inference.py', 'benchmark.py'))
    for name in ('COMPANION_FILES', 'FEEDBACK_FILES', 'EVIDENCE_FILES', 'DOC_FILES',
                 'NOTICE_FILES', 'UI_NOTICE_FILES', 'UI_SOURCE_PREFIXES'):
        monkeypatch.setattr(package, name, ())
    monkeypatch.setattr(package, 'OWW_FILES', (
        'experiments/iris-openwakeword-publication-v1/findings.json',))
    monkeypatch.setattr(package, 'UI_ROOT_FILES', ('SpeakerUI/pnpm-lock.yaml',))
    root = tmp_path/'source'
    # Reuse existing history, not new commits or edits to the integration tree.
    subprocess.run(['git', 'clone', '--quiet', '--no-local', '--depth=1',
                    '--no-checkout', '--sparse', str(ROOT), str(root)], check=True)
    names = ('.gitattributes', '.gitignore', 'SpeakerUI/.gitignore',
             *package.RUNTIME_FILES, *package.MODEL_FILES, *package.OWW_FILES,
             *package.UI_ROOT_FILES)
    subprocess.run(['git', '-C', str(root), 'sparse-checkout', 'set', '--no-cone', '--stdin'],
                   input=''.join('/' + name + '\n' for name in names), text=True, check=True)
    subprocess.run(['git', '-C', str(root), 'checkout', '--quiet', '--detach', 'HEAD'], check=True)
    for name in ('SpeakerUI/dist/index.html', 'SpeakerUI/dist-server/server/main.js'):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'/* inert packaging fixture; never executed */\n')
    receipt = dict(package.ui_binding(root), fixture_only=True,
                   tasks_passed=['check', 'test', 'build'], node='v24.13.0', pnpm='12.5.1')
    path = root/package.UI_BUILD_RECEIPT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(package.json_bytes(receipt))
    evidence = tmp_path/'evidence'
    evidence.mkdir()
    (evidence/'verification.json').write_bytes(b'{"fixture_only": true}\n')
    # No acoustic rows: the evidence validator has its own independent coverage.
    monkeypatch.setattr(package, 'benchmark_evidence', lambda directory: {
        package.FULL_EVIDENCE_PREFIX + 'verification.json':
            package.read_source(directory, 'verification.json')})
    monkeypatch.setattr(package, 'ROOT', root)
    source_state = package.git_state(root)
    assert source_state['source_is_clean'] is True
    output = root/'release-pair'
    assert not output.exists()
    assert subprocess.run(['git', '-C', str(root), 'check-ignore',
                           str(output/'iris-final-inference.zip')], capture_output=True).returncode == 1

    def forbidden(*args, **kwargs):
        pytest.fail('Packaging must not install, capture, call providers or fit')

    original_popen, original_import = subprocess.Popen, builtins.__import__

    def read_only_git(command, *args, **kwargs):
        if (kwargs.get('shell') or list(command[:3]) != ['git', '-C', str(root)]
                or command[3] not in ('rev-parse', 'status', 'ls-files')):
            forbidden()
        return original_popen(command, *args, **kwargs)

    def safe_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'onnxruntime', 'sounddevice'}:
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', read_only_git)
    monkeypatch.setattr(builtins, '__import__', safe_import)
    monkeypatch.setattr(socket, 'socket', forbidden)
    monkeypatch.setattr(package, 'build_ui', forbidden)
    monkeypatch.setattr(iris_epoch44, 'Runtime', forbidden)
    original_read, original_state = package.read_source, package.git_state
    late_reads = []

    def observed_read(source_root, name):
        if output.exists():
            late_reads.append(name)
        return original_read(source_root, name)

    def observed_state(source_root):
        if output.exists():
            late_reads.append('git_state')
        return original_state(source_root)

    monkeypatch.setattr(package, 'read_source', observed_read)
    monkeypatch.setattr(package, 'git_state', observed_state)
    original_writer = archive_io.write_archive
    written = []

    def observed_writer(entries, destination, metadata):
        written.append(metadata['target'])
        return original_writer(entries, destination, metadata)

    monkeypatch.setattr(archive_io, 'write_archive', observed_writer)
    argv = ['--output-dir', str(output), '--benchmark-evidence', str(evidence)]
    assert package.main(argv) == 0  # Deliberately no --allow-dirty.
    assert late_reads == []
    assert written == ['inference', 'benchmark']
    assert original_state(root)['source_is_clean'] is False  # Artifacts are not ignored.
    expected_names = {'iris-final-inference.zip', 'iris-final-benchmark.zip',
                      'BUILD-REPORT.json', 'SHA256SUMS.txt'}
    assert {path.name for path in output.iterdir()} == expected_names
    immutable = {path.name: path.read_bytes() for path in output.iterdir()}
    report = json.loads(immutable['BUILD-REPORT.json'])
    identity = dict(source_state, verification_only=False)
    assert {key: report[key] for key in identity} == identity
    assert report['actions_enabled'] is False and report['qualified'] is False
    assert len(report['artifacts']) == 2
    sums = []
    for target, artifact in zip(written, report['artifacts']):
        name = f'iris-final-{target}.zip'
        assert artifact['path'] == str(output/name)
        assert artifact['bytes'] == len(immutable[name])
        assert artifact['sha256'] == package.sha(immutable[name])
        sums.append(f"{artifact['sha256']}  {name}\n")
        with zipfile.ZipFile(output/name) as archive:
            assert archive.testzip() is None
            closure = json.loads(archive.read('SOURCE-CLOSURE.json'))
            assert {key: closure[key] for key in identity} == identity
            manifest = json.loads(archive.read('MANIFEST.json'))
            assert manifest['source'] == closure
            assert set(archive.namelist()) == set(manifest['files']) | {'MANIFEST.json'}
            for path, digest in manifest['files'].items():
                assert package.sha(archive.read(path)) == digest
            assert closure['files'] == {path: digest for path, digest in manifest['files'].items()
                                        if path != 'SOURCE-CLOSURE.json'}
            for path in package.RUNTIME_FILES + package.MODEL_FILES + package.OWW_FILES:
                assert closure['files'][path] == package.sha(original_read(root, path))
    assert (output/'SHA256SUMS.txt').read_text(encoding='utf-8') == ''.join(sums)
    with pytest.raises(SystemExit) as existing:
        package.main(argv)
    assert existing.value.code == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == immutable


def test_benchmark_population_instructions_are_explicit():
    package = package_module()
    entries = package.assemble_files(ROOT, target='benchmark', allow_dirty=True)
    protocol = json.loads(entries['BENCHMARK-PROTOCOL.json'])
    assert protocol['full_reference']['original_recordings'] == 4614
    assert protocol['full_reference']['eligible_recordings'] == 4427
    assert protocol['full_reference']['ambiguous_unscored'] == 187
    assert protocol['adapter_human']['original_recordings'] == 1146
    assert protocol['adapter_human']['eligible_recordings'] == 959
    assert protocol['repeats_increase_accuracy_denominators'] is False
    assert protocol['qualified'] is False
    assert protocol['actions_enabled'] is False


def test_linked_source_is_never_packaged(tmp_path):
    package = package_module()
    outside = tmp_path/'private'
    outside.mkdir()
    (outside/'secret.py').write_text('PRIVATE')
    root = tmp_path/'source'
    root.mkdir()
    try:
        (root/'scripts').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit creating directory links')
    with pytest.raises(ValueError, match='Linked package input'):
        package.read_source(root, 'scripts/secret.py')
