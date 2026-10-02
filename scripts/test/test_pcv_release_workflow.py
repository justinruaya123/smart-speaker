"""Workflow contracts. External fitting, render and package boundaries are stubbed."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if isinstance(data, dict) else data)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root, qualified=False):
    base = root / 'experiments/pcv-release-train-v1/runs/release-v1'
    files = {key: write(root / name, text) for key, name, text in [
        ('checkpoint', '.cache/model/selected.pt', 'checkpoint'),
        ('protocol', 'experiments/train/protocol.json', '{}'),
        ('source_receipt', 'experiments/source/acquisition-complete.json', '{}'),
        ('source_card', 'experiments/source/source-card.txt', 'license'),
        ('sample_manifest', 'experiments/prepared/test.jsonl', '{}\n'),
        ('summary', 'experiments/train/summary.json', '{}'),
    ]}
    relative = lambda p: p.relative_to(root).as_posix()
    selection = {'checkpoint': relative(files['checkpoint']), 'protocol': relative(files['protocol']),
                 'source_receipt': relative(files['source_receipt']),
                 'bindings': {relative(files[k]): digest(files[k]) for k in ('checkpoint', 'protocol', 'source_receipt')}}
    selected = write(base / 'selection-before-calibration.json', selection)
    graph = write(root / '.cache/bundle/model.onnx', 'onnx')
    release = {k: relative(v) for k, v in files.items() if k not in ('checkpoint', 'protocol')}
    release.update(bundle='.cache/bundle', qualified=qualified,
                   bindings={relative(v): digest(v) for v in [*files.values(), selected, graph]})
    write(base / 'release.json', release)
    return base, files


class ExternalStages:
    def __init__(self, root, failure=None):
        self.root, self.failure, self.commands = root, failure, []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        text = ' '.join(map(str, command))
        if self.failure and self.failure in text:
            raise subprocess.CalledProcessError(9, command)
        if '--summary' in command:
            folder = Path(command[command.index('--output') + 1])
            name = command[command.index('--name') + 1]
            write(folder / (name + '.pptx'), 'editable slide fixture')
            write(folder / (name + '.pdf'), 'PDF visual companion fixture')
        return subprocess.CompletedProcess(command, 0)


def packager(bundle, output, **kwargs):
    assert kwargs['diagnostic'] is True
    entries = {'model.onnx': b'onnx', 'pi_test.py': b'pass\n'}
    entries.update({k: Path(v).read_bytes() for k, v in kwargs['extra_files'].items()})
    manifest = {'files': {k: hashlib.sha256(v).hexdigest() for k, v in entries.items()}}
    with zipfile.ZipFile(output, 'x') as z:
        for key, val in entries.items(): z.writestr(key, val)
        z.writestr('MANIFEST.json', json.dumps(manifest))
    return {'path': str(output), 'diagnostic': True}


def test_skip_preserves_bound_source_and_runs_validation_and_diagnostic_package(tmp_path):
    from scripts.pcv_release import run_workflow
    base, files = fixture(tmp_path)
    write(tmp_path / 'experiments/pcv-source-refresh-v1/acquisition-complete.json', {'revision': 'unrelated-new'})
    stages = ExternalStages(tmp_path)
    result = run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager, test_modules=['fixture_test.py'])
    commands = '\n'.join(' '.join(map(str, c)) for c in stages.commands)
    assert 'pcv_release_data' not in commands
    assert 'pcv_release_prepare' not in commands
    assert 'pcv_release_training' not in commands
    assert 'pytest' in commands and 'pcv_release_qualification' in commands
    assert result['status'] == 'diagnostic_complete'
    assert result['qualified'] is False
    assert result['source_receipt'] == str(files['source_receipt'].resolve())
    assert Path(result['receipt']).is_file()
    assert Path(result['pdf']).is_file()
    assert result['pdf_sha256'] == digest(Path(result['pdf']))
    with zipfile.ZipFile(result['package']) as z:
        assert not any(x.endswith('.pt') for x in z.namelist())
        assert 'evidence/reproduction.json' in z.namelist()
        assert 'slides/me2-diagnostic.pdf' in z.namelist()
        assert z.read('evidence/selection-before-calibration.json') == (base / 'selection-before-calibration.json').read_bytes()


@pytest.mark.parametrize('key', ['checkpoint', 'protocol', 'source_receipt'])
def test_skip_rejects_changed_training_binding_before_any_stage(tmp_path, key):
    from scripts.pcv_release import run_workflow
    _, files = fixture(tmp_path)
    files[key].write_text('changed')
    stages = ExternalStages(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager)
    assert not stages.commands


def test_stage_failure_stops_before_package_and_records_failure(tmp_path):
    from scripts.pcv_release import run_workflow
    fixture(tmp_path)
    stages = ExternalStages(tmp_path, 'pytest')
    with pytest.raises(subprocess.CalledProcessError):
        run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager)
    receipts = list((tmp_path / 'experiments/pcv-release-train-v1/workflow').glob('*/workflow.json'))
    assert len(receipts) == 1
    value = json.loads(receipts[0].read_text())
    assert value['status'] == 'failed'
    assert value['stages'][-1]['status'] == 'failed'
    assert not list((tmp_path / 'dist').rglob('*.zip'))


def test_normal_run_calls_source_prepare_training_then_qualification(tmp_path):
    from scripts.pcv_release import run_workflow
    fixture(tmp_path)
    stages = ExternalStages(tmp_path)
    result = run_workflow(root=tmp_path, runner=stages, packager=packager, test_modules=['fixture_test.py'])
    modules = [c[c.index('-m') + 1] for c in stages.commands if '-m' in c]
    assert modules[:3] == ['scripts.train.pcv_release_data', 'scripts.train.pcv_release_prepare', 'scripts.train.pcv_release_training']
    assert modules.index('pytest') < modules.index('scripts.test.pcv_release_qualification')
    assert result['training_mode'] == 'resume_or_verify'


def test_release_requires_selection_hash_and_boolean_quality(tmp_path):
    from scripts.pcv_release import validate_release
    base, _ = fixture(tmp_path)
    release = json.loads((base / 'release.json').read_text())
    release['qualified'] = 'false'
    write(base / 'release.json', release)
    with pytest.raises(ValueError, match='boolean'):
        validate_release(tmp_path, base / 'release.json', base / 'selection-before-calibration.json')
    release['qualified'] = False
    release['bindings'].pop((base / 'selection-before-calibration.json').relative_to(tmp_path).as_posix())
    write(base / 'release.json', release)
    with pytest.raises(ValueError, match='selection'):
        validate_release(tmp_path, base / 'release.json', base / 'selection-before-calibration.json')


def test_release_changed_during_deck_is_not_packaged(tmp_path):
    from scripts.pcv_release import run_workflow
    base, _ = fixture(tmp_path)
    stages = ExternalStages(tmp_path)
    def changed(command, **kwargs):
        result = stages(command, **kwargs)
        if '--summary' in command:
            release = json.loads((base / 'release.json').read_text())
            release['changed_after_validation'] = True
            write(base / 'release.json', release)
        return result
    with pytest.raises(ValueError, match='release identity'):
        run_workflow(root=tmp_path, skip_train=True, runner=changed, packager=packager)
    assert not list((tmp_path / 'dist').rglob('*.zip'))


@pytest.mark.parametrize('member', ['../escape.txt', 'checkpoints/trained.pt'])
def test_package_rejects_escape_and_training_checkpoint(tmp_path, member):
    from scripts.pcv_release import verify_extract_package
    package = tmp_path / 'bad.zip'
    data = b'payload'
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr(member, data)
        archive.writestr('MANIFEST.json', json.dumps({'files': {member: hashlib.sha256(data).hexdigest()}}))
    with pytest.raises(ValueError):
        verify_extract_package(package, tmp_path / 'extract')
    assert not (tmp_path / 'escape.txt').exists()


def test_missing_pdf_stops_before_package(tmp_path):
    from scripts.pcv_release import run_workflow
    fixture(tmp_path)
    stages = ExternalStages(tmp_path)
    def no_pdf(command, **kwargs):
        result = stages(command, **kwargs)
        if '--summary' in command:
            folder = Path(command[command.index('--output') + 1])
            name = command[command.index('--name') + 1]
            (folder / (name + '.pdf')).unlink()
        return result
    with pytest.raises(ValueError, match='PDF'):
        run_workflow(root=tmp_path, skip_train=True, runner=no_pdf, packager=packager)
    assert not list((tmp_path / 'dist').rglob('*.zip'))


@pytest.mark.parametrize('bound', [True, False])
def test_model_results_ship_only_when_release_bound(tmp_path, bound):
    from scripts.pcv_release import run_workflow
    base, files = fixture(tmp_path)
    results = write(base / 'results.json', {'qualified': False, 'human': {'exact': 3, 'n': 10}})
    name = results.relative_to(tmp_path).as_posix()
    write(files['summary'], {'raw_results': name})
    release = json.loads((base / 'release.json').read_text())
    release['bindings'][files['summary'].relative_to(tmp_path).as_posix()] = digest(files['summary'])
    if bound:
        release['bindings'][name] = digest(results)
    write(base / 'release.json', release)
    stages = ExternalStages(tmp_path)
    if not bound:
        with pytest.raises(ValueError, match='model results.*bound'):
            run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager)
        assert not list((tmp_path / 'dist').rglob('*.zip'))
    else:
        result = run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager)
        with zipfile.ZipFile(result['package']) as archive:
            assert archive.read('evidence/model-results.json') == results.read_bytes()
