"""Producer cutover: no deployed app, no app build and honest source ownership."""
import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_model_repository_has_no_application_entrypoints():
    for name in ('SpeakerUI', 'scripts/runtime', 'inference.py', 'benchmark.py', 'start.py',
                 'scripts/mvp_delivery.py', 'scripts/speaker_publish.py', 'scripts/iris_final_package.py'):
        assert not (ROOT / name).exists(), name


def test_training_and_shipping_do_not_import_deployment_runtime():
    paths = list((ROOT / 'scripts/train').rglob('*.py'))
    paths += [ROOT / 'scripts' / name for name in (
        'archive_io.py', 'iris_release.py', 'iris_reproduce.py',
        'iris_reproduction.py', 'iris_publish.py', 'model_package.py')]
    for path in paths:
        assert path.is_file(), str(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = ([node.module] if isinstance(node, ast.ImportFrom)
                     else [alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else [])
            for name in names:
                assert not (name and (name == 'scripts.runtime'
                            or name.startswith('scripts.runtime.'))), str(path)


def test_new_source_closure_relocates_references_but_does_not_hide_missing_training():
    from scripts.repository_boundary import model_source_files
    source = ['scripts/train/iris_diversity.py', 'scripts/runtime/mvp_static.py',
              'SpeakerUI/package.json', 'start.py', 'scripts/runtime/speaker_worker.py',
              'scripts/mvp_delivery.py', 'scripts/missing_training_source.py']
    actual = model_source_files(source)
    assert actual == sorted({'scripts/train/iris_diversity.py', 'scripts/train/iris_validation.py',
                             'scripts/model_package.py', 'scripts/missing_training_source.py'})
    assert source[2] == 'SpeakerUI/package.json'  # historical input metadata was not rewritten


def test_model_presentation_has_its_own_browser_dependency():
    manifest = json.loads((ROOT / 'presentation/package.json').read_bytes())
    assert manifest['private'] is True
    assert manifest['devDependencies'] == {'playwright': '1.62.1'}
    source = (ROOT / 'presentation/qa.mjs').read_text(encoding='utf8')
    assert "resolve(root,'SpeakerUI')" not in source
    assert "resolve(root,'presentation')" in source


def test_static_release_integration_does_not_require_removed_inference_launcher(monkeypatch):
    from presentation.validate import validate_site
    original = Path.is_file
    def without_inference(path):
        return False if path == ROOT / 'inference.py' else original(path)
    monkeypatch.setattr(Path, 'is_file', without_inference)
    assert validate_site(ROOT, require_integration=True)['integration_paths_missing'] == []


def test_model_workflow_never_bootstraps_ui_or_runs_deployment_benchmarks(tmp_path, monkeypatch):
    from scripts import iris_release as module
    from scripts.test.test_iris_release import fixture, Stages
    from scripts.archive_io import write_archive
    files = fixture(tmp_path)
    # The inherited fixture serializes WindowsPath with backslashes. This new
    # fixture declares a valid POSIX receipt; production rejects backslashes.
    inputs = json.loads(files['receipt'].read_text())
    inputs['files'][0]['path'] = files['inputs'].relative_to(tmp_path).as_posix()
    files['receipt'].write_text(json.dumps(inputs))
    stages = Stages(tmp_path)
    def execute(command, **kwargs):
        assert not any(str(part) in ('pnpm', 'node', 'npm', 'scripts.test.iris_resources')
                       for part in command), 'Training attempted an application build/benchmark'
        return stages(command, **kwargs)
    def package(bundle, output, **kwargs):
        entries = {'models/default/model.onnx': (bundle / 'model.onnx').read_bytes(),
                   **kwargs['extra_files']}
        return write_archive(entries, output, {'format': 'smart-speaker-model-v1', 'qualified': False})
    result = module.run_workflow(root=tmp_path, runner=execute, packager=package,
                                python=__import__('sys').executable,
                                test_modules=['fixture.py'], run_id='model-boundary')
    assert result['status'] == 'complete'
    assert result['software_verification']['checks'] == ['python-tests']
    assert not any(item['name'].startswith('ui-') or item['name'] == 'pipeline-resources'
                   for item in result['stages'])
    fits = [c for c in stages.commands if 'scripts.train.iris_diversity' in c and 'train' in c]
    assert len(fits) == 6
    assert result['package']['path'].endswith('/iris-model.zip')
    manifest = result['package_verification']
    assert manifest['passed'] is True


def test_current_module_registry_only_points_to_model_sources():
    modules = json.loads((ROOT / 'scripts/modules.json').read_bytes())
    assert modules and 'iris.model_package' in modules
    for name, item in modules.items():
        assert item['runtime'] is False, name
        assert (ROOT / 'scripts' / item['path']).is_file(), (name, item['path'])
        assert not item['path'].startswith('runtime/'), name
