"""Release boundaries: pinned inputs, selected exports, archives and publication."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content) if isinstance(content, (dict, list)) else content)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    inputs = put(root, '.cache/iris-release-v1/inputs/initializer.pt', 'our checkpoint')
    source = put(root, 'scripts/train/iris_diversity.py', '# fixture source\n')
    protocol = put(root, 'experiments/iris-release-v1/protocol.json', {'experiment': 'iris-release-v1'})
    receipt = put(root, 'experiments/iris-release-v1/reproduction-inputs.json', {
        'schema_version': 1, 'experiment': 'iris-release-v1', 'files': [
            {'path': (inputs.relative_to(root)).as_posix(), 'sha256': sha(inputs), 'bytes': inputs.stat().st_size, 'role': 'vcm_initializer',
             'remote': {'repo_id': 'example/iris', 'repo_type': 'model', 'revision': 'a' * 40,
                        'filename': 'reproduction/initializer.pt'}}],
        'source_files': ['scripts/train/iris_diversity.py'],
        'data_requirements': [{'source': 'restricted original recordings', 'status': 'access required'}]})
    bundle = root / '.cache/iris-release-v1/diversity/release-export'
    graph = put(root, (bundle.relative_to(root) / 'model.onnx').as_posix(), 'finite direct model')
    put(root, (bundle.relative_to(root) / 'bundle.json').as_posix(), {
        'parameters': 969928, 'files': {'model.onnx': sha(graph)}})
    wake = root / '.cache/iris-release-v1/wake/export'
    wake_graph = put(root, (wake.relative_to(root) / 'wake.onnx').as_posix(), 'our hey iris')
    put(root, (wake.relative_to(root) / 'iris.json').as_posix(), {'files': {'wake.onnx': sha(wake_graph)}})
    owner = root / '.cache/iris-release-v1/owner/export'
    owner_graph = put(root, (owner.relative_to(root) / 'owner.onnx').as_posix(), 'owner embedding')
    put(root, (owner.relative_to(root) / 'owner.json').as_posix(), {'files': {'owner.onnx': sha(owner_graph)}})
    vad_graph = put(root, '.cache/iris-release-v1/inputs/vad/silero-vad.onnx', 'pinned VAD')
    put(root, 'artifacts/wake-models.json', {'files': {'silero-vad.onnx': {'sha256': sha(vad_graph)}}})
    put(root, 'artifacts/silero-LICENSE.txt', 'MIT source notice')
    selected = put(root, 'experiments/iris-release-v1/selection-before-test.json', {'test_used': False})
    return dict(inputs=inputs, source=source, protocol=protocol, receipt=receipt,
                bundle=bundle, wake=wake, owner=owner, selection=selected, graph=graph)


def freeze(root, files):
    from scripts.iris_release import freeze_release
    return freeze_release(root=root, bundle=files['bundle'], wake_folder=files['wake'],
                          owner_folder=files['owner'], input_receipt=files['receipt'],
                          selection=files['selection'], source_files=[files['source']])


class Stages:
    """Only the slow fitting/compiler process boundary is replaced."""
    def __init__(self, root, fail=None, mutate=None):
        self.root, self.fail, self.mutate, self.commands = root, fail, mutate, []

    def __call__(self, command, **kwargs):
        command = [str(x) for x in command]
        self.commands.append(command)
        if self.fail and self.fail in command:
            raise subprocess.CalledProcessError(9, command)
        if 'select' in command and 'scripts.train.iris_diversity' in command:
            put(self.root, 'experiments/iris-release-v1/width-selection-before-repeat.json',
                {'stage': 'repeat', 'model': 'compact', 'arms': ['ce', 'supcon'], 'seed': 67})
        if 'scripts.test.iris_resources' in command:
            pytest.fail('Model-only workflows cannot run deployment resource benchmarks')
        if 'scripts.test.iris_decision_parity' in command:
            output = command[command.index('--output') + 1] if '--output' in command else 'experiments/iris-release-v1/decision-parity.json'
            put(self.root, output, {'passed': True, 'models': 'fixture'})
        if self.mutate and 'scripts.test.iris_decision_parity' in command:
            self.mutate.write_text('source changed during build')
        return subprocess.CompletedProcess(command, 0)


def packager(bundle, output, **kwargs):
    from scripts.archive_io import write_archive
    entries = {'models/default/model.onnx': (bundle / 'model.onnx').read_bytes(),
               'models/default/wake/iris.json': b'{}', 'models/default/owner/owner.json': b'{}'}
    entries.update(kwargs['extra_files'])
    return write_archive(entries, output, {'format': 'smart-speaker-model-v1', 'qualified': False, 'actions_enabled': False, 'physical_pi_measured': False})


def test_missing_input_uses_exact_published_revision(tmp_path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    payload = files['inputs'].read_bytes()
    files['inputs'].unlink()
    download = put(tmp_path, 'download.pt', payload.decode())
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return str(download)
    result = prepare_inputs(tmp_path, files['receipt'], downloader=fetch)
    assert files['inputs'].read_bytes() == b'our checkpoint'
    assert calls[0]['revision'] == 'a' * 40
    assert calls[0]['filename'] == 'reproduction/initializer.pt'
    assert result['verified_files'] == 1


def test_independent_pinned_input_downloads_overlap_and_restore_in_inventory_order(tmp_path):
    from threading import Barrier
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    files['inputs'].unlink()
    inventory, payloads = [], {}
    for index in range(2):
        payload = put(tmp_path, 'download-' + str(index) + '.pt', 'verified bytes ' + str(index))
        inventory.append({'path': '.cache/iris-release-v1/inputs/' + payload.name,
                          'sha256': sha(payload), 'remote': {'repo_type': 'model', 'repo_id': 'example/iris',
                              'revision': 'a' * 40, 'filename': payload.name}})
        payloads[payload.name] = payload
    document = json.loads(files['receipt'].read_text())
    document['files'] = inventory
    files['receipt'].write_text(json.dumps(document))
    barrier = Barrier(2)
    def fetch(**kwargs):
        barrier.wait(timeout=3)
        return payloads[kwargs['filename']]
    result = prepare_inputs(tmp_path, files['receipt'], downloader=fetch)
    assert result['restored'] == [item['path'] for item in inventory]
    assert all(sha(tmp_path / item['path']) == item['sha256'] for item in inventory)


def test_changed_present_input_is_never_replaced_by_download(tmp_path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    files['inputs'].write_text('a different checkpoint')
    with pytest.raises(ValueError, match='identity'):
        prepare_inputs(tmp_path, files['receipt'], downloader=lambda **_: pytest.fail('unexpected download'))
    assert files['inputs'].read_text() == 'a different checkpoint'


@pytest.mark.parametrize('path', ['../escape.pt', '/tmp/escape.pt', '.cache/../escape.pt'])
def test_hydration_refuses_unsafe_input_destination(tmp_path, path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    receipt = json.loads(files['receipt'].read_text())
    receipt['files'][0]['path'] = path
    files['receipt'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='relative|Unsafe'):
        prepare_inputs(tmp_path, files['receipt'], downloader=lambda **_: pytest.fail('unexpected download'))


def test_absent_restricted_input_reports_requirement_without_substitution(tmp_path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    files['inputs'].unlink()
    receipt = json.loads(files['receipt'].read_text())
    receipt['files'][0].pop('remote')
    files['receipt'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='restricted original recordings'):
        prepare_inputs(tmp_path, files['receipt'])


def test_random_preflight_ignores_pinned_vcm_stage_weights_and_their_aliases(tmp_path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    inputs = json.loads(files['receipt'].read_text())
    inputs['files'].append({'path': '.cache/iris-release-v1/inputs/completed-stages/diversity/runs/compact-ce-s53/best.pt',
                            'sha256': 'a' * 64, 'role': 'vcm_completed_run'})
    inputs['cache_aliases'] = [{'path': '.cache/iris-release-v1/diversity/runs/compact-ce-s53/best.pt',
                               'source': inputs['files'][-1]['path'], 'sha256': 'a' * 64}]
    files['receipt'].write_text(json.dumps(inputs))
    files['inputs'].unlink()
    result = prepare_inputs(tmp_path, files['receipt'], from_scratch=True,
                            downloader=lambda **_: pytest.fail('random study fetched VCM weights'))
    assert result['verified_files'] == 0


def test_skip_refuses_changed_export_before_any_external_stage(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    freeze(tmp_path, files)
    files['graph'].write_text('other graph')
    stages = Stages(tmp_path)
    with pytest.raises(ValueError, match='identity'):
        run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager,
                     python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert stages.commands == []
    assert not list((tmp_path / 'dist').rglob('*.zip'))


def test_skip_rejects_alternate_component_routes_before_any_external_stage(tmp_path):
    import shutil
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    alternate = tmp_path / '.cache/iris-release-v1/alternate/export'
    alternate.parent.mkdir(parents=True)
    shutil.move(files['bundle'], alternate)
    files['bundle'] = alternate
    freeze(tmp_path, files)
    stages = Stages(tmp_path)
    with pytest.raises(ValueError, match='canonical'):
        run_workflow(root=tmp_path, skip_train=True, runner=stages, packager=packager,
                     python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert stages.commands == []


def test_skip_restores_missing_selected_export_from_pinned_input_alias(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    snapshot = put(tmp_path, '.cache/iris-release-v1/inputs/selected/vcm/model.onnx', files['graph'].read_text())
    inputs = json.loads(files['receipt'].read_text())
    inputs['files'].append({'path': (snapshot.relative_to(tmp_path)).as_posix(), 'sha256': sha(snapshot), 'bytes': snapshot.stat().st_size})
    inputs['cache_aliases'] = [{'path': (files['graph'].relative_to(tmp_path)).as_posix(),
                               'source': (snapshot.relative_to(tmp_path)).as_posix(), 'sha256': sha(snapshot)}]
    files['receipt'].write_text(json.dumps(inputs))
    freeze(tmp_path, files)
    files['graph'].unlink()
    result = run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path), packager=packager,
                          python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert result['status'] == 'complete'
    assert files['graph'].read_bytes() == b'finite direct model'


def test_skip_evaluates_rebuilds_and_hash_verifies_final_archive(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    freeze(tmp_path, files)
    result = run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path),
                          packager=packager, python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert result['status'] == 'complete'
    assert result['training_mode'] == 'verified_selected_exports'
    assert result['physical_pi_measured'] is False
    assert result['package_verification']['passed'] is True
    with zipfile.ZipFile(tmp_path / result['package']['path']) as archive:
        assert 'evidence/reproduction-release.json' in archive.namelist()
        assert all(not n.endswith('.pt') for n in archive.namelist())
    commands = [item['command'] for item in result['stages'] if 'command' in item]
    assert not any('train' in c or 'generate' in c for c in commands)
    assert sum('evaluate' in c for c in commands) == 3
    assert not any(part in ('pnpm', 'node', 'npm', 'scripts.test.iris_resources') for c in commands for part in c)
    assert 'resource_measurement' not in result
    assert result['decision_parity']['historical_release_parity_unchanged'] is True
    with zipfile.ZipFile(tmp_path / result['package']['path']) as archive:
        assert 'evidence/current-host-decision-parity.json' in archive.namelist()
        assert 'evidence/current-host-pipeline-resources.json' not in archive.namelist()
        assert 'start.py' not in archive.namelist()
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['qualified'] is False and manifest['actions_enabled'] is False


def test_normal_schedule_runs_four_primary_and_two_dev_selected_repeat_fits(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    stages = Stages(tmp_path)
    result = run_workflow(root=tmp_path, runner=stages, packager=packager,
                          python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    fits = [c for c in stages.commands if 'scripts.train.iris_diversity' in c and 'train' in c]
    assert len(fits) == 6
    assert [c[c.index('--seed') + 1] for c in fits] == ['53'] * 4 + ['67'] * 2
    assert [c[c.index('--model') + 1] for c in fits[-2:]] == ['compact', 'compact']
    assert result['status'] == 'complete'


def test_reports_render_after_acoustics_and_software_before_final_freeze(tmp_path):
    from scripts.iris_release import run_workflow
    fixture(tmp_path)
    stages = Stages(tmp_path)
    def execute(command, **kwargs):
        assert not any(str(part) in ('pnpm', 'node', 'npm') for part in command)
        assert 'scripts.test.iris_resources' not in command
        assert not (tmp_path / 'SpeakerUI').exists()
        return stages(command, **kwargs)
    result = run_workflow(root=tmp_path, runner=execute, packager=packager,
                         python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    names = [stage['name'] for stage in result['stages']]
    assert names.index('owner-evaluate') < names.index('decision-parity') < names.index('report-acoustic-evidence')
    assert names.index('python-tests') < names.index('report-final-evidence') < names.index('freeze-selected-release')
    assert not any(name.startswith('ui-') or name == 'pipeline-resources' for name in names)
    receipt = json.loads((tmp_path / 'experiments/iris-release-v1/software-verification.json').read_text())
    assert receipt['passed'] is True and receipt['checks'] == ['python-tests']
    assert not any('utc' in key or 'duration' in key for key in receipt)


def test_model_workflow_preserves_thread_overrides_without_deployment_profiler(tmp_path, monkeypatch):
    from scripts.iris_release import run_workflow
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '3')
    fixture(tmp_path)
    stages, observed = Stages(tmp_path), []
    def execute(command, **kwargs):
        observed.append((command, kwargs['env']))
        return stages(command, **kwargs)
    run_workflow(root=tmp_path, runner=execute, packager=packager,
                 python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert not any('scripts.test.iris_resources' in command for command, _ in observed)
    for command, environment in observed:
        assert environment['OPENBLAS_NUM_THREADS'] == '3'
        if 'train' in command:
            assert all(key not in environment for key in keys if key != 'OPENBLAS_NUM_THREADS')


def test_failed_software_stage_records_failure_and_never_packages(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    freeze(tmp_path, files)
    with pytest.raises(subprocess.CalledProcessError):
        run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path, fail='pytest'),
                     packager=packager, python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    receipts = list((tmp_path / 'experiments/iris-release-v1').glob('reproduction-run-*.json'))
    assert json.loads(receipts[0].read_text())['status'] == 'failed'
    assert not list((tmp_path / 'dist').rglob('*.zip'))


def test_source_change_during_build_prevents_packaging(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    freeze(tmp_path, files)
    with pytest.raises(ValueError, match='identity'):
        run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path, mutate=files['source']),
                     packager=packager, python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert not list((tmp_path / 'dist').rglob('*.zip'))


@pytest.mark.parametrize('member', ['../escape.txt', 'checkpoints/selected.pt', '.env'])
def test_package_validation_refuses_escape_checkpoint_or_private_settings(tmp_path, member):
    from scripts.iris_release import verify_package
    path = tmp_path / 'unsafe.zip'
    payload = b'private'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(member, payload)
        archive.writestr('MANIFEST.json', json.dumps({'files': {member: hashlib.sha256(payload).hexdigest()}}))
    with pytest.raises(ValueError):
        verify_package(path)


def test_publish_refuses_changed_verified_package_before_network(tmp_path):
    from scripts.iris_publish import validate_publication
    package = put(tmp_path, 'dist/iris.zip', 'reviewed bytes')
    plan = {'schema_version': 1, 'verification': {'passed': True}, 'files': [
        {'local_path': 'dist/iris.zip', 'path_in_repo': 'releases/iris.zip', 'sha256': sha(package),
         'repo_type': 'model', 'repo_id': 'example/iris'}]}
    package.write_text('changed bytes')
    with pytest.raises(ValueError, match='identity'):
        validate_publication(tmp_path, plan)


def test_final_publication_references_verified_input_capsule_without_reupload(tmp_path):
    from scripts.iris_release import run_workflow
    from scripts.iris_publish import build_publication_plan
    files = fixture(tmp_path)
    inputs = json.loads(files['receipt'].read_text())
    item = inputs['files'][0]
    item['publication'] = {'repo_type': 'model', 'repo_id': 'example/iris', 'path_in_repo': 'reproduction/initializer.pt'}
    proof = put(tmp_path, 'experiments/iris-release-v1/publication-input-proof.json',
                {'status': 'published_and_verified', 'repositories': [
                    {'repo_type': 'model', 'repo_id': 'example/iris', 'revision': 'a' * 40, 'verified': True,
                     'files': [{'path': 'reproduction/initializer.pt', 'sha256': item['sha256']}]}]})
    inputs['input_publication_receipt'] = {'path': (proof.relative_to(tmp_path)).as_posix(), 'sha256': sha(proof)}
    files['receipt'].write_text(json.dumps(inputs))
    freeze(tmp_path, files)
    workflow = run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path), packager=packager,
                             python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    plan = build_publication_plan(root=tmp_path, workflow=workflow)
    assert not any(x['local_path'] == (files['inputs'].relative_to(tmp_path)).as_posix() for x in plan['files'])
    assert any(x['path_in_repo'] == 'reproduction/publication-inputs.json' for x in plan['files'])
    destinations = {x['path_in_repo'] for x in plan['files']}
    assert not any('reproduction/source/' in name for name in destinations)
    assert 'reproduction/reproduction-inputs.json' in destinations
    assert 'reproduction/reproduction-release.json' in destinations
    assert plan['model_visibility'] == 'private'
    assert plan['source_visibility'] == 'preserve_existing'


def test_reconstruction_inventory_contains_all_model_source_reads():
    from scripts.iris_release import DEFAULT_INPUTS
    from scripts.repository_boundary import model_source_files
    project = Path(__file__).resolve().parents[2]
    original = json.loads((project / DEFAULT_INPUTS).read_text())['source_files']
    inventory = model_source_files(original)
    assert 'scripts/model_package.py' in inventory
    assert all(not name.startswith(('SpeakerUI/', 'scripts/runtime/')) for name in inventory)
    assert all((project / name).is_file() for name in inventory)
    assert 'scripts/train/iris_validation.py' in inventory


@pytest.mark.parametrize('available', [True, False])
def test_standard_publication_closes_card_linked_late_evidence(tmp_path, available):
    from scripts.iris_release import run_workflow
    from scripts.iris_publish import build_publication_plan, file_item
    files = fixture(tmp_path)
    name = 'experiments/iris-release-v1/ui-final-integration.json'
    put(tmp_path, 'MODEL_CARD.md', '# Measured Iris\n[Native proof](' + name + ')\n')
    if available:
        evidence = put(tmp_path, name, {'status': 'passed', 'scope': 'fixture'})
    release = freeze(tmp_path, files)
    assert name not in release['source_bindings'] and name not in release['evaluation_receipts']
    workflow = run_workflow(root=tmp_path, skip_train=True, runner=Stages(tmp_path), packager=packager,
                            python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    if not available:
        with pytest.raises(ValueError, match='[Cc]ard-linked'):
            build_publication_plan(root=tmp_path, workflow=workflow)
        return
    plan = build_publication_plan(root=tmp_path, workflow=workflow,
        extra_files=[file_item(tmp_path, evidence, 'evidence/ui-final-integration.json')])
    destinations = {item['path_in_repo']: item for item in plan['files']}
    assert destinations['evidence/ui-final-integration.json']['sha256'] == sha(evidence)
    assert sum(item['path_in_repo'] == 'evidence/ui-final-integration.json' for item in plan['files']) == 1
    assert not any('reproduction/source/' in destination for destination in destinations)


def test_input_capsule_reuses_verified_immutable_preupload_and_pins_new_files(tmp_path, monkeypatch):
    from scripts import iris_publish
    first = put(tmp_path, '.cache/iris-release-v1/inputs/first.pt', 'immutable earlier input')
    second = put(tmp_path, '.cache/iris-release-v1/inputs/second.pt', 'new selected weights')
    inventory = [{'path': (p.relative_to(tmp_path)).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size,
                  'publication': {'repo_type': 'model', 'repo_id': 'example/iris',
                                  'path_in_repo': 'reproduction/' + p.name}} for p in (first, second)]
    inputs = put(tmp_path, 'experiments/iris-release-v1/reproduction-inputs.json', {'files': inventory})
    proof = put(tmp_path, 'experiments/iris-release-v1/publication-preupload.json',
                {'status': 'published_and_verified', 'repositories': [
                    {'repo_type': 'model', 'repo_id': 'example/iris', 'revision': 'a' * 40, 'verified': True,
                     'private': False, 'files': [{'path': 'reproduction/first.pt', 'sha256': sha(first),
                                                 'bytes': first.stat().st_size}]}]})
    captured = []
    def publish(*, root, plan):
        captured.extend(plan['files'])
        grouped = {}
        for item in plan['files']:
            repository = grouped.setdefault((item['repo_type'], item['repo_id']), {
                'repo_type': item['repo_type'], 'repo_id': item['repo_id'], 'revision': 'b' * 40,
                'verified': True, 'private': False, 'files': []})
            repository['files'].append({'path': item['path_in_repo'], 'sha256': item['sha256'], 'bytes': item['bytes']})
        receipt = 'experiments/iris-release-v1/publication-new-inputs.json'
        result = {'status': 'published_and_verified', 'receipt': receipt, 'repositories': list(grouped.values())}
        put(root, receipt, result)
        return result
    monkeypatch.setattr(iris_publish, 'publish_plan', publish)
    result = iris_publish.publish_reproduction_inputs(root=tmp_path, input_receipt=inputs,
                verified_receipts=[{'path': (proof.relative_to(tmp_path)).as_posix(), 'sha256': sha(proof)}])
    assert not any(item['local_path'] == (first.relative_to(tmp_path)).as_posix() for item in captured)
    assert any(item['local_path'] == (second.relative_to(tmp_path)).as_posix() for item in captured)
    pinned = json.loads(inputs.read_text())
    assert [item['remote']['revision'] for item in pinned['files']] == ['a' * 40, 'b' * 40]
    merged = json.loads((tmp_path / result['receipt']).read_text())
    assert all(repository['verified'] for repository in merged['repositories'])
    assert merged['reused_files'] == 1
    assert pinned['input_publication_receipt']['sha256'] == sha(tmp_path / result['receipt'])


@pytest.mark.parametrize('changed', ['input', 'proof'])
def test_input_capsule_reuse_rejects_changed_identity_before_network(tmp_path, monkeypatch, changed):
    from scripts import iris_publish
    source = put(tmp_path, '.cache/iris-release-v1/inputs/frozen.pt', 'frozen bytes')
    inputs = put(tmp_path, 'experiments/iris-release-v1/reproduction-inputs.json', {'files': [
        {'path': (source.relative_to(tmp_path)).as_posix(), 'sha256': sha(source),
         'publication': {'repo_type': 'model', 'repo_id': 'example/iris', 'path_in_repo': 'frozen.pt'}}]})
    proof = put(tmp_path, 'experiments/iris-release-v1/publication-preupload.json',
                {'status': 'published_and_verified', 'repositories': []})
    pointer = {'path': (proof.relative_to(tmp_path)).as_posix(), 'sha256': sha(proof)}
    (source if changed == 'input' else proof).write_text('changed bytes')
    monkeypatch.setattr(iris_publish, 'publish_plan', lambda **_: pytest.fail('network before identity check'))
    with pytest.raises(ValueError, match='identity'):
        iris_publish.publish_reproduction_inputs(root=tmp_path, input_receipt=inputs, verified_receipts=[pointer])


def test_publish_requires_explicit_audio_redistribution_and_bound_notice(tmp_path):
    from scripts.iris_publish import validate_publication
    audio = put(tmp_path, '.cache/iris-release-v1/generated.wav', 'audio')
    plan = {'schema_version': 1, 'verification': {'passed': True}, 'files': [
        {'local_path': (audio.relative_to(tmp_path)).as_posix(), 'path_in_repo': 'audio/example.wav',
         'sha256': sha(audio), 'repo_type': 'dataset', 'repo_id': 'example/iris'}]}
    with pytest.raises(ValueError, match='redistribution'):
        validate_publication(tmp_path, plan)


def test_private_settings_cannot_be_published_under_an_innocent_name(tmp_path):
    from scripts.iris_publish import validate_publication
    private = put(tmp_path, '.env', 'private local settings')
    plan = {'schema_version': 1, 'verification': {'passed': True}, 'files': [
        {'local_path': '.env', 'path_in_repo': 'README.md', 'sha256': sha(private),
         'repo_type': 'model', 'repo_id': 'example/iris'}]}
    with pytest.raises(ValueError, match='Private'):
        validate_publication(tmp_path, plan)


def test_dataset_card_leads_with_current_release_and_preserves_original_terms(tmp_path):
    from scripts.iris_publish import write_cards
    old = '---\nlicense: original-source-terms\n---\n\n# Earlier reader architecture\nOriginal source and consent terms.\n'
    put(tmp_path, '.cache/iris-release-v1/inputs/provenance/dataset-card-before-release.md', old)
    put(tmp_path, 'docs/DATASETS.md', '# Current data audit\nMeasured split denominators.\n')
    _, dataset, _ = write_cards(tmp_path, {'evaluation_receipts': []}, {}, tmp_path / 'cards')
    result = dataset.read_text()
    assert result.startswith('---\nlicense: original-source-terms\n---\n')
    assert result.index('# Current Iris release') < result.index('# Earlier reader architecture')
    assert 'Original source and consent terms.' in result
    assert 'Measured split denominators.' in result


def test_authoritative_model_card_retains_hub_metadata_and_quantitative_body(tmp_path):
    from scripts.iris_publish import write_cards
    from urllib.parse import urlsplit
    put(tmp_path, 'MODEL_CARD.md', '# Iris measured results\n35/36 accepted.\n[Evidence](experiments/iris-release-v1/recognition-results.json)\n')
    model, dataset, _ = write_cards(tmp_path, {'evaluation_receipts': []}, {}, tmp_path / 'cards')
    result = model.read_text()
    assert result.startswith('---\n')
    assert 'license: other\n' in result
    for card in (model, dataset):
        link = next(line.removeprefix('license_link: ') for line in card.read_text().splitlines() if line.startswith('license_link: '))
        parsed = urlsplit(link)
        assert parsed.scheme == 'https' and parsed.netloc == 'huggingface.co'
        assert parsed.path.endswith('/NOTICES.md')
    assert '35/36 accepted.' in result
    assert '/blob/main/evidence/recognition-results.json' in result


def test_unpinned_remote_input_revision_is_rejected_before_download(tmp_path):
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    files['inputs'].unlink()
    receipt = json.loads(files['receipt'].read_text())
    receipt['files'][0]['remote']['revision'] = 'main'
    files['receipt'].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='revision'):
        prepare_inputs(tmp_path, files['receipt'], downloader=lambda **_: pytest.fail('unexpected download'))


@pytest.mark.parametrize('manifest_name', [
    'experiments/iris-release-v1/reproduction-inputs.json', '.cache/test-inputs.json'])
def test_audio_map_resolves_original_immutable_row_and_checks_hash(tmp_path, monkeypatch, manifest_name):
    from scripts.iris_release import resolve_audio
    files = fixture(tmp_path)
    audio = put(tmp_path, '.cache/iris-release-v1/inputs/audio/pinned.wav', 'original whole recording')
    receipt = json.loads(files['receipt'].read_text())
    receipt['audio_map'] = {'old/run/recording.wav': {'path': (audio.relative_to(tmp_path)).as_posix(), 'sha256': sha(audio)}}
    put(tmp_path, manifest_name, receipt)
    monkeypatch.setenv('IRIS_REPRODUCTION_INPUTS', manifest_name)
    row = {'audio': 'old/run/recording.wav', 'sha256': sha(audio)}
    assert resolve_audio(row, root=tmp_path) == audio
    audio.write_text('changed')
    with pytest.raises(ValueError, match='identity'):
        resolve_audio(row, root=tmp_path)


def test_pinned_parquet_restores_original_embedded_bytes_without_relabeling(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.iris_release import prepare_inputs
    files = fixture(tmp_path)
    source = tmp_path / 'download.parquet'
    payload = b'whole original recording bytes'
    pq.write_table(pa.Table.from_pylist([{'audio': {'bytes': b'other'}}, {'audio': {'bytes': payload}}]), source)
    destination = 'data/audio/pool/original.wav'
    receipt = json.loads(files['receipt'].read_text())
    receipt['audio_map'] = {destination: {'path': destination, 'sha256': hashlib.sha256(payload).hexdigest(),
                                        'parquet': 'source-shard', 'parquet_row': 1}}
    receipt['parquet_sources'] = [{'id': 'source-shard', 'path': '.cache/iris-release-v1/inputs/sources/shard.parquet',
                                  'sha256': sha(source), 'bytes': source.stat().st_size,
                                  'remote': {'repo_id': 'example/source', 'repo_type': 'dataset',
                                             'revision': 'a' * 40, 'filename': 'data/shard.parquet'}}]
    files['receipt'].write_text(json.dumps(receipt))
    result = prepare_inputs(tmp_path, files['receipt'], downloader=lambda **_: source)
    assert (tmp_path / destination).read_bytes() == payload
    assert result['restored_audio'] == 1
    assert not (tmp_path / 'data/audio/pool/other.wav').exists()


def test_publication_reads_back_exact_commit_and_retains_dataset_visibility(tmp_path):
    from scripts.iris_publish import publish_plan
    card = put(tmp_path, 'card.md', 'reviewed source card')
    plan = {'schema_version': 1, 'run_id': 'test-publication', 'verification': {'passed': True}, 'files': [
        {'local_path': 'card.md', 'path_in_repo': 'README.md', 'sha256': sha(card),
         'repo_type': kind, 'repo_id': 'example/iris'} for kind in ('dataset', 'model')]}
    published = {}
    class Hub:
        def whoami(self):
            return {'name': 'fixture-user'}
        def create_repo(self, **kwargs):
            assert kwargs['repo_type'] == 'model' and kwargs['private'] is True
        def repo_info(self, **kwargs):
            return SimpleNamespace(sha='a' * 40, private=True)
        def auth_check(self, **kwargs):
            pass
        def create_commit(self, **kwargs):
            assert kwargs['parent_commit'] == 'a' * 40
            for operation in kwargs['operations']:
                published[(kwargs['repo_type'], operation.path_in_repo)] = Path(operation.path_or_fileobj)
            return SimpleNamespace(oid='b' * 40)
    def fetch(**kwargs):
        assert kwargs['revision'] == 'b' * 40
        return published[(kwargs['repo_type'], kwargs['filename'])]
    result = publish_plan(root=tmp_path, plan=plan, api=Hub(), downloader=fetch)
    assert result['status'] == 'published_and_verified'
    assert [(r['repo_type'], r['private'], r['verified']) for r in result['repositories']] == [
        ('dataset', True, True), ('model', True, True)]
    assert 'token' not in json.dumps(result)


def test_remote_readback_mismatch_never_claims_publication_complete(tmp_path):
    from scripts.iris_publish import publish_plan
    card = put(tmp_path, 'card.md', 'reviewed card')
    wrong = put(tmp_path, 'remote.md', 'different remote content')
    plan = {'schema_version': 1, 'run_id': 'bad-remote', 'verification': {'passed': True}, 'files': [
        {'local_path': 'card.md', 'path_in_repo': 'README.md', 'sha256': sha(card),
         'repo_type': 'model', 'repo_id': 'example/iris'}]}
    class Hub:
        def whoami(self): return {'name': 'fixture-user'}
        def create_repo(self, **kwargs): pass
        def repo_info(self, **kwargs): return SimpleNamespace(sha='a' * 40, private=True)
        def auth_check(self, **kwargs): pass
        def create_commit(self, **kwargs): return SimpleNamespace(oid='b' * 40)
    with pytest.raises(ValueError, match='remote.*identity'):
        publish_plan(root=tmp_path, plan=plan, api=Hub(), downloader=lambda **_: wrong)
    receipt = json.loads((tmp_path / 'experiments/iris-release-v1/publication-hf-bad-remote.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['repositories'][0]['verified'] is False


def test_random_initializer_is_resumable_and_never_reads_published_weights(tmp_path):
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.iris_release import fit_random_initializer
    records = []
    for index, (synthetic, label) in enumerate([(False, 0), (False, 1), (True, 0), (True, 1)]):
        audio = tmp_path / ('audio-' + str(index) + '.wav')
        sf.write(audio, np.sin(np.arange(4000) * .07).astype('float32') * .05, 16000, subtype='PCM_16')
        records.append({'id': str(index), 'audio': audio.name, 'sha256': sha(audio),
                        'speaker': 'person-' + str(index), 'synthetic': synthetic, 'class_index': label})
    norm = {'mean': [0.] * 40, 'std': [1.] * 40}
    result = fit_random_initializer(root=tmp_path, folder=tmp_path / 'fit', train_rows=records,
                                    dev_rows=records, norm=norm, classes=2, width=128,
                                    epochs=1, steps=1, batch=4, device='cpu')
    saved = torch.load(tmp_path / result['checkpoint'], map_location='cpu', weights_only=True)
    assert saved['config'] == {'classes': 2, 'width': 128}
    assert saved['initialization'] == 'random'
    assert 'optimizer' in saved and 'torch_rng' in saved and 'numpy_rng' in saved
    checkpoint_hash = sha(tmp_path / result['checkpoint'])
    again = fit_random_initializer(root=tmp_path, folder=tmp_path / 'fit', train_rows=records,
                                   dev_rows=records, norm=norm, classes=2, width=128,
                                   epochs=1, steps=1, batch=4, device='cpu')
    assert again['checkpoint_sha256'] == checkpoint_hash
    records[0]['class_index'] = 1
    with pytest.raises(ValueError, match='identity'):
        fit_random_initializer(root=tmp_path, folder=tmp_path / 'fit', train_rows=records,
                                dev_rows=records, norm=norm, classes=2, width=128,
                                epochs=1, steps=1, batch=4, device='cpu')


def test_existing_root_environment_is_never_synchronized(tmp_path):
    import os
    from scripts.iris_release import ensure_source_environment
    location = '.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python'
    python = put(tmp_path, location, 'existing active interpreter')
    assert ensure_source_environment(tmp_path, runner=lambda *a, **k: pytest.fail('active environment mutation')) == python


def test_optional_initializer_cuda_rng_only_touches_requested_device():
    from scripts.iris_release import _capture_selected_cuda_rng, _restore_selected_cuda_rng
    calls, state = [], object()
    def capture(device):
        calls.append(('capture', device))
        return state
    def restore(value, device):
        assert value is state
        calls.append(('restore', device))
    fake = SimpleNamespace(cuda=SimpleNamespace(get_rng_state=capture, set_rng_state=restore))
    assert _capture_selected_cuda_rng(fake, 'cpu') is None
    _restore_selected_cuda_rng(fake, 'cpu', None)
    assert calls == []
    assert _capture_selected_cuda_rng(fake, 'cuda:3') is state
    _restore_selected_cuda_rng(fake, 'cuda:3', state)
    assert calls == [('capture', 'cuda:3'), ('restore', 'cuda:3')]


def completed_owner_fixture(root):
    from dataclasses import asdict
    from scripts.iris_release import ROOT
    from scripts.train.iris_owner import DEFAULT, PCM_NORMALIZATION
    import shutil
    sources = ('scripts/models/iris_owner.py', 'scripts/train/iris_owner_reference.py', 'scripts/train/iris_owner.py',
               'scripts/train/iris_audio.py', 'scripts/train/iris_session.py', 'scripts/iris_release.py',
               'pipeline.toml', 'requirements.lock')
    for name in sources:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    prefit = root / '.cache/iris-release-v1/inputs/provenance/owner-prefit-driver.py'
    prefit.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / 'scripts/iris_release.py', prefit)
    views = [put(root, '.cache/iris-release-v1/inputs/base-views/' + name + '.jsonl', '{}\n')
             for name in ('train', 'dev', 'calibration', 'test', 'personal-test')]
    view_records = [{'path': (p.relative_to(root)).as_posix(), 'sha256': sha(p)} for p in views]
    reservation = put(root, '.cache/iris-release-v1/inputs/owner-reservations.json', {})
    cache = '.cache/iris-release-v1/owner/'
    manifest = put(root, cache + 'manifest.jsonl', '{"id":"human"}\n')
    proxies = put(root, cache + 'synthetic-proxies.jsonl', '{"id":"synthetic"}\n')
    protocol = put(root, 'experiments/iris-release-v1/owner-protocol.json', {
        'frontend': asdict(DEFAULT), 'pcm_normalization': PCM_NORMALIZATION,
        'source_files': {'scripts/models/iris_owner.py': sha(root / 'scripts/models/iris_owner.py')},
        'data': {'vcm_view_receipts': view_records}})
    normalization = put(root, cache + 'normalization.json', {'frontend': asdict(DEFAULT), 'manifest_sha256': sha(manifest)})
    prepared = put(root, cache + 'preparation.json', {'protocol_sha256': sha(protocol),
        'manifest_sha256': sha(manifest), 'synthetic_proxy_manifest_sha256': sha(proxies)})
    binding = {'source_files': {name: sha(root / name) for name in sources},
        'protocol_sha256': sha(protocol), 'manifest_sha256': sha(manifest),
        'synthetic_proxy_manifest_sha256': sha(proxies), 'normalization_sha256': sha(normalization),
        'reservation_sha256': sha(reservation), 'base_view_equivalence_sha256': None, 'base_views': view_records}
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    implementation = put(root, 'experiments/iris-release-v1/owner-implementation-fixture.json',
                         {'binding': binding, 'fingerprint': fingerprint})
    pointer = {'path': (implementation.relative_to(root)).as_posix(), 'sha256': sha(implementation), 'fingerprint': fingerprint}
    execution = put(root, 'experiments/iris-release-v1/owner-execution.json', {'implementation_before_fitting': pointer})
    checkpoint = put(root, cache + 'selected.pt', 'published selected weights')
    selected = put(root, cache + 'selection-before-test.json', {'checkpoint_sha256': sha(checkpoint),
        'manifest_sha256': sha(manifest), 'normalization_sha256': sha(normalization),
        'protocol_sha256': sha(protocol), 'execution_sha256': sha(execution), 'selection_completed_before_test': True})
    files, aliases = [], []
    for path in (prepared, checkpoint, selected, normalization, manifest, proxies):
        destination = root / '.cache/iris-release-v1/inputs/owner-fixture' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files.append({'path': (destination.relative_to(root)).as_posix(), 'sha256': sha(destination)})
        aliases.append({'path': (path.relative_to(root)).as_posix(), 'source': files[-1]['path'], 'sha256': sha(path)})
    files.append({'path': (prefit.relative_to(root)).as_posix(), 'sha256': sha(prefit)})
    receipt = put(root, 'experiments/iris-release-v1/reproduction-inputs.json', {'files': files, 'cache_aliases': aliases})
    driver = root / 'scripts/iris_release.py'
    driver.write_text(driver.read_text() + '\n# Unrelated packaging metadata changed after fitting.\n')
    return receipt, driver


def test_completed_owner_reuse_allows_unrelated_driver_change_without_raw_data(tmp_path, monkeypatch):
    from scripts.iris_release import verify_completed_owner
    from scripts.train import iris_owner
    receipt, _ = completed_owner_fixture(tmp_path)
    monkeypatch.setattr(iris_owner, '_bind_implementation', lambda *a: pytest.fail('rewrote original source lock'))
    result = verify_completed_owner(tmp_path, receipt)
    assert result['status'] == 'verified_completed_owner_preparation'
    assert result['raw_librispeech_hydrated'] is False
    assert not (tmp_path / '.cache/iris-release-v1/owner/audio').exists()


@pytest.mark.parametrize('name', ['scripts/models/iris_owner.py', 'scripts/train/iris_audio.py',
                                  '.cache/iris-release-v1/owner/manifest.jsonl', 'scripts/iris_release.py'])
def test_completed_owner_reuse_rejects_model_frontend_manifest_or_resolver_change(tmp_path, name):
    from scripts.iris_release import verify_completed_owner
    receipt, driver = completed_owner_fixture(tmp_path)
    changed = tmp_path / name
    if changed == driver:
        changed.write_text(changed.read_text().replace('return path\n', 'return path.resolve()\n', 1))
    else:
        changed.write_text(changed.read_text() + '\n# changed critical input\n')
    with pytest.raises(ValueError, match='identity|resolver|source'):
        verify_completed_owner(tmp_path, receipt)


def completed_diversity_fixture(root):
    import shutil
    prefix = '.cache/iris-release-v1/diversity/runs/compact-ce-s53/'
    paths = {'protocol_sha256': 'experiments/iris-release-v1/protocol.json',
             'data_audit_sha256': '.cache/iris-release-v1/diversity/data-audit.json',
             'initializer_sha256': '.cache/iris-release-v1/inputs/compact-initializer.pt',
             'model_source_sha256': 'scripts/models/mvp_static.py',
             'contrastive_source_sha256': 'scripts/models/speaker_demo.py',
             'frontend_sha256': 'vcm/frontend.py', 'policy_source_sha256': 'scripts/train/iris_validation.py'}
    binding = {key: sha(put(root, name, 'frozen source or data')) for key, name in paths.items()}
    source = 'def train():\n    return "frozen fit"\n\ndef evaluate():\n    return "old cache"\n'
    trainer = put(root, 'scripts/train/iris_diversity.py', source)
    prefit = put(root, '.cache/iris-release-v1/inputs/provenance/diversity-prefit-trainer.py', source)
    binding.update(trainer_sha256=sha(trainer), model='compact', arm='ce', seed=53, host='original-host')
    checkpoint = put(root, prefix + 'best.pt', 'selected weights')
    states = [checkpoint, put(root, prefix + 'last.pt', 'optimizer and RNG'),
              put(root, prefix + 'selection.json', {'checkpoint_sha256': sha(checkpoint),
                  'model': 'compact', 'arm': 'ce', 'seed': 53, 'test_used_for_selection': False,
                  'checkpoint': prefix + 'best.pt'}),
              put(root, prefix + 'run-inputs.json', binding), put(root, prefix + 'history.json', [])]
    files = [{'path': (prefit.relative_to(root)).as_posix(), 'sha256': sha(prefit)}]
    aliases = []
    for path in states:
        destination = root / '.cache/iris-release-v1/inputs/fit-fixture' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files.append({'path': (destination.relative_to(root)).as_posix(), 'sha256': sha(destination)})
        aliases.append({'path': (path.relative_to(root)).as_posix(), 'source': files[-1]['path'], 'sha256': sha(path)})
    receipt = put(root, 'experiments/iris-release-v1/reproduction-inputs.json', {'files': files, 'cache_aliases': aliases})
    trainer.write_text(source.replace('old cache', 'bound atomic replay'))
    return receipt, trainer


def test_completed_diversity_allows_only_evaluation_source_amendment(tmp_path):
    from scripts.iris_release import verify_completed_diversity
    receipt, trainer = completed_diversity_fixture(tmp_path)
    result = verify_completed_diversity(tmp_path, receipt, 'compact', 'ce', 53)
    assert result['status'] == 'verified_completed_diversity_fit'
    trainer.write_text(trainer.read_text().replace('frozen fit', 'different loss'))
    with pytest.raises(ValueError, match='fit source'):
        verify_completed_diversity(tmp_path, receipt, 'compact', 'ce', 53)


def test_completed_diversity_rejects_changed_model_source(tmp_path):
    from scripts.iris_release import verify_completed_diversity
    receipt, _ = completed_diversity_fixture(tmp_path)
    (tmp_path / 'scripts/models/mvp_static.py').write_text('changed architecture')
    with pytest.raises(ValueError, match='identity'):
        verify_completed_diversity(tmp_path, receipt, 'compact', 'ce', 53)


def test_diversity_fit_contract_only_normalizes_redundant_selection_keyword():
    from scripts.iris_release import _diversity_fit_contract
    before = '''def train():
    return "frozen fit"

def select():
    chosen = max(selections, key=key)
    write(destination, dict(**chosen, all_runs=selections,
        rule=cfg['selection'], test_used_for_selection=False))
    return chosen

def evaluate():
    return "old replay"
'''
    after = before.replace(", test_used_for_selection=False", "").replace('old replay', 'bound replay')
    assert _diversity_fit_contract(before) == _diversity_fit_contract(after)
    for changed in (
        after.replace('max(selections, key=key)', 'min(selections, key=key)'),
        after.replace("cfg['selection']", "cfg['other_rule']"),
        before.replace('test_used_for_selection=False', 'test_used_for_selection=True'),
    ):
        assert _diversity_fit_contract(before) != _diversity_fit_contract(changed)


def test_interrupted_initializer_replays_from_saved_optimizer_rng_boundary(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    import torch
    from scripts.iris_release import fit_random_initializer
    records = []
    for index, (synthetic, label) in enumerate([(False, 0), (False, 1), (True, 0), (True, 1)]):
        audio = tmp_path / ('recording-' + str(index) + '.wav')
        sf.write(audio, np.sin(np.arange(4000) * .04).astype('float32') * .05, 16000, subtype='PCM_16')
        records.append({'id': str(index), 'audio': audio.name, 'sha256': sha(audio),
                        'speaker': 'speaker-' + str(index), 'synthetic': synthetic, 'class_index': label})
    settings = dict(root=tmp_path, train_rows=records, dev_rows=records,
                    norm={'mean': [0.] * 40, 'std': [1.] * 40}, classes=2, width=128,
                    epochs=1, steps=17, batch=4, device='cpu')
    complete = fit_random_initializer(folder=tmp_path / 'complete', **settings)
    original = torch.optim.AdamW.step
    count = 0
    def interrupt_after_step(optimizer, *args, **kwargs):
        nonlocal count
        result = original(optimizer, *args, **kwargs)
        count += 1
        if count == 17:
            raise RuntimeError('simulated process interruption')
        return result
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt_after_step)
    with pytest.raises(RuntimeError, match='interruption'):
        fit_random_initializer(folder=tmp_path / 'resumed', **settings)
    monkeypatch.setattr(torch.optim.AdamW, 'step', original)
    resumed = fit_random_initializer(folder=tmp_path / 'resumed', **settings)
    expected = torch.load(tmp_path / complete['checkpoint'], map_location='cpu', weights_only=True)
    actual = torch.load(tmp_path / resumed['checkpoint'], map_location='cpu', weights_only=True)
    assert actual['history'] == expected['history']
    assert torch.equal(actual['torch_rng'], expected['torch_rng'])
    assert all(torch.equal(actual['weights'][name], value) for name, value in expected['weights'].items())


def test_from_scratch_creates_separate_release_without_published_initializer(tmp_path):
    from scripts.iris_release import run_workflow
    files = fixture(tmp_path)
    protocol_before = files['protocol'].read_bytes()
    files['inputs'].unlink()
    stages = Stages(tmp_path)
    def execute(command, **kwargs):
        if '--fit-from-scratch' in command:
            identifier = command[command.index('--run-id') + 1]
            protocol = put(tmp_path, 'experiments/iris-release-v1/reproduction-random-protocol.json', {'initialization': 'random'})
            selection = put(tmp_path, '.cache/iris-release-v1/from-scratch/selection.json', {'test_used': False})
            put(tmp_path, 'experiments/iris-release-v1/reproduction-from-scratch-' + identifier + '.json',
                {'bundle': (files['bundle'].relative_to(tmp_path)).as_posix(), 'protocol': (protocol.relative_to(tmp_path)).as_posix(),
                 'selection': (selection.relative_to(tmp_path)).as_posix(), 'historical_v2_reproduced': False})
        return stages(command, **kwargs)
    result = run_workflow(root=tmp_path, from_scratch=True, runner=execute, packager=packager,
                          python=sys.executable, ui_command=['pnpm'], test_modules=['fixture.py'])
    assert 'reproduction-release-from-scratch-' in result['release']
    assert not (tmp_path / 'experiments/iris-release-v1/reproduction-release.json').exists()
    assert files['protocol'].read_bytes() == protocol_before
    assert result['canonical_random_study']['historical_v2_reproduced'] is False
    assert not any('scripts.train.iris_diversity' in command for command in stages.commands)
