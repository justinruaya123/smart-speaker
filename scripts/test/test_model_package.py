"""Model-only distribution boundaries; no fitting, audio or application boot."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import archive_io

ROOT = Path(__file__).resolve().parents[2]
REVISION = '94c0021d7e22c9aa719246ac97105d7140a9f2a2'
FROZEN = ROOT / 'submission/models/optionb-tiny'
FROZEN_NAMES = {'commands.json', 'model.onnx', 'normalization.json', 'policy.json', 'recipe.json'}


def test_writer_does_not_import_application(tmp_path):
    code = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.startswith(('scripts.runtime', 'scripts.mvp_delivery', 'scripts.speaker_publish')):
        raise AssertionError('Application import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from scripts.archive_io import write_archive
import sys
write_archive({'model/test.onnx': b'graph'}, sys.argv[1], {})
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path / 'pure.zip')],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('entries', [
    {'model/a': b'a', 'model/A': b'b'},
    {'MANIFEST.json': b'forged'},
    {'manifest.JSON': b'forged'},
    {'../escape': b'bad'},
    {'model\\graph': b'bad'},
    {'model/CON.onnx': b'bad'},
    {'model/a\x00tail': b'bad'},
    {'model/a\n': b'bad'},
    {'model/graph': b'a', 'MODEL/graph': b'b'},
    {'model/graph': b'a', 'model/graph/child': b'b'},
    {'MANIFEST.json/child': b'forged'},
])
def test_writer_rejects_unsafe_or_ambiguous_inventory(tmp_path, entries):
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError):
        archive_io.write_archive(entries, output, {})
    assert not output.exists()


def test_writer_computes_hashes_and_sizes_not_caller_claims(tmp_path):
    output = tmp_path / 'checked.zip'
    data = b'exact\r\nbytes\x00'
    archive_io.write_archive({'model/graph': data}, output,
                             {'files': {'lie': 'fake'}, 'sizes': {'lie': 123}})
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['files'] == {'model/graph': hashlib.sha256(data).hexdigest()}
        assert manifest['sizes'] == {'model/graph': len(data)}
        assert archive.read('model/graph') == data
    with pytest.raises(FileExistsError):
        archive_io.write_archive({}, output, {})


def test_frozen_archive_preserves_exact_five_payloads_and_contract(tmp_path):
    from scripts import model_package
    output = tmp_path / 'epoch44.zip'
    receipt = model_package.build_archive(FROZEN, output, source_revision=REVISION)
    expected = {'model/' + name for name in FROZEN_NAMES}
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == expected | {'MANIFEST.json'}
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['format'] == 'smart-speaker-model-v1'
        assert manifest['source_revision'] == REVISION
        assert manifest['original_source_revision'] == '3c05b3beb46f06c3108089e0d8c79474670c6cb7'
        assert manifest['qualified'] is False
        assert manifest['actions_enabled'] is False
        assert manifest['contract'] == json.loads((FROZEN / 'recipe.json').read_bytes())
        assert set(manifest['files']) == set(manifest['sizes']) == expected
        for name in FROZEN_NAMES:
            payload = (FROZEN / name).read_bytes()
            member = 'model/' + name
            assert archive.read(member) == payload
            assert manifest['files'][member] == hashlib.sha256(payload).hexdigest()
            assert manifest['sizes'][member] == len(payload)
    assert receipt['sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt['qualified'] is False
    other = tmp_path / 'repeat.zip'
    model_package.build_archive(FROZEN, other, source_revision=REVISION)
    assert other.read_bytes() == output.read_bytes()


def test_legacy_release_is_models_only_and_matches_reproduction_inventory(tmp_path):
    from scripts import model_package
    from scripts.iris_reproduce import compare_bundle
    reference = ROOT / 'artifacts/iris'
    output = tmp_path / 'iris-model.zip'
    evidence = b'{"source_version":{"commit":"' + REVISION.encode() + b'"}}\n'
    receipt = model_package.package_release(
        reference, output, wake_folder=reference / 'wake', owner_folder=reference / 'owner',
        vad_folder=reference / 'wake', extra_files={
            'evidence/reproduction-release.json': evidence,
            'MODEL_CARD.md': b'# Unqualified finite model\n',
            'docs/HANDOFF-iris-rpi-v1.md': b'Application instructions stay with the consumer.\n'})
    expected = json.loads((reference / 'integrity.json').read_bytes())['files']
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            *('models/default/' + name for name in expected),
            'licenses/silero-LICENSE.txt', 'evidence/reproduction-release.json',
            'MODEL_CARD.md', 'MANIFEST.json'}
        for name in expected:
            assert archive.read('models/default/' + name) == (reference / name).read_bytes()
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['format'] == 'smart-speaker-model-v1'
        assert manifest['qualified'] is False
        assert manifest['actions_enabled'] is False
        assert manifest['source_revision'] == REVISION
        assert manifest['contract']['bundle_format'] == 'mvp-static-v1'
        assert archive.read('evidence/reproduction-release.json') == evidence
    comparison = compare_bundle(reference, output)
    assert set(comparison) == set(expected)
    assert all(item['matches'] for item in comparison.values())
    assert receipt['qualified'] is False
    assert receipt['excluded_application_files'] == ['docs/HANDOFF-iris-rpi-v1.md']


def test_frozen_archive_accepts_only_verified_model_companions(tmp_path):
    from scripts import model_package
    reference = ROOT / 'artifacts/iris'
    output = tmp_path / 'companions.zip'
    model_package.build_archive(FROZEN, output, source_revision=REVISION, companions={
        'wake': reference / 'wake', 'owner': reference / 'owner', 'vad': reference / 'wake'})
    integrity = json.loads((reference / 'integrity.json').read_bytes())['files']
    expected = {'model/' + name for name in FROZEN_NAMES}
    expected |= {'model/' + name for name in integrity if name.startswith(('wake/', 'owner/'))}
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == expected | {'MANIFEST.json', 'licenses/silero-LICENSE.txt'}
        for name in integrity:
            if name.startswith(('wake/', 'owner/')):
                assert archive.read('model/' + name) == (reference / name).read_bytes()
        manifest = json.loads(archive.read('MANIFEST.json'))
        assert manifest['qualified'] is False
        assert manifest['companion_source']['wake_metadata_sha256'] == integrity['wake/iris.json']['sha256']


@pytest.mark.parametrize('companions', [{'runtime': ROOT}, {'../wake': ROOT}, {'wake': None}])
def test_frozen_archive_rejects_unknown_or_missing_companions(tmp_path, companions):
    from scripts import model_package
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError):
        model_package.build_archive(FROZEN, output, source_revision=REVISION, companions=companions)
    assert not output.exists()


@pytest.fixture
def iris_export(tmp_path):
    import shutil
    return Path(shutil.copytree(ROOT / 'artifacts/iris', tmp_path / 'iris-export'))


def legacy(folder, output, **kwargs):
    from scripts.model_package import package_release
    return package_release(folder, output, wake_folder=folder / 'wake', owner_folder=folder / 'owner',
                           vad_folder=folder / 'wake', **kwargs)


def change_json(path, edit):
    value = json.loads(path.read_bytes())
    edit(value)
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def rebind(folder, metadata, name):
    change_json(folder / metadata,
                lambda value: value['files'].__setitem__(name, hashlib.sha256((folder / name).read_bytes()).hexdigest()))


@pytest.mark.parametrize('relative', sorted(FROZEN_NAMES))
def test_frozen_payload_substitution_rejected_before_output(tmp_path, relative):
    from scripts.model_package import build_archive
    import shutil
    bundle = Path(shutil.copytree(FROZEN, tmp_path / 'epoch44'))
    target = bundle / relative
    target.write_bytes(target.read_bytes() + b'\n')
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError, match='Frozen payload identity mismatch'):
        build_archive(bundle, output, source_revision=REVISION)
    assert not output.exists()


@pytest.mark.parametrize('revision', ['', 'main', REVISION.upper(), REVISION + '\n', None])
def test_frozen_revision_is_not_repaired_or_resolved(tmp_path, revision):
    from scripts.model_package import build_archive
    with pytest.raises(ValueError, match='source_revision'):
        build_archive(FROZEN, tmp_path / 'bad.zip', source_revision=revision)


@pytest.mark.parametrize('mutation', ['missing', 'checkpoint', 'nested'])
def test_frozen_inventory_is_exact(tmp_path, mutation):
    from scripts.model_package import build_archive
    import shutil
    bundle = Path(shutil.copytree(FROZEN, tmp_path / 'epoch44'))
    if mutation == 'missing':
        (bundle / 'policy.json').unlink()
    elif mutation == 'checkpoint':
        (bundle / 'checkpoint.pt').write_bytes(b'private checkpoint')
    else:
        (bundle / 'application').mkdir()
    with pytest.raises(ValueError, match='five-file inventory'):
        build_archive(bundle, tmp_path / 'bad.zip', source_revision=REVISION)


@pytest.mark.parametrize('metadata', ['bundle.json', 'ncnn/engine.json', 'wake/iris.json', 'owner/owner.json'])
@pytest.mark.parametrize('mutation', ['missing', 'extra', 'bad_hash'])
def test_legacy_exact_export_inventory_and_hashes(iris_export, tmp_path, metadata, mutation):
    path = iris_export / metadata
    def edit(value):
        name = next(iter(value['files']))
        if mutation == 'missing':
            del value['files'][name]
        elif mutation == 'extra':
            value['files']['private/checkpoint.pt'] = value['files'][name]
        else:
            value['files'][name] = hashlib.sha256(b'not the export').hexdigest()
    change_json(path, edit)
    output = tmp_path / 'bad.zip'
    with pytest.raises(ValueError, match='inventory|identity mismatch'):
        legacy(iris_export, output)
    assert not output.exists()


@pytest.mark.parametrize('relative', [
    'model.onnx', 'ncnn/model.ncnn.param', 'ncnn/model.ncnn.bin', 'ncnn/position-frequency.json',
    'wake/iris.onnx', 'wake/iris.ncnn.param', 'wake/iris.ncnn.bin', 'owner/owner.onnx', 'wake/silero-vad.onnx',
])
def test_legacy_rejects_changed_graph_or_export_payload(iris_export, tmp_path, relative):
    path = iris_export / relative
    path.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='identity mismatch'):
        legacy(iris_export, tmp_path / 'bad.zip')


@pytest.mark.parametrize('name', [
    'start.py', 'scripts/runtime/worker.py', 'SpeakerUI/package.json', 'samples/audio.wav',
    'models/default/policy.json', '.env', 'config.toml', 'checkpoint.pt', 'MANIFEST.json',
    'evidence/checkpoint.pt', 'evidence/private/config.json', '../evidence/x.json',
])
def test_legacy_rejects_application_and_private_extras(iris_export, tmp_path, name):
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip', extra_files={name: b'{}'})


@pytest.mark.parametrize('corruption', ['negative_std', 'catalog_slot', 'unqualified_policy', 'class_policy'])
def test_legacy_keeps_semantic_model_validation(iris_export, tmp_path, corruption):
    if corruption == 'negative_std':
        name = 'normalization.json'
        edit = lambda value: value['std'].__setitem__(0, -1)
    elif corruption == 'catalog_slot':
        name = 'catalog.json'
        edit = lambda value: value[1].__setitem__('command', {'intent': 'LIGHT_ON', 'slots': {'light_id': 'contact.home'}})
    elif corruption == 'unqualified_policy':
        name = 'policy.json'
        edit = lambda value: value.__setitem__('qualified', True)
    else:
        name = 'policy.json'
        edit = lambda value: value['by_class'].pop(next(iter(value['by_class'])))
    change_json(iris_export / name, edit)
    rebind(iris_export, 'bundle.json', name)
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip')


@pytest.mark.parametrize('metadata', ['bundle.json', 'wake/iris.json', 'owner/owner.json'])
def test_legacy_rejects_ambiguous_json_metadata(iris_export, tmp_path, metadata):
    target = iris_export / metadata
    data = target.read_bytes()
    target.write_bytes(b'{"files": {},' + data.lstrip()[1:])
    with pytest.raises(ValueError, match='Duplicate JSON'):
        legacy(iris_export, tmp_path / 'bad.zip')


@pytest.mark.parametrize('evidence', [b'{"invalid":NaN}', b'{"x":1,"x":2}', b'{"overflow":1e999}'])
def test_legacy_rejects_ambiguous_or_nonfinite_evidence(iris_export, tmp_path, evidence):
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip', extra_files={'evidence/check.json': evidence})


@pytest.mark.parametrize('edit', [
    lambda value: value['parity'].__setitem__('max_absolute_error', 1.),
    lambda value: value['parity'].__setitem__('passed', 1),
    lambda value: value.__setitem__('onnx_sha256', hashlib.sha256(b'other model').hexdigest()),
    lambda value: value['parity'].__setitem__('tolerance', float('nan')),
])
def test_legacy_rejects_unverified_or_contradictory_ncnn_parity(iris_export, tmp_path, edit):
    change_json(iris_export / 'ncnn/engine.json', edit)
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip')


@pytest.mark.parametrize('metadata,name', [
    ('bundle.json', 'normalization.json'), ('bundle.json', 'policy.json'),
    ('ncnn/engine.json', 'ncnn/position-frequency.json'),
])
def test_legacy_rejects_ambiguous_json_payloads_even_if_rehashed(iris_export, tmp_path, metadata, name):
    target = iris_export / name
    if name.endswith('position-frequency.json'):
        target.write_bytes(b'[NaN]')
    else:
        raw = target.read_bytes()
        target.write_bytes(b'{"unused":1,"unused":2,' + raw.lstrip()[1:])
    folder = (iris_export / metadata).parent
    rebind(folder, Path(metadata).name, Path(name).name)
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip')


@pytest.mark.parametrize('component,edit', [
    ('wake/iris.json', lambda value: value.__setitem__('release', value['threshold'])),
    ('wake/iris.json', lambda value: value['normalization']['std'].__setitem__(0, 0)),
    ('owner/owner.json', lambda value: value['model'].__setitem__('path', '../model.onnx')),
    ('owner/owner.json', lambda value: value['model'].__setitem__('bytes', 1)),
    ('owner/owner.json', lambda value: value['policy'].__setitem__('timeout_seconds', 0)),
    ('owner/owner.json', lambda value: value.__setitem__('calibration_kind', 'undeclared')),
])
def test_legacy_keeps_wake_and_owner_contract_checks(iris_export, tmp_path, component, edit):
    change_json(iris_export / component, edit)
    with pytest.raises(ValueError):
        legacy(iris_export, tmp_path / 'bad.zip')


def test_legacy_ignores_unlisted_source_application_and_training_files(iris_export, tmp_path):
    (iris_export / 'checkpoint.pt').write_bytes(b'must not ship')
    (iris_export / '.env').write_bytes(b'PRIVATE_TOKEN=must-not-ship')
    (iris_export / 'wake/start.py').write_bytes(b'raise SystemExit("must not ship")')
    output = tmp_path / 'bounded.zip'
    legacy(iris_export, output)
    with zipfile.ZipFile(output) as archive:
        assert not any(name.endswith(('.pt', '.env', '.py', '.wav')) for name in archive.namelist())


def test_packagers_never_import_application_or_open_network(tmp_path):
    code = '''
import builtins, socket, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.startswith(('scripts.runtime', 'scripts.mvp_delivery', 'scripts.speaker_publish', 'SpeakerUI')):
        raise AssertionError('Application import: ' + name)
    return original(name, *args, **kwargs)
def offline(*args, **kwargs):
    raise AssertionError('Network access is forbidden')
builtins.__import__ = checked
socket.socket.connect = offline
socket.create_connection = offline
from pathlib import Path
from scripts.model_package import build_archive, package_release
root = Path.cwd()
build_archive(root / 'submission/models/optionb-tiny', Path(sys.argv[1]) / 'frozen.zip',
              source_revision=sys.argv[2])
assert not any(n in sys.modules for n in ('numpy', 'torch', 'onnxruntime'))
reference = root / 'artifacts/iris'
package_release(reference, Path(sys.argv[1]) / 'legacy.zip', wake_folder=reference / 'wake',
                owner_folder=reference / 'owner', vad_folder=reference / 'wake', extra_files={})
assert not any(n.startswith('scripts.runtime') for n in sys.modules)
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path), REVISION],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frozen_linked_payload_is_rejected(tmp_path):
    from scripts.model_package import build_archive
    import shutil
    bundle = Path(shutil.copytree(FROZEN, tmp_path / 'epoch44'))
    target = bundle / 'model.onnx'
    target.unlink()
    try:
        target.symlink_to(FROZEN / 'model.onnx')
    except OSError as error:
        pytest.skip('Host does not permit file symlinks: ' + str(error))
    with pytest.raises(ValueError, match='Linked model source'):
        build_archive(bundle, tmp_path / 'bad.zip', source_revision=REVISION)


def test_read_simulating_absent_path_is_junction_cpython311(monkeypatch, tmp_path):
    from scripts.model_package import _read
    for cls in type(Path()).mro():
        if 'is_junction' in cls.__dict__:
            monkeypatch.delattr(cls, 'is_junction')
    assert not hasattr(Path(), 'is_junction')
    data = _read(FROZEN, 'commands.json')
    assert data == (FROZEN / 'commands.json').read_bytes()

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    target = tmp_path / 'target.json'
    target.write_bytes(b'{}')
    link = bundle / 'commands.json'
    try:
        link.symlink_to(target)
        with pytest.raises(ValueError, match='Linked model source'):
            _read(bundle, 'commands.json')
    except OSError:
        pass


def test_symlink_or_junction_host_ancestor_allows_contained_bundle_and_rejects_internal_link(tmp_path):
    from scripts.model_package import _read, build_archive
    import shutil

    def make_dir_link(target, link):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if sys.platform == 'win32':
                import _winapi
                _winapi.CreateJunction(str(target), str(link))
            else:
                raise

    real_host = tmp_path / 'real_host'
    real_host.mkdir()
    bundle_real = real_host / 'epoch44'
    shutil.copytree(FROZEN, bundle_real)

    link_host = tmp_path / 'link_host'
    make_dir_link(real_host, link_host)
    assert link_host.is_symlink() or getattr(link_host, 'is_junction', lambda: False)()

    bundle_via_host = link_host / 'epoch44'
    assert not bundle_via_host.is_symlink()
    assert not getattr(bundle_via_host, 'is_junction', lambda: False)()

    # Contained regular bundle under symlink/junction host ancestor must be readable.
    data = _read(bundle_via_host, 'commands.json')
    assert data == (FROZEN / 'commands.json').read_bytes()
    output = tmp_path / 'contained.zip'
    receipt = build_archive(bundle_via_host, output, source_revision=REVISION)
    assert output.exists()
    assert receipt['qualified'] is False

    # Internal linked member must still be rejected.
    bundle_linked_member = real_host / 'epoch44_linked_member'
    shutil.copytree(FROZEN, bundle_linked_member)
    (bundle_linked_member / 'commands.json').unlink()
    try:
        (bundle_linked_member / 'commands.json').symlink_to(FROZEN / 'commands.json')
        with pytest.raises(ValueError, match='Linked model source'):
            _read(link_host / 'epoch44_linked_member', 'commands.json')
    except OSError as error:
        # On Windows without symlink privileges, file symlinks cannot be created
        pass

    # Internal linked directory must still be rejected (junction used if symlinks restricted).
    bundle_linked_dir = real_host / 'epoch44_linked_dir'
    bundle_linked_dir.mkdir()
    target_sub = tmp_path / 'target_sub'
    target_sub.mkdir()
    (target_sub / 'item.json').write_bytes(b'{}')
    make_dir_link(target_sub, bundle_linked_dir / 'sub')
    with pytest.raises(ValueError, match='Linked model source'):
        _read(link_host / 'epoch44_linked_dir', 'sub/item.json')

    # Linked bundle root itself must still be rejected.
    bundle_root_link = tmp_path / 'bundle_root_link'
    make_dir_link(bundle_real, bundle_root_link)
    with pytest.raises(ValueError, match='Linked model source'):
        _read(bundle_root_link, 'commands.json')
