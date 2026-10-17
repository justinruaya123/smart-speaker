"""Fresh training must not hydrate completed VCM fits or mutate source evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest


def put(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else json.dumps(content).encode())
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_input_plan_keeps_initializer_but_excludes_completed_fits_and_scores(tmp_path):
    from scripts.iris_reproduce import training_inputs
    items = [
        {'path': '.cache/iris-release-v1/inputs/compact-initializer.pt', 'role': 'vcm_initializer'},
        {'path': '.cache/iris-release-v1/inputs/completed-stages/diversity/runs/compact-ce-s53/best.pt', 'role': 'vcm_completed_run'},
        {'path': '.cache/iris-release-v1/inputs/selected/vcm/model.onnx', 'role': 'vcm_selected'},
        {'path': '.cache/iris-release-v1/inputs/selected/vcm-candidate/model.onnx', 'role': 'vcm_candidate'},
        {'path': '.cache/iris-release-v1/inputs/completed-stages/diversity/official-results/results.json'},
        {'path': '.cache/iris-release-v1/inputs/fallback-bundles/compact/model.onnx', 'role': 'vcm_fallback'},
        {'path': '.cache/iris-release-v1/inputs/completed-stages/owner/selected.pt'}]
    base = {'files': items, 'cache_aliases': [
        {'source': item['path'], 'path': '.cache/iris-release-v1/alias-' + str(i)} for i, item in enumerate(items)]}
    result = training_inputs(base)
    assert [i['path'] for i in result['files']] == [items[0]['path'], items[5]['path'], items[6]['path']]
    assert [i['path'] for i in result['cache_aliases']] == [
        '.cache/iris-release-v1/alias-0', '.cache/iris-release-v1/alias-5', '.cache/iris-release-v1/alias-6']
    assert result['force_vcm_fitting'] is True
    assert len(base['files']) == 7


def test_git_recipe_bytes_are_restored_without_a_huggingface_download(tmp_path):
    from scripts.iris_reproduce import hydrate_recipe_sources
    source = put(tmp_path, 'recipe/source.py', b'print("frozen recipe")\n')
    target = '.cache/iris-release-v1/inputs/provenance/source.py'
    item = {'source_kind': 'git', 'source_path': 'recipe/source.py', 'input_path': target,
            'sha256': sha(source), 'bytes': source.stat().st_size}
    hydrate_recipe_sources(tmp_path, {'files': [item]}, fetch=lambda _: pytest.fail('network used for Git-owned code'))
    assert (tmp_path / target).read_bytes() == b'print("frozen recipe")\n'


def test_changed_recipe_cache_is_never_overwritten(tmp_path):
    from scripts.iris_reproduce import hydrate_recipe_sources
    source = put(tmp_path, 'recipe/source.py', b'good')
    target = put(tmp_path, '.cache/iris-release-v1/inputs/provenance/source.py', b'changed')
    with pytest.raises(ValueError, match='identity'):
        hydrate_recipe_sources(tmp_path, {'files': [{'source_kind': 'git', 'source_path': 'recipe/source.py',
            'input_path': str(target.relative_to(tmp_path)), 'sha256': sha(source), 'bytes': 4}]})
    assert target.read_bytes() == b'changed'


def test_upstream_recipe_refuses_unpinned_or_non_github_code(tmp_path):
    from scripts.iris_reproduce import hydrate_recipe_sources
    item = {'source_kind': 'upstream_github', 'input_path': '.cache/iris-release-v1/inputs/scorer.py',
            'sha256': 'a' * 64, 'bytes': 1, 'upstream': {'revision': 'a' * 40,
                'raw_url': 'https://huggingface.co/example/resolve/main/scorer.py'}}
    with pytest.raises(ValueError, match='GitHub'):
        hydrate_recipe_sources(tmp_path, {'files': [item]}, fetch=lambda _: pytest.fail('unsafe fetch'))


def test_no_download_does_not_fetch_missing_upstream_recipe(tmp_path):
    from scripts.iris_reproduce import hydrate_recipe_sources
    item = {'source_kind': 'upstream_github', 'input_path': '.cache/iris-release-v1/inputs/scorer.py',
            'sha256': 'a' * 64, 'bytes': 1, 'upstream': {'revision': 'a' * 40,
                'raw_url': 'https://raw.githubusercontent.com/owner/repo/' + 'a' * 40 + '/scorer.py'}}
    with pytest.raises(ValueError, match='no-download'):
        hydrate_recipe_sources(tmp_path, {'files': [item]}, download=False,
                               fetch=lambda _: pytest.fail('network despite no-download'))


def test_bundle_comparison_detects_changed_native_graph_and_wake_asset(tmp_path):
    from scripts.iris_reproduce import compare_bundle
    reference = tmp_path / 'reference'
    payloads = {'model.onnx': b'command', 'ncnn/model.ncnn.bin': b'native',
                'wake/iris.onnx': b'wake', 'wake/silero-vad.onnx': b'vad', 'owner/owner.onnx': b'owner'}
    inventory = {}
    for name, data in payloads.items():
        p = put(reference, name, data)
        inventory[name] = {'sha256': sha(p), 'bytes': len(data)}
    put(reference, 'integrity.json', {'files': inventory})
    package = tmp_path / 'delivery.zip'
    with zipfile.ZipFile(package, 'w') as archive:
        for name, data in {**payloads, 'ncnn/model.ncnn.bin': b'changed-native',
                           'wake/iris.onnx': b'changed-wake'}.items():
            archive.writestr('models/default/' + name, data)
    comparison = compare_bundle(reference, package)
    assert comparison['model.onnx']['matches'] is True
    assert comparison['ncnn/model.ncnn.bin']['matches'] is False
    assert comparison['wake/iris.onnx']['matches'] is False
    assert comparison['wake/silero-vad.onnx']['matches'] is True
    assert comparison['owner/owner.onnx']['matches'] is True


def test_derived_results_are_recoverable_and_unlisted_protocols_stay_untouched(tmp_path):
    from scripts.iris_reproduce import preserve_derived_results
    result = put(tmp_path, 'experiments/study/result.json', {'old': True})
    protocol = put(tmp_path, 'experiments/study/protocol.json', {'frozen': True})
    receipt = preserve_derived_results(tmp_path, tmp_path / 'reference', ['experiments/study/result.json'])
    assert not result.exists()
    assert (tmp_path / 'reference/experiments/study/result.json').read_bytes() == b'{"old": true}'
    assert protocol.read_bytes() == b'{"frozen": true}'
    assert receipt['experiments/study/result.json'] == hashlib.sha256(b'{"old": true}').hexdigest()
