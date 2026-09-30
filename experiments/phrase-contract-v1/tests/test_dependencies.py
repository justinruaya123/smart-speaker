from pathlib import Path
import zipfile
import pytest


def wheel(path, name, version, tag, requires=()):
    filename = path / f'{name}-{version}-{tag}.whl'
    with zipfile.ZipFile(filename, 'w') as z:
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.10\n'
        metadata += ''.join(f'Requires-Dist: {r}\n' for r in requires)
        z.writestr(f'{name}-{version}.dist-info/METADATA', metadata)
        z.writestr(f'{name}-{version}.dist-info/licenses/LICENSE', 'Test license')
    return filename


def test_target_lock_checks_wheel_tags_transitives_and_hashes(tmp_path):
    from pcvcm.dependencies import lock_wheels, verify_lock
    wheels = tmp_path / 'wheels'; wheels.mkdir()
    a = wheel(wheels, 'numpy', '2.5.3', 'cp312-cp312-manylinux_2_28_aarch64')
    wheel(wheels, 'einops', '0.8.2', 'py3-none-any', ['numpy>=2'])
    out = tmp_path / 'lock'
    result = lock_wheels(wheels, out, required={'numpy', 'einops'})
    assert result['target']['machine'] == 'aarch64'
    assert '--hash=sha256:' in (out / 'requirements.txt').read_text()
    verify_lock(out, wheels)
    a.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        verify_lock(out, wheels)


def test_x64_wheel_and_missing_transitive_dependency_fail(tmp_path):
    from pcvcm.dependencies import lock_wheels
    path = wheel(tmp_path, 'numpy', '2.5.3', 'cp312-cp312-win_amd64')
    with pytest.raises(ValueError, match='target'):
        lock_wheels(tmp_path, tmp_path / 'out', required={'numpy'})
    path.unlink()
    wheel(tmp_path, 'numpy', '2.5.3', 'cp312-cp312-manylinux_2_28_aarch64', ['missing>=1'])
    with pytest.raises(ValueError, match='dependency'):
        lock_wheels(tmp_path, tmp_path / 'out', required={'numpy'})


def test_runtime_notice_outside_dist_info_is_preserved(tmp_path):
    from pcvcm.dependencies import lock_wheels
    path = wheel(tmp_path, 'onnxruntime', '1.30.0', 'cp312-cp312-manylinux_2_28_aarch64')
    with zipfile.ZipFile(path, 'a') as z:
        z.writestr('onnxruntime/ThirdPartyNotices.txt', 'Additional dependencies notice')
    lock_wheels(tmp_path, tmp_path / 'out', required={'onnxruntime'})
    assert 'Additional dependencies notice' in (tmp_path / 'out/THIRD_PARTY_NOTICES.txt').read_text()
