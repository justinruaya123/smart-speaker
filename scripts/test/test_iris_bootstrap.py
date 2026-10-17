"""Pinned local uv bootstrap without executing downloaded installers."""
from io import BytesIO
import hashlib
import os
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.runtime import speaker_tools


def wheel_payload(*, binary=True):
    stream = BytesIO()
    with ZipFile(stream, 'w') as wheel:
        if binary:
            wheel.writestr('uv-0.12.5.data/scripts/uv', b'fixture uv executable')
            wheel.writestr('uv-0.12.5.data/scripts/uv.exe', b'fixture uv executable')
        wheel.writestr('uv-0.12.5.data/scripts/uvx', b'unrequested executable')
        wheel.writestr('../../unrequested.txt', b'never extract archive paths')
    return stream.getvalue()


def absent_uv(monkeypatch):
    monkeypatch.setattr(speaker_tools.shutil, 'which', lambda name: None)
    monkeypatch.setattr(speaker_tools.urllib.request, 'urlretrieve',
                        lambda *a, **k: pytest.fail('downloaded shell installer attempted'))
    monkeypatch.setattr(speaker_tools.subprocess, 'run',
                        lambda *a, **k: pytest.fail('downloaded shell installer executed'))


def serve_wheel(monkeypatch, payload, *, digest=None):
    calls = []
    pins = {key: (url, digest or hashlib.sha256(payload).hexdigest())
            for key, (url, _) in getattr(speaker_tools, 'UV_WHEELS', {}).items()}
    monkeypatch.setattr(speaker_tools, 'UV_WHEELS', pins, raising=False)
    def fetch(url, **kwargs):
        calls.append(url)
        return BytesIO(payload)
    monkeypatch.setattr(speaker_tools.urllib.request, 'urlopen', fetch)
    return calls


@pytest.mark.parametrize(('system', 'machine', 'filename', 'executable'), [
    ('Linux', 'x86_64', 'uv-0.12.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'uv'),
    ('Linux', 'aarch64', 'uv-0.12.5-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl', 'uv'),
    ('Darwin', 'x86_64', 'uv-0.12.5-py3-none-macosx_10_12_x86_64.whl', 'uv'),
    ('Darwin', 'arm64', 'uv-0.12.5-py3-none-macosx_11_0_arm64.whl', 'uv'),
    ('Windows', 'AMD64', 'uv-0.12.5-py3-none-win_amd64.whl', 'uv.exe'),
    ('Windows', 'ARM64', 'uv-0.12.5-py3-none-win_arm64.whl', 'uv.exe'),
])
def test_absent_uv_uses_matching_pinned_pypi_wheel_only(tmp_path, monkeypatch, system, machine, filename, executable):
    absent_uv(monkeypatch)
    monkeypatch.setattr(speaker_tools.platform, 'system', lambda: system)
    monkeypatch.setattr(speaker_tools.platform, 'machine', lambda: machine)
    calls = serve_wheel(monkeypatch, wheel_payload())
    original_path = os.environ.get('PATH')

    result = Path(speaker_tools.find_uv(tmp_path))

    assert result == tmp_path / '.runtime/tools/uv' / executable
    assert result.read_bytes() == b'fixture uv executable'
    assert os.access(result, os.X_OK)
    assert len(calls) == 1 and calls[0].startswith('https://files.pythonhosted.org/packages/')
    assert calls[0].endswith('/' + filename)
    assert [p for p in tmp_path.rglob('*') if p.is_file()] == [result]
    assert os.environ.get('PATH') == original_path


def test_changed_wheel_checksum_never_installs_uv(tmp_path, monkeypatch):
    absent_uv(monkeypatch)
    monkeypatch.setattr(speaker_tools.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(speaker_tools.platform, 'machine', lambda: 'x86_64')
    serve_wheel(monkeypatch, wheel_payload(), digest='0' * 64)

    with pytest.raises(RuntimeError, match='checksum'):
        speaker_tools.find_uv(tmp_path)

    assert not any(p.is_file() for p in tmp_path.rglob('*'))


def test_wheel_without_expected_binary_never_installs_other_members(tmp_path, monkeypatch):
    absent_uv(monkeypatch)
    monkeypatch.setattr(speaker_tools.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(speaker_tools.platform, 'machine', lambda: 'x86_64')
    serve_wheel(monkeypatch, wheel_payload(binary=False))

    with pytest.raises(RuntimeError, match='binary'):
        speaker_tools.find_uv(tmp_path)

    assert not any(p.is_file() for p in tmp_path.rglob('*'))


def test_existing_local_uv_is_never_downloaded_or_overwritten(tmp_path, monkeypatch):
    absent_uv(monkeypatch)
    monkeypatch.setattr(speaker_tools.urllib.request, 'urlopen',
                        lambda *a, **k: pytest.fail('existing uv caused a download'))
    existing = tmp_path / '.runtime/tools/uv' / ('uv.exe' if os.name == 'nt' else 'uv')
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'user-owned local uv')

    assert speaker_tools.find_uv(tmp_path) == str(existing)
    assert existing.read_bytes() == b'user-owned local uv'


def test_installed_uv_is_preferred_without_local_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_tools.shutil, 'which', lambda name: '/opt/user-tools/uv')
    monkeypatch.setattr(speaker_tools.urllib.request, 'urlopen',
                        lambda *a, **k: pytest.fail('installed uv caused a download'))

    assert speaker_tools.find_uv(tmp_path) == '/opt/user-tools/uv'
    assert not list(tmp_path.iterdir())


def test_unsupported_host_fails_before_downloading(tmp_path, monkeypatch):
    absent_uv(monkeypatch)
    monkeypatch.setattr(speaker_tools.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(speaker_tools.platform, 'machine', lambda: 'riscv64')
    monkeypatch.setattr(speaker_tools.urllib.request, 'urlopen',
                        lambda *a, **k: pytest.fail('unsupported host caused a download'))

    with pytest.raises(RuntimeError, match='Supported hosts'):
        speaker_tools.find_uv(tmp_path)

    assert not list(tmp_path.iterdir())
