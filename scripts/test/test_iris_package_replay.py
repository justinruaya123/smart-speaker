"""The offline audit must fail before extracting or overwriting evidence."""
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def invoke(package, digest, output):
    return subprocess.run([sys.executable, '-m', 'scripts.test.iris_package_replay',
                           '--package', str(package), '--sha256', digest,
                           '--output', str(output)], cwd=ROOT, capture_output=True, text=True)


def test_wrong_hash_is_rejected_before_extraction(tmp_path):
    package = tmp_path / 'candidate.zip'
    package.write_bytes(b'Invalid archive: the digest guard must run first.')
    output = tmp_path / 'new-evidence'
    result = invoke(package, '0' * 64, output)
    assert result.returncode != 0
    assert 'SHA-256' in result.stderr
    assert not output.exists()
    assert package.read_bytes() == b'Invalid archive: the digest guard must run first.'


def test_existing_output_is_preserved_before_any_archive_read(tmp_path):
    package = tmp_path / 'candidate.zip'
    package.write_bytes(b'Not an archive; existing evidence takes priority.')
    output = tmp_path / 'partial-evidence'
    output.mkdir()
    sentinel = output / 'verification.json'
    sentinel.write_bytes(b'preserved interrupted evidence\n')
    result = invoke(package, hashlib.sha256(package.read_bytes()).hexdigest(), output)
    assert result.returncode != 0
    assert 'already exists' in result.stderr
    assert list(output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'preserved interrupted evidence\n'
