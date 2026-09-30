"""Frozen contract generator staging must include its immutable payload parts."""
from pathlib import Path
import json
import shutil
import pytest


def test_fresh_initialization_reproduces_frozen_bytes_on_host(tmp_path):
    from pcvcm.initialize import initialize, _stage_generator
    from pcvcm.contract import file_hash
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'spec').mkdir()
    (tmp_path / 'reports').mkdir()
    _stage_generator(root / 'spec', tmp_path / 'spec')
    shutil.copyfile(root / 'reports/contract-binding.json',
                    tmp_path / 'reports/contract-binding.json')
    result = initialize(tmp_path)
    binding = json.loads((tmp_path / 'reports/contract-binding.json').read_text())
    assert result['contract_sha256'] == binding['contract_sha256']
    assert all(file_hash(tmp_path / 'spec' / n) == h for n, h in binding['files'].items())
    (tmp_path / 'spec/allowed_commands.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Existing frozen spec changed'):
        initialize(tmp_path)


def test_generator_staging_copies_payload_parts(tmp_path):
    from pcvcm.initialize import _stage_generator

    spec = tmp_path / 'spec'
    parts = spec / 'generator_parts'
    parts.mkdir(parents=True)
    (spec / 'build_contract.py').write_text('print("ok")\n', encoding='utf-8')
    (parts / 'part00.b64').write_text('YWJj', encoding='ascii')
    (parts / 'part01.b64').write_text('ZA==', encoding='ascii')
    destination = tmp_path / 'staged'
    destination.mkdir()

    _stage_generator(spec, destination)

    assert (destination / 'build_contract.py').read_text() == 'print("ok")\n'
    assert (destination / 'generator_parts' / 'part00.b64').read_text() == 'YWJj'
    assert (destination / 'generator_parts' / 'part01.b64').read_text() == 'ZA=='


def test_generated_artifact_scan_ignores_staging_directories(tmp_path):
    from pcvcm.initialize import _generated_artifacts

    (tmp_path / 'generator_parts').mkdir()
    (tmp_path / 'generator_parts' / 'part00.b64').write_text('YWJj')
    (tmp_path / 'artifact.json').write_text('{}\n')

    assert [path.name for path in _generated_artifacts(tmp_path)] == ['artifact.json']
