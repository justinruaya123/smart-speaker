"""Regenerate missing review artifacts from committed source, never overwrite."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .contract import Contract, file_hash


def _stage_generator(spec: Path, destination: Path) -> None:
    shutil.copyfile(spec / 'build_contract.py', destination / 'build_contract.py')
    parts = spec / 'generator_parts'
    if parts.is_dir():
        shutil.copytree(parts, destination / 'generator_parts')


def _generated_artifacts(directory: Path) -> list[Path]:
    return [path for path in sorted(directory.iterdir()) if path.is_file()]


def initialize(root: Path) -> dict:
    spec = root / 'spec'
    binding = json.loads((root / 'reports/contract-binding.json').read_text())
    if all((spec / name).exists() for name in binding['files']):
        for name, expected in binding['files'].items():
            if file_hash(spec / name) != expected:
                raise ValueError(f'Existing frozen spec changed: {name}')
        return {'status': 'VERIFIED_EXISTING', 'contract_sha256': Contract(spec).sha256}
    with tempfile.TemporaryDirectory(prefix='pcv1-spec-') as temporary:
        temp = Path(temporary)
        _stage_generator(spec, temp)
        subprocess.run([sys.executable, str(temp / 'build_contract.py')], check=True)
        for name, expected in binding['files'].items():
            if file_hash(temp / name) != expected:
                raise ValueError(f'Generated spec differs: {name}')
        for path in _generated_artifacts(temp):
            target = spec / path.name
            if target.exists():
                if file_hash(target) != file_hash(path):
                    raise ValueError(f'Refusing to overwrite existing spec: {target}')
            else:
                with target.open('xb') as stream:
                    stream.write(path.read_bytes())
    return {'status': 'GENERATED_AND_VERIFIED', 'contract_sha256': Contract(spec).sha256}
