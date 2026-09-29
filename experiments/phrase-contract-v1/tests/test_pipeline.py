"""Pipeline identity and CLI guards; real audio/ONNX tests remain separate."""
import json
from pathlib import Path

import pytest

from pcvcm.contract import Contract, file_hash

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_requires_its_frozen_manifest(tmp_path):
    from pcvcm.pipeline import verify_confirmation
    path = tmp_path / 'final.jsonl'
    path.write_text('{}\n')
    contract = Contract(ROOT / 'spec')
    receipt = {'contract_sha256': contract.sha256,
               'confirmation_manifest_sha256': 'wrong'}
    with pytest.raises(ValueError):
        verify_confirmation(path, receipt, contract)


def test_confirmation_commitment_has_independent_speaker_identities(tmp_path):
    from pcvcm.pipeline import verify_confirmation
    path = tmp_path / 'final.jsonl'
    path.write_text('{}\n')
    contract = Contract(ROOT / 'spec')
    receipt = {'contract_sha256': contract.sha256,
               'confirmation_manifest_sha256': file_hash(path)}
    with pytest.raises(ValueError):
        verify_confirmation(path, receipt, contract)


def test_package_refuses_false_release_claim_before_onnx_loading(tmp_path):
    from pcvcm.pipeline import package_diagnostic
    with pytest.raises(ValueError, match='qualified'):
        package_diagnostic(tmp_path, tmp_path / 'runtime.zip',
                           Contract(ROOT / 'spec'), qualified=True)
