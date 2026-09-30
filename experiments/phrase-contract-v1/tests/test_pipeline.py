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


def test_calibration_refuses_unselected_endpoint_before_replay(tmp_path, monkeypatch):
    from pcvcm.pipeline import calibrate_bundle
    import pcvcm.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'verify_bundle', lambda *a: {'binding': 'x'})
    with pytest.raises((FileNotFoundError, ValueError), match='endpoint|Endpoint'):
        calibrate_bundle(tmp_path, tmp_path / 'manifest', tmp_path, tmp_path / 'out', Contract(ROOT / 'spec'))


def test_confirmation_access_is_reserved_once_before_predictions(tmp_path):
    from pcvcm.pipeline import reserve_confirmation
    ledger = tmp_path / 'access'
    receipt = reserve_confirmation('manifesthash', 'binding', 'policy', ledger)
    assert receipt.exists()
    assert json.loads(receipt.read_text())['state'] == 'INFERENCE_RESERVED_NOT_UNTOUCHED'
    with pytest.raises(FileExistsError):
        reserve_confirmation('manifesthash', 'other-candidate', 'policy', ledger)
