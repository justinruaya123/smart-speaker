"""Test deployed control flow without substituting fixtures for acoustic evidence."""
from pathlib import Path

import numpy as np
import pytest

from pcvcm.contract import Contract, compile_catalog
from pcvcm.endpoint import EndpointConfig

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_does_not_finalize_at_eof_and_fault_is_sticky():
    from pcvcm.runtime import StreamEngine
    contract = Contract(ROOT / 'spec')
    catalog = compile_catalog(contract, ROOT / 'configs/lexicon.json')
    calls = []

    def encoder(features):
        calls.append(features.shape[1])
        return np.zeros((features.shape[1], 40), dtype=np.float32)

    engine = StreamEngine(contract, catalog, encoder,
                          {'mean': [0.] * 40, 'std': [1.] * 40},
                          EndpointConfig(),
                          {'binding': 'x', 'enabled': False}, 'x')
    result = engine.feed(np.ones(8000, dtype=np.float32) * .1, 0)
    assert not result['accepted']
    assert result['reason'] == 'tentative'
    assert not engine.eof()['accepted']
    assert engine.eof()['reason'] == 'incomplete'
    result = engine.feed(np.zeros(320, dtype=np.float32), 9000)
    assert result['reason'] == 'capture_fault'
    assert engine.feed(np.zeros(16000, dtype=np.float32), 8000)['reason'] == 'capture_fault'
    engine.reset()
    assert engine.sequence == 0


def test_endpoint_truncation_does_not_decode_late_audio():
    from pcvcm.runtime import StreamEngine
    contract = Contract(ROOT / 'spec')
    catalog = compile_catalog(contract, ROOT / 'configs/lexicon.json')
    lengths = []

    def encoder(features):
        lengths.append(features.shape[1])
        return np.zeros((features.shape[1], 40), dtype=np.float32)

    engine = StreamEngine(contract, catalog, encoder,
                          {'mean': [0.] * 40, 'std': [1.] * 40},
                          EndpointConfig(),
                          {'binding': 'x', 'enabled': False}, 'x')
    pcm = np.r_[np.ones(8000) * .1, np.zeros(24000)].astype(np.float32)
    result = engine.feed(pcm, 0)
    assert result['reason'] == 'policy_disabled'
    assert engine.endpoint.end_sample == 19200
    assert sum(lengths) == 1 + (19200 - 400) // 160
    assert engine.diagnostic['endpoint_detected']
    assert all(n <= 4 for n in lengths)
    with pytest.raises(ValueError):
        engine.feed(np.zeros(320), len(pcm))


def test_export_does_not_claim_success_without_a_selected_checkpoint(tmp_path):
    from pcvcm.exporting import export_selected
    with pytest.raises((FileNotFoundError, ValueError)):
        export_selected(tmp_path / 'absent', tmp_path / 'bundle',
                        Contract(ROOT / 'spec'), ROOT / 'configs/lexicon.json',
                        EndpointConfig())
    assert not (tmp_path / 'bundle' / 'recipe.json').exists()
