"""Independent decoder tests: artificial tensors, never acoustic qualification."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcvcm.decoder import PrefixCTC


@pytest.mark.parametrize('seed', [11, 53, 71])
def test_shared_prefix_matches_torch_ctc(seed):
    rng = np.random.default_rng(seed)
    sequences = [[1], [1, 2], [1, 1], [1, 2, 3], [2, 2, 3]]
    x = rng.normal(size=(32, 40))
    actual = PrefixCTC(sequences).score(x)
    logp = torch.tensor(x).log_softmax(-1)
    expected = []
    for sequence in sequences:
        loss = torch.nn.functional.ctc_loss(
            logp[:, None], torch.tensor(sequence), torch.tensor([32]),
            torch.tensor([len(sequence)]), reduction='sum')
        expected.append(-float(loss) / 32)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_shared_prefix_arbitrary_chunk_boundaries():
    x = np.random.default_rng(53).normal(size=(43, 40))
    decoder = PrefixCTC([[1, 2, 3], [1, 1, 3], [1, 2, 4]])
    expected = decoder.score(x)
    decoder.reset()
    for i in range(0, len(x), 7):
        actual = decoder.feed(x[i:i + 7])
    np.testing.assert_array_equal(actual, expected)


def test_shared_prefix_invalid_logits_rejected():
    with pytest.raises(ValueError):
        PrefixCTC([[1]]).feed(np.full((3, 40), np.nan))
