"""The real baseline network, causal frontend, and training loss—not speech accuracy."""
from pathlib import Path
import numpy as np
import pytest
from pcvcm.data import verified_features, batch_arrays
from pcvcm.training import training_smoke, select_checkpoint


def test_baseline_parameter_count_cache_equivalence_and_loss():
    report = training_smoke(steps=12)
    assert report['parameters'] == 113064
    assert report['cache_bytes'] == 32064
    assert report['maximum_chunk_error'] < 2e-5
    assert report['final_loss'] < report['initial_loss']
    assert report['human_recordings'] == 0
    assert report['qualified'] is False


def test_selection_first_protects_phrase_coverage_then_worst_use_case():
    a = {'phrases_passing': 569, 'worst_use_case': .98, 'macro_exact': .99,
         'correct': 10000, 'seed': 53, 'epoch': 40}
    b = {'phrases_passing': 570, 'worst_use_case': .96, 'macro_exact': .98,
         'correct': 9999, 'seed': 71, 'epoch': 36}
    assert select_checkpoint([a, b]) == b


def test_feature_hash_mismatch_refused(tmp_path):
    path = tmp_path / 'feature.npy'
    np.save(path, np.zeros((40, 30), dtype='float32'))
    with pytest.raises(ValueError):
        verified_features({'features': 'feature.npy', 'features_sha256': 'wrong'}, tmp_path)


def test_batch_padding_and_ctc_lengths():
    arrays = [np.ones((40, 5), dtype='float32'), np.ones((40, 8), dtype='float32')]
    padded, lengths = batch_arrays(arrays)
    assert padded.shape == (2, 40, 8)
    assert lengths == [5, 8]
    assert np.all(padded[0, :, 5:] == 0)


def test_training_sampling_cannot_let_one_use_case_dominate():
    from pcvcm.training import sampling_weights
    rows = []
    for uc, n in [('UC-01', 2), ('UC-04', 20)]:
        for i in range(n):
            rows.append(dict(reference_class='supported_positive', use_case=uc,
                             phrase_id=f'{uc}-{i}', speaker_id='s',
                             expected_command={'value': i}))
    rows.append(dict(reference_class='unknown_no_action', use_case='UC-01',
                     phrase_id=None, speaker_id='n', expected_command=None))
    weights = sampling_weights(rows, .8, 1.)
    assert abs(weights.sum() - 1) < 1e-12
    assert abs(weights[:2].sum() - .4) < 1e-12
    assert abs(weights[2:22].sum() - .4) < 1e-12
    assert abs(weights[-1] - .2) < 1e-12
