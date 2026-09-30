"""Training-only folds must not leak or reuse whole-pool normalization."""
import copy
import numpy as np
import pytest


def rows():
    return [dict(recording_id=f'r{i}', speaker_id=f's{i}', session_id=f'session{i}',
                 utterance_parent_id=f'parent{i}', pcm_sha256=f'pcm{i}',
                 audio_sha256=f'audio{i}', split='train') for i in range(12)]


def test_fixed_folds_hold_each_speaker_once_and_ignore_input_order():
    from pcvcm.cv import speaker_folds
    source = rows()
    folds = speaker_folds(source)
    assert folds == speaker_folds(list(reversed(source)))
    assert len(folds) == 3
    held = [s for f in folds for s in f['held_speakers']]
    assert sorted(held) == sorted(r['speaker_id'] for r in source)
    for f in folds:
        assert len(f['held_speakers']) == 4
        assert not set(f['fit_speakers']) & set(f['held_speakers'])


@pytest.mark.parametrize('field', ['session_id', 'utterance_parent_id', 'pcm_sha256', 'audio_sha256'])
def test_shared_identity_and_nontraining_rows_refused(field):
    from pcvcm.cv import speaker_folds
    source = rows()
    source[1][field] = source[0][field]
    with pytest.raises(ValueError, match='identity|leakage'):
        speaker_folds(source)
    source = rows()
    source[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='training'):
        speaker_folds(source)


def test_normalization_fits_only_fit_speakers():
    from pcvcm.cv import fold_normalization
    source = rows()[:2]
    arrays = {'r0': np.full((40, 10), 3, dtype='float32'),
              'r1': np.full((40, 100), 1000, dtype='float32')}
    norm = fold_normalization(source[:1], arrays)
    assert norm['mean'] == [3.] * 40
    assert norm['fit_speakers'] == ['s0']
    assert norm['frames'] == 10


def test_cv_gate_is_per_fold_and_per_use_case():
    from pcvcm.cv import cv_passed
    good = [dict(macro_exact=.8, worst_use_case=.7, required_phrases=570,
                 observed_phrases=570, observed_use_cases=10) for _ in range(3)]
    assert cv_passed(good)
    bad = copy.deepcopy(good); bad[1]['worst_use_case'] = .699
    assert not cv_passed(bad)
    assert not cv_passed(good[:2])


def test_global_budget_is_shared_and_survives_new_instances(tmp_path):
    from pcvcm.compute import ComputeBudget
    path = tmp_path / 'budget.json'
    with ComputeBudget(path, 'cv', epoch_limit=2, seconds_limit=100) as budget:
        budget.start_epoch('fold0/1')
        budget.check()
    with ComputeBudget(path, 'primary', epoch_limit=2, seconds_limit=100) as budget:
        budget.start_epoch('seed53/1')
        with pytest.raises(RuntimeError, match='compute budget'):
            budget.start_epoch('seed53/2')
    with pytest.raises(ValueError, match='limits'):
        with ComputeBudget(path, 'ablation', epoch_limit=200, seconds_limit=100):
            pass


def test_prediction_metrics_recompute_typed_exactness_and_count_missing_phrases():
    from pcvcm.training import prediction_metrics
    records = [dict(recording_id='r', phrase_id='p', use_case='UC-01',
        reference_class='supported_positive', expected={'intent': 'VOLUME', 'value': 37},
        candidate={'intent': 'VOLUME', 'value': 73}, raw_exact=True)]
    result = prediction_metrics(records, required_phrases={'p', 'missing'}, required_use_cases={'UC-01', 'UC-02'})
    assert result['correct'] == 0
    assert result['macro_exact'] == 0
    assert result['observed_phrases'] == 1
    assert result['required_phrases'] == 2
