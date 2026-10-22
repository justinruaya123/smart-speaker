"""Balanced OptionB demo software fixtures, not acoustic quality evidence."""
import importlib

import numpy as np
import pytest


def module():
    try:
        return importlib.import_module('scripts.test.speaker_optionb_tiny_demo')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_tiny_demo':
            raise
        pytest.fail('Balanced OptionB demo selector is missing')


def fixture():
    rows, scores = [], {}
    for split in ('dev', 'calibration'):
        for i, (target, winner, value) in enumerate(((1, 1, 4.), (1, 1, 8.), (0, 0, 3.), (0, 1, 2.))):
            identifier = f'{split}-{i}'
            rows.append(dict(id=identifier, split=split, kind='human', training_eligible=False,
                             metric_eligible=True, class_index=target))
            values = np.zeros(32)
            values[winner] = value
            scores[identifier] = values.tolist()
    return rows, scores


def test_balanced_complete_command_f1_accepts_correct_optionb_below_099():
    rows, scores = fixture()
    result = module().select_demo_policy(rows, scores,
        dict(probability_grid=[.99, .5, 0.], margin_grid=[0., .3]))
    assert result['policy'] == dict(enabled=True, min_probability=.5, min_margin=.3)
    assert result['metrics']['pooled']['tp'] == 4
    assert result['metrics']['pooled']['fp'] == result['metrics']['pooled']['fn'] == 0
    assert result['metrics']['pooled']['f1'] == 1.
    assert result['selected_on'] == ['dev', 'calibration']
    assert result['optimizer_updates'] == 0


@pytest.mark.parametrize('change', ['test_split', 'synthetic', 'training', 'duplicate', 'extra_score'])
def test_policy_selection_rejects_nonfitting_scope_or_identity_drift(change):
    rows, scores = fixture()
    if change == 'test_split':
        rows[0]['split'] = 'test'
    elif change == 'synthetic':
        rows[0]['kind'] = 'synthetic'
    elif change == 'training':
        rows[0]['training_eligible'] = True
    elif change == 'duplicate':
        rows.append(dict(rows[0]))
    else:
        scores['extra-id'] = scores[rows[0]['id']]
    with pytest.raises(ValueError, match='unique nonfitting human dev/calibration'):
        module().select_demo_policy(rows, scores, dict(probability_grid=[.5], margin_grid=[0.]))


def cache_fixture():
    rows, scores = fixture()
    cached = [dict(id=r['id'], split=r['split'], class_index=r['class_index'], logits=scores[r['id']])
              for r in rows]
    return rows, scores, cached


def test_cached_onnx_scores_require_exact_reference_alignment():
    rows, scores, cached = cache_fixture()
    assert module().validated_cached_scores(rows, cached) == scores


@pytest.mark.parametrize('change', ['class', 'split', 'duplicate', 'missing', 'extra', 'shape', 'nan'])
def test_cached_score_tampering_is_rejected(change):
    rows, _, cached = cache_fixture()
    if change == 'class':
        cached[0]['class_index'] = 0
    elif change == 'split':
        cached[0]['split'] = 'test'
    elif change == 'duplicate':
        cached.append(dict(cached[0]))
    elif change == 'missing':
        cached.pop()
    elif change == 'extra':
        cached.append(dict(cached[0], id='extra'))
    elif change == 'shape':
        cached[0]['logits'] = [0.] * 31
    else:
        cached[0]['logits'][0] = float('nan')
    with pytest.raises(ValueError):
        module().validated_cached_scores(rows, cached)
