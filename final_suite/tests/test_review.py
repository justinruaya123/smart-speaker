"""Selection and evidence guardrails: a software pass is not deployment."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('review', ROOT / 'final_suite/review.py')


def load_module():
    assert SPEC.origin and Path(SPEC.origin).is_file(), 'review implementation absent'
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def test_research_selection_uses_development_not_test():
    review = load_module()
    candidates = [dict(id='parent', dev_error=.25, test_accuracy=.8, protocol_id='matched'),
                  dict(id='adapted', dev_error=.28, test_accuracy=.99, protocol_id='matched')]
    assert review.select_research(candidates)['id'] == 'parent'


def test_missing_measures_cannot_pass_release():
    review = load_module()
    assert review.release_decision({})['eligible'] is False


@pytest.mark.parametrize('bad', [True, '1', -1, float('nan')])
def test_invalid_selection_metric_is_rejected(bad):
    review = load_module()
    with pytest.raises(ValueError):
        review.select_research([dict(id='invalid', dev_error=bad)])


def test_raw_success_is_not_accepted_success():
    review = load_module()
    row = dict(id='x', text='volume fifty', target={'value': 50},
               candidate={'value': 50}, raw_correct=True,
               acceptance_enabled=False, status='needs_clarification')
    normalized = review.normalize_row(row, 'v6', 'test')
    assert normalized['raw_exact']
    assert not normalized['accepted']
    assert normalized['outcome'] == 'correct_candidate_rejected'


def test_negatives_are_not_in_valid_recall_denominator():
    review = load_module()
    a = review.normalize_row(dict(id='a', text='a', target={'x': 1},
        candidate={'x': 1}, status='needs_clarification'), 'v6', 'test')
    b = review.normalize_row(dict(id='b', text='b', target=None,
        candidate={'x': 1}, status='needs_clarification'), 'v6', 'test')
    metrics = review.summarize([a, b])
    assert metrics['valid'] == 1 and metrics['negatives'] == 1
    assert metrics['raw_correct'] == 1 and metrics['accepted_correct'] == 0


def test_duplicates_are_not_new_independent_test_examples():
    review = load_module()
    rows = [dict(trial='v4', id='a', pcm_sha256='same'),
            dict(trial='v5', id='a', pcm_sha256='same')]
    assert review.duplicate_audit(rows)['unique_recordings_by_hash'] == 1


def test_negative_raw_candidate_does_not_count_as_false_accept():
    review = load_module()
    row = review.normalize_row(dict(id='x', text='nonsense', target=None,
        candidate={'value': 50}, status='needs_clarification'), 'v6', 'test')
    assert row['outcome'] == 'negative_rejected'


def test_inconsistent_recorded_exact_flag_is_reported():
    review = load_module()
    row = review.normalize_row(dict(id='x', text='x', target={'value': 73},
        candidate={'value': 3}, raw_correct=True), 'v6', 'test')
    assert row['inconsistent_saved_flag'] is True
    assert row['raw_exact'] is False

def test_selection_requires_matching_protocols():
    review = load_module()
    with pytest.raises(ValueError):
        review.select_research([
            dict(id='a', dev_error=.1, protocol_id='catalog-a'),
            dict(id='b', dev_error=.2, protocol_id='catalog-b')])


def test_selection_refuses_missing_protocol_identity():
    review = load_module()
    with pytest.raises(ValueError):
        review.select_research([dict(id='a', dev_error=.1)])


def test_textual_accepted_boolean_is_not_silently_true():
    review = load_module()
    with pytest.raises(ValueError):
        review.normalize_row(dict(id='x', accepted='false'), 'x', 'x')


def test_ambiguity_requires_an_actual_reported_tie():
    review = load_module()
    result = review.normalize_row(dict(id='x', category='ambiguous',
        candidate={'id': 'only-one'}), 'x', 'x')
    assert result['outcome'] == 'ambiguous_rejected_without_tie'


def test_schema_of_every_retained_prediction():
    review = load_module()
    root = ROOT / 'experiments/vcm_services_v6/reports'
    if not root.exists():
        pytest.skip('Full evidence payload not restored')
    rows, sources = review.read_predictions(ROOT)
    assert len(rows) >= 36972
    assert len(sources) >= 79
    assert not any(r['inconsistent_saved_flag'] for r in rows)
    assert not any(r['action_executed'] for r in rows)


def test_boolean_parameter_is_not_counted_as_integer_match():
    review = load_module()
    result = review.normalize_row({
        "id": "typed", "target": {"intent": "SET_VOLUME", "value": 1},
        "candidate": {"intent": "SET_VOLUME", "value": True},
    }, "test", "test")
    assert result["raw_exact"] is False


def test_nonfinite_parameter_cannot_enter_evidence():
    review = load_module()
    with pytest.raises(ValueError):
        review.normalize_row({
            "id": "typed", "target": {"value": 1},
            "candidate": {"value": float("nan")},
        }, "test", "test")
