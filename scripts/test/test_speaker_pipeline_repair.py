"""Selection must favor recognition; promotion cannot spend the error budget."""
import copy
import importlib

import pytest


def adapter():
    assert importlib.util.find_spec('scripts.train.speaker_pipeline_repair') is not None, 'pipeline repair adapter is missing'
    return importlib.import_module('scripts.train.speaker_pipeline_repair')


def metric(raw=144, macro=.66, exact=69, precision=.945):
    return dict(human=dict(raw_exact=raw, raw_macro_speaker_recall=macro,
                          raw_negative_errors=40, exact=exact, precision=precision,
                          macro_speaker_recall=.3, negative_errors=0, false_accept_rate=0.),
                synthetic=dict(raw_recall=.93, recall=.8))


def test_recognition_beats_sparse_perfect_precision():
    weak = metric(raw=125, macro=.528, exact=13, precision=1.)
    strong = metric()
    assert adapter().recognition_key(strong, 24) > adapter().recognition_key(weak, 8)


def test_speaker_balanced_recognition_precedes_pooled_count():
    assert adapter().recognition_key(metric(raw=140, macro=.7), 4) > adapter().recognition_key(metric(raw=150, macro=.6), 8)


def test_recognition_selection_is_independent_of_acceptance_confidence():
    original = metric()
    changed = copy.deepcopy(original)
    changed['human'].update(exact=0, precision=None, macro_speaker_recall=0., negative_errors=8)
    changed['synthetic']['recall'] = .1
    assert adapter().recognition_key(original, 4) == adapter().recognition_key(changed, 4)
    assert adapter().recognition_key(original, 4) > adapter().recognition_key(original, 8)


def views():
    return dict(development=dict(correct=96, wrong=6, false_accepts=1),
                calibration=dict(correct=40, wrong=0, false_accepts=0))


def test_promotion_requires_a_real_gain():
    baseline = views()
    assert adapter().promotion_gate(baseline, baseline)['pass'] is False
    candidate = copy.deepcopy(baseline)
    candidate['calibration']['correct'] = 41
    assert adapter().promotion_gate(candidate, baseline)['pass'] is True


@pytest.mark.parametrize('view,field,value', [
    ('development', 'correct', 95), ('calibration', 'correct', 39),
    ('development', 'wrong', 7), ('calibration', 'wrong', 1),
    ('development', 'false_accepts', 2), ('calibration', 'false_accepts', 1),
])
def test_promotion_does_not_hide_per_view_regressions(view, field, value):
    baseline = views(); candidate = copy.deepcopy(baseline)
    candidate['development']['correct'] = 100
    candidate['calibration']['correct'] = 41
    candidate[view][field] = value
    assert adapter().promotion_gate(candidate, baseline)['pass'] is False


def rows():
    return [dict(id='human/1', synthetic=False, expected={'intent':'TIME','slots':{}}),
            dict(id='human/2', synthetic=False, expected=None)]


def test_comparison_allows_order_changes_but_keeps_references():
    assert adapter().verify_comparable(rows(), list(reversed(rows()))) is None


@pytest.mark.parametrize('change', ['reference', 'missing', 'duplicate', 'synthetic'])
def test_comparison_rejects_changed_populations(change):
    candidate = rows()
    if change == 'reference': candidate[0]['expected'] = None
    if change == 'missing': candidate.pop()
    if change == 'duplicate': candidate.append(copy.deepcopy(candidate[0]))
    if change == 'synthetic': candidate[0]['synthetic'] = True
    with pytest.raises(ValueError):
        adapter().verify_comparable(candidate, rows())


def test_regression_gate_does_not_promote_equal_recall_or_added_errors():
    human_test = dict(exact=230, accepted=251, negative_errors=2)
    human_holdout = dict(exact=24, accepted=25, negative_errors=1)
    assert adapter().regression_gate(human_test, human_holdout)['pass'] is True
    assert adapter().regression_gate({**human_test, 'exact':229}, human_holdout)['pass'] is False
    assert adapter().regression_gate({**human_test, 'accepted':252}, human_holdout)['pass'] is False
    assert adapter().regression_gate({**human_test, 'negative_errors':3}, human_holdout)['pass'] is False
    assert adapter().regression_gate(human_test, {**human_holdout, 'accepted':26})['pass'] is False


def test_frozen_protocol_rejects_drift_in_nested_control_predictions(tmp_path, monkeypatch):
    import hashlib
    import json
    m = adapter()
    control = tmp_path/'control'; control.mkdir()
    here = tmp_path/'repair'; here.mkdir()
    predictions = control/'test-predictions.json'
    predictions.write_text('[]\n')
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = control/'results.json'
    receipt.write_text(json.dumps({'bindings':{str(predictions):digest(predictions)}}))
    (here/'protocol.json').write_text(json.dumps(dict(
        bindings={str(receipt):digest(receipt)}, installed_metadata={})))
    monkeypatch.setattr(m, 'CONTROL', control)
    monkeypatch.setattr(m, 'HERE', here)
    predictions.write_text('[{"id":"changed"}]\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        m.freeze()


def test_comparison_preserves_capture_fault_status():
    candidate = rows(); candidate[0]['capture_fault'] = True
    with pytest.raises(ValueError, match='population|references|fault'):
        adapter().verify_comparable(candidate, rows())
