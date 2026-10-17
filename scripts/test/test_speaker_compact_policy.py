import copy
import importlib

import pytest

from scripts.runtime.pcv_schema import INTENTS


def record(probability=.8, *, expected='TIME', predicted='TIME', slot=None):
    command = {'intent': predicted, 'slots': slot or {}} if predicted else None
    reference = {'intent': expected, 'slots': {}} if expected else None
    return dict(id=str(probability), expected=reference, candidate=command, synthetic=False,
                intent_index=INTENTS.index(predicted)+1 if predicted else 0,
                intent_probability=probability, slot_confidence={}, target_named_probability=0.,
                top_intent_probability=probability, score=-.01, margin=.1)


def fitter():
    return importlib.import_module('scripts.train.speaker_compact_policy')


def calibration(rows):
    return [{**copy.deepcopy(r), 'id':'cal/'+r['id']} for r in rows]


def runtime():
    return importlib.import_module('scripts.runtime.speaker_compact')


def test_fit_recovers_correct_commands_without_added_errors():
    rows = [record(.99), record(.8), record(.7, expected=None)]
    policy = fitter().fit_policy(rows, calibration(rows), {'model_sha256': 'abc'})
    assert policy['intent_thresholds']['TIME'] == .8
    assert policy['views']['development']['selected']['correct'] == 2
    assert policy['views']['development']['selected']['wrong'] == 0
    assert policy['views']['calibration']['selected']['false_accepts'] == 0


def test_fit_does_not_trade_unknown_errors_for_recall():
    dev = [record(.99), record(.8)]
    cal = [record(.99), record(.8), record(.9, expected=None)]
    policy = fitter().fit_policy(dev, calibration(cal), {})
    assert policy['intent_thresholds']['TIME'] == .95


def test_fit_does_not_trade_wrong_commands_for_recall():
    rows = [record(.99), record(.8), record(.9, expected='WEATHER')]
    policy = fitter().fit_policy(rows, calibration(rows), {})
    assert policy['intent_thresholds']['TIME'] == .95


def test_missing_calibration_intent_keeps_original_threshold():
    policy = fitter().fit_policy([record(.8)], [record(.99, expected='WEATHER', predicted='WEATHER')], {})
    assert policy['intent_thresholds']['TIME'] == .95
    assert 'TIME' in policy['unrelaxed_missing_human_coverage']


def test_synthetic_rows_cannot_fit_human_policy():
    r = record(); r['synthetic'] = True
    with pytest.raises(ValueError, match='human'):
        fitter().fit_policy([r], [record()], {})


def test_overlapping_policy_views_are_rejected():
    rows = [record(.99), record(.8)]
    with pytest.raises(ValueError, match='overlap'):
        fitter().fit_policy(rows, copy.deepcopy(rows), {})


def policy():
    return dict(format='speaker-compact-recall-v1', binding={'model_sha256': 'abc'},
                intent_thresholds={i: .8 for i in INTENTS}, slot_threshold=.9,
                qualified=False)


def event(r=None):
    return dict(output={'schema_version':3, 'accepted':False, 'command':None, 'reason':'unqualified_model'},
                diagnostic_candidate=r or record(), timing={'total_ms':1.}, audio_seconds=1.)


def test_adapter_accepts_demo_but_preserves_qualified_output():
    original = event(); before = copy.deepcopy(original)
    result = runtime().decorate_event(original, policy(), mode='demo')
    assert result['output']['command'] == {'intent':'TIME','slots':{}}
    assert result['qualified_output'] == before['output']
    assert result['decision_mode'] == 'demo'
    assert original == before
    assert runtime().decorate_event(original, policy(), mode='qualified')['output'] == before['output']


def test_adapter_never_bypasses_capture_fault_or_slot_checks():
    bad = event(); bad['output']['reason'] = 'capture_fault'
    assert runtime().decorate_event(bad, policy(), mode='demo')['output']['reason'] == 'capture_fault'
    bad = event(record(.99, predicted='TIMER', slot={'duration_seconds':10}))
    assert runtime().decorate_event(bad, policy(), mode='demo')['output'] == {
        'schema_version':3,'accepted':False,'command':None,'reason':'invalid_argument'}


def test_policy_rejects_binding_drift_and_unsafe_thresholds():
    runtime().validate_policy(policy(), {'model_sha256':'abc'})
    with pytest.raises(ValueError, match='binding'):
        runtime().validate_policy(policy(), {'model_sha256':'changed'})
    bad = policy(); bad['slot_threshold'] = .1
    with pytest.raises(ValueError, match='slot'):
        runtime().validate_policy(bad, bad['binding'])
    bad = policy(); bad['intent_thresholds']['TIME'] = float('nan')
    with pytest.raises(ValueError, match='threshold'):
        runtime().validate_policy(bad, bad['binding'])


def test_runtime_rejects_policy_file_changed_after_recipe_lock(tmp_path):
    import hashlib, json
    path = tmp_path/'recall-policy.json'; path.write_text(json.dumps(policy()))
    recipe = dict(binding={'model_sha256':'abc'}, recall_policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert runtime().load_bound_policy(tmp_path, recipe)['intent_thresholds']['TIME'] == .8
    changed = policy(); changed['intent_thresholds']['TIME'] = .5
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='identity'):
        runtime().load_bound_policy(tmp_path, recipe)
