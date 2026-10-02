"""Review regressions for qualification without changing frozen training."""
import hashlib
from pathlib import Path

import pytest

from scripts.test.pcv_public_qualification import precise_metrics, verify_bindings, decision
from scripts.runtime.pcv_registry import reconciled_registry, compile_registry
from scripts.runtime.pcv_schema import DEFAULT_CUSTOM


def test_intent_precision_counts_false_predictions_from_other_intents():
    time = {'intent': 'TIME', 'slots': {}}
    weather = {'intent': 'WEATHER', 'slots': {}}
    rows = [{'source': 'fixture', 'expected': time, 'candidate': time},
            {'source': 'fixture', 'expected': weather, 'candidate': time},
            {'source': 'fixture', 'expected': None, 'candidate': time}]
    metric = precise_metrics(rows)['per_intent']['TIME']
    assert metric['predicted_n'] == 3
    assert metric['complete_precision'] == 1/3
    assert metric['intent_precision'] == 1/3
    assert metric['complete_recall'] == 1


def test_changed_normalization_fails_before_inference(tmp_path):
    path = tmp_path/'normalization.json'
    path.write_text('original')
    binding = {'normalization.json': hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_bindings(tmp_path, binding)
    path.write_text('changed')
    with pytest.raises(ValueError, match='identity'):
        verify_bindings(tmp_path, binding)


@pytest.mark.parametrize('target_probability', [.5, .999])
def test_confident_explicit_target_cannot_become_all_lights(target_probability):
    record = {'candidate': {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.all'}},
              'score': 0, 'margin': 1, 'intent_index': 4, 'intent_probability': .999,
              'top_intent_probability': .999, 'target_named_probability': target_probability}
    policy = {'enabled': True, 'min_score': -1, 'min_margin': 0, 'min_intent_probability': .7}
    assert decision(record, policy) == {'schema_version': 3, 'accepted': False, 'command': None, 'reason': 'invalid_argument'}


def test_uncertain_intent_is_oov():
    record = {'candidate': None, 'score': None, 'margin': 0, 'intent_index': 0,
              'intent_probability': .2, 'top_intent_probability': .2, 'target_named_probability': .5}
    policy = {'enabled': True, 'min_score': -1, 'min_margin': 0, 'min_intent_probability': .7}
    assert decision(record, policy)['reason'] == 'OOV'


def test_registry_retains_legacy_ids_without_relabeling_and_reads_file(tmp_path):
    import json
    from scripts.train.pcv_public_data import _words
    base = {'entries': [{'id': 'legacy', 'text': 'call alex kim', 'command': {'intent': 'CALL', 'slots': {'contact_id': 'contact.alex_kim'}}}]}
    registry = reconciled_registry(base, DEFAULT_CUSTOM)
    assert 'contact.alex_kim' in {r['id'] for r in registry['contacts']}
    registry['contacts'].append({'id': 'contact.dad', 'name': 'Dad', 'aliases': ['father']})
    path = tmp_path/'custom.json'
    path.write_text(json.dumps(registry))
    before = path.read_bytes()
    _, result = compile_registry(base, path, _words())
    by_text = {r['text']:r['command'] for r in result['entries']}
    assert by_text['call father']['slots']['contact_id'] == 'contact.dad'
    assert by_text['call mother']['slots']['contact_id'] == 'contact.mom'
    assert by_text['call alex kim']['slots']['contact_id'] == 'contact.alex_kim'
    assert path.read_bytes() == before
    assert result['requires_calibration']
