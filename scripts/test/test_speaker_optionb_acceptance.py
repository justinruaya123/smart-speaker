"""OptionB-only evaluation must preserve negatives and complete typed arguments."""
from copy import deepcopy
import importlib
from typing import Any
import pytest


def implementation():
    try:
        return importlib.import_module('scripts.test.speaker_optionb_acceptance')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_acceptance':
            raise
        raise AssertionError('OptionB acceptance investigation is missing') from exc


def row(identity, expected, *, candidate=None, accepted=None, eligible=True, reason=None) -> dict[str, Any]:
    return dict(id=identity, split='holdout', kind='human', synthetic=False,
                metric_eligible=eligible, training_eligible=False, expected=expected,
                candidate=candidate, application_command=accepted, application_reason=reason,
                speaker='fixture-speaker', text='fixture only, not acoustic evidence')


def test_optionb_reference_scope_keeps_outside_commands_and_unknowns_as_separate_negatives():
    weather = dict(intent='WEATHER', slots={})
    named = dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'})
    records = [row('inside', weather, candidate=weather, accepted=weather),
               row('unknown', None, candidate=weather, reason='intent_confidence'),
               row('outside', named, candidate=named, accepted=named),
               row('ambiguous', None, candidate=weather, accepted=weather, eligible=False)]
    original = deepcopy(records)
    scoped = implementation().scope_records(records)
    assert len(scoped) == len(records)
    assert [r['reference_scope'] for r in scoped] == ['optionb', 'true_unknown', 'outside_optionb', 'ambiguous']
    assert scoped[0]['expected'] == weather
    assert scoped[1]['expected'] is None and scoped[2]['expected'] is None
    assert scoped[2]['reference_before_scope'] == named
    assert scoped[2]['application_command'] == named  # A false accept must not vanish.
    assert scoped[3]['metric_eligible'] is False
    assert records == original
    assert all(r['training_eligible'] is False and r['split'] == 'holdout' for r in scoped)


def test_output_scope_can_only_reject_an_unsupported_complete_command():
    weather = dict(intent='WEATHER', slots={})
    named = dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'})
    records = [row('inside', weather, candidate=weather, accepted=weather),
               row('outside', named, candidate=named, accepted=named),
               row('false-accept', None, candidate=weather, accepted=weather)]
    original = deepcopy(records)
    scoped = implementation().scope_records(records, output_allowlist=True)
    assert scoped[0]['application_command'] == weather
    assert scoped[1]['application_command'] is None
    assert scoped[1]['application_reason'] == 'outside_optionb_output_scope'
    assert scoped[1]['candidate'] == named  # The raw prediction is not repaired.
    assert scoped[1]['application_before_scope'] == named
    assert scoped[2]['application_command'] == weather  # A valid-looking false accept remains an error.
    assert records == original


def test_complete_metrics_counts_wrong_parameters_as_both_fp_and_fn():
    ten = dict(intent='TIMER', slots={'duration_seconds': 10})
    thirty = dict(intent='TIMER', slots={'duration_seconds': 30})
    records = [row('wrong-value', thirty, candidate=ten, accepted=ten),
               row('correct', thirty, candidate=thirty, accepted=thirty),
               row('unknown-false-accept', None, candidate=ten, accepted=ten),
               row('ambiguous', None, candidate=ten, accepted=ten, eligible=False)]
    original = deepcopy(records)
    metrics = implementation().complete_metrics(records)
    overall = metrics['overall']
    assert (overall['tp'], overall['fp'], overall['fn']) == (1, 2, 1)
    assert (overall['positive_n'], overall['negative_n'], overall['ambiguous_n']) == (2, 1, 1)
    assert overall['precision'] == pytest.approx(1 / 3)
    assert overall['recall'] == pytest.approx(1 / 2)
    assert overall['f1'] == pytest.approx(2 / (2 + 2 + 1))
    assert overall['negative_errors'] == 1
    assert metrics['per_intent']['TIMER']['f1'] == overall['f1']
    assert records == original


def test_optionb_catalog_keeps_every_supported_alias_and_every_negative_competitor():
    from scripts.runtime.pcv_schema import baseline_cases, canonical

    entries: list[dict[str, Any]] = [dict(id=case['id'], command=case['command'], text=case['text'], phones=[1])
               for case in baseline_cases()]
    entries += [dict(id='additional-alias', command=dict(intent='WEATHER', slots={}), phones=[2]),
                dict(id='unknown', command=None, phones=[3]),
                dict(id='negation', command=None, phones=[4]),
                dict(id='named-target', command=dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'}), phones=[5]),
                dict(id='unsupported-value', command=dict(intent='TIMER', slots={'duration_seconds': 120}), phones=[6])]
    original = deepcopy(entries)
    selected = implementation().optionb_catalog(entries)
    assert {e['id'] for e in selected} == {e['id'] for e in entries} - {'named-target', 'unsupported-value'}
    assert [e for e in selected if e['command'] is None] == [e for e in entries if e['command'] is None]
    assert len({canonical(e['command']) for e in selected if e['command'] is not None}) == 31
    assert len([e for e in selected if e['id'].startswith('B1-')]) == 93
    assert entries == original


def test_gate_audit_separates_wrong_head_from_underconfidence_and_hidden_target_failures():
    from scripts.runtime.pcv_schema import INTENTS
    from scripts.runtime.speaker_confidence import application_decision

    config = dict(mode='demo', intent_threshold=.95, slot_threshold=.90)

    def trial(identity, expected, candidate, *, wrong_head=False, probability=.99,
              named_probability=.05, slot_confidence=None):
        index = 0 if wrong_head else INTENTS.index(candidate['intent']) + 1
        diagnostic = dict(candidate=candidate, intent_index=index, intent_probability=probability,
                          top_intent_probability=.99 if wrong_head else probability,
                          target_named_probability=named_probability,
                          slot_confidence=slot_confidence or {})
        event = dict(diagnostic_candidate=diagnostic, output={'reason': 'unqualified_model'})
        application = application_decision(event, config)
        return {**row(identity, expected, candidate=candidate, accepted=application['command'],
                      reason=application['reason']), 'diagnostic_candidate': diagnostic}

    weather = dict(intent='WEATHER', slots={})
    clock = dict(intent='TIME', slots={})
    light = dict(intent='LIGHT_ON', slots={'light_id': 'light.all'})
    red = dict(intent='COLOR', slots={'light_id': 'light.all', 'color': 'red'})
    ten = dict(intent='TIMER', slots={'duration_seconds': 10})
    thirty = dict(intent='TIMER', slots={'duration_seconds': 30})
    records = [trial('accepted', weather, weather),
               trial('wrong-head', clock, clock, wrong_head=True, probability=.02),
               trial('underconfident', clock, clock, probability=.8),
               trial('target', light, light, named_probability=.4, slot_confidence={'light_id': 1.}),
               trial('slot-and-target', red, red, named_probability=.7,
                     slot_confidence={'light_id': 1., 'color': .4}),
               trial('wrong-raw', ten, thirty, probability=.5, slot_confidence={'duration_seconds': 1.})]
    original = deepcopy(records)
    audit = implementation().gate_audit(records, config)
    assert audit['reference_n'] == len(records)
    assert audit['raw_correct'] == 5 and audit['raw_correct_rejected'] == 4
    assert audit['head_correct'] == 5
    assert audit['disjoint_outcomes'] == {
        'correct_accepted': 1, 'correct_raw_wrong_intent_head': 1,
        'correct_raw_matching_head_below_threshold': 1, 'correct_raw_target_confidence': 1,
        'correct_raw_slot_confidence': 1, 'incorrect_raw_rejected': 1}
    assert audit['independent_failed_gates']['target_confidence'] == 2
    assert audit['first_rejections']['target_confidence'] == 1
    assert audit['first_rejections']['intent_confidence'] == 3
    assert audit['gate_replayed_n'] == len(records)
    assert records == original


@pytest.mark.parametrize('remove_intent,keep_negative', [('LIGHT_OFF', True), (None, False)])
def test_restricted_catalog_cannot_omit_an_optionb_command_or_all_negative_competitors(remove_intent, keep_negative):
    from scripts.runtime.pcv_schema import baseline_cases

    entries: list[dict[str, Any]] = [dict(id=case['id'], command=case['command'], phones=[1])
               for case in baseline_cases() if case['command']['intent'] != remove_intent]
    if keep_negative:
        entries.append(dict(id='negative', command=None, phones=[2]))
    with pytest.raises(ValueError, match='OptionB|negative'):
        implementation().optionb_catalog(entries)


@pytest.mark.parametrize('changes', [{'split': 'train'}, {'split': 'dev'},
                                    {'training_eligible': True}, {'metric_eligible': 1}])
def test_scope_refuses_fitting_or_non_holdout_rows(changes):
    example = row('not-holdout', dict(intent='WEATHER', slots={}))
    example.update(changes)
    with pytest.raises(ValueError, match='evaluation-only holdout'):
        implementation().scope_records([example])


def test_scope_refuses_duplicate_recordings():
    example = row('duplicated', dict(intent='WEATHER', slots={}))
    with pytest.raises(ValueError, match='unique'):
        implementation().scope_records([example, deepcopy(example)])


def test_empty_and_all_rejected_metrics_do_not_invent_precision():
    empty = implementation().complete_metrics([])['overall']
    assert empty['precision'] is None and empty['recall'] is None and empty['f1'] is None
    rejected = implementation().complete_metrics([row('rejected', dict(intent='WEATHER', slots={}))])['overall']
    assert rejected['precision'] is None
    assert rejected['recall'] == 0 and rejected['f1'] == 0


def test_gate_audit_refuses_a_stored_decision_that_does_not_replay():
    weather = dict(intent='WEATHER', slots={})
    example = row('tampered-decision', weather, candidate=weather, reason='intent_confidence')
    example['diagnostic_candidate'] = dict(candidate=weather, intent_index=2, intent_probability=.99,
                                           top_intent_probability=.99, slot_confidence={})
    with pytest.raises(ValueError, match='does not replay'):
        implementation().gate_audit([example], dict(mode='demo', intent_threshold=.95, slot_threshold=.90))
