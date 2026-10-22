"""Scorecard arithmetic and evidence boundaries; fixtures are not speech."""
import copy
import importlib.util
from statistics import NormalDist

import pytest


def test_wilson_keeps_missing_evidence_undefined_and_handles_extreme_counts():
    assert importlib.util.find_spec('scripts.test.speaker_human_scorecard'), 'Per-intent scorecard metrics are missing'
    from scripts.test.speaker_human_scorecard import wilson_interval
    assert wilson_interval(0, 0) is None
    confidence = 2 * NormalDist().cdf(1) - 1  # z=1, analytically exact endpoints
    assert wilson_interval(1, 1, confidence=confidence) == pytest.approx([.5, 1.])
    assert wilson_interval(0, 1, confidence=confidence) == pytest.approx([0., .5])
    assert wilson_interval(5, 10)[0] == pytest.approx(1 - wilson_interval(5, 10)[1])
    for good, total in ((2, 1), (-1, 2), (0, -1)):
        with pytest.raises(ValueError):
            wilson_interval(good, total)


def test_data_profile_separates_head_masks_speakers_sources_and_command_cells():
    from scripts.test import speaker_human_scorecard as score
    from scripts.test.test_speaker_human_data import row
    from scripts.train.speaker_human_data import human_supervision, LABELS
    assert hasattr(score, 'dataset_profile'), 'Comprehensive data profile is missing'
    generic = {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.all'}}
    named = {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.kitchen'}}
    rows = [human_supervision(row('a', expected=generic, speaker='a')),
            human_supervision(row('b', expected=named, speaker='b')),
            human_supervision(row('c', expected=named, speaker='b'))]
    for i, item in enumerate(rows):
        item.update(text='turn on '+str(i), publisher_source='FSC' if i == 0 else 'SNIPS',
                    transcript_source='dataset' if i < 2 else 'whisper')
    rows[2].update(phones=[], loss_mask=dict(ctc=False, intent=True, target=False))
    audio = [dict(id=r['id'], duration_seconds=i+1, flags=['internal_pause_endpoint_risk'] if i == 2 else [])
             for i, r in enumerate(rows)]
    definitions = [{'text': 'generic', 'command': generic}, {'text': 'named', 'command': named}]
    result = score.dataset_profile(rows, audio, definitions, definitions)
    assert set(result['per_intent']) == set(LABELS)
    light = result['per_intent']['LIGHT_ON']
    train = light['splits']['train']
    assert train['rows'] == 3 and train['speakers'] == 2
    assert train['speaker_effective_count'] == pytest.approx(9/5)
    assert train['largest_speaker_fraction'] == pytest.approx(2/3)
    assert train['full_ctc_rows'] == 2 and train['semantic_only_rows'] == 1
    assert train['transcript_origins'] == {'dataset': 2, 'whisper': 1}
    assert train['duration']['total_seconds'] == 6
    assert train['flags']['internal_pause_endpoint_risk'] == 1
    assert light['coverage']['baseline_commands']['train'] == dict(covered=2, total=2)
    assert {s['source']: s['rows'] for s in light['training_sources']} == {'FSC': 1, 'SNIPS': 2}
    assert light['qualified_best_source'] is None
    assert result['per_intent']['MESSAGE']['splits']['train']['rows'] == 0
    assert result['per_intent']['MESSAGE']['splits']['train']['duration']['total_seconds'] is None
    with pytest.raises(ValueError, match='admitted original-human'):
        score.dataset_profile([{**rows[0], 'synthetic': True}], [], [], [])


def test_complete_command_metrics_penalize_wrong_slots_and_use_predicted_not_gold_head():
    from scripts.test import speaker_human_scorecard as score
    from scripts.test.test_speaker_human_data import row
    from scripts.runtime.pcv_schema import INTENTS
    assert hasattr(score, 'prediction_metrics'), 'Comprehensive prediction metrics are missing'
    low = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 20}}
    high = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 60}}
    stop = {'intent': 'STOP', 'slots': {}}
    rows = [row('correct', split='dev', expected=low), row('slot-error', split='dev', expected=low),
            row('unknown', split='dev'), row('rejected', split='dev', expected=stop),
            row('ambiguous', split='dev', metric=False)]
    for item, candidate, accepted, head in zip(rows, [low, high, high, stop, low],
            [low, high, high, None, low], ['STOP', 'BRIGHTNESS', 'BRIGHTNESS', 'STOP', 'BRIGHTNESS']):
        item.update(candidate=candidate, application_command=accepted, publisher_source='fixture',
                    application_reason='intent_confidence' if accepted is None else None,
                    intent_index=INTENTS.index('BRIGHTNESS')+1,
                    diagnostic_candidate={'intent_index': INTENTS.index(head)+1})
    result = score.prediction_metrics(rows)
    overall = result['overall']
    assert overall['positive_n'] == 3 and overall['negative_n'] == 1 and overall['ambiguous_n'] == 1
    assert overall['accepted_correct'] == 1 and overall['raw_correct'] == 2
    assert overall['accepted_precision'] == pytest.approx(1/3)
    assert overall['accepted_recall'] == pytest.approx(1/3)
    assert overall['intent_head_correct'] == 2 and overall['intent_head_scored_rows'] == 4
    assert overall['unknown_false_accepts'] == 1 and overall['wrong_positive_accepts'] == 1
    light = result['per_intent']['BRIGHTNESS']
    assert (light['tp'], light['fp'], light['fn']) == (1, 2, 1)
    assert light['accepted_precision'] == pytest.approx(1/3)
    assert light['accepted_recall'] == .5 and light['accepted_f1'] == pytest.approx(2/5)
    assert light['slots']['percent']['raw_conditional_n'] == 2
    assert light['slots']['percent']['raw_conditional_correct'] == 1
    assert result['per_intent']['MESSAGE']['accepted_recall'] is None
    assert result['per_intent']['UNKNOWN']['false_accept_rate'] == 1.
    assert result['confusion']['intent_head']['BRIGHTNESS']['STOP'] == 1
    with pytest.raises(ValueError, match='original-human'):
        score.prediction_metrics([{**rows[0], 'synthetic': True}])


def test_speaker_bootstrap_reports_sparse_and_undefined_evidence_explicitly():
    from scripts.test import speaker_human_scorecard as score
    from scripts.test.test_speaker_human_data import row
    assert hasattr(score, 'speaker_bootstrap'), 'Speaker-cluster uncertainty is missing'
    stop = {'intent': 'STOP', 'slots': {}}
    a = row('a', split='dev', expected=stop, speaker='single-speaker')
    a.update(candidate=stop, application_command=stop)
    result = score.speaker_bootstrap([a], resamples=128, seed=53)
    assert result['speaker_n'] == 1
    assert result['accepted_recall95'] == [1., 1.]
    assert result['insufficient_speaker_evidence'] is True
    assert result['unknown_false_accept_rate95'] is None
    rejected = {**a, 'application_command': None}
    assert score.speaker_bootstrap([rejected], resamples=128)['accepted_precision95'] is None
    assert score.speaker_bootstrap([], resamples=128)['accepted_recall95'] is None
    b = {**a, 'id': 'b', 'speaker': 'second-speaker', 'application_command': None}
    one = score.speaker_bootstrap([a, b], resamples=128, seed=53)
    two = score.speaker_bootstrap([a, b], resamples=128, seed=53)
    assert one == two and one['accepted_recall95'] == [0., 1.]
    missing = score.speaker_bootstrap([a, b], label='MESSAGE', resamples=128)
    assert missing['accepted_recall95'] is None


def test_rejection_references_are_not_counted_as_typed_commands():
    from scripts.test import speaker_human_scorecard as score
    from scripts.test.test_speaker_human_data import row
    negative = row('negative')
    result = score.dataset_profile([negative], [], [], [])
    unknown = result['per_intent']['UNKNOWN']['splits']['train']
    assert unknown['typed_commands'] == 0
    assert unknown['rejection_rows'] == 1
    assert unknown['commands'] == {}


def test_head_f1_counts_false_positive_intents_and_excludes_missing_predictions():
    from scripts.test import speaker_human_scorecard as score
    from scripts.test.test_speaker_human_data import row
    from scripts.runtime.pcv_schema import INTENTS
    stop = {'intent': 'STOP', 'slots': {}}
    examples = [row('positive', split='dev', expected=stop), row('negative', split='dev'),
                row('missing-head', split='dev', expected=stop)]
    for item in examples:
        item.update(candidate=stop, application_command=stop, intent_index=INTENTS.index('STOP')+1)
    for item in examples[:2]:
        item['diagnostic_candidate'] = {'intent_index': INTENTS.index('STOP')+1}
    result = score.prediction_metrics(examples)
    metric = result['per_intent']['STOP']
    assert metric['intent_head_f1'] == pytest.approx(2/3)
    assert metric['intent_head_reference_n'] == 1
    assert metric['intent_head_precision'] == .5 and metric['intent_head_recall'] == 1.
    assert result['per_intent']['UNKNOWN']['intent_head_f1'] == 0.
    absent = score.prediction_metrics(examples[2:])['per_intent']['STOP']
    assert absent['intent_head_f1'] is None and absent['intent_head_reference_n'] == 0
