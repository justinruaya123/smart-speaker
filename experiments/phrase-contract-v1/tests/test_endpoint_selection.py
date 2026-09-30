import numpy as np
import pytest
import json
from pathlib import Path


def test_proposed_config_is_directly_loadable_and_requires_300ms_speech():
    from pcvcm.endpoint import EndpointConfig
    path = Path(__file__).resolve().parents[1] / 'configs/endpoint.proposed.json'
    config = EndpointConfig(**json.loads(path.read_text()))
    assert config.minimum_speech_ms == 300
    assert config.rms_threshold == pytest.approx(10 ** (-45 / 20))


def test_selection_uses_misses_then_absolute_delay_then_distance_to_minus45():
    from pcvcm.endpoint_selection import choose_threshold
    records = [dict(dbfs=d, misses=0, median_absolute_delay_ms=700.)
               for d in [-50, -45, -40, -35]]
    assert choose_threshold(records)['dbfs'] == -45
    records[1]['misses'] = 1
    records[2]['median_absolute_delay_ms'] = 680
    assert choose_threshold(records)['dbfs'] == -40
    with pytest.raises(ValueError):
        choose_threshold(records[:-1])


@pytest.mark.parametrize('category', ['number', 'name', 'negation'])
def test_internal_600ms_pause_does_not_end_utterance(category):
    from pcvcm.endpoint_selection import evaluate_endpoint
    pcm = np.r_[np.zeros(4800), np.full(6400, .1), np.zeros(9600),
                np.full(6400, .1), np.zeros(16000)].astype('float32')
    row = dict(split='dev', speech_end_sample=27200,
               pause_category=category, reviewed_internal_pauses=[[11200, 20800]])
    for db in [-50, -45, -40, -35]:
        result = evaluate_endpoint(pcm, row, db)
        assert result['endpoint_sample'] == 38400
        assert result['miss'] is False
        assert result['premature_pause'] is False


def test_endpoint_cannot_select_from_confirmation_or_unannotated_audio():
    from pcvcm.endpoint_selection import evaluate_endpoint
    for row in [dict(split='confirmation', speech_end_sample=100), dict(split='dev')]:
        with pytest.raises(ValueError):
            evaluate_endpoint(np.zeros(16000, dtype='float32'), row, -45)


def test_premature_endpoint_is_miss_even_when_detected():
    from pcvcm.endpoint_selection import evaluate_endpoint
    pcm = np.r_[np.ones(4800)*.1, np.zeros(16000), np.ones(4800)*.1].astype('float32')
    result = evaluate_endpoint(pcm, dict(split='dev', speech_end_sample=len(pcm)), -45)
    assert result['miss'] is True


def test_extra_pause_category_cannot_mask_missing_negation():
    from pcvcm.endpoint_selection import validate_pause_coverage
    rows = [dict(pause_category=k, reviewed_internal_pauses=[[1, 2]]) for k in ['number', 'name', 'other']]
    with pytest.raises(ValueError, match='negation'):
        validate_pause_coverage(rows)
