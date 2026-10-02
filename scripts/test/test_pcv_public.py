"""Behavioral boundaries for the separately versioned public-speech study."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from einops import pack

from scripts.runtime.pcv_schema import baseline_cases, default_slots, validate_command, compile_inventory, output
from scripts.models.pcv_residual import ResidualVCM, export_onnx
from scripts.train.pcv_public_data import speaker_split, audit_split_rows
from scripts.test.pcv_public_eval import calibrate, accepted_candidate
from scripts.train.pcv_public_training import pooled_logits


def test_baseline_is_exhaustive_and_typed():
    cases = baseline_cases()
    assert len(cases) == 93
    assert len({r['id'] for r in cases}) == 93
    assert len({r['text'].lower() for r in cases}) == 93
    assert len({r['command']['intent'] for r in cases}) == 18
    for row in cases:
        validate_command(row['command'])
    by_text = {r['text']: r['command'] for r in cases}
    assert by_text['Call']['slots'] == {'contact_id': 'contact.home'}
    assert by_text['Message']['slots'] == {}
    assert by_text['Volume up']['intent'] == 'VOLUME'
    assert by_text['Timer 1 minute']['slots'] == {'duration_seconds': 60}
    assert by_text['Remind me to Study']['slots'] == {'task_id': 'task.study'}


def test_defaults_do_not_rescue_explicit_names():
    assert default_slots('LIGHT_ON', None) == {'light_id': 'light.all'}
    with pytest.raises(ValueError, match='invalid_argument'):
        default_slots('LIGHT_ON', 'nonexistent')
    assert default_slots('CALL', None) == {'contact_id': 'contact.home'}
    assert default_slots('CALL', 'Mom') == {'contact_id': 'contact.mom'}


@pytest.mark.parametrize('value', [True, -1, 101, 1.2, '20', None])
def test_invalid_numeric_types_and_bounds(value):
    with pytest.raises(ValueError):
        validate_command({'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': value}})


def test_message_cannot_acquire_implicit_payload_and_reject_has_no_command():
    with pytest.raises(ValueError):
        validate_command({'intent': 'MESSAGE', 'slots': {'recipient': 'Mom'}})
    assert output(None, 'OOV') == {'schema_version': 3, 'accepted': False, 'command': None, 'reason': 'OOV'}


def test_catalog_conflicting_alias_is_rejected():
    rows = [{'id': 'a', 'text': 'same words', 'command': {'intent': 'TIME', 'slots': {}}},
            {'id': 'b', 'text': 'same words', 'command': {'intent': 'WEATHER', 'slots': {}}}]
    with pytest.raises(ValueError, match='collision'):
        compile_inventory(rows, {'same': ['S', 'EY', 'M'], 'words': ['W', 'ER', 'D', 'Z']})


def test_causal_state_and_full_chunk_equivalence():
    torch.manual_seed(37)
    model = ResidualVCM(width=16, dilations=(1, 2, 4)).eval()
    x = torch.randn(2, 40, 41)
    states = [torch.randn(2, c, n) for _, c, n in model.state_shapes]
    with torch.no_grad():
        full = model(x, states)
        parts = []
        for start, stop in [(0, 1), (1, 4), (4, 17), (17, 41)]:
            result = model(x[..., start:stop], states)
            states = result[-1]
            parts.append(result)
        for head in range(3):
            joined, _ = pack([p[head] for p in parts], 'b c *')
            torch.testing.assert_close(joined, full[head], rtol=1e-5, atol=1e-5)
        for actual, expected in zip(states, full[-1], strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_onnx_has_generic_state_contract(tmp_path):
    model = ResidualVCM(width=16, dilations=(1, 2)).eval()
    receipt = export_onnx(model, tmp_path / 'model.onnx')
    assert receipt['pass'] and receipt['state_count'] == 3
    assert receipt['nonzero_cache_tested']
    assert receipt['max_absolute_error'] < 2e-5


def test_speaker_and_paired_recording_leakage_are_errors():
    assert speaker_split('slurp/speaker-1') == speaker_split('slurp/speaker-1')
    rows = [{'id': 'a', 'speaker': 's1', 'pair': 'p1', 'pcm_sha256': 'h1', 'split': 'train'},
            {'id': 'b', 'speaker': 's2', 'pair': 'p1', 'pcm_sha256': 'h2', 'split': 'test'}]
    with pytest.raises(ValueError, match='pair'):
        audit_split_rows(rows)
    rows[1]['pair'] = 'p2'
    rows[1]['speaker'] = 's1'
    with pytest.raises(ValueError, match='speaker'):
        audit_split_rows(rows)


def test_padding_does_not_change_pooled_intent():
    logits = torch.tensor([[[2., 4., 100.], [4., 6., -100.]]])
    torch.testing.assert_close(pooled_logits(logits, torch.tensor([2])), torch.tensor([[3., 5.]]))


def test_explicit_light_signal_cannot_default_to_all():
    row = {'candidate': {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.all'}},
           'score': 0., 'margin': 1., 'intent_index': 4, 'intent_probability': .999,
           'target_named_probability': .99}
    policy = {'min_score': -1., 'min_margin': 0., 'min_intent_probability': .5}
    assert accepted_candidate(row, policy) is None


def test_all_reject_does_not_pass_calibration():
    row = {'id': 'x', 'source': 'fixture', 'expected': {'intent': 'TIME', 'slots': {}},
           'candidate': None, 'score': None, 'margin': 0., 'intent_index': 0,
           'intent_probability': 0., 'target_named_probability': 0.}
    assert calibrate([row])['enabled'] is False
