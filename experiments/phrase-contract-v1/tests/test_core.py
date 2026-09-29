"""Acceptance contract and independently checked CTC state transitions."""
from pathlib import Path
import numpy as np
import pytest

from pcvcm.contract import Contract, compile_catalog, canonical
from pcvcm.decoder import PrefixCTC, rank_commands
from pcvcm.endpoint import Endpoint, EndpointConfig

ROOT = Path(__file__).resolve().parents[1]


def test_contract_enumerates_all_required_values():
    contract = Contract(ROOT / 'spec')
    assert len(contract.positive) == 570
    assert len(contract.commands) == 402
    assert len(contract.negative) == 200
    for entity, count in [('light.kitchen', 101), ('light.bedroom', 202)]:
        rows = [r for r in contract.positive if r['expected_command']['intent'] == 'BRIGHTNESS'
                and r['expected_command']['entity_id'] == entity]
        assert len(rows) == count
        assert {r['expected_command']['value'] for r in rows} == set(range(101))


@pytest.mark.parametrize('value', [True, False, 73.0, '73', None, -1, 101])
def test_integer_and_membership_are_strict(value):
    contract = Contract(ROOT / 'spec')
    with pytest.raises(ValueError):
        contract.validate({'intent': 'BRIGHTNESS', 'entity_id': 'light.kitchen', 'value': value})


def test_missing_target_or_cross_product_is_rejected():
    contract = Contract(ROOT / 'spec')
    for command in [
        {'intent': 'BRIGHTNESS', 'value': 73},
        {'intent': 'COLOR_SET', 'entity_id': 'light.bedroom', 'color': 'red'},
        {'intent': 'REMINDER_SET', 'reminder_id': 'reminder.water',
         'content_id': 'reminder_content.water', 'delay_seconds': 300},
    ]:
        with pytest.raises(ValueError):
            contract.validate(command)


def test_compilation_and_letter_name_override():
    contract = Contract(ROOT / 'spec')
    catalog = compile_catalog(contract, ROOT / 'configs/lexicon.json')
    assert len(catalog['entries']) == 770
    assert catalog['collisions'] == []
    alarms = [r for r in catalog['entries'] if r['text'] == 'set an alarm for six a m']
    phones = catalog['phone_inventory']
    assert [phones[i] for i in alarms[0]['phones']][-3:] == ['EY', 'EH', 'M']


@pytest.mark.parametrize('seed', range(5))
def test_shared_prefix_matches_independent_torch_ctc(seed):
    import torch
    torch.set_num_threads(2)
    sequences = [[1], [1, 1], [1, 2], [1, 2, 1], [3, 4, 4], [3, 4]]
    logits = np.random.default_rng(seed).normal(size=(24, 40))
    decoder = PrefixCTC(sequences)
    score = decoder.feed(logits)
    targets = torch.tensor([p for seq in sequences for p in seq])
    logp = torch.tensor(logits).log_softmax(1)[:, None, :].repeat(1, len(sequences), 1)
    expected = -torch.nn.functional.ctc_loss(
        logp, targets, torch.full((len(sequences),), len(logits)),
        torch.tensor([len(s) for s in sequences]), reduction='none',
    ).numpy() / len(logits)
    np.testing.assert_allclose(score, expected, atol=1e-12, rtol=1e-12)
    decoder.reset()
    for chunk in np.array_split(logits, 7):
        actual = decoder.feed(chunk)
    np.testing.assert_array_equal(actual, score)


def test_repeated_phone_requires_blank_and_invalid_logits_fail():
    decoder = PrefixCTC([[1], [1, 1]])
    assert np.isneginf(decoder.feed(np.zeros((1, 40)))[1])
    for logits in [np.zeros((3, 39)), np.full((3, 40), np.nan), np.zeros((2, 3, 40))]:
        with pytest.raises(ValueError):
            decoder.feed(logits)


def test_aliases_do_not_compete_with_same_command():
    a = {'intent': 'POWER_ON', 'entity_id': 'light.kitchen'}
    b = {'intent': 'POWER_OFF', 'entity_id': 'light.kitchen'}
    entries = [{'command': a, 'id': 'a', 'use_case': 'UC-03'},
               {'command': a, 'id': 'aa', 'use_case': 'UC-03'},
               {'command': b, 'id': 'b', 'use_case': 'UC-03'}]
    result = rank_commands(np.array([-0.1, -0.11, -1.0]), entries)
    assert result['command'] == a
    assert result['margin'] == pytest.approx(0.9)
    entries[2]['command'] = None
    assert rank_commands(np.array([-2., -2.1, -1.]), entries)['command'] is None


def test_endpoint_is_chunk_invariant_and_tolerates_internal_pause():
    audio = np.concatenate([np.zeros(4800), np.ones(4800) * .1,
                            np.zeros(9600), np.ones(4800) * .1,
                            np.zeros(16000)]).astype('float32')
    ends = []
    for chunk in [137, 320, 640, 5000]:
        endpoint = Endpoint(EndpointConfig())
        events = []
        for start in range(0, len(audio), chunk):
            event = endpoint.feed(audio[start:start + chunk], start)
            events.append(event)
            if event['ended']:
                ends.append(event['end_sample'])
                break
        assert not any(e['ended'] for e in events[:-1])
    assert len(set(ends)) == 1
    assert ends[0] == 4800 + 4800 + 9600 + 4800 + 11200


@pytest.mark.parametrize('fault', ['gap', 'nan', 'overflow'])
def test_capture_fault_is_sticky(fault):
    endpoint = Endpoint(EndpointConfig())
    x = np.ones(320, dtype='float32') * .1
    endpoint.feed(x, 0)
    result = endpoint.feed(np.full(320, np.nan) if fault == 'nan' else x,
                           999 if fault == 'gap' else 320, overflow=fault == 'overflow')
    assert result['fault'] is not None
    assert endpoint.feed(np.zeros(16000), 640)['fault'] == result['fault']
    endpoint.reset()
    assert endpoint.feed(x, 0)['fault'] is None


def test_eof_never_manufactures_endpoint():
    endpoint = Endpoint(EndpointConfig())
    assert not endpoint.feed(np.ones(16000) * .1, 0)['ended']
