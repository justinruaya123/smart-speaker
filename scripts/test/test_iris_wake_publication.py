"""Offline software tests; never synthesize, fit, or contact a provider."""
import importlib
import importlib.util
import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest


def test_protocol_write_is_immutable(tmp_path):
    m = publication()
    assert hasattr(m, 'write_once'), 'Missing immutable protocol writer'
    target = tmp_path / 'protocol.json'
    m.write_once(target, {'test': False})
    m.write_once(target, {'test': False})
    with pytest.raises(ValueError, match='immutable'):
        m.write_once(target, {'test': True})


def test_generation_requires_frozen_case_binding(tmp_path):
    m = publication()
    assert hasattr(m, 'verify_generated'), 'Missing generated-input verifier'
    row = {'id': 'case-1', 'synthetic': True}
    target = tmp_path / 'clip.wav'
    target.write_bytes(b'fixture')
    receipt = {'case_sha256': m.object_digest(row), 'wav_sha256': m.digest(target)}
    m.verify_generated(row, receipt, target)
    with pytest.raises(ValueError, match='case'):
        m.verify_generated({**row, 'id': 'case-2'}, receipt, target)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio'):
        m.verify_generated(row, receipt, target)


def test_freeze_binds_actual_baseline_before_generation(tmp_path):
    m = publication()
    assert hasattr(m, 'freeze_baseline'), 'Missing pre-generation protocol freeze'
    available = [{'ShortName': v, 'Gender': 'metadata-only', 'Locale': v[:5]} for v in m.BASELINE_VOICES]
    result = m.freeze_baseline(available, out=tmp_path)
    assert result['positive_recall_minimum'] == .9
    assert result['false_wakes_maximum'] == 0
    assert result['baseline_policy']['threshold'] == .9999
    assert result['baseline_policy']['consecutive'] == 2
    assert len(result['cases']) == 100
    assert (tmp_path / 'baseline-protocol.json').exists()
    assert result['qualified'] is False


def test_synthesis_receipt_is_verified_not_reused_blindly(tmp_path):
    m = publication()
    assert hasattr(m, 'generate_cases'), 'Missing bounded synthesis helper'
    import wave
    calls = []

    async def synth(case, path):
        calls.append(case['id'])
        path.write_bytes(b'provider-fixture')

    def decode(mp3, wav):
        with wave.open(str(wav), 'wb') as f:
            f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            f.writeframes(b'\0\0' * 3200)

    cases = [{'id': 'fixture', 'voice': 'test-only', 'text': 'Hey Iris', 'rate': '+0%',
              'wake_positive': True, 'synthetic': True, 'kind': 'positive'}]
    result = asyncio.run(m.generate_cases(cases, tmp_path, 'frozen', synthesizer=synth, decoder=decode))
    assert len(result) == 1
    assert result[0]['protocol_sha256'] == 'frozen'
    assert result[0]['sample_rate'] == 16000
    assert result[0]['samples'] == 3200
    asyncio.run(m.generate_cases(cases, tmp_path, 'frozen', synthesizer=synth, decoder=decode))
    assert calls == ['fixture']
    (tmp_path / 'audio/fixture.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio'):
        asyncio.run(m.generate_cases(cases, tmp_path, 'frozen', synthesizer=synth, decoder=decode))


def publication():
    assert importlib.util.find_spec('scripts.iris_wake_publication') is not None, 'Missing publication acceptance helper'
    return importlib.import_module('scripts.iris_wake_publication')


def test_baseline_predeclares_complete_panel():
    m = publication()
    voices = [{'ShortName': name, 'Gender': 'metadata-only', 'Locale': name[:5]}
              for name in m.BASELINE_VOICES]
    cases = m.baseline_cases(voices)
    assert Counter(r['kind'] for r in cases) == {'positive': 32, 'negative': 64, 'control': 4}
    assert len({r['id'] for r in cases}) == len(cases)
    assert all(r['synthetic'] is True for r in cases)
    assert {r['text'] for r in cases if r['kind'] == 'negative'} == set(m.NEGATIVES)
    assert {r['rate'] for r in cases if r['kind'] == 'positive'} == {'-10%', '+10%'}
    assert {r['voice'] for r in cases if r['kind'] == 'positive'} == set(m.BASELINE_VOICES)
