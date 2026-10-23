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


def test_streaming_uses_exact_native_hysteresis():
    m = publication()
    assert hasattr(m, 'stream_pcm'), 'Missing actual native stream replay'
    import numpy as np
    from scripts.runtime.mvp_wake import RollingWake
    values = iter([0., .99, .99] + [0.] * 200)
    gate = RollingWake(lambda pcm: next(values), threshold=.9)
    result = m.stream_pcm(gate, np.zeros(1280, dtype='float32'))
    assert result['hits'] == 1
    assert result['score_count'] > 3


def test_panel_metrics_reject_incomplete_or_repeated_membership():
    m = publication()
    assert hasattr(m, 'panel_metrics'), 'Missing exact-cohort acceptance scorer'
    cases = [{'id': 'p', 'wake_positive': True, 'kind': 'positive'},
             {'id': 'n', 'wake_positive': False, 'kind': 'negative'}]
    records = [{'id': 'p', 'hits': 1, 'seconds': 1.}, {'id': 'n', 'hits': 0, 'seconds': 2.}]
    metric = m.panel_metrics(cases, records)
    assert metric['synthetic_smoke_pass'] is True
    assert metric['positive_n'] == metric['detected'] == 1
    assert metric['negative_seconds'] == 2.
    for invalid in ([], records[:1], records + records[:1]):
        with pytest.raises(ValueError, match='cohort'):
            m.panel_metrics(cases, invalid)
    records[1]['hits'] = 1
    assert m.panel_metrics(cases, records)['synthetic_smoke_pass'] is False


def test_adaptation_panel_is_voice_disjoint_with_unseen_test_renderings():
    m = publication()
    assert hasattr(m, 'adaptation_cases'), 'Missing independent companion protocol'
    names = set(m.BASELINE_VOICES)
    names.update(v for group in m.ADAPTATION_VOICES.values() for v in group)
    available = [{'ShortName': v, 'Gender': 'metadata-only', 'Locale': v[:5]} for v in names]
    rows = m.adaptation_cases(available)
    assert len({r['id'] for r in rows}) == len(rows)
    groups = {s: {r['voice'] for r in rows if r['split'] == s and r['voice']} for s in m.ADAPTATION_VOICES}
    for split, voices in groups.items():
        assert voices.isdisjoint(m.BASELINE_VOICES)
        for other in groups:
            if split != other:
                assert voices.isdisjoint(groups[other])
    train = [r for r in rows if r['split'] == 'train' and r['wake_positive']]
    test = [r for r in rows if r['split'] == 'test' and r['wake_positive']]
    assert len(test) == 32
    assert {r['text'] for r in test}.isdisjoint(r['text'] for r in train)
    assert {r['rate'] for r in test}.isdisjoint(r['rate'] for r in train)
    assert all(r['synthetic'] for r in rows)


def test_training_inputs_cannot_admit_other_splits():
    m = publication()
    assert hasattr(m, 'assert_fit_membership'), 'Missing split isolation guard'
    m.assert_fit_membership([{'split': 'train', 'synthetic': True, 'voice': 'new'}], 'train')
    for row in ({'split': 'test', 'synthetic': True, 'voice': 'new'},
                {'split': 'train', 'synthetic': False, 'voice': 'new'}):
        with pytest.raises(ValueError, match='fit membership'):
            m.assert_fit_membership([row], 'train')


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
