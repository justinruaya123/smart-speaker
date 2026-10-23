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


def test_train_only_normalization_rejects_test_before_reader():
    m = publication()
    assert hasattr(m, 'fit_normalization'), 'Missing train-only normalization'
    import numpy as np
    rows = [{'id': 'train', 'split': 'train', 'wake_positive': True, 'synthetic': True}]
    rng = np.random.default_rng(12)
    norm = m.fit_normalization(rows, lambda row: rng.normal(0, .01, 12000).astype('float32'))
    assert len(norm['mean']) == len(norm['std']) == 40
    assert norm['source_ids'] == ['train']
    assert min(norm['std']) >= .1
    with pytest.raises(ValueError, match='fit membership'):
        m.fit_normalization([{**rows[0], 'split': 'test'}], lambda row: pytest.fail('leaked reader'))


def test_recipe_rejects_unbounded_training_before_importing_optimizer():
    m = publication()
    assert hasattr(m, 'validate_recipe'), 'Missing bounded training validator'
    cfg = dict(epochs=30, cpu_threads=4, steps_per_epoch=64, batch_size=64, parameters=82114)
    m.validate_recipe(cfg)
    for change in ({'epochs': 31}, {'cpu_threads': 8}, {'parameters': 258912}, {'steps_per_epoch': 1000}):
        with pytest.raises(ValueError, match='bounded'):
            m.validate_recipe({**cfg, **change})


def test_development_selection_key_never_uses_test():
    m = publication()
    assert hasattr(m, 'development_key'), 'Missing development-only selection objective'
    import numpy as np
    logits = np.asarray([[4., 0.], [0., 4.], [1., 2.]])
    target = np.asarray([0, 1, 1])
    key, metric = m.development_key(logits, target)
    assert key[0] == 1.
    assert metric['recall'] == metric['specificity'] == 1.
    assert key[1] < 0.


def test_calibration_grid_uses_no_test_rows_and_preserves_native_policy():
    m = publication()
    assert hasattr(m, 'select_calibration'), 'Missing calibration-only threshold selection'
    records = [dict(id='p', split='calibration', positive=True, synthetic=True, seconds=1., scores=[0., .96, .96, 0.]),
               dict(id='n', split='calibration', positive=False, synthetic=True, seconds=1., scores=[0., .7, .7, 0.])]
    selected = m.select_calibration(records, [.5, .8, .9, 1.])
    assert selected['threshold'] == .8
    assert selected['metrics']['detected'] == 1
    assert selected['metrics']['false_wakes'] == 0
    with pytest.raises(ValueError, match='calibration'):
        m.select_calibration([{**records[0], 'split': 'test'}], [.5, 1.])


def test_test_lock_rejects_dropped_cases_and_changed_inputs(tmp_path):
    m = publication()
    assert hasattr(m, 'validate_test_lock'), 'Missing once-only heldout lock validation'
    cases = [{'id': 't', 'split': 'test'}]
    inputs = [{'id': 't', 'wav_sha256': 'fixed'}]
    path = tmp_path / 'protocol.json'
    m.write_once(path, {'cases': cases})
    lock = dict(protocol_sha256=m.digest(path), test_cases_sha256=m.object_digest(cases),
                test_inputs_sha256=m.object_digest(inputs))
    m.validate_test_lock(path, cases, inputs, lock)
    with pytest.raises(ValueError, match='locked test'):
        m.validate_test_lock(path, [], [], lock)
    with pytest.raises(ValueError, match='locked test'):
        m.validate_test_lock(path, cases, [{'id': 't', 'wav_sha256': 'changed'}], lock)


def test_cli_has_no_implicit_generation_or_fit(monkeypatch):
    m = publication()
    assert hasattr(m, 'main'), 'Missing explicit opt-in CLI'
    monkeypatch.setattr(m, 'fit_companion', lambda *a, **k: pytest.fail('implicit fit'))
    with pytest.raises(SystemExit) as error:
        m.main([])
    assert error.value.code == 2


def test_publication_verdict_cannot_promote_a_failed_smoke_gate():
    m = publication()
    assert hasattr(m, 'publication_verdict'), 'Missing honest publication verdict'
    baseline = dict(positive_n=32, detected=1, false_wakes=0, duplicate_detections=0, synthetic_smoke_pass=False)
    failed = dict(positive_n=32, detected=21, false_wakes=2, duplicate_detections=0, synthetic_smoke_pass=True)
    verdict = m.publication_verdict(baseline, failed)
    assert verdict['synthetic_acceptance_pass'] is False
    assert verdict['publish_companion'] is False
    assert verdict['qualified'] is False
    assert verdict['release_blocker'] is True
    passed = {**failed, 'detected': 30, 'false_wakes': 0}
    assert m.publication_verdict(baseline, passed)['publish_companion'] is True
    assert m.publication_verdict(baseline, passed)['qualified'] is False


def test_published_receipts_reconcile_but_do_not_imply_acceptance():
    m = publication()
    summary = m.study_summary()
    assert summary['baseline']['positive_n'] == 32
    assert summary['baseline']['detected'] == 1
    assert summary['companion_test']['positive_n'] == 32
    assert summary['companion_test']['detected'] == 21
    assert summary['companion_test']['false_wakes'] == 2
    assert summary['verdict']['release_blocker'] is True
    assert summary['verdict']['qualified'] is False


def test_summary_rejects_changed_training_input_receipt(tmp_path):
    m = publication()
    import shutil
    out = tmp_path / 'receipts'
    shutil.copytree(m.EXPERIMENT, out)
    target = out / 'companion-inputs.json'
    rows = json.loads(target.read_text())
    rows[0]['wav_sha256'] = '0' * 64
    target.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='generation manifest'):
        m.study_summary(out)


def test_summary_cli_exits_nonzero_for_real_failed_acceptance(tmp_path, capsys):
    m = publication()
    import shutil
    out = tmp_path / 'receipts'
    shutil.copytree(m.EXPERIMENT, out)
    with pytest.raises(SystemExit) as error:
        m.main(['summary', '--output', str(out)])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)['verdict']['synthetic_acceptance_pass'] is False


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
