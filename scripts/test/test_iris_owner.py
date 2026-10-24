"""Independent owner backend software checks; no acoustic quality claims."""
from __future__ import annotations

import json
import io
from pathlib import Path
import tarfile
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import soundfile as sf

from scripts.models.iris_owner import OwnerEncoder, speaker_metric_loss
from scripts.train.iris_owner_reference import continuity_spans, owner_features, speech_mask
from scripts.train.iris_owner import (
    _base_view_receipts, _canonical_rows_digest, _ensure_audio_inputs,
    base_view_directory, calibrate, configure_training_device, digest, historical_installed_integrity,
    historical_splits, mix_waveforms, prepare, restore_training_state,
    save_training_state,
)


def test_speaker_encoder_padding_cannot_change_identity_or_overlap():
    torch.manual_seed(73)
    model = OwnerEncoder(7).eval()
    speech = torch.randn(2, 40, 83)
    lengths = torch.tensor([61, 83])
    tail = torch.randn(2, 40, 42) * 100
    with torch.no_grad():
        expected = model(speech, lengths)
        actual = model(torch.cat((speech, tail), dim=2), lengths)
    assert expected[0].shape == (2, 7)
    assert expected[1].shape == (2, 128)
    assert expected[2].shape == (2,)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(expected[1].norm(dim=1), torch.ones(2))


def test_speaker_metric_uses_identity_pairs_and_has_finite_gradients():
    embeddings = torch.randn(8, 128, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = speaker_metric_loss(embeddings, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()
    unique = speaker_metric_loss(embeddings, torch.arange(8))
    assert unique.item() == 0


def test_energy_windows_cover_every_detected_frame_and_reject_silence():
    rate = 16000
    signal = np.sin(2 * np.pi * 173 * np.arange(5 * rate) / rate).astype('float32') * .08
    signal[rate:rate + rate // 4] = 0
    mask = speech_mask(signal)
    spans = continuity_spans(signal)
    assert spans
    covered = np.zeros(len(signal), dtype=bool)
    for start, end in spans:
        assert 0 <= start < end <= len(signal)
        covered[start:end] = True
    for index in np.flatnonzero(mask):
        assert covered[index * 160:index * 160 + 400].all()
    assert continuity_spans(np.zeros(rate, dtype='float32')) is None
    assert continuity_spans(signal[:2000]) is None
    quiet_edges = np.pad(signal[:rate], (4000, 4000))
    edge_spans = continuity_spans(quiet_edges)
    assert min(a for a, b in edge_spans) == 0
    assert max(b for a, b in edge_spans) == len(quiet_edges)
    with pytest.raises(ValueError):
        speech_mask(np.array([np.nan], dtype='float32'))


def test_overlap_construction_is_split_safe_and_peak_bounded():
    a = np.sin(np.arange(22000) * .15).astype('float32') * .1
    b = np.cos(np.arange(19000) * .12).astype('float32') * .12
    result = mix_waveforms(a, b, seed=3)
    assert result.shape == (max(len(a), len(b)),)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= .95
    assert not np.array_equal(result[:len(a)], a)


def test_declared_level_normalization_removes_mixture_loudness_shortcut():
    waveform = np.sin(np.arange(23000, dtype=np.float32) * .07) * .02
    np.testing.assert_allclose(owner_features(waveform), owner_features(waveform * 3), atol=1e-3, rtol=1e-4)


def test_historical_reservations_never_turn_into_training(tmp_path: Path):
    data = tmp_path / 'data' / 'parameter-study'
    data.mkdir(parents=True)
    (data / 'manifest.jsonl').write_text(json.dumps({
        'speaker': 'libri/19', 'split': 'test', 'audio': 'ignored',
        'text': 'must not become a feature',
    }) + '\n')
    split, receipts = historical_splits(tmp_path)
    assert split['libri/19'] == 'test'
    assert receipts[0]['sha256']


def test_portable_base_views_are_preferred_and_historical_artifacts_are_metadata(tmp_path):
    portable = tmp_path / '.cache/iris-release-v1/inputs/base-views'
    portable.mkdir(parents=True)
    legacy = tmp_path / '.cache/vcm-mvp-hey-iris-v2/views'
    legacy.mkdir(parents=True)
    for name in ('train', 'dev', 'calibration', 'test', 'personal-test'):
        (portable / (name + '.jsonl')).write_text('')
        (legacy / (name + '.jsonl')).write_text('historical')
    protocol = tmp_path / 'experiments/iris-release-v1/owner-protocol.json'
    protocol.parent.mkdir(parents=True)
    integrity = {'artifacts/bundle.json': 'a' * 64, 'artifacts/manifest.json': 'b' * 64}
    protocol.write_text(json.dumps({'installed_preserved': integrity}))
    assert base_view_directory(tmp_path) == portable
    assert historical_installed_integrity(tmp_path) == integrity
    assert not (tmp_path / 'artifacts').exists()


def test_cached_prepare_detects_changed_protocol_before_reusing_data(tmp_path):
    cache = tmp_path / '.cache/iris-release-v1/owner'
    cache.mkdir(parents=True)
    manifest = cache / 'manifest.jsonl'
    proxy = cache / 'synthetic-proxies.jsonl'
    manifest.write_text('')
    proxy.write_text('')
    protocol = tmp_path / 'experiments/iris-release-v1/owner-protocol.json'
    protocol.parent.mkdir(parents=True)
    protocol.write_text('{}')
    prepared = {'manifest_sha256': digest(manifest), 'synthetic_proxy_manifest_sha256': digest(proxy),
                'protocol_sha256': digest(protocol)}
    (cache / 'preparation.json').write_text(json.dumps(prepared))
    protocol.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='protocol'):
        prepare(tmp_path, cache)


def test_portable_view_reserialization_requires_pinned_full_row_equivalence(tmp_path):
    portable = tmp_path / '.cache/iris-release-v1/inputs/base-views'
    portable.mkdir(parents=True)
    old = tmp_path / 'old.jsonl'
    old.write_text('{"speaker":"reserved","split":"test"}\n')
    expected, equivalences = [], []
    for name in ('train', 'dev', 'calibration', 'test', 'personal-test'):
        filename = name + '.jsonl'
        path = portable / filename
        path.write_text('{"speaker": "reserved", "split": "test"}\n')
        canonical, count = _canonical_rows_digest(path)
        expected.append({'path': 'historical/' + filename, 'sha256': digest(old)})
        equivalences.append({'name': filename, 'original_sha256': digest(old),
                             'portable_sha256': digest(path), 'canonical_rows_sha256': canonical,
                             'row_count': count})
    proof = tmp_path / 'experiments/iris-release-v1/owner-base-view-equivalence.json'
    proof.parent.mkdir(parents=True)
    proof.write_text(json.dumps({'views': equivalences}))
    protocol = {'data': {'vcm_view_receipts': expected}}
    assert len(_base_view_receipts(tmp_path, protocol)) == 5
    (portable / 'train.jsonl').write_text('{"speaker": "reserved", "split": "train"}\n')
    with pytest.raises(ValueError, match='frozen protocol'):
        _base_view_receipts(tmp_path, protocol)


def test_portable_libri_hydration_preserves_manifest_and_existing_payload(tmp_path, monkeypatch):
    from scripts.train import iris_owner
    cache = tmp_path / '.cache/iris-release-v1/owner'
    cache.mkdir(parents=True)
    payload = io.BytesIO()
    sf.write(payload, np.sin(np.arange(16000) * .1).astype('float32') * .02, 16000, format='FLAC')
    waveform = payload.getvalue()
    member_name = 'LibriSpeech/train-clean-100/19/198/19-198-0000.flac'
    archive = cache / 'software-fixture.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo(member_name)
        member.size = len(waveform)
        tar.addfile(member, io.BytesIO(waveform))
    destination = cache / 'audio' / member_name
    import hashlib
    row = {'source': 'LibriSpeech-train-clean-100', 'audio': destination.relative_to(tmp_path).as_posix(),
           'sha256': hashlib.sha256(waveform).hexdigest()}
    manifest = cache / 'manifest.jsonl'
    manifest.write_text(json.dumps(row) + '\n')
    (cache / 'synthetic-proxies.jsonl').write_text('')
    original_manifest = manifest.read_bytes()
    monkeypatch.setattr(iris_owner, '_download', lambda root, cache: (archive, {}))
    report = _ensure_audio_inputs(tmp_path, cache)
    assert report['libri_files_restored'] == 1
    assert digest(destination) == row['sha256']
    assert manifest.read_bytes() == original_manifest
    destination.write_bytes(b'changed by another owner')
    with pytest.raises(ValueError, match='identity changed'):
        _ensure_audio_inputs(tmp_path, cache)
    assert destination.read_bytes() == b'changed by another owner'


def test_calibration_does_not_claim_human_wake_qualification():
    scores = {'same': [.91, .89, .93], 'impostor': [.1, .3, .4],
              'clean_overlap': [.05, .1, .2], 'mixed_overlap': [.8, .9, .7]}
    selected = calibrate(scores)
    assert .4 < selected['threshold'] <= .89
    assert .2 < selected['overlap_threshold'] <= .7
    assert selected['calibration_kind'] == 'short_utterance_proxy+synthetic_wake_proxy'
    assert selected['experimental'] is True


def test_boundary_checkpoint_restores_model_optimizer_schedule_and_rng(tmp_path):
    torch.manual_seed(47)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    rng = np.random.default_rng(9)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / 'resume.pt'
    bindings = {'manifest': 'abc', 'protocol': 'def'}
    save_training_state(path, model, optimizer, scheduler, rng,
                        {'epoch': 2, 'next_step': 31}, bindings)
    expected = [p.detach().clone() for p in model.parameters()]
    expected_random = rng.integers(100000, size=10)
    expected_torch = torch.rand(8)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    progress = restore_training_state(path, model, optimizer, scheduler, rng, bindings)
    assert progress == {'epoch': 2, 'next_step': 31}
    for current, saved in zip(model.parameters(), expected):
        torch.testing.assert_close(current, saved)
    np.testing.assert_array_equal(rng.integers(100000, size=10), expected_random)
    torch.testing.assert_close(torch.rand(8), expected_torch)
    assert scheduler.last_epoch == 1
    with pytest.raises(ValueError, match='binding'):
        restore_training_state(path, model, optimizer, scheduler, rng, {'manifest': 'changed'})


def test_portable_cuda_selector_uses_one_logical_device_without_real_allocation():
    for selector in (None, '0', '3', '5', 'GPU-01234567-89ab-cdef-0123-456789abcdef'):
        cuda = SimpleNamespace(set_device=Mock(), set_per_process_memory_fraction=Mock())
        environment = {} if selector is None else {'CUDA_VISIBLE_DEVICES': selector}
        result = configure_training_device('cuda', cuda=cuda, environ=environment)
        cuda.set_device.assert_called_once_with(0)
        cuda.set_per_process_memory_fraction.assert_called_once_with(.12, 0)
        assert result['cuda_visible_devices'] == selector
        assert result['logical_gpu'] == 0
    cuda = SimpleNamespace(set_device=Mock(), set_per_process_memory_fraction=Mock())
    result = configure_training_device('cuda:2', cuda=cuda, environ={})
    assert result['logical_gpu'] == 2
    cuda.set_device.assert_called_once_with(2)
    cuda.set_per_process_memory_fraction.assert_called_once_with(.12, 2)
    for selector in ('0,1', '', '-1', 'not-a-gpu'):
        cuda = SimpleNamespace(set_device=Mock(), set_per_process_memory_fraction=Mock())
        with pytest.raises(ValueError, match='single'):
            configure_training_device('cuda', cuda=cuda, environ={'CUDA_VISIBLE_DEVICES': selector})
        cuda.set_device.assert_not_called()
        cuda.set_per_process_memory_fraction.assert_not_called()
    cuda = SimpleNamespace(set_device=Mock(), set_per_process_memory_fraction=Mock())
    result = configure_training_device('cpu', cuda=cuda, environ={})
    assert result['logical_gpu'] is None
    cuda.set_device.assert_not_called()
    cuda.set_per_process_memory_fraction.assert_not_called()
