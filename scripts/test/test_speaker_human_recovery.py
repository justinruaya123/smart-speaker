"""Safety tests for original-human recovery; fixtures are not speech evidence."""
import copy
import importlib.util
import io
import json
from pathlib import Path

import pytest

from scripts.runtime.pcv_schema import canonical
from scripts.train.pcv_updated_prepare import assign_split


def metadata(**changes) -> dict:
    value = dict(source='SLURP', speaker_id='FO-488', file='train_12_clip.wav',
                 transcript='pause the audiobook', command='PAUSE', is_synthetic=0,
                 transcript_source='dataset', out_of_scope=0, whisper_check='',
                 published_split='train', slot_value='')
    value.update(changes)
    return value


def location() -> dict:
    return dict(repository='fixture/repo', revision='a' * 40,
                file='data/train-00000-of-00002.parquet', sha256='b' * 64, row=12)


def test_missing_pronunciation_keeps_semantics_without_inventing_ctc():
    assert importlib.util.find_spec('scripts.train.speaker_human_recovery'), 'Human recovery path is missing'
    from scripts.train.speaker_human_recovery import candidate_from_metadata
    meta = metadata()
    before = copy.deepcopy(meta)
    old = dict(id='train/train-00000-of-00002/000012', split=assign_split(meta),
               reference_status='supported', reason='missing_pronunciation')
    row = candidate_from_metadata(meta, location(), old)
    assert row['id'] == 'fixture/repo/primary/train-00000-of-00002/000012'
    assert row['recovery_status'] == 'metadata_semantic_candidate'
    assert row['expected'] == dict(intent='PAUSE', slots={})
    assert row['loss_mask'] == dict(ctc=False, intent=True, target=False)
    assert row['phones'] == []
    assert row['intent_index'] > 0
    assert row['metric_eligible'] is False
    assert row['training_eligible'] is False  # must still pass audio and split checks
    assert row['listening_status'] == 'not_independently_reviewed'
    assert meta == before
    assert canonical(row['location']) == canonical(location())


def test_synthetic_and_malformed_source_identity_cannot_be_human_proposals():
    from scripts.train.speaker_human_recovery import candidate_from_metadata
    with pytest.raises(ValueError, match='original human'):
        candidate_from_metadata(metadata(is_synthetic=1), location())
    for changes in ({'row': '12'}, {'revision': 'a' * 39},
                    {'file': 'data/../data/train-00000-of-00002.parquet'}):
        with pytest.raises(ValueError, match='Malformed pinned'):
            candidate_from_metadata(metadata(), {**location(), **changes})
    old = dict(id='train/train-00001-of-00002/000012', split='train')
    with pytest.raises(ValueError, match='exact source row'):
        candidate_from_metadata(metadata(), location(), old)


@pytest.mark.parametrize('changes,reason', [
    ({'transcript_source': 'whisper'}, 'missing_pronunciation'),
    ({'whisper_check': 'not understood'}, 'missing_pronunciation'),
    ({'command': 'TIMER', 'transcript': 'set timer for one hour one minute',
      'slot_value': 'one minute'}, 'missing_pronunciation'),
    ({}, 'ambiguous_reference'),
])
def test_disputed_semantics_never_become_unknown_from_a_lexical_repair(changes, reason):
    from scripts.train.speaker_human_recovery import candidate_from_metadata
    meta = metadata(**changes)
    old = dict(id='train/train-00000-of-00002/000012', split=assign_split(meta),
               reference_status='supported', reason=reason)
    item = candidate_from_metadata(meta, location(), old)
    assert item['recovery_status'] == 'review_required'
    assert not any(item['loss_mask'].values())
    assert item['training_eligible'] is False


def test_recovery_prepares_only_train_audio_and_preserves_immutable_parent(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.train import speaker_human_recovery as recovery
    from scripts.train.speaker_human_data import SPLITS, file_sha256, write_json, write_rows
    assert hasattr(recovery, 'recover_dataset'), 'Executable human recovery is missing'
    pcm = np.random.default_rng(11).uniform(-.1, .1, 16000).astype('float32')
    stream = io.BytesIO()
    sf.write(stream, pcm, 16000, format='WAV', subtype='FLOAT')
    records = [metadata(), metadata(transcript_source='whisper', transcript='bin lights', command='BRIGHTNESS'),
               metadata(source='real_voice', speaker_id='202322013_speaker2')]
    for i, meta in enumerate(records):
        meta['audio'] = {'bytes': stream.getvalue() if i < 2 else b'evaluation audio must not be decoded', 'path': None}
    source = tmp_path / 'source'
    path = source / location()['file']
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    loc = {k: v for k, v in location().items() if k != 'row'}
    loc['sha256'] = file_sha256(path)
    prefix = loc['repository'] + '/primary/' + path.stem + '/'
    parent = tmp_path / 'parent'
    for split in SPLITS:
        write_rows(parent / (split + '.jsonl'), [])
    write_rows(parent / 'exclusions.jsonl', [])
    write_json(parent / 'audit.json', dict(
        outputs={s + '.jsonl': file_sha256(parent / (s + '.jsonl')) for s in SPLITS},
        metadata_receipt=dict(shards=[loc], omitted_rows=[{'id': prefix + f'{i:06d}'} for i in range(3)]),
        split_leakage=[], conflicting_pcm_labels=[]))
    historical = tmp_path / 'historical.json'
    write_json(historical, [dict(id='train/' + path.stem + '/' + f'{i:06d}',
        split=assign_split(meta), reference_status='supported' if i != 1 else 'ambiguous',
        reason='missing_pronunciation' if i != 1 else 'ambiguous_reference') for i, meta in enumerate(records)])
    protected = {str(p): file_sha256(p) for p in (historical, parent / 'audit.json')}
    output = tmp_path / 'recovery'
    result = recovery.recover_dataset(parent, source, historical, output, protected_paths=list(protected))
    assert result['omitted_rows'] == 3
    assert result['admitted_semantic_rows'] == 1
    assert result['audio_checked_rows'] == 2
    assert result['full_training_ready'] is False
    assert result['independently_listened_rows'] == 0
    assert {str(p): file_sha256(p) for p in (historical, parent / 'audit.json')} == protected
    queue = recovery.read_rows(output / 'review-queue.jsonl')
    assert len(queue) == 2
    assert not any(r['training_eligible'] for r in queue)
    dev = next(r for r in queue if r['split'] == 'dev')
    assert 'audio' not in dev and 'features' not in dev
    admitted = recovery.read_rows(output / 'recovered-train.jsonl')[0]
    assert admitted['phones'] == [] and admitted['loss_mask']['ctc'] is False
    assert admitted['intent_index'] > 0 and admitted['metric_eligible'] is True
    assert Path(admitted['audio']).exists() and Path(admitted['features']).exists()
    assert hasattr(recovery, 'masked_optimizer_smoke'), 'Recovered supervision is not exercised'
    smoke = recovery.masked_optimizer_smoke(output, tmp_path / 'smoke',
        architecture=dict(width=8, layers=1, heads=2, feedforward=16, dropout=0.))
    assert smoke['weights_changed'] is True and smoke['optimizer_updates'] == 1
    assert smoke['supervised_rows'] == dict(ctc=0, intent=1, target=0)
    assert smoke['full_training_completed'] is False and smoke['accuracy_gain_claimed'] is False
    assert smoke['normalization_human_rows'] == 1
    with pytest.raises(FileExistsError):
        recovery.recover_dataset(parent, source, historical, output)
    path.write_bytes(b'changed pinned shard')
    with pytest.raises(ValueError, match='shard checksum'):
        recovery.recover_dataset(parent, source, historical, tmp_path / 'tampered')


def test_masked_optimizer_uses_real_semantics_without_empty_ctc_targets():
    import numpy as np
    import torch
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.test.test_speaker_human_data import row
    from scripts.train import speaker_human_recovery as recovery
    assert hasattr(recovery, 'masked_optimization_step'), 'Head-masked human optimizer is missing'
    full = recovery.human_supervision(row('full', expected=None))
    semantic = recovery.human_supervision(row('semantic', expected={'intent': 'PAUSE', 'slots': {}}))
    full['loss_mask'] = dict(ctc=True, intent=True, target=False)
    semantic.update(phones=[], loss_mask=dict(ctc=False, intent=True, target=False))
    rows = [full, semantic]
    arrays = {r['id']: np.random.default_rng(i).normal(size=(40, 40)).astype('float32')
              for i, r in enumerate(rows)}
    torch.set_num_threads(2)
    torch.manual_seed(53)
    models = [PooledContextVCM(width=8, layers=1, heads=2, feedforward=16, dropout=0.) for _ in range(2)]
    models[1].load_state_dict(models[0].state_dict())
    before = {k: v.clone() for k, v in models[0].state_dict().items()}
    results = []
    for microbatch, model in enumerate(models, 1):
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        model.train()
        results.append(recovery.masked_optimization_step(model, rows, arrays, optimizer,
            device='cpu', microbatch=microbatch))
    assert results[0]['supervised_rows'] == dict(ctc=1, intent=2, target=0)
    assert np.isfinite(results[0]['losses']).all()
    assert any(not torch.equal(before[k], v) for k, v in models[0].state_dict().items())
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], atol=1e-6, rtol=1e-5)
    bad = {**semantic, 'split': 'test'}
    optimizer = torch.optim.SGD(models[0].parameters(), lr=.001)
    before = {k: v.clone() for k, v in models[0].state_dict().items()}
    with pytest.raises(ValueError, match='Unsafe human'):
        recovery.masked_optimization_step(models[0], [bad], arrays, optimizer, device='cpu')
    assert all(torch.equal(before[k], v) for k, v in models[0].state_dict().items())


def test_review_requires_two_independent_matching_listeners_and_bound_audio():
    from scripts.train import speaker_human_recovery as recovery
    assert hasattr(recovery, 'adjudicate_review'), 'Independent adjudication guard is missing'
    row = recovery.candidate_from_metadata(metadata(), location())
    row['sha256'] = 'c' * 64
    decision = dict(id=row['id'], location=row['location'], sha256=row['sha256'],
        reviewer_id='reviewer-a', independent=True, listened=True, boundary_status='complete',
        decision='supported', expected={'intent': 'PAUSE', 'slots': {}}, transcript='pause the audiobook')
    assert recovery.adjudicate_review(row, [decision])['status'] == 'awaiting_second_reviewer'
    second = {**decision, 'reviewer_id': 'reviewer-b'}
    result = recovery.adjudicate_review(row, [decision, second])
    assert result['status'] == 'consensus_semantic_proposal'
    assert result['candidate']['training_eligible'] is False
    assert result['candidate']['loss_mask']['ctc'] is False
    assert result['candidate']['phones'] == []
    assert result['reviewer_ids'] == ['reviewer-a', 'reviewer-b']
    with pytest.raises(ValueError, match='Duplicate reviewer'):
        recovery.adjudicate_review(row, [decision, decision])
    with pytest.raises(ValueError, match='audio identity'):
        recovery.adjudicate_review(row, [decision, {**second, 'sha256': 'd' * 64}])
    unresolved = recovery.adjudicate_review(row, [decision, {**second, 'decision': 'ambiguous', 'expected': None}])
    assert unresolved['status'] == 'disputed' and unresolved['candidate'] is None
    assert row['training_eligible'] is False and row['listening_status'] == 'not_independently_reviewed'


def test_cli_keeps_adjudication_proposals_separate_from_training(tmp_path, capsys):
    from scripts.train import speaker_human_recovery as recovery
    from scripts.train.speaker_human_data import write_rows
    assert hasattr(recovery, 'main'), 'Recovery/review CLI is missing'
    with pytest.raises(SystemExit) as exit:
        recovery.main(['prepare', '--help'])
    assert exit.value.code == 0
    assert '--require-ready' in capsys.readouterr().out
    item = recovery.candidate_from_metadata(metadata(), location())
    item['sha256'] = 'c' * 64
    queue, reviews, output = tmp_path / 'queue.jsonl', tmp_path / 'reviews.jsonl', tmp_path / 'review-result'
    write_rows(queue, [item])
    write_rows(reviews, [])
    assert recovery.main(['adjudicate', '--queue', str(queue), '--reviews', str(reviews), '--output', str(output)]) == 2
    outcome = recovery.read_rows(output / 'adjudications.jsonl')[0]
    assert outcome['status'] == 'awaiting_second_reviewer'
    assert outcome['candidate'] is None
    assert not (output / 'train.jsonl').exists()


@pytest.mark.parametrize('coarse,expected', [('VOLUME_UP', 'VOLUME'), ('VOLUME_DOWN', 'VOLUME'),
                                              ('OUT_OF_SCOPE', 'UNKNOWN'), ('PAUSE', 'PAUSE')])
def test_review_intent_does_not_lose_coarse_volume_or_unknown_rows(coarse, expected):
    from scripts.train.speaker_human_recovery import candidate_from_metadata
    item = candidate_from_metadata(metadata(command=coarse), location())
    assert item.get('review_intent') == expected
    assert item['publisher_command'] == coarse


def test_recovery_module_is_explicitly_excluded_from_runtime():
    modules = json.loads((Path(__file__).resolve().parents[1] / 'modules.json').read_text())
    assert modules.get('speaker.human_recovery') == dict(path='train/speaker_human_recovery.py', runtime=False)
