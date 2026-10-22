"""Fine-tune safety/data-selection fixtures, never acoustic accuracy evidence."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


def audit_module():
    try:
        return importlib.import_module('scripts.test.speaker_optionb_tiny_ft_audit')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_tiny_ft_audit':
            raise
        raise AssertionError('Train-only tiny fine-tune audit is missing') from exc


def row(identifier, command=None, *, split='train', kind='human'):
    from scripts.train.speaker_optionb_tiny import projected_row
    return projected_row(dict(id=identifier, speaker=identifier + '-speaker',
        split=split, kind=kind, synthetic=kind == 'synthetic', expected=command,
        metric_eligible=True, training_eligible=split == 'train',
        sha256=identifier + '-file', pcm_sha256=identifier + '-pcm',
        listening_status='not_independently_reviewed', transcript_source='dataset'))


def logits(index, probability):
    values = np.full(32, (1 - probability) / 31, dtype=np.float64)
    values[index] = probability
    return np.log(values)


def test_audit_preserves_source_references_and_fails_closed_on_stale_projection():
    audit = audit_module()
    outside = dict(intent='LIGHT_ON', slots={'light_id': 'light.kitchen'})
    rows = [row('outside', outside), row('unknown')]
    original = deepcopy(rows)
    result = audit.audit_training_rows(rows, [])
    assert rows == original and result['rows'] == 2
    assert result['counts']['human']['outside_optionb'] == 1
    assert result['counts']['human']['original_unknown'] == 1
    for change in ({'class_index': 1}, {'class_index': False},
                   {'scope': 'optionb'}, {'reference_status': 'ambiguous'},
                   {'capture_fault': True}, {'split': 'holdout'}):
        with pytest.raises(ValueError):
            audit.audit_training_rows([{**rows[0], **change}], [])


def test_audit_checks_all_declared_reserved_groups_not_just_human_speakers():
    audit = audit_module()
    train = row('synthetic', dict(intent='WEATHER', slots={}), kind='synthetic')
    reserved = row('reserved', split='test', kind='synthetic')
    for key in ('speaker', 'parent_recording_id', 'speaker_alias_group'):
        with pytest.raises(ValueError):
            audit.audit_training_rows([{**train, key: 'same'}], [{**reserved, key: 'same'}])


def test_text_conflicts_are_review_proposals_not_automatic_relabels():
    audit = audit_module()
    rows = [row('positive', dict(intent='WEATHER', slots={})), row('negative')]
    rows[0]['text'], rows[1]['text'] = 'weather', 'Weather!'
    result = audit.audit_training_rows(rows, [])
    assert len(result['text_reference_conflicts']) == 1
    assert result['labels_changed'] is False and result['independent_listening'] is False
    assert rows[1]['expected'] is None


def test_wrong_positive_command_is_both_false_accept_and_missed_true_command():
    audit = audit_module()
    rows = [row('positive', dict(intent='WEATHER', slots={})), row('negative')]
    values = np.stack([logits(1, .99), logits(1, .99)])
    result = audit.score_logits(rows, values, dict(enabled=True, min_probability=.9, min_margin=0.))
    assert result['tp'] == 0 and result['fp'] == 2 and result['fn'] == 1
    assert result['negative_false_accepts'] == 1
    with pytest.raises(ValueError):
        audit.score_logits(rows, np.zeros((2, 31)))


def test_hard_negative_mining_never_accepts_reserved_or_positive_recordings():
    audit = audit_module()
    rows = [row('hard'), row('easy'), row('positive', dict(intent='WEATHER', slots={}))]
    values = np.stack([logits(1, .9), logits(0, .99), logits(1, .99)])
    assert audit.hard_negative_ids(rows, values, fraction=.25) == ['hard']
    with pytest.raises(ValueError):
        audit.hard_negative_ids([{**rows[0], 'split': 'dev'}], values[:1])


def test_development_selection_ranks_safe_accepted_recall_not_raw_accuracy():
    audit = audit_module()
    weather, stop = dict(intent='WEATHER', slots={}), dict(intent='STOP', slots={})
    rows = [row('p1', weather, split='dev'), row('p2', stop, split='dev'), row('n', split='dev')]
    first, second = rows[0]['class_index'], rows[1]['class_index']
    unsafe = np.stack([logits(first, .98), logits(second, .98), logits(first, .999)])
    safe = np.stack([logits(first, .995), logits(0, .99), logits(0, .99)])
    config = dict(probability_grid=[.9, .99, .9999], margin_grid=[0., .3])
    a = audit.dev_operating_point(rows, unsafe, config)
    b = audit.dev_operating_point(rows, safe, config)
    assert b['rank'] > a['rank'] and b['metrics']['recall'] == .5
    assert b['metrics']['negative_false_accepts'] == 0
    assert audit.score_logits(rows, unsafe)['f1'] > audit.score_logits(rows, safe)['f1']
    with pytest.raises(ValueError):
        audit.dev_operating_point([{**r, 'split': 'holdout'} for r in rows], safe, config)


def fit_module():
    try:
        return importlib.import_module('scripts.train.speaker_optionb_tiny_ft')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.train.speaker_optionb_tiny_ft':
            raise
        raise AssertionError('Versioned tiny fine-tuning workflow is missing') from exc


def test_historical_gap_holdouts_are_omitted_never_relabelled_as_negatives():
    train = fit_module()
    rows = [row(name, dict(intent='WEATHER', slots={}), kind='synthetic')
            for name in ('train', 'val', 'test', 'blank')]
    for item in rows:
        item['location'] = dict(repository='martinnavs/ai231-fil-supplemental-data',
            file='gap_fill/train-00000-of-00001.parquet')
    metadata = {r['id']: dict(manifest_split='' if r['id'] == 'blank' else r['id']) for r in rows}
    original = deepcopy(rows)
    kept, excluded = train.fine_tuning_view(rows, [], metadata)
    assert [r['id'] for r in kept] == ['train']
    assert {r['id'] for r in excluded} == {'val', 'test', 'blank'}
    assert rows == original and all(r['class_index'] > 0 for r in rows)


def test_fine_tune_executes_optimizer_on_human_only_parent_and_rejects_mixed_parent(tmp_path):
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.train.speaker_optionb_tiny import train_normalization
    train = fit_module()
    torch.set_num_threads(2)
    rows = [row('human-positive', dict(intent='WEATHER', slots={})), row('human-negative'),
            row('synthetic-positive', dict(intent='WEATHER', slots={}), kind='synthetic')]
    dev = [row('dev-p', dict(intent='WEATHER', slots={}), split='dev'), row('dev-n', split='dev')]
    rng = np.random.default_rng(73)
    arrays = {r['id']: rng.normal(-5, 1, (40, 61)).astype(np.float32) for r in [*rows, *dev]}
    human = [r for r in rows if r['kind'] == 'human']
    norm = train_normalization(human, arrays)
    parent = dict(state_dict=OptionBTinyVCM().state_dict(), epoch=1,
        config=dict(human_epochs=1), sample_counts={'human-positive': 1})
    config = dict(seed=73, epochs=1, steps_per_epoch=1, learning_rate=.0004,
        minimum_learning_rate=.0001, weight_decay=.0001, context_max_frames=8, threads=2,
        selection=dict(probability_grid=[.5, .99], margin_grid=[0.]))
    arm = dict(name='negative_focus', label_smoothing=0., human_positive=8,
        human_negative=16, synthetic_positive=8, hard_negative_draws=8)
    result = train.fit_continuation(rows, dev, arrays, norm, parent, ['human-negative'],
        tmp_path / 'good', config, arm)
    assert result['optimizer_steps'] == 1 and result['weights_changed'] is True
    assert result['parameters'] == 258912 and result['fitting_splits'] == ['train']
    assert result['selected_on'] == 'dev' and (tmp_path / 'good/best.pt').is_file()
    assert set(result['sample_counts']) <= {r['id'] for r in rows}
    with pytest.raises(ValueError, match='human-only'):
        train.fit_continuation(rows, dev, arrays, norm,
            {**parent, 'sample_counts': {'synthetic-positive': 1}}, ['human-negative'],
            tmp_path / 'bad', config, arm)


def test_fine_tune_export_and_final_replay_require_completed_stage_receipts(tmp_path):
    try:
        evaluate = importlib.import_module('scripts.test.speaker_optionb_tiny_ft_eval')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.test.speaker_optionb_tiny_ft_eval':
            raise
        raise AssertionError('Guarded fine-tune export/evaluation is missing') from exc
    import json
    (tmp_path / 'protocol.json').write_text(json.dumps({'outputs': {'run': str(tmp_path / 'run')}}))
    with pytest.raises(FileNotFoundError, match='completed fine-tune'):
        evaluate.select_candidate(tmp_path)
    with pytest.raises(FileNotFoundError, match='selection'):
        evaluate.evaluate_candidate(tmp_path)
