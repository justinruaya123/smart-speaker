"""Direct-continuation provenance and unchanged-parent fallback fixtures."""
import importlib

import pytest

from scripts.test.test_speaker_optionb_tiny_ft import row


def module():
    try:
        return importlib.import_module('scripts.train.speaker_optionb_tiny_direct_ft')
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts.train.speaker_optionb_tiny_direct_ft':
            raise
        raise AssertionError('Explicit mixed-parent direct continuation is missing') from exc


def test_pretrained_lineage_allows_actual_train_origins_not_reserved_exposure():
    train = [row('human'), row('synthetic', dict(intent='WEATHER', slots={}), kind='synthetic')]
    reserved = [row('reserved', split='holdout')]
    parent = dict(epoch=44, sample_counts={'human': 2, 'synthetic': 3})
    result = module().validate_parent_lineage(parent, train, reserved, expected_epoch=44)
    assert result['human_draws'] == 2 and result['synthetic_draws'] == 3
    for bad in ({**parent, 'sample_counts': {'reserved': 1}}, {**parent, 'epoch': 43}):
        with pytest.raises(ValueError):
            module().validate_parent_lineage(bad, train, reserved, expected_epoch=44)


def test_bounded_replay_keeps_humans_and_limits_synthetic_per_class_speaker():
    from copy import deepcopy
    items = [row('h'), *[row('s' + str(i), dict(intent='WEATHER', slots={}), kind='synthetic') for i in range(5)]]
    for item in items[1:]:
        item['speaker'] = 'one-tts-voice'
    original = deepcopy(items)
    kept = module().bounded_replay_view(items)
    assert len(kept) == 3 and kept[0]['kind'] == 'human'
    assert items == original and module().bounded_replay_view(items) == kept


def test_real_direct_optimizer_keeps_unchanged_parent_when_development_ties(tmp_path):
    import numpy as np
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
    from scripts.train.speaker_optionb_tiny import train_normalization
    torch.set_num_threads(2)
    weather = dict(intent='WEATHER', slots={})
    items = [row('h-pos', weather), row('h-neg'), row('s-pos', weather, kind='synthetic')]
    dev = [row('dev-pos', weather, split='dev'), row('dev-neg', split='dev')]
    rng = np.random.default_rng(83)
    arrays = {r['id']: rng.normal(-5, 1, (40, 61)).astype(np.float32) for r in [*items, *dev]}
    norm = train_normalization([r for r in items if r['kind'] == 'human'], arrays)
    parent = dict(state_dict=OptionBTinyVCM().state_dict(), epoch=44,
        sample_counts={'h-pos': 1, 's-pos': 1})
    config = dict(seed=83, epochs=1, steps_per_epoch=1, learning_rate=0., minimum_learning_rate=0.,
        weight_decay=.0001, context_max_frames=8, threads=2,
        selection=dict(probability_grid=[.5, .99], margin_grid=[0.]))
    arm = dict(name='direct', label_smoothing=.01, human_positive=12, human_negative=12,
               synthetic_positive=8, hard_negative_draws=6)
    result = module().fit_direct(items, dev, arrays, norm, parent, items, ['h-neg'], tmp_path, config, arm)
    assert result['optimizer_steps'] == 1 and result['sample_draws'] == 32
    assert result['selected_epoch'] == 0 and result['selected_parent_unchanged'] is True
    checkpoint = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert checkpoint['parent_epoch'] == 44 and checkpoint['sample_counts'] == {}
    assert all(torch.equal(v, parent['state_dict'][k]) for k, v in checkpoint['state_dict'].items())
