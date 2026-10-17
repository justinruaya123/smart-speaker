import importlib

import pytest
import torch


def compact():
    return importlib.import_module('scripts.train.speaker_compact')


def test_full_compact_model_fits_hard_parameter_limit():
    from scripts.models.pcv_release_pooling import PooledContextVCM
    module = compact()
    model = PooledContextVCM(**module.ARCHITECTURE)
    assert module.check_parameters(model) == 935567
    with pytest.raises(ValueError, match='parameter'):
        module.check_parameters(torch.nn.Linear(1200000, 1))


def test_filtered_control_preserves_sources_and_exclusion_reasons():
    rows = [dict(id='human/good', slot_metadata_conflict=False),
            dict(id='human/conflict', slot_metadata_conflict=True),
            dict(id='old/synthetic', slot_metadata_conflict=False),
            dict(id='qsc1/new', slot_metadata_conflict=False)]
    accepted, excluded = compact().control_rows(rows)
    assert [r['id'] for r in accepted] == ['human/good', 'old/synthetic']
    assert {r['id']: r['compact_exclusion'] for r in excluded} == {
        'human/conflict': 'unresolved_slot_metadata_conflict', 'qsc1/new': 'new_qwen_not_control'}
    assert all('compact_exclusion' not in r for r in rows)


def test_shared_trainer_instance_does_not_rebind_parent_paths():
    from scripts.train import speaker_final_train as parent
    before = (parent.HERE, parent.CACHE, parent.ARCHITECTURE.copy())
    isolated = compact().shared_trainer()
    assert isolated is not parent
    assert isolated.HERE != parent.HERE and isolated.CACHE != parent.CACHE
    assert (parent.HERE, parent.CACHE, parent.ARCHITECTURE) == before
