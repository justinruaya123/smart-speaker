"""Release diversity contracts: fixed splits, useful pairs and acceptance selection."""
import importlib
from collections import Counter

import numpy as np
import torch

from scripts.models.mvp_static import StaticCommandNet


def test_training_embedding_preserves_authoritative_logits_and_padding():
    torch.manual_seed(1)
    model = StaticCommandNet(5, width=16).eval()
    x = torch.randn(2, 40, 21)
    lengths = torch.tensor([21, 13])
    with torch.no_grad():
        logits, embedding = model.forward_with_embedding(x, lengths)
        assert torch.equal(logits, model(x, lengths))
        assert embedding.shape == (2, 16)
        short, _ = model.forward_with_embedding(x[1:2, :, :13], lengths[1:2])
        assert torch.allclose(logits[1], short[0], atol=2e-6)


def test_new_voice_splits_are_frozen_disjoint_designed_identities():
    module = importlib.import_module('scripts.train.iris_diversity')
    voices = module.designed_voices()
    assert len({v['id'] for v in voices}) == 24
    assert Counter(v['split'] for v in voices) == {'train':12,'dev':4,'calibration':4,'voice-test':4}
    assert len({(v['prompt'],v['design_seed']) for v in voices}) == 24
    assert all(v['identity_source']=='original_synthetic_voice_design' and not v['accent_verified'] for v in voices)


def test_paired_sampling_preserves_source_balance_and_real_cross_voice_pairs():
    module = importlib.import_module('scripts.train.iris_diversity')
    rows = [dict(id=f'{kind}/{label}/{speaker}', class_index=label, speaker=f'{kind}/{speaker}',synthetic=kind)
            for kind in (False,True) for label in (0,1,2) for speaker in range(3)]
    batch=module.sample_batch(rows,np.random.default_rng(53),[1,2])
    assert len(batch)==32 and sum(r['synthetic'] for r in batch)==16
    for start in range(0,32,8):
        micro=batch[start:start+8]
        assert any(a['class_index']>0 and a['class_index']==b['class_index'] and a['speaker']!=b['speaker']
                   for a in micro for b in micro)


def test_development_key_counts_accepted_commands_instead_of_raw_argmax():
    module = importlib.import_module('scripts.train.iris_diversity')
    records=[dict(speaker='a',synthetic=False,expected_index=1,accepted_index=None,raw_index=1),
             dict(speaker='b',synthetic=False,expected_index=1,accepted_index=1,raw_index=1),
             dict(speaker='s',synthetic=True,expected_index=2,accepted_index=2,raw_index=2)]
    assert module.accepted_key(records)==(.5,1,1)


def test_gain_is_training_only_and_does_not_mutate_cached_features():
    module = importlib.import_module('scripts.train.iris_diversity')
    values=np.ones((40,17),dtype=np.float32)
    original=values.copy()
    norm={'std':[2.]*40}
    augmented=module.gain_features(values,norm,np.random.default_rng(53))
    assert np.array_equal(values,original)
    assert not np.array_equal(augmented,original)
    assert module.gain_features(values,norm,None) is values


def test_promotion_keeps_personal_and_human_error_guards_independent():
    module=importlib.import_module('scripts.train.iris_diversity')
    baseline={'personal':{'overall':{'positive_n':36,'correct_accepts':30}},
        'project':{'test':{'strata':{'human':{'correct_accepts':192,'unknown_accepts':15}}}},
        'official':{'accepted':{'command_level':{'n':196,'accuracy':.75}}}}
    results={'personal-test':{'overall':{'positive_n':36,'correct_accepts':31}},
        'test':{'strata':{'human':{'correct_accepts':200,'unknown_accepts':16}}}}
    official={'accepted':{'command_level':{'n':196,'accuracy':.80}}}
    guards=module.promotion_guards(results,official,baseline)
    assert guards['personal_30_of_36'] is True
    assert guards['human_correct_not_below_baseline'] is True
    assert guards['human_unknown_not_above_baseline'] is False
    assert not all(guards.values())
    results['test']['strata']['human']['unknown_accepts']=15
    assert all(module.promotion_guards(results,official,baseline).values())
    results['personal-test']['overall']['positive_n']=35
    assert module.promotion_guards(results,official,baseline)['personal_population_unchanged'] is False


def test_completed_fit_cache_rejects_changed_inputs_or_weights(tmp_path):
    import json,hashlib
    import pytest
    module=importlib.import_module('scripts.train.iris_diversity')
    model=tmp_path/'best.pt';model.write_bytes(b'frozen project weights')
    selected={'checkpoint_sha256':hashlib.sha256(model.read_bytes()).hexdigest()}
    (tmp_path/'selection.json').write_text(json.dumps(selected))
    (tmp_path/'run-inputs.json').write_text(json.dumps({'protocol_sha256':'p','host':'original'}))
    assert module.completed_selection(tmp_path,{'protocol_sha256':'p'})==selected
    with pytest.raises(ValueError,match='inputs'):
        module.completed_selection(tmp_path,{'protocol_sha256':'changed'})
    model.write_bytes(b'changed project weights')
    with pytest.raises(ValueError,match='checkpoint'):
        module.completed_selection(tmp_path,{'protocol_sha256':'p'})


def test_checkpoint_rng_touches_only_selected_gpu(monkeypatch):
    module=importlib.import_module('scripts.train.iris_diversity')
    state=torch.tensor([1,2],dtype=torch.uint8);calls=[]
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:(calls.append(('get',device)) or state))
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda value,device:calls.append(('set',device)))
    def forbidden(*args,**kwargs):raise AssertionError('Must not initialize other GPUs')
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',forbidden)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',forbidden)
    assert module.capture_device_rng('cpu') is None
    assert torch.equal(module.capture_device_rng('cuda:2'),state)
    module.restore_device_rng(state,'cuda:2')
    assert calls==[('get','cuda:2'),('set','cuda:2')]
def test_final_development_selection_preserves_existing_no_test_flag(tmp_path,monkeypatch):
    import json
    from scripts.train import iris_diversity as study
    cache=tmp_path/'cache';here=tmp_path/'evidence'
    for model,arm,seed,key,parameters in [
        ('compact','ce',53,.4,100),('compact','supcon',53,.5,100),
        ('full','ce',53,.3,200),('full','supcon',53,.35,200),
        ('compact','ce',67,.55,100),('compact','supcon',67,.6,100)]:
        folder=cache/'runs'/f'{model}-{arm}-{seed}';folder.mkdir(parents=True)
        (folder/'selection.json').write_text(json.dumps(dict(model=model,arm=arm,seed=seed,
            key=[key,10,20],parameters=parameters,test_used_for_selection=False)))
    monkeypatch.setattr(study,'CACHE',cache);monkeypatch.setattr(study,'HERE',here)
    monkeypatch.setattr(study,'freeze',lambda:{'selection':'frozen development only'})
    chosen=study.select()
    locked=json.loads((here/'development-selection.json').read_text())
    assert chosen['arm']=='supcon' and chosen['seed']==67
    assert locked['test_used_for_selection'] is False and len(locked['all_runs'])==6
