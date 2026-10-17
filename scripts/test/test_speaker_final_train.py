"""Protect the last fixed-architecture refinement's sampling and time axes."""
import importlib

import numpy as np
import pytest
import torch


def adapter():
    name = 'scripts.train.speaker_final_train'
    assert importlib.util.find_spec(name) is not None, 'final refinement trainer is missing'
    return importlib.import_module(name)


def records():
    rows = []
    for group, kind, prefix, label in (
        ('human_positive', 'human', 'h', 1), ('human_negative', 'human', 'n', 0),
        ('old_synthetic', 'synthetic', 'old', 1), ('new_synthetic', 'synthetic', 'qsc1/', 1),
    ):
        for i in range(4):
            command = {'intent': 'MESSAGE', 'slots': {}} if label else None
            rows.append(dict(id=prefix+str(i), split='train', kind=kind, synthetic=kind=='synthetic',
                             expected=command, command=command, intent_index=label, target_index=-100,
                             speaker=group+str(i), speaker_id=group+str(i), training_eligible=True,
                             phones=[1, 2], reference_status='supported' if label else 'unsupported'))
    return rows


def test_training_masks_preserve_original_shape_and_limit_erasure():
    m = adapter()
    source = np.ones((40, 200), dtype=np.float32)
    masked = m.augment(source, np.random.default_rng(1), probability=1.)
    assert source.min() == 1 and masked.shape == source.shape
    assert np.any(masked == 0)
    assert np.all(np.isin(masked, [0., 1.]))
    assert np.sum(np.all(masked == 0, axis=1)) <= 4
    assert np.sum(np.all(masked == 0, axis=0)) <= 10


def test_no_augmentation_changes_evaluation_features():
    source = np.arange(400, dtype=np.float32).reshape(40, 10)
    result = adapter().augment(source, np.random.default_rng(4), probability=0.)
    assert np.array_equal(source, result) and not np.shares_memory(source, result)


def test_short_commands_keep_sufficient_time_content():
    source = np.ones((40, 9), dtype=np.float32)
    result = adapter().augment(source, np.random.default_rng(4), probability=1.)
    assert not np.any(np.all(result == 0, axis=0))


def test_mixed_sampling_preserves_human_half_and_matches_human_draws():
    m = adapter(); pools = m.TrainingPools(records())
    control = m.sample_batch(pools, seed=53, phase='mixed', epoch=3, step=4, arm='control')
    completion = m.sample_batch(pools, seed=53, phase='mixed', epoch=3, step=4, arm='completion')
    assert len(control) == len(completion) == 32
    assert sum(r['kind']=='human' for r in control) == 16
    assert sum(r['kind']=='human' for r in completion) == 16
    assert [r['id'] for r in control if r['kind']=='human'] == [r['id'] for r in completion if r['kind']=='human']
    assert sum(r['kind']=='human' and r['expected'] is None for r in completion) == 4
    assert sum(r['id'].startswith('qsc1/') for r in completion) == 4
    assert not any(r['id'].startswith('qsc1/') for r in control)


def test_foundation_never_samples_synthetic():
    m=adapter(); batch=m.sample_batch(m.TrainingPools(records()), seed=53, phase='human', epoch=1, step=0, arm='shared')
    assert len(batch)==32 and all(r['kind']=='human' for r in batch)
    assert sum(r['expected'] is None for r in batch)==8


def test_sampler_rejects_calibration_or_test_rows():
    m=adapter(); rows=records(); rows[0]['split']='calibration'
    with pytest.raises(ValueError, match='train'):
        m.TrainingPools(rows)


def test_empty_required_sampling_pool_fails_before_training():
    m=adapter(); rows=[r for r in records() if not r['id'].startswith('qsc1/')]
    with pytest.raises(ValueError, match='pool'):
        m.sample_batch(m.TrainingPools(rows), seed=53, phase='mixed', epoch=1, step=0, arm='completion')


def test_selection_does_not_reward_rejecting_everything():
    m=adapter()
    empty={'precision':None,'false_accept_rate':0.,'exact':0,'macro_speaker_recall':0.,'raw_macro_speaker_recall':.8}
    good={'precision':1.,'false_accept_rate':0.,'exact':1,'macro_speaker_recall':.1,'raw_macro_speaker_recall':.7}
    assert m.selection_key({'human':good,'synthetic':good},4)>m.selection_key({'human':empty,'synthetic':empty},8)


def test_macro_speaker_recall_keeps_small_speakers_visible():
    m=adapter(); command={'intent':'MESSAGE','slots':{}}
    rows=[dict(expected=command,candidate=command,application_command=command,synthetic=False,speaker='large') for _ in range(9)]
    rows += [dict(expected=command,candidate=None,application_command=None,synthetic=False,speaker='small')]
    metric=m.summarize(rows)['human']
    assert metric['recall']==.9 and metric['macro_speaker_recall']==.5


def test_model_receives_input_lengths_and_ctc_receives_downsampled_lengths():
    m=adapter(); torch.set_num_threads(1)
    from scripts.models.pcv_release_pooling import PooledContextVCM

    class Spy(PooledContextVCM):
        def forward(self, features, lengths=None):
            self.seen_lengths=lengths.detach().cpu().tolist()
            return super().forward(features,lengths)

    model=Spy(width=16,layers=1,heads=4,feedforward=32,dropout=0.)
    rows=records()[:2]; rows[1]['target_index']=0
    arrays={rows[0]['id']:np.ones((40,15),dtype=np.float32), rows[1]['id']:np.ones((40,9),dtype=np.float32)}
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    loss=m.optimization_step(model,rows,arrays,opt,device='cpu',seed=53,augment_training=False,microbatch=2)
    assert model.seen_lengths==[15,9]
    assert all(np.isfinite(value) for value in loss)


def test_ctc_impossible_repeats_fail_before_optimizer_update():
    m=adapter(); row=records()[0]; row['phones']=[1,1,1,1]
    from scripts.models.pcv_release_pooling import PooledContextVCM
    model=PooledContextVCM(width=16,layers=1,heads=4,feedforward=32,dropout=0.)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with pytest.raises(ValueError,match='CTC'):
        m.optimization_step(model,[row],{row['id']:np.ones((40,7),dtype=np.float32)},
                            torch.optim.AdamW(model.parameters()),device='cpu',seed=53,augment_training=False)
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())


def test_padding_never_reaches_finite_decoder():
    m=adapter(); rows=records()[:2]
    for row in rows:row.update(source='fixture', text='message')
    arrays={rows[0]['id']:np.ones((40,15),dtype=np.float32),rows[1]['id']:np.ones((40,9),dtype=np.float32)}
    from scripts.models.pcv_release_pooling import PooledContextVCM
    model=PooledContextVCM(width=16,layers=1,heads=4,feedforward=32,dropout=0.)
    entries=[dict(command=rows[0]['expected'],phones=[1,2])]

    class Scorer:
        lengths=[]
        def map(self,values):
            for value in values:
                self.lengths.append(value.shape[0])
                yield dict(prediction=dict(command=entries[0]['command'],score=-1.,margin=.1,phrase_id='fixture'),scores=np.array([-1.]))

    scorer=Scorer(); result=m.score_model(model,rows,arrays,entries,scorer,device='cpu')
    assert len(result)==2 and scorer.lengths==[8,5]
