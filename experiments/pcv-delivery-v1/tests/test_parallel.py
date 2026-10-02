"""Execution-only acceleration must preserve the exact finite decoder."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from pcvcm.decoder import PrefixCTC,rank_commands


def module():
    try:return importlib.import_module('scripts.pcv_parallel')
    except ModuleNotFoundError:pytest.fail('Parallel evaluator is not implemented')


def entries():
    return [dict(id='one',phones=[1,1,2],command={'intent':'A','value':1},use_case='UC-01'),
            dict(id='alias',phones=[1,2],command={'intent':'A','value':1},use_case='UC-01'),
            dict(id='two',phones=[1,3],command={'intent':'A','value':2},use_case='UC-01'),
            dict(id='reject',phones=[4,4],command=None,use_case='UC-01')]


def test_parallel_scores_ranking_and_order_match_exact_reference():
    api=module();catalog=entries();rng=np.random.default_rng(812)
    arrays=[rng.normal(size=(frames,40)).astype(np.float32) for frames in (1,7,501,23,12)]
    decoder=PrefixCTC([e['phones'] for e in catalog])
    expected=[(decoder.score(x).copy(),None) for x in arrays]
    with api.ParallelScorer(catalog,workers=2,return_scores=True) as scorer:
        actual=list(scorer.map(arrays))
    assert len(actual)==len(arrays)
    for item,(scores,_) in zip(actual,expected):
        np.testing.assert_array_equal(item['scores'],scores)
        assert item['prediction']==rank_commands(scores,catalog)


def test_parallel_propagates_invalid_recording_instead_of_omitting_it():
    api=module()
    with api.ParallelScorer(entries(),workers=2) as scorer:
        with pytest.raises(ValueError,match='finite raw logits'):
            list(scorer.map([np.zeros((2,40)),np.full((3,40),np.nan),np.zeros((2,40))]))


def test_parallel_submission_is_bounded_and_stop_does_not_consume_corpus():
    api=module();consumed=[]
    def source():
        for i in range(100):
            consumed.append(i);yield np.zeros((3,40))
    with api.ParallelScorer(entries(),workers=2,max_pending=4) as scorer:
        results=scorer.map(source());next(results)
        assert len(consumed)<=4
        results.close()


def test_development_preserves_predictions_and_metrics_on_artificial_features():
    api=module()
    import torch
    from pcvcm.training import development
    from vcm.model import CausalVCM
    torch.manual_seed(81);torch.set_num_threads(1)
    model=CausalVCM();catalog=entries();rng=np.random.default_rng(27)
    rows=[dict(recording_id=str(i),phrase_id=str(i),split='dev',use_case='UC-01',
               reference_class='supported_positive' if i<2 else 'negative',
               expected_command=catalog[i]['command']) for i in range(4)]
    arrays={str(i):rng.normal(size=(40,40+i*7)).astype(np.float32) for i in range(4)}
    expected=development(model,rows,arrays,catalog,'cpu')
    random_before=torch.get_rng_state().clone();checks=[]
    actual=api.development(model,rows,arrays,catalog,'cpu',lambda:checks.append(True),workers=2)
    assert actual==expected and checks
    assert torch.equal(random_before,torch.get_rng_state())


def test_parallel_rejects_invalid_worker_budget():
    api=module()
    for n in (0,33):
        with pytest.raises(ValueError):api.ParallelScorer(entries(),workers=n)
