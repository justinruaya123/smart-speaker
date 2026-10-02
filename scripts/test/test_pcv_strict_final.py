"""Strict source/profile contracts; numerical probes are not speech accuracy tests."""
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
from scripts.test.pcv_strict_audit import (expand_schema, route_source, gate,
    cp_upper, load_reference, ctc_min_frames, split_components)


def test_schema_is_93_phrases_31_semantics_19_intents():
    rows = expand_schema(ROOT/'experiments/pcv-strict-final-v1/option_b.csv')
    assert len(rows) == 93
    assert len({r['semantic_key'] for r in rows}) == 31
    assert len({r['intent'] for r in rows}) == 19
    assert len({r['phrase'].lower() for r in rows}) == 93


@pytest.mark.parametrize('split', ['test','holdout'])
def test_reserved_voice_cannot_train_despite_train_filename(split):
    row = dict(repo='airimonda/ai231-me2-voice-commands', config='supplemental_synth',
               published_split='train', voice_split=split)
    assert route_source(row) == split


def test_unknown_supplement_voice_fails_closed():
    with pytest.raises(ValueError):
        route_source(dict(repo='airimonda/ai231-me2-voice-commands',
                          config='supplemental_synth', published_split='train'))


@pytest.mark.parametrize('repo', ['SLURP/slurp','google/speech_commands','custom/tts'])
def test_unapproved_repositories_rejected(repo):
    with pytest.raises(ValueError):
        route_source(dict(repo=repo, config='default', published_split='train'))


def test_optional_source_requires_explicit_opt_in():
    row=dict(repo='martinnavs/ai231-fil-supplemental-data',config='default',published_split='train')
    with pytest.raises(ValueError): route_source(row)
    assert route_source(row,allow_optional=True)=='train'


def test_numerals_auxiliary_not_commands():
    row=dict(repo='airimonda/ai231-me2-voice-commands',config='default',published_split='numerals')
    assert route_source(row)=='auxiliary'


def test_negative_train_and_test_not_merged():
    for s in ['train','test']:
        assert route_source(dict(repo='airimonda/ai231-me2-voice-commands',config='synthetic_negatives',published_split=s))==s


def test_derived_audio_parent_leakage_detected():
    rows=[dict(id='a',split='train',pcm='p1',parents=[]),
          dict(id='b',split='test',pcm='p2',parents=['a'])]
    with pytest.raises(ValueError,match='leak'): split_components(rows)


def test_exact_pcm_leakage_detected():
    with pytest.raises(ValueError,match='leak'):
        split_components([dict(id='a',split='train',pcm='same',parents=[]),dict(id='b',split='test',pcm='same',parents=[])])


def test_missing_parent_rejected():
    with pytest.raises(ValueError,match='parent'):
        split_components([dict(id='a',split='train',pcm='p1',parents=['missing'])])


def test_zero_negatives_is_not_evidence():
    assert cp_upper(0,0) is None
    result=gate(exact=95,positive_n=100,wrong_accepted=0,accepted=95,fa=0,negative_n=0,human_intents=19,pi_verified=True)
    assert not result['qualified']


def test_zero_250_cannot_certify_one_percent_at_95_one_sided():
    assert cp_upper(0,250) > .01
    assert cp_upper(0,299) < .01


def test_disabled_or_low_coverage_never_qualifies():
    result=gate(exact=100,positive_n=100,wrong_accepted=0,accepted=100,fa=0,negative_n=1000,human_intents=18,pi_verified=True)
    assert not result['qualified']


def test_end_to_end_denominator_includes_rejections():
    result=gate(exact=80,positive_n=100,wrong_accepted=0,accepted=80,fa=0,negative_n=1000,human_intents=19,pi_verified=True)
    assert result['recall']==.8
    assert result['precision']==1
    assert not result['qualified']


def test_ctc_repeat_minimum():
    assert ctc_min_frames([1,1,2,2])==6
    assert ctc_min_frames([1,2,1])==3


@pytest.mark.parametrize('seed', range(8))
def test_exact_prefix_matches_independent_torch_ctc(seed):
    import torch
    PrefixCTC=load_reference('decoder')['PrefixCTC']
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(19,40))
    paths=[[1],[1,1],[1,2],[1,2,3],[2,1,2],[3,3,3],[1,2]]
    got=PrefixCTC(paths).score(x)
    expected=[]
    for p in paths:
        loss=torch.nn.functional.ctc_loss(torch.tensor(x).log_softmax(-1)[:,None,:],torch.tensor(p),
              torch.tensor([len(x)]),torch.tensor([len(p)]),reduction='sum',zero_infinity=False)
        expected.append(-loss.item()/len(x))
    np.testing.assert_allclose(got,expected,rtol=0,atol=1e-12)


def test_prefix_matches_exhaustive_path_sum():
    PrefixCTC=load_reference('decoder')['PrefixCTC']
    x=np.full((4,40),-1000.);x[:,:3]=np.array([[1,2,3],[2,1,3],[3,2,1],[1,3,2]])
    p=np.exp(x[:,:3]-x[:,:3].max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    targets=[(1,),(1,1),(1,2),(2,1)]
    totals=dict.fromkeys(targets,0.)
    for path in itertools.product(range(3),repeat=4):
        collapsed=tuple(k for i,k in enumerate(path) if k and (i==0 or path[i-1]!=k))
        if collapsed in totals:totals[collapsed]+=np.prod([p[t,k] for t,k in enumerate(path)])
    got=PrefixCTC([list(t) for t in targets]).score(x)
    np.testing.assert_allclose(got,[np.log(totals[t])/4 for t in targets],atol=1e-12)


def test_chunk_equivalence():
    PrefixCTC=load_reference('decoder')['PrefixCTC'];x=np.random.default_rng(8).normal(size=(37,40))
    a=PrefixCTC([[1,1,2],[2,3,2]]);expected=a.score(x)
    a.reset()
    for chunk in [x[:1],x[1:7],x[7:19],x[19:]]:got=a.feed(chunk)
    np.testing.assert_allclose(got,expected,rtol=0,atol=0)


def test_subset_keeps_likelihood_not_margin():
    mod=load_reference('decoder');PrefixCTC=mod['PrefixCTC'];rank=mod['rank_commands']
    x=np.random.default_rng(12).normal(size=(80,40));paths=[[1,2],[1,2,3],[3,2],[5,6,7]]
    full=PrefixCTC(paths).score(x);subset=PrefixCTC(paths[:2]).score(x)
    np.testing.assert_array_equal(full[:2],subset)
    entries=[dict(id=str(i),command={'id':i},use_case='probe') for i in range(3)]
    a=rank(np.array([-.1,-.2,-.11]),entries)
    b=rank(np.array([-.1,-.2]),entries[:2])
    assert a['command']==b['command'] and b['margin']>a['margin']


def test_aliases_not_distinct_command_rivals():
    rank=load_reference('decoder')['rank_commands']
    entries=[dict(id=str(i),command={'id':k},use_case='probe') for i,k in enumerate([1,1,2])]
    assert rank(np.array([-.1,-.11,-.5]),entries)['margin']==.4


def test_blank_tail_changes_normalized_score():
    PrefixCTC=load_reference('decoder')['PrefixCTC']
    x=np.zeros((10,40));x[2,1]=4;x[6,2]=4
    tail=np.full((100,40),-10.);tail[:,0]=10.
    a=PrefixCTC([[1,2]]).score(x)[0];b=PrefixCTC([[1,2]]).score(np.concatenate([x,tail]))[0]
    assert b>a+1  # This exposes score-duration dependence, not improved recognition.


@pytest.mark.parametrize('frames', [1,3,37,100,199])
def test_reference_pooling_shape_and_padding(frames):
    import torch
    Model=load_reference('model')['PooledContextVCM']
    torch.manual_seed(42);m=Model(width=32,layers=2,heads=4,feedforward=64,dropout=0.).eval()
    x=torch.randn(1,40,frames)
    with torch.inference_mode():
        a=m(x,torch.tensor([frames]));b=m(torch.nn.functional.pad(x,(0,31)),torch.tensor([frames]))
    n=(frames+1)//2
    for actual,padded,c in zip(a[:3],b[:3],[40,19,2]):
        assert actual.shape==(1,c,n)
        torch.testing.assert_close(actual,padded[:,:,:n],rtol=1e-5,atol=2e-6)


def test_strict_candidates_masks_shapes_and_gradients():
    import torch
    from scripts.models.pcv_strict_candidates import StrictTinyVCM
    for kind in ['tcn','crnn']:
        torch.manual_seed(2);m=StrictTinyVCM(kind=kind).eval()
        x=torch.randn(2,40,83);lengths=torch.tensor([61,83])
        y=m(x,lengths)
        assert y['command'].shape==(2,32) and y['phrase'].shape==(2,94)
        assert y['ctc'].shape==(2,40,42)
        with torch.inference_mode():
            a=m(x[:1,:,:61],torch.tensor([61]));b=m(x[:1],torch.tensor([61]))
            torch.testing.assert_close(a['command'],b['command'],rtol=1e-4,atol=3e-6)
        loss=y['command'].square().mean()+y['phrase'].square().mean()+y['ctc'].square().mean()
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
