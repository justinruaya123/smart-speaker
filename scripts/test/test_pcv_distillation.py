"""Distillation must preserve the finite decoder's scores and JSON competition."""
import numpy as np
import torch
from einops import rearrange


def test_differentiable_scores_match_finite_decoder():
    from scripts.train.pcv_distillation import finite_scores
    from scripts.pcv_parallel import ParallelScorer
    from pcvcm.decoder import PrefixCTC
    torch.manual_seed(912)
    logits = torch.randn(2, 40, 11, requires_grad=True)
    paths = [[[1, 2], [2, 2, 3], [4]], [[4, 1], [1, 3], [3, 3]]]
    actual = finite_scores(logits, [11, 8], paths)
    for i, n in enumerate((11, 8)):
        expected = PrefixCTC(paths[i]).score(rearrange(logits[i, :, :n].detach(), 'c t -> t c').numpy())
        np.testing.assert_allclose(actual[i].detach().numpy(), expected, atol=5e-7)
    actual.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[1, :, 8:]) == 0


def test_teacher_competition_collapses_same_command_aliases():
    from scripts.train.pcv_distillation import top_distinct
    time = {'intent': 'TIME', 'slots': {}}
    weather = {'intent': 'WEATHER', 'slots': {}}
    entries = [{'command': time}, {'command': time}, {'command': weather}, {'command': None}]
    indices, scores, valid = top_distinct(np.array([-.1, -.11, -.2, -.3]), entries, count=4)
    assert indices[:3] == [0, 2, 3]
    assert valid == [True, True, True, False]


def test_distribution_loss_excludes_padding_and_is_zero_at_match():
    from scripts.train.pcv_distillation import distribution_loss
    scores = torch.tensor([[-.1, -.2, -.3, 90.]], requires_grad=True)
    teacher = torch.tensor([[-.1, -.2, -.3, -90.]])
    valid = torch.tensor([[True, True, True, False]])
    loss = distribution_loss(scores, teacher, valid, temperature=.03)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert scores.grad[0, 3] == 0
