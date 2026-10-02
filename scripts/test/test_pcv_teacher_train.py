"""Numerical preflight for the trainable finite head, without downloading audio."""
import torch
from einops import rearrange


def test_ctc_frame_exclusions_account_for_repeated_cues():
    from scripts.train.pcv_teacher_train import ctc_eligible
    row = {'phones':[1, 1, 2, 3, 3]}
    assert not ctc_eligible(row, 6)
    assert ctc_eligible(row, 7)


def test_finite_head_has_finite_training_gradients():
    from scripts.models.pcv_teacher import FiniteTeacherHead
    from scripts.train.pcv_public_training import pooled_logits
    torch.manual_seed(733)
    model = FiniteTeacherHead(width=16, dilations=(1, 2))
    ctc, intent, target, _ = model(torch.randn(2, 768, 16))
    lengths = torch.tensor([16, 12])
    loss = torch.nn.CTCLoss()(rearrange(ctc.log_softmax(1), 'b c t -> t b c'),
                             torch.tensor([1, 2, 3, 3, 4]), lengths, torch.tensor([3, 2]))
    loss += torch.nn.functional.cross_entropy(pooled_logits(intent, lengths), torch.tensor([1, 0]))
    loss += torch.nn.functional.cross_entropy(pooled_logits(target, lengths), torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
