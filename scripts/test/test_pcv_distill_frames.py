import torch
from einops import rearrange


def test_teacher_frame_alignment_uses_valid_centres_only():
    from scripts.train.pcv_distillation import aligned_teacher_logits
    teacher = rearrange(torch.tensor([2., 4., 6.]), 't -> 1 t')
    torch.testing.assert_close(aligned_teacher_logits(teacher, 5),
                               torch.tensor([[2., 2., 4., 4., 6.]]))
    torch.testing.assert_close(aligned_teacher_logits(teacher, 6),
                               torch.tensor([[2., 2., 4., 4., 6., 6.]]))
