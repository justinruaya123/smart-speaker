"""Sequence-level distillation of finite command scores, without text decoding."""
import numpy as np
import torch
from einops import rearrange, repeat
from scripts.runtime.pcv_schema import canonical


def finite_scores(logits, lengths, paths):
    """Differentiable counterpart of PrefixCTC's length-normalized log likelihood."""
    batch, _, _ = logits.shape
    count = len(paths[0])
    if len(paths) != batch or any(len(p) != count for p in paths):
        raise ValueError('Finite candidate count must be equal across the minibatch')
    expanded = repeat(logits.log_softmax(1), 'b c t -> t (b k) c', k=count)
    labels = [phone for row in paths for path in row for phone in path]
    target_lengths = [len(path) for row in paths for path in row]
    input_lengths = [n for n in lengths for _ in range(count)]
    loss = torch.nn.functional.ctc_loss(expanded,
        torch.tensor(labels, device=logits.device, dtype=torch.long),
        torch.tensor(input_lengths), torch.tensor(target_lengths),
        blank=0, reduction='none', zero_infinity=False)
    score = rearrange(-loss, '(b k) -> b k', b=batch, k=count)
    return score/rearrange(torch.tensor(lengths, device=logits.device), 'b -> b 1')


def top_distinct(scores, entries, count=8):
    """Keep one winning phrase per typed command, including the rejection candidate."""
    indices, chosen, valid, seen = [], [], [], set()
    for index in np.argsort(-np.asarray(scores), kind='stable'):
        if not np.isfinite(scores[index]): continue
        key = canonical(entries[index]['command'])
        if key in seen: continue
        seen.add(key); indices.append(int(index)); chosen.append(float(scores[index])); valid.append(True)
        if len(indices) == count: break
    if not indices: raise ValueError('No finite teacher candidate')
    while len(indices) < count:
        indices.append(indices[0]); chosen.append(chosen[0]); valid.append(False)
    return indices, chosen, valid


def distribution_loss(student_scores, teacher_scores, valid, temperature=.03):
    student_logp = (student_scores/temperature).masked_fill(~valid, -10000.).log_softmax(1)
    teacher_p = (teacher_scores/temperature).masked_fill(~valid, -10000.).softmax(1)
    return torch.nn.functional.kl_div(student_logp, teacher_p, reduction='batchmean')


def aligned_teacher_logits(teacher, student_frames):
    """Zero-order interpolation from 20 ms teacher centres to 10 ms student centres."""
    if teacher.shape[1] != (student_frames+1)//2:
        raise ValueError('Teacher/student time grids do not describe the same waveform')
    return repeat(teacher, 'c t -> c (t two)', two=2)[:, :student_frames]
