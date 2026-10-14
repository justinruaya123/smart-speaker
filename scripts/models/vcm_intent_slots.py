"""Random-weight complete-utterance temporal CNN with independent semantic queries."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from scripts.runtime.vcm_intent_slots_contract import verify_contract


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(128, 128, 5, padding=2*dilation,
                                   dilation=dilation, groups=128)
        self.norm = nn.LayerNorm(128)
        self.pointwise = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        y = self.depthwise(x * mask)
        y = rearrange(self.norm(rearrange(y, "b c t -> b t c")), "b t c -> b c t")
        y = self.dropout(self.pointwise(F.relu(y)))
        return (x + y) * mask


class IntentSlotsVCM(nn.Module):
    def __init__(self, contract):
        super().__init__()
        verify_contract(contract)
        self.contract = contract
        self.stem = nn.Conv1d(40, 128, 5, stride=2, padding=2)
        self.blocks = nn.ModuleList(TemporalBlock(d) for d in [1, 2, 4, 8, 1, 2, 4, 8])
        self.keys = nn.Linear(128, 64)
        self.queries = nn.Parameter(torch.randn(len(contract["roles"])+1, 64) * 0.02)
        self.intent = nn.Linear(128, 19)
        self.validity = nn.Linear(128, 4)
        self.states = nn.ModuleList(nn.Linear(128, 4) for _ in contract["roles"])
        self.values = nn.ModuleDict({r: nn.Linear(128, len(v))
                                     for r, v in contract["finite_values"].items()})
        self.entities = nn.ModuleList(nn.Linear(128, 64) for _ in contract["entity_roles"])

    def forward(self, features, lengths):
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if (features.ndim != 3 or features.shape[1] != 40
                    or lengths.shape != (features.shape[0],)
                    or lengths.dtype not in (torch.int32, torch.int64)
                    or (lengths < 1).any() or (lengths > features.shape[2]).any()
                    or not torch.isfinite(features).all()):
                raise ValueError("Expected finite [B,40,T] and positive valid frame lengths")
        mask = (torch.arange(features.shape[2], device=features.device)[None, :] < lengths[:, None])
        x = self.stem(features * mask[:, None, :])
        valid = (lengths + 1) // 2
        mask2 = (torch.arange(x.shape[2], device=x.device)[None, :] < valid[:, None])
        x = x * mask2[:, None, :]
        for block in self.blocks:
            x = block(x, mask2[:, None, :])
        frames = rearrange(x, "b c t -> b t c")
        logits = torch.matmul(self.keys(frames), self.queries.t()) / 8
        logits = logits.masked_fill(~mask2[:, :, None], -10000)
        weights = logits.softmax(dim=1)
        pooled = torch.matmul(rearrange(weights, "b t r -> b r t"), frames)
        roles = {r: pooled[:, n+1] for n, r in enumerate(self.contract["roles"])}
        return {
            "intent_logits": self.intent(pooled[:, 0]),
            "request_validity_logits": self.validity(pooled[:, 0]),
            "role_state_logits": torch.stack([h(pooled[:, n+1])
                                             for n, h in enumerate(self.states)], dim=1),
            "value_logits": {r: h(roles[r]) for r, h in self.values.items()},
            "entity_embeddings": torch.stack([
                F.normalize(h(roles[r]), dim=-1, eps=1e-8)
                for r, h in zip(self.contract["entity_roles"], self.entities)], dim=1),
            "role_frame_logits": logits, "encoded_lengths": valid,
        }

    def parameter_counts(self):
        counts = {}
        for name, p in self.named_parameters():
            group = name.split(".")[0]
            counts[group] = counts.get(group, 0) + p.numel()
        return {**counts, "total": sum(p.numel() for p in self.parameters())}


def joint_loss(prediction, target):
    """Normalized observed supervision; unknown labels use -100.

    Entity positives require the same role and entity from different people.
    No whole-command contrastive or CTC term is enabled in this version.
    Spans are (batch, role_plus_one, start_frame, exclusive_end_frame),
    supplied only by reviewed annotations, in encoded-frame coordinates.
    """
    zero = prediction["intent_logits"].sum() * 0
    counts = {}

    def ce(logits, labels, name):
        if labels is None:
            counts[name] = 0
            return zero
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        mask = labels != -100
        counts[name] = int(mask.sum().detach())
        return F.cross_entropy(logits[mask], labels[mask]) if mask.any() else zero

    loss = ce(prediction["intent_logits"], target.get("intent"), "intents")
    loss = loss + ce(prediction["role_state_logits"], target.get("role_states"), "role_states")
    loss = loss + ce(prediction["request_validity_logits"], target.get("validity"), "validity")
    terms, observed = [], 0
    for role, logits in prediction["value_logits"].items():
        term = ce(logits, target.get("values", {}).get(role), role)
        n = counts[role]
        if n:
            terms.append(term * n)
            observed += n
    loss = loss + (sum(terms) / observed if observed else zero)
    counts["observed_values"] = observed
    embeddings = prediction["entity_embeddings"]
    ids = target.get("entity_ids", [[None]*embeddings.shape[1] for _ in embeddings])
    speakers = target.get("speaker_ids", [None]*embeddings.shape[0])
    metric = []
    for role in range(embeddings.shape[1]):
        for a in range(embeddings.shape[0]):
            if ids[a][role] is None or speakers[a] is None:
                continue
            positives = [b for b in range(embeddings.shape[0])
                         if ids[b][role] == ids[a][role] and speakers[b] is not None
                         and speakers[b] != speakers[a]]
            negatives = [b for b in range(embeddings.shape[0])
                         if ids[b][role] is not None and ids[b][role] != ids[a][role]]
            for b in positives:
                positive = (embeddings[a, role]*embeddings[b, role]).sum()
                term = 1 - positive
                if negatives:
                    hardest = (embeddings[a, role]*embeddings[negatives, role]).sum(dim=-1).max()
                    term = term + F.relu(hardest - positive + 0.2)
                metric.append(term)
    counts["entity_pairs"] = len(metric)
    loss = loss + (0.1 * torch.stack(metric).mean() if metric else zero)
    localization = []
    for batch, role, start, end in target.get("spans", []):
        length = int(prediction["encoded_lengths"][batch])
        if not 0 <= start < end <= length:
            raise ValueError("Reviewed span outside valid acoustic frames")
        logp = prediction["role_frame_logits"][batch, :length, role].log_softmax(dim=0)
        localization.append(-torch.logsumexp(logp[start:end], dim=0))
    counts["reviewed_spans"] = len(localization)
    loss = loss + (0.1 * torch.stack(localization).mean() if localization else zero)
    return loss, counts
