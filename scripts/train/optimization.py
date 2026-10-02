"""Balanced train-only sampling and unchanged acceptance criteria."""
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from einops import reduce


def balanced_draws(rows, annotations, rng, per_family=600):
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Sampling may use training recordings only")
    families = defaultdict(list)
    for i, row in enumerate(rows):
        families[row["command"]].append(i)
    draws = [rng.choice(ids) for ids in families.values() for _ in range(per_family)]
    # Add value-balanced full commands, not isolated words or synthetic audio.
    for head in ("operation", "amount", "target", "color"):
        groups = defaultdict(list)
        for i, annotation in enumerate(annotations):
            value = annotation["labels"].get(head)
            if value not in {None, "NONE", "OTHER"}:
                groups[value].append(i)
        keys = sorted(groups)
        if keys:
            draws.extend(rng.choice(groups[rng.choice(keys)]) for _ in range(per_family))
    rng.shuffle(draws)
    return draws


def head_weights(annotations, vocabulary, device):
    weights = {}
    for name, values in vocabulary.items():
        counts = Counter(a["labels"].get(name) for a in annotations)
        count = torch.tensor([counts[value] for value in values], dtype=torch.float32, device=device)
        supported = count > 0
        w = torch.zeros_like(count)
        w[supported] = (reduce(count[supported], "c ->", "mean") / count[supported]).sqrt().clamp(.25, 4.)
        weights[name] = w
    return weights


def balanced_losses(outputs, targets, masks, weights):
    losses = {}
    for name, logits in outputs.items():
        mask = masks[name]
        losses[name] = (F.cross_entropy(logits[mask], targets[name][mask], weight=weights[name])
                        if mask.any() else reduce(logits, "b c ->", "sum") * 0.)
    return losses


def fit_intent_policy(records, labels, temperature=1.):
    """Development only: >=20 accepts, >=95% class precision, <=1% UNKNOWN.

    A small integer-budget dynamic program shares the UNKNOWN allowance across
    classes while maximizing correct accepts. The runtime decision rule is unchanged.
    """
    from .inference import softmax
    prepared = []
    for row in records:
        p = softmax(np.log(np.maximum(row["probabilities"], 1e-30)), temperature)
        prepared.append((row["gold"], labels[int(p.argmax())], float(p.max())))
    budget = int(sum(gold == "UNKNOWN" for gold, _, _ in prepared) * .01)
    options = {}
    for label in labels[1:]:
        candidates = sorted([(confidence, gold) for gold, predicted, confidence in prepared if predicted == label], reverse=True)
        by_cost = {0: (0, 0, 1.1)}
        correct = unknown = 0
        for i, (confidence, gold) in enumerate(candidates, 1):
            correct += gold == label
            unknown += gold == "UNKNOWN"
            if i < len(candidates) and candidates[i][0] == confidence:
                continue
            if i >= 20 and correct / i >= .95 and unknown <= budget:
                option = (correct, -i, confidence)
                if option > by_cost.get(unknown, (-1, 0, 0)):
                    by_cost[unknown] = option
        options[label] = by_cost
    states = {0: (0, 0, {})}
    for label in labels[1:]:
        updated = {}
        for used, (good, negative_count, thresholds) in states.items():
            for cost, (correct, negative_n, threshold) in options[label].items():
                if used + cost > budget:
                    continue
                candidate = (good + correct, negative_count + negative_n, {**thresholds, label: threshold})
                if used + cost not in updated or candidate[:2] > updated[used + cost][:2]:
                    updated[used + cost] = candidate
        states = updated
    selected = max(states.values(), key=lambda item: item[:2])
    return {"temperature": temperature, "acceptance_thresholds": selected[2],
            "partition": "development policy_fit only; not final calibration", "minimum_accepts": 20,
            "rule": ">=95% per-family observed precision, >=20 accepts, shared <=1% UNKNOWN budget",
            "unknown_budget": budget}
