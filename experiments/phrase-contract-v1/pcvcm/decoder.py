"""Exact CTC likelihood using shared prefixes; no beam or phrase pruning."""
from __future__ import annotations

import numpy as np
from .contract import canonical


class PrefixCTC:
    """Two CTC states per phone prefix, with independent terminal probabilities.

    Each node has a blank-ending and a nonblank-ending probability. Sharing a
    prefix does not merge different complete commands or approximate their score.
    Repeated labels may transition from the parent's blank state only.
    """

    def __init__(self, sequences: list[list[int]]):
        if not sequences or any(not seq or any(type(p) is not int or not 1 <= p < 40
                                              for p in seq) for seq in sequences):
            raise ValueError('Nonempty phone sequences in 1..39 are required')
        parents, labels, edges, terminals = [0], [0], {}, []
        for sequence in sequences:
            parent = 0
            for phone in sequence:
                key = (parent, phone)
                if key not in edges:
                    edges[key] = len(parents)
                    parents.append(parent)
                    labels.append(phone)
                parent = edges[key]
            terminals.append(parent)
        self.parent = np.asarray(parents, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.terminals = np.asarray(terminals, dtype=np.int64)
        self.can_skip = self.labels != self.labels[self.parent]
        self.node_count = len(parents)
        self.reset()

    def reset(self) -> None:
        self.blank = np.full(self.node_count, -np.inf)
        self.nonblank = np.full(self.node_count, -np.inf)
        self.blank[0] = 0.0
        self.frames = 0
        self.best_frame_logp = 0.0

    def feed(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 40 or not np.isfinite(x).all():
            raise ValueError('Expected finite raw logits shaped [frames,40]')
        if not len(x):
            return self.scores()
        x = x - x.max(axis=1, keepdims=True)
        logp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
        self.best_frame_logp += float(logp.max(axis=1).sum())
        for frame in logp:
            next_blank = np.logaddexp(self.blank, self.nonblank) + frame[0]
            parent_nonblank = np.where(self.can_skip,
                                       self.nonblank[self.parent], -np.inf)
            incoming = np.logaddexp(self.blank[self.parent], parent_nonblank)
            next_nonblank = np.logaddexp(self.nonblank, incoming) + frame[self.labels]
            next_nonblank[0] = -np.inf
            self.blank, self.nonblank = next_blank, next_nonblank
            self.frames += 1
        return self.scores()

    def scores(self) -> np.ndarray:
        return np.logaddexp(self.blank[self.terminals],
                           self.nonblank[self.terminals]) / max(1, self.frames)

    def score(self, logits: np.ndarray) -> np.ndarray:
        self.reset()
        return self.feed(logits)


def rank_commands(scores: np.ndarray, entries: list[dict]) -> dict:
    """Compare distinct full JSON commands, not competing aliases of one command."""
    scores = np.asarray(scores)
    if scores.shape != (len(entries),) or np.isnan(scores).any():
        raise ValueError('Invalid candidate score array')
    order = np.argsort(-scores, kind='stable')
    top = int(order[0])
    best = entries[top]
    key = canonical(best['command'])
    rivals = [int(i) for i in order[1:]
              if canonical(entries[int(i)]['command']) != key]
    margin = float(scores[top] - scores[rivals[0]]) if rivals else 0.0
    if not np.isfinite(scores[top]):
        return {'command': None, 'phrase_id': None, 'score': None,
                'margin': 0.0, 'use_case': None}
    return {'command': best['command'], 'phrase_id': best['id'],
            'score': float(scores[top]), 'margin': margin,
            'use_case': best['use_case']}
