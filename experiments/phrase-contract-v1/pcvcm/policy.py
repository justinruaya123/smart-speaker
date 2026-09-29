"""One decision predicate for calibration and deployed endpoint acceptance."""
from __future__ import annotations

import math
import numpy as np

from .contract import Contract, canonical
from .evidence import audit_partitions, validate_reference


def candidate_allowed(candidate: dict, limits: list,
                      endpoint: bool, fault: str | None = None) -> bool:
    if fault or not endpoint or candidate.get('command') is None:
        return False
    score, margin = candidate.get('score'), candidate.get('margin')
    return bool(limits is not None and score is not None and margin is not None
                and math.isfinite(score) and math.isfinite(margin)
                and margin > 0 and score >= limits[0] and margin >= limits[1])


def decide(candidate: dict, policy: dict, binding: str, *,
           endpoint: bool, fault: str | None = None) -> dict:
    reason = None
    if fault:
        reason = 'capture_fault'
    elif policy.get('binding') != binding:
        reason = 'invalid_bundle'
    elif not endpoint:
        reason = 'tentative'
    elif policy.get('enabled') is not True:
        reason = 'policy_disabled'
    else:
        thresholds = policy.get('thresholds', {})
        limits = thresholds.get(candidate.get('use_case'),
                                thresholds.get('global'))
        if not candidate_allowed(candidate, limits, endpoint):
            reason = 'unknown'
    accepted = reason is None
    return {'schema_version': 2, 'accepted': accepted,
            'command': candidate['command'] if accepted else None,
            'reason': 'accepted' if accepted else reason}


def calibrate(rows: list[dict], contract: Contract, binding: str) -> dict:
    if not rows or any(r.get('split') != 'calibration' for r in rows):
        raise ValueError('Only frozen calibration data can fit acceptance')
    audit_partitions(rows)
    for row in rows:
        validate_reference(row, contract)
        if (row.get('human_reviewed') is not True
                or row.get('evidence_type') != 'human_waveform'
                or row.get('replay_verified') is not True
                or row.get('binding') != binding):
            raise ValueError('Calibration requires identity-bound human replay')
    for phrase in contract.positive + contract.negative:
        key = phrase.get('phrase_id', phrase.get('rejection_id'))
        selected = [r for r in rows if r.get('phrase_id') == key]
        if len(selected) < 8 or len({r['speaker_id'] for r in selected}) < 4:
            raise ValueError(f'Incomplete independent calibration coverage: {key}')
    diagnostic = [r.get('diagnostic', {}) for r in rows]
    scores = np.array([d['score'] if d.get('score') is not None else np.nan
                       for d in diagnostic])
    margins = np.array([d['margin'] if d.get('margin') is not None else np.nan
                        for d in diagnostic])
    commands = [d.get('command') for d in diagnostic]
    good = np.array([c is not None and canonical(c) ==
                     canonical(r.get('expected_command'))
                     for r, c in zip(rows, commands)])
    valid = np.array([r['reference_class'] == 'supported_positive' for r in rows])
    critical = np.array([
        contract.negative_by_id.get(r.get('phrase_id'), {}).get(
            'critical_zero_accept', False) for r in rows])
    base = np.array([candidate_allowed(d, [-float('inf'), 0.],
                                      r.get('endpoint_detected') is True,
                                      r.get('capture_fault'))
                     for d, r in zip(diagnostic, rows)])
    finite = np.isfinite(scores) & np.isfinite(margins) & base
    masks = {}
    for uc in [f'UC-{i:02}' for i in range(1, 11)]:
        actual = np.array([r['use_case'] == uc for r in rows])
        predicted = np.array([
            contract.intent_use_case.get((c or {}).get('intent')) == uc
            for c in commands])
        negatives = [actual & np.array([r['reference_class'] == kind for r in rows])
                     for kind in ('unknown_no_action', 'unsupported_valid_request')]
        masks[uc] = (actual & valid, predicted, negatives)
    score_grid = (np.unique(np.quantile(scores[finite], np.linspace(0, 1, 31)))
                  if finite.any() else [])
    margin_grid = (np.unique(np.quantile(margins[finite], np.linspace(0, 1, 21)))
                   if finite.any() else [])
    best, history = None, []
    for threshold in score_grid:
        for margin in margin_grid:
            accept = finite & (scores >= threshold) & (margins >= margin)
            successes = int((accept & good).sum())
            passed = bool(accept.any()) and not (accept & critical).any()
            for positive, predicted, negatives in masks.values():
                n = int(positive.sum())
                emitted = int((accept & predicted).sum())
                recall = int((accept & good & positive).sum()) / n if n else 0.
                precision = (int((accept & good & predicted).sum()) / emitted
                             if emitted else 0.)
                passed &= recall >= .9 and precision >= .99
                for negative in negatives:
                    n = int(negative.sum())
                    far = int((accept & negative).sum()) / n if n else 1.
                    passed &= n >= 50 and far <= .01
            history.append({'threshold': float(threshold), 'margin': float(margin),
                            'correct': successes, 'emitted': int(accept.sum()),
                            'feasible': bool(passed)})
            if passed and (best is None or successes > best['correct']):
                best = history[-1]
    return {'enabled': best is not None, 'binding': binding,
            'thresholds': {'global': [best['threshold'], best['margin']]
                           if best else [0., 1.]},
            'status': 'CALIBRATED_NOT_QUALIFIED' if best else 'BLOCKED_CALIBRATION',
            'search_evaluations': len(history), 'history': history,
            'minimum_per_uc_per_negative_class_calibration': 50,
            'model_selection_after_calibration': False}
