"""Shared deployed acceptance predicate, without calibration tooling."""
from __future__ import annotations
import math


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


