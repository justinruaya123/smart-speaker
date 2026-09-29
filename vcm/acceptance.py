import math


def candidate_reason(command, score, margin, min_score, min_margin):
    """The same candidate gate is used for calibration and deployment."""
    if not math.isfinite(score) or not math.isfinite(margin): return 'invalid_score'
    if command is None or margin <= 0.: return 'ambiguous_or_unsupported'
    if score < min_score or margin < min_margin: return 'below_threshold'
    return 'accepted'


def decide(command, score, margin, policy, binding, *, endpoint):
    reason = 'accepted'
    if not endpoint: reason='tentative'
    elif policy.get('binding') != binding: reason='stale_policy'
    elif not policy.get('enabled',False): reason='uncalibrated'
    else: reason=candidate_reason(command,score,margin,policy['min_score'],policy['min_margin'])
    return {'accepted':reason=='accepted','reason':reason,'command':command if reason=='accepted' else None,
            'candidate':command,'score':score if math.isfinite(score) else None,
            'margin':margin if math.isfinite(margin) else None}
