"""Finite human dev/calibration fit; no test or synthetic evidence is accepted."""
from scripts.runtime.pcv_schema import INTENTS
from scripts.runtime.speaker_compact import THRESHOLDS, apply_policy


def metrics(rows, thresholds, *, intent=None):
    policy = dict(intent_thresholds=thresholds)
    correct = wrong = false_accepts = 0
    for row in rows:
        event = dict(output={'accepted':False, 'reason':'capture_fault' if row.get('capture_fault') else 'unqualified_model'},
                     diagnostic_candidate=row)
        command = apply_policy(event, policy)['command']
        if command is None or (intent is not None and command['intent'] != intent):
            continue
        exact = command == row['expected']
        correct += exact
        wrong += not exact
        false_accepts += row['expected'] is None
    return dict(correct=correct, wrong=wrong, false_accepts=false_accepts)


def fit_policy(development, calibration, binding):
    views = dict(development=list(development), calibration=list(calibration))
    if any(r.get('synthetic') is not False for rows in views.values() for r in rows):
        raise ValueError('Only original human development/calibration can fit recall policy')
    identities = [{r['id'] for r in rows} for rows in views.values()]
    if identities[0] & identities[1]:
        raise ValueError('Human development/calibration identity overlap')
    baseline = {intent:.95 for intent in INTENTS}
    selected = dict(baseline); missing = []
    for intent in INTENTS:
        if any(not any(r['expected'] and r['expected']['intent'] == intent for r in rows)
               for rows in views.values()):
            missing.append(intent)
            continue
        before = {name:metrics(rows, baseline, intent=intent) for name,rows in views.items()}
        best = (sum(m['correct'] for m in before.values()), .95)
        for threshold in THRESHOLDS:
            trial = {**baseline, intent:threshold}
            after = {name:metrics(rows, trial, intent=intent) for name,rows in views.items()}
            if all(after[name]['wrong'] <= before[name]['wrong'] and
                   after[name]['false_accepts'] <= before[name]['false_accepts'] for name in views):
                key = (sum(m['correct'] for m in after.values()), threshold)
                if key > best:
                    best = key; selected[intent] = threshold
    return dict(format='speaker-compact-recall-v1', binding=dict(binding), qualified=False,
                intent_thresholds=selected, slot_threshold=.9,
                unrelaxed_missing_human_coverage=missing,
                selection='Maximize correct accepts without additional wrong or UNKNOWN accepts on either human view; higher threshold on ties',
                views={name:dict(baseline=metrics(rows,baseline), selected=metrics(rows,selected), rows=len(rows))
                       for name,rows in views.items()}, test_used=False, synthetic_used=False)
