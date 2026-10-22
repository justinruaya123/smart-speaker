"""Evaluation-only OptionB scope and confidence accounting; no fitting or rescue."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
import math

from scripts.runtime.pcv_schema import INTENTS, baseline_cases, canonical, validate_command
from scripts.runtime.speaker_confidence import application_decision

OPTIONB_COMMANDS = frozenset(canonical(case['command']) for case in baseline_cases())


def optionb_catalog(entries):
    """Restrict complete positive commands, never aliases or negative competitors."""
    selected = []
    for entry in entries:
        command = entry['command']
        if command is not None:
            validate_command(command)
        if command is None or canonical(command) in OPTIONB_COMMANDS:
            selected.append(deepcopy(entry))
    present = {canonical(e['command']) for e in selected if e['command'] is not None}
    if present != OPTIONB_COMMANDS:
        raise ValueError('Catalog does not cover all OptionB complete commands')
    if not any(e['command'] is None for e in selected):
        raise ValueError('Catalog must retain negative competitors')
    return selected


def scope_records(records, *, output_allowlist=False):
    """Keep every holdout row, distinguishing outside-scope negatives from UNKNOWN."""
    records = list(records)
    if (len({r['id'] for r in records}) != len(records)
            or any(r.get('split') != 'holdout' or r.get('training_eligible') is not False
                   or type(r.get('metric_eligible')) is not bool for r in records)):
        raise ValueError('Expected unique, evaluation-only holdout rows')
    result = []
    for row in records:
        expected = row['expected']
        if expected is not None:
            validate_command(expected)
        if not row['metric_eligible']:
            scope = 'ambiguous'
        elif expected is None:
            scope = 'true_unknown'
        elif canonical(expected) in OPTIONB_COMMANDS:
            scope = 'optionb'
        else:
            scope = 'outside_optionb'
        accepted = row['application_command']
        if accepted is not None:
            validate_command(accepted)
        reject_output = (output_allowlist and accepted is not None
                         and canonical(accepted) not in OPTIONB_COMMANDS)
        result.append({**row, 'reference_before_scope': expected, 'reference_scope': scope,
                       'expected': None if scope == 'outside_optionb' else expected,
                       'application_before_scope': accepted,
                       'application_command': None if reject_output else accepted,
                       'application_reason': ('outside_optionb_output_scope' if reject_output
                                              else row.get('application_reason'))})
    return result


def complete_metrics(records, key='application_command'):
    """Reconcile exact typed-command errors; wrong slots incur both FP and FN."""
    from scripts.test.pcv_public_qualification import precise_metrics

    records = list(records)
    eligible = [r for r in records if r['metric_eligible']]
    result = precise_metrics(eligible, key)
    overall = result['overall']
    tp = overall['exact']
    fp, fn = overall['accepted'] - tp, overall['positive_n'] - tp
    denominator = 2 * tp + fp + fn
    overall.update(tp=tp, fp=fp, fn=fn, f1=2 * tp / denominator if denominator else None,
                   ambiguous_n=len(records) - len(eligible),
                   known_rejections=sum(r['expected'] is not None and r[key] is None for r in eligible),
                   wrong_positive_accepts=fp - overall['negative_errors'])
    for values in result['per_intent'].values():
        tp = values['complete_correct']
        fp, fn = values['predicted_n'] - tp, values['reference_n'] - tp
        denominator = 2 * tp + fp + fn
        values.update(tp=tp, fp=fp, fn=fn, f1=2 * tp / denominator if denominator else None)
    result['negative_groups'] = {}
    for name in ('true_unknown', 'outside_optionb'):
        members = [r for r in eligible if r.get('reference_scope',
                   'true_unknown' if r['expected'] is None else 'optionb') == name]
        n = len(members)
        errors = sum(r[key] is not None for r in members)
        result['negative_groups'][name] = dict(n=n, false_accepts=errors,
                                             false_accept_rate=errors / n if n else None)
    return result


def _failed_gates(record, config):
    command = record.get('candidate')
    try:
        validate_command(command)
    except (ValueError, TypeError, KeyError):
        return ['invalid_command']

    def passes(value, threshold):
        return type(value) in (int, float) and math.isfinite(value) and threshold <= value <= 1

    failed = []
    if not passes(record.get('intent_probability'), config['intent_threshold']):
        failed.append('intent_confidence')
    if record.get('intent_index') != INTENTS.index(command['intent']) + 1:
        failed.append('intent_disagreement')
    if any(not passes(record.get('slot_confidence', {}).get(name), config['slot_threshold'])
           for name in command['slots']):
        failed.append('slot_confidence')
    if command['intent'] in {'LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'}:
        value = record.get('target_named_probability')
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            failed.append('target_confidence')
        else:
            named = command['slots']['light_id'] != 'light.all'
            if not passes(value if named else 1 - value, config['slot_threshold']):
                failed.append('target_confidence')
    return failed


def gate_audit(records, config):
    """Replay real decisions and expose masked disagreements without changing them."""
    if config.get('mode') != 'demo':
        raise ValueError('This diagnostic accounts for the declared demo policy only')
    records = [r for r in records if r['metric_eligible']]
    outcomes, first, independent = Counter(), Counter(), Counter()
    details = []
    for row in records:
        diagnostic = row.get('diagnostic_candidate') or {}
        capture = row.get('application_reason') == 'capture_fault'
        event = dict(diagnostic_candidate=diagnostic,
                     output={'reason': 'capture_fault' if capture else 'unqualified_model'})
        application = application_decision(event, config)
        if (application['command'] != row['application_command']
                or application['reason'] != row.get('application_reason')
                or diagnostic.get('candidate') != row.get('candidate')):
            raise ValueError('Stored decision does not replay: ' + row['id'])
        expected = row['expected']
        if expected is None or canonical(expected) not in OPTIONB_COMMANDS:
            continue
        raw_correct = row.get('candidate') is not None and canonical(row['candidate']) == canonical(expected)
        index = diagnostic.get('intent_index')
        head = (INTENTS[index - 1] if index else 'UNKNOWN') if type(index) is int and 0 <= index <= len(INTENTS) else None
        head_correct = head == expected['intent']
        flags = ['capture_fault'] if capture else _failed_gates(diagnostic, config)
        if (flags[0] if flags else None) != application['reason']:
            raise ValueError('Independent gate flags disagree with actual decision: ' + row['id'])
        if application['command'] is not None:
            category = 'correct_accepted' if application['command'] == expected else 'wrong_accepted'
        elif not raw_correct:
            category = 'incorrect_raw_rejected'
        elif head is None:
            category = 'correct_raw_missing_intent_head'
        elif not head_correct:
            category = 'correct_raw_wrong_intent_head'
        elif 'intent_confidence' in flags:
            category = 'correct_raw_matching_head_below_threshold'
        else:
            category = 'correct_raw_' + str(application['reason'])
        outcomes[category] += 1
        independent.update(flags)
        if application['command'] is None:
            first[application['reason']] += 1
        details.append(dict(id=row['id'], text=row.get('text'), speaker=row.get('speaker'),
                            expected=expected, candidate=row.get('candidate'),
                            application_command=application['command'], raw_correct=raw_correct,
                            predicted_head=head, head_correct=head_correct,
                            intent_probability=diagnostic.get('intent_probability'),
                            top_intent_probability=diagnostic.get('top_intent_probability'),
                            slot_confidence=diagnostic.get('slot_confidence', {}),
                            target_named_probability=diagnostic.get('target_named_probability'),
                            first_rejection=application['reason'], failed_gates=flags, outcome=category))
    return dict(reference_n=len(details), raw_correct=sum(r['raw_correct'] for r in details),
                raw_correct_rejected=sum(r['raw_correct'] and r['application_command'] is None for r in details),
                head_reference_n=sum(r['predicted_head'] is not None for r in details),
                head_correct=sum(r['head_correct'] for r in details),
                positive_speakers=len({r['speaker'] for r in details}), gate_replayed_n=len(records),
                disjoint_outcomes=dict(outcomes), first_rejections=dict(first),
                independent_failed_gates=dict(independent), records=details)
