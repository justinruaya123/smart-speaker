"""Development-only cascade: separate acoustic mistakes from disabled policy values."""
from collections import Counter, defaultdict
import json
import torch

from .complete_data import ROOT, DATA, EXP, CACHE, read_rows, save, digest
from .finite_policy import finite_reference, slot_records, command_metrics
from .family_contract import family_candidate, required_heads
from .metrics import policy_rows
from .routing import PARAMETER_COMMANDS, parse_parameters
from .complete_training import experiment_for


def diagnose(arm):
    folder = CACHE / arm
    bundle = json.loads((folder / 'bundle.json').read_text())
    rows = [r for r in read_rows(DATA / 'manifest.jsonl') if r['kind'] == 'human' and r['split'] == 'dev']
    outputs = torch.load(folder / 'development-logits.pt', weights_only=True)
    labels = bundle['vcm']['labels']
    probabilities = torch.softmax(outputs[0], -1).tolist()
    records = [{'id': r['id'], 'gold': r['command'], 'probabilities': p} for r, p in zip(rows, probabilities)]
    intents = policy_rows(records, labels, bundle['routing']['variants']['fp32'])
    slots = slot_records(outputs, bundle['acoustic_slots']['vocabulary'])
    selected = [i for i, r in enumerate(rows) if r['fold'] == 'selection']
    command, details = command_metrics([rows[i] for i in selected], [rows[i]['annotation'] for i in selected],
        [intents[i] for i in selected], [slots[i] for i in selected], bundle['acoustic_slots'], bundle['catalog'])
    families, failures, errors = defaultdict(Counter), Counter(), []
    policy = bundle['acoustic_slots']['variants']['fp32']
    for i, detail in zip(selected, details):
        row, intent, values = rows[i], intents[i], slots[i]
        gold = finite_reference(row['annotation'])
        if gold is None:
            continue
        family = row['command']
        raw_family = labels[max(range(len(labels)), key=probabilities[i].__getitem__)]
        candidate = family_candidate(family, values, bundle['acoustic_slots'], bundle['catalog'])
        raw_command = candidate['result'].get('command') if candidate['result'] else None
        if family not in PARAMETER_COMMANDS:
            raw_command = parse_parameters(family, '', bundle['catalog']).get('command')
        raw_correct = raw_family == family and raw_command == gold
        family_metrics = families[family]
        family_metrics.update(references=1, raw_family_correct=raw_family == family,
            accepted_family_correct=intent['accepted'] and intent['predicted'] == family,
            raw_command_correct=raw_correct, correct_complete=detail['correct_complete'])
        wrong_roles = [name for name in required_heads(family) if row['annotation']['labels'].get(name) is not None
            and candidate['values'].get(name) != row['annotation']['labels'][name]]
        blocked = [name + '=' + str(value) for name, value in candidate['values'].items() if name != 'guard'
            and candidate['confidence'][name] < policy['value_thresholds'].get(name, {}).get(value, 1.1)]
        if detail['correct_complete']:
            reason = 'correct'
        elif raw_family != family:
            reason = 'family_acoustic_error'
        elif not intent['accepted']:
            reason = 'family_policy_rejection'
        elif raw_command != gold:
            reason = 'argument_or_guard_acoustic_error'
        elif blocked:
            reason = 'value_policy_rejection'
        else:
            reason = 'joint_policy_rejection'
        failures[reason] += 1
        family_metrics[reason] += 1
        errors.append({'id': row['id'], 'family': family, 'reason': reason,
            'wrong_roles': wrong_roles, 'blocked_values': blocked, 'reference': gold,
            'raw_command': raw_command, 'raw_family': raw_family})
    result = {'arm': arm, 'partition': 'original human development selection only',
        'test_used': False, 'graph_sha256': digest(folder / 'vcm.fp32.onnx'),
        'commands': command, 'failure_cascade': failures, 'per_family': families,
        'note': 'Raw commands are diagnostic argmax outputs, not safe emitted commands. Gold-family role analysis is conditional.'}
    save(experiment_for(arm) / (arm + '-diagnosis.json'), result)
    with (folder / 'development-errors.jsonl').open('w', encoding='utf8') as stream:
        for error in errors:
            stream.write(json.dumps(error) + '\n')
    print(json.dumps(result), flush=True)
    return result
