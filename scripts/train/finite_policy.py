from collections import Counter
import torch
from .family_contract import LABELS, family_candidate, route_family
from .slot_contract import apply_value_policy
from .routing import PARAMETER_COMMANDS, parse_parameters

def finite_reference(annotation: dict) -> dict | None:
    reference = annotation["reference"]
    if not reference or reference["status"] != "accepted":
        return None
    if set(reference["command"]["arguments"]) & {"title", "artist", "selection_text"}:
        return None
    return reference["command"]

def threshold(scores: list[float], correct: list[bool], precision: float = .95) -> float:
    options = []
    for value in sorted(set(scores)):
        accepted = [good for score, good in zip(scores, correct) if score >= value]
        if len(accepted) >= 20 and sum(accepted)/len(accepted) >= precision:
            options.append((sum(accepted), -len(accepted), value))
    return max(options)[2] if options else 1.1

def slot_records(outputs: tuple, vocab: dict) -> list[dict]:
    values = {name: torch.softmax(output, -1).cpu().numpy()
              for name, output in zip(vocab, outputs[1:])}
    return [{name: p[i] for name, p in values.items()} for i in range(len(outputs[0]))]

def provisional_slots(rows, annotations, accepted, slots, schema, catalog):
    policy = {"value_thresholds": {}, "acceptance_thresholds": {},
              "custom_media_threshold": 1.1, "custom_guard_threshold": 1.1,
              "clarification_thresholds": {}, "partition": "development policy_fit only"}
    candidates = [family_candidate(r["predicted"], p, schema, catalog)
                  if r["predicted"] != "UNKNOWN" else None for r, p in zip(accepted, slots)]
    for name, values in schema["vocabulary"].items():
        if name == "guard":
            continue
        policy["value_thresholds"][name] = {}
        for value in values:
            scores, correct = [], []
            for row, annotation, intent, candidate in zip(rows, annotations, accepted, candidates):
                if (not intent["accepted"] or intent["predicted"] != row["command"] or not candidate
                        or candidate["values"].get(name) != value):
                    continue
                gold = annotation["labels"].get(name)
                if gold is not None:
                    scores.append(candidate["confidence"][name])
                    correct.append(gold == value)
            policy["value_thresholds"][name][value] = threshold(scores, correct, .99 if name == "color" else .95)
    for family in LABELS[1:]:
        scores, correct = [], []
        for row, annotation, intent, candidate in zip(rows, annotations, accepted, candidates):
            if not intent["accepted"] or intent["predicted"] != family or not candidate:
                continue
            candidate = apply_value_policy(candidate, policy)
            if candidate["score"] > 0:
                scores.append(candidate["score"])
                correct.append(row["command"] == family and candidate["result"]["command"] == finite_reference(annotation))
        policy["acceptance_thresholds"][family] = threshold(scores, correct)
    # Custom routing remains off; this pilot does not establish a custom-parameter gate.
    return policy

def command_metrics(rows, annotations, accepted, slots, config, catalog):
    emitted = correct = resolvable = 0
    per_family = {c: Counter() for c in LABELS}
    errors = []
    for row, annotation, intent, values in zip(rows, annotations, accepted, slots):
        reference = finite_reference(annotation)
        resolvable += reference is not None
        per_family[row["command"]]["resolvable"] += reference is not None
        decision = {"decision": "rejected"}
        if intent["accepted"]:
            decision = route_family(intent["predicted"], values, config, catalog, "fp32")
            if decision["decision"] == "fallback" and intent["predicted"] not in PARAMETER_COMMANDS:
                decision = {"decision": "accepted", **parse_parameters(intent["predicted"], "", catalog)}
        command = decision.get("command") if decision["decision"] == "accepted" else None
        success = command is not None and row["command"] == intent["predicted"] and command == reference
        emitted += command is not None
        correct += success
        per_family[row["command"]]["correct_complete"] += success
        per_family[row["command"]]["emitted"] += command is not None
        errors.append({"id": row["id"], "gold": row["command"], "predicted": intent["predicted"],
                       "intent_accepted": intent["accepted"], "command": command,
                       "reference": reference, "correct_complete": success,
                       "decision": decision["decision"], "parameter_invoked": False})
    return {"correct_complete": correct, "resolvable_finite_references": resolvable,
            "emitted": emitted, "exact_command_success": correct/max(1, resolvable),
            "accepted_command_precision": correct/max(1, emitted), "per_family": per_family,
            "parameter_calls": 0, "command_promotion_assessed": False,
            "denominator": "all silver resolvable finite commands; open-text media reported as clarification"}, errors
