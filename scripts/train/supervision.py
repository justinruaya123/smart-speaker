"""Recover explicit source supervision without changing the runtime grammar."""
import copy
from collections import Counter
import re

from .family_contract import assemble_family


def repair_reference(row, annotation, catalog):
    """FSC supplies a direction action for these human recordings.

    Recover only missing relative-volume labels, never overwrite known operations,
    infer a numeric amount, or borrow a source action for contradictory text.
    This deterministic rule can be applied to any split; only train labels fit
    weights and only development labels select a candidate or provisional policy.
    """
    repaired = copy.deepcopy(annotation)
    direction = {"increase/volume/none": "INCREASE", "decrease/volume/none": "DECREASE"}.get(row.get("source_intent"))
    if (row.get("source") != "FluentSpeechCommands" or row.get("command") != "VOLUME"
            or direction is None or annotation["labels"].get("operation") is not None):
        return repaired, None
    text = row.get("transcript", "").lower()
    # Do not turn a numeric, compound, negated, mute, or endpoint request into a default step.
    ambiguous = r"\d|\b(percent|percentage|half|double|maximum|minimum|max|min|mute|unmute|stop|zero|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|do not|don't|but|then|and)\b"
    if re.search(ambiguous, text):
        return repaired, None
    opposite = (r"\b(quieter|softer|decrease|reduce|lower)\b|\b(turn|volume|sound)\b.*\bdown\b|\btoo loud\b"
                if direction == "INCREASE" else
                r"\b(louder|increase|raise)\b|\b(turn|volume|sound)\b.*\bup\b|\btoo quiet\b")
    if re.search(opposite, text):
        return repaired, None
    repaired["labels"].update(operation=direction, amount="NONE", guard="READY")
    reference = assemble_family("VOLUME", repaired["labels"], catalog)
    if reference["status"] != "accepted":
        raise ValueError("Recovered source action must assemble under the unchanged contract")
    repaired["reference"] = reference
    repaired["quality"] = "source_action_with_existing_default_contract"
    repair = {"id": row["id"], "split": row["split"], "source_intent": row["source_intent"],
              "before": annotation["labels"], "after": repaired["labels"],
              "rule": "FSC explicit relative volume action; no numeric/compound/contradictory text"}
    repaired["supervision_provenance"] = repair["rule"]
    return repaired, repair


def prepare_references(rows, references, catalog):
    annotations, repairs = [], []
    for row in rows:
        annotation, repair = repair_reference(row, references[row["id"]], catalog)
        annotations.append(annotation)
        if repair:
            repairs.append(repair)
    return annotations, repairs


def coverage(rows, annotations):
    result = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [a for r, a in zip(rows, annotations) if r["split"] == split]
        heads = {name: dict(Counter(str(a["labels"].get(name)) for a in selected))
                 for name in ("operation", "amount", "target", "color")}
        numeric = {key: count for key, count in heads["amount"].items() if key.isdigit()}
        result[split] = {"examples": len(selected), "heads": heads,
                         "numeric_values": len(numeric), "numeric_recordings": sum(numeric.values()),
                         "missing_numeric_values": sorted(set(map(str, range(101))) - set(numeric), key=int)}
    return result
