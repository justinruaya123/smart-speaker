from collections import Counter
import numpy as np

def wilson(good: int, count: int) -> list[float] | None:
    if not count:
        return None
    z = 1.959963984540054
    p = good / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    radius = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [float(center-radius), float(center+radius)]

def metrics(records: list[dict], labels: list[str]) -> dict:
    counts = {c: Counter() for c in labels}
    confusion = {c: Counter({label: 0 for label in labels}) for c in labels}
    for row in records:
        gold, predicted, accepted = row["gold"], row["predicted"], row["accepted"]
        counts[gold]["support"] += 1
        confusion[gold][predicted] += 1
        counts[gold]["raw_correct"] += gold == predicted
        counts[gold]["abstentions"] += not accepted
        if accepted:
            counts[predicted]["accepted_predictions"] += 1
            counts[predicted]["wrong_accepts"] += gold != predicted
            counts[gold]["correct_accepted"] += gold == predicted
            counts[gold]["false_accepts_on_reference"] += gold != predicted
    good = sum(v["correct_accepted"] for v in counts.values())
    emitted = sum(v["accepted_predictions"] for v in counts.values())
    for c, row in counts.items():
        n, a = row["support"], row["accepted_predictions"]
        row["raw_recall"] = row["raw_correct"] / n if n else None
        row["correct_accepted_recall"] = row["correct_accepted"] / n if n else None
        row["accepted_precision"] = row["correct_accepted"] / a if a else None
        row["raw_recall_wilson95"] = wilson(row["raw_correct"], n)
        row["correct_accepted_recall_wilson95"] = wilson(row["correct_accepted"], n)
        row["accepted_precision_wilson95"] = wilson(row["correct_accepted"], a)
    unknown = counts["UNKNOWN"]
    low_raw = [c for c, r in counts.items() if r["raw_recall"] is None or r["raw_recall"] < .8]
    low_accepted = [c for c, r in counts.items() if c != "UNKNOWN" and
                    (r["correct_accepted_recall"] is None or r["correct_accepted_recall"] < .6)]
    low_precision = [c for c, r in counts.items() if c != "UNKNOWN" and
                     (r["accepted_precision"] is None or r["accepted_precision"] < .95)]
    unknown_rate = unknown["false_accepts_on_reference"] / max(1, unknown["support"])
    precision = good / max(1, emitted)
    return {"per_family": counts, "confusion": confusion,
            "correct_accepted": good, "accepted": emitted, "accepted_precision": precision,
            "unknown_false_accepts": unknown["false_accepts_on_reference"],
            "unknown_support": unknown["support"], "unknown_false_accept_rate": unknown_rate,
            "unknown_rate_wilson95": wilson(unknown["false_accepts_on_reference"], unknown["support"]),
            "gates": {"low_raw_recall": low_raw, "low_accepted_recall": low_accepted,
                      "additional_low_per_intent_precision": low_precision,
                      "global_precision_pass": precision >= .95, "unknown_pass": unknown_rate <= .01,
                      "inherited_pass": not low_raw and not low_accepted and precision >= .95 and unknown_rate <= .01},
            "uncertainty_note": "Wilson intervals describe recording counts, not independent-speaker or seed uncertainty."}

def policy_rows(records: list[dict], labels: list[str], policy: dict) -> list[dict]:
    from tinyvcm.inference import softmax
    output = []
    for row in records:
        p = softmax(np.log(np.maximum(row["probabilities"], 1e-30)), policy.get("temperature", 1.))
        command = labels[int(p.argmax())]
        confidence = float(p.max())
        output.append({**row, "predicted": command, "confidence": confidence,
                       "accepted": command != "UNKNOWN" and confidence >= policy["acceptance_thresholds"][command]})
    return output
