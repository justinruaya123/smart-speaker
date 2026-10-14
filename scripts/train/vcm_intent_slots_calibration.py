"""Separate human calibration; frozen score meaning, never threshold lowering."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json

import numpy as np
from scipy.optimize import minimize

from scripts.runtime.pcv_schema import INTENTS
from scripts.runtime.vcm_intent_slots_contract import digest
from scripts.runtime.vcm_intent_slots_decode import prediction_from_tensors, propose
from scripts.test.speaker_demo_suite import read_audio
from scripts.train.vcm_intent_slots_data import eligible, write_new


def fit_score_calibration(observations, binding, output):
    if not observations or any(r.get("split") != "calibration" or r.get("kind") != "human"
                               for r in observations):
        raise ValueError("Only separate human calibration observations may fit scores")
    if not all(binding.get(k) for k in ("contract_hash", "checkpoint", "normalization", "manifest", "frontend")):
        raise ValueError("Incomplete calibration binding")
    if set(r["intent_label"] for r in observations) != set(range(19)):
        raise ValueError("Missing human calibration intent classes; no qualification")
    fitted = _fit_numeric_scores(observations)
    policy = {"contract_hash": binding["contract_hash"], "binding": binding,
              "intent_temperature": fitted["intent_temperature"], "slot_calibrators": fitted["slot_calibrators"],
              "entities": binding.get("entities", {}),
              "missing_slot_calibration": fitted["missing_slot_calibration"], "require_calibration": True,
              "thresholds": {"intent": .95, "slot": .90},
              "calibration_people": sorted({r["speaker_id"] for r in observations}),
              "status": "frozen_partial_calibration" if fitted["missing_slot_calibration"] else "frozen_scores_unqualified",
              "qualification": False}
    policy["calibration_id"] = digest(policy)
    write_new(Path(output)/"calibration.json", policy)
    return policy


def _fit_numeric_scores(observations):
    """Identical temperature/logistic fitting after the caller validates provenance."""
    logits = np.asarray([r["intent_logits"] for r in observations], float)
    truth = np.asarray([r["intent_label"] for r in observations])

    def temperature_nll(log_t):
        t = np.exp(log_t[0])
        z = logits/t
        z = z-z.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(len(z)), truth]))

    opt = minimize(temperature_nll, [0.], method="L-BFGS-B", bounds=[(-3, 3)])
    if not opt.success:
        raise ValueError("Intent calibration optimization failed")
    temperature = float(np.exp(opt.x[0]))
    groups = defaultdict(list)
    for r in observations:
        for field, item in r.get("slots", {}).items():
            key = field+"/"+item["provenance"]+"/"+str(len(item["scores"]))
            groups[key].append(item)
    calibrators, missing = {}, []
    for key, group in groups.items():
        x = np.asarray([r["scores"] for r in group], float)
        y = np.asarray([r["correct"] for r in group], float)
        if set(y) != {0., 1.}:
            missing.append(key)
            continue

        def objective(w):
            z = x @ w[:-1]+w[-1]
            return float(np.mean(np.logaddexp(0, z)-y*z)+.001*np.square(w[:-1]).sum())

        fit = minimize(objective, np.zeros(x.shape[1]+1), method="BFGS")
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError("Slot calibration optimization failed: "+key)
        z = x @ fit.x[:-1]+fit.x[-1]
        probability = 1/(1+np.exp(-np.clip(z, -700, 700)))
        calibrators[key] = {"weights": fit.x[:-1].tolist(), "bias": float(fit.x[-1]),
                            "n": len(group), "positive_n": int(y.sum()),
                            "brier": float(np.mean((probability-y)**2)),
                            "feature_order": "role/value/request constituent scores; no probability product"}
    return {"intent_temperature": temperature, "slot_calibrators": calibrators,
            "missing_slot_calibration": missing}


def collect_calibration(rows, runtime, audio_root):
    observations = []
    for row in rows:
        eligible(row)
        if row["split"] != "calibration":
            raise ValueError("Separate calibration partition required")
        pcm = read_audio(row, Path(audio_root))
        from vcm.frontend import features
        tensors = runtime.infer_features(features(pcm, runtime.mean, runtime.std))
        prediction = prediction_from_tensors(tensors, runtime.contract, runtime.memory,
                                             {"entities": runtime.policy.get("entities", {})})
        proposal = propose(prediction, runtime.contract, runtime.catalog)
        candidate = proposal.get("candidate", {})
        expected = row.get("command") or {}
        actual_slots = candidate.get("slots", {})
        observations.append({
            "id": row["id"], "split": row["split"], "kind": row["kind"],
            "speaker_id": row["speaker_id"], "intent_logits": tensors["intent_logits"].tolist(),
            "intent_label": runtime.contract["intent_order"].index(expected.get("intent", "OOV")),
            "slots": {field: {"scores": evidence["constituent_scores"],
                              "provenance": evidence["provenance"],
                              "correct": int(candidate.get("intent") == expected.get("intent")
                                             and actual_slots.get(field) == expected.get("slots", {}).get(field))}
                      for field, evidence in proposal["slot_evidence"].items()},
        })
    return observations
