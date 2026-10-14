"""Action-free replay and denominator-preserving, partition-bound selection."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from scripts.runtime.pcv_schema import canonical
from scripts.train.vcm_intent_slots_data import sha, write_new
from scripts.test.vcm_intent_slots_diversity import known_identity

LOCK_KEYS = {"protocol", "manifest", "checkpoint", "contract", "catalog",
             "normalization", "calibration", "frontend", "runtime"}


def freeze_selection(bindings, output):
    if not LOCK_KEYS <= set(bindings):
        raise ValueError("Incomplete selection-before-test binding")
    files = {name: {"path": str(Path(path).resolve()), "sha256": sha(path)}
             for name, path in bindings.items()}
    lock = {"version": 1, "complete": True, "files": files,
            "thresholds": {"intent": .95, "slot": .90},
            "fresh_test_access": "only after this immutable selection lock"}
    write_new(output, lock)
    return lock


def verify_selection(path):
    lock = json.loads(Path(path).read_text())
    if lock.get("complete") is not True or not LOCK_KEYS <= set(lock.get("files", {})):
        raise ValueError("Incomplete selection lock")
    for name, binding in lock["files"].items():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Stale selection binding: " + name)
    return lock


def verify_evaluation_context(lock, runtime, manifest, protocol=None):
    if (runtime.policy.get("require_calibration") is not True
            or not runtime.policy.get("calibration_id")):
        raise ValueError("Fresh runtime needs frozen calibration")
    files = lock.get("files", {})
    expected = {"runtime": sha(runtime.root/"recipe.json"),
                "checkpoint": runtime.recipe["checkpoint_source_sha256"],
                "manifest": sha(manifest)}
    for name, filename in {"contract": "contract.json", "catalog": "catalog.json",
                           "normalization": "normalization.json", "calibration": "policy.json"}.items():
        expected[name] = runtime.recipe["files"][filename]
    if protocol is not None:
        expected["protocol"] = sha(protocol)
    if any(files.get(k, {}).get("sha256") != v for k, v in expected.items()):
        raise ValueError("Selection lock does not bind this runtime/data/protocol")
    if "fixture" in runtime.recipe.get("checkpoint_status", ""):
        raise ValueError("Software fixture cannot authorize fresh acoustic evaluation")


def _metrics(pairs):
    m = dict(trials=len(pairs), supported_n=0, negative_n=0, ambiguous_n=0,
             accepted_n=0, correct=0, false_accepts=0, correct_rejections=0,
             faults=0, missing=0, slot_counts={})
    for trial, prediction in pairs:
        ambiguous = trial.get("reference_status") == "ambiguous"
        expected = trial.get("command")
        if ambiguous:
            m["ambiguous_n"] += 1
        elif expected is None:
            m["negative_n"] += 1
        else:
            m["supported_n"] += 1
        if not ambiguous:
            for field in (expected or {}).get("slots", {}):
                c = m["slot_counts"].setdefault(field, {"n": 0, "correct_accepted": 0})
                c["n"] += 1
        if prediction is None:
            m["missing"] += 1
            continue
        if prediction.get("fault"):
            m["faults"] += 1
            continue
        app = prediction.get("final_application", prediction.get("application", {}))
        accepted = app.get("accepted") is True and app.get("command") is not None
        actual = app.get("command") if accepted else None
        if ambiguous:
            continue
        m["accepted_n"] += int(accepted)
        exact = accepted and canonical(actual) == canonical(expected)
        m["correct"] += int(exact)
        m["false_accepts"] += int(expected is None and accepted)
        m["correct_rejections"] += int(expected is None and not accepted)
        for field, value in (expected or {}).get("slots", {}).items():
            c = m["slot_counts"].setdefault(field, {"n": 0, "correct_accepted": 0})
            c["correct_accepted"] += int(accepted and actual.get("slots", {}).get(field) == value)
    m["recall"] = m["correct"]/m["supported_n"] if m["supported_n"] else None
    m["precision"] = m["correct"]/m["accepted_n"] if m["accepted_n"] else None
    m["false_accept_rate"] = m["false_accepts"]/m["negative_n"] if m["negative_n"] else None
    return m


def _pairs(trials, predictions):
    identifiers = [t["id"] for t in trials]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate trial ID")
    index = {}
    for p in predictions:
        if p["id"] in index:
            raise ValueError("Duplicate trial prediction")
        index[p["id"]] = p
    if set(index) - set(identifiers):
        raise ValueError("Unexpected trial predictions")
    return [(t, index.get(t["id"])) for t in trials]


@lru_cache(maxsize=1)
def _optionb_outputs():
    from scripts.runtime.pcv_schema import baseline_cases
    return {canonical(c["command"]) for c in baseline_cases() if c["command"] is not None}


def trial_tier(trial):
    if trial.get("tier"):
        return trial["tier"]
    if trial.get("reference_status") == "ambiguous":
        return "ambiguous"
    command = trial.get("command")
    if command is None:
        return "negative"
    return "optionb" if canonical(command) in _optionb_outputs() else "extension"


def _host_timing(pairs):
    milliseconds, rtf = [], []
    for trial, prediction in pairs:
        if not prediction or prediction.get("fault"):
            continue
        value = prediction.get("timing", {}).get("total_ms")
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            continue
        milliseconds.append(value)
        duration = trial.get("decoded_duration_seconds")
        if duration is None and trial.get("sample_rate"):
            duration = trial.get("samples", 0)/trial["sample_rate"]
        if type(duration) in (float, int) and math.isfinite(duration) and duration > 0:
            rtf.append(value/(duration*1000))
    def quantile(values, q):
        return float(np.quantile(values, q)) if values else None
    return {"measured_n": len(milliseconds), "rtf_n": len(rtf),
            "p50_ms": quantile(milliseconds, .5), "p95_ms": quantile(milliseconds, .95),
            "p50_rtf": quantile(rtf, .5), "p95_rtf": quantile(rtf, .95),
            "physical_pi_evidence": False}


def summarize_trials(trials, predictions):
    pairs = _pairs(trials, predictions)
    tracks, people, intents, kinds = (defaultdict(list) for _ in range(4))
    for pair in pairs:
        t = pair[0]
        tracks[t["kind"]+"/"+trial_tier(t)].append(pair)
        if known_identity(t.get("speaker_id")):
            people[(t["kind"], t["speaker_id"])].append(pair)
        kinds[t["kind"]].append(pair)
        intents[(t.get("command") or {}).get("intent", "OOV")].append(pair)
    person_metrics = {k: _metrics(v) for k, v in people.items()}
    summaries = {}
    for kind, group in kinds.items():
        metrics = [v for (k, _), v in person_metrics.items() if k == kind]
        recalls = [v["recall"] for v in metrics if v["recall"] is not None]
        summaries[kind] = {
            "declared_ids": len(metrics), "supported_ids": len(recalls),
            "unidentified_trials": sum(not known_identity(t.get("speaker_id")) for t, _ in group),
            "macro_recall": float(np.mean(recalls)) if recalls else None,
            "worst_recall": min(recalls) if recalls else None,
            "identity_scope": "declared human IDs" if kind == "human" else "synthetic voice IDs; not human people"}
    human = summaries.get("human", {})
    return {"all": _metrics(pairs), "tracks": {k: _metrics(v) for k, v in tracks.items()},
            "people": {kind+"/"+person: v for (kind, person), v in person_metrics.items()},
            "kinds": {k: _metrics(v) for k, v in kinds.items()}, "person_summary": summaries,
            "intents": {k: _metrics(v) for k, v in intents.items()},
            "macro_person_recall": human.get("macro_recall"),
            "worst_person_recall": human.get("worst_recall"),
            "macro_person_population": "human declared IDs only",
            "tier_definition": "Supplied tier or typed-output membership in OptionB; not 93-phrase acoustic coverage.",
            "host_timing": _host_timing(pairs),
            "uncertainty": "People are the resampling unit; repeated sessions/pairs are dependent.",
            "evidence": "actual replay only if predictions came from a bound runtime"}


def cluster_interval(trials, predictions, seed=53, repetitions=1000, *, kind="human"):
    groups = defaultdict(list)
    for trial, prediction in _pairs(trials, predictions):
        if trial["kind"] == kind and known_identity(trial.get("speaker_id")):
            groups[trial["speaker_id"]].append((trial, prediction))
    if not groups:
        return None
    # Aggregate once; repeated recordings retain their within-person dependence.
    counts = []
    for person in sorted(groups):
        metric = _metrics(groups[person])
        counts.append([metric["correct"], metric["supported_n"]])
    counts = np.asarray(counts, dtype=np.int64)
    values = []
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        correct, supported = counts[rng.choice(len(counts), len(counts))].sum(axis=0)
        if supported:
            values.append(correct/supported)
    return np.quantile(values, [.025, .975]).tolist() if values else None


def replay(rows, runtime, audio_root, output, *, split, selection=None, regression=False,
           manifest=None, protocol=None):
    from scripts.test.speaker_demo_suite import read_audio
    if any(row["split"] != split for row in rows):
        raise ValueError("Replay partition mismatch")
    _pairs(rows, [])
    if split in ("test", "holdout") and not regression:
        if selection is None:
            raise ValueError("Fresh evaluation requires selection-before-test")
        lock = verify_selection(selection)
        if manifest is None:
            raise ValueError("Missing fresh evaluation manifest binding")
        verify_evaluation_context(lock, runtime, manifest, protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions = []
    with (output/"predictions.jsonl").open("x") as stream:
        for row in rows:
            try:
                event = runtime.recognize(read_audio(row, Path(audio_root)), 16000)
                p = {"id": row["id"], "final_application": event["final_application"],
                     "benchmark": event["benchmark"], "timing": event["timing"]}
                if event["final_application"].get("reason") == "capture_fault":
                    p["fault"] = "capture_fault"
            except Exception as error:
                p = {"id": row["id"], "fault": type(error).__name__+": "+str(error)}
            predictions.append(p)
            stream.write(json.dumps(p, allow_nan=False)+"\n")
            stream.flush()
    metrics = summarize_trials(rows, predictions)
    metrics["people_cluster_recall_interval"] = cluster_interval(rows, predictions)
    metrics["evaluation_role"] = "historical_regression" if regression else split
    metrics["runtime"] = {"engine": runtime.engine,
                          "checkpoint_status": runtime.recipe.get("checkpoint_status"),
                          "checkpoint_sha256": runtime.recipe["checkpoint_source_sha256"],
                          "calibration_id": runtime.policy.get("calibration_id"),
                          "actions_enabled": False}
    metrics["qualification"] = False
    write_new(output/"metrics.json", metrics)
    return metrics
