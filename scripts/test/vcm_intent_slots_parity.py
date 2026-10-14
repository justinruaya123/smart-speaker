"""Full-decision parity on unpadded features and immutable real PCM."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime
from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors
from scripts.runtime.pcv_schema import canonical
from scripts.test.speaker_demo_suite import read_audio
from scripts.train.vcm_intent_slots_data import write_new
from vcm.frontend import features


def verify_parity(bundle, output, *, recordings=(), audio_root=".", engines=("onnxruntime", "ncnn")):
    reference = IntentSlotsRuntime(bundle, "pytorch", 1)
    runtimes = {engine: IntentSlotsRuntime(bundle, engine, 1) for engine in engines}
    cases = [("numerical/"+str(n), np.random.default_rng(n).normal(size=(40, n)).astype(np.float32))
             for n in (1, 2, 3, 17, 32, 101, 231, 511, 798)]
    pcm_cases = []
    for row in recordings:
        pcm = read_audio(row, Path(audio_root))
        cases.append((row["id"], features(pcm, reference.mean, reference.std)))
        pcm_cases.append((row, pcm))
    observations = []
    started = time.perf_counter()

    def fields(out):
        return [out["intent_logits"], out["role_state_logits"], out["request_validity_logits"],
                *out["value_logits"].values(), out["entity_embeddings"]]

    for name, x in cases:
        a = reference.infer_features(x)
        expected = decide(prediction_from_tensors(a, reference.contract, reference.memory, reference.policy),
                          reference.contract, reference.catalog, reference.policy)
        for engine, runtime in runtimes.items():
            b = runtime.infer_features(x)
            actual = decide(prediction_from_tensors(b, runtime.contract, runtime.memory, runtime.policy),
                            runtime.contract, runtime.catalog, runtime.policy)
            error = max(float(np.max(abs(i-j))) for i, j in zip(fields(a), fields(b)))
            repeated = runtime.infer_features(x)
            stable = all(np.array_equal(i, j) for i, j in zip(fields(b), fields(repeated)))
            same = (actual["accepted"], canonical(actual["command"]), actual["reason"]) == (
                expected["accepted"], canonical(expected["command"]), expected["reason"])
            observations.append({"id": name, "frames": x.shape[1], "engine": engine,
                                 "max_error": error, "typed_decision_agreement": same,
                                 "repeat_stable": stable})
    # Controlled logits straddling every preserved application threshold.
    boundary_cases = []
    c = reference.contract
    for value in (.9499, .9501):
        t = reference.infer_features(np.zeros((40, 3), np.float32))
        t["intent_logits"] = np.zeros(19, np.float32)
        idx = c["intent_order"].index("STOP")
        t["intent_logits"][idx] = np.log(18*value/(1-value))
        t["role_state_logits"] = np.tile([12., 0., 0., 0.], (len(c["roles"]), 1)).astype("float32")
        t["request_validity_logits"] = np.asarray([12., 0., 0., 0.], np.float32)
        actual = decide(prediction_from_tensors(t, c), c, c, {})
        boundary_cases.append({"kind": "intent", "input_score": value,
                               "accepted": actual["accepted"], "expected": value >= .95})
    for value in (.8999, .9001):
        t["intent_logits"][:] = 0
        t["intent_logits"][c["intent_order"].index("VOLUME")] = 12
        t["role_state_logits"][c["roles"].index("volume_operation")] = [0, 12, 0, 0]
        t["value_logits"]["volume_operation"] = np.asarray([np.log(3*value/(1-value)), 0, 0, 0], np.float32)
        actual = decide(prediction_from_tensors(t, c), c, c, {})
        boundary_cases.append({"kind": "slot", "input_score": value,
                               "accepted": actual["accepted"], "expected": value >= .90})
    final_pcm = []
    for row, pcm in pcm_cases:
        expected = reference.recognize(pcm)
        for engine, runtime in runtimes.items():
            actual = runtime.recognize(pcm)
            same = canonical(actual["final_application"]["command"]) == canonical(expected["final_application"]["command"])
            final_pcm.append({"id": row["id"], "sha256": row["sha256"],
                              "pcm_sha256": row["pcm_sha256"], "engine": engine,
                              "accepted_equal": actual["final_application"]["accepted"] == expected["final_application"]["accepted"],
                              "typed_command_equal": same, "timing": actual["timing"]})
    passed = all(r["max_error"] <= 2e-5 and r["typed_decision_agreement"] and r["repeat_stable"]
                 for r in observations) and all(r["accepted"] == r["expected"] for r in boundary_cases)
    passed = passed and all(r["accepted_equal"] and r["typed_command_equal"] for r in final_pcm)
    result = {"passed": passed, "engines": list(engines), "observations": observations,
              "numerical_threshold_cases": boundary_cases, "real_pcm_pipeline": final_pcm,
              "elapsed_seconds": time.perf_counter()-started,
              "checkpoint_status": reference.recipe["checkpoint_status"],
              "evidence": "Export/software parity. Untrained weights cannot establish acoustic quality; controlled logits are numerical fixtures.",
              "physical_pi": False}
    write_new(output, result)
    return result
