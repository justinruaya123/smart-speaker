"""Fresh ONNX Runtime replay of the frozen selected research checkpoint.

No training, threshold changes, device actions, or external weights are used.
Timings concern a desktop CPU, not a Raspberry Pi.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.special import log_softmax
import torch

from experiment import (
    GRAPH, NORM, OUT, PARENT, ROOT, V3, digest, external_rows, model_from,
    own_rows, save,
)
from run_stream import Engine
from stream_policy import CommandRanker
from trie_ctc import TrieCTC


def verify(audio: Path) -> None:
    selected_file = OUT / "reports/selected.json"
    selected_hash = digest(selected_file)
    selected = json.loads(selected_file.read_text())
    if selected["epoch"] != 0 or selected["sha256"] != digest(PARENT):
        raise ValueError("Export a promoted candidate before replaying it")
    torch.set_num_threads(1)
    model = model_from(PARENT)
    graph = json.loads(GRAPH.read_text())
    scorer = TrieCTC([f["target"] for f in graph["forms"]], 40, "native")
    ranker = CommandRanker(graph)
    engines = [Engine(V3 / f"models/phone_s53_{ms}ms.onnx", "onnxruntime")
               for ms in (40, 80)]
    records = []
    maximum = {4: 0.0, 8: 0.0}
    frame_changes, command_changes = Counter(), Counter()
    costs = defaultdict(list)
    # These datasets are regression/evaluation only, never passed to selection.
    partitions = {"own_v6_test": own_rows("test"),
                  "external_brightness_test": external_rows(audio, "test")}
    started = time.perf_counter()
    with torch.inference_mode():
        for domain, rows in partitions.items():
            for index, row in enumerate(rows):
                if index % 100 == 0:
                    print(domain, index, "/", len(rows), flush=True)
                x = row["array"]
                expected = log_softmax(
                    model(torch.from_numpy(x)[None])[0].T.numpy(), axis=1)
                scorer.reset()
                reference = ranker.rank(scorer.push(expected))
                for engine in engines:
                    for cache in engine.states:
                        cache.fill(0)
                    outputs = []
                    for start in range(0, x.shape[1], engine.frames):
                        block = x[:, start:start + engine.frames]
                        valid = block.shape[1]
                        padded = np.pad(block, ((0, 0),
                            (0, engine.frames - valid)))[None]
                        clock = time.perf_counter()
                        logits = engine.step(padded)
                        costs[engine.frames].append(
                            (time.perf_counter() - clock) * 1000)
                        outputs.append(logits[:valid])
                    actual = log_softmax(np.concatenate(outputs), axis=1)
                    error = float(np.max(np.abs(actual - expected)))
                    maximum[engine.frames] = max(maximum[engine.frames], error)
                    np.testing.assert_allclose(actual, expected,
                                               atol=3e-4, rtol=3e-4)
                    frame_changes[engine.frames] += int(not np.array_equal(
                        actual.argmax(1), expected.argmax(1)))
                    scorer.reset()
                    compared = ranker.rank(scorer.push(actual))
                    command_changes[engine.frames] += int(
                        compared["candidate"] != reference["candidate"]
                        or compared["ties"] != reference["ties"])
                correct = (row["target"] is not None
                           and row["target"] == reference["candidate"]
                           and not reference["ties"])
                records.append({
                    "domain": domain, "id": row["id"], "split": "test",
                    "pcm_sha256": row["pcm_sha256"], "text": row["text"],
                    "category": row["category"], "target": row["target"],
                    "candidate": reference["candidate"],
                    "ties": reference["ties"], "raw_correct": correct,
                    "accepted": False, "action_executed": False,
                })
    assert digest(selected_file) == selected_hash
    summaries = {}
    for domain in partitions:
        group = [r for r in records if r["domain"] == domain]
        categories = defaultdict(Counter)
        for row in group:
            categories[row["category"]]["recordings"] += 1
            categories[row["category"]]["raw_correct"] += row["raw_correct"]
        summaries[domain] = {
            "recordings": len(group),
            "valid": sum(r["target"] is not None for r in group),
            "raw_correct": sum(r["raw_correct"] for r in group),
            "categories": {k: dict(v) for k, v in categories.items()},
        }
    previous = json.loads((ROOT / "experiments/vcm_services_v6/reports/"
                           "probe_predictions.json").read_text())
    current = {r["id"]: r for r in records if r["domain"] == "own_v6_test"}
    differences = []
    for row in previous:
        new = current[row["id"]]
        assert row["pcm_sha256"] == new["pcm_sha256"]
        if row["candidate"] != new["candidate"] or row["ties"] != new["ties"]:
            differences.append(row["id"])
    import onnxruntime as ort
    save("replay_predictions.json", records)
    save("replay_summary.json", {
        "engine": f"ONNX Runtime {ort.__version__} CPUExecutionProvider",
        "host": platform.machine(), "python": platform.python_version(),
        "torch": torch.__version__, "model_sha256": digest(PARENT),
        "normalization_sha256": digest(NORM), "graph_sha256": digest(GRAPH),
        "selected_policy_sha256_before_and_after": selected_hash,
        "summaries": summaries, "max_absolute_logp_error": maximum,
        "changed_frame_paths": dict(frame_changes),
        "changed_commands_or_ties": dict(command_changes),
        "changed_vs_saved_v6": differences,
        "engine_step_ms": {str(frames): {
            "calls": len(values), "median": float(np.median(values[10:])),
            "p95": float(np.quantile(values[10:], .95)),
            "scope": ("Neural call and cache transfer only; "
                      "exclude frontend/decoder.")
        } for frames, values in costs.items()},
        "elapsed_seconds": time.perf_counter() - started,
        "physical_pi4_tested": False, "acceptance_enabled": False,
        "external_pretrained_weights": False,
        "root_v12_trained_weights_tested": False,
    })
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    verify(args.audio)
