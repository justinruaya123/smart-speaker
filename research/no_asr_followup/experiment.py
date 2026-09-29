"""Bounded own-checkpoint experiment; external weights are never fetched.

Option B supplies recordings, not an initialization. The only trainable tensors
are the original first convolution. All later acoustic layers remain frozen.
The development command guard is stricter than historical PER selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import gcd
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.signal import resample_poly
from scipy.special import log_softmax
import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
V3 = ROOT / "experiments/vcm_streaming_v3"
V6 = ROOT / "experiments/vcm_services_v6"
for folder in (ROOT, V3 / "src", V3 / "runtime"):
    sys.path.insert(0, str(folder))

from frontend import Frontend
from phone_model import PhoneTCN
from phonetic_catalog import PHONES, pronounce
from research.no_asr_followup.selection import assess_candidate
from stream_policy import CommandRanker
from train_phone import batch_arrays, collapse, edit_distance
from trie_ctc import TrieCTC


PARENT = V3 / "models/phone_s53.pt"
NORM = V3 / "runtime/normalization.npz"
GRAPH = V6 / "data/services_compiled.json"
SOURCE_REVISION = "ed245708c2573188eb769096aba76588f79b8de1"
SUBSET_SHA256 = (
    "ac4370d63c155e8d5a2217fef93b61077e17a18d3c1147567fd5119d44658b8e"
)
PARENT_SHA256 = (
    "2cb858cffd656fe87537f4af80d94c9502354d6f0bebbe6a70f716ea9c15d4aa"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, value: dict | list) -> None:
    folder = OUT / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(value, indent=2, allow_nan=False))


def own_rows(split: str) -> list[dict]:
    norm = np.load(NORM)
    manifest = V6 / "data/probe_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    result = []
    for row in rows:
        if row["split"] != split:
            continue
        array = np.load(V6 / row["features"])
        array = (array - norm["mean"][:, None]) / norm["std"][:, None]
        result.append({**row, "array": array})
    return result


def external_rows(folder: Path, split: str, gain: float = 1.0) -> list[dict]:
    from semantics import number_words

    norm = np.load(NORM)
    manifest_path = folder / "manifest.json"
    if digest(manifest_path) != SUBSET_SHA256:
        raise ValueError("External subset manifest has changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest["commit"] != SOURCE_REVISION:
        raise ValueError("Unreviewed external dataset revision")
    result = []
    for row in manifest["records"]:
        if row["split"] != split or row["intent"] != "BRIGHTNESS":
            continue
        path = folder / row["path"]
        if digest(path) != row["file_sha256"]:
            raise ValueError("External recording checksum changed")
        pcm, rate = sf.read(path, dtype="float32")
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Expected finite mono audio")
        factor = gcd(rate, 16000)
        pcm = resample_poly(pcm, 16000 // factor, rate // factor).astype(
            np.float32)
        pcm = np.r_[pcm, np.zeros(1600, np.float32)]
        array = Frontend().push(pcm * gain)
        array = (array - norm["mean"][:, None]) / norm["std"][:, None]
        text = re.sub(r"\b\d+\b", lambda m: number_words(int(m[0])),
                      row["transcript"].lower())
        labels = list(pronounce(text))
        required = len(labels) + sum(
            a == b for a, b in zip(labels, labels[1:]))
        if required > array.shape[1]:
            raise ValueError("CTC target is infeasible")
        result.append({
            "id": row["path"], "speaker": row["speaker"],
            "condition": row["variant_id"], "split": split,
            "text": row["transcript"], "category": "external_brightness",
            "target": {"intent": "SET_BRIGHTNESS", "entity_id": None,
                       "unit": "percent",
                       "value": int(row["label"].split("_")[1])},
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "labels": labels, "frames": array.shape[1], "array": array,
            "source_sample_rate": rate,
        })
    if not result:
        raise ValueError("Requested partition is absent")
    return result


def model_from(path: Path) -> PhoneTCN:
    if path.resolve() == PARENT.resolve() and digest(path) != PARENT_SHA256:
        raise ValueError("Expected the pinned project-trained checkpoint")
    model = PhoneTCN().eval()
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(loaded["weights"])
    return model


def evaluate(model: PhoneTCN, rows: list[dict], graph: dict) -> dict:
    scorer = TrieCTC([f["target"] for f in graph["forms"]], 40, "native")
    ranker = CommandRanker(graph)
    model.eval()
    records, errors, phones = [], 0, 0
    with torch.inference_mode():
        for start in range(0, len(rows), 24):
            batch = rows[start:start + 24]
            print(f"evaluate {start}/{len(rows)}", flush=True)
            x, _, lengths, _ = batch_arrays(batch)
            logits = model(x).numpy()
            for row, sequence, length in zip(batch, logits, lengths.tolist()):
                sequence = sequence[:, :length].T
                predicted = collapse(sequence.argmax(1).tolist())
                errors += edit_distance(row["labels"], predicted)
                phones += len(row["labels"])
                scorer.reset()
                probabilities = log_softmax(sequence, axis=1).astype(
                    np.float64)
                scores = scorer.push(probabilities)
                result = ranker.rank(scores)
                record = {key: row[key] for key in (
                    "id", "pcm_sha256", "split", "text", "category", "target")}
                record.update(
                    candidate=result["candidate"], ties=result["ties"],
                              accepted=False, action_executed=False)
                record["raw_correct"] = (row["target"] is not None
                    and row["target"] == result["candidate"]
                    and not result["ties"])
                records.append(record)
    return {"records": records, "phone_error_rate": errors / phones,
            "raw_correct": sum(r["raw_correct"] for r in records),
            "valid": sum(r["target"] is not None for r in records)}


def protocol() -> dict:
    return {
        "seed": 811, "epochs": 6, "batch_size": 24,
        "learning_rate": 0.0002, "weight_decay": 0.0001,
        "minimum_learning_rate": 0.00002, "gradient_clip": 3.0,
        "trainable": ["blocks.0.conv.weight", "blocks.0.conv.bias"],
        "initialization": "own_from_scratch_lineage_checkpoint_continuation",
        "external_pretrained_weights": False,
        "parent_sha256": digest(PARENT), "normalization_sha256": digest(NORM),
        "graph_sha256": digest(GRAPH), "external_revision": SOURCE_REVISION,
        "selection": ("external dev command gain; "
                      "zero lost correct own dev commands"),
        "test_used_for_selection": False, "automatic_acceptance": False,
    }


def train_epoch(folder: Path, epoch: int) -> None:
    if not 1 <= epoch <= 6:
        raise ValueError("This bounded protocol has six epochs")
    torch.set_num_threads(2)
    torch.manual_seed(811 + epoch)
    rng = np.random.default_rng(811 + epoch)
    path = OUT / ".cache/stem_trial.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = protocol()
    model = model_from(PARENT if epoch == 1 else path)
    for key, parameter in model.named_parameters():
        parameter.requires_grad_(key in config["trainable"])
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=6, eta_min=config["minimum_learning_rate"])
    if epoch > 1:
        state = torch.load(path, weights_only=True, map_location="cpu")
        if state["epoch"] != epoch - 1 or state["protocol"] != config:
            raise ValueError("Resume protocol/epoch mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    external = external_rows(folder, "train")
    replay = own_rows("train")
    # Balanced batches prevent the three-value dataset from dominating replay.
    rng.shuffle(external)
    rng.shuffle(replay)
    batches = []
    for start in range(0, len(external), 12):
        left = external[start:start + 12]
        right = [replay[(start + i) % len(replay)] for i in range(len(left))]
        batches.append(left + right)
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)
    model.train()
    losses = []
    started = time.perf_counter()
    for batch in batches:
        # No aggressive time/frequency masking of short numeric phonemes.
        x, labels, lengths, label_lengths = batch_arrays(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x).log_softmax(1).permute(2, 0, 1),
                         labels, lengths, label_lengths)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite CTC loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        losses.append(float(loss.detach()))
    scheduler.step()
    parent = torch.load(
        PARENT, weights_only=True, map_location="cpu")["weights"]
    assert all(torch.equal(parent[k], v) for k, v in model.state_dict().items()
               if k not in config["trainable"])
    torch.save({"weights": model.state_dict(), "phones": PHONES,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch, "protocol": config}, path)
    graph = json.loads(GRAPH.read_text())
    own_dev = evaluate(model, own_rows("dev"), graph)
    ext_dev = evaluate(model, external_rows(folder, "val"), graph)
    baseline_path = OUT / "reports/baseline_development.json"
    baseline = json.loads(baseline_path.read_text())
    guard = assess_candidate(
        baseline["own"]["records"], own_dev["records"],
        baseline["external"]["records"], ext_dev["records"])
    result = {"epoch": epoch, "mean_loss": float(np.mean(losses)),
              "train_seconds": time.perf_counter() - started,
              "own": own_dev, "external": ext_dev, "guard": guard}
    save(f"stem_epoch_{epoch}.json", result)
    if guard["eligible"]:
        chosen = OUT / ".cache/eligible_candidate.pt"
        prior = json.loads((OUT / "reports/selected.json").read_text())
        if ext_dev["raw_correct"] > prior["external_correct"]:
            torch.save({"weights": copy.deepcopy(model.state_dict()),
                        "phones": PHONES, "epoch": epoch,
                        "protocol": config}, chosen)
            save("selected.json", {"path": str(chosen.relative_to(ROOT)),
                 "epoch": epoch, "sha256": digest(chosen),
                 "external_correct": ext_dev["raw_correct"],
                 "protocol": config})
    print(json.dumps({"epoch": epoch, "own_correct": own_dev["raw_correct"],
        "external_correct": ext_dev["raw_correct"],
        "eligible": guard["eligible"],
        "lost_own": len(guard["replay"]["lost_ids"]),
        "own_per": own_dev["phone_error_rate"],
        "external_per": ext_dev["phone_error_rate"],
        "seconds": result["train_seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()
    if args.epoch:
        train_epoch(args.audio, args.epoch)
        return
    torch.set_num_threads(2)
    graph = json.loads(GRAPH.read_text())
    model = model_from(PARENT)
    own = evaluate(model, own_rows("dev"), graph)
    ext = evaluate(model, external_rows(args.audio, "val"), graph)
    save("protocol.json", protocol())
    save("baseline_development.json", {"own": own, "external": ext})
    save("selected.json", {"path": str(PARENT.relative_to(ROOT)), "epoch": 0,
         "sha256": digest(PARENT), "external_correct": ext["raw_correct"],
         "protocol": protocol()})
    print({"own": own["raw_correct"],
           "external": ext["raw_correct"]}, flush=True)


if __name__ == "__main__":
    main()
