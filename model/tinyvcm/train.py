"""Reproducible, independent training jobs with validation-only model selection."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, reduce

from .audio import DEFAULT, cached_features
from .data import ROOT, read_jsonl, write_json, normalize_text

_FEATURE_MEMORY = {}


def training_features(record):
    value = _FEATURE_MEMORY.get(record["id"])
    return value if value is not None else cached_features(record, ROOT)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def warm_cache(rows, workers=6, memory_mib=0):
    started = last = time.monotonic()
    memory_bytes = sum(array.nbytes for array in _FEATURE_MEMORY.values()) if memory_mib else 0
    def cache(record):
        array = training_features(record)
        return array if memory_mib else array.shape[1]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, value in enumerate(pool.map(cache, rows), 1):
            frames = value.shape[1] if memory_mib else value
            rows[i - 1]["frames"] = frames
            if memory_mib and rows[i - 1]["id"] not in _FEATURE_MEMORY:
                memory_bytes += value.nbytes
                if memory_bytes > memory_mib * 2**20:
                    raise RuntimeError(f"Feature cache exceeds the requested {memory_mib} MiB budget")
                value.flags.writeable = False
                _FEATURE_MEMORY[rows[i - 1]["id"]] = value
            if time.monotonic() - last > 20:
                print(f"FEATURES {i}/{len(rows)} elapsed={time.monotonic()-started:.0f}s", flush=True)
                last = time.monotonic()
    if memory_mib:
        print(f"RAM FEATURE CACHE {len(_FEATURE_MEMORY)} arrays, {memory_bytes/2**20:.1f} MiB", flush=True)


def normalization(rows):
    total = np.zeros(40, dtype=np.float64)
    squares = np.zeros(40, dtype=np.float64)
    count = 0
    for row in rows:
        x = training_features(row).astype(np.float64)
        total += reduce(x, "m t -> m", "sum")
        squares += reduce(x * x, "m t -> m", "sum")
        count += x.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-4))
    return {"mean": mean.tolist(), "std": std.tolist(), "training_frames": count}


def batches(rows, batch_size, rng=None, frame_budget=16000):
    indices = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(indices)
    output = []
    for offset in range(0, len(indices), batch_size * 32):
        bucket = sorted(indices[offset:offset + batch_size * 32], key=lambda i: rows[i].get("frames", rows[i]["duration"] * 100))
        current, maximum = [], 0
        for index in bucket:
            length = rows[index].get("frames", math.ceil(rows[index]["duration"] * 100))
            if current and (len(current) >= batch_size or max(maximum, length) * (len(current) + 1) > frame_budget):
                output.append(current)
                current, maximum = [], 0
            current.append(index)
            maximum = max(maximum, length)
        if current:
            output.append(current)
    if rng is not None:
        rng.shuffle(output)
    return output


def make_batch(rows, indices, stats, augment=False, rng=None):
    arrays = [training_features(rows[i]) for i in indices]
    lengths = torch.tensor([x.shape[1] for x in arrays], dtype=torch.long)
    maximum = int(math.ceil(int(lengths.max()) / 32) * 32)
    batch = np.zeros((len(arrays), 40, maximum), dtype=np.float32)
    mean = rearrange(np.array(stats["mean"], dtype=np.float32), "m -> m 1")
    std = rearrange(np.array(stats["std"], dtype=np.float32), "m -> m 1")
    for j, x in enumerate(arrays):
        x = (x - mean) / std
        if augment:
            x = x.copy()
            # Feature-space augmentation is explicitly reported; no synthetic room claim.
            x += rng.uniform(-0.15, 0.15)
            width = rng.randrange(0, 6)
            begin = rng.randrange(0, 40 - width + 1)
            x[begin:begin + width, :] = 0
            width = rng.randrange(0, min(25, x.shape[1] // 10) + 1)
            begin = rng.randrange(0, x.shape[1] - width + 1)
            x[:, begin:begin + width] = 0
        batch[j, :, :x.shape[1]] = x
    return torch.from_numpy(batch), lengths


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


@torch.inference_mode()
def validate_vcm(model, rows, stats, labels, device, batch_size=32):
    model.eval()
    lookup = {c: i for i, c in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    loss_sum = 0.0
    outputs = []
    for indices in batches(rows, batch_size):
        x, lengths = make_batch(rows, indices, stats)
        target = torch.tensor([lookup[rows[i]["command"]] for i in indices], device=device)
        logits = model(x.to(device), lengths.to(device))
        loss_sum += F.cross_entropy(logits, target, reduction="sum").item()
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        for row_index, probability in zip(indices, probs):
            predicted = int(probability.argmax())
            gold = lookup[rows[row_index]["command"]]
            confusion[gold, predicted] += 1
            outputs.append({"id": rows[row_index]["id"], "gold": labels[gold], "predicted": labels[predicted], "probabilities": probability.tolist()})
    tp = np.diag(confusion)
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    present = confusion.sum(1) > 0
    return {"loss": loss_sum / len(rows), "accuracy": float(tp.sum() / confusion.sum()), "macro_f1": float(f1[present].mean()), "confusion": confusion.tolist(), "samples": len(rows)}, outputs


@torch.inference_mode()
def validate_asr(model, rows, stats, device, batch_size=16):
    from .asr import decode_greedy
    model.eval()
    words = word_errors = chars = char_errors = exact = 0
    outputs = []
    for indices in batches(rows, batch_size):
        x, lengths = make_batch(rows, indices, stats)
        logits = model(x.to(device), lengths.to(device))
        sequence = logits.argmax(-1).cpu()
        output_lengths = model.output_lengths(lengths)
        for j, index in enumerate(indices):
            hypothesis = decode_greedy(sequence[:int(output_lengths[j]), j])
            reference = normalize_text(rows[index]["transcript"])
            word_errors += edit_distance(reference.split(), hypothesis.split())
            words += len(reference.split())
            char_errors += edit_distance(reference, hypothesis)
            chars += len(reference)
            exact += reference == hypothesis
            outputs.append({"id": rows[index]["id"], "reference": reference, "hypothesis": hypothesis, "source": rows[index]["source"]})
    return {"wer": word_errors / max(1, words), "cer": char_errors / max(1, chars), "exact_match": exact / len(rows), "samples": len(rows), "reference_words": words}, outputs


def save_checkpoint(path, model, optimizer, stats, args, epoch, best, labels=None):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "model_config": model.config, "optimizer": optimizer.state_dict(), "normalization": stats, "args": vars(args), "epoch": epoch, "best": best, "labels": labels, "frontend": DEFAULT.__dict__, "timing_supervision": False}, temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["vcm", "asr"])
    parser.add_argument("--stage", choices=["all", "acoustic", "commands"], default="all")
    parser.add_argument("--run", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--pooling", choices=["mean", "mean_max"], default="mean")
    parser.add_argument("--kernels", type=int, nargs="+", default=[5, 7, 9, 11, 13])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--initialize")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--feature-cache-mib", type=int, default=0, help="Preload immutable features in RAM with this explicit budget; 0 uses disk")
    args = parser.parse_args(argv)
    _FEATURE_MEMORY.clear()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Expected GPU training; explicitly configure a CPU experiment instead of silently running for days")
    device = torch.device("cuda")
    run = ROOT / "runs" / args.run
    if (run / "last.pt").exists() and not args.resume:
        raise RuntimeError("Run already exists; use --resume or a new run name")
    run.mkdir(parents=True, exist_ok=True)
    all_rows = read_jsonl(ROOT / f"data/manifests/{args.task}.jsonl")
    if args.task == "asr" and args.stage == "acoustic":
        all_rows = [r for r in all_rows if r["source"] == "LibriSpeech"]
    if args.task == "asr" and args.stage == "commands":
        # A deterministic 10% read-speech mixture retains general acoustic support.
        from .data import stable_number
        all_rows = [r for r in all_rows if r["source"] == "SLURP-real" or r["split"] == "train" and stable_number(r["id"]) % 10 == 0]
    train_rows = [r for r in all_rows if r["split"] == "train"]
    dev_rows = [r for r in all_rows if r["split"] == "dev"]
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    if args.limit_dev:
        dev_rows = dev_rows[:args.limit_dev]
    if not train_rows or not dev_rows:
        raise RuntimeError("Empty training or development partition")
    print(f"PREPARE {args.task}/{args.stage}: train={len(train_rows)} dev={len(dev_rows)} GPU={torch.cuda.get_device_name(0)}", flush=True)
    warm_cache(train_rows + dev_rows, memory_mib=args.feature_cache_mib)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if args.resume else None
    initial = torch.load(args.initialize, map_location="cpu", weights_only=False) if args.initialize else None
    stats = (checkpoint or initial or {}).get("normalization") or normalization(train_rows)
    labels = None
    if args.task == "vcm":
        from .vcm import VCM
        labels = json.loads((ROOT / "data/manifests/labels.json").read_text())
        model = VCM(**checkpoint["model_config"]) if checkpoint else VCM(len(labels), width=args.width or 128, kernels=args.kernels, pooling=args.pooling)
        lookup = {label: i for i, label in enumerate(labels)}
        counts = Counter(r["command"] for r in train_rows)
        class_weights = torch.tensor([1 / math.sqrt(counts[c]) for c in labels], device=device)
        class_weights /= class_weights.mean()
    else:
        from .asr import ParameterASR, encode_text
        config = (checkpoint or initial or {}).get("model_config", {"width": args.width or 256})
        if args.stage == "commands" and not checkpoint:
            # Mask the padded batch positions during assistant adaptation so that
            # valid frames match single-recording deployment. Acoustic pretraining
            # retains its explicitly recorded unmasked baseline configuration.
            config = {**config, "mask_padding": True}
        model = ParameterASR(**config)
    if checkpoint or initial:
        model.load_state_dict((checkpoint or initial)["model"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -float("inf")
    first_epoch = 1
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        best = checkpoint["best"]
        first_epoch = checkpoint["epoch"] + 1
        if (run / "config.json").exists() and not (run / "config.initial.json").exists():
            (run / "config.initial.json").write_bytes((run / "config.json").read_bytes())
        with (run / "restarts.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"resumed_from_epoch": checkpoint["epoch"], "feature_cache_mib": args.feature_cache_mib, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rng_note": "Weights and optimizer restored; initialization seeds reset, so dropout RNG continuation is not bitwise identical."}) + "\n")
    parameters = sum(p.numel() for p in model.parameters())
    limit = 1_000_000 if args.task == "vcm" else 5_000_000
    if parameters > limit:
        raise RuntimeError(f"Parameter budget exceeded: {parameters} > {limit}")
    manifest = ROOT / f"data/manifests/{args.task}.jsonl"
    write_json(run / "config.json", {"args": vars(args), "model": model.config, "parameters": parameters, "train_samples": len(train_rows), "dev_samples": len(dev_rows), "train_hours": sum(r["duration"] for r in train_rows)/3600, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "training_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "normalization": stats, "timing_supervision": False, "augmentation": "gain offset and frequency/time feature masks" if not args.no_augment else "none"})
    print(f"TRAIN {parameters:,} parameters", flush=True)
    stale = 0
    for epoch in range(first_epoch, args.epochs + 1):
        started = last_log = time.monotonic()
        model.train()
        rng = random.Random(args.seed * 1000 + epoch)
        order = batches(train_rows, args.batch_size, rng)
        loss_sum = examples = 0
        optimizer.zero_grad(set_to_none=True)
        for step, indices in enumerate(order):
            progress = ((epoch - 1) + step / len(order)) / args.epochs
            scale = min(1, (progress + 0.001) / 0.05) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            x, lengths = make_batch(train_rows, indices, stats, not args.no_augment, rng)
            x = x.to(device)
            if args.task == "vcm":
                target = torch.tensor([lookup[train_rows[i]["command"]] for i in indices], device=device)
                logits = model(x, lengths.to(device))
                loss = F.cross_entropy(logits, target, weight=class_weights, label_smoothing=0.03)
            else:
                targets = [encode_text(train_rows[i]["transcript"]) for i in indices]
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
                flattened = torch.tensor([token for target in targets for token in target], dtype=torch.long)
                logits = model(x, lengths.to(device))
                loss = F.ctc_loss(F.log_softmax(logits.float(), -1), flattened, model.output_lengths(lengths), target_lengths, blank=0, zero_infinity=False)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch={epoch} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * len(indices)
            examples += len(indices)
            if time.monotonic() - last_log > 30:
                print(f"STEP epoch={epoch} batch={step+1}/{len(order)} loss={loss_sum/examples:.4f} elapsed={time.monotonic()-started:.0f}s", flush=True)
                last_log = time.monotonic()
        if args.task == "vcm":
            metrics, _ = validate_vcm(model, dev_rows, stats, labels, device)
            score = metrics["macro_f1"]
        else:
            metrics, predictions = validate_asr(model, dev_rows, stats, device)
            score = -metrics["wer"]
        result = {"epoch": epoch, "training_loss": loss_sum / examples, "validation": metrics, "seconds": time.monotonic()-started, "peak_gpu_mib": torch.cuda.max_memory_allocated()/2**20, "lr": optimizer.param_groups[0]["lr"]}
        with (run / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        improved = score > best + 1e-4
        if improved:
            best, stale = score, 0
            save_checkpoint(run / "best.pt", model, optimizer, stats, args, epoch, best, labels)
        else:
            stale += 1
        save_checkpoint(run / "last.pt", model, optimizer, stats, args, epoch, best, labels)
        write_json(run / "progress.json", {**result, "best_score": best, "best_updated": improved, "status": "training"})
        print("EPOCH " + json.dumps({k: v for k, v in result.items() if k != "validation"} | {"validation": {k: v for k, v in metrics.items() if k != "confusion"}}), flush=True)
        if stale >= args.patience:
            print(f"EARLY_STOP patience={args.patience}", flush=True)
            break
    write_json(run / "complete.json", {"status": "complete", "epochs": epoch, "best_validation_score": best, "checkpoint": "best.pt", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


if __name__ == "__main__":
    main()
