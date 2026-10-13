"""Random-init human-only fixed-demo ablations; no installation or test
fitting.

Run as a module from the repository root. Reuses vcm.frontend.features
unchanged.
A completed training run is an unqualified candidate, not a shipping artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from scripts.models.speaker_demo import (
    DemoKeywordNet, class_speaker_weights, cross_speaker_contrastive,
)
from scripts.test.speaker_demo_suite import (
    audit, canonical,
    file_digest, read_audio, read_json, read_jsonl, write_new,
)


def optimization_step(model, batch, optimizer, *, weight: float) -> dict:
    """One finite-gradient update; logs whether SupCon had usable positives."""
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Contrastive weight must be finite and nonnegative")
    features, lengths, labels, speakers = batch
    optimizer.zero_grad(set_to_none=True)
    logits, embeddings = model(features, lengths)
    classification = nn.functional.cross_entropy(logits, labels)
    contrastive = cross_speaker_contrastive(embeddings, labels, speakers)
    loss = classification + weight * contrastive
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss; optimizer not updated")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    positives = (
        (labels[:, None] == labels[None, :])
        & (speakers[:, None] != speakers[None, :])
        & (labels[:, None] > 0)
    )
    return {
        "loss": float(loss.detach()),
        "cross_entropy": float(classification.detach()),
        "contrastive": float(contrastive.detach()),
        "paired_anchor_fraction": float(positives.any(dim=1).float().mean()),
    }


class FeatureRows(Dataset):
    def __init__(self, rows, mean, std):
        self.rows, self.mean, self.std = rows, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        values = np.load(row["feature_path"], allow_pickle=False)
        values = (values - self.mean[:, None]) / self.std[:, None]
        return torch.from_numpy(values), row["label"], row["speaker_index"]


def collate(batch):
    lengths = torch.tensor([sample[0].shape[1] for sample in batch])
    features = torch.zeros(len(batch), 40, int(lengths.max()))
    for index, (values, _, _) in enumerate(batch):
        features[index, :, :values.shape[1]] = values
    return (features, lengths, torch.tensor([r[1] for r in batch]),
            torch.tensor([r[2] for r in batch]))


def fit(args: argparse.Namespace) -> dict:
    """Fit only train; select on dev. Calibration/test audio never
    feeds model."""
    from vcm.frontend import SPEC, features

    if args.output.exists():
        raise FileExistsError(
            "Use a new run directory; do not overwrite evidence"
        )
    if min(args.epochs, args.steps_per_epoch, args.batch_size) < 1:
        raise ValueError("Positive training sizes required")
    if not np.isfinite(args.contrastive_weight) or args.contrastive_weight < 0:
        raise ValueError("Invalid contrastive weight")
    rows = read_jsonl(args.manifest)
    audit_result = audit(rows, check_audio=True)
    commands = read_json(args.catalog)["commands"]
    from scripts.runtime.pcv_schema import validate_command
    for command in commands:
        validate_command(command)
    class_keys = [canonical(None)] + [canonical(c) for c in commands]
    if len(class_keys) < 2 or len(class_keys) != len(set(class_keys)):
        raise ValueError("Catalog needs unique complete commands")
    label_ids = {key: index for index, key in enumerate(class_keys)}
    selected = [r for r in rows if r["kind"] == "human"
                and r["role"] == "command" and r["split"] in {"train", "dev"}]
    speakers = {name: index for index, name in enumerate(
        sorted({row["speaker_id"] for row in selected}))}
    prepared = []
    for row in selected:
        prepared.append({
            **row, "label": label_ids.get(canonical(row["command"]), 0),
            "speaker_index": speakers[row["speaker_id"]],
        })
    for split in ("train", "dev"):
        counts = Counter(r["label"] for r in prepared if r["split"] == split)
        missing = set(range(len(class_keys))) - set(counts)
        if missing:
            raise ValueError(
                f"Missing human {split} classes: {sorted(missing)}"
            )
    if args.contrastive_weight > 0:
        for label in range(1, len(class_keys)):
            distinct = {r["speaker_id"] for r in prepared
                        if r["split"] == "train" and r["label"] == label}
            if len(distinct) < 2:
                raise ValueError(
                    "Contrastive training needs two speakers per keyword"
                )
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    write_new(args.output / "protocol-before-training.json", {
        "arguments": config, "manifest_sha256": file_digest(args.manifest),
        "catalog_sha256": file_digest(args.catalog), "frontend": SPEC,
        "frontend_source_sha256": file_digest(Path("vcm/frontend.py")),
        "model_source_sha256": file_digest(
            Path("scripts/models/speaker_demo.py")
        ),
        "trainer_source_sha256": file_digest(Path(__file__)),
        "audit": audit_result,
        "scope": (
            'human-only fixed demo; other complete commands are UNKNOWN in '
            'this mode'
        ),
        "pretrained_weights": False, "test_used_for_selection": False,
    })
    cache = args.output / "features"
    cache.mkdir()
    total = np.zeros(40, dtype=np.float64)
    squared = total.copy()
    frame_count = 0
    for row in prepared:
        raw = features(read_audio(row))
        if raw.shape[1] < 1:
            raise ValueError(f"No frontend frames: {row['id']}")
        path = cache / f"{row['pcm_sha256']}.npy"
        with path.open("xb") as stream:
            np.save(stream, raw, allow_pickle=False)
        row["feature_path"] = str(path)
        row["feature_sha256"] = file_digest(path)
        if row["split"] == "train":
            values = raw.astype(np.float64)
            total += values.sum(axis=1)
            squared += (values ** 2).sum(axis=1)
            frame_count += values.shape[1]
    mean = (total / frame_count).astype(np.float32)
    variance = squared / frame_count - (total / frame_count) ** 2
    std = np.maximum(np.sqrt(np.maximum(variance, 0)), 1e-5).astype(np.float32)
    write_new(args.output / "normalization.json", {
        "mean": mean.tolist(), "std": std.tolist(), "frontend": SPEC,
        "fit_split": "train", "frames": frame_count,
    })
    write_new(args.output / "feature-manifest.jsonl", prepared, jsonl=True)
    train = [r for r in prepared if r["split"] == "train"]
    dev = [r for r in prepared if r["split"] == "dev"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model = DemoKeywordNet(len(class_keys), width=args.width).to(device)
    weights = (
        class_speaker_weights(train)
        if args.sampling == "balanced" else np.ones(len(train))
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, args.steps_per_epoch * args.batch_size, replacement=True,
        generator=generator,
    )
    loader = DataLoader(FeatureRows(train, mean, std), sampler=sampler,
                        batch_size=args.batch_size, collate_fn=collate)
    validation = DataLoader(FeatureRows(dev, mean, std), shuffle=False,
                            batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )
    best_key, selected_checkpoint = None, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        updates = [optimization_step(
            model, tuple(t.to(device) for t in batch), optimizer,
            weight=args.contrastive_weight,
        ) for batch in loader]
        model.eval()
        positives = correct = negatives = false_accepts = 0
        speaker_totals, speaker_correct = Counter(), Counter()
        with torch.inference_mode():
            for batch in validation:
                x, lengths, labels, speaker_ids = (
                    t.to(device) for t in batch
                )
                logits, _ = model(x, lengths)
                predictions = logits.argmax(dim=1)
                valid = labels > 0
                positives += int(valid.sum())
                correct += int(((predictions == labels) & valid).sum())
                negatives += int((~valid).sum())
                false_accepts += int(((predictions > 0) & ~valid).sum())
                for label, predicted, speaker in zip(
                    labels.cpu().tolist(), predictions.cpu().tolist(),
                    speaker_ids.cpu().tolist(),
                ):
                    if label > 0:
                        speaker_totals[speaker] += 1
                        speaker_correct[speaker] += int(label == predicted)
        speaker_recalls = [
            speaker_correct[speaker] / count
            for speaker, count in speaker_totals.items()
        ]
        macro_recall = float(np.mean(speaker_recalls))
        key = (macro_recall, -false_accepts / negatives, correct / positives)
        row = {
            "dev_macro_speaker_recall": macro_recall,
            "dev_worst_speaker_recall": min(speaker_recalls),
            "epoch": epoch, "dev_positive_n": positives,
            "dev_raw_correct": correct,
            "dev_negative_n": negatives,
            "dev_raw_negative_errors": false_accepts,
            **{
                name: float(
                    np.mean([item[name] for item in updates])
                ) for name in updates[0]
            },
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            selected_checkpoint = (
                args.output / f"selected-epoch-{epoch:03d}.pt"
            )
            checkpoint = {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "classes": len(class_keys),
                "width": args.width,
                "epoch": epoch,
            }
            torch.save(checkpoint, selected_checkpoint)
        print(canonical(row), flush=True)
    write_new(args.output / "history.json", history)
    receipt = {
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": file_digest(selected_checkpoint),
        "manifest_sha256": file_digest(args.manifest),
        "catalog_sha256": file_digest(args.catalog),
        "normalization_sha256": file_digest(
            args.output / "normalization.json"
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "status": "unqualified_candidate",
        "selection": "dev macro-speaker recall, negative-error tie break",
        "calibration_evaluated": False, "test_evaluated": False,
        "onnx_exported": False, "installed": False,
    }
    write_new(args.output / "selection-before-calibration.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--sampling", choices=("uniform", "balanced"), default="balanced"
    )
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    args = parser.parse_args()
    if (
        not args.output.resolve().is_relative_to(
            (Path.cwd() / ".cache/speaker-demo-v1").resolve()
        )
    ):
        parser.error("Write outputs under .cache/speaker-demo-v1")
    fit(args)


if __name__ == "__main__":
    main()
