"""Reproducible cached comparison; deployment contract and current bundle stay fixed."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import torch

from .candidate import fit_candidate, head_metrics
from .data import read_jsonl
from .supervision import prepare_references
from .metrics import metrics, policy_rows
from .finite_policy import provisional_slots, command_metrics
from .optimization import fit_intent_policy
from .inference import Recognizer
from .audio import load_audio


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def outputs(bundle, rows, root):
    recognizer = Recognizer(bundle, threads=2)
    records, slots = [], []
    for i, row in enumerate(rows, 1):
        pcm, _ = load_audio(root / row["audio"])
        p, heads, _ = recognizer.classify_with_slots(pcm, temperature=1.)
        records.append({"id": row["id"], "gold": row["command"], "probabilities": p.tolist()})
        slots.append({name: value.tolist() for name, value in heads.items()})
        if i % 500 == 0:
            print(f"DEPLOYMENT INFERENCE {i}/{len(rows)}", flush=True)
    return records, slots


def score(rows, annotations, original, records, slots, metadata):
    accepted = policy_rows(records, metadata["vcm"]["labels"], metadata["routing"]["variants"]["fp32"])
    commands, _ = command_metrics(rows, annotations, accepted, slots, metadata["acoustic_slots"], metadata["catalog"])
    old_commands, _ = command_metrics(rows, [original[r["id"]] for r in rows], accepted, slots,
                                     metadata["acoustic_slots"], metadata["catalog"])
    vocab = metadata["acoustic_slots"]["vocabulary"]
    tensors = (torch.tensor([r["probabilities"] for r in records]),
               *[torch.tensor([values[name] for values in slots]) for name in vocab])
    return {"intent": metrics(accepted, metadata["vcm"]["labels"]), "commands": commands,
            "original_reference_commands": old_commands, "heads": head_metrics(tensors, annotations, vocab)}


def development_check(bundle, root, *, refit=False, mode="frozen"):
    metadata = json.loads((bundle / "bundle.json").read_text())
    original = {r["id"]: r for r in read_jsonl(root / "data/references.jsonl")}
    rows = [r for r in read_jsonl(root / "data/vcm.jsonl") if r["split"] == "dev"]
    folds = json.loads((root / "data/development-folds.json").read_text())
    if not refit:
        rows = [r for r in rows if folds[r["id"]] == "selection"]
    annotations, _ = prepare_references(rows, original, metadata["catalog"])
    records, slots = outputs(bundle, rows, root)
    fit_ids = [i for i, row in enumerate(rows) if folds[row["id"]] == "policy_fit"]
    selection_ids = [i for i, row in enumerate(rows) if folds[row["id"]] == "selection"]
    if refit:
        policy = metadata["routing"]["variants"]["fp32"]
        fit_records = [records[i] for i in fit_ids]
        if mode == "finetune":
            policy = fit_intent_policy(fit_records, metadata["vcm"]["labels"], policy.get("temperature", 1.))
            metadata["routing"]["variants"]["fp32"] = policy
        accepted = policy_rows(fit_records, metadata["vcm"]["labels"], policy)
        policy_slots = provisional_slots([rows[i] for i in fit_ids], [annotations[i] for i in fit_ids], accepted,
                                          [slots[i] for i in fit_ids], metadata["acoustic_slots"], metadata["catalog"])
        policy_slots["inference_path"] = "single-recording FP32 ONNX CPU"
        metadata["acoustic_slots"]["variants"]["fp32"] = policy_slots
        ck = torch.load(bundle / "vcm.pt", map_location="cpu", weights_only=False)
        ck.update(intent_policy=policy, slot_policy=policy_slots)
        torch.save(ck, bundle / "vcm.pt")
        checkpoint_hash = hashlib.sha256((bundle / "vcm.pt").read_bytes()).hexdigest()
        metadata["vcm"]["checkpoint_sha256"] = checkpoint_hash
        metadata["experiment"]["checkpoint_sha256"] = checkpoint_hash
        save(bundle / "bundle.json", metadata)
        manifest = {"format_version": 1, "name": "cached-vcm-training-candidate", "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(bundle.iterdir()) if path.is_file() and path.name != "manifest.json"}}
        save(bundle / "manifest.json", manifest)
    return score([rows[i] for i in selection_ids], [annotations[i] for i in selection_ids], original,
                 [records[i] for i in selection_ids], [slots[i] for i in selection_ids], metadata)


def selection_rank(result):
    command = result["commands"]
    recalls = [command["per_family"][f]["correct_complete"] / max(1, command["per_family"][f]["resolvable"])
               for f in ("BRIGHTNESS", "VOLUME")]
    macro_raw = sum(v["raw_recall"] or 0 for v in result["intent"]["per_family"].values()) / 10
    return (command["accepted_command_precision"] >= .95, min(recalls), sum(recalls), macro_raw,
            command["exact_command_success"])


def compare(config, root, *, argument_heads=False):
    from .training import export_graphs
    from .validation import run_checks, evaluate
    root = Path(root)
    if config["training"]["max_train_examples"] or config["training"]["max_dev_examples"]:
        raise ValueError("A comparable improvement run requires the complete train/development partitions")
    parent = root / (".cache/argument-refinement" if argument_heads else ".cache/improvement-20260918")
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "output": str(output.relative_to(root)),
              "runtime_pipeline_changed": False, "architecture_changed": argument_heads, "current_artifacts_updated": False,
              "calibration_used": False, "selection_split": "grouped development selection only",
              "comparison_note": ("Matched frozen-backbone training: shared pooled heads versus added frame-attention residuals for target, operation and amount. Same labels, splits, sampling, losses, seed and acceptance criteria. Family logits remain fixed."
                                  if argument_heads else "Balanced frozen heads vs balanced encoder fine-tuning with feature augmentation; a training-strategy comparison, not an isolated architecture ablation."),
              "config": config, "candidates": {}}
    result["input_hashes"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
        "data/vcm.jsonl", "data/references.jsonl", "data/development-folds.json", "artifacts/vcm.pt", "artifacts/bundle.json")}
    save(output / "experiment.json", result)
    print(f"EXPERIMENT {output}", flush=True)
    result["baseline"] = development_check(root / "artifacts", root)
    save(output / "experiment.json", result)
    trials = [("shared", "frozen", False), ("attention", "frozen", True)] if argument_heads else [("frozen", "frozen", False), ("finetune", "finetune", False)]
    for name, mode, attention in trials:
        bundle = output / name
        bundle.mkdir()
        for path in (root / "artifacts").iterdir():
            if path.is_file() and path.name not in {"family-metrics.json", "pilot-results.json", "training-history.json", "export-verification.json"}:
                shutil.copy2(path, bundle / path.name)
        metadata, training = fit_candidate(config, root, bundle, mode, slot_attention=attention)
        errors = export_graphs(bundle, ["vcm"], metadata)
        save(bundle / "bundle.json", metadata)
        deployment = development_check(bundle, root, refit=True, mode=mode)
        checks = run_checks(config, bundle)
        result["candidates"][name] = {"training": training, "export_errors": errors,
                                      "development": deployment, "checks": checks, "bundle": str(bundle.relative_to(root))}
        save(output / "experiment.json", result)
    selected = max(result["candidates"], key=lambda mode: selection_rank(result["candidates"][mode]["development"]))
    result["selected"] = selected
    result["selection_locked_before_test"] = True
    save(output / "selection.json", {"selected": selected, "basis": "single-recording ONNX development selection",
                                     "ranking": {m: selection_rank(v["development"]) for m, v in result["candidates"].items()}})
    save(output / "experiment.json", result)
    # Test inference runs after the training strategy, epoch, and policy are locked.
    test_rows = [r for r in read_jsonl(root / "data/vcm.jsonl") if r["split"] == "test"]
    refs = {r["id"]: r for r in read_jsonl(root / "data/references.jsonl")}
    cfg = copy.deepcopy(config)
    cfg["tests"].update(vad=True, vcm=True, pipeline=True)
    bundle = output / selected
    test, predictions = evaluate(cfg, bundle, test_rows, root, refs)
    save(output / "test-original-references.json", test)
    with (output / "test-predictions.jsonl").open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction) + "\n")
    result["test"] = test
    result["promotion"] = "Not automatic: inspect per-family quality gates, label-reference limitations and an independent future holdout before replacing the current bundle."
    result["test_note"] = "The existing test was previously diagnosed; this is a regression evaluation, not a fresh unseen benchmark. No test results select this run's training, epoch or thresholds."
    save(output / "experiment.json", result)
    save(parent / "latest.json", {"output": result["output"], "selected": selected})
    return result
