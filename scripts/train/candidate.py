"""VCM training with repaired source labels and balanced finite-head supervision."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F

from .data import read_jsonl
from .train_helpers import warm_cache, batches, make_batch, seed_everything
from .slot_model import SlotVCM
from .supervision import prepare_references, coverage
from .optimization import balanced_draws, head_weights, balanced_losses, fit_intent_policy
from .finite_policy import provisional_slots, slot_records, command_metrics
from .metrics import policy_rows, metrics
from .slot_contract import catalog_signature


def head_metrics(outputs, annotations, vocabulary):
    result = {}
    for name, output in zip(vocabulary, outputs[1:]):
        predicted = output.argmax(-1).tolist()
        per_value = {}
        for i, annotation in enumerate(annotations):
            gold = annotation["labels"].get(name)
            if gold is None:
                continue
            counts = per_value.setdefault(gold, {"correct": 0, "support": 0})
            counts["support"] += 1
            counts["correct"] += vocabulary[name][predicted[i]] == gold
        result[name] = {"per_value": per_value,
                        "correct": sum(v["correct"] for v in per_value.values()),
                        "support": sum(v["support"] for v in per_value.values())}
    return result


def fit_candidate(config, root, output, mode="frozen", progress=None, *, slot_attention=False):
    if mode not in {"frozen", "finetune"}:
        raise ValueError("Encoder mode must be frozen or finetune")
    cfg = config["training"]
    seed_everything(cfg["seed"])
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ck = torch.load(root / "artifacts/vcm.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((root / "artifacts/bundle.json").read_text())
    schema = json.loads((root / "data/contract.json").read_text())
    original = {r["id"]: r for r in read_jsonl(root / "data/references.jsonl")}
    all_rows = read_jsonl(root / "data/vcm.jsonl")
    folds = json.loads((root / "data/development-folds.json").read_text())
    training = [r for r in all_rows if r["split"] == "train"]
    development = [r for r in all_rows if r["split"] == "dev"]
    if cfg["max_train_examples"]:
        training = training[:cfg["max_train_examples"]]
    if cfg["max_dev_examples"]:
        limit = max(1, cfg["max_dev_examples"] // 2)
        development = [r for fold in ("policy_fit", "selection")
                       for r in [x for x in development if folds[x["id"]] == fold][:limit]]
    rows = training + development
    annotations, repairs = prepare_references(rows, original, metadata["catalog"])
    (output / "label-repairs.json").write_text(json.dumps(repairs, indent=2) + "\n")
    (output / "coverage.json").write_text(json.dumps(coverage(rows, annotations), indent=2) + "\n")
    fit_ids = [i for i, r in enumerate(rows) if r["split"] == "dev" and folds[r["id"]] == "policy_fit"]
    select_ids = [i for i, r in enumerate(rows) if r["split"] == "dev" and folds[r["id"]] == "selection"]
    if not training or not fit_ids or not select_ids:
        raise ValueError("Need train and both grouped development folds")
    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose CPU explicitly")
    print(f"PREPARE mode={mode} train={len(training)} dev={len(development)} repairs={len(repairs)}", flush=True)
    warm_cache(rows, memory_mib=4096)
    model_config = {**ck["config"]}
    if slot_attention:
        model_config["slot_attention"] = True
    model = SlotVCM(**model_config).to(device).eval()
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    permitted = ("heads.attention.", "heads.temporal.") if slot_attention else ()
    if unexpected or any(not name.startswith(permitted) for name in missing):
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(mode == "finetune")
    vocab = ck["config"]["vocabulary"]
    targets = {name: torch.tensor([values.index(a["labels"][name]) if a["labels"][name] is not None else 0
                                  for a in annotations], device=device) for name, values in vocab.items()}
    masks = {name: torch.tensor([a["labels"][name] is not None for a in annotations], device=device) for name in vocab}
    families = torch.tensor([ck["labels"].index(row["command"]) for row in rows], device=device)
    weights = head_weights(annotations[:len(training)], vocab, device)
    vectors = family_logits = None
    frame_cache = [None] * len(rows) if model.slot_attention and mode == "frozen" else None
    if mode == "frozen":
        vectors = torch.empty(len(rows), model.heads.project[0].normalized_shape[0], device=device)
        family_logits = torch.empty(len(rows), len(ck["labels"]), device=device)
        with torch.inference_mode():
            for batch_no, ids in enumerate(batches(rows, cfg["batch_size"])):
                x, lengths = make_batch(rows, ids, ck["normalization"])
                encoded = model.encode(x.to(device), lengths.to(device))
                vectors[ids], family_logits[ids] = encoded[:2]
                if frame_cache is not None:
                    for j, index in enumerate(ids):
                        frame_cache[index] = encoded[2][j, :, :int(lengths[j])].cpu().clone()
                if batch_no % 100 == 0:
                    print(f"ENCODE {batch_no}", flush=True)
    groups = [{"params": [p for p in model.heads.parameters() if p.requires_grad],
               "lr": cfg.get("head_learning_rate", .001)}]
    if mode == "finetune":
        groups.append({"params": model.backbone.parameters(), "lr": cfg.get("encoder_learning_rate", .00003)})
    optimizer = torch.optim.AdamW(groups, weight_decay=.01)
    train_annotations = annotations[:len(training)]

    def frozen_values(ids):
        if frame_cache is None:
            return model.heads(vectors[ids], family_logits[ids])
        lengths = torch.tensor([frame_cache[i].shape[-1] for i in ids], device=device)
        frames = torch.zeros(len(ids), frame_cache[ids[0]].shape[0], int(lengths.max()), device=device)
        for j, index in enumerate(ids):
            frames[j, :, :frame_cache[index].shape[-1]] = frame_cache[index].to(device)
        return model.heads(vectors[ids], family_logits[ids], frames, lengths)

    def infer(ids):
        outputs = [torch.empty(len(ids), size) for size in [len(ck["labels"]), *map(len, vocab.values())]]
        positions = {index: pos for pos, index in enumerate(ids)}
        model.eval()
        with torch.inference_mode():
            if mode == "frozen":
                for start in range(0, len(ids), 256):
                    part = ids[start:start + 256]
                    values = frozen_values(part)
                    for target, value in zip(outputs, values):
                        target[start:start + len(part)] = value.cpu()
            else:
                for part in batches(rows, cfg["batch_size"], indices=ids):
                    x, lengths = make_batch(rows, part, ck["normalization"])
                    values = model(x.to(device), lengths.to(device))
                    for target, value in zip(outputs, values):
                        target[[positions[i] for i in part]] = value.cpu()
        return tuple(outputs)

    def family_records(ids, outputs):
        probabilities = torch.softmax(outputs[0], -1).tolist()
        return [{"id": rows[i]["id"], "gold": rows[i]["command"], "probabilities": p} for i, p in zip(ids, probabilities)]

    best, stale, history, best_state = None, 0, [], None
    started = time.perf_counter()
    for epoch in range(1, cfg["vcm_epochs"] + 1):
        model.train(mode == "finetune")
        model.heads.train()
        rng = random.Random(cfg["seed"] * 1000 + epoch)
        draws = balanced_draws(training, train_annotations, rng)
        losses_seen = []
        for batch_no, ids in enumerate(batches(rows, cfg["batch_size"], rng=rng, indices=draws)):
            if mode == "frozen":
                values = frozen_values(ids)
            else:
                x, lengths = make_batch(rows, ids, ck["normalization"], augment=True, rng=rng,
                                        speed_augment=True, gain_augment_db=6.)
                values = model(x.to(device), lengths.to(device))
            head_loss = balanced_losses(dict(zip(vocab, values[1:])),
                                       {k: v[ids] for k, v in targets.items()},
                                       {k: v[ids] for k, v in masks.items()}, weights)
            loss = sum(.5 * value for value in head_loss.values())
            if mode == "finetune":
                loss = loss + 2. * F.cross_entropy(values[0], families[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite VCM loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            losses_seen.append(float(loss.detach()))
            if batch_no % 80 == 0:
                print(f"TRAIN mode={mode} epoch={epoch} batch={batch_no} elapsed={time.perf_counter()-started:.0f}s", flush=True)
        fit_outputs, selection_outputs = infer(fit_ids), infer(select_ids)
        fit_records = family_records(fit_ids, fit_outputs)
        intent_policy = ck["intent_policy"] if mode == "frozen" else fit_intent_policy(
            fit_records, ck["labels"], ck["intent_policy"].get("temperature", 1.))
        fit_accepted = policy_rows(fit_records, ck["labels"], intent_policy)
        select_accepted = policy_rows(family_records(select_ids, selection_outputs), ck["labels"], intent_policy)
        fit_slots, selection_slots = slot_records(fit_outputs, vocab), slot_records(selection_outputs, vocab)
        policy = provisional_slots([rows[i] for i in fit_ids], [annotations[i] for i in fit_ids],
                                   fit_accepted, fit_slots, schema, metadata["catalog"])
        slot_config = {**schema, "catalog_signature": catalog_signature(metadata["catalog"]), "variants": {"fp32": policy}}
        selection_rows = [rows[i] for i in select_ids]
        selection_annotations = [annotations[i] for i in select_ids]
        commands, _ = command_metrics(selection_rows, selection_annotations, select_accepted, selection_slots, slot_config, metadata["catalog"])
        old_commands, _ = command_metrics(selection_rows, [original[r["id"]] for r in selection_rows],
                                         select_accepted, selection_slots, slot_config, metadata["catalog"])
        intent = metrics(select_accepted, ck["labels"])
        heads = head_metrics(selection_outputs, selection_annotations, vocab)
        focus = [commands["per_family"][family]["correct_complete"] / max(1, commands["per_family"][family]["resolvable"])
                 for family in ("BRIGHTNESS", "VOLUME")]
        macro_raw = sum(v["raw_recall"] or 0 for v in intent["per_family"].values()) / len(ck["labels"])
        rank = (commands["accepted_command_precision"] >= .95, min(focus), sum(focus), macro_raw,
                commands["exact_command_success"])
        entry = {"epoch": epoch, "loss": sum(losses_seen)/len(losses_seen), "commands": commands,
                 "original_reference_commands": old_commands, "intent": intent, "heads": heads,
                 "rank": rank, "elapsed_seconds": time.perf_counter()-started}
        history.append(entry)
        if best is None or rank > best:
            best, stale = rank, 0
            best_state = {**ck, "config": model.config,
                          "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "epoch": epoch, "seed": cfg["seed"], "intent_policy": intent_policy,
                          "slot_policy": policy, "metrics": entry, "training_mode": mode}
            torch.save(best_state, output / "vcm.pt")
        else:
            stale += 1
        (output / "training-history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"EPOCH mode={mode} epoch={epoch} focus={focus} precision={commands['accepted_command_precision']:.3f} macro_raw={macro_raw:.3f}", flush=True)
        if progress:
            progress(entry)
        if stale >= cfg["patience"]:
            break
    metadata["acoustic_slots"]["variants"]["fp32"] = best_state["slot_policy"]
    metadata["routing"]["variants"]["fp32"] = best_state["intent_policy"]
    metadata["experiment"].update(seed=cfg["seed"], epoch=best_state["epoch"], status="cached_training_candidate")
    (output / "training-config.json").write_text(json.dumps({
        "training": cfg, "encoder_mode": mode, "schema": schema, "model_config": model.config,
        "source_checkpoint_sha256": hashlib.sha256((root / "artifacts/vcm.pt").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((root / "data/vcm.jsonl").read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256((root / "data/references.jsonl").read_bytes()).hexdigest(),
        "label_overlay": "label-repairs.json", "selected_epoch": best_state["epoch"],
        "partitions": {"train": len(training), "policy_fit": len(fit_ids), "selection": len(select_ids)},
        "selection_rule": "95% observed command precision, minimum focal command recall, focal sum, macro raw family recall, overall command recall",
        "calibration_or_test_used_for_fitting": False}, indent=2) + "\n")
    return metadata, {"mode": mode, "epochs": len(history), "selected_epoch": best_state["epoch"],
                      "train_examples": len(training), "dev_examples": len(development),
                      "source_label_repairs": len(repairs), "encoder_frozen": mode == "frozen",
                      "calibration_or_test_used_for_fitting": False, "selection": best_state["metrics"],
                      "augmentation": "feature masks, speed and gain" if mode == "finetune" else "cached original embeddings",
                      "architecture_changed": model.slot_attention != ck["config"].get("slot_attention", False)}
