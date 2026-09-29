"""Continue the current checkpoint; train/dev only, with explicit artifact replacement."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import time
import numpy as np
import torch
from einops import pack
from .data import read_jsonl
from .train_helpers import batches, make_batch, seed_everything, warm_cache


def subset(rows, split, maximum):
    selected = [r for r in rows if r["split"] == split]
    return selected[:maximum] if maximum else selected


def fit_vcm(config, root, output):
    from .slot_model import SlotVCM, masked_head_losses
    from .finite_policy import provisional_slots, slot_records, command_metrics
    from .metrics import policy_rows
    from .slot_contract import catalog_signature
    cfg = config["training"]
    ck = torch.load(root / "artifacts/vcm.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((root / "artifacts/bundle.json").read_text(encoding="utf-8"))
    schema = json.loads((root / "data/contract.json").read_text(encoding="utf-8"))
    refs = {r["id"]: r for r in read_jsonl(root / "data/references.jsonl")}
    rows = read_jsonl(root / "data/vcm.jsonl")
    training = subset(rows, "train", cfg["max_train_examples"])
    development = subset(rows, "dev", cfg["max_dev_examples"])
    folds = json.loads((root / "data/development-folds.json").read_text(encoding="utf-8"))
    # If a debug limit is used, still take examples from both fixed disjoint folds.
    if cfg["max_dev_examples"]:
        maximum = max(1, cfg["max_dev_examples"] // 2)
        development = [r for fold in ("policy_fit", "selection")
                       for r in [x for x in rows if x["split"] == "dev" and folds[x["id"]] == fold][:maximum]]
    rows = training + development
    if not training or not development:
        raise ValueError("Training requires nonempty train and development records")
    train_ids = list(range(len(training)))
    fit_ids = [i for i, r in enumerate(rows) if r["split"] == "dev" and folds[r["id"]] == "policy_fit"]
    select_ids = [i for i, r in enumerate(rows) if r["split"] == "dev" and folds[r["id"]] == "selection"]
    if not fit_ids or not select_ids:
        raise ValueError("Need both grouped development folds")
    device = cfg["device"]
    model = SlotVCM(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["model"])
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    warm_cache(rows)
    vectors = torch.empty(len(rows), model.heads.project[0].normalized_shape[0])
    logits = torch.empty(len(rows), len(ck["labels"]))
    with torch.inference_mode():
        for n, ids in enumerate(batches(rows, cfg["batch_size"])):
            x, lengths = make_batch(rows, ids, ck["normalization"])
            a, b = model.encode(x.to(device), lengths.to(device))
            vectors[ids], logits[ids] = a.cpu(), b.cpu()
            if n % 100 == 0:
                print(f"VCM ENCODE batch={n}", flush=True)
    vectors, logits = vectors.to(device), logits.to(device)
    vocab = ck["config"]["vocabulary"]
    annotations = [refs[r["id"]] for r in rows]
    targets = {name: torch.tensor([values.index(a["labels"][name]) if a["labels"][name] is not None else 0
                                  for a in annotations], device=device) for name, values in vocab.items()}
    masks = {name: torch.tensor([a["labels"][name] is not None for a in annotations], device=device) for name in vocab}
    # Frozen family encoder => preserve the exact current family acceptance policy.
    intent_policy = ck["intent_policy"]
    def accepted(ids):
        return policy_rows([{"id": rows[i]["id"], "gold": rows[i]["command"],
                             "probabilities": torch.softmax(logits[i], -1).cpu().tolist()} for i in ids],
                           ck["labels"], intent_policy)
    accepted_fit, accepted_selection = accepted(fit_ids), accepted(select_ids)
    optimizer = torch.optim.AdamW([p for p in model.heads.parameters() if p.requires_grad],
                                 lr=cfg["learning_rate"], weight_decay=.01)
    best, best_state, history, stale = None, None, [], 0
    groups = {label: [i for i in train_ids if rows[i]["command"] == label] for label in ck["labels"]}
    groups = {label: ids for label, ids in groups.items() if ids}
    for epoch in range(1, cfg["vcm_epochs"] + 1):
        model.heads.train()
        rng = random.Random(cfg["seed"] * 1000 + epoch)
        draws = [rng.choice(ids) for ids in groups.values() for _ in range(min(600, len(training)))]
        rng.shuffle(draws)
        losses_seen = []
        for offset in range(0, len(draws), cfg["batch_size"]):
            ids = draws[offset:offset + cfg["batch_size"]]
            outputs = model.heads(vectors[ids], logits[ids])
            losses = masked_head_losses(dict(zip(vocab, outputs[1:])),
                {name: v[ids] for name, v in targets.items()}, {name: v[ids] for name, v in masks.items()})
            loss = sum(.5 * v for v in losses.values())
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 3.)
            optimizer.step(); losses_seen.append(float(loss.detach()))
        model.heads.eval()
        with torch.inference_mode():
            fit_outputs = model.heads(vectors[fit_ids], logits[fit_ids])
            selection_outputs = model.heads(vectors[select_ids], logits[select_ids])
        torch.testing.assert_close(selection_outputs[0], logits[select_ids], atol=0., rtol=0.)
        fit_slots, selection_slots = slot_records(fit_outputs, vocab), slot_records(selection_outputs, vocab)
        policy = provisional_slots([rows[i] for i in fit_ids], [annotations[i] for i in fit_ids],
                                   accepted_fit, fit_slots, schema, metadata["catalog"])
        slot_config = {**schema, "catalog_signature": catalog_signature(metadata["catalog"]), "variants": {"fp32": policy}}
        command, _ = command_metrics([rows[i] for i in select_ids], [annotations[i] for i in select_ids],
            accepted_selection, selection_slots, slot_config, metadata["catalog"])
        rank = (command["accepted_command_precision"] >= .95, command["exact_command_success"],
                command["accepted_command_precision"])
        entry = {"epoch": epoch, "loss": sum(losses_seen)/max(1, len(losses_seen)), "commands": command}
        history.append(entry)
        if best is None or rank > best:
            best, stale = rank, 0
            best_state = {**ck, "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                          "epoch": epoch, "seed": cfg["seed"], "slot_policy": policy, "metrics": entry}
        else:
            stale += 1
        print(f"VCM epoch={epoch} exact={command['exact_command_success']:.4f}", flush=True)
        if stale >= cfg["patience"]:
            break
    torch.save(best_state, output / "vcm.pt")
    (output / "training-history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata["acoustic_slots"]["variants"]["fp32"] = best_state["slot_policy"]
    return metadata, {"epochs": len(history), "selected_epoch": best_state["epoch"],
                      "train_examples": len(training), "dev_examples": len(development), "encoder_frozen": True}




def export_graphs(output, selected, metadata):
    """Export only the VCM; reject obsolete task names before opening files."""
    import onnx
    import onnxruntime as ort
    from .slot_model import SlotVCM
    if selected != ["vcm"]:
        raise ValueError("Only the VCM export is supported")
    ck = torch.load(output / "vcm.pt", map_location="cpu", weights_only=False)
    model = SlotVCM(**ck["config"]).eval()
    model.load_state_dict(ck["model"])
    x, lengths = torch.zeros(1, 40, 160), torch.tensor([160])
    path = output / "vcm.fp32.onnx"
    names = ["logits", *["slot_" + n for n in ck["config"]["vocabulary"]]]
    axes = {"features": {0: "batch", 2: "frames"}, "lengths": {0: "batch"},
            **{name: {0: "batch"} for name in names}}
    torch.onnx.export(model, (x, lengths), str(path),
                      input_names=["features", "lengths"], output_names=names,
                      dynamic_axes=axes, opset_version=17, dynamo=False)
    graph = onnx.load(str(path))
    for key, value in (("vcm_labels", ck["labels"]),
                       ("slot_vocabulary", ck["config"]["vocabulary"])):
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, json.dumps(value)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    maximum = 0.0
    for frames in (61, 160, 317):
        x, lengths = torch.randn(1, 40, frames), torch.tensor([frames])
        with torch.inference_mode():
            expected = model(x, lengths)
        actual = session.run(None, {"features": x.numpy(), "lengths": lengths.numpy()})
        if len(expected) != len(actual):
            raise ValueError("VCM output count changed at export")
        for left, right in zip(expected, actual):
            np.testing.assert_allclose(left.numpy(), right, atol=2e-4, rtol=2e-4)
            maximum = max(maximum, float(np.max(np.abs(left.numpy() - right))))
    metadata["files"][path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata["vcm"].update(checkpoint="vcm.pt", checkpoint_sha256=
        hashlib.sha256((output / "vcm.pt").read_bytes()).hexdigest())
    return {"vcm": maximum}


def fit(config, root):
    root = Path(root); cfg = config["training"]
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; set training.device = 'cpu' explicitly")
    selected = [task for task in ("vcm",) if cfg[task]]
    if not selected:
        raise ValueError("No training component selected")
    if cfg["update_artifacts"] and (cfg["max_train_examples"] or cfg["max_dev_examples"]):
        raise ValueError("Limited debug training cannot replace the current artifacts; set update_artifacts = false")
    seed_everything(cfg["seed"])
    parent = root / ".cache/training"; parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="candidate-", dir=parent))
    for path in (root / "artifacts").iterdir():
        if path.is_file() and not path.name.startswith("asr"):
            shutil.copy2(path, output / path.name)
    metadata = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    result = {"selected": selected, "output": str(output.relative_to(root)),
              "continuation_from_current_weights": True, "calibration_or_test_used_for_fitting": False,
              "scope": "frozen-family encoder with finite-head continuation"}
    if "vcm" in selected:
        metadata, result["vcm"] = fit_vcm(config, root, output)
    result["export_max_errors"] = export_graphs(output, selected, metadata)
    metadata["experiment"].update(status="experimental_continuation", training=result)
    (output / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from .validation import run_checks
    test_config = copy.deepcopy(config); test_config["tests"] = {key: True for key in config["tests"]}
    result["checks"] = run_checks(test_config, output)
    if cfg["update_artifacts"]:
        # Keep a single previous-current recovery copy until the next successful fit.
        backup = parent / "previous-current"
        backup.mkdir(exist_ok=True)
        for path in (root / "artifacts").iterdir():
            if path.is_file() and not path.name.startswith("asr"):
                shutil.copy2(path, backup / path.name)
        for path in output.iterdir():
            if path.is_file() and path.name != "manifest.json":
                shutil.copy2(path, root / "artifacts" / path.name)
    result["artifacts_updated"] = cfg["update_artifacts"]
    return result
