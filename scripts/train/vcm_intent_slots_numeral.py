"""Isolated human numeral-category component experiment, never command fitting."""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from scripts.train.vcm_intent_slots_data import sha, write_new


def _rank(seed, value):
    return hashlib.sha256(f"{seed}/{value}".encode()).hexdigest()


def partition(data, reserved, attributes, seed=53):
    """One take/person/value; stable 70/15/15 person split; originals unchanged.

    MLEnd category labels are checked against the original attributes CSV.
    They provide no command, role-presence, grammar, span or identity labels.
    """
    exclusions = Counter()
    candidates, seen = [], set()
    keys = ("speaker_id", "sha256", "pcm_sha256", "pair_id")
    held = {key: {r[key] for r in reserved if r.get(key)} for key in keys}
    for row in data:
        metadata = row.get("publisher_metadata", {})
        if metadata.get("source") != "MLEnd_numerals":
            continue
        if row["id"] in seen:
            raise ValueError("duplicate numeral id")
        seen.add(row["id"])
        if (row.get("kind") != "human" or row.get("published_split") != "numerals"
                or row.get("reference_status") != "auxiliary"
                or row.get("qa_eligible") is not True or row.get("audio_flags")
                or row.get("command") is not None
                or row.get("sample_rate") != 16000 or row.get("channels") != 1
                or not 400 <= row.get("samples", 0) <= 128000
                or not row.get("license") or not row.get("source_revision")
                or not all(row.get(k) for k in keys)):
            exclusions["not_original_eligible_human"] += 1
            continue
        match = re.search(r"_MLEnd_numerals_(\d+)\.wav$", metadata.get("file", ""))
        note = attributes.get(match.group(1)) if match else None
        if (not note or note["Speaker"] != metadata.get("speaker_id")
                or row["speaker_id"] != "MLEnd_numerals/" + note["Speaker"]
                or note["Numeral"] != metadata.get("numerals")
                or not note["Numeral"].isdigit()
                or note["Intonation"] not in ("neutral", "bored", "excited", "question")):
            exclusions["original_metadata_conflict"] += 1
            continue
        value = int(note["Numeral"])
        if value > 100:
            exclusions["outside_numeric_head"] += 1
            continue
        if any(row[k] in held[k] for k in keys):
            exclusions["reserved_identity"] += 1
            continue
        candidates.append({**row, "numeric_value": value, "intonation": note["Intonation"],
                           "original_view_split": row["split"]})
    groups = {key: defaultdict(list) for key in keys[1:]}
    for n, row in enumerate(candidates):
        for key in groups:
            groups[key][row[key]].append(n)
    conflicts = set()
    for index in groups.values():
        for ns in index.values():
            if len({candidates[n]["numeric_value"] for n in ns}) > 1:
                conflicts.update(ns)
    used, cells = {key: set() for key in groups}, set()
    kept = []
    for n in sorted(range(len(candidates)), key=lambda n: _rank(seed, candidates[n]["id"])):
        row = candidates[n]
        if n in conflicts:
            exclusions["conflicting_identity_labels"] += 1
            continue
        if any(row[key] in used[key] for key in used):
            exclusions["duplicate_identity"] += 1
            continue
        # Include capped takes in the duplicate index so their copies cannot migrate.
        for key in used:
            used[key].add(row[key])
        cell = (row["speaker_id"], row["numeric_value"])
        if cell in cells:
            exclusions["repeated_take_cap"] += 1
            continue
        cells.add(cell)
        kept.append(row)
    people = sorted({r["speaker_id"] for r in kept}, key=lambda p: _rank(seed, p))
    if len(people) < 3:
        raise ValueError("Insufficient people for three separate partitions")
    ntrain, ndev = max(1, int(.7 * len(people))), max(1, int(.15 * len(people)))
    assignment = {p: ("train" if n < ntrain else "dev" if n < ntrain + ndev else "test")
                  for n, p in enumerate(people)}
    views = {s: sorted([{**r, "split": "numeral_" + s} for r in kept
                        if assignment[r["speaker_id"]] == s], key=lambda r: r["id"])
             for s in ("train", "dev", "test")}
    report = {"source_rows": len(seen), "selected": len(kept), "exclusions": dict(exclusions),
              "alias_resolution": "Declared IDs only; cross-source aliases remain unverified",
              "gender_policy": "No inference or gender routing; person-disjoint first",
              "cap": "one original take per person/numeral before model fitting",
              "partitions": {s: {"rows": len(rs), "people": len({r["speaker_id"] for r in rs}),
                  "values": dict(Counter(str(r["numeric_value"]) for r in rs)),
                  "intonation": dict(Counter(r["intonation"] for r in rs)),
                  "publisher_accent_group": dict(Counter(
                      r["publisher_metadata"].get("accent_group", "unknown") for r in rs)),
                  "gender": {"unknown": len({r["speaker_id"] for r in rs})}}
                  for s, rs in views.items()}}
    for key in keys:
        sets = [{r[key] for r in rs} for rs in views.values()]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Cross-component partition leakage: " + key)
    report["declared_cross_partition_overlaps"] = {k: 0 for k in keys}
    return views, report


def require_coverage(views):
    values = {r["numeric_value"] for rs in views.values() for r in rs}
    for split in ("train", "dev", "test"):
        if values - {r["numeric_value"] for r in views[split]}:
            raise ValueError("Missing observed numeric category coverage: " + split)


def numeric_optimizer(model):
    """Unobserved head weights and inactive queries do not decay or update."""
    parameters = []
    for name, parameter in model.named_parameters():
        active = name == "queries" or name.startswith(
            ("stem.", "blocks.", "keys.", "values.brightness_percent."))
        parameter.requires_grad_(active)
        if active and name != "queries":
            parameters.append(parameter)
    return torch.optim.AdamW([{"params": parameters, "weight_decay": .0001},
                              {"params": [model.queries], "weight_decay": 0.}], lr=.001)


def numeric_loss(prediction, values, contract):
    classes = contract["finite_values"]["brightness_percent"]
    labels = torch.tensor([classes.index(v) for v in values],
                          device=prediction["value_logits"]["brightness_percent"].device)
    return F.cross_entropy(prediction["value_logits"]["brightness_percent"], labels)


def sample_numeric(data, seed, count):
    cells = defaultdict(lambda: defaultdict(list))
    for n, row in enumerate(data):
        cells[row["numeric_value"]][row["speaker_id"]].append(n)
    rng, result = np.random.default_rng(seed), []
    values = sorted(cells)
    for _ in range(count):
        value = values[int(rng.integers(len(values)))]
        people = sorted(cells[value])
        person = people[int(rng.integers(len(people)))]
        takes = cells[value][person]
        result.append(takes[int(rng.integers(len(takes)))])
    return np.asarray(result)


class DevelopmentStop:
    """Select macro-person top1, then NLL; never extend the declared ceiling."""
    def __init__(self, max_checks=4, patience=2):
        if max_checks < 1 or patience < 1:
            raise ValueError("Invalid stopping budget")
        self.max_checks, self.patience = max_checks, patience
        self.checks, self.stale, self.best_check = 0, 0, None
        self.best, self.stopped = None, False

    def observe(self, macro_accuracy, nll):
        if self.stopped:
            raise ValueError("Experiment already stopped")
        if not np.isfinite([macro_accuracy, nll]).all():
            raise ValueError("Nonfinite development metric")
        self.checks += 1
        key = (macro_accuracy, -nll)
        improved = self.best is None or key[0] > self.best[0] + .0001 or (
            abs(key[0] - self.best[0]) <= .0001 and key[1] > self.best[1] + .0001)
        if improved:
            self.best, self.best_check, self.stale = key, self.checks, 0
        else:
            self.stale += 1
        self.stopped = self.checks >= self.max_checks or self.stale >= self.patience
        return improved, self.stopped


COMPONENT_BINDINGS = ("protocol", "audit", "train", "dev", "test", "features",
                      "contract", "normalization", "checkpoint", "initial_checkpoint",
                      "frontend", "model_source", "training_source", "batch_source",
                      "prepared", "catalog")


def lock_component(bindings, path):
    if set(bindings) != set(COMPONENT_BINDINGS):
        raise ValueError("Incomplete component binding")
    value = {"purpose": "numeral_category_component_only", "qualification": False,
             "calibration": None, "actions_enabled": False,
             "files": {key: {"path": str(Path(p).resolve()), "sha256": sha(p)}
                       for key, p in bindings.items()}}
    write_new(path, value)
    return value


def _protocol(path):
    value = json.loads(Path(path).read_text())
    if (value.get("experiment_id") != "vcm-intent-slots-numeral-aux-v1"
            or value.get("purpose") != "numeral_category_component_only"
            or value.get("seed") != 53 or value.get("patience") != 2
            or not 1 <= value.get("max_checks", 0) <= 4
            or not 1 <= value.get("steps_per_check", 0) <= 60
            or not 1 <= value.get("batch_size", 0) <= 64
            or value.get("update_artifacts") is not False
            or value.get("provider_actions_allowed") is not False
            or value.get("installation_allowed") is not False
            or value.get("calibration") is not None):
        raise ValueError("Invalid bounded numeric-only protocol")
    return value


def _binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def _check(bindings):
    for name, binding in bindings.items():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Input binding changed: " + name)


def prepare(inputs_path, auxiliary, attributes, normalization, protocol_path, output):
    """Freeze metadata/splits and verify each selected original WAV and feature."""
    from scripts.runtime.vcm_intent_slots_contract import verify_contract
    from scripts.runtime.vcm_intent_slots_runtime import validate_pcm
    from scripts.test.speaker_demo_suite import read_audio
    from scripts.train.vcm_intent_slots_data import rows
    from vcm.frontend import features
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    protocol = _protocol(protocol_path)
    inputs = json.loads(Path(inputs_path).read_text())
    if set(inputs.get("views", {})) != {"train", "dev", "calibration", "test", "holdout"}:
        raise ValueError("All five frozen parent partitions must be reserved")
    if sha(attributes) != protocol.get("sources", {}).get("metadata_sha256"):
        raise ValueError("Original metadata does not match the declared pin")
    _check({**inputs["views"], "contract": inputs["contract"], "catalog": inputs["catalog"]})
    contract = json.loads(Path(inputs["contract"]["path"]).read_text())
    verify_contract(contract)
    if contract["finite_values"]["brightness_percent"] != list(range(101)):
        raise ValueError("Complete 101-value numeric head required")
    norm = json.loads(Path(normalization).read_text())
    if (norm.get("fit_split") != "train" or len(norm.get("mean", [])) != 40
            or len(norm.get("std", [])) != 40
            or not np.isfinite(norm["mean"] + norm["std"]).all()
            or min(norm["std"]) <= 0):
        raise ValueError("Frozen train-only normalization required")
    notes = list(csv.DictReader(Path(attributes).open()))
    index = {r["Public filename"]: r for r in notes}
    if len(index) != len(notes):
        raise ValueError("Duplicate original audio attribute key")
    reserved = [r for b in inputs["views"].values() for r in rows(b["path"])]
    views, audit = partition(rows(auxiliary), reserved, index, protocol["seed"])
    require_coverage(views)
    output.mkdir(parents=True)
    write_new(output / "audit.json", audit)
    for split, rs in views.items():
        with (output / (split + ".jsonl")).open("x") as stream:
            for row in rs:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_new(output / "normalization.json", norm)
    write_new(output / "contract.json", contract)
    write_new(output / "protocol.json", protocol)
    source_root = Path(__file__).resolve().parents[2]
    source_paths = {"frontend": source_root / "vcm/frontend.py",
                    "model_source": source_root / "scripts/models/vcm_intent_slots.py",
                    "training_source": Path(__file__),
                    "batch_source": source_root / "scripts/train/vcm_intent_slots_train.py"}
    bindings = {k: _binding(output / (k + ".jsonl")) for k in views}
    bindings.update({k: _binding(output / (k + ".json")) for k in
                     ("audit", "protocol", "normalization", "contract")})
    bindings.update({k: _binding(p) for k, p in source_paths.items()})
    original = {"inputs": _binding(inputs_path), "auxiliary": _binding(auxiliary),
                "attributes": _binding(attributes), "normalization": _binding(normalization),
                "protocol": _binding(protocol_path), **inputs["views"],
                "catalog": inputs["catalog"], "contract": inputs["contract"]}
    receipt = {}
    feature_dir = output / "features"
    feature_dir.mkdir()
    for rs in views.values():
        for row in rs:
            try:
                pcm = read_audio(row, Path(inputs["audio_root"]))
                validate_pcm(pcm, 16000)
                x = features(pcm)
                path = feature_dir / (row["pcm_sha256"] + ".npy")
                with path.open("xb") as stream:
                    np.save(stream, x)
                receipt[row["id"]] = {**_binding(path), "frames": int(x.shape[1]),
                                       "pcm_sha256": row["pcm_sha256"]}
            except Exception as error:
                write_new(output / "preparation-fault.json", {"id": row["id"],
                    "audio": row["audio"], "fault": type(error).__name__ + ": " + str(error)})
                raise
    write_new(output / "features.json", receipt)
    bindings["features"] = _binding(output / "features.json")
    bindings["catalog"] = inputs["catalog"]
    write_new(output / "prepared.json", {"purpose": protocol["purpose"],
        "bindings": bindings, "original_inputs": original, "verified_audio_n": len(receipt),
        "normalization": "Reused frozen main human-train statistics; no dev/test fitting",
        "qualification": False, "actions_enabled": False})
    return audit


def _prepared(path):
    value = json.loads((Path(path) / "prepared.json").read_text())
    _check(value["original_inputs"])
    _check(value["bindings"])
    receipt = json.loads(Path(value["bindings"]["features"]["path"]).read_text())
    for name, binding in receipt.items():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("feature binding changed: " + name)
    return value, receipt


def _score_numeric(model, data, receipt, normalization):
    from scripts.train.vcm_intent_slots_train import feature_batch
    model.eval()
    people, values, modes = defaultdict(list), defaultdict(list), defaultdict(list)
    losses, correct = [], []
    classes = model.contract["finite_values"]["brightness_percent"]
    with torch.no_grad():
        for start in range(0, len(data), 32):
            batch = data[start:start + 32]
            x, length = feature_batch(batch, receipt, normalization, "cpu")
            logits = model(x, length)["value_logits"]["brightness_percent"]
            labels = torch.tensor([classes.index(r["numeric_value"]) for r in batch])
            hit = (logits.argmax(-1) == labels).tolist()
            losses.extend(F.cross_entropy(logits, labels, reduction="none").tolist())
            correct.extend(hit)
            for row, good in zip(batch, hit):
                people[row["speaker_id"]].append(good)
                values[str(row["numeric_value"])].append(good)
                modes[row.get("intonation", "not_supplied")].append(good)
    def counts(groups):
        return {key: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
                for key, v in sorted(groups.items())}
    accuracies = [sum(v) / len(v) for v in people.values()]
    return {"n": len(data), "people": len(people), "correct": sum(correct),
            "accuracy": float(np.mean(correct)), "nll": float(np.mean(losses)),
            "macro_person_accuracy": float(np.mean(accuracies)),
            "worst_person_accuracy": min(accuracies), "person_counts": counts(people),
            "numeric_category_counts": counts(values), "intonation_counts": counts(modes),
            "head_values": 101, "output_kind": "raw_numeral_category_top1"}


def fit(prepared, output):
    """Bounded random initialization, development-only selection and early stop."""
    from scripts.models.vcm_intent_slots import IntentSlotsVCM
    from scripts.train.vcm_intent_slots_data import rows
    from scripts.train.vcm_intent_slots_train import feature_batch
    output, prepared = Path(output).resolve(), Path(prepared).resolve()
    if output.exists():
        raise FileExistsError(output)
    value, receipt = _prepared(prepared)
    b = value["bindings"]
    protocol = _protocol(b["protocol"]["path"])
    train, dev = rows(b["train"]["path"]), rows(b["dev"]["path"])
    contract = json.loads(Path(b["contract"]["path"]).read_text())
    norm = json.loads(Path(b["normalization"]["path"]).read_text())
    torch.manual_seed(protocol["seed"])
    np.random.seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    model = IntentSlotsVCM(contract)
    optimizer = numeric_optimizer(model)
    output.mkdir(parents=True)
    write_new(output / "run-identity.json", {"protocol": protocol, "prepared": _binding(
        prepared / "prepared.json"), "initialization": "random", "qualification": False})
    def save(path):
        torch.save({"weights": model.state_dict(), "contract": contract,
                    "normalization": norm, "status": "unqualified_numeral_component_only"}, path)
    initial = output / "initial.pt"
    save(initial)
    initial_dev = _score_numeric(model, dev, receipt, norm)
    write_new(output / "initial-development.json", initial_dev)
    stop = DevelopmentStop(protocol["max_checks"], protocol["patience"])
    stop.best, stop.best_check = (initial_dev["macro_person_accuracy"], -initial_dev["nll"]), 0
    selected, updates, mass = initial, 0, Counter()
    history = []
    for check in range(protocol["max_checks"]):
        model.train()
        count = protocol["batch_size"] * protocol["steps_per_check"]
        draws = sample_numeric(train, protocol["seed"] + check * 1009, count)
        train_loss = 0.
        for start in range(0, count, protocol["batch_size"]):
            batch = [train[n] for n in draws[start:start + protocol["batch_size"]]]
            mass.update(r["id"] for r in batch)
            x, lengths = feature_batch(batch, receipt, norm, "cpu")
            optimizer.zero_grad()
            loss = numeric_loss(model(x, lengths), [r["numeric_value"] for r in batch], contract)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite numeric component loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
            train_loss += float(loss.detach())
        dev_metrics = _score_numeric(model, dev, receipt, norm)
        train_metrics = _score_numeric(model, train, receipt, norm)
        improved, stopped = stop.observe(dev_metrics["macro_person_accuracy"], dev_metrics["nll"])
        record = {"check": check + 1, "updates": updates,
                  "sampled_train_nll": train_loss / protocol["steps_per_check"],
                  "train": train_metrics, "development": dev_metrics, "selected": improved,
                  "train_minus_dev_macro_accuracy": train_metrics["macro_person_accuracy"] -
                      dev_metrics["macro_person_accuracy"]}
        history.append(record)
        write_new(output / f"check-{check + 1:02}.json", record)
        if improved:
            selected = output / f"checkpoint-{check + 1:02}.pt"
            save(selected)
        print(json.dumps({"check": check + 1, "updates": updates,
              "train_accuracy": train_metrics["accuracy"], "dev_accuracy": dev_metrics["accuracy"],
              "dev_nll": dev_metrics["nll"], "selected": improved, "stopped": stopped}), flush=True)
        if stopped:
            break
    result = {"qualification": False, "actions_enabled": False, "calibration": None,
              "checkpoint": _binding(selected), "selected_check": stop.best_check,
              "updates": updates, "checks": len(history),
              "stop_reason": "development_patience" if stop.stale >= stop.patience else "budget_ceiling",
              "sampled_examples": sum(mass.values()), "unique_sampled_recordings": len(mass),
              "max_exposures_per_recording": max(mass.values()),
              "loss_supervision": {"numeric_categories": sum(mass.values()),
                  "intent": 0, "presence": 0, "validity": 0, "other_slots": 0,
                  "entity": 0, "span": 0, "identity": 0},
              "history": history, "initial_development": initial_dev}
    write_new(output / "training-result.json", result)
    lock_component({**{key: Path(binding["path"]) for key, binding in b.items()},
                    "checkpoint": selected, "initial_checkpoint": initial,
                    "prepared": prepared / "prepared.json"},
                   output / "selection-before-component-test.json")
    return result


def score_run(run, output):
    """The held-person component test may run only against a complete lock."""
    from scripts.models.vcm_intent_slots import IntentSlotsVCM
    from scripts.train.vcm_intent_slots_data import rows
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    lock = verify_component(run / "selection-before-component-test.json")
    files = lock["files"]
    prepared = Path(files["prepared"]["path"]).parent
    _, receipt = _prepared(prepared)
    data = rows(files["test"]["path"])
    metrics = {}
    for key, binding in (("initial", files["initial_checkpoint"]), ("selected", files["checkpoint"])):
        checkpoint = torch.load(binding["path"], map_location="cpu", weights_only=False)
        model = IntentSlotsVCM(checkpoint["contract"])
        model.load_state_dict(checkpoint["weights"])
        metrics[key] = _score_numeric(model, data, receipt, checkpoint["normalization"])
    result = {"purpose": "numeral_category_component_only", "qualification": False,
              "actions_enabled": False, "calibration": None,
              "selection_lock": _binding(run / "selection-before-component-test.json"),
              "test_status": "New component split of previously available category-labeled speech; not fresh command holdout",
              **metrics}
    write_new(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    for name in ("inputs", "auxiliary", "attributes", "normalization", "protocol", "output"):
        p.add_argument("--" + name, required=True)
    p = sub.add_parser("fit")
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("score")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    if mode == "prepare":
        args["inputs_path"] = args.pop("inputs")
        args["protocol_path"] = args.pop("protocol")
        result = prepare(**args)
    elif mode == "fit":
        result = fit(**args)
    else:
        result = score_run(**args)
    print(json.dumps({k: v for k, v in result.items() if k not in (
        "history", "initial_development", "initial", "selected")}, indent=2))


def verify_component(path):
    value = json.loads(Path(path).read_text())
    if (value.get("purpose") != "numeral_category_component_only"
            or value.get("qualification") is not False or value.get("calibration") is not None
            or value.get("actions_enabled") is not False
            or set(value.get("files", {})) != set(COMPONENT_BINDINGS)):
        raise ValueError("Incomplete or wrong component binding")
    for key, binding in value["files"].items():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Component binding changed: " + key)
    return value


if __name__ == "__main__":
    main()
