"""Matched human-only optimization with development-only checkpoint selection."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from scripts.models.vcm_intent_slots import IntentSlotsVCM, joint_loss
from scripts.runtime.pcv_schema import canonical
from scripts.runtime.vcm_intent_slots_contract import digest
from scripts.runtime.vcm_intent_slots_entities import catalog_ids, enroll_examples
from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors
from scripts.test.vcm_intent_slots_eval import summarize_trials
from scripts.train.vcm_intent_slots_data import (
    eligible, prepare_features, rows, sha, write_new,
)

RECIPE = {"human_only": True, "update_artifacts": False, "optimizer": "AdamW",
          "learning_rate": .001, "weight_decay": .0001, "gradient_norm_limit": 1.,
          "batch_size": 64, "epochs": 30, "steps_per_epoch": 96,
          "whole_command_contrastive_weight": 0., "auxiliary_ctc_weight": 0.}


def require_full_scope_inputs(train, dev, inputs, contract):
    supervised = {role for row in train for role in row.get("role_labels", {})}
    if set(contract["roles"]) - supervised:
        raise ValueError("Missing reviewed role-state supervision: " +
                         ",".join(sorted(set(contract["roles"]) - supervised)))
    binding = inputs.get("enrollment_examples")
    thresholds = inputs.get("development_entity_policy")
    if not binding or not thresholds:
        raise ValueError("Missing full-scope enrollment and predeclared development entity policy")
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("Enrollment source binding mismatch")
    enrollment = rows(binding["path"])
    allowed = catalog_ids(contract)
    allowed["light"].discard("light.all")
    coverage = defaultdict(set)
    query_people = {r["speaker_id"] for r in dev}
    query_pcm = {r["pcm_sha256"] for r in train+dev}
    for ex in enrollment:
        if (ex.get("kind") != "human" or ex.get("split") != "enrollment"
                or ex["speaker_id"] in query_people or ex["pcm_sha256"] in query_pcm
                or ex["entity_id"] not in allowed.get(ex["entity_role"], set())):
            raise ValueError("Ineligible or overlapping human enrollment")
        coverage[(ex["entity_role"], ex["entity_id"])].add(ex["pcm_sha256"])
    missing = [entity for role, ids in allowed.items() for entity in ids
               if len(coverage[(role, entity)]) < 3]
    if missing or set(thresholds) != set(contract["entity_roles"]):
        raise ValueError("Missing enrollment/role policy coverage: "+",".join(missing))
    return enrollment, thresholds


def development_memory(model, enrollment, thresholds, contract, normalization, audio_root):
    """Re-encode original enrollment against each checkpoint under raw dev policy."""
    import hashlib
    from scripts.test.speaker_demo_suite import read_audio
    from vcm.frontend import features
    model.eval()
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    binding = {"encoder": h.hexdigest(), "frontend": sha(Path(__file__).resolve().parents[2]/"vcm/frontend.py"),
               "normalization": digest(normalization)}
    examples = []
    with torch.no_grad():
        for row in enrollment:
            x = features(read_audio(row, Path(audio_root)), normalization["mean"], normalization["std"])
            if x.shape[1] == 0:
                raise ValueError("No valid enrollment acoustic frames")
            out = model(torch.tensor(x[None], device=next(model.parameters()).device),
                        torch.tensor([x.shape[1]], device=next(model.parameters()).device))
            role = row["entity_role"]
            vector = out["entity_embeddings"][0, contract["entity_roles"].index(role)].cpu().numpy()
            examples.append({**row, "role": role, "embedding": vector.tolist()})
    memory = enroll_examples(examples, binding, contract)
    policies = {}
    for role in contract["entity_roles"]:
        count = len({ex["entity_id"] for ex in memory["examples"] if ex["role"] == role})
        policies[role] = {**thresholds[role], "encoder_binding": binding,
                          "catalog_hash": contract["catalog_hash"], "catalog_sizes": [count]}
    return memory, {"entities": policies, "require_calibration": False,
                    "status": "predeclared_raw_development_only"}


def validate_training_protocol(protocol, arm, seed):
    if arm not in ("uniform", "balanced") or seed not in (53, 71, 89):
        raise ValueError("Unapproved training arm/seed")
    if any(protocol["training"].get(k) != v for k, v in RECIPE.items()):
        raise ValueError("Recipe changes require a separate experiment")
    if protocol.get("installation_allowed") is not False or protocol.get("provider_actions_allowed") is not False:
        raise ValueError("Installation and actions must stay disabled")


def sample_indices(data, arm, seed, count):
    rng = np.random.default_rng(seed)
    if arm == "uniform":
        return rng.integers(0, len(data), count)
    if arm != "balanced":
        raise ValueError("Unknown sampler")
    cells = defaultdict(lambda: defaultdict(list))
    for n, row in enumerate(data):
        cells[canonical(row.get("command"))][row["speaker_id"]].append(n)
    commands = sorted(cells)
    result = []
    for _ in range(count):
        cmd = commands[int(rng.integers(len(commands)))]
        people = sorted(cells[cmd])
        person = people[int(rng.integers(len(people)))]
        recordings = cells[cmd][person]
        result.append(recordings[int(rng.integers(len(recordings)))])
    return np.asarray(result)


def make_targets(data, contract, device="cpu"):
    """Unknown role states, timer grammar and spans stay masked, never guessed."""
    b, r = len(data), len(contract["roles"])
    intent = torch.tensor([contract["intent_order"].index(
        (row.get("command") or {}).get("intent", "OOV")) for row in data], device=device)
    states = torch.full((b, r), -100, device=device)
    validity = torch.full((b,), -100, device=device)
    values = {role: torch.full((b,), -100, device=device)
              for role in contract["finite_values"]}
    entity_ids, spans = [], []
    for n, row in enumerate(data):
        cmd = row.get("command") or {}
        slots = cmd.get("slots", {})
        if row.get("request_validity") in contract["request_validity_states"]:
            validity[n] = contract["request_validity_states"].index(row["request_validity"])
        elif row.get("command") is not None:
            validity[n] = 0
        for role, state in row.get("role_labels", {}).items():
            states[n, contract["roles"].index(role)] = contract["role_states"].index(state)
        observed = dict(row.get("acoustic_values", {}))
        mapping = {"BRIGHTNESS": {"brightness_percent": "percent"},
                   "VOLUME": {"volume_operation": "operation", "volume_percent": "percent"},
                   "COLOR": {"color": "color"}, "TEMPERATURE": {"temperature_degrees": "degrees"}}
        for role, field in mapping.get(cmd.get("intent"), {}).items():
            if field in slots:
                observed[role] = slots[field]
        for role, value in observed.items():
            values[role][n] = contract["finite_values"][role].index(value)
        entity_ids.append([slots.get(role+"_id") if slots.get(role+"_id") != "light.all" else None
                           for role in contract["entity_roles"]])
        for span in row.get("reviewed_spans", []):
            spans.append((n, contract["roles"].index(span["role"])+1,
                          span["encoded_start"], span["encoded_end"]))
    return {"intent": intent, "role_states": states, "validity": validity,
            "values": values, "entity_ids": entity_ids,
            "speaker_ids": [row["speaker_id"] for row in data], "spans": spans}


def tensor_prediction(out, index=0):
    return {k: ({r: v[index].detach().cpu().numpy() for r, v in value.items()}
                if isinstance(value, dict) else value[index].detach().cpu().numpy())
            for k, value in out.items() if k not in ("role_frame_logits", "encoded_lengths")}


def feature_batch(data, receipt, normalization, device):
    xs = [(np.load(receipt[row["id"]]["path"]) -
           np.asarray(normalization["mean"])[:, None]) /
          np.asarray(normalization["std"])[:, None] for row in data]
    length = torch.tensor([x.shape[1] for x in xs], device=device)
    padded = np.zeros((len(xs), 40, max(x.shape[1] for x in xs)), np.float32)
    for n, x in enumerate(xs):
        padded[n, :, :x.shape[1]] = x
    return torch.tensor(padded, device=device), length


def development_score(model, data, receipt, normalization, contract, memory=None, policy=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for row in data:
            x, lengths = feature_batch([row], receipt, normalization, next(model.parameters()).device)
            pred = prediction_from_tensors(tensor_prediction(model(x, lengths)), contract, memory, policy)
            predictions.append({"id": row["id"], "application": decide(pred, contract, contract, policy or {})})
    metrics = summarize_trials(data, predictions)
    micro = metrics["all"]
    key = (metrics["macro_person_recall"] or 0, -(micro["false_accepts"] +
           micro["accepted_n"] - micro["correct"] - micro["false_accepts"]), micro["recall"] or 0)
    return key, metrics


def fit_run(protocol, inputs, contract, output, *, arm, seed, device="cpu"):
    validate_training_protocol(protocol, arm, seed)
    from scripts.train.vcm_intent_slots_data import audit_views
    output = Path(output)
    if output.exists():
        raise FileExistsError("Interrupted/existing run retained: " + str(output))
    audit = audit_views({s: v["path"] for s, v in inputs["views"].items()},
                        {"views": inputs["views"]}, output/"audit")
    if not audit["fit_ready"]:
        write_new(output/"blocked.json", {"missing_intents": audit["missing_intents"], "status": "blocked_not_fitted"})
        raise ValueError("Missing human training/development coverage; full scope cannot be narrowed")
    train, dev = rows(inputs["views"]["train"]["path"]), rows(inputs["views"]["dev"]["path"])
    for row in train+dev:
        eligible(row)
    enrollment, entity_thresholds = require_full_scope_inputs(train, dev, inputs, contract)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    model = IntentSlotsVCM(contract).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    receipt, normalization = prepare_features(train+dev, inputs["audio_root"], output/"features")
    write_new(output/"run-identity.json", {
        "arm": arm, "seed": seed, "protocol": protocol, "inputs": inputs,
        "contract_hash": contract["contract_hash"], "initialization": "random",
        "selection": "human development only", "status": "running"})
    best_key, history, selected = None, [], None
    total_counts, mass = defaultdict(int), defaultdict(int)
    for epoch in range(30):
        model.train()
        indices = sample_indices(train, arm, seed+epoch*1009, 64*96)
        epoch_loss = 0.
        for start in range(0, len(indices), 64):
            batch = [train[i] for i in indices[start:start+64]]
            for row in batch:
                mass["human"] += 1
            x, lengths = feature_batch(batch, receipt, normalization, device)
            optimizer.zero_grad()
            loss, counts = joint_loss(model(x, lengths), make_targets(batch, contract, device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            epoch_loss += float(loss.detach())
            for k, v in counts.items():
                total_counts[k] += v
        memory, dev_policy = development_memory(model, enrollment, entity_thresholds,
                                               contract, normalization, inputs["audio_root"])
        key, metrics = development_score(model, dev, receipt, normalization, contract, memory, dev_policy)
        record = {"epoch": epoch+1, "updates": 96, "loss": epoch_loss/96,
                  "selection_key": key, "development": metrics}
        history.append(record)
        write_new(output/f"epoch-{epoch+1:02}.json", record)
        if best_key is None or key > best_key:
            best_key, selected = key, epoch+1
            path = output/f"checkpoint-{epoch+1:02}.pt"
            torch.save({"weights": model.cpu().state_dict(), "contract": contract,
                        "normalization": normalization, "training": protocol["training"],
                        "seed": seed, "arm": arm, "epoch": selected,
                        "status": "unqualified_candidate", "inputs": inputs}, path)
            model.to(device)
    result = {"status": "unqualified_candidate", "epoch": selected,
              "checkpoint": str(output/f"checkpoint-{selected:02}.pt"),
              "checkpoint_sha256": sha(output/f"checkpoint-{selected:02}.pt"),
              "sampling_mass": dict(mass), "supervision_denominators": dict(total_counts),
              "normalization": str(output/"features/normalization.json"), "development_key": best_key}
    write_new(output/"selection-before-calibration.json", result)
    return result
