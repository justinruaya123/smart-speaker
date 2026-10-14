"""Distinct FP32 export; never touches models/default or installed artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from scripts.models.vcm_intent_slots import IntentSlotsVCM
from scripts.runtime.vcm_intent_slots_contract import verify_contract
from scripts.train.vcm_intent_slots_data import sha, write_new


class OrderedHeads(nn.Module):
    """Batch-one unpadded endpoint equivalent to the masked training graph."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        m = self.model
        x = m.stem(features)
        for block in m.blocks:
            y = block.depthwise(x)
            y = rearrange(block.norm(rearrange(y, "b c t -> b t c")), "b t c -> b c t")
            x = x + block.pointwise(F.relu(y))
        frames = rearrange(x, "b c t -> b t c")
        scores = torch.matmul(m.keys(frames), m.queries.t()) / 8
        pooled = torch.matmul(rearrange(scores.softmax(dim=1), "b t r -> b r t"), frames)
        roles = {r: pooled[:, n+1] for n, r in enumerate(m.contract["roles"])}
        return (
            m.intent(pooled[:, 0]),
            torch.cat([h(pooled[:, n+1]) for n, h in enumerate(m.states)], dim=1),
            m.validity(pooled[:, 0]),
            torch.cat([h(roles[r]) for r, h in m.values.items()], dim=1),
            torch.cat([F.normalize(h(roles[r]), dim=-1, eps=1e-8)
                       for r, h in zip(m.contract["entity_roles"], m.entities)], dim=1),
        )


def export_candidate(checkpoint, contract, output_dir, *, engines=None,
                     calibration=None, memory=None, entity_policy=None):
    verify_contract(contract)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if data["contract"]["contract_hash"] != contract["contract_hash"]:
        raise ValueError("Checkpoint contract binding mismatch")
    model = IntentSlotsVCM(contract).eval()
    model.load_state_dict(data["weights"], strict=True)
    wrapper = OrderedHeads(model).eval()
    torch.set_num_threads(1)
    rewrite_errors = []
    for n in (1, 3, 17, 32, 101, 231):
        x = torch.linspace(-1, 1, 40*n).reshape(1, 40, n)
        with torch.no_grad():
            old = model(x, torch.tensor([n]))
            reference = [old["intent_logits"], old["role_state_logits"].flatten(1),
                         old["request_validity_logits"],
                         torch.cat(list(old["value_logits"].values()), dim=1),
                         old["entity_embeddings"].flatten(1)]
            errors = [float((a-b).abs().max()) for a, b in zip(wrapper(x), reference)]
            rewrite_errors.append({"frames": n, "max_error": max(errors)})
    if max(r["max_error"] for r in rewrite_errors) > 2e-5:
        raise ValueError("Unpadded rewrite disagrees with training graph")
    write_new(output_dir/"contract.json", contract)
    write_new(output_dir/"catalog.json", {"entries": contract["catalog_entries"], "custom": contract["custom"]})
    write_new(output_dir/"normalization.json", data["normalization"])
    policy = json.loads(Path(calibration).read_text()) if calibration is not None else data.get(
        "policy", {"require_calibration": False, "contract_hash": contract["contract_hash"],
                   "status": "uncalibrated_unqualified"})
    memory = json.loads(Path(memory).read_text()) if memory is not None else data.get("memory")
    if entity_policy is not None:
        if policy.get("require_calibration"):
            raise ValueError("Frozen calibration already binds its entity policies; recalibrate after changing retrieval")
        policy["entities"] = json.loads(Path(entity_policy).read_text())
    root = Path(__file__).resolve().parents[2]
    from scripts.runtime.vcm_intent_slots_bindings import verify_evidence_bindings
    verify_evidence_bindings(contract, policy, memory, encoder=sha(checkpoint),
                             frontend=sha(root/"vcm/frontend.py"),
                             normalization=sha(output_dir/"normalization.json"))
    write_new(output_dir/"policy.json", policy)
    shutil.copyfile(checkpoint, output_dir/"reference.pt")
    names = ["intent", "role_states", "validity", "values", "entities"]
    torch.onnx.export(wrapper, torch.zeros(1, 40, 128), str(output_dir/"model.onnx"),
                      input_names=["features"], output_names=names,
                      dynamic_axes={"features": {2: "frames"}}, opset_version=17,
                      dynamo=False)
    status = {"onnxruntime": "exported"}
    if "ncnn" in (engines or ["onnxruntime", "ncnn"]):
        try:
            import pnnx
            pnnx.export(wrapper, str(output_dir/"network.pt"),
                        inputs=torch.zeros(1, 40, 128),
                        inputs2=torch.zeros(1, 40, 231), fp16=False)
            if not all((output_dir/n).is_file() for n in ("network.ncnn.param", "network.ncnn.bin")):
                raise ValueError("ncnn conversion did not produce both graph files")
            status["ncnn"] = "exported_unverified"
        except Exception as error:
            status["ncnn"] = "blocked: "+type(error).__name__+": "+str(error)
    files = ["contract.json", "catalog.json", "normalization.json", "policy.json",
             "reference.pt", "model.onnx"]
    if status.get("ncnn") == "exported_unverified":
        files += ["network.ncnn.param", "network.ncnn.bin"]
    if memory is not None:
        write_new(output_dir/"entities.json", memory)
        files.append("entities.json")
    sources = ["vcm/frontend.py", "scripts/runtime/pcv_schema.py",
               "scripts/models/vcm_intent_slots.py",
               "scripts/runtime/vcm_intent_slots_bindings.py",
               "scripts/runtime/vcm_intent_slots_contract.py",
               "scripts/runtime/vcm_intent_slots_entities.py",
               "scripts/runtime/vcm_intent_slots_decode.py",
               "scripts/runtime/vcm_intent_slots_runtime.py",
               "scripts/runtime/pcv_benchmark_adapter.py"]
    recipe = {
        "format": "vcm-intent-slots-v1", "contract_hash": contract["contract_hash"],
        "head_order": names, "value_offsets": contract["value_offsets"],
        "files": {n: sha(output_dir/n) for n in files},
        "source_bindings": {n: sha(root/n) for n in sources},
        "engine_status": status, "rewrite_parity": rewrite_errors,
        "checkpoint_source_sha256": sha(checkpoint), "encoder_id": sha(checkpoint),
        "checkpoint_status": data.get("status", "unqualified_candidate"),
        "parameters": model.parameter_counts(),
        "graph_bytes": (output_dir/"model.onnx").stat().st_size,
        "installation_allowed": False, "provider_actions_allowed": False,
        "acoustic_qualified": False,
    }
    write_new(output_dir/"recipe.json", recipe)
    return recipe
