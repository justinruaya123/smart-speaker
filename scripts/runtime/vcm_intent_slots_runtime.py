"""Explicit experimental engines and one action-free final application verdict."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from vcm.frontend import features
from scripts.runtime.vcm_intent_slots_contract import verify_contract
from scripts.runtime.vcm_intent_slots_decode import decide, prediction_from_tensors
from scripts.runtime.pcv_benchmark_adapter import log_record


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate_pcm(pcm, sample_rate):
    pcm = np.asarray(pcm, dtype=np.float32)
    if (sample_rate != 16000 or pcm.ndim != 1 or not 400 <= len(pcm) <= 128000
            or not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1):
        raise ValueError("Requires finite mono 16 kHz PCM, 25 ms..8 s, amplitude <=1")
    return pcm


class IntentSlotsRuntime:
    action_free = True

    def __init__(self, bundle, engine, threads=1):
        self.root, self.engine = Path(bundle).resolve(), engine
        recipe = json.loads((self.root/"recipe.json").read_text())
        if recipe.get("format") != "vcm-intent-slots-v1" or recipe.get("head_order") != [
                "intent", "role_states", "validity", "values", "entities"]:
            raise ValueError("Wrong experimental head layout")
        required = {"contract.json", "catalog.json", "normalization.json", "policy.json", "model.onnx"}
        if engine == "pytorch":
            required.add("reference.pt")
        if engine == "ncnn":
            required.update(("network.ncnn.param", "network.ncnn.bin"))
        if (self.root/"entities.json").exists():
            required.add("entities.json")
        if not required <= set(recipe.get("files", {})):
            raise ValueError("Missing bundle bindings")
        for name, expected in recipe["files"].items():
            path = (self.root/name).resolve()
            if not path.is_relative_to(self.root) or file_hash(path) != expected:
                raise ValueError("Bundle file hash binding mismatch: " + name)
        root = Path(__file__).resolve().parents[2]
        for name, expected in recipe.get("source_bindings", {}).items():
            path = (root/name).resolve()
            if not path.is_relative_to(root) or file_hash(path) != expected:
                raise ValueError("Runtime source binding mismatch: " + name)
        self.recipe = recipe
        self.contract = json.loads((self.root/"contract.json").read_text())
        verify_contract(self.contract)
        if self.contract["contract_hash"] != recipe["contract_hash"]:
            raise ValueError("Mixed contract binding")
        self.catalog = json.loads((self.root/"catalog.json").read_text())
        self.policy = json.loads((self.root/"policy.json").read_text())
        self.memory = json.loads((self.root/"entities.json").read_text()) if (self.root/"entities.json").exists() else None
        from scripts.runtime.vcm_intent_slots_bindings import verify_evidence_bindings
        if recipe.get("encoder_id") != recipe["checkpoint_source_sha256"]:
            raise ValueError("Missing actual command encoder identity")
        verify_evidence_bindings(self.contract, self.policy, self.memory,
                                 encoder=recipe["encoder_id"],
                                 frontend=recipe["source_bindings"]["vcm/frontend.py"],
                                 normalization=recipe["files"]["normalization.json"])
        norm = json.loads((self.root/"normalization.json").read_text())
        self.mean, self.std = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
        if (norm.get("fit_split") != "train" or self.mean.shape != (40,) or self.std.shape != (40,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0)):
            raise ValueError("Invalid training normalization")
        if engine == "onnxruntime":
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(self.root/"model.onnx"), options, providers=["CPUExecutionProvider"])
        elif engine == "ncnn":
            import ncnn
            if recipe["engine_status"].get("ncnn") != "exported_unverified":
                raise ValueError("No compatible ncnn export")
            self.ncnn = ncnn
            self.net = ncnn.Net()
            self.net.opt.num_threads = threads
            self.net.opt.use_vulkan_compute = False
            self.net.opt.use_fp16_packed = False
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
            self.net.opt.use_packing_layout = False
            if self.net.load_param(str(self.root/"network.ncnn.param")) != 0 or self.net.load_model(str(self.root/"network.ncnn.bin")) != 0:
                raise RuntimeError("ncnn experimental graph load failed")
        elif engine == "pytorch":
            import torch
            from scripts.models.vcm_intent_slots import IntentSlotsVCM
            from scripts.train.vcm_intent_slots_export import OrderedHeads
            torch.set_num_threads(threads)
            data = torch.load(self.root/"reference.pt", map_location="cpu", weights_only=False)
            model = IntentSlotsVCM(self.contract).eval()
            model.load_state_dict(data["weights"])
            self.torch, self.wrapper = torch, OrderedHeads(model).eval()
        else:
            raise ValueError("Explicit engine required; no fallback")

    def infer_features(self, x):
        if self.engine == "onnxruntime":
            outputs = self.session.run(None, {"features": np.ascontiguousarray(x[None], np.float32)})
        elif self.engine == "pytorch":
            with self.torch.no_grad():
                outputs = [v.numpy() for v in self.wrapper(self.torch.tensor(x[None]))]
        else:
            # Retain every buffer/Mat owner until all outputs have been copied.
            buffer = np.ascontiguousarray(x, dtype=np.float32)
            mat = self.ncnn.Mat(buffer).clone()
            outputs = []
            with self.net.create_extractor() as extractor:
                if extractor.input("in0", mat) != 0:
                    raise RuntimeError("ncnn feature input failed")
                for n in range(5):
                    code, value = extractor.extract(f"out{n}")
                    if code != 0:
                        raise RuntimeError("ncnn head extraction failed: " + str(n))
                    outputs.append(np.array(value, copy=True))
        sizes = [19, len(self.contract["roles"])*4, 4,
                 max(v[1] for v in self.contract["value_offsets"].values()), 5*64]
        arrays = []
        if len(outputs) != 5:
            raise ValueError("Corrupt experimental output count")
        for value, size in zip(outputs, sizes):
            if np.asarray(value).size != size or not np.isfinite(value).all():
                raise ValueError("Corrupt/nonfinite experimental graph output")
            arrays.append(np.asarray(value).reshape(-1))
        return {"intent_logits": arrays[0], "role_state_logits": arrays[1].reshape(-1, 4),
                "request_validity_logits": arrays[2],
                "value_logits": {r: arrays[3][a:b] for r, (a, b) in self.contract["value_offsets"].items()},
                "entity_embeddings": arrays[4].reshape(5, 64)}

    def recognize(self, pcm, sample_rate=16000):
        started = time.perf_counter()
        pcm = validate_pcm(pcm, sample_rate)
        x = features(pcm, self.mean, self.std)
        front = time.perf_counter()
        tensors = self.infer_features(x)
        modeled = time.perf_counter()
        pred = prediction_from_tensors(tensors, self.contract, self.memory, self.policy)
        application = decide(pred, self.contract, self.catalog, self.policy)
        done = time.perf_counter()
        event = {"event": "command", "output": application, "application": application,
                 "final_application": application, "actions_enabled": False,
                 "model_family": "vcm-intent-slots-v1", "audio_seconds": len(pcm)/16000,
                 "timing": {"frontend_ms": (front-started)*1000, "model_ms": (modeled-front)*1000,
                            "decision_ms": (done-modeled)*1000, "total_ms": (done-started)*1000}}
        event["benchmark"] = log_record(event, decision="final_application", custom=self.contract["custom"])
        return event
