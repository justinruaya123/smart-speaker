"""Export trained checkpoints; quantization ranges use training audio only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch
from einops import rearrange

from .audio import DEFAULT, cached_features
from .data import ROOT, read_jsonl, write_json, stable_number
from .vcm import VCM, StreamingExport


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def normalized(row, stats):
    mel = cached_features(row, ROOT)
    mean = rearrange(np.asarray(stats["mean"], dtype=np.float32), "m -> m 1")
    std = rearrange(np.asarray(stats["std"], dtype=np.float32), "m -> m 1")
    return rearrange((mel - mean) / std, "m t -> 1 m t")


class Reader(CalibrationDataReader):
    def __init__(self, inputs):
        self.inputs = iter(inputs)

    def get_next(self):
        return next(self.inputs, None)


def quantize(source, target, feeds):
    prepared = source.with_name(source.stem + ".prepared.onnx")
    quant_pre_process(str(source), str(prepared), skip_optimization=True, skip_symbolic_shape=True)
    quantize_static(str(prepared), str(target), Reader(feeds), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    prepared.unlink()


def choose_vcm():
    candidates = []
    for name in ("vcm_seed17", "vcm_context_seed17"):
        run = ROOT / "runs" / name
        if not (run / "complete.json").exists():
            raise RuntimeError(f"Candidate training incomplete: {name}")
        history = read_jsonl(run / "history.jsonl")
        best = max(history, key=lambda h: h["validation"]["macro_f1"])
        candidates.append({"run": name, "epoch": best["epoch"], **{k: v for k, v in best["validation"].items() if k != "confusion"}})
    selected = max(candidates, key=lambda c: c["macro_f1"])
    write_json(ROOT / "reports/model_selection.json", {"criterion": "highest development macro F1; test never used", "candidates": candidates, "selected": selected})
    return ROOT / "runs" / selected["run"] / "best.pt"


def training_calibration(task, count=200):
    rows = [r for r in read_jsonl(ROOT / f"data/manifests/{task}.jsonl") if r["split"] == "train"]
    if task == "asr":
        rows = [r for r in rows if r["source"] == "SLURP-real"]
    # Deterministic round-robin through VCM classes; no final-test or threshold-calibration rows.
    groups = {}
    for row in sorted(rows, key=lambda r: stable_number(r["id"])):
        groups.setdefault(row.get("command", "speech"), []).append(row)
    selected = []
    while groups and len(selected) < count:
        for key in sorted(list(groups)):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == count:
                break
    return selected


def file_facts(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["vcm", "asr"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--bundle", default="artifacts/tinyvcm")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    bundle = ROOT / args.bundle
    bundle.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else choose_vcm() if args.task == "vcm" else ROOT / "runs/asr_commands_seed17/best.pt"
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.task == "vcm":
        model = VCM(**ck["model_config"])
    else:
        from .asr import ParameterASR
        model = ParameterASR(**ck["model_config"])
    model.load_state_dict(ck["model"])
    model.eval()
    rows = training_calibration(args.task)
    feeds = []
    for row in rows:
        x = normalized(row, ck["normalization"])
        feed = {"features": x}
        if args.task == "vcm":
            feed["lengths"] = np.array([x.shape[-1]], dtype=np.int64)
        feeds.append(feed)
    dummy = torch.from_numpy(feeds[0]["features"])
    fp = bundle / f"{args.task}.fp32.onnx"
    inputs = (dummy, torch.tensor([dummy.shape[-1]])) if args.task == "vcm" else (dummy,)
    names = ["features", "lengths"] if args.task == "vcm" else ["features"]
    axes = {"features": {0: "batch", 2: "frames"}, "logits": {0: "batch"} if args.task == "vcm" else {0: "output_frames", 1: "batch"}}
    if args.task == "vcm":
        axes["lengths"] = {0: "batch"}
    with torch.inference_mode():
        torch.onnx.export(model, inputs, str(fp), input_names=names, output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(fp))
    runtime = session(fp)
    errors = []
    for feed in feeds[:12]:
        with torch.inference_mode():
            tensors = [torch.from_numpy(feed[name]) for name in names]
            expected = model(*tensors).numpy()
        actual = runtime.run(None, feed)[0]
        errors.append(float(np.max(np.abs(expected - actual))))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-4)
    print(f"EXPORT {args.task} FP32 parity max error={max(errors):.8f}", flush=True)
    quantize(fp, bundle / f"{args.task}.int8.onnx", feeds)
    info = {"checkpoint": str(checkpoint_path.relative_to(ROOT) if checkpoint_path.is_relative_to(ROOT) else checkpoint_path), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "epoch": ck["epoch"], "model_config": ck["model_config"], "parameters": sum(p.numel() for p in model.parameters()), "normalization": ck["normalization"], "quantization": {"format": "QDQ S8S8 per-channel Conv/Gemm/MatMul; remaining operations float", "calibration_split": "train", "calibration_ids": [r["id"] for r in rows]}, "fp32_torch_onnx_max_error": max(errors)}
    if args.task == "vcm":
        states = model.initial_state()
        info.update(labels=ck["labels"], state_shapes=[list(s.shape) for s in states], state_initial_values=[float(s[tuple(0 for _ in s.shape)]) for s in states])
        stream_path = bundle / "vcm_stream.fp32.onnx"
        input_names = ["features", *[f"state_{i}" for i in range(len(states))]]
        output_names = ["logits", *[f"next_{i}" for i in range(len(states))]]
        with torch.inference_mode():
            torch.onnx.export(StreamingExport(model), (dummy[:, :, :4], *states), str(stream_path), input_names=input_names, output_names=output_names, dynamic_axes={"features": {2: "chunk_frames"}}, opset_version=17, dynamo=False)
        onnx.checker.check_model(str(stream_path))
        stream = session(stream_path)
        stream_feeds = []
        stream_errors = []
        for row in rows[:40]:
            x = normalized(row, ck["normalization"])
            current = [s.numpy().copy() for s in model.initial_state()]
            capture = {0, (x.shape[-1] // 4 // 3) * 4, (x.shape[-1] // 4 * 2 // 3) * 4, ((x.shape[-1] - 1) // 4) * 4}
            for frame in range(0, x.shape[-1], 4):
                feed = {"features": x[:, :, frame:frame + 4], **{f"state_{i}": s for i, s in enumerate(current)}}
                if frame in capture:
                    stream_feeds.append(feed)
                logits, *current = stream.run(None, feed)
            final = runtime.run(None, {"features": x, "lengths": np.array([x.shape[-1]], dtype=np.int64)})[0]
            stream_errors.append(float(np.max(np.abs(logits - final))))
            np.testing.assert_allclose(logits, final, atol=3e-4, rtol=2e-4)
        info["fp32_stream_offline_max_error"] = max(stream_errors)
        info["state_bytes"] = sum(s.numel() * s.element_size() for s in states)
        quantize(stream_path, bundle / "vcm_stream.int8.onnx", stream_feeds)
    metadata_path = bundle / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"schema_version": 1, "frontend": DEFAULT.__dict__, "catalog": {"entities": []}, "routing": {"acceptance_threshold": 0.7, "trigger_threshold": 0.9, "stability_chunks": 3, "minimum_audio_seconds": 0.5, "lookback_seconds": 0.5}, "hardware_validated": [], "boundary_supervision": False}
    metadata[args.task] = info
    metadata["files"] = {p.name: file_facts(p) for p in sorted(bundle.glob("*.onnx"))}
    metadata["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(metadata_path, metadata)
    write_json(ROOT / f"reports/export_{args.task}.json", {"task": args.task, **info, "files": metadata["files"]})
    print(f"EXPORTED {args.task} to {bundle}", flush=True)


if __name__ == "__main__":
    main()
