"""Continue the current checkpoint; train/dev only, with explicit artifact replacement."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from .train_helpers import seed_everything


def subset(rows, split, maximum):
    selected = [r for r in rows if r["split"] == split]
    return selected[:maximum] if maximum else selected


def fit_vcm(config, root, output):
    from .candidate import fit_candidate
    return fit_candidate(config, root, output, config["training"].get("encoder", "finetune"))




def export_graphs(output, selected, metadata):
    import onnx
    import onnxruntime as ort
    from .slot_model import SlotVCM
    if set(selected) != {'vcm'}:
        raise ValueError('This exporter handles VCM; parameter models own their separate exporters')
    errors = {}
    for task in selected:
        ck = torch.load(output / f"{task}.pt", map_location="cpu", weights_only=False)
        model = SlotVCM(**ck["config"]).eval()
        model.load_state_dict(ck["model"])
        x, lengths = torch.zeros(1, 40, 160), torch.tensor([160])
        path = output / f"{task}.fp32.onnx"
        names = ["logits", *["slot_" + name for name in ck["config"]["vocabulary"]]]
        axes = {"features": {0: "batch", 2: "frames"}, "logits": {0: "batch"},
                'lengths':{0:'batch'}, **{name:{0:'batch'} for name in names[1:]}}
        torch.onnx.export(model, (x,lengths), str(path), input_names=["features", "lengths"],
                          output_names=names, dynamic_axes=axes, opset_version=17, dynamo=False)
        graph = onnx.load(str(path))
        for key, value in (("vcm_labels", ck["labels"]), ("slot_vocabulary", ck["config"]["vocabulary"])):
            entry = graph.metadata_props.add(); entry.key = key; entry.value = json.dumps(value)
        onnx.checker.check_model(graph); onnx.save(graph, str(path))
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        maximum = 0.
        for frames in (61, 160, 317):
            x, lengths = torch.randn(1, 40, frames), torch.tensor([frames])
            with torch.inference_mode():
                expected = model(x, lengths)
            actual = session.run(None, {"features": x.numpy(), "lengths": lengths.numpy()})
            for left, right in zip(expected, actual):
                np.testing.assert_allclose(left.numpy(), right, atol=2e-4, rtol=2e-4)
                maximum = max(maximum, float(np.max(np.abs(left.numpy() - right))))
        errors[task] = maximum
        metadata["files"][path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        metadata[task]["checkpoint"] = f"{task}.pt"
        metadata[task]["checkpoint_sha256"] = hashlib.sha256((output / f"{task}.pt").read_bytes()).hexdigest()
    (output / "export-verification.json").write_text(json.dumps({
        "torch_onnx_max_abs_error": errors, "frames_checked": [61, 160, 317],
        "files": metadata["files"], "physical_hardware_tested": False}, indent=2) + "\n")
    return errors


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
        if path.is_file() and path.name not in {"family-metrics.json", "pilot-results.json", "training-history.json", "export-verification.json"}:
            shutil.copy2(path, output / path.name)
    metadata = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    result = {"selected": selected, "output": str(output.relative_to(root)),
              "continuation_from_current_weights": True, "calibration_or_test_used_for_fitting": False,
              "scope": "balanced VCM family/finite-head training"}
    if "vcm" in selected:
        metadata, result["vcm"] = fit_vcm(config, root, output)
    result["export_max_errors"] = export_graphs(output, selected, metadata)
    metadata["experiment"].update(status="experimental_continuation", training=result)
    (output / "bundle.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if "vcm" in selected:
        from .improvement import development_check
        result["deployment_development"] = development_check(output, root, refit=True,
            mode=config["training"].get("encoder", "finetune"))
    from .validation import run_checks
    test_config = copy.deepcopy(config); test_config["tests"] = {key: True for key in config["tests"]}
    result["checks"] = run_checks(test_config, output)
    quality_passed = True
    if "vcm" in selected:
        development = result["deployment_development"]
        gates = development["intent"]["gates"]
        quality_passed = (gates["inherited_pass"] and not gates["additional_low_per_intent_precision"]
                          and development["commands"]["accepted_command_precision"] >= .95
                          and development["commands"]["exact_command_success"] >= .6)
    promote = cfg["update_artifacts"] and quality_passed
    result["development_quality_gate_passed"] = quality_passed
    if not quality_passed:
        result["promotion_blocked"] = "Per-family recognition/precision or complete-command development gates failed; candidate retained in cache."
    if promote:
        # Keep a single previous-current recovery copy until the next successful fit.
        backup = parent / "previous-current"
        backup.mkdir(exist_ok=True)
        for path in (root / "artifacts").iterdir():
            if path.is_file():
                shutil.copy2(path, backup / path.name)
        for path in output.iterdir():
            if path.is_file() and path.name != "manifest.json":
                shutil.copy2(path, root / "artifacts" / path.name)
    result["artifacts_updated"] = promote
    return result
