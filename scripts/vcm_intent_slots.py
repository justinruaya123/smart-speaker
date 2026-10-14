"""Thin explicit experiment pipeline. Never installs models or dispatches actions."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime.pcv_schema import DEFAULT_CUSTOM, baseline_cases, extended_cases
from scripts.runtime.vcm_intent_slots_contract import build_slot_contract
from scripts.train.vcm_intent_slots_data import audit_views, rows, sha, write_new

CACHE = ROOT/".cache/vcm-intent-slots-v1"


def input_bindings(path):
    inputs = json.loads(Path(path).read_text())
    for group in ("views",):
        for name, binding in inputs[group].items():
            if sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Input binding mismatch: "+name)
    receipt = inputs["preparation_receipt"]
    if sha(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Preparation receipt changed")
    for path, expected in inputs["source_receipts"].items():
        if sha(path) != expected:
            raise ValueError("Source receipt changed")
    catalog = inputs["catalog"]
    if sha(ROOT/catalog["path"]) != catalog["sha256"]:
        raise ValueError("Selected catalog changed")
    metadata = inputs.get("speaker_metadata")
    if metadata:
        if sha(metadata["path"]) != metadata["sha256"]:
            raise ValueError("Speaker metadata binding mismatch")
        source = metadata["source_metadata"]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Original speaker metadata binding mismatch")
    return inputs


def load_contract(inputs):
    path = ROOT/inputs["catalog"]["path"]
    catalog = json.loads(path.read_text())
    return build_slot_contract(baseline_cases()+extended_cases(), catalog["entries"], DEFAULT_CUSTOM)


def preflight(inputs, protocol, output):
    versions = {}
    for name in ("torch", "onnx", "onnxruntime", "ncnn", "pnnx", "numpy", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import torch
    contract = load_contract(inputs)
    audit = audit_views({s: v["path"] for s, v in inputs["views"].items()},
                        {"views": inputs["views"]}, output/"data-audit")
    missing = [f"{'/'.join(audit['missing_intents'][s])} human {s}"
               for s in ("train", "dev", "calibration") if audit["missing_intents"][s]]
    missing += ["independently reviewed role states, timer/alarm grammar labels and role spans",
               "3..5 original enrollment recordings per named entity with disjoint queries",
               "genuinely fresh person-disjoint holdout", "verified same-person wake/command pairs",
               "separate trained/calibrated identity, overlap and continuity backend",
               "physical Raspberry Pi 4/5 and approved numerical resource limits"]
    if not inputs.get("shipping_bundle"):
        missing.append("exact publication-v2-shipping checkpoint/ONNX/ncnn payloads and crop spans")
    result = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
              "design_commit": "ede9d3caa036876cb112907b8d85e239b799c618",
              "python": sys.version, "interpreter": sys.executable,
              "platform": platform.platform(), "versions": versions,
              "cuda_available": torch.cuda.is_available(), "inputs": inputs,
              "catalog_entries": len(contract["catalog_entries"]), "case_definitions": len(contract["cases"]),
              "contract_hash": contract["contract_hash"], "blockers": missing,
              "training_update_artifacts": protocol["training"]["update_artifacts"],
              "installation_allowed": False, "provider_actions_allowed": False}
    write_new(output/"preflight.json", result)
    write_new(output/"slot-contract.json", contract)
    return result


def run(args):
    protocol = json.loads(args.protocol.read_text())
    if protocol["training"]["update_artifacts"] or protocol["installation_allowed"] or protocol["provider_actions_allowed"]:
        raise ValueError("Experiment cannot install artifacts or dispatch actions")
    inputs = input_bindings(args.inputs)
    contract = load_contract(inputs)
    output = CACHE/args.run_id
    if args.command == "preflight":
        return preflight(inputs, protocol, output)
    if args.command == "audit":
        return audit_views({s:v["path"] for s,v in inputs["views"].items()},
                           {"views":inputs["views"]}, output,
                           audio_root=inputs["audio_root"] if args.check_audio else None)
    if args.command == "diversity":
        from scripts.test.vcm_intent_slots_diversity import audit_diversity
        binding = inputs.get("speaker_metadata")
        metadata = json.loads(Path(binding["path"]).read_text()) if binding else None
        result = audit_diversity({s: rows(v["path"]) for s, v in inputs["views"].items()},
                                 speaker_metadata=metadata)
        result["bindings"] = {"views": inputs["views"], "inputs_sha256": sha(args.inputs),
                              "source_receipts": inputs["source_receipts"],
                              "speaker_metadata": binding}
        write_new(output/"diversity.json", result)
        return result
    if args.command == "train":
        from scripts.train.vcm_intent_slots_train import fit_run
        results = []
        for seed in args.seeds:
            try:
                r = fit_run(protocol, inputs, contract, output/f"{args.arm}-{seed}",
                            arm=args.arm, seed=seed, device=args.device)
            except ValueError as error:
                r = {"arm": args.arm, "seed": seed, "status": "blocked_not_fitted", "reason": str(error)}
            results.append(r)
        write_new(output/"grid.json", results)
        if any(r["status"] == "blocked_not_fitted" for r in results):
            raise ValueError("Grid blocked; exact reasons preserved in "+str(output/"grid.json"))
        return results
    if args.command == "diagnose":
        from scripts.test.vcm_intent_slots_diagnose import diagnose
        return diagnose(inputs, rows(inputs["views"]["dev"]["path"]), output)
    if args.command == "export":
        if args.checkpoint is None:
            raise ValueError("Missing selected checkpoint; use --checkpoint")
        from scripts.train.vcm_intent_slots_export import export_candidate
        return export_candidate(args.checkpoint, contract, output/"bundle", engines=args.engines,
                                calibration=args.calibration, memory=args.memory,
                                entity_policy=args.entity_policy)
    if args.command in ("evaluate", "calibrate", "parity", "owner-replay", "lock"):
        if args.bundle is None:
            raise ValueError("Missing explicit experimental bundle; use --bundle")
        from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime
        if args.command == "parity":
            from scripts.test.vcm_intent_slots_parity import verify_parity
            # Original human dev recordings; selected by ID order, never errors.
            dev = rows(inputs["views"]["dev"]["path"])
            dev = sorted(dev, key=lambda r:(r["samples"], r["id"]))
            probe = dev[:5]+dev[len(dev)//2:len(dev)//2+5]+dev[-5:]
            return verify_parity(args.bundle, output/"parity.json", recordings=probe,
                                 audio_root=inputs["audio_root"], engines=args.engines)
        runtime = IntentSlotsRuntime(args.bundle, args.engine, 1)
        if args.command == "lock":
            from scripts.test.vcm_intent_slots_eval import freeze_selection, verify_evaluation_context
            files = {"protocol": args.protocol, "manifest": inputs["views"][args.split]["path"],
                     "checkpoint": args.bundle/"reference.pt", "contract": args.bundle/"contract.json",
                     "catalog": args.bundle/"catalog.json", "normalization": args.bundle/"normalization.json",
                     "calibration": args.bundle/"policy.json", "frontend": ROOT/"vcm/frontend.py",
                     "runtime": args.bundle/"recipe.json", "inputs": args.inputs}
            # Validate all context before publishing the immutable selection lock.
            context = {"files": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()}}
            verify_evaluation_context(context, runtime, inputs["views"][args.split]["path"], args.protocol)
            return freeze_selection(files, output/"selection-before-test.json")
        if args.command == "calibrate":
            from scripts.train.vcm_intent_slots_calibration import collect_calibration, fit_score_calibration
            observations = collect_calibration(rows(inputs["views"]["calibration"]["path"]),
                                               runtime, inputs["audio_root"])
            binding = {"contract_hash": contract["contract_hash"],
                       "checkpoint": runtime.recipe["checkpoint_source_sha256"],
                       "normalization": runtime.recipe["files"]["normalization.json"],
                       "frontend": runtime.recipe["source_bindings"]["vcm/frontend.py"],
                       "entities": runtime.policy.get("entities", {}),
                       "manifest": inputs["views"]["calibration"]["sha256"]}
            return fit_score_calibration(observations, binding, output)
        if args.command == "evaluate":
            from scripts.test.vcm_intent_slots_eval import replay
            return replay(rows(inputs["views"][args.split]["path"]), runtime, inputs["audio_root"],
                          output, split=args.split, selection=args.selection, regression=args.regression,
                          manifest=inputs["views"][args.split]["path"], protocol=args.protocol)
        if not inputs.get("owner_pairs"):
            raise ValueError("Missing genuine human wake/command pairs, calibration and separate validated owner backend")
        from scripts.test.vcm_intent_slots_owner_replay import replay_owner
        binding = inputs["owner_pairs"]
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Owner-pair binding mismatch")
        return replay_owner(json.loads(Path(binding["path"]).read_text()), runtime, inputs["audio_root"], output)
    if args.command == "summarize":
        return {"status": "implemented_unqualified", "coverage": json.loads(
            (ROOT/"experiments/vcm-intent-slots-v1/coverage.json").read_text()),
            "receipts": sorted(str(p.relative_to(ROOT)) for p in CACHE.glob("*/blocked.json")),
            "installed_artifacts_changed": False, "actions_enabled": False}
    raise ValueError("Unknown pipeline stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "audit", "diagnose", "train", "calibrate",
                                          "evaluate", "export", "parity", "owner-replay", "summarize", "lock", "diversity"])
    parser.add_argument("--protocol", type=Path, default=ROOT/"experiments/vcm-intent-slots-v1/protocol.json")
    parser.add_argument("--inputs", type=Path, default=ROOT/"experiments/vcm-intent-slots-v1/local-inputs.json")
    parser.add_argument("--run-id", default="run-01")
    parser.add_argument("--arm", choices=["uniform", "balanced"], default="uniform")
    parser.add_argument("--seeds", type=int, nargs="+", default=[53, 71, 89])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--entity-policy", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--engine", choices=["pytorch", "onnxruntime", "ncnn"], default="ncnn")
    parser.add_argument("--engines", nargs="+", choices=["onnxruntime", "ncnn"], default=["onnxruntime", "ncnn"])
    parser.add_argument("--split", choices=["dev", "calibration", "test", "holdout"], default="dev")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--regression", action="store_true", help="Explicit previously examined test evidence")
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args()
    if not args.run_id or "/" in args.run_id or "\\" in args.run_id or args.run_id in (".", ".."):
        parser.error("Use a simple unique run ID")
    try:
        result = run(args)
        print(json.dumps(result, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        output = CACHE/args.run_id
        blocked = {"command": args.command, "status": "blocked", "reason": str(error),
                   "actions_enabled": False, "installed_artifacts_changed": False}
        try:
            write_new(output/"blocked.json", blocked)
        except FileExistsError:
            pass
        print(json.dumps(blocked), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
