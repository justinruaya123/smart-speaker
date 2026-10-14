"""Hash-bound IntentSlots Pi handoff using the existing experimental WAV worker."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "vcm-intent-slots-pi-handoff-v1"
BENCHMARK_REVISION = "ab39857cf73dd067d068fed388c321ab160dc191"
SOURCE_FILES = (
    "scripts/runtime/speaker_worker.py", "scripts/runtime/speaker_config.py",
    "scripts/runtime/pcv_final_runtime.py", "scripts/runtime/speaker_confidence.py",
    "scripts/runtime/pcv_schema.py", "scripts/runtime/pcv_benchmark_adapter.py",
    "scripts/runtime/vcm_intent_slots_runtime.py", "scripts/runtime/vcm_intent_slots_bindings.py",
    "scripts/runtime/vcm_intent_slots_contract.py", "scripts/runtime/vcm_intent_slots_entities.py",
    "scripts/runtime/vcm_intent_slots_decode.py", "vcm/frontend.py", "vcm/catalog.py",
    "final/pi_finalize.py", "final/package_pi.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or name != PurePosixPath(name).as_posix()
            or any(p in (".", "..") for p in name.split("/"))):
        raise ValueError("Unsafe package path: " + str(name))
    return name


def source_bytes(root, name):
    safe_name(name)
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Source path escapes root: " + name)
    return path.read_bytes()


def decoded(data):
    def invalid(value):
        raise ValueError("Nonfinite JSON evidence: " + value)
    return json.loads(data, parse_constant=invalid)


def extra_files(extra, root):
    if isinstance(extra, dict):
        return list(extra.items())
    result = []
    for item in extra:
        text = str(item)
        if "=" in text:
            name, path = text.split("=", 1)
        else:
            path = Path(text).resolve()
            name = path.relative_to(root).as_posix() if path.is_relative_to(root) else "extras/" + path.name
        result.append((name, Path(path)))
    return result


def package(bundle, card, output, *, root=ROOT, extra=()):
    root, bundle, card, output = (Path(p).resolve() for p in (root, bundle, card, output))
    if output.exists():
        raise FileExistsError("ZIP outputs are immutable: " + str(output))
    recipe_bytes = (bundle / "recipe.json").read_bytes()
    recipe = decoded(recipe_bytes)
    if (recipe.get("format") != "vcm-intent-slots-v1"
            or recipe.get("head_order") != ["intent", "role_states", "validity", "values", "entities"]):
        raise ValueError("Requires the selected IntentSlots recipe")
    checkpoint = recipe.get("checkpoint_source_sha256")
    if (not isinstance(checkpoint, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint)
            or recipe.get("encoder_id") != checkpoint
            or "fixture" in recipe.get("checkpoint_status", "").lower()):
        raise ValueError("Missing actual selected checkpoint binding")
    required = {"contract.json", "catalog.json", "normalization.json", "policy.json", "model.onnx"}
    files = recipe.get("files", {})
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Incomplete candidate file bindings")
    entries = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in candidate bundle")
        if path.is_file():
            name = path.relative_to(bundle).as_posix()
            safe_name(name)
            entries["models/selected/" + name] = path.read_bytes()
    if entries.get("models/selected/recipe.json") != recipe_bytes:
        raise ValueError("Candidate recipe changed during packaging")
    for name, expected in files.items():
        safe_name(name)
        payload = entries.get("models/selected/" + name)
        if payload is None or digest(payload) != expected:
            raise ValueError("Candidate file hash binding mismatch: " + name)
    if files.get("reference.pt") != checkpoint:
        raise ValueError("Selected checkpoint identity does not match preserved reference.pt bytes")
    sources = recipe.get("source_bindings", {})
    if not isinstance(sources, dict) or not {"vcm/frontend.py", "scripts/runtime/vcm_intent_slots_runtime.py"} <= set(sources):
        raise ValueError("Incomplete runtime source bindings")
    for name, expected in sources.items():
        payload = source_bytes(root, name)
        if digest(payload) != expected:
            raise ValueError("Runtime source binding mismatch: " + name)
        entries[name] = payload
    contract = decoded(entries["models/selected/contract.json"])
    if (contract.get("contract_hash") != recipe.get("contract_hash")
            or contract.get("thresholds") != {"intent": 0.95, "slot": 0.90}):
        raise ValueError("Contract identity or fixed threshold binding mismatch")
    card_bytes = card.read_bytes()
    baseline = decoded(card_bytes)
    binding = {"checkpoint_sha256": checkpoint, "recipe_sha256": digest(recipe_bytes),
               "contract_hash": recipe["contract_hash"]}
    if (baseline.get("format") != "vcm-model-card-v2"
            or any(baseline.get("bindings", {}).get(key) != value for key, value in binding.items())):
        raise ValueError("Semi-final card candidate binding mismatch")
    for name in SOURCE_FILES:
        if name not in entries:
            entries[name] = source_bytes(root, name)
    for name in ("contract.py", "decoder.py"):
        entries["pcvcm/" + name] = source_bytes(root, "experiments/phrase-contract-v1/pcvcm/" + name)
    for name in ("scripts/__init__.py", "scripts/runtime/__init__.py", "vcm/__init__.py", "pcvcm/__init__.py"):
        entries[name] = source_bytes(root, name) if (root / name).is_file() else b""
    entries["requirements.lock"] = source_bytes(root, "requirements.lock")
    for name in ("pyproject.toml", "uv.lock"):
        entries[name] = source_bytes(root, "scripts/runtime/" + name)
    entries["licenses/ncnn-LICENSE.txt"] = source_bytes(root, "scripts/runtime/ncnn-LICENSE.txt")
    for path in sorted((root / "data/sources").glob("*")):
        if path.is_file() and path.name.endswith(("LICENSE.txt", "NOTICE.txt")):
            entries["licenses/" + path.name] = source_bytes(root, path.relative_to(root).as_posix())
    entries["final/card.json"] = card_bytes
    if (root / "final/MODEL_CARD.md").is_file():
        entries["final/MODEL_CARD.md"] = source_bytes(root, "final/MODEL_CARD.md")
    else:
        entries["final/MODEL_CARD.md"] = ("# Semi-final VCM model card\n\n```json\n"
            + json.dumps(baseline, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n").encode()
    protocol = "experiments/vcm-intent-slots-v1/protocol.json"
    if (root / protocol).is_file():
        entries[protocol] = source_bytes(root, protocol)
    ncnn_available=(recipe.get('engine_status',{}).get('ncnn')=='exported_unverified'
                    and {'network.ncnn.param','network.ncnn.bin'}<=set(files))
    engine='ncnn' if ncnn_available else 'onnxruntime'
    entries["README-Pi.md"] = README.replace('--engine ncnn --threads 1',
                                             '--engine '+engine+' --threads 1').encode("utf-8")
    for name, path in extra_files(extra, root):
        safe_name(name)
        if name == "MANIFEST.json" or name in entries:
            raise ValueError("Extra path collision: " + name)
        entries[name] = Path(path).read_bytes()
    if len({name.casefold() for name in entries}) != len(entries):
        raise ValueError("Case-colliding package paths")
    manifest = {"format": FORMAT, "files": {name: digest(data) for name, data in sorted(entries.items())},
                "model_binding": binding, "physical_pi_measured": False, "actions_enabled": False,
                "benchmark_required_revision": BENCHMARK_REVISION,
                "benchmark_note": "Pinned source/default/reference files, when supplied, retain their original bytes. No benchmark is run by packaging."}
    entries["MANIFEST.json"] = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    verify_package(output)
    return {"path": str(output), "sha256": digest(output.read_bytes()), "bytes": output.stat().st_size,
            "files": manifest["files"], "model_binding": binding, "physical_pi_measured": False}


def verify_package(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError("Duplicate or case-colliding package members")
        for info in archive.infolist():
            safe_name(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlink in package")
        manifest = decoded(archive.read("MANIFEST.json"))
        if manifest.get("format") != FORMAT or set(names) != set(manifest.get("files", {})) | {"MANIFEST.json"}:
            raise ValueError("Package format/inventory mismatch")
        for name, expected in manifest["files"].items():
            if digest(archive.read(name)) != expected:
                raise ValueError("Package checksum mismatch: " + name)
    return manifest


README = """# Pi benchmark handoff

Physical Pi results are deferred. This package preserves the selected model, calibration, entity enrollment, references and semi-final card. Packaging does not run inference or certify accuracy.

Use the packaged runtime lock (Python 3.11–3.13):

```sh
uv sync --locked --no-dev --no-build
.venv/bin/python -m scripts.runtime.speaker_worker --experimental-intent-slots --bundle models/selected --engine ncnn --threads 1
```

This is the existing resident JSONL WAV service. Send `{"op":"wav","id":"trial-1","data":"BASE64_WAV_BYTES"}`, then `{"op":"status","id":"status"}` or `{"op":"shutdown"}`. WAVs must be mono 16 kHz. A one-shot WAV uses the same command with `--wav command.wav`. The command service executes no actions.

The existing engines are ncnn CPU and ONNX Runtime CPU (`--engine onnxruntime`). This package adds no backend, affinity, memory cap or hardware qualification. Keep the mandatory vcm-benchmark source/defaults at revision ab39857cf73dd067d068fed388c321ab160dc191 and leave all supplied references unchanged. Replay and live microphone evidence remain separate; command-only service traffic does not certify wake, owner or microphone capture.

Verify ZIP bytes without loading a model:

```sh
python final/package_pi.py --verify /path/to/the-handoff.zip
```

After the actual Pi benchmark, write an actual JSON receipt with `kind="physical_pi_benchmark"`, `bindings` matching the semi-final card's checkpoint_sha256, recipe_sha256 and contract_hash, nonempty actual `hardware` and `environment`, `metrics` with trials, supported_n, negative_n, correct, false_accepts, faults and missing integer counts, and explicit boolean measured `gates`. Positive and negative denominators must be positive; an optional ambiguous_n remains separate. Include additional actual slot/timing/resource metrics without dropping failures or missing trials.

```sh
python final/pi_finalize.py --baseline final/card.json --receipt results/pi-receipt.json --output results
```

The updater preserves the frozen packaged card and writes the derived card to results/card.json and results/MODEL_CARD.md. It copies all non-Pi findings and failed verbatims, hash-binds the receipt, and only advances the overall status when non-Pi accuracy gates were already ready. Failed Pi gates, faults or missing trials remain failed. No threshold, enrollment, selection or training change is performed.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--card", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extra", action="append", default=[], metavar="ARCHIVE_PATH=LOCAL_PATH")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        if any(v is not None for v in (args.bundle, args.card, args.output)) or args.extra:
            parser.error("--verify is separate from packaging")
        manifest = verify_package(args.verify)
        print(json.dumps({"status": "package_hashes_verified", "files": len(manifest["files"])}))
    else:
        if any(v is None for v in (args.bundle, args.card, args.output)):
            parser.error("Packaging requires --bundle, --card and --output")
        receipt = package(args.bundle, args.card, args.output, extra=args.extra)
        print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, allow_nan=False))


if __name__ == "__main__":
    main()
