"""Remove the independent transcription component from the known notebook.

Run on the research branch, never main. Acoustic VCM weights are not changed.
Historical recognition reports are not new validation of the modified pipeline.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

MARKER = "vcm-only-v1"


def replace(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Migration anchor missing: {old[:85]!r}")
    return source.replace(old, new)


def remove_function(source: str, name: str) -> str:
    lines = source.splitlines(keepends=True)
    matches = [n for n in ast.walk(ast.parse(source))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one function {name}")
    node = matches[0]
    return "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])


def cut(source: str, start: str, end: str, new: str = "") -> str:
    a = source.index(start)
    b = source.index(end, a)
    return source[:a] + new + source[b:]


EXPORT = '''def export_graphs(output, selected, metadata):
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
'''


def transform(name: str, source: str) -> str:
    if name == "inference":
        source = replace(source, "bundle, asr=False, threads=2", "bundle, threads=2")
        source = replace(source, "        self.asr_enabled = asr\n        self.asr = None\n", "")
        source = replace(source, "PARAMETER_COMMANDS, parameter_slice, parse_parameters", "PARAMETER_COMMANDS, parse_parameters")
        source = replace(source, "from .text import decode_greedy\n", "")
        for function in ("transcribe", "transcribe_features"):
            source = remove_function(source, function)
        source = cut(source, "            from .slot_contract import custom_media_eligibility", '        result["processing_ms"]',
            '            result.update(status="needs_parameters", parameter_source="vcm",\n'
            '                          reason="catalog_decoder_required" if command == "MEDIA_PLAY"\n'
            '                          else "finite_parameters_require_clarification")\n')
        source = replace(source, '        started = time.perf_counter()\n',
            '        started = time.perf_counter()\n'
            '        pcm = np.asarray(pcm, dtype=np.float32)\n'
            '        if pcm.ndim != 1 or not np.isfinite(pcm).all():\n'
            '            return {"status": "rejected", "reason": "invalid_audio"}\n')
        source = source.replace(', "asr_invoked": False', '')
    elif name == "pipeline":
        source = replace(source, "self.recognizer = self.asr = None", "self.recognizer = None")
        source = replace(source, 'bundle, asr=switches["asr"], threads=', 'bundle, threads=')
        source = cut(source, '        elif switches["asr"]:', '\n    def utterance')
        source = cut(source, '        if self.asr:', '        return {"status": "speech_detected"')
        source = source.replace(', "asr_invoked": False', '')
    elif name == "bundle":
        source = replace(source, '*([' + '"vcm_stream"] if streaming else []), "asr"',
                         '*( ["vcm_stream"] if streaming else [])')
        source = replace(source, '{"vcm", "vcm_stream", "asr"}', '{"vcm", "vcm_stream"}')
        source = replace(source, '        if component == "asr":\n            continue\n', '')
        source = source.replace("without requiring optional ASR artifacts", "without a transcription component")
    elif name == "config":
        source = source.replace('"vcm", "asr", ', '"vcm", ')
        source = source.replace('"vcm", "asr"', '"vcm"')
        source = source.replace('"vcm_epochs", "asr_epochs", ', '"vcm_epochs", ')
        source = replace(source, '    for key in ("vcm", "update_artifacts"):',
            '    if any(k in config["training"] for k in ("asr", "asr_epochs")):\n'
            '        raise ValueError("Transcription training switches have been removed")\n'
            '    for key in ("vcm", "update_artifacts"):')
    elif name == "text":
        source = remove_function(source, "decode_greedy")
        source = '\n'.join(line for line in source.splitlines() if not line.startswith('VOCAB =')) + '\n'
    elif name == "slot_contract":
        source = remove_function(source, "custom_media_eligibility")
    elif name == "train_helpers":
        source = remove_function(source, "validate_asr")
        source = source.replace("deployment/ASR frontend", "deployment frontend")
    elif name == "training":
        source = remove_function(source, "fit_asr")
        source = cut(source, 'def export_graphs(', '\ndef fit(', EXPORT + '\n')
        source = replace(source, '("vcm", "asr")', '("vcm",)')
        source = replace(source, '    if "asr" in selected:\n        result["asr"] = fit_asr(config, root, output, metadata)\n', '')
        source = replace(source, '; independent CTC ASR continuation', '')
        source = source.replace('if path.is_file():', 'if path.is_file() and not path.name.startswith("asr"):' )
    elif name == "validation":
        source = replace(source, '{"vad": True, "asr": False, "pipeline": "vad and not asr"}',
                         '{"vad": True, "vcm": True, "pipeline": "vad and vcm"}')
        source = source.replace('bundle, asr=False, threads=', 'bundle, threads=')
        source = replace(source, 'and not silence.get("asr_invoked")', '')
        source = replace(source, 'model.asr is None and model.stream is None', 'model.stream is None')
        source = replace(source, '"asr_loaded": False, ', '')
        source = cut(source, '    if config["tests"]["asr"]:', '    if config["tests"]["pipeline"]:')
        source = source.replace(', "asr": True', '')
        source = cut(source, '        calls = []\n', '    return checks\n',
            '        result = model.predict(pcm, stream=False)\n'
            '        require(result["status"] == "needs_parameters", "Custom media must clarify")\n'
            '        require(result["reason"] == "catalog_decoder_required", "Unexpected fallback")\n'
            '        require(not hasattr(model, "transcribe"), "Transcription API reintroduced")\n'
            '        family[0] = "UNKNOWN"\n'
            '        require(model.predict(pcm, stream=False)["status"] == "rejected", "Unknown accepted")\n'
            '        model.metadata = original\n'
            '        checks["pipeline"] = {"status": "passed", "real_vad_vcm_replay": True,\n'
            '            "routing_fixtures": ["custom media clarifies", "UNKNOWN rejected"],\n'
            '            "production_thresholds_modified": False}\n')
        source = replace(source, '    splits = {}\n', '    splits = {}\n    required_audio = set()\n')
        source = replace(source, '("vcm", "asr")', '("vcm",)')
        source = replace(source, '        splits[task] = dict(Counter(r["split"] for r in rows))',
                         '        splits[task] = dict(Counter(r["split"] for r in rows))\n'
                         '        required_audio.update(row["audio"] for row in rows)')
        source = replace(source, '    for relative, expected in index.items():',
                         '    for relative in sorted(required_audio):\n        expected = index[relative]')
        source = replace(source, '"audio_files_verified": len(index)', '"audio_files_verified": len(required_audio)')
        source = replace(source, '    vcm = asr = pipeline = None', '    vcm = pipeline = None')
        source = cut(source, '    if selected["asr"]:', '    if selected["pipeline"]:')
        source = replace(source, '    word_errors = words = char_errors = chars = exact_text = 0\n', '')
        source = replace(source, '            counts["asr_calls"] += sum(bool(r.get("asr_invoked")) for r in results)\n', '')
        source = cut(source, '        if asr:', '        predictions.append(entry)')
        source = cut(source, '    if asr:', '    command_paths =')
        source = source.replace('ASR-only or VAD-only', 'VAD-only')
        source = source.replace('custom-ASR acceptance', 'dynamic-catalog acceptance')
        source = cut(source, '    # Current VCM and ASR', '    if maximum:',
                     '    rows = [r for r in read_rows(root / "data/vcm.jsonl") if r["split"] == split]\n')
    elif name == "family_contract":
        source = source.replace('device names through ASR', 'unrestricted device names')
    elif name == "routing":
        source = source.replace('acoustic slots and ASR-derived arguments', 'typed acoustic-slot arguments')
        source = remove_function(source, "parse_custom_media")
    elif name == "finite_policy":
        source = source.replace('custom-ASR gate', 'dynamic-catalog gate')
        source = source.replace(', "asr_invoked": False', '')
        source = source.replace('"asr_calls": 0, ', '')
    compile(source, name, "exec")
    return source


INTRO = '''# VCM-only: architecture, training and Pi pipeline

The independent character-transcription model has been removed from this branch.
Only the existing v12 family encoder and finite heads are used by this notebook.
Unresolved parameters and custom media return clarification, not transcription.

The separate `phone_s53` research model uses constrained phoneme CTC for numbers
and dynamic catalogs; it is not automatically substituted for this v12 graph.
No acoustic weights were changed by removal. Historical validation is in Git
history. Run current tests before interpreting any prior accuracy as current.

Commands: `python train.py`, `python train.py --fit`, `python publish.py`.
Restored local VCM weights and referenced audio remain prerequisites; they are
not downloaded automatically. See `research/no_asr_review/RESULTS.md`.
'''
ARCHITECTURE = '''## Architecture and contracts

```mermaid
flowchart LR
    A[Microphone / WAV] --> B[Energy VAD and endpoint]
    B --> C[16 kHz causal log-mel frontend]
    C --> D[VCM family encoder + finite parameter heads]
    D --> E[Family and per-value acceptance]
    E --> F[Typed command or clarification]
```

The v12 network remains unchanged: 398,422 parameters including its frozen
309,130-parameter family encoder. There is no independent transcription session,
transcription trainer, character decoder, or transcription payload. The source
classifier and the sequence-preserving research CNN are separate checkpoints.
Neither service executes device actions. Pi and live-microphone qualification
remain outstanding.
'''
DATA = '''## Data and reproducibility

Training and validation read `data/vcm.jsonl` only. The original declared split
has 18,434 train, 5,364 development, 4,850 calibration and 4,314 test examples.
Those counts are historical metadata, not a newly replayed dataset.
The data audit checks hashes, speaker/pair/recording separation and only the audio
referenced by this VCM manifest. Unused files in an old shared audio index do not
become extra training requirements. Labels remain programmatic silver.

Finite-head continuation retains the encoder and uses train/development only;
calibration/test never select training weights. Parameter uncertainty cannot
invoke a second network. The original research packet must be restored separately
for the streaming phoneme experiments and visualization.
'''


def apply(root: Path) -> None:
    notebook_path = root / "Architecture.ipynb"
    notebook = json.loads(notebook_path.read_text())
    if notebook.get("metadata", {}).get("vcm_pipeline") == MARKER:
        print("Already migrated; no files changed")
        return
    sources = {c.get("metadata", {}).get("checkpoint", {}).get("module")
               for c in notebook["cells"]}
    if not {"tinyvcm.asr", "tinyvcm.asr_runtime"} <= sources:
        raise ValueError("Unexpected notebook; refuse partial migration")
    updates = {}
    validation_path = root / "validation.json"
    if validation_path.exists():
        historical = json.loads(validation_path.read_text())
        historical["historical_result_only"] = True
        historical["pipeline_revision_at_recording"] = "v12-with-independent-transcription"
        historical["current_pipeline_note"] = (
            "The separate transcription component was removed later. These are "
            "preserved pre-removal measurements, not current validation results.")
        updates["validation.json"] = json.dumps(historical, indent=2) + "\n"
    cells = []
    for cell in notebook["cells"]:
        module = cell.get("metadata", {}).get("checkpoint", {}).get("module")
        if module in {"tinyvcm.asr", "tinyvcm.asr_runtime"} or cell.get("id") in {"text-28", "text-43"}:
            continue
        if module:
            header, source = "".join(cell["source"]).split("\n", 1)
            changed = transform(module.rsplit(".", 1)[-1], source)
            if changed != source:
                cell["source"] = (header + "\n" + changed).splitlines(keepends=True)
                cell["execution_count"], cell["outputs"] = None, []
                cell["metadata"].pop("execution", None)
        if cell.get("id") in {"text-0", "text-1", "text-2"}:
            cell["source"] = {"text-0": INTRO, "text-1": ARCHITECTURE,
                              "text-2": DATA}[cell["id"]].splitlines(keepends=True)
        if cell.get("metadata", {}).get("workflow_result"):
            cell["source"] = ["## Current validation status\n\nTranscription removal changes the pipeline. "
                              "Historical run results are retained in Git history, not reported as a new run. "
                              "Run `train.py` with the local VCM artifacts and human recordings to refresh.\n"]
        if cell.get("id") == "text-56":
            cell["source"] = ["## Acceptance limits\n\nNo current human-speech or physical-Pi qualification "
                              "is claimed. The historical low accepted-recall results must not be hidden "
                              "by removal. Custom names require the separate catalog research path, not "
                              "unrestricted transcription.\n"]
        cells.append(cell)
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["vcm_pipeline"] = MARKER
    updates["Architecture.ipynb"] = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
    text = (root / "pipeline.toml").read_text()
    text = '\n'.join(line for line in text.splitlines() if not line.startswith(("asr =", "asr_epochs ="))) + '\n'
    updates["pipeline.toml"] = text.replace('vad and vcm and asr', 'vad and vcm')
    text = (root / "train.py").read_text()
    text = cut(text, '        if "asr" in validation:', '        for name in ("vcm", "pipeline"):', '')
    text = text.replace('v12-family-finite-with-independent-local-asr', 'v12-family-finite-vcm-only')
    text = text.replace('if p.is_file() and p.name != "manifest.json"',
                        'if p.is_file() and p.name != "manifest.json" and not p.name.startswith("asr")')
    updates["train.py"] = text
    text = (root / "publish.py").read_text()
    text = text.replace('VAD, VCM, finite heads and ASR', 'VAD, VCM and finite heads')
    text = replace(text, '\ndef download_json(',
                   '\nMODEL_FILES = ("bundle.json", "vcm.fp32.onnx")\n'
                   'RUNTIME_COMPONENTS = ["vad", "vcm", "finite_heads", "routing", "microphone"]\n\n'
                   'def download_json(')
    text = replace(text, '("bundle.json", "vcm.fp32.onnx", "asr.fp32.onnx")', 'MODEL_FILES')
    text = replace(text, '["vad", "vcm", "finite_heads", "asr", "routing", "microphone"]', 'RUNTIME_COMPONENTS')
    updates["publish.py"] = text
    metadata = json.loads((root / "artifacts/bundle.json").read_text())
    metadata.pop("asr", None)
    metadata["runtime_components"] = [v for v in metadata["runtime_components"] if v != "asr"]
    metadata["files"] = {k: v for k, v in metadata["files"].items() if not k.startswith("asr")}
    metadata["pipeline_revision"] = MARKER
    updates["artifacts/bundle.json"] = json.dumps(metadata, indent=2) + "\n"
    descriptor = json.loads((root / "data/dataset.json").read_text())
    for key in ("current_manifests", "origin_manifests"):
        descriptor[key].pop("asr", None)
    descriptor["notes"] += " VCM-only pipeline: the total audio inventory is historical; only VCM-referenced files are required."
    updates["data/dataset.json"] = json.dumps(descriptor, indent=2) + "\n"
    training = json.loads((root / "artifacts/training-config.json").read_text())
    training.pop("asr", None)
    updates["artifacts/training-config.json"] = json.dumps(training, indent=2) + "\n"
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    manifest["files"] = {k: v for k, v in manifest["files"].items() if not k.startswith("asr")}
    manifest["name"] = "v12-family-finite-vcm-only"
    for path in (root / "artifacts").iterdir():
        if path.name == "manifest.json" or path.name.startswith("asr") or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = updates[relative].encode() if relative in updates else path.read_bytes()
        manifest["files"][path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    updates["artifacts/manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    for name, text in updates.items():
        if name.endswith(".py"):
            compile(text, name, "exec")
    for name, text in updates.items():
        (root / name).write_text(text, encoding="utf-8")
    (root / "artifacts/asr-training-config.json").unlink(missing_ok=True)
    print("Migrated", sorted(updates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    apply(args.root)
