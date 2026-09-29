"""VCM-only contract checks. Random weights are not recognition evidence."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vcm import legacy_workflow as train


@pytest.fixture(scope="module")
def modules():
    return train.register_sources(ROOT)


def test_registered_sources_have_no_transcription_model(modules):
    assert "tinyvcm.asr" not in modules
    assert "tinyvcm.asr_runtime" not in modules
    for source in modules.values():
        assert not any(isinstance(n, ast.ImportFrom) and
                       n.module in {"asr", "asr_runtime"}
                       for n in ast.walk(ast.parse(source)))


def test_config_has_no_asr_switches(modules):
    from tinyvcm.config import load_config
    config = load_config(ROOT / "pipeline.toml")
    for section in ("runtime", "training", "tests"):
        assert "asr" not in config[section]
    assert "asr_epochs" not in config["training"]
    assert config["tests"]["pipeline"]


@pytest.mark.parametrize("section", ["runtime", "training", "tests"])
def test_obsolete_switch_is_rejected(modules, tmp_path, section):
    from tinyvcm.config import load_config
    text = (ROOT / "pipeline.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(text.replace(f"[{section}]", f"[{section}]\nasr = true"))
    with pytest.raises(ValueError):
        load_config(path)


def test_bundle_is_vcm_only(modules):
    from tinyvcm.bundle import runtime_components
    data = json.loads((ROOT / "artifacts/bundle.json").read_text())
    assert "asr" not in data
    assert runtime_components(data) == ["vcm"]
    assert not any(name.startswith("asr") for name in data["files"])
    data["runtime_components"] = ["vcm", "asr"]
    with pytest.raises(ValueError):
        runtime_components(data)


def test_no_transcribe_api(modules):
    from inspect import signature
    from tinyvcm.inference import Recognizer
    assert not hasattr(Recognizer, "transcribe")
    assert not hasattr(Recognizer, "transcribe_features")
    assert "asr" not in signature(Recognizer).parameters


def test_no_transcription_trainer(modules):
    from tinyvcm import training, train_helpers
    assert not hasattr(training, "fit_asr")
    assert not hasattr(train_helpers, "validate_asr")


def test_exporter_rejects_other_tasks_before_file_access(modules, tmp_path):
    from tinyvcm.training import export_graphs
    with pytest.raises(ValueError, match="VCM"):
        export_graphs(tmp_path, ["asr"], {})


def test_legacy_custom_media_never_falls_back(modules):
    from tinyvcm.inference import Recognizer
    model = object.__new__(Recognizer)
    model.metadata = json.loads((ROOT / "artifacts/bundle.json").read_text())
    model.labels = model.metadata["vcm"]["labels"]
    model.catalog = model.metadata["catalog"]
    model.variant = "fp32"
    vocab = model.metadata["acoustic_slots"]["vocabulary"]

    def onehot(values, value):
        p = np.zeros(len(values))
        p[values.index(value)] = 1.0
        return p

    slots = {key: onehot(values, values[0]) for key, values in vocab.items()}
    for key, value in [("guard", "READY"), ("media", "OTHER"), ("media_mode", "CUSTOM")]:
        slots[key] = onehot(vocab[key], value)
    family = onehot(model.labels, "MEDIA_PLAY")
    model.classify_with_slots = lambda pcm: (family, slots, None)
    policy = model.metadata["acoustic_slots"]["variants"]["fp32"]
    policy.update(custom_media_threshold=0.0, custom_guard_threshold=0.0)
    result = model.predict(np.ones(16000, dtype=np.float32) * .1, stream=False)
    assert result["status"] == "needs_parameters"
    assert result["reason"] == "catalog_decoder_required"
    assert "parameter_transcript" not in result
    assert "asr_invoked" not in result


@pytest.mark.parametrize("bad", [np.full(16000, np.nan), np.full(16000, np.inf),
                                 np.zeros((16000, 2))])
def test_invalid_audio_rejected_before_model_call(modules, bad):
    from tinyvcm.inference import Recognizer
    result = object.__new__(Recognizer).predict(bad, stream=False)
    assert result["status"] == "rejected"
    assert result["reason"] == "invalid_audio"


def publisher_module():
    spec = importlib.util.spec_from_file_location("checked_publisher", ROOT / "vcm/legacy_publish.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    return publisher


def test_publisher_cannot_ship_asr(modules):
    publisher = publisher_module()
    assert publisher.MODEL_FILES == ("bundle.json", "vcm.fp32.onnx")
    assert "asr" not in publisher.RUNTIME_COMPONENTS


def test_artifact_manifest_no_asr():
    data = json.loads((ROOT / "artifacts/manifest.json").read_text())
    assert not any(name.startswith("asr") for name in data["files"])
    assert not (ROOT / "artifacts/asr-training-config.json").exists()


def test_dataset_descriptor_only_requires_vcm():
    data = json.loads((ROOT / "data/dataset.json").read_text())
    assert set(data["current_manifests"]) == {"vcm"}


@pytest.fixture(scope="module")
def graph_bundle(modules, tmp_path_factory):
    import torch
    from tinyvcm.slot_model import SlotVCM
    from tinyvcm.training import export_graphs
    folder = tmp_path_factory.mktemp("vcm_fixture")
    data = json.loads((ROOT / "artifacts/bundle.json").read_text())
    config = {
        "backbone_config": {
            "classes": len(data["vcm"]["labels"]), "width": 16,
            "kernels": [5, 7], "dropout": 0., "pooling": "mean_max",
            "dilations": [1, 2]},
        "vocabulary": data["acoustic_slots"]["vocabulary"],
        "hidden": 16, "adapt_intent": False}
    torch.manual_seed(20260929)
    torch.set_num_threads(1)
    model = SlotVCM(**config).eval()
    with torch.no_grad():
        model.backbone.head.weight.zero_()
        model.backbone.head.bias.fill_(-20)
        model.backbone.head.bias[0] = 20
    torch.save({"config": config, "model": model.state_dict(),
                "labels": data["vcm"]["labels"]}, folder / "vcm.pt")
    assert export_graphs(folder, ["vcm"], data)["vcm"] < 2e-4
    (folder / "bundle.json").write_text(json.dumps(data))
    return folder


def test_real_graph_runs_without_second_session(modules, graph_bundle):
    from tinyvcm.inference import Recognizer
    model = Recognizer(graph_bundle, threads=1)
    x = np.sin(np.arange(16000) * .05).astype(np.float32) * .1
    p, slots, _ = model.classify_with_slots(x)
    assert len(p) == 10 and len(slots) == 7
    assert model.predict(x, stream=False)["vcm_command"] == "UNKNOWN"
    assert model.stream is None
    assert not hasattr(model, "asr")
    assert not (graph_bundle / "asr.fp32.onnx").exists()


def test_component_smoke_checks_execute_vcm_only(modules, graph_bundle):
    from tinyvcm.config import load_config
    from tinyvcm.validation import run_checks
    checks = run_checks(load_config(ROOT / "pipeline.toml"), graph_bundle)
    assert set(checks) == {"config", "vad", "vcm", "pipeline"}
    assert all(result["status"] == "passed" for result in checks.values())


def test_package_has_one_graph_even_if_old_file_is_present(modules, graph_bundle,
                                                         tmp_path, monkeypatch):
    import shutil
    import soundfile as sf
    from tinyvcm.config import load_config
    publisher = publisher_module()
    config = load_config(ROOT / "pipeline.toml")
    artifacts = tmp_path / "artifacts"
    shutil.copytree(graph_bundle, artifacts)
    (artifacts / "asr.fp32.onnx").write_bytes(b"obsolete local artifact")
    (artifacts / "manifest.json").write_text("{}")
    (tmp_path / "data/audio").mkdir(parents=True)
    shutil.copy2(ROOT / "Architecture.ipynb", tmp_path / "Architecture.ipynb")
    labels = json.loads((artifacts / "bundle.json").read_text())["vcm"]["labels"]
    rows = []
    for i, label in enumerate(labels):
        audio = f"data/audio/{i}.wav"
        sf.write(tmp_path / audio, np.zeros(1600), 16000)
        rows.append({"id": str(i), "command": label, "split": "test", "audio": audio})
    (tmp_path / "data/vcm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "data/references.jsonl").write_text("".join(json.dumps({"id": r["id"]}) + "\n" for r in rows))
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    entries, manifest = publisher.package_sources(config)
    assert {name for name in entries if name.endswith(".onnx")} == {"model/vcm.fp32.onnx"}
    assert not any("asr" in name.lower() for name in entries)
    assert "asr" not in manifest["components"]


@pytest.mark.parametrize("name,digest", [
    ("vcm", "a47b0ce30616bdc6c53b0976c2c51d039fe051d85a5b84f6886f95fe2a42c7fa"),
    ("slot_model", "1d3f5b8107494675328462566d8d8a64d5bdb712974b377ffc3d78695fdfb38b"),
    ("audio", "b097aecf362f0636bffbe267888e24ed80727d208e0f20f819a44fe43cdfd0b5")])
def test_removal_preserves_vcm_model_and_frontend(name, digest):
    source = f'%%module tinyvcm.{name}\n' + (ROOT / 'vcm/legacy/tinyvcm' / (name+'.py')).read_text(encoding='utf-8')
    assert hashlib.sha256(source.encode()).hexdigest() == digest


def test_old_metrics_are_labelled_historical():
    assert json.loads((ROOT / "validation.json").read_text())["historical_result_only"] is True
