"""Packaging tests check bytes and provenance without loading any recognizer."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "package_pi.py"
SOURCES = (
    "scripts/runtime/speaker_worker.py", "scripts/runtime/speaker_config.py",
    "scripts/runtime/pcv_final_runtime.py", "scripts/runtime/speaker_confidence.py",
    "scripts/runtime/pcv_schema.py", "scripts/runtime/pcv_benchmark_adapter.py",
    "scripts/runtime/vcm_intent_slots_runtime.py", "scripts/runtime/vcm_intent_slots_bindings.py",
    "scripts/runtime/vcm_intent_slots_contract.py", "scripts/runtime/vcm_intent_slots_entities.py",
    "scripts/runtime/vcm_intent_slots_decode.py", "vcm/frontend.py", "vcm/catalog.py",
    "scripts/models/vcm_intent_slots.py", "final/pi_finalize.py", "final/package_pi.py",
    "experiments/phrase-contract-v1/pcvcm/contract.py", "experiments/phrase-contract-v1/pcvcm/decoder.py",
)


def module():
    assert SCRIPT.is_file(), "Pi package builder has not been implemented"
    spec = importlib.util.spec_from_file_location("package_pi_under_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(data)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "project"
    for name in SOURCES:
        put(root, name, ("# " + name + "\n").encode())
    for name in ("requirements.lock", "scripts/runtime/pyproject.toml", "scripts/runtime/uv.lock",
                 "scripts/runtime/ncnn-LICENSE.txt", "data/sources/cmudict-LICENSE.txt"):
        put(root, name, ("fixture: " + name + "\n").encode())
    put(root, "final/MODEL_CARD.md", b"# Actual semi-final card\n")
    put(root, "experiments/vcm-intent-slots-v1/protocol.json", {
        "source_revisions_from_supplied_handoff_not_newly_downloaded": {
            "benchmark": "ab39857cf73dd067d068fed388c321ab160dc191"}})
    bundle = root / ".cache/selected/bundle"
    contract = {"contract_hash": "c" * 64, "thresholds": {"intent": 0.95, "slot": 0.90}}
    put(bundle, "contract.json", contract)
    put(bundle, "catalog.json", {"entries": []})
    put(bundle, "normalization.json", {"fit_split": "train"})
    put(bundle, "policy.json", {"require_calibration": True, "calibration_id": "d" * 64})
    for name in ("model.onnx", "reference.pt", "network.ncnn.param", "network.ncnn.bin"):
        put(bundle, name, ("opaque selected fixture: " + name).encode())
    put(bundle, "evidence/selection.json", {"epoch": 12})
    checkpoint = sha(bundle / "reference.pt")
    recipe = {"format": "vcm-intent-slots-v1", "contract_hash": "c" * 64,
              "head_order": ["intent", "role_states", "validity", "values", "entities"],
              "checkpoint_source_sha256": checkpoint, "encoder_id": checkpoint,
              "checkpoint_status": "unqualified_candidate", "engine_status": {"ncnn": "exported_unverified"},
              "files": {name: sha(bundle / name) for name in (
                  "contract.json", "catalog.json", "normalization.json", "policy.json",
                  "model.onnx", "reference.pt", "network.ncnn.param", "network.ncnn.bin")},
              "source_bindings": {name: sha(root / name) for name in (
                  "vcm/frontend.py", "scripts/runtime/vcm_intent_slots_runtime.py", "scripts/models/vcm_intent_slots.py")}}
    put(bundle, "recipe.json", recipe)
    card = {"format": "vcm-model-card-v2", "bindings": {"checkpoint_sha256": checkpoint,
            "recipe_sha256": sha(bundle / "recipe.json"), "contract_hash": "c" * 64},
            "gates": {"accuracy_ready": False}, "pi": {"status": "deferred", "results": None},
            "status": "failed: human gate"}
    card_path = put(root, "final/card.json", card)
    return root, bundle, card_path, recipe


def test_packages_every_bundle_byte_and_required_imports_without_model_loading(inputs, tmp_path):
    root, bundle, card, _ = inputs
    put(root, ".env", b"never publish this")
    result = module().package(bundle, card, tmp_path / "pi.zip", root=root)
    with zipfile.ZipFile(tmp_path / "pi.zip") as archive:
        assert archive.read("models/selected/model.onnx") == b"opaque selected fixture: model.onnx"
        assert archive.read("models/selected/evidence/selection.json") == (bundle / "evidence/selection.json").read_bytes()
        for path in bundle.rglob("*"):
            if path.is_file():
                assert archive.read("models/selected/" + path.relative_to(bundle).as_posix()) == path.read_bytes()
        assert archive.read("pcvcm/decoder.py") == (root / "experiments/phrase-contract-v1/pcvcm/decoder.py").read_bytes()
        assert archive.read("final/card.json") == card.read_bytes()
        assert "requirements.lock" in archive.namelist()
        assert "pyproject.toml" in archive.namelist()
        assert "uv.lock" in archive.namelist()
        assert "final/pi_finalize.py" in archive.namelist()
        assert ".env" not in archive.namelist()
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["physical_pi_measured"] is False
        assert manifest["model_binding"] == json.loads(card.read_text())["bindings"]
    assert result["sha256"] == sha(tmp_path / "pi.zip")
    assert result["physical_pi_measured"] is False


def test_identical_inputs_produce_identical_zip_bytes(inputs, tmp_path):
    root, bundle, card, _ = inputs
    builder = module()
    builder.package(bundle, card, tmp_path / "first.zip", root=root)
    builder.package(bundle, card, tmp_path / "second.zip", root=root)
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    with pytest.raises(FileExistsError):
        builder.package(bundle, card, tmp_path / "first.zip", root=root)


@pytest.mark.parametrize("path", ["model.onnx", "contract.json"])
def test_stale_bundle_bytes_are_rejected_before_publication(inputs, tmp_path, path):
    root, bundle, card, _ = inputs
    (bundle / path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding|hash"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)
    assert not (tmp_path / "pi.zip").exists()


def test_stale_runtime_source_is_rejected(inputs, tmp_path):
    root, bundle, card, _ = inputs
    (root / "vcm/frontend.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source|binding"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "recipe_sha256", "contract_hash"])
def test_card_for_another_candidate_is_rejected(inputs, tmp_path, field):
    root, bundle, card, _ = inputs
    data = json.loads(card.read_text())
    data["bindings"][field] = "e" * 64
    put(root, "final/card.json", data)
    with pytest.raises(ValueError, match="card|binding"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)


def test_explicit_report_and_pinned_benchmark_extras_preserve_bytes(inputs, tmp_path):
    root, bundle, card, _ = inputs
    report = put(root, "final/model.pdf", b"%PDF fixture actual receipt")
    parser = put(tmp_path, "parser.py", b"# pinned parser bytes\n")
    module().package(bundle, card, tmp_path / "pi.zip", root=root,
                     extra={"final/model.pdf": report, "benchmark/pinned/parser.py": parser})
    with zipfile.ZipFile(tmp_path / "pi.zip") as archive:
        assert archive.read("final/model.pdf") == report.read_bytes()
        assert archive.read("benchmark/pinned/parser.py") == parser.read_bytes()
        assert archive.read("experiments/vcm-intent-slots-v1/protocol.json") == (root / "experiments/vcm-intent-slots-v1/protocol.json").read_bytes()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "final/card.json"])
def test_unsafe_or_colliding_extra_paths_are_rejected(inputs, tmp_path, name):
    root, bundle, card, _ = inputs
    report = put(tmp_path, "report.json", {})
    with pytest.raises(ValueError, match="path|collision|collid"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root, extra={name: report})
    assert not (tmp_path / "pi.zip").exists()


def test_recipe_source_path_escape_is_rejected(inputs, tmp_path):
    root, bundle, card, recipe = inputs
    recipe["source_bindings"]["../outside.py"] = "f" * 64
    put(bundle, "recipe.json", recipe)
    with pytest.raises(ValueError, match="path|source"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)


def test_checkpoint_identity_must_match_preserved_reference_bytes(inputs, tmp_path):
    root, bundle, card, recipe = inputs
    recipe["checkpoint_source_sha256"] = recipe["encoder_id"] = "a" * 64
    put(bundle, "recipe.json", recipe)
    data = json.loads(card.read_text())
    data["bindings"].update(checkpoint_sha256="a" * 64, recipe_sha256=sha(bundle / "recipe.json"))
    put(root, "final/card.json", data)
    with pytest.raises(ValueError, match="checkpoint|reference"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)
    assert not (tmp_path / "pi.zip").exists()


def test_recipe_cannot_change_between_verification_and_packaging(inputs, tmp_path, monkeypatch):
    root, bundle, card, recipe = inputs
    original = Path.read_bytes
    recipe_reads = 0

    def changed_read(path):
        nonlocal recipe_reads
        if path == bundle / "recipe.json":
            recipe_reads += 1
            if recipe_reads > 1:
                return json.dumps({**recipe, "encoder_id": "f" * 64}).encode()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    with pytest.raises(ValueError, match="changed|recipe"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)
    assert not (tmp_path / "pi.zip").exists()


def test_verified_runtime_source_bytes_are_not_replaced_by_a_later_read(inputs, tmp_path, monkeypatch):
    root, bundle, card, _ = inputs
    original = Path.read_bytes
    frontend = root / "vcm/frontend.py"
    first_bytes = original(frontend)
    reads = 0

    def changed_read(path):
        nonlocal reads
        if path == frontend:
            reads += 1
            return first_bytes if reads == 1 else b"unbound later read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed_read)
    module().package(bundle, card, tmp_path / "pi.zip", root=root)
    with zipfile.ZipFile(tmp_path / "pi.zip") as archive:
        assert archive.read("vcm/frontend.py") == first_bytes


def test_bundle_symlink_is_not_followed(inputs, tmp_path):
    root, bundle, card, _ = inputs
    outside = put(tmp_path, "outside.dat", b"outside")
    (bundle / "external.dat").symlink_to(outside)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        module().package(bundle, card, tmp_path / "pi.zip", root=root)
    assert not (tmp_path / "pi.zip").exists()


def test_package_verifier_detects_modified_members(inputs, tmp_path):
    root, bundle, card, _ = inputs
    builder = module()
    builder.package(bundle, card, tmp_path / "pi.zip", root=root)
    assert builder.verify_package(tmp_path / "pi.zip")["physical_pi_measured"] is False
    with zipfile.ZipFile(tmp_path / "pi.zip") as original:
        entries = {name: original.read(name) for name in original.namelist()}
    entries["models/selected/model.onnx"] = b"tampered"
    with zipfile.ZipFile(tmp_path / "tampered.zip", "x") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="hash|checksum"):
        builder.verify_package(tmp_path / "tampered.zip")


def test_verification_cli_needs_only_standard_library(inputs, tmp_path):
    root, bundle, card, _ = inputs
    module().package(bundle, card, tmp_path / "pi.zip", root=root)
    result = subprocess.run([sys.executable, "-S", "-B", str(SCRIPT), "--verify", str(tmp_path / "pi.zip")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "package_hashes_verified"
