"""Filesystem/API compatibility only; never assemble a real card or run models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from final import assemble_card
from scripts.test import vcm_mixed_benchmark as benchmark

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "final/assemble_card_postfit_path_v1.py"


def adapter():
    assert SCRIPT.is_file(), "Versioned postfit path adapter has not been implemented"
    spec = importlib.util.spec_from_file_location("assemble_card_postfit_path_v1_under_test", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def receipt_file(tmp_path):
    path = tmp_path / "postfit-receipt.json"
    path.write_bytes(b'{"software_fixture": "no actual neural measurements"}\n')
    return path


def test_original_directory_only_verifier_reproduces_receipt_file_path_failure(tmp_path):
    path = receipt_file(tmp_path)
    original = path.read_bytes()
    with pytest.raises(NotADirectoryError) as error:
        benchmark.verify_postfit({}, path)
    assert str(path / "postfit-receipt.json") in str(error.value)
    assert path.read_bytes() == original


@pytest.mark.parametrize("path_type", [Path, str])
def test_existing_exact_receipt_file_routes_to_original_strict_verifier(tmp_path, monkeypatch, path_type):
    path = receipt_file(tmp_path)
    original_bytes = path.read_bytes()
    context, expected = {"bound_context": object()}, object()
    calls = []
    def strict(actual_context, actual_postfit):
        calls.append((actual_context, actual_postfit))
        assert actual_context is context
        assert actual_postfit == path.resolve().parent
        assert (actual_postfit / "postfit-receipt.json").read_bytes() == original_bytes
        return expected
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    with adapter().postfit_path_compatibility():
        assert benchmark.verify_postfit(context, path_type(path)) is expected
    assert benchmark.verify_postfit is strict
    assert len(calls) == 1 and path.read_bytes() == original_bytes


@pytest.mark.parametrize("path_type", [Path, str])
def test_ordinary_directory_argument_reaches_original_verifier_unchanged(tmp_path, monkeypatch, path_type):
    path = path_type(tmp_path)
    context = {"same": True}
    def strict(actual_context, actual_postfit):
        assert actual_context is context and actual_postfit is path
        return "strict result"
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    with adapter().postfit_path_compatibility():
        assert benchmark.verify_postfit(context, path) == "strict result"
    assert benchmark.verify_postfit is strict


@pytest.mark.parametrize("name", ["observations.json", "postfit-receipt-v2.json", "postfit_receipt.json"])
def test_other_existing_files_are_rejected_without_calling_or_weakening_verifier(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_bytes(b"software fixture")
    strict = lambda *args: pytest.fail("Wrong receipt filename must not route to another receipt")
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    with adapter().postfit_path_compatibility():
        with pytest.raises(ValueError, match="postfit-receipt.json"):
            benchmark.verify_postfit({}, path)
    assert benchmark.verify_postfit is strict and path.read_bytes() == b"software fixture"


def test_symlink_cannot_coerce_an_other_named_file_into_a_receipt_directory(tmp_path, monkeypatch):
    target = tmp_path / "other.json"
    target.write_bytes(b"software fixture")
    path = tmp_path / "postfit-receipt.json"
    path.symlink_to(target)
    strict = lambda *args: pytest.fail("Resolved non-receipt file must be rejected")
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    with adapter().postfit_path_compatibility():
        with pytest.raises(ValueError, match="postfit-receipt.json"):
            benchmark.verify_postfit({}, path)
    assert benchmark.verify_postfit is strict


def test_missing_argument_is_left_for_original_verifier_to_reject(tmp_path, monkeypatch):
    path = tmp_path / "missing" / "postfit-receipt.json"
    def strict(context, actual):
        assert actual is path
        raise FileNotFoundError("original strict missing input")
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    with adapter().postfit_path_compatibility():
        with pytest.raises(FileNotFoundError, match="original strict"):
            benchmark.verify_postfit({}, path)
    assert benchmark.verify_postfit is strict


def test_real_original_evidence_checks_still_reject_an_invalid_candidate(tmp_path):
    path = receipt_file(tmp_path)
    path.write_text(json.dumps({"selected": {"checkpoint_sha256": "b" * 64}}) + "\n")
    context = {"selected": {"checkpoint_sha256": "a" * 64}}
    original = benchmark.verify_postfit
    with pytest.raises(ValueError, match="selected-candidate postfit receipt"):
        original(context, path.parent)
    with adapter().postfit_path_compatibility():
        with pytest.raises(ValueError, match="selected-candidate postfit receipt"):
            benchmark.verify_postfit(context, path)
    assert benchmark.verify_postfit is original


def test_original_verifier_restored_when_body_or_strict_verification_fails(tmp_path, monkeypatch):
    path = receipt_file(tmp_path)
    def strict(*args):
        raise RuntimeError("original strict evidence check")
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    loaded = adapter()
    with pytest.raises(RuntimeError, match="original strict evidence"):
        with loaded.postfit_path_compatibility():
            benchmark.verify_postfit({}, path)
    assert benchmark.verify_postfit is strict
    with pytest.raises(KeyError, match="assembly failed"):
        with loaded.postfit_path_compatibility():
            raise KeyError("assembly failed")
    assert benchmark.verify_postfit is strict


def test_entrypoint_uses_original_assembly_main_and_restores_verifier(tmp_path, monkeypatch):
    path = receipt_file(tmp_path)
    strict = lambda context, actual: actual
    monkeypatch.setattr(benchmark, "verify_postfit", strict)
    calls = []
    def original_main():
        calls.append(list(sys.argv))
        assert benchmark.verify_postfit({}, path) == path.parent
        return 7
    monkeypatch.setattr(assemble_card, "main", original_main)
    arguments = [str(SCRIPT), "--output", "software-only-newdir"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert adapter().main() == 7
    assert calls == [arguments] and benchmark.verify_postfit is strict


def test_entrypoint_restores_verifier_on_original_assembly_failure(monkeypatch):
    strict = benchmark.verify_postfit
    def original_main():
        assert benchmark.verify_postfit is not strict
        raise ValueError("original assembly failure")
    monkeypatch.setattr(assemble_card, "main", original_main)
    with pytest.raises(ValueError, match="original assembly failure"):
        adapter().main()
    assert benchmark.verify_postfit is strict


def test_scoped_compatibility_does_not_mutate_frozen_sources_or_receipt_bytes(tmp_path):
    names = ("scripts/test/vcm_mixed_wave_parity_cuda.py", "scripts/test/vcm_mixed_benchmark.py",
             "final/assemble_card.py", "final/render_card.py")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    path = receipt_file(tmp_path)
    original_bytes = path.read_bytes()
    with adapter().postfit_path_compatibility():
        with pytest.raises(ValueError, match="selected-candidate postfit receipt"):
            benchmark.verify_postfit({"selected": {}}, path)
    assert {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names} == hashes
    assert path.read_bytes() == original_bytes


def test_cli_retains_original_assembly_arguments_without_any_actual_execution(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert all("--" + flag in result.stdout for flag in (
        "training", "inputs", "postfit", "evaluation", "bundle", "data-receipt", "output"))
    assert not list(tmp_path.iterdir())
