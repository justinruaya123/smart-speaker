"""One workstation entry point: setup, optional training, tests and validation.

    python train.py                  # configured setup/test/validate/report
    python train.py --fit            # also train the enabled current models
    python train.py --tests-only     # focused checks selected in pipeline.toml
    python train.py --no-setup       # use the already installed environment
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode="w", encoding="utf-8") as f:
        temporary = Path(f.name)
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    register_sources()
    from tinyvcm.config import load_config as read_config
    return read_config(path)


def notebook_sources(path=ROOT / "Architecture.ipynb", runtime_only=False):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for cell in notebook["cells"]:
        details = cell.get("metadata", {}).get("checkpoint", {})
        if cell["cell_type"] != "code" or not details or runtime_only and not details.get("runtime"):
            continue
        name = details["module"]
        if not name.startswith("tinyvcm.") or not all(p.isidentifier() for p in name.split(".")):
            raise ValueError(f"Invalid notebook module: {name}")
        first, source = "".join(cell["source"]).split("\n", 1)
        if first != f"%%module {name}" or name in result:
            raise ValueError(f"Invalid or duplicate source cell: {name}")
        compile(source, name, "exec")
        result[name] = source
    return result


def register_sources(root=ROOT):
    os.environ["SMART_SPEAKER_ROOT"] = str(root)
    sources = notebook_sources(root / "Architecture.ipynb")
    class Sources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            key = "tinyvcm.__init__" if fullname == "tinyvcm" else fullname
            if key in sources:
                return importlib.util.spec_from_loader(fullname, self, is_package=fullname == "tinyvcm")
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            key = "tinyvcm.__init__" if module.__name__ == "tinyvcm" else module.__name__
            module.__file__ = str(root / "Architecture.ipynb") + "#" + key
            exec(compile(sources[key], module.__file__, "exec"), module.__dict__)
    for name in list(sys.modules):
        if name == "tinyvcm" or name.startswith("tinyvcm."):
            del sys.modules[name]
    sys.meta_path.insert(0, Sources())
    return sources


def ensure_environment(config, *, notebook=False, no_setup=False):
    """Use one local environment; install only when requested or absent."""
    target = ROOT / ".venv"
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not no_setup and config["workflow"]["setup"]:
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(target)
        if subprocess.run([str(python), "-c", "import pip"], capture_output=True).returncode:
            subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60", "--retries", "5",
                        "--only-binary=:all:", "--require-hashes", "-r", str(ROOT / "requirements.lock")], check=True)
        if notebook:
            subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60",
                            "nbformat==5.11.1", "nbclient==0.11.0", "ipykernel==7.3.0",
                            "torch==2.13.0", "onnx==1.22.0", "onnxscript==0.7.2"], check=True)
    if python.exists() and Path(sys.prefix).resolve() != target.resolve():
        command = [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        if "--no-setup" not in command:
            command.append("--no-setup")
        raise SystemExit(subprocess.call(command))


def verify_artifacts(root=ROOT):
    folder = root / "artifacts"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError(f"Artifact integrity failed: {name}")
    metadata = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
    for name, expected in metadata["files"].items():
        if digest(folder / name) != expected["sha256"]:
            raise ValueError(f"Bundle metadata hash mismatch: {name}")
    return manifest


def refresh_artifact_manifest(root=ROOT):
    folder = root / "artifacts"
    files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(folder.iterdir()) if p.is_file() and p.name != "manifest.json" and not p.name.startswith("asr")}
    save_json(folder / "manifest.json", {"format_version": 1,
              "name": "v12-family-finite-vcm-only", "files": files})


def update_report(result, root=ROOT):
    path = root / "Architecture.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "## Current workflow result\n\nGenerated by `train.py` from the current artifacts and data.\n\n"
    text += f"Run: `{result.get('run_utc', 'not recorded')}`. Training performed: **{result['training_performed']}**.\n\n"
    if result.get("tests"):
        text += "| Check | Result |\n|---|---|\n"
        text += "".join(f"| {name} | {item['status']} |\n" for name, item in result["tests"].items()) + "\n"
    validation = result.get("validation")
    if validation:
        text += f"Evaluated **{validation['examples']:,}** recordings from `{validation['split']}`"
        text += " (limited run).\n\n" if validation["limited"] else " (full split).\n\n"
        text += "| Path | Exact finite commands | Correct / emitted | Finite-command recall |\n|---|---:|---:|---:|\n"
        for name in ("vcm", "pipeline"):
            if name not in validation:
                continue
            counts = validation[name]["commands"]
            text += (f"| {name} | {counts['correct_complete']} / {counts['finite_references']} | "
                     f"{counts['correct_complete']} / {counts['emitted']} | {validation[name]['exact_finite_command_recall']:.2%} |\n")
        text += "\n"
        for name in ("vcm", "pipeline"):
            if name not in validation:
                continue
            text += f"### Per-family results: {name}\n\n| Family | Support | Raw recall | Accepted recall | Accepted precision |\n|---|---:|---:|---:|---:|\n"
            def percent(value):
                return "not assessed" if value is None else f"{value:.1%}"
            for family, counts in validation[name]["intent"]["per_family"].items():
                text += (f"| {family} | {counts['support']} | {percent(counts['raw_recall'])} | "
                         f"{percent(counts['correct_accepted_recall'])} | {percent(counts['accepted_precision'])} |\n")
            text += "\n"
        text += f"Declared software quality gates passed: **{validation['quality_gate_passed']}**. Physical Pi and live microphone validation: **not performed**.\n\n"
    text += "<details><summary>Complete validation record and provenance</summary>\n\n```json\n" + json.dumps(result, indent=2, ensure_ascii=False) + "\n```\n\n</details>\n"
    cell = {"cell_type": "markdown", "id": "workflow-results", "metadata": {"workflow_result": True}, "source": text.splitlines(True)}
    notebook["cells"] = [c for c in notebook["cells"] if not c.get("metadata", {}).get("workflow_result")]
    notebook["cells"].insert(1, cell)
    save_json(path, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline.toml")
    parser.add_argument("--fit", action="store_true", help="train the selected components before testing")
    parser.add_argument("--tests-only", action="store_true", help="run only the selected component checks")
    parser.add_argument("--no-setup", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.fit:
        config["workflow"]["train"] = True
    if args.tests_only:
        config["workflow"].update(train=False, validate=False)
    ensure_environment(config, notebook=config["workflow"]["train"], no_setup=args.no_setup)
    os.chdir(ROOT)
    register_sources()
    verify_artifacts()
    from datetime import datetime, timezone
    result = {"run_utc": datetime.now(timezone.utc).isoformat(),
              "selected_checks": config["tests"], "training_performed": False,
              "physical_raspberry_pi_tested": False, "live_microphone_tested": False}
    if config["workflow"]["train"]:
        from tinyvcm.training import fit
        result["training"] = fit(config, ROOT)
        result["training_performed"] = True
        refresh_artifact_manifest()
    from tinyvcm.validation import run_checks, validate_dataset
    if config["workflow"]["test"]:
        result["tests"] = run_checks(config, ROOT / "artifacts")
    if config["workflow"]["validate"]:
        result["validation"] = validate_dataset(config, ROOT)
    result["artifact_manifest_sha256"] = digest(ROOT / "artifacts/manifest.json")
    result["config_sha256"] = digest(args.config.resolve())
    save_json(ROOT / "validation.json", result)
    if config["workflow"]["report"]:
        update_report(result)
    print(json.dumps(result, indent=2))
    if config["validation"]["require_quality_gate"] and not result.get("validation", {}).get("quality_gate_passed", False):
        raise RuntimeError("Current quality gates failed; see validation.json and the notebook")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError, ModuleNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Workflow failed: {exc}", file=sys.stderr)
        sys.exit(1)
