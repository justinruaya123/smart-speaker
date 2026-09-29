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

ROOT = Path(__file__).resolve().parents[1]


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


def notebook_sources(path=None, runtime_only=False):
    """Historical name retained for comparison tests; reads extracted Python."""
    folder = ROOT / 'vcm/legacy'
    modules = json.loads((folder / 'modules.json').read_text(encoding='utf-8'))
    return {name: (folder / (name.replace('.', '/') + '.py')).read_text(encoding='utf-8')
            for name, entry in modules.items() if not runtime_only or entry['runtime']}


def register_sources(root=ROOT):
    """Compatibility for historical tests; source now lives in ordinary files."""
    os.environ["SMART_SPEAKER_ROOT"] = str(root)
    folder = ROOT / "vcm/legacy"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return {"tinyvcm." + p.stem: p.read_text(encoding="utf-8")
            for p in (folder / "tinyvcm").glob("*.py")}


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


def update_report(*args, **kwargs):
    raise RuntimeError('Historical report writer retired; Architecture.ipynb is curated')
