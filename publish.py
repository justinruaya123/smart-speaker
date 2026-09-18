"""Package the complete current pipeline for Raspberry Pi 4.

    python publish.py
    python publish.py --config pipeline.toml

All runtime code comes from Architecture.ipynb. Tests may be selected in the
configuration; VAD, VCM, finite heads and ASR are always included in the package.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from train import ROOT, digest, load_config, notebook_sources, save_json, verify_artifacts


def download_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "smart-speaker-publisher/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def wheel_files(config):
    """Download hash-pinned CPython wheels compatible with 64-bit Pi OS."""
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    pins = re.findall(r"^([\w-]+)==([\w.]+)([^\n]*)", lock, flags=re.M)
    hashes = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", lock))
    cache = ROOT / ".cache/wheels"
    cache.mkdir(parents=True, exist_ok=True)
    metadata_path = cache / "pypi-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else []
    indexed = {(p["name"], p["version"]): p for p in metadata}
    versions = config["publish"]["python_versions"]
    selected = {}
    compatibility = {version: [] for version in versions}
    def compatible(filename, version):
        if not filename.endswith(".whl"):
            return False
        python_tag, abi, platform_tag = filename[:-4].rsplit("-", 3)[1:]
        minor = int(version.split(".")[1]); cp = "cp3" + str(minor)
        if platform_tag == "any":
            return bool(set(python_tag.split(".")) & {"py3", "py2.py3", cp}) and abi == "none"
        platforms = platform_tag.split(".")
        suitable_platform = any(tag == "manylinux2014_aarch64" or re.fullmatch(r"manylinux_2_(\d+)_aarch64", tag)
                               and int(tag.split("_")[2]) <= 28 for tag in platforms)
        if not suitable_platform:
            return False
        if abi == "abi3":
            return any(re.fullmatch(r"cp3\d+", tag) and int(tag[3:]) <= minor for tag in python_tag.split("."))
        return cp in python_tag.split(".") and abi in {cp, "none"}
    for name, version, marker in pins:
        key = name, version
        if key not in indexed:
            print(f"METADATA {name}=={version}", flush=True)
            payload = download_json(f"https://pypi.org/pypi/{name}/{version}/json")
            indexed[key] = {"name": name, "version": version, "requires_python": payload["info"].get("requires_python"),
                            "requires_dist": payload["info"].get("requires_dist"),
                            "files": [{"filename": item["filename"], "url": item["url"], "sha256": item["digests"]["sha256"]}
                                      for item in payload["urls"]]}
        for python_version in versions:
            # The lock has only these two explicit Python-version markers.
            condition = marker.strip().rstrip("\\").strip()
            if condition and condition not in {'; python_version < "3.12"', '; python_version >= "3.12"'}:
                raise ValueError(f"Unsupported lock marker: {condition}")
            if '< "3.12"' in condition and python_version != "3.11" or '>= "3.12"' in condition and python_version == "3.11":
                continue
            candidates = [item for item in indexed[key]["files"] if compatible(item["filename"], python_version)
                          and item["sha256"] in hashes]
            if not candidates:
                raise ValueError(f"No pinned ARM64 wheel for {name}=={version}, CPython {python_version}")
            item = sorted(candidates, key=lambda item: item["filename"])[0]
            selected[item["filename"]] = item
            compatibility[python_version].append(item["filename"])
    save_json(metadata_path, list(indexed.values()))
    def fetch(item):
        path = cache / item["filename"]
        if path.is_file() and digest(path) == item["sha256"]:
            return path
        print(f"WHEEL {path.name}", flush=True)
        request = urllib.request.Request(item["url"], headers={"User-Agent": "smart-speaker-publisher/1.0"})
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if digest(temporary) != item["sha256"]:
                raise ValueError(f"Wheel hash mismatch: {path.name}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, selected.values()))
    return sorted(paths), compatibility


def deterministic_zip(path, entries, *, executable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 17, 0, 0, 0))
            info.external_attr = (0o755 if executable and name == "__main__.py" else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data.read_bytes() if isinstance(data, Path) else data)


def package_sources(config):
    sources = notebook_sources(runtime_only=True)
    entries = {}
    for name, source in sources.items():
        target = "__main__.py" if name == "tinyvcm.launcher" else name.replace(".", "/") + ".py"
        entries[target] = source.encode("utf-8")
    for name in ("bundle.json", "vcm.fp32.onnx", "asr.fp32.onnx"):
        entries["model/" + name] = (ROOT / "artifacts" / name).read_bytes()
    refs = {row["id"]: row for row in map(json.loads, (ROOT / "data/references.jsonl").read_text(encoding="utf-8").splitlines())}
    counts, samples = {}, []
    for row in map(json.loads, (ROOT / "data/vcm.jsonl").read_text(encoding="utf-8").splitlines()):
        if row["split"] != "test" or counts.get(row["command"], 0) >= config["publish"]["sample_per_intent"]:
            continue
        path = (ROOT / row["audio"]).resolve()
        if not path.is_relative_to((ROOT / "data/audio").resolve()):
            raise ValueError("Sample path outside the current dataset")
        name = f"audio/{len(samples):03d}{path.suffix}"
        entries["samples/" + name] = path.read_bytes()
        samples.append({**row, "audio": name, "annotation": refs[row["id"]]})
        counts[row["command"]] = counts.get(row["command"], 0) + 1
    labels = json.loads((ROOT / "artifacts/bundle.json").read_text(encoding="utf-8"))["vcm"]["labels"]
    if set(counts) != set(labels):
        raise ValueError("Packaged smoke samples must cover each current family")
    entries["samples/samples.jsonl"] = "".join(json.dumps(row) + "\n" for row in samples).encode()
    for path in (ROOT / "data/sources").glob("*.json"):
        entries["samples/sources/" + path.name] = path.read_bytes()
    manifest = {"format_version": 1, "components": ["vad", "vcm", "finite_heads", "asr", "routing", "microphone"],
                "artifact_manifest_sha256": digest(ROOT / "artifacts/manifest.json"),
                "notebook_sha256": digest(ROOT / "Architecture.ipynb"),
                "sample_counts": counts,
                "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in entries.items()}}
    entries["manifest.json"] = json.dumps(manifest, indent=2).encode()
    return entries, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline.toml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    verify_artifacts()
    entries, manifest = package_sources(config)
    output = (ROOT / config["publish"]["output"]).resolve()
    if not output.is_relative_to(ROOT / "dist") or output.suffix != ".zip":
        raise ValueError("Publish output must be a ZIP inside the workspace dist directory")
    staging_parent = ROOT / ".cache/publish"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-", dir=staging_parent) as temporary:
        staging = Path(temporary)
        application = staging / "pipeline.pyz"
        deterministic_zip(application, entries, executable=True)
        shutil.copy2(config_path, staging / "pipeline.toml")
        shutil.copy2(ROOT / "requirements.lock", staging / "requirements.lock")
        python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError("Run python train.py first to set up the environment and validate")
        print("Checking the isolated executable with packaged human-speech samples...", flush=True)
        result = subprocess.run([str(python), str(application), "--test"], cwd=staging, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + "\n" + result.stderr)
        release_check = json.loads((staging / "launch-validation.json").read_text(encoding="utf-8"))
        wheels, compatibility = wheel_files(config) if config["publish"]["include_wheels"] else ([], {})
        package = {"smart-speaker/pipeline.pyz": application,
                   "smart-speaker/pipeline.toml": config_path,
                   "smart-speaker/requirements.lock": ROOT / "requirements.lock",
                   "smart-speaker/README.md": ROOT / "README.md"}
        package.update({"smart-speaker/wheels/" + path.name: path for path in wheels})
        report = {"output": str(output.relative_to(ROOT)), "components": manifest["components"],
                  "pipeline_sha256": digest(application), "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
                  "notebook_sha256": manifest["notebook_sha256"], "python_arm64_wheels": compatibility,
                  "offline_python_dependencies": bool(wheels), "release_check": release_check,
                  "physical_pi_validation": False}
        package["smart-speaker/build.json"] = json.dumps(report, indent=2).encode()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = output.with_suffix(".building")
        try:
            deterministic_zip(temporary_zip, package)
            os.replace(temporary_zip, output)
        finally:
            temporary_zip.unlink(missing_ok=True)
        report.update(zip_sha256=digest(output), zip_bytes=output.stat().st_size)
        save_json(output.parent / "build.json", report)
        output.with_suffix(output.suffix + ".sha256").write_text(report["zip_sha256"] + "  " + output.name + "\n", encoding="ascii")
        print(json.dumps({key: report[key] for key in ("output", "components", "zip_sha256", "zip_bytes", "offline_python_dependencies")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Publish failed: {exc}", file=sys.stderr)
        sys.exit(1)
