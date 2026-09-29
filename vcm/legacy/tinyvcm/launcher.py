"""Self-contained pipeline: setup, component checks, microphone or audio-file inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import venv
import zipfile


def materialize(archive_path):
    archive_path = Path(archive_path).resolve()
    fingerprint = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    destination = archive_path.parent / ".runtime" / fingerprint[:20]
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        manifest = json.loads(archive.read("manifest.json"))
        if len(names) != len(set(names)) or set(names) != {*manifest["files"], "manifest.json"}:
            raise ValueError("Incomplete or duplicate pipeline archive entries")
        for name, expected in manifest["files"].items():
            parts = PurePosixPath(name).parts
            if (not parts or ".." in parts or "\\" in name or name.startswith("/")
                    or parts[0] not in {"__main__.py", "tinyvcm", "model", "samples"}):
                raise ValueError(f"Invalid pipeline archive entry: {name}")
            content = archive.read(name)
            if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError(f"Pipeline checksum mismatch: {name}")
            if parts[0] not in {"model", "samples"}:
                continue
            path = destination / name
            if path.is_file() and path.read_bytes() == content:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name); stream.write(content)
            try:
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return destination, manifest


def setup(root, argv):
    parser = argparse.ArgumentParser(description="Install into a local .venv; bundled Pi wheels are used offline by default")
    parser.add_argument("--system-packages", action="store_true", help="install Pi OS Python/audio packages with apt")
    parser.add_argument("--online", action="store_true", help="fetch Python dependencies instead of using bundled wheels")
    args = parser.parse_args(argv)
    if sys.version_info[:2] not in {(3, 11), (3, 12), (3, 13)}:
        raise RuntimeError("Use CPython 3.11, 3.12 or 3.13")
    if args.system_packages:
        if not sys.platform.startswith("linux"):
            raise ValueError("--system-packages supports Raspberry Pi OS/Debian only")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "python3-venv", "libportaudio2", "libsndfile1"], check=True)
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:", "--require-hashes", "-r", str(root / "requirements.lock")]
    if not args.online:
        if not (root / "wheels").is_dir():
            raise RuntimeError("No bundled wheels; republish with include_wheels=true or use setup --online")
        command += ["--no-index", "--find-links", str(root / "wheels")]
    subprocess.run(command, check=True)
    print("Installed. Run python3 pipeline.pyz --test, then --list-mics and --mic NUMBER.", flush=True)


def activate(root):
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if python.exists() and Path(sys.prefix).resolve() != environment.resolve():
        command = [str(python), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
        if os.name == "nt":
            raise SystemExit(subprocess.call(command))
        os.execv(str(python), command)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    archive = Path(sys.argv[0]).resolve()
    root = archive.parent
    if sys.argv[1:2] == ["setup"]:
        materialize(archive)
        setup(root, sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="run configured checks and packaged human-speech samples")
    mode.add_argument("--list-mics", action="store_true")
    mode.add_argument("--mic", type=int, help="numeric microphone index")
    mode.add_argument("--wav", type=Path, help="decode/resample an audio file and run the configured stages")
    parser.add_argument("--config", type=Path, default=root / "pipeline.toml")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--capture-rate", type=int)
    args = parser.parse_args()
    if args.channel < 0 or args.mic is not None and args.mic < 0 or args.capture_rate is not None and args.capture_rate <= 0:
        parser.error("Microphone/channel indices must be nonnegative and capture rate positive")
    activate(root)
    destination, manifest = materialize(archive)
    from tinyvcm.config import load_config
    config = load_config(args.config.resolve())
    if args.list_mics:
        import sounddevice as sd
        hosts = sd.query_hostapis()
        emit({"microphones": [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
              "default_sample_rate": d["default_samplerate"], "host_api": hosts[d["hostapi"]]["name"]}
              for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]})
        return
    if args.test:
        from tinyvcm.validation import run_checks, validate_samples
        result = {"components": manifest["components"], "selected_checks": config["tests"],
                  "tests": run_checks(config, destination / "model"),
                  "sample_validation": validate_samples(config, destination / "model", destination / "samples"),
                  "hardware": Path("/proc/device-tree/model").read_text(encoding="utf-8").strip("\x00")
                     if Path("/proc/device-tree/model").exists() else sys.platform,
                  "live_microphone_tested": False}
        (root / "launch-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(result)
        if config["validation"]["require_quality_gate"] and not result["sample_validation"]["quality_gate_passed"]:
            raise RuntimeError("Quality gates unmet; see launch-validation.json")
        return
    from tinyvcm.pipeline import Pipeline
    pipeline = Pipeline(destination / "model", config)
    if args.wav:
        from tinyvcm.audio import load_audio
        pcm, rate = load_audio(args.wav.expanduser().resolve(), channel=args.channel)
        emit({"input_sample_rate": rate, "results": pipeline.process(pcm)})
        return
    from tinyvcm.capture import listen
    emit({"stage": "service", "status": "starting", "microphone": args.mic, "runtime": config["runtime"]})
    try:
        listen(pipeline, args)
    finally:
        emit({"stage": "service", "status": "stopped"})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Pipeline failed: {exc}\nInstall once with python3 pipeline.pyz setup. "
              "Pi OS prerequisites: python3-venv, libportaudio2, libsndfile1.", file=sys.stderr)
        sys.exit(1)
