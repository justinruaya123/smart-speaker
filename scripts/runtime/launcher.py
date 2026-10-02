"""Self-contained pipeline: setup, component checks, microphone or audio-file inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import urllib.request
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


def find_uv(root):
    """Use installed uv, or bootstrap the official pinned tool beside the release."""
    available = shutil.which('uv')
    if available:
        return available
    directory = root / '.tools' / 'uv'
    executable = directory / ('uv.exe' if os.name == 'nt' else 'uv')
    if executable.is_file():
        return str(executable)
    suffix = 'ps1' if os.name == 'nt' else 'sh'
    environment = os.environ.copy()
    environment['UV_UNMANAGED_INSTALL'] = str(directory)
    environment['UV_NO_MODIFY_PATH'] = '1'
    print('Installing uv 0.12.5 locally (first setup only)...', flush=True)
    with tempfile.TemporaryDirectory(prefix='uv-bootstrap-', dir=root) as temporary:
        installer = Path(temporary) / ('install.' + suffix)
        urllib.request.urlretrieve('https://astral.sh/uv/0.12.5/install.' + suffix, installer)
        command = (['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(installer)]
                   if os.name == 'nt' else ['sh', str(installer)])
        subprocess.run(command, check=True, env=environment, cwd=root)
    if not executable.is_file():
        raise RuntimeError('uv installation did not produce ' + str(executable))
    return str(executable)


def setup(root, argv):
    root = Path(root).resolve()
    parser = argparse.ArgumentParser(description="Create/update this release's .venv with uv sync --locked")
    parser.add_argument("--system-packages", action="store_true", help="install Pi OS Python/audio packages with apt")
    parser.add_argument("--online", action="store_true", help="compatibility flag; setup downloads missing dependencies by default")
    args = parser.parse_args(argv)
    if sys.version_info[:2] not in {(3, 11), (3, 12), (3, 13)}:
        raise RuntimeError("Use CPython 3.11, 3.12 or 3.13")
    if not all((root / name).is_file() for name in ('pyproject.toml', 'uv.lock')):
        raise RuntimeError('Release needs pyproject.toml and uv.lock; extract the complete new ZIP')
    if args.system_packages:
        if not sys.platform.startswith("linux"):
            raise ValueError("--system-packages supports Raspberry Pi OS/Debian only")
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "python3-venv", "libportaudio2", "libsndfile1"], check=True)
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    process_environment = os.environ.copy()
    process_environment.pop('VIRTUAL_ENV', None)
    process_environment['UV_PROJECT_ENVIRONMENT'] = str(environment)
    process_environment['UV_CACHE_DIR'] = str(root / '.cache' / 'uv')
    command = [find_uv(root), 'sync', '--locked', '--no-dev', '--no-build', '--no-python-downloads',
               '--project', str(root), '--python', str(python if python.is_file() else sys.executable)]
    subprocess.run(command, check=True, cwd=root, env=process_environment)
    if not (root / 'pipeline.toml').exists() and (root / 'pipeline.default.toml').is_file():
        shutil.copyfile(root / 'pipeline.default.toml', root / 'pipeline.toml')
    print("Synchronized. Run python3 pipeline.pyz --test, then --list-mics and --mic NUMBER.", flush=True)


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
        if archive.suffix != '.pyz':
            raise ValueError('For the source checkout, use python train.py --tests-only to set up the root environment')
        materialize(archive)
        setup(root, sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="run configured checks and packaged human-speech samples")
    mode.add_argument('--benchmark', action='store_true', help='measure packaged CPU latency and peak RAM without opening a microphone')
    mode.add_argument("--list-mics", action="store_true")
    mode.add_argument("--mic", type=int, help="numeric microphone index")
    mode.add_argument("--wav", type=Path, help="decode/resample an audio file and run the configured stages")
    default_config = root / 'pipeline.toml'
    if not default_config.is_file():
        default_config = root / 'pipeline.default.toml'
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--backend", choices=("none",), help="standalone VCM; no transcription backend")
    parser.add_argument("--wake", action="store_true", help="treat --wav as continuous audio containing Hey Jarvis and a command")
    parser.add_argument("--no-wake", action="store_true", help="explicit diagnostic microphone mode without wake activation")
    parser.add_argument("--seconds", type=float, default=0, help="stop microphone capture after this many seconds; 0 listens until Ctrl+C")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--capture-rate", type=int)
    parser.add_argument("--vad-debug", action="store_true", help="print input peak, wake score and speech probability once per second")
    args = parser.parse_args()
    import math
    if (args.channel < 0 or args.mic is not None and args.mic < 0 or args.capture_rate is not None and args.capture_rate <= 0
            or not math.isfinite(args.seconds) or args.seconds < 0 or args.wake and args.no_wake):
        parser.error("Microphone/channel indices must be nonnegative and capture rate positive")
    activate(root)
    if archive.suffix == '.pyz':
        destination, manifest = materialize(archive)
        bundle = destination / 'model'
    else:
        bundle = root / 'artifacts'
        manifest = {'components': ['vcm', 'wake', 'silero_vad'], 'source_checkout': True}
    from tinyvcm.config import load_config, select_backend
    config = load_config(args.config.resolve())
    if args.backend:
        config = select_backend(config, args.backend)
    if args.no_wake:
        config['wake']['enabled'] = False
    if args.wake:
        if not all(config['runtime'].values()) or config['audio']['vad_backend'] != 'silero':
            raise ValueError('--wake requires runtime VAD/VCM and audio.vad_backend="silero"')
        config['wake']['enabled'] = True
    if config['parameters']['enabled'] and config['parameters']['backend'] not in manifest['components']:
        raise ValueError('This runtime supports standalone VCM only; disable transcription parameters')
    if args.list_mics:
        import sounddevice as sd
        hosts = sd.query_hostapis()
        emit({"microphones": [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
              "default_sample_rate": d["default_samplerate"], "host_api": hosts[d["hostapi"]]["name"]}
              for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]})
        return
    if args.benchmark:
        if archive.suffix != '.pyz':
            raise ValueError('Use --benchmark on the published pipeline.pyz containing bounded smoke samples')
        from tinyvcm.benchmark import benchmark
        result = benchmark(bundle, config, destination / 'samples')
        (root / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        emit(result)
        return
    if args.test:
        from tinyvcm.validation import run_checks, validate_samples
        result = {"components": manifest["components"], "selected_checks": config["tests"],
                  "tests": run_checks(config, bundle),
                  "sample_validation": validate_samples(config, bundle, destination / "samples") if archive.suffix == '.pyz' else {'quality_gate_passed': False, 'note': 'Use the experiment replay harness for source-checkout acoustic evaluation.'},
                  "hardware": Path("/proc/device-tree/model").read_text(encoding="utf-8").strip("\x00")
                     if Path("/proc/device-tree/model").exists() else sys.platform,
                  "live_microphone_tested": False}
        receipt = root / 'launch-validation.json' if archive.suffix == '.pyz' else root / '.cache/wake-pipeline-v1/source-checks.json'
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        emit(result)
        if config["validation"]["require_quality_gate"] and not result["sample_validation"]["quality_gate_passed"]:
            raise RuntimeError("Quality gates unmet; see launch-validation.json")
        return
    from tinyvcm.pipeline import Pipeline
    pipeline = Pipeline(bundle, config)
    if args.wav:
        from tinyvcm.audio import load_audio
        pcm, rate = load_audio(args.wav.expanduser().resolve(), channel=args.channel)
        emit({"input_sample_rate": rate, "wake_required": args.wake,
              "results": pipeline.live_stream().replay(pcm) if args.wake else pipeline.process(pcm)})
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
