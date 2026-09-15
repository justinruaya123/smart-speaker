"""Run in the fresh runtime-only environment, where torch must be absent."""
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
assert importlib.util.find_spec("torch") is None, "Use the isolated runtime-only environment"
from tinyvcm.audio import load_audio
from tinyvcm.data import read_jsonl, write_json
from tinyvcm.inference import Recognizer

rows = read_jsonl(ROOT / "data/manifests/voice_test_suite.jsonl")
selected = [next(r for r in rows if r["command"] == command) for command in ("POWER_ON", "MEDIA_PLAY", "TIME_QUERY", "UNKNOWN")]
outputs = []
for variant in ("fp32", "int8"):
    recognizer = Recognizer(ROOT / "artifacts/tinyvcm", asr=True, variant=variant)
    for row in selected:
        pcm, _ = load_audio(ROOT / row["audio"])
        result = recognizer.predict(pcm)
        text = recognizer.transcribe(pcm)
        outputs.append({"variant": variant, "id": row["id"], "gold": row["command"], "result": result, "full_audio_asr": text})
    assert "torch" not in sys.modules
cli = []
for asr in ("off", "on"):
    command = [sys.executable, "-B", "-m", "tinyvcm", "predict", "--wav", str(ROOT / selected[1]["audio"]), "--asr", asr]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    cli.append({"asr": asr, "result": json.loads(process.stdout)})
process = subprocess.run([sys.executable, "-B", "-m", "tinyvcm", "devices"], cwd=ROOT, capture_output=True, text=True, check=True)
report = {"status": "passed", "host": platform.platform(), "python": platform.python_version(), "executable": sys.executable, "pytorch_installed": False, "pytorch_imported": "torch" in sys.modules, "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "einops", "onnxruntime", "soundfile", "sounddevice")}, "voice_inference_cases": len(outputs), "cli_predict_calls": cli, "device_enumeration_succeeded": bool(process.stdout.strip()), "evidence": "Fresh venv installed only requirements-runtime.lock; both FP32 and INT8 models infer without PyTorch/ASR service/network access. This does not test a physical Pi or live microphone speech.", "predictions": outputs}
write_json(ROOT / "reports/runtime_smoke.json", report)
print(json.dumps({k: v for k, v in report.items() if k != "predictions"}, indent=2))
