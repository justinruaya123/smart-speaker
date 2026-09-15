"""Run this unchanged on Pi 4; records the actual host and CPU measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnxruntime as ort
from tinyvcm.audio import load_audio
from tinyvcm.inference import Recognizer
from tinyvcm.data import read_jsonl, write_json


def peak_rss():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        values = Counters()
        values.cb = ctypes.sizeof(values)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if get_memory(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
            return int(values.PeakWorkingSetSize)
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def pi_state():
    model = Path("/proc/device-tree/model")
    result = {"model": model.read_bytes().decode().strip("\x00") if model.exists() else None}
    if shutil.which("vcgencmd"):
        for command in ("measure_temp", "get_throttled"):
            result[command] = subprocess.run(["vcgencmd", command], capture_output=True, text=True, timeout=3).stdout.strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="artifacts/tinyvcm")
    parser.add_argument("--variant", choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", default="reports/benchmark_host.json")
    parser.add_argument("--asr", choices=["on", "off"], default="on")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "data/manifests/voice_test_suite.jsonl")
    audio = [(row, load_audio(ROOT / row["audio"])[0]) for row in rows]
    before = pi_state()
    started = time.perf_counter()
    recognizer = Recognizer(ROOT / args.bundle, asr=args.asr == "on", threads=args.threads, variant=args.variant)
    if args.asr == "on":
        recognizer.transcribe(audio[0][1])
    cold_ms = (time.perf_counter() - started) * 1000
    for _, pcm in audio[:3]:
        recognizer.classify(pcm)
        recognizer.stream_events(pcm)
    values = {"vcm_whole_file": [], "vcm_stream_replay": []}
    if args.asr == "on":
        values["asr_whole_file"] = []
        values["routed_pipeline"] = []
    for _ in range(args.repeats):
        for row, pcm in audio:
            jobs = {"vcm_whole_file": recognizer.classify, "vcm_stream_replay": recognizer.stream_events}
            if args.asr == "on":
                jobs.update(asr_whole_file=recognizer.transcribe, routed_pipeline=recognizer.predict)
            for name, function in jobs.items():
                started = time.perf_counter()
                function(pcm)
                elapsed = time.perf_counter() - started
                values[name].append({"id": row["id"], "seconds": elapsed, "audio_seconds": len(pcm)/16000, "rtf": elapsed/(len(pcm)/16000)})
    summary = {}
    for name, items in values.items():
        summary[name] = {"runs": len(items), "p50_ms": float(np.median([r["seconds"] for r in items])*1000), "p95_ms": float(np.quantile([r["seconds"] for r in items], .95)*1000), "p95_rtf": float(np.quantile([r["rtf"] for r in items], .95)), "total_audio_seconds": sum(r["audio_seconds"] for r in items), "total_processing_seconds": sum(r["seconds"] for r in items)}
    report = {"host": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": platform.python_version(), "onnxruntime": ort.__version__}, "pi_before": before, "pi_after": pi_state(), "is_raspberry_pi4": "Raspberry Pi 4" in (before["model"] or ""), "threads": args.threads, "variant": args.variant, "asr": args.asr, "session_load_and_first_asr_ms": cold_ms, "process_peak_rss_bytes": peak_rss(), "measurements": summary, "interpretation": "Offline CPU replay including feature extraction. No real-time audio sleeps, microphone/endpoint latency, sustained thermal soak, or hardware success is implied.", "bundle_sha256": hashlib.sha256((ROOT / args.bundle / "bundle.json").read_bytes()).hexdigest(), "details": values}
    write_json(ROOT / args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
