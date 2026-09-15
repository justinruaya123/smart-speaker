"""Microphone-selectable offline voice command CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import sys
import time
import tomllib


def resolve_microphone(devices, hostapis, selection=None, host_api=None, default=None):
    candidates = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0 and (host_api is None or hostapis[d["hostapi"]]["name"].lower() == host_api.lower())]
    if selection is None:
        if default is None or default < 0:
            raise ValueError("No default input; select --mic from the devices list")
        selection = str(default)
    if str(selection).isdigit():
        matches = [(i, d) for i, d in candidates if i == int(selection)]
    else:
        matches = [(i, d) for i, d in candidates if str(selection).lower() in d["name"].lower()]
    if len(matches) != 1:
        raise ValueError("Microphone selection is missing, output-only, or ambiguous; use devices and --host-api")
    return matches[0]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"train-vcm", "train-asr"}:
        from .train import main as train
        return train([argv[0].removeprefix("train-"), *argv[1:]])
    parser = argparse.ArgumentParser(prog="tinyvcm")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("devices")
    for name in ["predict", "listen"]:
        p = sub.add_parser(name)
        p.add_argument("--config")
        p.add_argument("--bundle")
        p.add_argument("--asr", choices=["on", "off"])
        p.add_argument("--variant", choices=["fp32", "int8"])
        p.add_argument("--catalog")
        p.add_argument("--threads", type=int)
        if name == "predict":
            p.add_argument("--wav", required=True)
            p.add_argument("--parameter-start-ms", type=float)
            p.add_argument("--save-parameter")
        else:
            p.add_argument("--mic")
            p.add_argument("--host-api")
            p.add_argument("--channel", type=int)
            p.add_argument("--capture-rate", type=int)
    for task in ["vcm", "asr"]:
        p = sub.add_parser("train-" + task)
        p.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command.startswith("train-"):
        from .train import main as train
        return train([args.command.removeprefix("train-"), *args.training_args])
    if args.command == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        return
    config = {}
    if args.config:
        config = tomllib.loads(Path(args.config).read_text(encoding="utf-8"))
    for key, value in config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    args.bundle = args.bundle or "artifacts/tinyvcm"
    args.asr = args.asr or "off"
    args.variant = args.variant or "fp32"
    args.threads = args.threads or 2
    from .inference import Recognizer
    from .audio import load_audio
    recognizer = Recognizer(args.bundle, asr=args.asr == "on", threads=args.threads, variant=args.variant)
    if args.catalog:
        recognizer.catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    if args.command == "predict":
        pcm, original_rate = load_audio(args.wav)
        result = recognizer.predict(pcm, parameter_start_ms=args.parameter_start_ms)
        if args.save_parameter and "parameter_audio" in result:
            import soundfile as sf
            sf.write(args.save_parameter, pcm[result["parameter_audio"]["start_sample_16k"]:], 16000)
        print(json.dumps(result, ensure_ascii=False))
        return
    listen(recognizer, args)


def listen(recognizer, args):
    import numpy as np
    from einops import reduce
    import sounddevice as sd
    from .audio import resample_audio
    devices, hostapis = sd.query_devices(), sd.query_hostapis()
    index, device = resolve_microphone(devices, hostapis, args.mic, args.host_api, sd.default.device[0])
    channel = args.channel if args.channel is not None else 0
    if not 0 <= channel < device["max_input_channels"]:
        raise ValueError("Input channel is unavailable")
    rate = args.capture_rate or 16000
    try:
        sd.check_input_settings(device=index, channels=channel + 1, samplerate=rate, dtype="float32")
    except Exception:
        if args.capture_rate:
            raise
        rate = int(device["default_samplerate"])
        sd.check_input_settings(device=index, channels=channel + 1, samplerate=rate, dtype="float32")
    print(json.dumps({"microphone": device["name"], "index": index, "host_api": hostapis[device["hostapi"]]["name"], "channel": channel, "capture_rate": rate, "model_rate": 16000}), file=sys.stderr, flush=True)
    chunks = queue.Queue(maxsize=200)
    overflow = [False]
    def callback(indata, frames, time_info, status):
        if status:
            overflow[0] = True
        try:
            chunks.put_nowait(indata[:, channel].copy())
        except queue.Full:
            overflow[0] = True
    buffer = []
    preroll = []
    active, silence, elapsed = False, 0.0, 0.0
    state = None
    next_inference = 0.16
    noise_rms = 0.003
    with sd.InputStream(device=index, samplerate=rate, channels=channel + 1, dtype="float32", blocksize=int(rate * 0.04), callback=callback) as stream:
        try:
            while stream.active:
                chunk = chunks.get(timeout=3)
                duration = len(chunk) / rate
                if overflow[0]:
                    print(json.dumps({"status": "rejected", "reason": "audio_overflow"}), flush=True)
                    buffer, preroll = [], []
                    active, silence, elapsed = False, 0.0, 0.0
                    overflow[0] = False
                    while not chunks.empty():
                        try:
                            chunks.get_nowait()
                        except queue.Empty:
                            break
                    continue
                rms = float(np.sqrt(reduce(chunk * chunk, "t ->", "mean")))
                speech = rms > max(0.008, noise_rms * 3)
                if not active:
                    preroll.append(chunk)
                    preroll = preroll[-5:]
                    if not speech:
                        noise_rms = 0.98 * noise_rms + 0.02 * rms
                        continue
                    active = True
                    buffer = list(preroll)
                    elapsed = sum(len(x) for x in buffer) / rate
                    state = recognizer.begin_stream()
                    next_inference = 0.16
                else:
                    buffer.append(chunk)
                    elapsed += duration
                silence = 0.0 if speech else silence + duration
                if elapsed >= next_inference:
                    from einops import pack
                    original, _ = pack(buffer, "*")
                    pcm = resample_audio(original, rate)
                    for event in recognizer.advance_stream(pcm, state):
                        print(json.dumps(event), flush=True)
                    next_inference = elapsed + 0.16
                if elapsed >= 8 or silence >= 0.4:
                    if elapsed >= 8:
                        result = {"status": "rejected", "reason": "utterance_timeout"}
                    else:
                        from einops import pack
                        original, _ = pack(buffer, "*")
                        pcm = resample_audio(original, rate)
                        for event in recognizer.advance_stream(pcm, state, final=True):
                            print(json.dumps(event), flush=True)
                        result = recognizer.predict(pcm, events=state["events"])
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    buffer, preroll = [], []
                    active, silence, elapsed = False, 0.0, 0.0
        except KeyboardInterrupt:
            return
        except (queue.Empty, sd.PortAudioError) as exc:
            print(json.dumps({"status": "rejected", "reason": "microphone_disconnected_or_stalled", "detail": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
