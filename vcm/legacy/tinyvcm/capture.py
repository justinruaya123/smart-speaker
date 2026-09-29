"""Numeric microphone selection and capture using the same VAD as replay."""
import json
import queue
import sys


def resolve_microphone(devices, hostapis, selection=None, host_api=None, default=None):
    selection = default if selection is None else selection
    if selection is None or not str(selection).isdigit():
        raise ValueError("Select a numeric microphone index from --list-mics")
    index = int(selection)
    if not 0 <= index < len(devices) or devices[index]["max_input_channels"] < 1:
        raise ValueError("Selected index is not an input device")
    return index, devices[index]


def listen(pipeline, args):
    import sounddevice as sd
    from .audio import resample_audio
    from .pipeline import EnergyVAD
    if not pipeline.config["runtime"]["vad"]:
        raise ValueError("Microphone capture requires runtime.vad = true; use --wav for presegmented audio")
    devices, hosts = sd.query_devices(), sd.query_hostapis()
    index, device = resolve_microphone(devices, hosts, args.mic, default=sd.default.device[0])
    if not 0 <= args.channel < device["max_input_channels"]:
        raise ValueError("Input channel unavailable")
    rate = args.capture_rate or 16000
    try:
        sd.check_input_settings(device=index, channels=args.channel + 1, samplerate=rate, dtype="float32")
    except sd.PortAudioError:
        if args.capture_rate:
            raise
        rate = int(device["default_samplerate"])
        sd.check_input_settings(device=index, channels=args.channel + 1, samplerate=rate, dtype="float32")
    print(json.dumps({"microphone": device["name"], "index": index, "capture_rate": rate,
                      "model_rate": 16000, "channel": args.channel}), file=sys.stderr, flush=True)
    chunks, overflow = queue.Queue(maxsize=200), [False]
    def callback(indata, frames, timing, status):
        if status:
            overflow[0] = True
        try:
            chunks.put_nowait(indata[:, args.channel].copy())
        except queue.Full:
            overflow[0] = True
    vad = EnergyVAD(pipeline.config["audio"], rate)
    blocksize = round(rate * pipeline.config["audio"]["block_ms"] / 1000)
    with sd.InputStream(device=index, samplerate=rate, channels=args.channel + 1,
                        dtype="float32", blocksize=blocksize, callback=callback) as stream:
        while stream.active:
            try:
                chunk = chunks.get(timeout=3)
            except queue.Empty as exc:
                raise RuntimeError("Microphone disconnected or stalled") from exc
            if overflow[0]:
                vad.reset()
                while not chunks.empty():
                    try:
                        chunks.get_nowait()
                    except queue.Empty:
                        break
                overflow[0] = False
                print(json.dumps({"status": "rejected", "reason": "audio_overflow"}), flush=True)
                continue
            was_active = vad.active
            result = vad.push(chunk)
            if not was_active and vad.active:
                print(json.dumps({"stage": "vad", "status": "speech_started"}), flush=True)
            if result is not None:
                pcm, timeout = result
                print(json.dumps(pipeline.utterance(resample_audio(pcm, rate), timeout),
                                 ensure_ascii=False, allow_nan=False), flush=True)
