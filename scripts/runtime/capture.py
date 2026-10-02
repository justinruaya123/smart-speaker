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
    if pipeline.config.get('wake', {}).get('enabled') or pipeline.config['audio'].get('vad_backend') == 'silero':
        return listen_neural(pipeline, args)
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
    vad = EnergyVAD(pipeline.config["audio"], rate, calibrate=True)
    def calibration_event(reason):
        print(json.dumps({"stage": "vad", "status": "calibrating", "reason": reason,
                          "duration_ms": pipeline.config["audio"].get("calibration_ms", 1000),
                          "message": "Keep quiet until the ready event"}), flush=True)
    debug_samples = 0
    blocksize = round(rate * pipeline.config["audio"]["block_ms"] / 1000)
    with sd.InputStream(device=index, samplerate=rate, channels=args.channel + 1,
                        dtype="float32", blocksize=blocksize, callback=callback) as stream:
        calibration_event("startup")
        while stream.active:
            try:
                chunk = chunks.get(timeout=3)
            except queue.Empty as exc:
                raise RuntimeError("Microphone disconnected or stalled") from exc
            if overflow[0]:
                vad.reset(calibrate=True)
                while not chunks.empty():
                    try:
                        chunks.get_nowait()
                    except queue.Empty:
                        break
                overflow[0] = False
                print(json.dumps({"status": "rejected", "reason": "audio_overflow"}), flush=True)
                calibration_event("audio_overflow")
                continue
            was_active, was_calibrating = vad.active, vad.calibrating
            result = vad.push(chunk)
            if was_calibrating and not vad.calibrating:
                print(json.dumps({"stage": "vad", "status": "ready", **vad.diagnostics()}), flush=True)
            if not was_active and vad.active:
                print(json.dumps({"stage": "vad", "status": "speech_started", **vad.diagnostics()}), flush=True)
            debug_samples += len(chunk)
            if getattr(args, "vad_debug", False) and debug_samples >= rate:
                print(json.dumps({"stage": "vad", "status": "levels", **vad.diagnostics()}), flush=True)
                debug_samples %= rate
            if result is not None:
                pcm, timeout = result
                outcome = pipeline.utterance(resample_audio(pcm, rate), timeout)
                if timeout or getattr(args, "vad_debug", False):
                    outcome = {**outcome, "vad": vad.last_endpoint}
                print(json.dumps(outcome, ensure_ascii=False, allow_nan=False), flush=True)
                if timeout:
                    calibration_event("utterance_timeout")


def listen_neural(pipeline, args):
    """Bounded-buffer microphone -> continuous resampler -> pretrained wake/VAD."""
    import time
    import numpy as np
    import sounddevice as sd
    from .audio import CaptureResampler
    devices, hosts = sd.query_devices(), sd.query_hostapis()
    index, device = resolve_microphone(devices, hosts, args.mic, default=sd.default.device[0])
    if not 0 <= args.channel < device['max_input_channels']:
        raise ValueError('Input channel unavailable')
    # Prefer the device's native clock; some Windows virtual devices advertise
    # 16 kHz through check_input_settings but fail when the stream is opened.
    rate = args.capture_rate or int(device['default_samplerate'])
    try:
        sd.check_input_settings(device=index, channels=args.channel + 1, samplerate=rate, dtype='float32')
    except sd.PortAudioError:
        if args.capture_rate:
            raise
        rate = 16000
        sd.check_input_settings(device=index, channels=args.channel + 1, samplerate=rate, dtype='float32')
    engine, resampler = pipeline.live_stream(), CaptureResampler(rate)
    chunks, overflow = queue.Queue(maxsize=25), [False]
    def emit(event):
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)
    def callback(indata, frames, timing, status):
        if status:
            overflow[0] = True
        try:
            chunks.put_nowait(indata[:, args.channel].copy())
        except queue.Full:
            overflow[0] = True
    print(json.dumps({'microphone': device['name'], 'index': index, 'capture_rate': rate,
                      'model_rate': 16000, 'channel': args.channel}), file=sys.stderr, flush=True)
    started = time.monotonic()
    deadline = started + getattr(args, 'seconds', 0) if getattr(args, 'seconds', 0) else None
    debug_samples, blocks, overflows, peak = 0, 0, 0, 0.
    with sd.InputStream(device=index, samplerate=rate, channels=args.channel + 1,
                        dtype='float32', blocksize=resampler.size, callback=callback) as stream:
        emit(engine.ready())
        while stream.active and (deadline is None or time.monotonic() < deadline):
            try:
                chunk = chunks.get(timeout=3)
            except queue.Empty as exc:
                raise RuntimeError('Microphone disconnected or stalled') from exc
            if overflow[0]:
                engine.reset(); resampler.reset(); overflows += 1
                while not chunks.empty():
                    try: chunks.get_nowait()
                    except queue.Empty: break
                overflow[0] = False
                emit({'status': 'rejected', 'reason': 'audio_overflow', 'wake_reset': True})
                continue
            blocks += 1
            peak = max(peak, float(np.max(np.abs(chunk))))
            for event in engine.push(resampler.push(chunk)):
                emit(event)
            debug_samples += len(chunk)
            if getattr(args, 'vad_debug', False) and debug_samples >= rate:
                emit({'stage': 'capture', 'status': 'levels', 'state': engine.state,
                      'peak': float(np.max(np.abs(chunk))), 'wake_score': engine.wake.score if engine.wake else None,
                      'speech_probability': engine.vad.probability})
                debug_samples %= rate
    emit({'stage': 'capture', 'status': 'stopped', 'blocks': blocks, 'overflows': overflows,
          'peak': peak, 'elapsed_seconds': time.monotonic() - started, 'audio_saved': False})
