"""Shared endpointing for microphone and replay, with independently selectable stages."""
from collections import deque
import numpy as np
from einops import pack, reduce


class EnergyVAD:
    def __init__(self, config, rate=16000, *, calibrate=False):
        self.config, self.rate = config, rate
        self.live_calibration = calibrate
        self.noise = .003
        self.raw_rms = self.rms = self.dc_offset = 0.
        self.last_endpoint = None
        self.silence_limit = round(rate * config["silence_ms"] / 1000)
        self.utterance_limit = round(rate * config["max_utterance_seconds"])
        self.reset(calibrate=calibrate)

    @property
    def calibrating(self):
        return self.calibration_remaining > 0

    @property
    def threshold(self):
        return max(self.config["energy_floor"], self.noise * self.config["noise_multiplier"])

    def diagnostics(self):
        return {"state": "calibrating" if self.calibrating else "speech" if self.active else "listening",
                "rms": self.raw_rms, "ac_rms": self.rms, "dc_offset": self.dc_offset,
                "noise_rms": self.noise, "threshold": self.threshold,
                "utterance_seconds": self.elapsed_samples / self.rate,
                "silence_ms": self.silence_samples * 1000 / self.rate}

    def reset(self, *, calibrate=False):
        self.preroll = deque(maxlen=max(1, round(self.config["preroll_ms"] / self.config["block_ms"])))
        self.buffer, self.active = [], False
        self.elapsed_samples = self.silence_samples = 0
        self.calibration_remaining = round(self.rate * self.config.get("calibration_ms", 1000) / 1000) if calibrate else 0
        self.calibration_levels = []

    def push(self, chunk):
        """Return (original PCM, timed_out) on an endpoint, otherwise None.

        Live capture has an ambient-only calibration preamble. A timeout starts
        fresh calibration so a raised noise floor cannot cause endless retries.
        DC removal applies only to VAD energy; model input is kept unchanged.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1 or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError("VAD expects finite mono audio")
        self.raw_rms = float(np.sqrt(reduce(chunk * chunk, "t ->", "mean")))
        self.dc_offset = float(reduce(chunk, "t ->", "mean"))
        centered = chunk - self.dc_offset
        self.rms = float(np.sqrt(reduce(centered * centered, "t ->", "mean")))
        if self.calibrating:
            self.calibration_levels.append(self.rms)
            self.calibration_remaining = max(0, self.calibration_remaining - len(chunk))
            if not self.calibrating:
                # A brief click or spoken fragment must not dominate the baseline.
                self.noise = float(np.quantile(self.calibration_levels, .2))
                self.calibration_levels.clear()
            return None
        speech = self.rms > self.threshold
        if not self.active:
            self.preroll.append(chunk.copy())
            if not speech:
                self.noise = .98 * self.noise + .02 * self.rms
                return None
            self.active, self.buffer = True, list(self.preroll)
            self.elapsed_samples = sum(len(c) for c in self.buffer)
        else:
            self.buffer.append(chunk.copy())
            self.elapsed_samples += len(chunk)
        self.silence_samples = 0 if speech else self.silence_samples + len(chunk)
        timeout = self.elapsed_samples >= self.utterance_limit
        if timeout or self.silence_samples >= self.silence_limit:
            pcm, _ = pack(self.buffer, "*")
            self.last_endpoint = self.diagnostics()
            self.reset(calibrate=timeout and self.live_calibration)
            return pcm, timeout
        return None

    def segment(self, pcm):
        """Replay without a calibration preamble; a file can start with speech."""
        self.reset()
        self.noise = .003  # Independent recordings must not inherit a noise estimate.
        size = round(self.rate * self.config["block_ms"] / 1000)
        padded = np.pad(pcm, (0, round(self.rate * (self.config["silence_ms"] + self.config["block_ms"]) / 1000)))
        output = []
        for offset in range(0, len(padded), size):
            result = self.push(padded[offset:offset + size])
            if result is not None:
                output.append(result)
        return output


class Pipeline:
    def __init__(self, bundle, config):
        self.config = config
        self.bundle = bundle
        switches = config["runtime"]
        self.vad = None
        if switches['vad']:
            if config['audio'].get('vad_backend', 'energy') == 'silero':
                from .wake_models import SileroVAD
                self.vad = SileroVAD(bundle, config['audio'])
            else:
                self.vad = EnergyVAD(config['audio'])
        self.recognizer = None
        if switches["vcm"]:
            from .inference import Recognizer
            parameters = config.get("parameters", {"enabled": False})
            self.recognizer = Recognizer(bundle, threads=config["validation"]["threads"],
                parameter_backend=parameters["backend"] if parameters["enabled"] else None)

    def utterance(self, pcm, timeout=False):
        if timeout:
            return {"status": "rejected", "reason": "utterance_timeout", "parameter_invoked": False}
        if self.recognizer:
            result = self.recognizer.predict(pcm, stream=False)
            required = {"POWER_ON": ["target"], "POWER_OFF": ["target"], "BRIGHTNESS": ["target", "level"],
                        "COLOR_SET": ["target", "color"], "MEDIA_PLAY": ["selection_text"], "VOLUME": ["level"],
                        "MEDIA_RESUME": [], "TIME_QUERY": [], "DATE_QUERY": []}
            known = result.get("intent_accepted", False) and result.get("status") != "rejected"
            family = result.get("vcm_command") if known else None
            fields = required.get(family) if known else None
            return {**result, "stage": "result", "provisional": False, "intent": family,
                    "required_parameters": fields,
                    "parameters_needed": bool(fields) and result.get("status") != "accepted" if known else None}
        return {"status": "speech_detected", "seconds": len(pcm) / 16000, "parameter_invoked": False}

    def process(self, pcm):
        """Presegmented command-file benchmark; intentionally does not infer a wake boundary."""
        if self.vad:
            segments = self.vad.segment(pcm)
            return [self.utterance(*item) for item in segments] or [
                {"status": "rejected", "reason": "no_speech", "parameter_invoked": False}]
        if not any(self.config["runtime"].values()):
            return [{"status": "disabled", "reason": "all_runtime_stages_disabled"}]
        return [self.utterance(pcm)]

    def live_stream(self):
        from .activation import WakeCommandStream
        from .wake_models import SileroVAD
        # File replay and microphone capture are mutually exclusive CLI modes.
        # Reuse the loaded VAD session instead of allocating a second graph.
        return WakeCommandStream(self, vad=self.vad if isinstance(self.vad, SileroVAD) else None)
