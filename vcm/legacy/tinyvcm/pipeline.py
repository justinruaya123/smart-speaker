"""Shared endpointing for microphone and replay, with independently selectable stages."""
from collections import deque
import numpy as np
from einops import pack, reduce


class EnergyVAD:
    def __init__(self, config, rate=16000):
        self.config, self.rate = config, rate
        self.noise = .003
        self.reset()

    def reset(self):
        self.preroll = deque(maxlen=max(1, round(self.config["preroll_ms"] / self.config["block_ms"])))
        self.buffer, self.active, self.elapsed, self.silence = [], False, 0., 0.

    def push(self, chunk):
        """Return (PCM, timed_out) on an endpoint, otherwise None."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 1 or not len(chunk) or not np.isfinite(chunk).all():
            raise ValueError("VAD expects finite mono audio")
        duration = len(chunk) / self.rate
        rms = float(np.sqrt(reduce(chunk * chunk, "t ->", "mean")))
        speech = rms > max(self.config["energy_floor"], self.noise * self.config["noise_multiplier"])
        if not self.active:
            self.preroll.append(chunk.copy())
            if not speech:
                self.noise = .98 * self.noise + .02 * rms
                return None
            self.active, self.buffer = True, list(self.preroll)
            self.elapsed = sum(len(c) for c in self.buffer) / self.rate
        else:
            self.buffer.append(chunk.copy())
            self.elapsed += duration
        self.silence = 0. if speech else self.silence + duration
        timeout = self.elapsed + 1e-9 >= self.config["max_utterance_seconds"]
        if timeout or self.silence * 1000 + 1e-9 >= self.config["silence_ms"]:
            pcm, _ = pack(self.buffer, "*")
            self.reset()
            return pcm, timeout
        return None

    def segment(self, pcm):
        """Replay as a stream, including trailing silence to close the last utterance."""
        self.reset()
        self.noise = .003  # Independent recordings must not inherit another recording's noise estimate.
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
        switches = config["runtime"]
        self.vad = EnergyVAD(config["audio"]) if switches["vad"] else None
        self.recognizer = None
        if switches["vcm"]:
            from .inference import Recognizer
            self.recognizer = Recognizer(bundle, threads=config["validation"]["threads"])

    def utterance(self, pcm, timeout=False):
        if timeout:
            return {"status": "rejected", "reason": "utterance_timeout"}
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
        return {"status": "speech_detected", "seconds": len(pcm) / 16000}

    def process(self, pcm):
        if self.vad:
            segments = self.vad.segment(pcm)
            return [self.utterance(*item) for item in segments] or [
                {"status": "rejected", "reason": "no_speech"}]
        if not any(self.config["runtime"].values()):
            return [{"status": "disabled", "reason": "all_runtime_stages_disabled"}]
        return [self.utterance(pcm)]
