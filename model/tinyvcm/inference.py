"""Offline CPU inference and a causal intent-to-parameter handoff."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from einops import rearrange
import onnxruntime as ort

from .audio import features, load_audio
from .routing import PARAMETER_COMMANDS, parameter_slice, parse_parameters
from .text import decode_greedy


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    exponents = np.exp(values - values.max())
    return exponents / exponents.sum()


class Recognizer:
    def __init__(self, bundle, asr=False, threads=2, variant="fp32"):
        self.bundle = Path(bundle)
        self.metadata = json.loads((self.bundle / "bundle.json").read_text(encoding="utf-8"))
        self.labels = self.metadata["vcm"]["labels"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.options = options
        self.variant = variant
        self.vcm = ort.InferenceSession(str(self.bundle / f"vcm.{variant}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.stream = ort.InferenceSession(str(self.bundle / f"vcm_stream.{variant}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.asr_enabled = asr
        self.asr = None
        self.catalog = self.metadata.get("catalog", {"entities": []})

    def normalize(self, mel, task="vcm"):
        stats = self.metadata[task]["normalization"]
        mean = rearrange(np.array(stats["mean"], dtype=np.float32), "m -> m 1")
        std = rearrange(np.array(stats["std"], dtype=np.float32), "m -> m 1")
        return rearrange((mel - mean) / std, "m t -> 1 m t")

    def classify(self, pcm):
        mel = self.normalize(features(pcm))
        logits = self.vcm.run(None, {"features": mel, "lengths": np.array([mel.shape[-1]], dtype=np.int64)})[0]
        return softmax(logits[0])

    def stream_events(self, pcm):
        state = self.begin_stream()
        self.advance_stream(pcm, state, final=True)
        return state["events"]

    def begin_stream(self):
        shapes = self.metadata["vcm"]["state_shapes"]
        fills = self.metadata["vcm"].get("state_initial_values", [0.0] * len(shapes))
        return {"states": [np.full(shape, fill, dtype=np.float32) for shape, fill in zip(shapes, fills)], "frame": 0, "previous": None, "stable": 0, "events": []}

    def advance_stream(self, pcm, state, final=False):
        """Process only new frames; live capture retains 80 ms for resampling edges."""
        mel = self.normalize(features(pcm))
        states = state["states"]
        policy = self.metadata.get("routing", {})
        threshold = policy.get("trigger_threshold", 0.9)
        required = policy.get("stability_chunks", 3)
        minimum = policy.get("minimum_audio_seconds", 0.5)
        chunk = 4
        previous, stable = state["previous"], state["stable"]
        events = state["events"]
        before = len(events)
        available = mel.shape[-1] if final else max(0, (mel.shape[-1] - 8) // chunk * chunk)
        for frame in range(state["frame"], available, chunk):
            x = mel[:, :, frame:min(frame + chunk, available)]
            if x.shape[-1] == 0:
                break
            feed = {"features": x, **{f"state_{i}": state for i, state in enumerate(states)}}
            logits, *states = self.stream.run(None, feed)
            probabilities = softmax(logits[0])
            index = int(probabilities.argmax())
            command, confidence = self.labels[index], float(probabilities[index])
            timestamp = (frame + x.shape[-1] - 1) * 0.01 + 0.025
            stable = (stable + 1 if command == previous else 1) if confidence >= threshold else 0
            previous = command
            if not events and command != "UNKNOWN" and confidence >= threshold and stable >= required and timestamp >= minimum:
                events.append({"stage": "intent", "vcm_command": command, "confidence": confidence, "audio_time_s": timestamp, "provisional": True})
            state["frame"] = frame + x.shape[-1]
        state.update(states=states, previous=previous, stable=stable)
        return events[before:]

    def transcribe(self, pcm):
        if self.asr is None:
            path = self.bundle / f"asr.{self.variant}.onnx"
            if not path.exists():
                raise RuntimeError("Parameter ASR is not trained/exported yet; use --asr off")
            self.asr = ort.InferenceSession(str(path), sess_options=self.options, providers=["CPUExecutionProvider"])
        logits = self.asr.run(None, {"features": self.normalize(features(pcm), "asr")})[0]
        return decode_greedy(logits[:, 0, :].argmax(-1))

    def predict(self, pcm, parameter_start_ms=None, stream=True, events=None):
        started = time.perf_counter()
        if not 0.1 <= len(pcm) / 16000 <= 8.2:
            return {"status": "rejected", "reason": "audio_duration_outside_supported_range"}
        probabilities = self.classify(pcm)
        index = int(probabilities.argmax())
        command, confidence = self.labels[index], float(probabilities[index])
        threshold = self.metadata.get("routing", {}).get("acceptance_threshold", 0.7)
        if command == "UNKNOWN" or confidence < threshold:
            return {"status": "rejected", "reason": "unknown_or_low_confidence", "vcm_command": command, "confidence": confidence, "processing_ms": (time.perf_counter()-started)*1000}
        events = events if events is not None else self.stream_events(pcm) if stream else []
        event = events[0] if events else None
        result = {"vcm_command": command, "confidence": confidence, "events": events}
        if event and event["vcm_command"] != command:
            result.update(status="rejected", reason="early_intent_retracted")
        elif command not in PARAMETER_COMMANDS:
            result.update(parse_parameters(command, "", self.catalog))
        else:
            if parameter_start_ms is not None:
                if not 0 <= parameter_start_ms / 1000 < len(pcm) / 16000:
                    raise ValueError("Parameter boundary is outside the recording")
                start = int(round(parameter_start_ms * 16))
                tail, policy = pcm[start:], "explicit_boundary"
            elif event:
                tail, start = parameter_slice(pcm, event["audio_time_s"], self.metadata.get("routing", {}).get("lookback_seconds", 0.5))
                policy = "trigger_with_lookback_unverified_boundary"
            else:
                result.update(status="needs_parameters", reason="no_early_trigger_to_define_parameter_slice")
                tail, start, policy = pcm[:0], len(pcm), "unavailable"
            result["parameter_audio"] = {"start_sample_16k": start, "end_sample_16k": len(pcm), "policy": policy, "timing_verified": False, "boundary_source": "user_supplied" if parameter_start_ms is not None else "predicted"}
            if len(tail) and self.asr_enabled:
                text = self.transcribe(tail)
                result["parameter_transcript"] = text
                result.update(parse_parameters(command, text, self.catalog))
            elif len(tail):
                result.update(status="needs_parameters", reason="asr_disabled")
        result["processing_ms"] = (time.perf_counter()-started)*1000
        return result
