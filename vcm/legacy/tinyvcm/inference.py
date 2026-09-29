"""Offline CPU inference and a causal intent-to-parameter handoff."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from einops import rearrange
import onnxruntime as ort

from .audio import features, load_audio
from .bundle import load_runtime_contract, runtime_components, validate_outputs
from .routing import PARAMETER_COMMANDS, parse_parameters


def softmax(values, temperature=1.0):
    values = np.asarray(values, dtype=np.float64) / temperature
    exponents = np.exp(values - values.max())
    return exponents / exponents.sum()


class Recognizer:
    def __init__(self, bundle, threads=2, variant="fp32"):
        self.bundle = Path(bundle)
        self.metadata = load_runtime_contract(self.bundle, variant)
        self.labels = self.metadata["vcm"]["labels"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.options = options
        self.variant = variant
        self.vcm = ort.InferenceSession(str(self.bundle / f"vcm.{variant}.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        validate_outputs(self.vcm, self.metadata)
        self.stream = ort.InferenceSession(str(self.bundle / f"vcm_stream.{variant}.onnx"), sess_options=options, providers=["CPUExecutionProvider"]) if "vcm_stream" in runtime_components(self.metadata) else None
        self.catalog = self.metadata.get("catalog", {"entities": []})

    def routing_policy(self):
        policy = self.metadata.get("routing", {})
        return {**policy, **policy.get("variants", {}).get(self.variant, {})}

    def normalize(self, mel, task="vcm"):
        stats = self.metadata[task]["normalization"]
        mean = rearrange(np.array(stats["mean"], dtype=np.float32), "m -> m 1")
        std = rearrange(np.array(stats["std"], dtype=np.float32), "m -> m 1")
        return rearrange((mel - mean) / std, "m t -> 1 m t")

    def classify(self, pcm, *, temperature=None):
        if not len(pcm):
            raise ValueError("An utterance must contain audio samples")
        mel = self.normalize(features(pcm))
        logits = self.vcm.run(["logits"], {"features": mel, "lengths": np.array([mel.shape[-1]], dtype=np.int64)})[0]
        return softmax(logits[0], self.routing_policy().get("temperature", 1.0) if temperature is None else temperature)

    def classify_with_slots(self, pcm, *, temperature=None):
        if not len(pcm):
            raise ValueError("An utterance must contain audio samples")
        mel = features(pcm)
        x = self.normalize(mel)
        names = list(self.metadata["acoustic_slots"]["vocabulary"])
        outputs = self.vcm.run(["logits", *["slot_" + name for name in names]], {"features": x, "lengths": np.array([x.shape[-1]], dtype=np.int64)})
        slots = {name: softmax(value[0]) for name, value in zip(names, outputs[1:])}
        return softmax(outputs[0][0], self.routing_policy().get("temperature", 1.0) if temperature is None else temperature), slots, mel

    def stream_events(self, pcm):
        state = self.begin_stream()
        self.advance_stream(pcm, state, final=True)
        return state["events"]

    def begin_stream(self):
        shapes = [] if self.metadata.get("acoustic_slots") or not self.metadata["vcm"].get("streaming_supported", True) else self.metadata["vcm"]["state_shapes"]
        fills = self.metadata["vcm"].get("state_initial_values", [0.0] * len(shapes))
        return {"states": [np.full(shape, fill, dtype=np.float32) for shape, fill in zip(shapes, fills)], "frame": 0, "previous": None, "stable": 0, "events": []}

    def advance_stream(self, pcm, state, final=False):
        """Process only new frames; live capture retains 80 ms for resampling edges."""
        policy = self.routing_policy()
        if self.metadata.get("acoustic_slots") or not self.metadata.get("vcm", {}).get("streaming_supported", True) or policy.get("trigger_threshold", .9) > 1:
            return []  # A disabled provisional trigger cannot produce an event.
        mel = self.normalize(features(pcm))
        states = state["states"]
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
            probabilities = softmax(logits[0], policy.get("temperature", 1.0))
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



    def predict(self, pcm, parameter_start_ms=None, stream=True, events=None):
        started = time.perf_counter()
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            return {"status": "rejected", "reason": "invalid_audio"}
        if not 0.1 <= len(pcm) / 16000 <= 8.2:
            return {"status": "rejected", "reason": "audio_duration_outside_supported_range"}
        slot_config = self.metadata.get("acoustic_slots")
        slot_probabilities, shared_mel, slot_decision = None, None, None
        if slot_config and slot_config.get("enabled", True):
            probabilities, slot_probabilities, shared_mel = self.classify_with_slots(pcm)
        else:
            probabilities = self.classify(pcm)
        index = int(probabilities.argmax())
        command, confidence = self.labels[index], float(probabilities[index])
        policy = self.routing_policy()
        threshold = policy.get("acceptance_thresholds", {}).get(command, policy.get("acceptance_threshold", 0.7))
        if slot_probabilities is not None and command != "UNKNOWN" and confidence >= threshold:
            from .slot_contract import route_acoustic
            slot_decision = route_acoustic(command, slot_probabilities, slot_config, self.catalog, self.variant, intent_confidence=confidence)
        if command == "UNKNOWN" or confidence < threshold:
            return {"status": "rejected", "reason": "unknown_or_low_confidence", "vcm_command": command, "confidence": confidence, "intent_accepted": False, "processing_ms": (time.perf_counter()-started)*1000}
        events = events if events is not None else self.stream_events(pcm) if stream else []
        event = events[0] if events else None
        result = {"vcm_command": command, "confidence": confidence, "events": events,
                  "intent_accepted": True, "intent_acceptance_source": "intent"}
        if event and event["vcm_command"] != command:
            result.update(status="rejected", reason="early_intent_retracted", intent_accepted=False)
        elif slot_decision is not None and slot_decision["decision"] == "clarify":
            result.update(status="needs_clarification", reason=slot_decision["reason"], parameter_source="vcm")
        elif command not in PARAMETER_COMMANDS:
            result.update(parse_parameters(command, "", self.catalog))
            result["parameter_source"] = "none"
        else:
            if parameter_start_ms is not None and not 0 <= parameter_start_ms / 1000 < len(pcm) / 16000:
                raise ValueError("Parameter boundary is outside the recording")
            if slot_probabilities is not None:
                result["acoustic_parameters"] = {k: v for k, v in slot_decision.items() if k in {"decision", "reason", "candidate"}}
                if slot_decision["decision"] in {"accepted", "clarify"}:
                    result.update({k: v for k, v in slot_decision.items() if k in {"status", "command", "reason"}})
                    result.update(parameter_source="vcm", processing_ms=(time.perf_counter()-started)*1000)
                    return result
            result.update(status="needs_parameters", parameter_source="vcm",
                          reason="catalog_decoder_required" if command == "MEDIA_PLAY"
                          else "finite_parameters_require_clarification")
        result["processing_ms"] = (time.perf_counter()-started)*1000
        return result
