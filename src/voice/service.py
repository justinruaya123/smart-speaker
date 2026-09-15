"""Reuse TinyVCM's capture/VAD loop and report intent parameter requirements."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


DEFAULT_BUNDLE = Path(__file__).resolve().parents[2] / "model/artifacts/tinyvcm"

# User-supplied fields in tinyvcm.routing. Values implied by the intent (such as
# POWER_ON's `on=True`) are not additional parameters. Relative brightness has
# an optional amount, with a default of ten percentage points in the router.
REQUIRED_PARAMETERS = {
    "POWER_ON": ("target",),
    "POWER_OFF": ("target",),
    "BRIGHTNESS_SET": ("target", "percent"),
    "BRIGHTNESS_UP": ("target",),
    "BRIGHTNESS_DOWN": ("target",),
    "COLOR_SET": ("target", "color"),
    "VOLUME_SET": ("percent",),
    "MEDIA_PLAY": ("selection_text",),
    "MEDIA_PAUSE": (),
    "MEDIA_RESUME": (),
    "MEDIA_STOP": (),
    "TIME_QUERY": (),
    "DATE_QUERY": (),
    "TIMER_SET": ("duration_s",),
}


def _emit(**event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


def describe_intent(result):
    """Add debug fields without promoting a rejected candidate to an intent.

    Parameter fields are null for rejected/unknown predictions, false/[] for
    parameterless intents, and true/[names] for intents needing user arguments.
    The service leaves parameter transcription disabled.
    """
    command = result.get("vcm_command")
    known = result.get("status") != "rejected" and command in REQUIRED_PARAMETERS
    required = list(REQUIRED_PARAMETERS[command]) if known else None
    return {
        **result,
        "intent": command if known else None,
        "parameters_needed": bool(required) if known else None,
        "required_parameters": required,
    }


class _DebugRecognizer:
    """Adapt the existing listener's recognizer interface, preserving its state."""

    def __init__(self, recognizer):
        self.recognizer = recognizer

    def begin_stream(self):
        # The capture loop calls this only after its VAD detects speech.
        _emit(stage="vad", status="speech_started")
        return self.recognizer.begin_stream()

    def advance_stream(self, pcm, state, final=False):
        return [
            describe_intent(event)
            for event in self.recognizer.advance_stream(pcm, state, final=final)
        ]

    def predict(self, pcm, *, events):
        result = self.recognizer.predict(pcm, events=events)
        return describe_intent({**result, "stage": "result", "provisional": False})


def start_service(
    microphone: int,
    *,
    bundle: str | Path = DEFAULT_BUNDLE,
    channel: int = 0,
    capture_rate: int | None = None,
) -> None:
    """Listen on a numeric sounddevice input index until Ctrl+C or disconnect.

    TinyVCM owns channel selection, rate conversion, energy VAD, pre-roll,
    streaming recognition, endpoint detection, overflow handling and cleanup.
    Model paths are independent of the caller's current working directory.
    """
    if type(microphone) is not int or microphone < 0:
        raise ValueError("microphone must be a non-negative integer device index")
    if type(channel) is not int or channel < 0:
        raise ValueError("channel must be a non-negative integer")
    if capture_rate is not None and (type(capture_rate) is not int or capture_rate <= 0):
        raise ValueError("capture_rate must be a positive integer")
    bundle = Path(bundle).expanduser().resolve()
    for name in ("bundle.json", "vcm.fp32.onnx", "vcm_stream.fp32.onnx"):
        if not (bundle / name).is_file():
            raise FileNotFoundError(f"Missing model file: {bundle / name}; select a bundle with --bundle")

    from tinyvcm.cli import listen
    from tinyvcm.inference import Recognizer

    _emit(stage="service", status="starting", microphone=microphone, parameter_asr=False)
    recognizer = Recognizer(bundle, asr=False)
    args = SimpleNamespace(mic=microphone, host_api=None, channel=channel, capture_rate=capture_rate)
    try:
        listen(_DebugRecognizer(recognizer), args)
    except KeyboardInterrupt:
        pass
    finally:
        _emit(stage="service", status="stopped")
