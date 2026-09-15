"""Contract and capture-loop tests; no real microphone or ONNX files required."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from einops import repeat
import numpy as np
import sounddevice as sd

from src.voice import start_service
from src.voice.__main__ import main
from src.voice.service import REQUIRED_PARAMETERS, _DebugRecognizer, describe_intent
from tinyvcm.routing import ACTION, PARAMETER_COMMANDS


DEVICES = [
    {"name": "Speakers", "max_input_channels": 0, "hostapi": 0, "default_samplerate": 48000},
    {"name": "Test mic", "max_input_channels": 2, "hostapi": 0, "default_samplerate": 48000},
]


class FakeRecognizer:
    def __init__(self, commands=("COLOR_SET", "TIME_QUERY")):
        self.commands = commands
        self.states = []
        self.recordings = []
        self.advances = []

    def begin_stream(self):
        state = {"events": []}
        self.states.append(state)
        return state

    def advance_stream(self, pcm, state, final=False):
        self.advances.append((len(pcm), final))
        if len(pcm) < 8000 or state["events"]:
            return []
        event = {"stage": "intent", "vcm_command": self.commands[len(self.recordings)],
                 "confidence": 0.9999, "provisional": True, "audio_time_s": 0.6}
        state["events"].append(event)
        return [event]

    def predict(self, pcm, *, events):
        command = self.commands[len(self.recordings)]
        self.recordings.append(pcm.copy())
        return {"vcm_command": command, "confidence": 0.999,
                "status": "needs_parameters" if command in PARAMETER_COMMANDS else "accepted",
                "events": events}


class IntentDebugTests(unittest.TestCase):
    def test_requirements_cover_the_existing_router(self):
        self.assertEqual(set(REQUIRED_PARAMETERS), set(ACTION))
        self.assertEqual({key for key, fields in REQUIRED_PARAMETERS.items() if fields}, PARAMETER_COMMANDS)

    def test_rejected_candidates_never_become_detected_intents(self):
        for command, status in (("POWER_ON", "rejected"), ("UNKNOWN", "rejected")):
            with self.subTest(command=command):
                result = describe_intent({"vcm_command": command, "status": status})
                self.assertEqual(result["vcm_command"], command)
                self.assertIsNone(result["intent"])
                self.assertIsNone(result["parameters_needed"])
                self.assertIsNone(result["required_parameters"])

    def test_stream_debugging_preserves_provisional_state(self):
        raw = FakeRecognizer()
        adapter = _DebugRecognizer(raw)
        with redirect_stdout(io.StringIO()):
            state = adapter.begin_stream()
        event = adapter.advance_stream(np.zeros(16000, dtype=np.float32), state)[0]
        self.assertTrue(event["provisional"])
        self.assertTrue(event["parameters_needed"])
        self.assertEqual(event["required_parameters"], ["target", "color"])
        self.assertNotIn("intent", state["events"][0])
        result = adapter.predict(np.zeros(16000, dtype=np.float32), events=state["events"])
        self.assertEqual(result["stage"], "result")
        self.assertFalse(result["provisional"])


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        for name in ("bundle.json", "vcm.fp32.onnx", "vcm_stream.fp32.onnx"):
            (self.bundle / name).touch()

    def run_capture(self, levels, *, fallback=False, interrupt=False):
        """Drive the real tinyvcm listener's callback and VAD with 40 ms blocks."""
        recognizer = FakeRecognizer()
        blocks = iter(levels)
        capture = {}

        class InputStream:
            def __init__(self, **settings):
                self.settings = settings
                capture.update(settings)
                capture["closed"] = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                capture["closed"] = True

            @property
            def active(self):
                try:
                    value = next(blocks)
                except StopIteration:
                    if interrupt:
                        raise KeyboardInterrupt
                    return False
                level, status = value if isinstance(value, tuple) else (value, False)
                pcm = np.full(self.settings["blocksize"], level, dtype=np.float32)
                data = repeat(pcm, "samples -> samples channels", channels=self.settings["channels"])
                self.settings["callback"](data, len(pcm), None, status)
                return True

        def check_settings(**settings):
            if fallback and settings["samplerate"] == 16000:
                raise sd.PortAudioError("16 kHz unsupported by test device")

        output, errors = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            factory = stack.enter_context(patch("tinyvcm.inference.Recognizer", return_value=recognizer))
            stack.enter_context(patch("sounddevice.query_devices", return_value=DEVICES))
            stack.enter_context(patch("sounddevice.query_hostapis", return_value=[{"name": "Test API"}]))
            stack.enter_context(patch("sounddevice.check_input_settings", side_effect=check_settings))
            stack.enter_context(patch("sounddevice.InputStream", InputStream))
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(errors))
            start_service(1, bundle=self.bundle)
        factory.assert_called_once_with(self.bundle, asr=False)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["status"], "starting")
        self.assertEqual(events[-1]["status"], "stopped")
        self.assertTrue(capture["closed"])
        self.assertEqual(capture["device"], 1)
        return recognizer, events, capture

    def test_silence_never_starts_intent_recognition(self):
        recognizer, events, _ = self.run_capture([0.0] * 30)
        self.assertEqual(recognizer.states, [])
        self.assertEqual(recognizer.advances, [])
        self.assertEqual(len(events), 2)

    def test_two_utterances_have_preroll_final_results_and_fresh_state(self):
        utterance = [0.0] * 8 + [0.05] * 15 + [0.0] * 12
        recognizer, events, _ = self.run_capture(utterance * 2)
        self.assertEqual([event["stage"] for event in events],
                         ["service", "vad", "intent", "result", "vad", "intent", "result", "service"])
        results = [event for event in events if event["stage"] == "result"]
        self.assertEqual(results[0]["intent"], "COLOR_SET")
        self.assertEqual(results[0]["required_parameters"], ["target", "color"])
        self.assertFalse(results[1]["parameters_needed"])
        self.assertEqual(results[1]["required_parameters"], [])
        self.assertIsNot(recognizer.states[0], recognizer.states[1])
        self.assertTrue(np.all(recognizer.recordings[0][:4 * 640] == 0))
        self.assertTrue(np.all(recognizer.recordings[0][4 * 640:5 * 640] > 0))
        np.testing.assert_array_equal(*recognizer.recordings)

    def test_native_capture_rate_is_resampled_before_inference(self):
        recognizer, _, capture = self.run_capture([0.0] * 8 + [0.05] * 15 + [0.0] * 12, fallback=True)
        self.assertEqual(capture["samplerate"], 48000)
        self.assertGreater(len(recognizer.recordings[0]), 16000)
        self.assertLess(len(recognizer.recordings[0]), 24000)

    def test_overflow_discards_partial_audio_and_recovers(self):
        levels = [0.05] * 8 + [(0.05, True)] + [0.0] * 8 + [0.05] * 15 + [0.0] * 12
        recognizer, events, _ = self.run_capture(levels)
        self.assertEqual(len(recognizer.states), 2)
        self.assertEqual(len(recognizer.recordings), 1)
        self.assertEqual(sum(event.get("reason") == "audio_overflow" for event in events), 1)

    def test_utterance_limit_rejects_instead_of_classifying_truncated_audio(self):
        recognizer, events, _ = self.run_capture([0.05] * 200 + [0.0] * 12)
        self.assertEqual(recognizer.recordings, [])
        self.assertTrue(any(event.get("reason") == "utterance_timeout" for event in events))

    def test_keyboard_interrupt_closes_capture(self):
        _, events, _ = self.run_capture([0.0] * 3, interrupt=True)
        self.assertEqual(events[-1]["status"], "stopped")

    def test_numeric_input_and_bundle_validation(self):
        for invalid in (-1, True, "1", 1.5):
            with self.subTest(microphone=invalid), self.assertRaisesRegex(ValueError, "integer device index"):
                start_service(invalid, bundle=self.bundle)
        with self.assertRaisesRegex(FileNotFoundError, "Missing model file"):
            start_service(1, bundle=self.bundle / "missing")


class CommandLineTests(unittest.TestCase):
    def test_missing_portaudio_library_reports_a_startup_error(self):
        original_import = __import__

        def unavailable(name, *args, **kwargs):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return original_import(name, *args, **kwargs)

        errors = io.StringIO()
        with patch("builtins.__import__", side_effect=unavailable), redirect_stderr(errors):
            self.assertEqual(main(["--list-mics"]), 1)
        self.assertIn("PortAudio library not found", errors.getvalue())

    def test_list_preserves_device_indices_and_filters_output_devices(self):
        output = io.StringIO()
        with patch("sounddevice.query_devices", return_value=DEVICES), \
             patch("sounddevice.query_hostapis", return_value=[{"name": "Test API"}]), \
             patch("src.voice.__main__.start_service") as start, redirect_stdout(output):
            self.assertEqual(main(["--list-mics"]), 0)
        start.assert_not_called()
        self.assertIn("1: Test mic", output.getvalue())
        self.assertNotIn("Speakers", output.getvalue())

    def test_microphone_errors_return_nonzero(self):
        errors = io.StringIO()
        with patch("src.voice.__main__.start_service", side_effect=sd.PortAudioError("device unavailable")), \
             redirect_stderr(errors):
            self.assertEqual(main(["--mic", "1"]), 1)
        self.assertEqual(json.loads(errors.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
