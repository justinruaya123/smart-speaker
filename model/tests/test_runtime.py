import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tinyvcm.cli import resolve_microphone
from tinyvcm.inference import Recognizer
from tinyvcm.routing import parse_parameters


CATALOG = {"entities": [
    {"id": "lamp.1", "name": "Blue Moon lamp", "kind": "light", "aliases": []},
    {"id": "lamp.2", "name": "lamp two", "kind": "light", "aliases": []},
    {"id": "plug.1", "name": "wemo plug", "kind": "switch", "aliases": []},
]}


def test_asr_tail_cannot_reverse_the_vcm_command():
    result = parse_parameters("POWER_ON", "turn off the Blue Moon lamp", CATALOG)
    assert result["status"] == "needs_clarification"
    assert result["vcm_command"] == "POWER_ON"
    result = parse_parameters("POWER_ON", "Blue Moon lamp", CATALOG)
    assert result["command"]["arguments"]["on"] is True
    assert result["command"]["action"] == "SET_POWER"


def test_catalog_normalizes_apostrophes_and_accented_spelling_consistently():
    catalog = {"entities": [{"id": "lamp", "name": "René's lamp", "kind": "light"}]}
    assert parse_parameters("COLOR_SET", "rene s lamp red", catalog)["command"]["arguments"]["color"] == "red"


def test_empty_media_asr_does_not_become_a_default_play_action():
    assert parse_parameters("MEDIA_PLAY", "")["status"] == "needs_clarification"
    assert parse_parameters("MEDIA_PLAY", "music")["status"] == "accepted"


def test_configurable_names_do_not_become_color_or_numeric_arguments():
    result = parse_parameters("COLOR_SET", "Blue Moon lamp red", CATALOG)
    assert result["command"]["arguments"]["color"] == "red"
    result = parse_parameters("BRIGHTNESS_DOWN", "lamp two", CATALOG)
    assert result["command"]["arguments"]["delta_percentage_points"] == -10


@pytest.mark.parametrize("command,text", [
    ("POWER_ON", "unknown lamp"),
    ("POWER_OFF", "Blue Moon lamp and lamp two"),
    ("COLOR_SET", "wemo plug red"),
    ("COLOR_SET", "Blue Moon lamp red or green"),
    ("POWER_ON", "Blue Moon lamp but cancel"),
])
def test_unsafe_or_ambiguous_parameter_bindings_abstain(command, text):
    assert parse_parameters(command, text, CATALOG)["status"] == "needs_clarification"


def test_microphone_selection_obeys_index_name_and_host_api():
    devices = [{"name": "USB mic", "max_input_channels": 1, "hostapi": 0},
               {"name": "USB mic", "max_input_channels": 1, "hostapi": 1},
               {"name": "Speakers", "max_input_channels": 0, "hostapi": 0}]
    hosts = [{"name": "ALSA"}, {"name": "PulseAudio"}]
    assert resolve_microphone(devices, hosts, "0")[0] == 0
    assert resolve_microphone(devices, hosts, "USB", "ALSA")[0] == 0
    assert resolve_microphone(devices, hosts, default=1)[0] == 1
    for selection in ("USB", "2", "missing"):
        with pytest.raises(ValueError):
            resolve_microphone(devices, hosts, selection)


def test_intent_detection_operates_with_asr_disabled():
    recognizer = Recognizer.__new__(Recognizer)
    recognizer.labels = ["UNKNOWN", "POWER_ON"]
    recognizer.metadata = {"routing": {"acceptance_threshold": 0.7, "lookback_seconds": 0.5}}
    recognizer.catalog = CATALOG
    recognizer.asr_enabled = False
    recognizer.classify = lambda pcm: np.array([0.01, 0.99])
    recognizer.stream_events = lambda pcm: [{"vcm_command": "POWER_ON", "audio_time_s": 0.6}]
    recognizer.transcribe = lambda pcm: pytest.fail("ASR must not run when disabled")
    result = recognizer.predict(np.zeros(16000, dtype=np.float32))
    assert result["vcm_command"] == "POWER_ON"
    assert result["status"] == "needs_parameters"
    assert result["parameter_audio"]["start_sample_16k"] == 1600


def test_low_confidence_never_calls_asr():
    recognizer = Recognizer.__new__(Recognizer)
    recognizer.labels = ["UNKNOWN", "POWER_ON"]
    recognizer.metadata = {"routing": {"acceptance_threshold": 0.7}}
    recognizer.classify = lambda pcm: np.array([0.49, 0.51])
    recognizer.stream_events = lambda pcm: pytest.fail("Rejected input must not trigger ASR")
    assert recognizer.predict(np.zeros(16000, dtype=np.float32))["status"] == "rejected"


def test_runtime_import_does_not_require_pytorch():
    code = """import sys, importlib.abc
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('PyTorch is unavailable for this runtime test')
sys.meta_path.insert(0, BlockTorch())
import tinyvcm.inference, tinyvcm.cli
assert 'torch' not in sys.modules
print('runtime_without_torch_ok')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "runtime_without_torch_ok" in result.stdout
