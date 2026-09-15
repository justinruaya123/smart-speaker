import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from einops import pack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinyvcm.audio import features, StreamingFrontend, resample_audio
from tinyvcm.data import command_label
from tinyvcm.vcm import VCM
from tinyvcm.asr import encode_text, decode_greedy


@pytest.mark.parametrize("pooling", ["mean", "mean_max"])
def test_causal_vcm_matches_cached_stream_and_has_no_future_dependence(pooling):
    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = VCM(5, width=16, kernels=(5, 7), pooling=pooling).eval()
    x = torch.randn(1, 40, 48)
    with torch.inference_mode():
        expected = model.prefix_logits(x)
        states = model.initial_state()
        for offset in range(0, 48, 4):
            actual, states = model.step(x[:, :, offset:offset + 4], states)
            torch.testing.assert_close(actual, expected[:, offset + 3, :], atol=2e-6, rtol=2e-5)
        changed = x.clone()
        changed[:, :, 24:] += 100
        torch.testing.assert_close(expected[:, :24], model.prefix_logits(changed)[:, :24])


def test_streaming_features_equal_file_features_for_irregular_capture_chunks():
    rng = np.random.default_rng(17)
    wave = rng.normal(0, 0.01, 12037).astype(np.float32)
    frontend = StreamingFrontend()
    frames = [frontend.push(wave[i:i + 317]) for i in range(0, len(wave), 317)]
    actual, _ = pack(frames, "m *")
    np.testing.assert_allclose(actual, features(wave), rtol=2e-5, atol=2e-5)


def test_audio_rate_is_converted_not_relabelled():
    wave = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
    actual = resample_audio(wave, 48000)
    assert len(actual) == 16000
    peak = np.fft.rfftfreq(len(actual), 1 / 16000)[np.argmax(abs(np.fft.rfft(actual)))]
    assert peak == pytest.approx(440)


@pytest.mark.parametrize("text,source,expected", [
    ("turn on the lamp", "iot_hue_lighton", "POWER_ON"),
    ("turn off the lamp", "iot_hue_lightoff", "POWER_OFF"),
    ("turn on the Blue Moon lamp", "iot_hue_lighton", "POWER_ON"),
    ("the wemo plug should be turned off on", "iot_wemo_off", "UNKNOWN"),
    ("pause the music", "audio_volume_mute", "MEDIA_PAUSE"),
    ("set an alarm for seven", "alarm_set", "UNKNOWN"),
    ("set a timer for ten minutes", "alarm_set", "TIMER_SET"),
    ("do not turn on the lights", "iot_hue_lighton", "UNKNOWN"),
    ("play the beatles", "play_music", "MEDIA_PLAY"),
    ("play a podcast", "play_podcasts", "UNKNOWN"),
    ("lower the brightness to thirty percent", "iot_hue_lightdim", "BRIGHTNESS_SET"),
    ("dim the lights by ten percent", "iot_hue_lightdim", "BRIGHTNESS_DOWN"),
])
def test_semantic_mapping_preserves_spoken_distinctions(text, source, expected):
    assert command_label(text, source) == expected


def test_ctc_repeated_characters_require_a_blank():
    indices = encode_text("anna")
    assert decode_greedy(indices) == "ana"
    assert decode_greedy([indices[0], indices[1], 0, indices[2], indices[3]]) == "anna"


def test_asr_valid_frames_ignore_other_batch_padding():
    from tinyvcm.asr import ParameterASR
    torch.manual_seed(7)
    model = ParameterASR(width=16, kernels=(5, 7), repeat=2, mask_padding=True).eval()
    short = torch.randn(1, 40, 35)
    padded = torch.nn.functional.pad(short, (0, 45))
    with torch.inference_mode():
        expected = model(short)
        actual = model(padded, torch.tensor([35]))[:18]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_training_augmentation_preserves_the_readonly_feature_cache():
    import random
    from tinyvcm import train
    array = np.random.default_rng(3).normal(size=(40, 80)).astype(np.float32)
    expected = array.copy()
    array.flags.writeable = False
    train._FEATURE_MEMORY['unit/cache'] = array
    try:
        row = {'id': 'unit/cache', 'duration': .8}
        stats = {'mean': [0.] * 40, 'std': [1.] * 40}
        batch, lengths = train.make_batch([row], [0], stats, augment=True, rng=random.Random(17))
        assert int(lengths[0]) == 80
        np.testing.assert_array_equal(array, expected)
    finally:
        train._FEATURE_MEMORY.pop('unit/cache')
