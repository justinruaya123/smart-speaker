"""Causal/numerical contracts; tensor probes are not recognition accuracy."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def model_module():
    name = "scripts.models.pcv_streaming_v2"
    assert importlib.util.find_spec(name) is not None, "Causal model missing"
    return importlib.import_module(name)


def new_model():
    torch.manual_seed(43)
    return model_module().StreamingVCM().eval()


def test_old_tcn_is_not_causal():
    from scripts.models.pcv_strict_candidates import StrictTinyVCM

    torch.manual_seed(43)
    model = StrictTinyVCM().eval()
    prefix = torch.randn(1, 40, 40)
    a = torch.cat([prefix, torch.zeros(1, 40, 100)], dim=-1)
    b = torch.cat([prefix, torch.randn(1, 40, 100) * 10], dim=-1)
    with torch.inference_mode():
        difference = (model(a)["ctc"][..., :20]
                      - model(b)["ctc"][..., :20]).abs().max().item()
    assert difference > 1e-4


@pytest.mark.parametrize("frames", [1, 3, 4, 17, 83, 251])
def test_arbitrary_chunks_equal_independent_full_pooling(frames):
    model = new_model()
    x = torch.randn(1, 40, frames)
    state = model.initial_state()
    outputs = []
    cursor = 0
    sizes = [1, 4, 3, 8, 2, 17]
    with torch.inference_mode():
        reference = model.full(x)
        while cursor < frames:
            count = sizes[len(outputs) % len(sizes)]
            result, state = model.step(x[..., cursor:cursor + count], state)
            outputs.append(result["acoustic"])
            cursor += count
    for key in ("semantic", "percent", "intent", "operation", "target"):
        torch.testing.assert_close(result[key], reference[key],
                                   atol=5e-6, rtol=2e-5)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), reference["acoustic"],
                               atol=5e-6, rtol=2e-5)


@pytest.mark.parametrize("prefix", [1, 7, 41, 100])
def test_future_audio_cannot_change_past_features(prefix):
    model = new_model()
    a = torch.randn(1, 40, prefix + 90)
    b = a.clone()
    b[..., prefix:] = torch.randn_like(b[..., prefix:]) * 12
    with torch.inference_mode():
        ha, _ = model.encode(a, model.initial_state()[:7])
        hb, _ = model.encode(b, model.initial_state()[:7])
    torch.testing.assert_close(ha[..., :prefix], hb[..., :prefix],
                               atol=0, rtol=0)


def test_reset_is_repeatable_and_streams_do_not_mix():
    model = new_model()
    a, b = torch.randn(1, 40, 29), torch.randn(1, 40, 31)
    with torch.inference_mode():
        first, state_a = model.step(a, model.initial_state())
        second, state_b = model.step(b, model.initial_state())
        repeated, _ = model.step(a, model.initial_state())
        continued, _ = model.step(a, state_a)
    torch.testing.assert_close(first["semantic"], repeated["semantic"],
                               atol=0, rtol=0)
    assert not torch.allclose(first["semantic"], continued["semantic"])
    assert state_a[0].data_ptr() != state_b[0].data_ptr()
    assert not torch.equal(first["semantic"], second["semantic"])


def test_streaming_states_are_fixed_size():
    model = new_model()
    state = model.initial_state()
    shapes = [tuple(x.shape) for x in state]
    with torch.inference_mode():
        for _ in range(400):
            _, state = model.step(torch.randn(1, 40, 4), state)
            assert [tuple(x.shape) for x in state] == shapes
    assert sum(x.numel() * x.element_size() for x in state) < 110_000
    assert sum(p.numel() for p in model.parameters()) < 150_000


def test_online_pooling_handles_extreme_scores():
    module = model_module()
    h = torch.randn(2, 96, 31)
    scores = torch.randn(2, 5, 31) * 1000
    m, z, v = module.initial_pool(2, 96, h.device, h.dtype)
    for start, stop in [(0, 2), (2, 17), (17, 31)]:
        pooled, (m, z, v) = module.merge_pool(
            h[..., start:stop], scores[..., start:stop], (m, z, v)
        )
    reference = torch.einsum("bqt,bct->bqc", scores.softmax(-1), h)
    torch.testing.assert_close(pooled, reference, atol=2e-5, rtol=2e-5)
    assert all(torch.isfinite(x).all() for x in (m, z, v, pooled))


@pytest.mark.parametrize("profile", ["benchmark", "demo"])
def test_flat_export_interface_matches_streaming(profile):
    model = new_model()
    wrapper = model_module().StreamingExport(model, profile)
    state = model.initial_state()
    x = torch.randn(1, 40, 4)
    with torch.inference_mode():
        heads, expected_state = model.step(x, state)
        flat = wrapper(x, *state)
    names = wrapper.head_names
    for actual, key in zip(flat[:len(names)], names, strict=True):
        expected = (heads[key].transpose(1, 2)
                    if key in {"acoustic", "speaker_frames"} else heads[key])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if profile == "benchmark":
        expected_state = expected_state[:10] + state[10:]
    for actual, expected in zip(
            flat[len(names):], expected_state, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if profile == "benchmark":
        assert names == ("semantic", "percent")


def test_backpropagates_every_head_from_random_initialization():
    model = new_model()
    result = model.full(torch.randn(2, 40, 37))
    loss = sum(x.square().mean() for x in result.values())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_empty_stream_chunk_rejected():
    model = new_model()
    with pytest.raises(ValueError, match="nonempty"):
        model.step(torch.empty(1, 40, 0), model.initial_state())


def test_invalid_profile_rejected():
    with pytest.raises(ValueError, match="profile"):
        model_module().StreamingExport(new_model(), "auto")


def contract_module():
    name = "scripts.runtime.pcv_streaming_contract_v2"
    assert importlib.util.find_spec(name), "Stream contract missing"
    return importlib.import_module(name)


def test_benchmark_volume_numbers_but_no_named_entities():
    module = contract_module()
    assert module.profile_allows("benchmark", {
        "intent": "VOLUME", "slots": {"operation": "set", "percent": 73}
    })
    assert not module.profile_allows("benchmark", {
        "intent": "LIGHT_ON", "slots": {"light_id": "light.kitchen"}
    })
    assert not module.profile_allows("benchmark", {
        "intent": "PLAY_MUSIC", "slots": {"song_id": "song.butter"}
    })


@pytest.mark.parametrize("value", [-1, 101, 2.5, True, None, "73"])
def test_invalid_percent_is_never_clamped(value):
    module = contract_module()
    assert not module.profile_allows("benchmark", {
        "intent": "VOLUME", "slots": {"operation": "set", "percent": value}
    })


def test_partial_seventy_can_change_to_seventy_three_without_execution():
    gate = contract_module().EventGate(
        "benchmark", "model-hash", qualified=True)
    gate.start("utterance-1")
    a = {"intent": "VOLUME", "slots": {"operation": "set", "percent": 70}}
    b = {"intent": "VOLUME", "slots": {"operation": "set", "percent": 73}}
    event = gate.partial(a, 640)
    assert not event["accepted"] and not event["actionable"]
    assert gate.partial(b, 1280)["candidate"] == b
    final = gate.finish(b, endpoint_confirmed=True, policy_pass=True)
    assert final["accepted"] and final["command"] == b
    assert gate.finish(b, endpoint_confirmed=True, policy_pass=True) is None


def test_unqualified_model_cannot_enable_demo_acceptance():
    gate = contract_module().EventGate("demo", "model-hash", qualified=False)
    gate.start("u")
    out = gate.finish({"intent": "PLAY_MUSIC", "slots": {}},
                      endpoint_confirmed=True, policy_pass=True)
    assert not out["accepted"] and out["reason"] == "unqualified_model"


def test_missing_endpoint_and_backwards_sample_sequence_rejected():
    gate = contract_module().EventGate("benchmark", "hash", qualified=True)
    gate.start("u")
    command = {"intent": "PLAY_MUSIC", "slots": {}}
    with pytest.raises(ValueError, match="endpoint"):
        gate.finish(command, endpoint_confirmed=False, policy_pass=True)
    gate.partial(command, 1280)
    with pytest.raises(ValueError, match="sequence"):
        gate.partial(command, 640)




@pytest.mark.parametrize("size", [1, 127, 160, 321, 512, 640])
def test_existing_frontend_chunk_parity(size):
    from vcm.frontend import StreamingFrontend, features

    pcm = np.random.default_rng(89).normal(size=16087).astype("float32")
    pcm *= 0.05
    frontend = StreamingFrontend()
    chunks = [frontend.feed(pcm[i:i + size])
              for i in range(0, len(pcm), size)]
    actual = np.concatenate(chunks, axis=1)
    np.testing.assert_allclose(actual, features(pcm), atol=2e-6, rtol=1e-6)
    assert len(frontend.pending) < 400


def test_pcm_to_heads_matches_whole_waveform_computation():
    from vcm.frontend import StreamingFrontend, features

    pcm = (np.random.default_rng(8).normal(size=32657) * 0.1).astype("float32")
    model = new_model()
    frontend = StreamingFrontend()
    state = model.initial_state()
    with torch.inference_mode():
        reference = model.full(torch.from_numpy(features(pcm)).unsqueeze(0))
        for start in range(0, len(pcm), 512):
            x = frontend.feed(pcm[start:start + 512])
            if x.shape[-1]:
                result, state = model.step(torch.from_numpy(x).unsqueeze(0),
                                           state)
    for key in ("semantic", "percent", "intent", "operation", "target"):
        torch.testing.assert_close(result[key], reference[key],
                                   atol=8e-6, rtol=2e-5)


def test_training_batch_padding_does_not_change_valid_heads():
    model = new_model()
    x = torch.randn(2, 40, 73)
    lengths = torch.tensor([17, 73])
    with torch.inference_mode():
        batch = model.full(x, lengths)
        one = model.full(x[:1, :, :17])
    for key in ("semantic", "percent", "intent", "operation", "target"):
        torch.testing.assert_close(batch[key][:1], one[key],
                                   atol=5e-6, rtol=2e-5)


@pytest.mark.parametrize("profile", ["benchmark", "demo"])
def test_torch_trace_warm_state_and_final_fragment(profile):
    model = new_model()
    wrapper = model_module().StreamingExport(model, profile)
    state = model.initial_state()
    traced = torch.jit.trace(wrapper, (torch.randn(1, 40, 4), *state))
    with torch.inference_mode():
        for frames in [4, 1, 3, 7, 4]:
            x = torch.randn(1, 40, frames)
            expected = wrapper(x, *state)
            actual = traced(x, *state)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, atol=8e-6, rtol=2e-5)
            state = actual[-14:]


@pytest.mark.parametrize("value", list(range(101)))
def test_every_integer_volume_is_representable(value):
    assert contract_module().profile_allows("benchmark", {
        "intent": "VOLUME", "slots": {"operation": "set", "percent": value}
    })


def test_benchmark_does_not_claim_extra_timer_or_brightness_values():
    module = contract_module()
    assert not module.profile_allows("benchmark", {
        "intent": "TIMER", "slots": {"duration_seconds": 73}
    })
    assert not module.profile_allows("benchmark", {
        "intent": "BRIGHTNESS",
        "slots": {"light_id": "light.all", "percent": 73}
    })
