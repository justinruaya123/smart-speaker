"""Mechanism fixtures only: no claim of acoustic or speaker accuracy."""
import importlib
import importlib.util

import numpy as np
import pytest
import torch


def helper():
    name = 'scripts.runtime.pcv_enrollment_owner_v2'
    assert importlib.util.find_spec(name), 'Helper is not implemented'
    return importlib.import_module(name)


def model():
    name = 'scripts.models.pcv_streaming_v2'
    assert importlib.util.find_spec(name), 'Model is not implemented'
    torch.manual_seed(231)
    return importlib.import_module(name).StreamingVCM().eval()


def test_deployed_outputs_have_no_transcription_or_phone_inventory():
    net = model()
    result = net.full(torch.randn(1, 40, 71))
    assert set(result) == {
        'semantic', 'percent', 'intent', 'operation', 'target',
                           'light', 'song', 'artist', 'entity_queries',
                           'acoustic', 'speaker_frames'}
    assert result['entity_queries'].shape == (1, 3, 64)
    assert result['acoustic'].shape == (1, 64, 71)
    assert result['speaker_frames'].shape == (1, 64, 71)


def test_speaker_loss_does_not_update_content_encoder():
    net = model()
    frames = net.full(torch.randn(2, 40, 71))['speaker_frames']
    frames.square().sum().backward()
    assert all(p.grad is None for n, p in net.named_parameters()
               if not n.startswith('speaker.'))
    assert all(p.grad is not None for p in net.speaker.parameters())


def test_content_loss_does_not_update_speaker_tower():
    net = model()
    net.full(torch.randn(2, 40, 71))['semantic'].square().sum().backward()
    assert all(p.grad is None for p in net.speaker.parameters())


def bank(entries=None):
    h = helper()
    entries = entries or [('light.kitchen', np.array([1., 0., 0.])),
                          ('light.bedroom', np.array([0., 1., 0.]))]
    return h.ExemplarBank('light', entries, 'encoder-1', 'frontend-1')


def test_enrollment_does_not_modify_input_or_query_arrays():
    v = np.array([2., 0., 0.])
    b = bank([('light.kitchen', v), ('light.bedroom', np.array([0., 1., 0.]))])
    v[:] = 999.
    result = b.rank(np.array([1., 0., 0.]))
    assert result['id'] == 'light.kitchen'
    assert result['similarity'] == pytest.approx(1.)


def test_duplicate_aliases_do_not_inflate_evidence():
    b = bank()
    duplicate = bank([('light.kitchen', np.array([1., 0., 0.]))] * 15 +
                     [('light.bedroom', np.array([0., 1., 0.]))])
    q = np.array([0.8, 0.6, 0.])
    assert b.rank(q) == duplicate.rank(q)


def test_phonetically_indistinguishable_ids_are_ambiguous():
    b = bank([('light.a', np.array([1., 0., 0.])),
              ('light.b', np.array([1., 0., 0.]))])
    assert b.rank(np.array([1., 0., 0.]))['margin'] == 0.
    assert not b.accept(np.array([1., 0., 0.]), .8, .1, b.binding,
                        acoustic_pass=True, qualified=True)['accepted']


def test_single_name_bank_has_no_rival_evidence():
    b = bank([('light.a', np.array([1., 0., 0.]))])
    assert b.rank(np.array([1., 0., 0.]))['margin'] == 0.


@pytest.mark.parametrize('acoustic,qualified', [(False, True), (True, False),
                                               (False, False)])
def test_perfect_template_cannot_override_other_rejection(acoustic, qualified):
    b = bank()
    out = b.accept(np.array([1., 0., 0.]), .8, .1, b.binding,
                   acoustic_pass=acoustic, qualified=qualified)
    assert not out['accepted']


def test_unknown_name_cannot_be_forced_to_nearest_id():
    b = bank()
    assert not b.accept(np.array([0., 0., 1.]), .8, .1, b.binding,
                        acoustic_pass=True, qualified=True)['accepted']


def test_enrollment_change_invalidates_calibration():
    a = bank()
    b = bank([('light.kitchen', np.array([.9, .1, 0.])),
              ('light.bedroom', np.array([0., 1., 0.]))])
    with pytest.raises(ValueError, match='binding'):
        b.accept(np.array([1., 0., 0.]), .8, .1, a.binding,
                 acoustic_pass=True, qualified=True)


@pytest.mark.parametrize('v', [np.array([0., 0., 0.]),
                               np.array([np.nan, 0., 0.]),
                               np.array([np.inf, 1., 0.]), np.zeros((2, 3))])
def test_invalid_enrollment_rejected(v):
    with pytest.raises(ValueError):
        bank([('light.a', v)])


def test_wrong_role_cannot_enter_bank():
    with pytest.raises(ValueError, match='role'):
        bank([('song.a', np.array([1., 0., 0.]))])


def test_bank_budget_is_bounded():
    with pytest.raises(ValueError, match='budget'):
        bank([('light.a', np.ones(3))] * 97)


def owner(mode='same_speaker', qualified=True):
    return helper().OwnerGuard(mode, .75, min_reference_ms=600,
                               min_command_ms=400, qualified=qualified)


def test_owner_reference_is_not_poisoned_by_later_voice():
    guard = owner()
    reference = np.array([1., 0., 0.])
    guard.arm(reference, 800)
    reference[:] = 999
    guard.observe(np.array([0., 1., 0.]), 500)
    assert not guard.finish()['pass']
    assert np.array_equal(guard.reference, np.array([1., 0., 0.]))


def test_unfamiliar_but_same_speaker_does_not_need_saved_identity():
    guard = owner()
    guard.arm(np.array([.2, .7, -.3]), 800)
    guard.observe(np.array([.2, .7, -.3]), 500)
    assert guard.finish()['pass']


def test_intruder_cannot_be_hidden_in_long_average():
    guard = owner()
    guard.arm(np.array([1., 0., 0.]), 800)
    for _ in range(10):
        guard.observe(np.array([1., 0., 0.]), 100)
    guard.observe(np.array([0., 1., 0.]), 100)
    for _ in range(10):
        guard.observe(np.array([1., 0., 0.]), 100)
    assert not guard.finish()['pass']


def test_overlap_and_unknown_are_fail_closed():
    for overlap, embedding in [(True, np.array([1., 0., 0.])), (False, None)]:
        guard = owner()
        guard.arm(np.array([1., 0., 0.]), 800)
        guard.observe(embedding, 500, overlap=overlap)
        assert not guard.finish()['pass']


def test_short_wake_or_command_is_insufficient_not_confirmed():
    guard = owner()
    guard.arm(np.array([1., 0., 0.]), 100)
    guard.observe(np.array([1., 0., 0.]), 500)
    assert not guard.finish()['pass']
    guard.arm(np.array([1., 0., 0.]), 800)
    guard.observe(np.array([1., 0., 0.]), 100)
    assert not guard.finish()['pass']


def test_official_spliced_benchmark_has_explicit_owner_bypass():
    guard = owner('official_spliced')
    guard.arm(np.array([1., 0., 0.]), 800)
    guard.observe(np.array([0., 1., 0.]), 500)
    out = guard.finish()
    assert out['pass'] and out['owner_enforced'] is False
    assert out['reason'] == 'official_spliced_not_owner_test'


def test_unqualified_speaker_model_never_certifies_same_speaker():
    guard = owner(qualified=False)
    guard.arm(np.array([1., 0., 0.]), 800)
    guard.observe(np.array([1., 0., 0.]), 500)
    assert not guard.finish()['pass']


def test_silence_is_not_a_new_speaker_or_evidence():
    guard = owner()
    guard.arm(np.array([1., 0., 0.]), 800)
    guard.observe(None, 200, voiced=False)
    guard.observe(np.array([1., 0., 0.]), 500)
    assert guard.finish()['pass']


def test_mode_cannot_be_implicitly_auto_selected():
    with pytest.raises(ValueError):
        owner('auto')


def test_mono_overlap_is_not_identifiable_from_mixture_alone():
    rng = np.random.default_rng(4)
    a, b, perturbation = rng.normal(size=(3, 2000))
    np.testing.assert_allclose(a + b, (a + perturbation) + (b - perturbation),
                               atol=1e-12)
    assert np.linalg.norm(perturbation) > 1


@pytest.mark.parametrize('flag', ['true', 1, None])
def test_string_or_integer_qualification_is_not_authorization(flag):
    name = 'scripts.runtime.pcv_streaming_contract_v2'
    contract = importlib.import_module(name)
    with pytest.raises(ValueError):
        contract.EventGate('demo', 'hash', qualified=flag).start('u')


def test_owner_required_cannot_be_bypassed_by_semantic_confidence():
    contract = importlib.import_module(
        'scripts.runtime.pcv_streaming_contract_v2')
    guard = contract.EventGate('demo', 'hash', qualified=True,
                               ownership_required=True)
    guard.start('u')
    result = guard.finish({'intent': 'PLAY_MUSIC', 'slots': {}},
                          endpoint_confirmed=True, policy_pass=True,
                          owner_pass=False)
    assert not result['accepted'] and result['reason'] == 'speaker_unverified'


def test_explicit_unknown_light_never_defaults_to_all():
    contract = importlib.import_module(
        'scripts.runtime.pcv_streaming_contract_v2')
    guard = contract.EventGate('demo', 'hash', qualified=True)
    guard.start('u')
    command = {'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.all'}}
    result = guard.finish(command,
                          endpoint_confirmed=True, policy_pass=True,
                          explicit_target_unresolved=True)
    assert not result['accepted']


def test_audio_sequence_gaps_and_duplicate_chunks_fault():
    h = helper()
    tracker = h.AudioSequence()
    tracker.feed(0, 320)
    tracker.feed(320, 640)
    for offset in (320, 961):
        with pytest.raises(ValueError, match='sequence'):
            tracker.feed(offset, 320)
    tracker.reset()
    tracker.feed(0, 1)


def test_speaker_frame_chunk_equivalence():
    net = model()
    x = torch.randn(1, 40, 273)
    parts = []
    state = net.initial_state()
    with torch.inference_mode():
        reference = net.full(x)['speaker_frames']
        for start in range(0, x.shape[-1], 7):
            out, state = net.step(x[..., start:start + 7], state)
            parts.append(out['speaker_frames'])
    torch.testing.assert_close(torch.cat(parts, -1), reference,
                               atol=1e-5, rtol=2e-5)


def test_all_slot_heads_and_query_vectors_equal_full_sequence():
    net = model()
    x = torch.randn(1, 40, 101)
    state = net.initial_state()
    with torch.inference_mode():
        reference = net.full(x)
        for start in range(0, 101, 3):
            out, state = net.step(x[..., start:start + 3], state)
    for key in ('light', 'song', 'artist', 'entity_queries'):
        torch.testing.assert_close(out[key], reference[key],
                                   atol=1e-5, rtol=2e-5)


def test_reference_with_overlap_cannot_arm_owner_mode():
    guard = owner()
    guard.arm(np.array([1., 0., 0.]), 1000, overlap=True)
    guard.observe(np.array([1., 0., 0.]), 500)
    assert not guard.finish()['pass']


@pytest.mark.parametrize('command', [
    {'intent': 'TIMER', 'slots': {'duration_seconds': 73}},
    {'intent': 'CALL', 'slots': {'contact_id': 'contact.unknown'}},
    {'intent': 'TEMPERATURE', 'slots': {'degrees': 25, 'unit': 'C'}},
    {'intent': 'COLOR',
     'slots': {'light_id': 'light.kitchen', 'color': 'cyan'}},
])
def test_demo_does_not_silently_expand_untrained_use_cases(command):
    contract = importlib.import_module(
        'scripts.runtime.pcv_streaming_contract_v2')
    assert not contract.profile_allows('demo', command)
