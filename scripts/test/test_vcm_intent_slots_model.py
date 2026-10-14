"""Numerical architecture, padding and supervised-loss contracts."""
import pytest
import torch
import torch.nn.functional as F

from scripts.test.test_vcm_intent_slots_contract import contract
from scripts.models.vcm_intent_slots import IntentSlotsVCM, joint_loss


@pytest.mark.parametrize("frames", [1, 2, 3, 17, 32, 101])
def test_exact_layout_and_finite_gradients(frames):
    c = contract()
    model = IntentSlotsVCM(c)
    out = model(torch.randn(2, 40, frames), torch.tensor([frames, frames]))
    assert out["intent_logits"].shape == (2, 19)
    assert out["role_state_logits"].shape == (2, len(c["roles"]), 4)
    assert out["request_validity_logits"].shape == (2, 4)
    assert out["entity_embeddings"].shape == (2, 5, 64)
    assert out["role_frame_logits"].shape == (2, (frames+1)//2, len(c["roles"])+1)
    for role, values in c["finite_values"].items():
        assert out["value_logits"][role].shape == (2, len(values))
    out["intent_logits"].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(model.blocks) == 8
    assert [b.depthwise.dilation[0] for b in model.blocks] == [1, 2, 4, 8, 1, 2, 4, 8]


@pytest.mark.parametrize("length", [1, 3, 20, 43])
def test_padding_and_other_batch_member_invariant(length):
    torch.manual_seed(53)
    model = IntentSlotsVCM(contract()).eval()
    x = torch.randn(1, 40, length)
    base = model(x, torch.tensor([length]))
    padded = torch.cat([x, torch.randn(1, 40, 65-length)*100], dim=2)
    batched = torch.cat([padded, torch.randn(1, 40, 65)], dim=0)
    out = model(batched, torch.tensor([length, 65]))
    for key in ("intent_logits", "role_state_logits", "request_validity_logits", "entity_embeddings"):
        torch.testing.assert_close(base[key][0], out[key][0], atol=3e-6, rtol=1e-5)
    for role in base["value_logits"]:
        torch.testing.assert_close(base["value_logits"][role][0], out["value_logits"][role][0], atol=3e-6, rtol=1e-5)


@pytest.mark.parametrize("length", [0, -1, 9])
def test_invalid_or_all_padding_rejected(length):
    with pytest.raises(ValueError):
        IntentSlotsVCM(contract())(torch.randn(1, 40, 8), torch.tensor([length]))


def test_unknown_labels_spans_and_pairs_masked():
    model = IntentSlotsVCM(contract())
    out = model(torch.randn(2, 40, 13), torch.tensor([13, 7]))
    target = {"intent": torch.tensor([1, 0]),
              "role_states": torch.full((2, len(model.contract["roles"])), -100),
              "validity": torch.tensor([-100, -100]), "values": {},
              "entity_ids": [[None]*5]*2, "speaker_ids": ["a", "b"],
              "spans": []}
    loss, detail = joint_loss(out, target)
    torch.testing.assert_close(loss, F.cross_entropy(out["intent_logits"], target["intent"]))
    assert detail["entity_pairs"] == detail["reviewed_spans"] == detail["observed_values"] == 0
    loss.backward()
    assert torch.isfinite(loss)


def test_numeric_twenty_sixty_hundred_are_distinct():
    c = contract()
    assert len({c["finite_values"]["brightness_percent"].index(n) for n in (20, 60, 100)}) == 3
    assert c["roles"].index("volume_percent") != c["roles"].index("brightness_percent")
