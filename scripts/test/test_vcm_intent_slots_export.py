import json
import numpy as np
import pytest
import torch

from scripts.models.vcm_intent_slots import IntentSlotsVCM
from scripts.test.test_vcm_intent_slots_contract import contract
from scripts.train.vcm_intent_slots_export import OrderedHeads, export_candidate
from scripts.runtime.vcm_intent_slots_runtime import IntentSlotsRuntime, validate_pcm


@pytest.mark.parametrize("n", [1, 3, 19, 32, 101])
def test_unpadded_export_rewrite_matches_masked_training(n):
    model = IntentSlotsVCM(contract()).eval()
    x = torch.randn(1, 40, n)
    out = model(x, torch.tensor([n]))
    ordered = OrderedHeads(model)(x)
    torch.testing.assert_close(ordered[0], out["intent_logits"])
    torch.testing.assert_close(ordered[1], out["role_state_logits"].flatten(1))
    torch.testing.assert_close(ordered[2], out["request_validity_logits"])
    torch.testing.assert_close(ordered[3], torch.cat(list(out["value_logits"].values()), dim=1))
    torch.testing.assert_close(ordered[4], out["entity_embeddings"].flatten(1))


@pytest.mark.parametrize("pcm,sr", [
    (np.zeros(399), 16000), (np.zeros(128001), 16000),
    (np.zeros((400, 1)), 16000), (np.zeros(400), 8000),
    (np.full(400, np.nan), 16000), (np.full(400, 2.), 16000),
])
def test_pcm_rejected(pcm, sr):
    with pytest.raises(ValueError):
        validate_pcm(pcm, sr)


def test_export_and_onnx_decision_parity_and_corrupt_bundle(tmp_path):
    torch.manual_seed(53)
    c = contract()
    model = IntentSlotsVCM(c).eval()
    checkpoint = tmp_path/"checkpoint.pt"
    torch.save({"weights": model.state_dict(), "contract": c,
                "normalization": {"mean": [0.]*40, "std": [1.]*40, "fit_split": "train"},
                "status": "numerical_fixture_not_trained"}, checkpoint)
    bundle = tmp_path/"bundle"
    receipt = export_candidate(checkpoint, c, bundle, engines=["onnxruntime"])
    assert receipt["engine_status"]["onnxruntime"] == "exported"
    runtime = IntentSlotsRuntime(bundle, "onnxruntime", 1)
    pcm = np.random.default_rng(53).normal(0, .05, 16000).astype("float32")
    event = runtime.recognize(pcm)
    assert not event["final_application"]["accepted"]
    assert event["actions_enabled"] is False
    assert event["benchmark"]["intent"] == "OUT_OF_SCOPE"
    (bundle/"normalization.json").write_text("{}")
    with pytest.raises(ValueError, match="binding|hash"):
        IntentSlotsRuntime(bundle, "onnxruntime", 1)


@pytest.mark.parametrize("kind,score,accepted", [
    ("intent", .9499, False), ("intent", .9501, True),
    ("slot", .8999, False), ("slot", .9001, True),
])
def test_actual_exported_engines_near_threshold_decisions(tmp_path, kind, score, accepted):
    pytest.importorskip("ncnn")
    pytest.importorskip("pnnx")
    c = contract()
    model = IntentSlotsVCM(c).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        for head in model.states:
            head.bias[0] = 12
        model.validity.bias[0] = 12
        intent = "STOP" if kind == "intent" else "VOLUME"
        model.intent.bias[c["intent_order"].index(intent)] = (
            np.log(18*score/(1-score)) if kind == "intent" else 12)
        if kind == "slot":
            model.states[c["roles"].index("volume_operation")].bias[:] = torch.tensor([0, 12, 0, 0])
            model.values["volume_operation"].bias[0] = np.log(3*score/(1-score))
    checkpoint = tmp_path/"fixture.pt"
    torch.save({"weights": model.state_dict(), "contract": c,
                "normalization": {"mean": [0.]*40, "std": [1.]*40, "fit_split": "train"},
                "status": "controlled_numeric_threshold_fixture"}, checkpoint)
    bundle = tmp_path/"bundle"
    receipt = export_candidate(checkpoint, c, bundle)
    assert receipt["engine_status"]["ncnn"] == "exported_unverified"
    pcm = np.zeros(400, np.float32)
    events = [IntentSlotsRuntime(bundle, engine, 1).recognize(pcm)
              for engine in ("pytorch", "onnxruntime", "ncnn")]
    assert all(e["final_application"]["accepted"] is accepted for e in events)
    assert all(e["final_application"]["command"] == events[0]["final_application"]["command"] for e in events)
