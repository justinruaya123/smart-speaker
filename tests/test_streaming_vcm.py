import numpy as np
import pytest


def test_frontend_chunk_invariance_and_reset():
    from vcm.frontend import StreamingFrontend, features
    x = np.random.default_rng(7).normal(0, .1, 16231).astype('float32')
    whole = features(x)
    f = StreamingFrontend()
    parts = [f.feed(x[i:i+137]) for i in range(0, len(x), 137)]
    np.testing.assert_allclose(np.concatenate(parts, axis=1), whole, atol=1e-5)
    f.reset()
    np.testing.assert_allclose(f.feed(x), whole)
    assert features(x[:399]).shape == (40, 0)


def test_causal_cache_streaming_and_onnx(tmp_path):
    import torch
    from vcm.model import CausalVCM, export_model
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = CausalVCM().eval()
    assert sum(p.numel() for p in model.parameters()) == 113064
    x = torch.randn(1, 40, 23)
    state = model.initial_states()
    with torch.no_grad():
        full, _ = model(x, state)
        outputs = []
        for start in range(0, 23, 4):
            y, state = model(x[..., start:start+4], state)
            outputs.append(y)
        torch.testing.assert_close(torch.cat(outputs, dim=2), full)
        changed = x.clone(); changed[..., 10:] += 3
        other, _ = model(changed, model.initial_states())
        torch.testing.assert_close(full[..., :10], other[..., :10])
    receipt = export_model(model, tmp_path/'model.onnx')
    assert receipt['max_absolute_error'] < 2e-5
    assert receipt['nonzero_cache_tested']


def test_ctc_repeats_and_underflow():
    from vcm.decoder import CTCRanker
    ranker = CTCRanker([[1, 1], [1], [2]])
    logits = np.full((3, 3), -40., dtype=np.float32)
    logits[0, 1] = 0; logits[1, 0] = 0; logits[2, 1] = 0
    scores = ranker.score(logits)
    assert scores.argmax() == 0
    assert np.isfinite(scores).all()
    assert np.isfinite(ranker.score(np.tile(logits, (100, 1)))).all()


def test_policy_binding_and_endpoint():
    from vcm.acceptance import decide
    binding = {'model': 'a', 'catalog': 'b', 'frontend': 'c'}
    policy = {'enabled': True, 'binding': binding, 'min_score': -2., 'min_margin': .2}
    result = decide({'intent': 'POWER_ON'}, -1., .5, policy, binding, endpoint=False)
    assert not result['accepted'] and result['reason'] == 'tentative'
    assert decide({'intent':'POWER_ON'}, -1., .5, policy, binding, endpoint=True)['accepted']
    assert not decide({}, -1., .5, policy, {**binding, 'catalog':'d'}, endpoint=True)['accepted']
    assert not decide({}, float('nan'), .5, policy, binding, endpoint=True)['accepted']
    policy['min_margin']=0.
    assert not decide({'intent':'POWER_ON'}, -1., 0., policy, binding, endpoint=True)['accepted']


def test_capture_fault_is_sticky():
    from vcm.runtime import CaptureGuard
    g = CaptureGuard()
    assert g.check(np.zeros(12), 0)
    assert not g.check(np.zeros(12), 20)
    assert not g.check(np.zeros(12), 12)
    g.reset()
    assert g.check(np.zeros(12), 0)
    assert not g.check(np.full(3, np.nan), 12)
