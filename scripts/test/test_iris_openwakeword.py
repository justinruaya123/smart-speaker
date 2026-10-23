"""Real CPU ONNX characterization; optional cached public assets, never downloads."""
from importlib.util import find_spec
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / '.cache/iris-openwakeword-publication-v1/models'


def test_additive_adapter_exists_and_cpu_loads():
    if not ASSETS.exists():
        pytest.skip('Run python -m scripts.iris_openwakeword_publication acquire for public hash-pinned assets')
    assert find_spec('scripts.runtime.iris_openwakeword') is not None, 'Missing additive adapter'
    from scripts.runtime.iris_openwakeword import HeyIrisOpenWakeWord
    model = HeyIrisOpenWakeWord(ASSETS)
    assert all(s.get_providers() == ['CPUExecutionProvider'] for s in model.sessions.values())
    assert model.sessions['hey_iris.onnx'].get_inputs()[0].shape[1:] == [16, 96]
    assert model.push(np.zeros(640, np.float32)) is None
    assert model.push(np.zeros(640, np.float32)) == 0.0
    assert model.samples == 1280
    model.reset()
    assert model.samples == 0


def test_invalid_pcm_is_rejected_without_advancing():
    if not ASSETS.exists():
        pytest.skip('Public assets not acquired')
    from scripts.runtime.iris_openwakeword import HeyIrisOpenWakeWord
    model = HeyIrisOpenWakeWord(ASSETS)
    for invalid in (np.zeros((2, 3)), np.array([np.nan]), np.array([np.inf]),
                    np.array([1.01]), np.zeros(16001), np.zeros(3, np.int16)):
        with pytest.raises(ValueError, match='PCM'):
            model.push(invalid)
        assert model.samples == 0


def test_identity_fails_before_onnx_load(tmp_path):
    from scripts.runtime.iris_openwakeword import HeyIrisOpenWakeWord
    (tmp_path / 'hey_iris.onnx').write_bytes(b'not a model')
    with pytest.raises(ValueError, match='identity'):
        HeyIrisOpenWakeWord(tmp_path)


def test_upstream_reference_numeric_and_chunk_parity():
    if not ASSETS.exists():
        pytest.skip('Public assets not acquired')
    from scripts.runtime.iris_openwakeword import HeyIrisOpenWakeWord
    from scripts.test.iris_openwakeword_reference import UpstreamFeatureReference
    model = HeyIrisOpenWakeWord(ASSETS)
    reference = UpstreamFeatureReference(model.sessions)
    pcm = np.random.RandomState(42).randint(-10000, 10000, 1280 * 150).astype(np.int16)
    pcm[:32000] = 0
    expected, actual = [], []
    for start in range(0, len(pcm), 1280):
        expected.append(reference.step(pcm[start:start + 1280]))
    # Non-aligned chunks prove NEW samples are consumed once, with bounded state.
    for start in range(0, len(pcm), 731):
        model.push(pcm[start:start + 731].astype(np.float32) / 32768)
        actual.extend(v for _, v in model.last_scores)
        assert len(model.pending) < 1280 and len(model.raw) <= 480
        assert len(model.mels) == 76 and len(model.embeddings) == 16
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    model.reset()
    repeat = []
    for start in range(0, len(pcm), 1280):
        model.push(pcm[start:start + 1280].astype(np.float32) / 32768)
        repeat.extend(v for _, v in model.last_scores)
    np.testing.assert_array_equal(actual, repeat)


def test_publication_gate_rejects_missing_or_duplicate_cases():
    assert find_spec('scripts.iris_openwakeword_publication') is not None, 'Missing locked evaluator'
    from scripts.iris_openwakeword_publication import validate_membership
    cases = [{'id': 'one'}, {'id': 'two'}]
    validate_membership(cases, cases)
    for rows in ([], cases[:1], [cases[0], cases[0]], cases[::-1]):
        with pytest.raises(ValueError, match='cohort'):
            validate_membership(cases, rows)


def test_recorded_regression_is_bound_and_never_promoted():
    import hashlib
    import json
    experiment = ROOT / 'experiments/iris-openwakeword-publication-v1'
    if not (experiment / 'results.json').exists():
        pytest.skip('No completed immutable regression receipt')
    from scripts.iris_openwakeword_publication import COHORTS, PROTOCOL_SHA
    from scripts.iris_wake_publication import panel_metrics
    assert hashlib.sha256((experiment / 'protocol.json').read_bytes()).hexdigest() == PROTOCOL_SHA
    lock = json.loads((experiment / 'selection-before-regression.json').read_text())
    result = json.loads((experiment / 'results.json').read_text())
    for path, expected in lock['source_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert lock['candidate_pcm_decoded_or_inferred_before_lock'] is False
    assert lock['policy'] == {'threshold': .5, 'release': .3, 'consecutive': 2, 'cooldown_samples': 16000}
    assert result['promoted'] is False and result['installed'] is False and result['qualified'] is False
    assert len(result['panels']) == len(COHORTS) == 2
    for (name, filename, split, protocol_hash), panel in zip(COHORTS, result['panels']):
        source = ROOT / 'experiments/iris-wake-publication-v1' / filename
        assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol_hash
        cases = [r for r in json.loads(source.read_text())['cases'] if r['split'] == split]
        path = experiment / (name + '-cases.json')
        rows = json.loads(path.read_text())
        assert len(rows) == len(cases) == len({r['id'] for r in rows}) == 100
        assert hashlib.sha256(path.read_bytes()).hexdigest() == panel['cases_sha256']
        assert panel_metrics(cases, rows) == panel['metrics']
        assert sum(c['wake_positive'] for c in cases) == 32
        assert panel['metrics']['negative_n'] == 68
    assert result['gate_passed'] == all(p['metrics']['synthetic_smoke_pass'] for p in result['panels'])


def test_real_score_traces_reconstruct_every_wake():
    import hashlib
    import json
    experiment = ROOT / 'experiments/iris-openwakeword-publication-v1'
    if not (experiment / 'results.json').exists() or not ASSETS.exists():
        pytest.skip('Original local score traces are deliberately not redistributed')
    result = json.loads((experiment / 'results.json').read_text())
    for panel in result['panels']:
        trace_path = ASSETS.parent / (panel['name'] + '-scores.json')
        if not trace_path.exists():
            pytest.skip('Private/local score trace unavailable')
        assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == panel['score_trace_sha256']
        traces = json.loads(trace_path.read_text())
        rows = json.loads((experiment / (panel['name'] + '-cases.json')).read_text())
        assert len(traces) == len(rows) == 100
        for trace, row in zip(traces, rows):
            assert trace['id'] == row['id']
            hits, run, cooldown, latched = [], 0, 0, False
            scores = trace['samples_scores']
            assert len(scores) == row['score_count']
            assert [t for t, _ in scores] == list(range(1280, 1280 * (len(scores) + 1), 1280))
            for sample, value in scores:
                assert np.isfinite(value) and 0 <= value <= 1
                cooldown = max(0, cooldown - 1280)
                if value < .3 and cooldown == 0:
                    latched = False
                run = run + 1 if value >= .5 else 0
                if run >= 2 and not latched and cooldown == 0:
                    hits.append((sample - 32000) / 16000)
                    latched, cooldown, run = True, 16000, 0
            assert hits == row['hit_seconds']
            assert len(hits) == row['hits']
            assert max(v for _, v in scores) == row['max_score']
