"""PC timing summaries and immutable-output guards, not acoustic fixtures."""
import importlib
import numpy as np
import pytest


def benchmark():
    try:
        return importlib.import_module('scripts.test.optionb_pc_benchmark')
    except ModuleNotFoundError as error:
        if error.name != 'scripts.test.optionb_pc_benchmark':
            raise
        pytest.fail('Frozen-model PC benchmark is missing')


def test_pc_timing_reports_observations_and_audio_normalized_latency():
    result = benchmark().timing_summary([2., 4., 6.], [1., 2., 3.])
    assert result['observations'] == 3
    assert result['p50_ms'] == 4.
    assert result['p95_ms'] == pytest.approx(float(np.percentile([2., 4., 6.], 95)))
    assert result['rtf_p95'] == pytest.approx(.002)


@pytest.mark.parametrize('times,durations', [([], []), ([1.], []), ([float('nan')], [1.]),
                                               ([-1.], [1.]), ([1.], [0.])])
def test_pc_timing_rejects_empty_misaligned_or_invalid_observations(times, durations):
    with pytest.raises(ValueError):
        benchmark().timing_summary(times, durations)


def test_pc_benchmark_never_overwrites_prior_output(tmp_path):
    (tmp_path / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError, match='Preserve'):
        benchmark().run_benchmark(tmp_path / 'missing-protocol.json', tmp_path)
    assert (tmp_path / 'sentinel').read_text() == 'keep'
