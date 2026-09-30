import numpy as np
import pytest


def test_real_onnx_nonzero_cache_parity_for_all_fixed_frame_sizes(tmp_path):
    import torch
    from vcm.model import CausalVCM, export_model
    from pcvcm.training import native_cpu_convolution
    from pcvcm.graph_validation import check_graph
    torch.set_num_threads(2); torch.manual_seed(53)
    model = CausalVCM()
    with native_cpu_convolution():
        export_model(model, tmp_path / 'test-only.onnx')
        result = check_graph(model, tmp_path / 'test-only.onnx')
    assert result['parameters'] == 113064
    assert result['graph_bytes'] < 1024 * 1024
    assert [r['frames'] for r in result['trials']] == [1, 3, 4, 8, 17]
    assert result['max_absolute_error'] < 2e-5
    assert result['precision'] == 'FP32'
    assert result['qualified'] is False
