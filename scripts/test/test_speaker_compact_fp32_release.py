import importlib

import pytest


def module():return importlib.import_module('scripts.train.speaker_compact_fp32_release')


def receipt(*,fidelity=False,numerical=True):
    return dict(fidelity={'pass':fidelity,'extra_human_negative_errors':2},
                export={'fp32_numerical_parity_pass':numerical})


def test_fp32_fallback_preserves_failed_int8_and_parameter_limit():
    original=receipt();result=module().fallback_gate(original,3864875,935567)
    assert result['precision']=='fp32' and result['parameters']==935567
    assert result['int8_fidelity_passed'] is False
    assert original['fidelity']['pass'] is False


@pytest.mark.parametrize('export,size,parameters',[
    (receipt(fidelity=True),3864875,935567),
    (receipt(numerical=False),3864875,935567),
    (receipt(),4*1024*1024,935567),
    (receipt(),3864875,1200001)])
def test_fp32_fallback_cannot_waive_quality_size_or_parameter_guards(export,size,parameters):
    with pytest.raises(ValueError):module().fallback_gate(export,size,parameters)
