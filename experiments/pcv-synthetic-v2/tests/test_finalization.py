import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision
import orchestrate
from final_report import parameter_metrics


def test_controller_handoff_requires_both_authorization_and_success(tmp_path,monkeypatch):
    monkeypatch.setattr(revision,'CACHE',tmp_path)
    with pytest.raises(RuntimeError):orchestrate.require_transition()
    receipt=tmp_path/'controller-transition-complete.json'
    revision.write_json(receipt,dict(controller_stopped=True,user_authorized=False))
    with pytest.raises(ValueError):orchestrate.require_transition()
    revision.write_json(receipt,dict(controller_stopped=True,user_authorized=True))
    assert orchestrate.require_transition()['controller_stopped']


def test_parameter_metrics_require_typed_exact_values_and_matching_intent():
    expected={'intent':'BRIGHTNESS','entity_id':'light.kitchen','value':1}
    rows=[dict(expected_command=expected,candidate={**expected,'value':True},output={'accepted':True,'command':{**expected,'value':True}}),
          dict(expected_command=expected,candidate={**expected,'intent':'VOLUME_ABSOLUTE'},output={'accepted':False,'command':None}),
          dict(expected_command=expected,candidate=expected,output={'accepted':True,'command':expected})]
    metrics=parameter_metrics(rows)
    assert metrics['value']==dict(trials=3,raw_correct=1,accepted_correct=1)
    assert metrics['entity_id']==dict(trials=3,raw_correct=2,accepted_correct=2)


def test_dead_original_generator_is_a_failure_not_an_infinite_wait(tmp_path,monkeypatch):
    monkeypatch.setattr(revision,'OLD_CACHE',tmp_path)
    assert not orchestrate.original_ready('qwen',None,alive=lambda _:True)
    with pytest.raises(RuntimeError,match='exited without'):
        orchestrate.original_ready('qwen',123,alive=lambda _:False)
