import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synthbench import metrics, audit_manifest, compile_synthetic_catalog, summarize
from study import build_verbatims


def test_wrong_parameter_is_not_a_correct_accept_and_all_reject_fails():
    expected = {'intent':'BRIGHTNESS','entity_id':'light.kitchen','value':37}
    wrong = {**expected, 'value':73}
    row = dict(expected_command=expected, candidate=wrong,
               output={'accepted':True,'command':wrong}, use_case='UC-04',
               phase=1, engine='qwen', accent_requested='British English', verbatim_id='x')
    m=metrics([row])
    assert m['correct_accepted']==0 and m['parameter_errors']==1 and m['intent_correct']==1
    assert m['accepted_precision']==0 and m['accepted_recall']==0
    row['output']={'accepted':False,'command':None}
    assert metrics([row])['accepted_precision'] is None


def test_phase2_compiles_without_altering_command_scope():
    catalog=compile_synthetic_catalog(build_verbatims())
    assert len(catalog['entries'])==2310
    assert len({str(r['command']) for r in catalog['entries'] if r['command']})==402


def test_manifest_must_be_complete_and_have_no_cross_split_pcm(tmp_path):
    with pytest.raises(ValueError, match='Incomplete'):
        audit_manifest([], [{'recording_id':'missing'}], tmp_path)


def test_perfect_single_use_case_is_not_a_full_benchmark_pass():
    c={'intent':'POWER_ON','entity_id':'light.kitchen'}
    rows=[dict(expected_command=c,candidate=c,output={'accepted':True,'command':c},
        phase=1,engine='qwen',use_case='UC-03',verbatim_id='x',accent_requested='British English')]
    rows.append({**rows[0],'expected_command':None,'candidate':None,
                 'output':{'accepted':False,'command':None},'verbatim_id':'negative'})
    assert summarize(rows)['synthetic_gates_passed'] is False


def test_precision_charges_wrong_emissions_to_predicted_use_case():
    light={'intent':'POWER_ON','entity_id':'light.kitchen'}
    clock={'intent':'TIME_QUERY','timezone':'Asia/Manila'}
    base=dict(phase=1,engine='qwen',accent_requested='British English',verbatim_id='x')
    rows=[{**base,'use_case':'UC-02','expected_command':clock,'candidate':clock,
           'output':{'accepted':True,'command':clock}}]*10
    rows += [{**base,'use_case':'UC-03','expected_command':light,'candidate':clock,
              'output':{'accepted':True,'command':clock}}]*5
    assert summarize(rows)['use_case']['UC-02']['accepted_precision']==pytest.approx(10/15)


def test_capture_failure_is_reported_separately_from_a_valid_rejection():
    row=dict(expected_command=None,candidate=None,output={'accepted':False,'command':None},
             endpoint_detected=False,capture_fault='timeout')
    result=metrics([row])
    assert result['invalid_trials']==1
    assert result['valid_negative_rejections']==0
