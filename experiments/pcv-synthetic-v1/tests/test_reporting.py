import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reporting import assemble_outputs


def test_unexecuted_expectation_never_becomes_actual_output():
    ref={'id':'x','text':'pause','expected_command':{'intent':'MEDIA_PAUSE'}}
    result=assemble_outputs([ref],[])[0]
    assert result['status']=='NOT_RUN' and result['observations']==[]


def test_disabled_policy_does_not_promote_correct_raw_candidate():
    command={'intent':'MEDIA_PAUSE'}
    ref={'id':'x','expected_command':command}
    row=dict(recording_id='qwen/test/x',verbatim_id='x',expected_command=command,
        candidate=command,output={'accepted':False,'command':None,'reason':'disabled'},
        engine='qwen',accent_requested='British English',capture_fault=None,endpoint_detected=True)
    result=assemble_outputs([ref],[row])[0]['observations'][0]
    assert result['raw_candidate']==command
    assert result['actual_output']['accepted'] is False and result['exact_correct'] is False
    with pytest.raises(ValueError,match='Duplicate'):assemble_outputs([ref],[row,row])
