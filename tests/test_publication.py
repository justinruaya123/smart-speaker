import json
from pathlib import Path
import pytest


def test_package_rejects_unqualified_and_unknown_bundle_files(tmp_path):
    from vcm.packaging import check_release
    with pytest.raises(ValueError,match='qualified'):
        check_release({'eligible':False},{'model':'m'})
    with pytest.raises(ValueError):
        check_release({'eligible':True,'binding':{'model':'wrong'}},{'model':'m'})


def test_manifest_paths_cannot_escape(tmp_path):
    from vcm.packaging import safe_members
    with pytest.raises(ValueError): safe_members({'../secret':b'x'})
    with pytest.raises(ValueError): safe_members({'C:/secret':b'x'})


def test_wrong_accepted_type_is_not_exact(tmp_path):
    from vcm.evaluation import write_reports
    rows=[{'id':'x','expected':{'intent':'BRIGHTNESS','value':1},'candidate':{'intent':'BRIGHTNESS','value':True},
           'command':{'intent':'BRIGHTNESS','value':True},'accepted':True,'valid_request':True}]
    assert write_reports(rows,tmp_path,{})['correct_accepted']==0
