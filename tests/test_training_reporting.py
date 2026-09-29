import json


def test_selection_protects_previously_correct_commands():
    from vcm.training import select_epoch
    assert select_epoch({'a'}, {'a','b'})
    assert not select_epoch({'a'}, {'b','c'})


def test_reports_reconcile_and_do_not_promote_rejections(tmp_path):
    from vcm.evaluation import write_reports
    rows=[{'id':'1','expected':{'intent':'POWER_ON'},'valid_request':True,'accepted':False,'candidate':{'intent':'POWER_ON'},'command':None,'reason':'uncalibrated'},
          {'id':'2','expected':None,'valid_request':False,'accepted':False,'candidate':None,'command':None,'reason':'below_threshold'},
          {'id':'3','expected':None,'valid_request':True,'accepted':False,'candidate':None,'command':None,'reason':'unsupported'}]
    result=write_reports(rows,tmp_path,{'model':'m'})
    assert result['valid_requests']==2
    assert result['correct_accepted']==0
    assert result['negative_false_accepts']==0
    assert len((tmp_path/'accepted_commands.csv').read_text().splitlines())==1
    assert not json.loads((tmp_path/'release.json').read_text())['eligible']
