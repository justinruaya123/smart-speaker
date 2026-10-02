from scripts.test.pcv_three_engine_report import metrics_with_coverage


def test_missing_positive_remains_in_recall_and_intent_denominator():
    command={'intent':'TIME','slots':{}}
    rows=[{'expected':command,'candidate':command,'present':True},
          {'expected':command,'candidate':None,'present':False}]
    result=metrics_with_coverage(rows)
    assert result['overall']['recall']==.5
    assert result['overall']['positive_n']==2
    assert result['overall']['missing_positive']==1
    assert result['per_intent']['TIME']['reference_n']==2


def test_missing_negative_is_unmeasured_not_a_correct_rejection():
    rows=[{'expected':None,'candidate':None,'present':True},
          {'expected':None,'candidate':None,'present':False}]
    result=metrics_with_coverage(rows)['overall']
    assert result['negative_n']==2 and result['measured_negative_n']==1
    assert result['false_accept_rate'] is None
    assert result['false_accept_rate_bounds']==[0,.5]


def test_complete_all_reject_keeps_zero_recall():
    rows=[{'expected':{'intent':'TIME','slots':{}},'candidate':None,'present':True},
          {'expected':None,'candidate':None,'present':True}]
    result=metrics_with_coverage(rows)['overall']
    assert result['recall']==0 and result['precision'] is None
    assert result['false_accept_rate']==0 and result['missing_n']==0
