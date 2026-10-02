import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.pcv_human_regression import summarize


def test_raw_human_regression_does_not_count_negatives_as_positive_accuracy():
    def row(expected,b,c):return dict(expected=expected,baseline=dict(raw_correct=b,capture_fault=None),candidate=dict(raw_correct=c,capture_fault=None))
    result=summarize([row({'intent':'POWER_ON'},True,False),row({'intent':'POWER_ON'},False,True),row(None,True,True)])
    assert result['examples']==3 and result['positives']==2 and result['negatives']==1
    assert result['baseline_raw_correct']==result['candidate_raw_correct']==1
    assert result['per_intent']['NEGATIVE']['trials']==1
    assert not result['accepted_accuracy_measured'] and not result['human_qualified']
