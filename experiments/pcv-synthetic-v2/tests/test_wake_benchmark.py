import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wake_benchmark import score_trial,summarize
import wake_benchmark
import pytest
import json


def test_wake_rejection_and_wrong_parameter_are_not_successful_positives():
    expected={'intent':'BRIGHTNESS','value':100,'entity_id':'light.kitchen'}
    reference={'expected_command':expected,'case_type':'wake_command','engine':'qwen','use_case':'UC-04'}
    wake={'stage':'wake','status':'detected'}
    wrong={'stage':'command','accepted':True,'command':{**expected,'value':0}}
    assert not score_trial(reference,[wake])['correct']
    row=score_trial(reference,[wake,wrong]);assert not row['correct'] and row['accepted']
    good={**wrong,'command':expected}
    assert score_trial(reference,[wake,good])['correct']
    assert not score_trial(reference,[wake,good,good])['correct']


def test_all_reject_and_spurious_wake_fail_and_no_pause_remains_diagnostic():
    base={'engine':'qwen','use_case':'UC-04'}
    positive=score_trial({**base,'case_type':'wake_command','expected_command':{'intent':'BRIGHTNESS','value':100}},[])
    negative=score_trial({**base,'case_type':'no_wake','expected_command':None},[{'stage':'wake','status':'detected'}])
    report=summarize([positive,negative])
    assert report['positive_recall']==0 and report['false_wakes']==1 and not report['wake_gates_passed']
    diagnostic=score_trial({**base,'case_type':'no_pause','diagnostic_only':True,'expected_command':{'intent':'POWER_ON'}},[])
    assert summarize([diagnostic])['positives']==0


def test_undetected_wake_only_cannot_hide_behind_successful_command_strata():
    rows=[]
    for phase in (1,2):
        for engine in ('qwen','parler','dia'):
            for uc in range(1,11):
                command={'intent':f'fixture-{uc}'}
                rows.append(score_trial(dict(phase=phase,engine=engine,use_case=f'UC-{uc:02}',
                    expected_command=command,case_type='wake_command'),[
                    {'stage':'wake','status':'detected'},{'stage':'command','accepted':True,'command':command}]))
    rows.append(score_trial(dict(phase=1,engine='qwen',use_case='UC-03',expected_command=None,case_type='wake_only'),[]))
    assert not summarize(rows)['wake_gates_passed']


def test_wake_test_reservation_is_dataset_locked_across_candidates(tmp_path,monkeypatch):
    monkeypatch.setattr(wake_benchmark,'CACHE',tmp_path)
    monkeypatch.setattr(wake_benchmark,'file_hash',lambda path:str(path))
    path=wake_benchmark.reserve_test_access('candidate-a',tmp_path/'first')
    assert json.loads(path.read_text())['binding']=='candidate-a'
    with pytest.raises(FileExistsError):
        wake_benchmark.reserve_test_access('candidate-b',tmp_path/'second')
