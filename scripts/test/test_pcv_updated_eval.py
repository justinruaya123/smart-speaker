"""Evaluation must expose ambiguity, missing phrases and all-reject policies."""
from scripts.runtime.pcv_schema import command


def test_missing_baseline_cell_remains_missing_not_passing():
    from scripts.test.pcv_updated_eval import phrase_table
    table=phrase_table([], 'candidate')
    assert len(table)==415
    assert all(r['n']==0 and r['accuracy'] is None and not r['pass'] for r in table)


def test_ambiguous_reference_does_not_become_a_negative():
    from scripts.test.pcv_updated_eval import measured
    row={'id':'r','synthetic':False,'source':'human','reference_status':'ambiguous',
         'expected':None,'candidate':command('TIME'),'phrase_id':None}
    result=measured([row],'candidate')
    assert result['all']['overall']['n']==0
    assert result['unresolved_reference_n']==1


def test_zero_acceptance_cannot_qualify():
    from scripts.test.pcv_updated_eval import gates
    assert not gates({'positive_n':93,'negative_n':20,'precision':None,'recall':0.,'false_accept_rate':0.})
