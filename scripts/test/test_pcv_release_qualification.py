from scripts.test.pcv_release_qualification import release_quality


def metrics(**changes):
    return dict({'positive_n':100,'negative_n':100,'precision':1.,'recall':.95,'false_accept_rate':0.},**changes)


def test_release_needs_acceptance_and_human_negative_evidence():
    assert release_quality(True,metrics(),metrics(negative_n=0),True)
    assert not release_quality(False,metrics(),metrics(),True)
    assert not release_quality(True,metrics(negative_n=0),metrics(),True)
    assert not release_quality(True,metrics(precision=None,recall=0),metrics(),True)
    assert not release_quality(True,metrics(),metrics(),False)


def test_high_synthetic_accuracy_cannot_hide_human_failure():
    assert not release_quality(True,metrics(recall=.89),metrics(),True)
    assert not release_quality(True,metrics(false_accept_rate=.02),metrics(),True)


def test_negative_phrase_requires_its_exact_typed_rejection_reason():
    from scripts.test.pcv_release_qualification import typed_phrase_table
    from scripts.runtime.pcv_schema import output
    row={'text':'turn on the mystery light','reference_status':'unsupported',
         'output':output(None,'unqualified_model'),'accepted_command':None}
    table=typed_phrase_table([row])
    case=next(r for r in table if r['text']=='turn on the mystery light')
    assert case['n']==1 and case['correct']==0 and not case['pass']
    row['output']=output(None,'invalid_argument')
    case=next(r for r in typed_phrase_table([row]) if r['text']=='turn on the mystery light')
    assert case['correct']==1 and case['pass']
