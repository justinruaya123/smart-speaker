"""Leakage and denominator checks for the expanded test-only corpus."""
import pytest

from scripts.train.pcv_three_engine import exact_reuse, audit_records


def prompt():
    return {'id':'B1-001','text':'Lights on','command':{'intent':'LIGHT_ON','slots':{'light_id':'light.all'}}}


def row(**changes):
    return {'id':'qwen/B1-001','phrase_id':'B1-001','text':'lights on','expected':prompt()['command'],
            'engine':'qwen','speaker':'qwen/test-gb-m','split':'test','pcm_sha256':'abc',**changes}


def test_reuse_requires_exact_command_and_reserved_test_voice():
    assert exact_reuse(prompt(),row())
    assert not exact_reuse(prompt(),row(speaker='qwen/train-us-m'))
    assert not exact_reuse(prompt(),row(expected={'intent':'LIGHT_ON','slots':{'light_id':'light.kitchen'}}))
    assert not exact_reuse(prompt(),row(text='turn on the lights'))


def test_audit_keeps_missing_cells_in_denominator():
    value=audit_records([prompt()],['qwen','dia'],[row()],[])
    assert value['planned']==2 and value['present']==1 and not value['complete']
    assert value['missing']==['dia/B1-001']


def test_duplicate_pcm_and_training_overlap_reject():
    with pytest.raises(ValueError,match='duplicate PCM'):
        audit_records([prompt()],['qwen','dia'],[row(),row(id='dia/B1-001',engine='dia',speaker='dia/test-gb-m')],[])
    with pytest.raises(ValueError,match='non-test overlap'):
        audit_records([prompt()],['qwen'],[row()],[row(split='train')])


def test_unexpected_or_wrongly_labelled_cell_rejects():
    with pytest.raises(ValueError,match='cell'):
        audit_records([prompt()],['qwen'],[row(phrase_id='nope')],[])
    with pytest.raises(ValueError,match='label'):
        audit_records([prompt()],['qwen'],[row(expected=None)],[])
