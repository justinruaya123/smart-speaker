"""Recipe and target masking checks, without acoustic-model inference."""
import json
from pathlib import Path

import pytest

from scripts.train.vcm_intent_slots_mixed_train import targets, validate_training
from scripts.train.vcm_intent_slots_mixed_data import annotate_reference
from scripts.vcm_intent_slots import load_contract

ROOT=Path(__file__).resolve().parents[2]


def protocol():
    return json.loads((ROOT/'experiments/vcm-mixed-final-v2/protocol.json').read_text())


def test_training_preserves_full_matched_recipe_and_cpu_hold():
    p=protocol()
    validate_training(p, 'uniform', 53, 'cuda')
    validate_training(p, 'balanced', 89, 'cuda')
    for arm,seed,device in [('uniform',53,'cpu'),('other',53,'cuda'),('uniform',17,'cuda')]:
        with pytest.raises(ValueError):
            validate_training(p,arm,seed,device)
    p['training']['epochs']=1
    with pytest.raises(ValueError):
        validate_training(p,'uniform',53,'cuda')


def test_bare_call_has_no_invented_acoustic_contact_target():
    c=load_contract(json.loads((ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json').read_text()))
    bare=next(r for r in c['cases'] if r.get('text')=='Call')
    r=annotate_reference({**bare,'speaker_id':'Kokoro-82M/af_sarah'},c)
    t=targets([r],c,'cpu')
    assert t['entity_ids'][0][c['entity_roles'].index('contact')] is None
    named=next(r for r in c['cases'] if (r.get('command') or {}).get('slots',{}).get('contact_id')=='contact.mom')
    r=annotate_reference({**named,'speaker_id':'Kokoro-82M/af_sarah'},c)
    assert targets([r],c,'cpu')['entity_ids'][0][c['entity_roles'].index('contact')]=='contact.mom'


def test_grid_tie_selects_earlier_epoch():
    from scripts.train.vcm_intent_slots_mixed_train import choose_candidate
    later={'arm':'uniform','seed':53,'epoch':20,'development_key':[.8,-2,.8]}
    earlier={'arm':'balanced','seed':71,'epoch':3,'development_key':[.8,-2,.8]}
    assert choose_candidate([later,earlier]) is earlier


def test_training_requires_complete_named_enrollment_and_separation():
    import hashlib
    from scripts.train.vcm_intent_slots_mixed_train import require_enrollment
    from scripts.train.vcm_intent_slots_mixed_corpus import make_plan
    p=protocol()
    c=load_contract(json.loads((ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json').read_text()))
    enrollment=[]
    for row in make_plan(p,c):
        if row['split']!='enrollment':continue
        row=annotate_reference(row,c)
        row.update(sha256=hashlib.sha256(row['id'].encode()).hexdigest(),
                   pcm_sha256=hashlib.sha256(('pcm'+row['id']).encode()).hexdigest())
        enrollment.append(row)
    require_enrollment({'enrollment':enrollment,'train':[]},p,c)
    with pytest.raises(ValueError,match='three'):
        require_enrollment({'enrollment':enrollment[:-1],'train':[]},p,c)
    with pytest.raises(ValueError,match='overlapping'):
        require_enrollment({'enrollment':enrollment,'train':[enrollment[0]]},p,c)
