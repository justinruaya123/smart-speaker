"""Corpus planning preserves references and reserves final evaluation voices."""
import json
from pathlib import Path

import pytest

from scripts.train.vcm_intent_slots_mixed_corpus import make_plan
from scripts.vcm_intent_slots import load_contract

ROOT = Path(__file__).resolve().parents[2]


def inputs():
    protocol = json.loads((ROOT/'experiments/vcm-mixed-final-v2/protocol.json').read_text())
    contract = load_contract(json.loads((ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json').read_text()))
    return protocol, contract


def test_plan_keeps_every_form_and_separates_voice_groups():
    protocol, contract = inputs()
    plan = make_plan(protocol, contract)
    text = {r.get('spoken_text', r['text']) for r in contract['cases']+contract['catalog_entries']}
    for split, n in [('train', 2), ('dev', 1), ('calibration', 1)]:
        subset = [r for r in plan if r['split'] == split]
        assert len(subset) == n * len(text)
        assert {r['transcript'] for r in subset} == text
    final = set(protocol['final_tts_voices'])
    assert not final & {r['voice_id'] for r in plan}
    assert len({r['id'] for r in plan}) == len(plan)
    assert len({r['recording_group'] for r in plan}) == len(plan)
    assert all(r['acoustic_listening_verified'] is False for r in plan)


def test_enrollment_covers_three_separate_voices_per_named_entity():
    protocol, contract = inputs()
    plan = make_plan(protocol, contract)
    from scripts.runtime.vcm_intent_slots_entities import catalog_ids
    entities = catalog_ids(contract)
    entities['light'].remove('light.all')
    for role, names in entities.items():
        for entity in names:
            rows = [r for r in plan if r['split']=='enrollment' and r['entity_role']==role and r['entity_id']==entity]
            assert len(rows)==3
            assert len({r['voice_id'] for r in rows})==3
            assert all(r['command']['slots'][role+'_id']==entity for r in rows)
    from scripts.train.vcm_intent_slots_mixed_data import annotate_reference
    assert all(annotate_reference(r, contract)['role_labels'].get(r['entity_role'])=='present'
               for r in plan if r['split']=='enrollment')


def test_final_voice_cannot_be_reassigned_for_fitting():
    protocol, contract = inputs()
    protocol['voices']['train'][0] = 'af_heart'
    with pytest.raises(ValueError):
        make_plan(protocol, contract)


def test_correction_builds_the_full_plan_once(tmp_path, monkeypatch):
    from scripts.train import vcm_intent_slots_mixed_corpus as module
    from scripts.train.vcm_intent_slots_data import sha
    old=[{'id':'old'+str(n),'entity_id':'contact.home','split':'enrollment'} for n in range(3)]
    new=[{'id':'new'+str(n),'entity_id':'contact.home','split':'enrollment'} for n in range(3)]
    parent=tmp_path/'old.json';parent.write_text(json.dumps(old))
    receipt=tmp_path/'experiments/vcm-mixed-final-v2/preparation-correction-v1.json'
    receipt.parent.mkdir(parents=True);receipt.write_text(json.dumps({'original_generation_plan_sha256':sha(parent)}))
    calls=[]
    def build(*args):
        calls.append(args)
        return new
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'make_plan',build)
    assert module.correction_plan(parent,{}, {})==new
    assert len(calls)==1


def test_generated_manifest_cannot_drop_or_relabel_locked_records(tmp_path):
    from scripts.train.vcm_intent_slots_mixed_corpus import verify_generation
    from scripts.train.vcm_intent_slots_data import sha
    from collections import Counter
    plan=[{'id':'a','split':'train','command':None},{'id':'b','split':'dev','command':None}]
    planned=tmp_path/'plan.json';planned.write_text(json.dumps(plan))
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    identity={'plan':{'path':str(planned),'sha256':sha(planned)},
              'protocol':{'path':str(protocol),'sha256':sha(protocol)}}
    (tmp_path/'run-identity.json').write_text(json.dumps(identity))
    manifest=tmp_path/'manifest.json'
    values=[{**r,'status':'generation_fault'} for r in plan]
    def write(values):
        manifest.write_text(json.dumps(values))
        receipt={'manifest_sha256':sha(manifest),'planned':2,
                 'status_counts':dict(Counter(r['status'] for r in values))}
        (tmp_path/'generation-receipt.json').write_text(json.dumps(receipt))
    write(values)
    assert verify_generation(manifest,protocol)==values
    write(values[:1])
    with pytest.raises(ValueError,match='Incomplete'):
        verify_generation(manifest,protocol)
    values[0]['command']={'intent':'STOP','slots':{}}
    write(values)
    with pytest.raises(ValueError,match='frozen plan'):
        verify_generation(manifest,protocol)
