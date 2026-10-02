import sys
from pathlib import Path
from collections import Counter
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import revision


def test_preserves_original_commands_and_all_brightness_values():
    rows=revision.build_verbatims()
    old=revision.read_rows(revision.V1/'verbatims.jsonl')
    current={r['id']:r for r in rows}
    assert all(current[r['id']]==r for r in old if r['expected_command'] is not None)
    assert all(current[r['id']]==r or current[r['id']]['supersedes_reference_sha256']==revision.identity(r)
               for r in old)
    assert {r['use_case'] for r in rows}=={f'UC-{i:02}' for i in range(1,11)}
    for entity in ('light.kitchen','light.bedroom'):
        assert {r['expected_command']['value'] for r in rows if r['expected_command']
                and r['expected_command']['intent']=='BRIGHTNESS'
                and r['expected_command']['entity_id']==entity}==set(range(101))


def test_optional_artist_disambiguates_duplicate_title_and_rejects_mismatch():
    rows={r['text'].lower():r for r in revision.build_verbatims()}
    assert rows['play butter']['expected_command']==rows['play butter by bts']['expected_command']
    assert rows['play stay']['expected_command'] is None
    assert rows['play stay by bts']['expected_command']!=rows['play stay by blackpink']['expected_command']
    assert rows['play butter by blackpink']['expected_command'] is None
    assert rows['play unknown song']['expected_command'] is None


def test_reminder_durations_clocks_colors_and_typed_rejection():
    rows={r['text'].lower():r for r in revision.build_verbatims()}
    assert rows['set a reminder for sixty minutes']['expected_command']['delay_seconds']==3600
    assert rows['set a reminder for twelve hours']['expected_command']['delay_seconds']==43200
    assert rows['set a reminder at twelve a m']['expected_command']['time_local']=='00:00'
    assert rows['set a reminder at twelve p m']['expected_command']['time_local']=='12:00'
    assert rows['set a reminder at seven']['expected_command'] is None
    assert rows['set kitchen lamp to cyan']['expected_command']['color']=='cyan'
    assert rows['set kitchen lamp to one hundred one percent']['expected_command'] is None
    contract=revision.RevisionContract(rows=revision.build_verbatims())
    valid=rows['set kitchen lamp to one percent']['expected_command']
    for value in (-1,101,True,1.0,'1'):
        with pytest.raises(ValueError):contract.validate({**valid,'value':value})


def test_dia_subset_is_frozen_balanced_and_does_not_reduce_qwen_parler():
    rows=revision.build_verbatims();plan=revision.generation_plan(rows)
    dia=[r for r in plan if r['engine']=='dia']
    assert len(dia)==600
    assert len({r['verbatim_id'] for r in dia})==100
    assert len({r['recording_id'] for r in plan})==len(plan)
    assert len([r for r in plan if r['engine']=='qwen'])==len(rows)*6
    assert len([r for r in plan if r['engine']=='parler'])==len(rows)*6
    assert plan==revision.generation_plan(rows)
    lookup={r['id']:r for r in rows}
    strata=Counter((lookup[r['verbatim_id']]['phase'],lookup[r['verbatim_id']]['use_case'],r['split'],
                    lookup[r['verbatim_id']]['expected_command'] is not None) for r in dia)
    assert len(strata)==160
    assert all(n>0 for n in strata.values())
    assert len({r['voice_group'] for r in dia})==6


def test_compiled_catalog_has_no_conflicting_acoustic_targets():
    rows=revision.build_verbatims()
    catalog=revision.compile_catalog(rows)
    assert len(catalog['entries'])==len(rows)
    assert all(0<p<40 for r in catalog['entries'] for p in r['phones'])
    contract=revision.RevisionContract(rows=rows)
    for row in catalog['entries']:
        if row['command'] is not None:contract.validate(row['command'])


def test_catalog_compiles_from_frozen_lexicon_not_live_upstream(monkeypatch):
    def changed_upstream(_rows):
        raise AssertionError('Live dictionary must not be read after freezing')
    monkeypatch.setattr(revision,'lexicon',changed_upstream)
    catalog=revision.compile_catalog(revision.read_rows(revision.HERE/'verbatims.jsonl'))
    assert catalog['pronunciation_sha256']==revision.identity(__import__('json').loads(
        (revision.HERE/'lexicon.json').read_text()))
