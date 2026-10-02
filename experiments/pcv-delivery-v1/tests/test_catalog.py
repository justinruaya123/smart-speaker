import csv
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
from scripts.runtime.pcv_deploy.catalog import compile_csv,CatalogContract


@pytest.fixture
def lexicon():return json.loads((ROOT/'experiments/pcv-synthetic-v2/lexicon.json').read_text())['words']


def inventory(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['kind','id','name','artist','aliases','capabilities'])
        writer.writeheader();writer.writerows(rows)
    return path


def test_csv_dynamic_title_artist_collision_and_recalibration(tmp_path,lexicon):
    path=inventory(tmp_path/'catalog.csv',[dict(kind='song',id='track.bts_stay',name='stay',artist='BTS'),
        dict(kind='song',id='track.blackpink_stay',name='stay',artist='Blackpink'),
        dict(kind='song',id='track.bts_butter',name='butter',artist='BTS')])
    result=compile_csv(path,lexicon)
    by_text={r['text']:r['command'] for r in result['entries']}
    assert by_text['play stay'] is None
    assert by_text['play stay by bts']['track_id']=='track.bts_stay'
    assert by_text['play stay by blackpink']['track_id']=='track.blackpink_stay'
    assert by_text['play butter']==by_text['play butter by bts']
    assert result['requires_calibration'] is True
    assert result==compile_csv(path,lexicon)


def test_light_capabilities_all_101_values_and_strict_types(tmp_path,lexicon):
    result=compile_csv(inventory(tmp_path/'catalog.csv',[dict(kind='light',id='light.demo',name='kitchen lamp',
        aliases='bedroom lamp',capabilities='power|brightness|color')]),lexicon)
    commands=[r['command'] for r in result['entries'] if r['command']]
    assert {r['value'] for r in commands if r['intent']=='BRIGHTNESS'}==set(range(101))
    assert {r['color'] for r in commands if r['intent']=='COLOR_SET'}>={'red','blue','cyan','magenta'}
    contract=CatalogContract(result)
    good=next(c for c in commands if c.get('value')==1)
    contract.validate(good)
    for value in (-1,101,True,1.0,'1'):
        with pytest.raises(ValueError):contract.validate({**good,'value':value})


@pytest.mark.parametrize('rows',[
    [dict(kind='song',id='track.one',name='gibberishword')],
    [dict(kind='light',id='../escape',name='kitchen lamp',capabilities='power')],
    [dict(kind='light',id='light.one',name='kitchen lamp',capabilities='launch')],
    [dict(kind='song',id='track.same',name='stay'),dict(kind='song',id='track.same',name='butter')],
])
def test_unsupported_inventory_fails_closed(tmp_path,lexicon,rows):
    with pytest.raises(ValueError):compile_csv(inventory(tmp_path/'catalog.csv',rows),lexicon)


def test_catalog_bound_prevents_unbounded_pi_decoder(tmp_path,lexicon):
    path=inventory(tmp_path/'catalog.csv',[dict(kind='light',id='light.demo',name='kitchen lamp',capabilities='brightness')])
    with pytest.raises(ValueError,match='budget'):compile_csv(path,lexicon,max_entries=20)


def test_inventory_merge_keeps_other_use_cases_and_explicit_rejections():
    from scripts.runtime.pcv_deploy.catalog import merge_inventory
    def row(i,intent,phones):return dict(id=i,phones=phones,command={'intent':intent} if intent else None)
    base=dict(entries=[row('old','MEDIA_PLAY',[1]),row('timer','TIMER_SET',[2]),row('invalid',None,[3])])
    inventory=dict(entries=[row('new','MEDIA_PLAY',[4])])
    result=merge_inventory(base,inventory)
    assert {r['id'] for r in result['entries']}=={'new','timer','invalid'}
    assert result['requires_calibration'] and not result['acoustic_generalization_measured']
    inventory['entries'][0]['phones']=[3]
    with pytest.raises(ValueError,match='explicit protocol amendment'):merge_inventory(base,inventory)
