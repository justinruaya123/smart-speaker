import pytest


def test_numbers_are_complete_and_typed():
    from vcm.catalog import number_words, validate_command
    assert number_words(73)=='seventy three'
    for n in range(101):
        assert validate_command({'intent':'BRIGHTNESS','value':n})
    for n in (True,-1,101,3.5,float('nan')):
        assert not validate_command({'intent':'BRIGHTNESS','value':n})
    assert not validate_command({'intent':'BRIGHTNESS','value':7,'extra':3})


def test_catalog_ties_capabilities_and_music_pairs():
    from vcm.catalog import resolve_entity, music_phrases
    entities=[{'id':'light.a','aliases':['lamp'],'area':'kitchen','exposed':True,'available':False,'capabilities':['power']},
              {'id':'light.b','aliases':['lamp'],'area':'bedroom','exposed':True,'available':True,'capabilities':['power','color']}]
    assert resolve_entity(entities,'lamp')['reason']=='ambiguous'
    assert resolve_entity(entities,'lamp','kitchen')['reason']=='unavailable'
    assert resolve_entity(entities,'lamp','bedroom','color')['id']=='light.b'
    assert resolve_entity(entities,'lamp','kitchen','color')['reason']=='unsupported_capability'
    music=music_phrases([{'id':'song1','title':'seventy three','artists':['alpha']}, {'id':'song2','title':'seventy three','artists':['beta']}])
    assert music['seventy three'] is None
    assert music['seventy three by alpha']['track_id']=='song1'
    assert 'seventy three by gamma' not in music


def test_split_audit_rejects_leakage():
    from vcm.data import audit_rows
    rows=[{'id':'a','speaker':'s','split':'train','pcm_sha256':'a'}, {'id':'b','speaker':'s','split':'test','pcm_sha256':'b'}]
    with pytest.raises(ValueError,match='speaker'): audit_rows(rows)
    rows[1]['speaker']='t'; rows[1]['pcm_sha256']='a'
    with pytest.raises(ValueError,match='hash'): audit_rows(rows)
