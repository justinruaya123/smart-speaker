from pathlib import Path


def test_fixture_compiler_keeps_to_by_and_rejects_substrings():
    from vcm.catalog import compile_phrases
    fixtures={'entities':[{'id':'light.2','aliases':['lamp two'],'area':'bedroom','exposed':True,'available':True,'capabilities':['power','brightness','color']}],
              'tracks':[{'id':'track73','title':'seventy three','artists':['alpha']}],'colors':['red']}
    phrases=compile_phrases(fixtures)
    assert phrases['set lamp two to seventy three percent']=={'intent':'BRIGHTNESS','value':73,'entity_id':'light.2'}
    assert phrases['increase lamp two by seventy three percent']['delta']==73
    assert phrases['play seventy three by alpha']['track_id']=='track73'
    for bad in ['set lamp two to seventy three percent nine','do not turn on lamp two','set lamp two to one hundred one percent','set lamp two to minus one percent','set lamp two to three point five percent']:
        assert bad not in phrases
