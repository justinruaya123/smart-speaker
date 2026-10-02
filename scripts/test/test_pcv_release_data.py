def test_refresh_reuses_only_identical_payloads():
    from scripts.train.pcv_release_data import payload_plan
    old={'files':[{'path':'.cache/old/source/data/train.parquet','sha256':'a','bytes':10}]}
    current={'siblings':[{'rfilename':'README.md'},
        {'rfilename':'data/train.parquet','size':10,'lfs':{'sha256':'a'}},
        {'rfilename':'data/test.parquet','size':12,'lfs':{'sha256':'b'}}]}
    result=payload_plan(current,old)
    assert result[0]['reuse']=='.cache/old/source/data/train.parquet'
    assert result[1]['reuse'] is None
    current['siblings'][1]['size']=11
    assert payload_plan(current,old)[0]['reuse'] is None


def test_movement_audit_identifies_inherited_checkpoint_contamination():
    from scripts.train.pcv_release_data import movement_audit
    old=[{'id':'a','pcm_sha256':'one','published_split':'train','source':'s','speaker_id':'A'},
         {'id':'b','pcm_sha256':'two','published_split':'test','source':'s','speaker_id':'B'}]
    new=[{'id':'c','pcm_sha256':'one','published_split':'test','source':'s','speaker_id':'A'},
         {'id':'d','pcm_sha256':'three','published_split':'train','source':'s','speaker_id':'B'}]
    result=movement_audit(old,new)
    assert result['prior_train_in_new_test_pcm']==['one']
    assert result['prior_test_speakers_in_new_train']==['s/B']
    assert result['new_pcm_n']==1
    assert not result['inherited_model_clean_for_new_test']
