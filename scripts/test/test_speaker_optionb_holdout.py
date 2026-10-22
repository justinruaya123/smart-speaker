"""Offline OptionB data contracts; no fixture counts as recorded human speech."""
import importlib.util
import pytest


def test_cue_match_keeps_negation_values_actions_and_meridiem():
    assert importlib.util.find_spec('scripts.test.speaker_optionb_holdout'), 'OptionB dataset audit is missing'
    from scripts.test.speaker_optionb_holdout import cue_match
    from scripts.runtime.pcv_schema import baseline_cases
    definitions=baseline_cases()
    brightness=next(r for r in definitions if r['text']=='Brightness 20 percent')
    assert cue_match('could you please adjust the brightness to 20 percent for me',
                     next(r for r in definitions if r['text']=='Adjust brightness to 20 percent'))
    assert cue_match('please brightness twenty percent now',brightness)
    for text in ('do not brightness twenty percent','brightness sixty percent',
                 'brightness twenty percent and then stop','brightness twenty percent for kitchen lights',
                 'the phrase brightness twenty percent is an example not a request'):
        assert not cue_match(text,brightness)
    alarm=next(r for r in definitions if r['text']=='Alarm 6:00 AM')
    assert cue_match('please alarm six a m',alarm)
    assert not cue_match('alarm six p m',alarm)


def test_wording_coverage_separates_exact_padding_conflicts_and_missing_cells():
    from scripts.test import speaker_optionb_holdout as audit
    from scripts.runtime.pcv_schema import baseline_cases
    from scripts.test.test_speaker_human_data import row
    assert hasattr(audit,'wording_audit'), 'Per-verbatim data audit is missing'
    definitions=[r for r in baseline_cases() if r['text'] in ('Brightness 20 percent','Brightness 60 percent')]
    low,high=[d['command'] for d in definitions]
    examples=[row('exact',expected=low),row('padded',expected=low,speaker='b'),
              row('dev',split='dev',expected=low),row('conflicting',expected=high),
              row('ambiguous',expected=None,metric=False)]
    for item in examples:item['text']='Brightness 20 percent'
    examples[1]['text']='please Brightness 20 percent now'
    result=audit.wording_audit(examples,definitions)
    assert len(result)==2
    train=result[0]['strata']['human/train']
    assert train['exact_n']==1 and train['cue_nonexact_n']==1 and train['typed_reference_n']==2
    assert train['reference_conflict_n']==1 and train['ambiguous_matching_n']==1
    assert result[0]['strata']['human/holdout']['scored_n']==0
    assert result[0]['strata']['human/holdout']['accepted_recall'] is None
    assert result[1]['strata']['human/train']['exact_n']==0
    with pytest.raises(ValueError,match='original evidence'):
        audit.wording_audit([{**examples[0],'kind':'constructed_silence_stress','synthetic':None}],definitions)


def test_label_review_preserves_conflicts_and_never_auto_admits_metadata():
    from scripts.test import speaker_optionb_holdout as audit
    from scripts.test.test_speaker_human_data import row
    assert hasattr(audit,'annotation_review'), 'Fresh label consistency review is missing'
    old={'intent':'TIMER','slots':{'duration_seconds':60}}
    item=row('timer',expected=old)
    metadata=dict(transcript='set a timer for one hour and one minute',command='TIMER',
                  slot_value='one minute',out_of_scope=0,is_synthetic=0,transcript_source='whisper')
    review=audit.annotation_review(item,metadata)
    assert review['metadata_derived']['slots']['duration_seconds']==3660
    assert 'publisher_slot_or_family_conflict' in review['flags']
    assert 'semantic_reference_disagreement' in review['flags']
    assert review['admit_to_training'] is False and review['label_changed'] is False
    assert item['expected']==old
    negative=row('ambiguous-negative')
    unresolved={**metadata,'transcript':'please mystery garbled request','command':'WEATHER'}
    flags=audit.annotation_review(negative,unresolved)['flags']
    assert 'unresolved_reference_was_scored' in flags


def test_human_loader_checks_membership_hashes_head_labels_and_masks(tmp_path):
    import numpy as np
    from scripts.test import speaker_optionb_holdout as audit
    from scripts.test.test_speaker_human_data import row
    from scripts.train.speaker_human_data import file_sha256,human_supervision
    assert hasattr(audit,'load_admitted_features'), 'Verified human feature loader is missing'
    light={'intent':'LIGHT_ON','slots':{'light_id':'light.all'}}
    pause={'intent':'PAUSE','slots':{}}
    rows=[human_supervision(row('full',expected=light,speaker='a')),
          human_supervision(row('partial',expected=pause,speaker='b'))]
    for i,item in enumerate(rows):
        path=tmp_path/(item['id']+'.npy');np.save(path,np.full((40,11),3+i,dtype=np.float32))
        item.update(features=str(path),features_sha256=file_sha256(path))
    rows[1].update(phones=[],loss_mask=dict(ctc=False,intent=True,target=False))
    norm=dict(mean=[1.]*40,std=[2.]*40,source_ids=['full','partial'],human_training_utterances=2)
    arrays,receipt=audit.load_admitted_features(rows,norm)
    assert np.all(arrays['full']==1.) and np.all(arrays['partial']==1.5)
    assert receipt['full_ctc_rows']==1 and receipt['semantic_only_rows']==1
    assert receipt['normalization_membership_verified'] is True
    for invalid in ({**rows[0],'split':'dev'},{**rows[0],'synthetic':True},
                    {**rows[0],'metric_eligible':False}):
        with pytest.raises(ValueError,match='admitted human train'):
            audit.load_admitted_features([invalid,rows[1]],norm)
    with pytest.raises(ValueError,match='dependent supervision'):
        audit.load_admitted_features([{**rows[0],'target_index':1},rows[1]],norm)
    with pytest.raises(ValueError,match='normalization membership'):
        audit.load_admitted_features(rows,{**norm,'source_ids':['full','eval']})
    reserved=row('reserved',split='holdout',speaker='a')
    with pytest.raises(ValueError,match='reserved split overlap'):
        audit.load_admitted_features(rows,norm,reserved=[reserved])
    with pytest.raises(ValueError,match='feature checksum'):
        audit.load_admitted_features([{**rows[0],'features_sha256':'0'*64},rows[1]],norm)


def test_loader_rejects_supervised_target_mask_without_a_target(tmp_path):
    import numpy as np
    from scripts.test.speaker_optionb_holdout import load_admitted_features
    from scripts.test.test_speaker_human_data import row
    from scripts.train.speaker_human_data import human_supervision,file_sha256
    item=human_supervision(row('pause',expected={'intent':'PAUSE','slots':{}}))
    path=tmp_path/'features.npy';np.save(path,np.ones((40,11),dtype=np.float32))
    item.update(features=str(path),features_sha256=file_sha256(path),
                loss_mask=dict(ctc=True,intent=True,target=True))
    norm=dict(source_ids=['pause'],human_training_utterances=1,mean=[0.]*40,std=[1.]*40)
    with pytest.raises(ValueError,match='target mask'):
        load_admitted_features([item],norm)


@pytest.mark.parametrize('field,value',[('intent_index',4.0),('target_index',False)])
def test_loader_does_not_coerce_malformed_semantic_indices(tmp_path,field,value):
    import numpy as np
    from scripts.test.speaker_optionb_holdout import load_admitted_features
    from scripts.test.test_speaker_human_data import row
    from scripts.train.speaker_human_data import human_supervision,file_sha256
    item=human_supervision(row('light',expected={'intent':'LIGHT_ON','slots':{'light_id':'light.all'}}))
    path=tmp_path/'features.npy';np.save(path,np.ones((40,11),dtype=np.float32))
    item.update(features=str(path),features_sha256=file_sha256(path));item[field]=value
    norm=dict(source_ids=['light'],human_training_utterances=1,mean=[0.]*40,std=[1.]*40)
    with pytest.raises(ValueError,match='dependent supervision'):
        load_admitted_features([item],norm)


def test_label_review_can_replay_explicit_frozen_approval_without_reverting_aliases():
    from scripts.test.speaker_optionb_holdout import annotation_review
    from scripts.test.test_speaker_human_data import row
    expected={'intent':'COLOR','slots':{'color':'green','light_id':'light.all'}}
    item=row('approved-alias',split='dev',expected=expected)
    text='can the lights be changed to green'
    metadata=dict(transcript=text,command='COLOR',out_of_scope=0,is_synthetic=0,transcript_source='dataset')
    policy=dict(version='frozen-fixture',supported={text:expected},ambiguous={})
    review=annotation_review(item,metadata,policy=policy)
    assert review['metadata_derived']==expected
    assert 'semantic_reference_disagreement' not in review['flags']
    assert review['annotation_rule_version']=='frozen-fixture'
    out_of_scope=annotation_review(row('oov',split='dev'),{**metadata,'out_of_scope':1},policy=policy)
    assert out_of_scope['metadata_derived'] is None
    assert out_of_scope['admit_to_training'] is False
