def test_final_catalog_covers_every_approved_phrase_and_keeps_parent():
    import json
    from scripts.train.pcv_public_data import ROOT,_words
    from scripts.test.pcv_final_qualification import complete_catalog
    parent=json.loads((ROOT/'experiments/pcv-updated-voice-v1/catalog.json').read_text())
    prompts=json.loads((ROOT/'experiments/pcv-b1-three-engine-v1/prompts.json').read_text())
    catalog,receipt=complete_catalog(parent,prompts,_words())
    assert receipt['approved_phrases']==415
    assert receipt['missing_after']==0
    assert receipt['added_ids']==['CUSTOM-CALL-mother']
    assert catalog['entries'][:len(parent['entries'])]==parent['entries']
    assert catalog['acoustic_qualification'] is False


def test_quantization_fidelity_does_not_hide_human_regression():
    from scripts.test.pcv_final_qualification import quantization_fidelity
    reference={'human':{'exact':140,'negative_errors':20},'synthetic':{'exact':108}}
    close={'human':{'exact':139,'negative_errors':21},'synthetic':{'exact':107}}
    assert quantization_fidelity(reference,close)
    assert not quantization_fidelity(reference,{**close,'human':{'exact':138,'negative_errors':20}})
    assert not quantization_fidelity(reference,{**close,'human':{'exact':140,'negative_errors':22}})
