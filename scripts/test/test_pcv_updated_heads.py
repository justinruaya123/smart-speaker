from scripts.runtime.pcv_schema import INTENTS


def test_unknown_contact_retains_known_call_intent():
    from scripts.train.pcv_updated_heads import semantic_label
    row={'expected':None,'command':'CALL','reference_status':'unsupported','reason':'finite_name_or_content_scope','intent_index':0}
    assert semantic_label(row)==INTENTS.index('CALL')+1
    assert semantic_label({**row,'command':'OUT_OF_SCOPE','reason':'publisher_out_of_scope'})==0


def test_joint_ranking_cannot_rescue_rejected_intent():
    import numpy as np
    from scripts.train.pcv_updated_heads import joint_rank
    entries=[{'id':'p','command':{'intent':'TIME','slots':{}},'use_case':'TIME'},
             {'id':'n','command':None,'use_case':'OOV'}]
    probability=np.zeros(19);probability[0]=1
    assert joint_rank(np.array([-.1,-3]),entries,probability,.1)['command'] is None
