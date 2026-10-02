def test_a_worse_new_run_cannot_replace_parent():
    from scripts.test.pcv_distill_eval import choose_with_parent
    def candidate(human, synthetic, errors, epoch):
        return {'development':{'human':{'recall':human,'negative_errors':errors},
                               'synthetic':{'recall':synthetic}},'epoch':epoch}
    parent=candidate(.8,.98,20,24)
    arms={'arms':{'hard':candidate(.8,.98,20,4),'distilled':candidate(.79,.97,19,8)}}
    assert choose_with_parent(arms,parent)=='parent_v3'
    arms['arms']['distilled']=candidate(.91,1.,10,12)
    assert choose_with_parent(arms,parent)=='distilled'
