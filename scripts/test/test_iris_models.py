import importlib
import json
import numpy as np
import pytest
import torch

def test_observed_policy_grid_can_separate_near_one_errors_without_rejecting_all():
    from scripts.runtime.mvp_static import fit_policy,decide
    import numpy as np
    catalog=[{'command':None},{'command':{'intent':'PAUSE','slots':{}}}]
    def row(probability,expected):
        return dict(logits=[float(np.log1p(-probability)),float(np.log(probability))],
                    expected_index=expected,synthetic=False,split='calibration')
    rows=[row(.99611,0),row(.99789,1),row(.9981,1)]
    assert fit_policy(rows,[0,1])['fitting']['correct']==0
    policy=fit_policy(rows,[0,1],threshold_strategy='observed')
    assert policy['fitting']['correct']==2 and policy['fitting']['unknown_accepts']==0
    assert .99611<policy['threshold']<.99789
    assert decide(rows[0]['logits'],catalog,policy)['command'] is None



def test_class_policy_does_not_transfer_one_songs_errors_to_a_different_song():
    from scripts.runtime.mvp_static import fit_policy,decide
    import numpy as np
    catalog=[{'command':None},
             {'command':{'intent':'PLAY_MUSIC','slots':{'song_id':'song.a'}}},
             {'command':{'intent':'PLAY_MUSIC','slots':{'song_id':'song.b'}}}]
    def row(index,probability,expected):
        p=np.full(3,(1-probability)/2);p[index]=probability
        return dict(logits=np.log(p).tolist(),expected_index=expected,synthetic=True,split='dev')
    rows=[row(1,.8,1),row(1,.81,1),row(2,.99,0),row(2,.995,2),row(2,.996,2)]
    policy=fit_policy(rows,[0,1,2],catalog=catalog,threshold_strategy='observed',grouping='class')
    assert policy['fitting']['synthetic_correct']==4 and policy['fitting']['unknown_accepts']==0
    assert decide(rows[0]['logits'],catalog,policy)['command']==catalog[1]['command']
    assert decide(rows[2]['logits'],catalog,policy)['command'] is None



def test_missing_class_limit_never_falls_back_to_a_looser_global_threshold():
    from scripts.runtime.mvp_static import decide
    catalog=[{'command':None},{'command':{'intent':'PAUSE','slots':{}}}]
    policy=dict(threshold=.2,margin=0.,trained_classes=[0,1],by_class={},grouping='class')
    assert decide([0.,8.],catalog,policy)['command'] is None



@pytest.mark.parametrize('keys',[{}, {'01':{'threshold':.9,'margin':0.}}])
def test_runtime_rejects_incomplete_or_noncanonical_class_policy(tmp_path,keys):
    import json
    from scripts.runtime.mvp_static import Runtime,digest
    (tmp_path/'model.onnx').write_bytes(b'not loaded for invalid metadata')
    files={'catalog.json':[{'command_id':0,'command':None},{'command_id':1,'command':{'intent':'PAUSE','slots':{}}}],
           'normalization.json':{'mean':[0]*40,'std':[1]*40},
           'policy.json':dict(threshold=.2,margin=0.,trained_classes=[0,1],qualified=False,grouping='class',by_class=keys)}
    for name,value in files.items():(tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'bundle.json').write_text(json.dumps(dict(format='mvp-static-v1',
        files={name:digest(tmp_path/name) for name in ('model.onnx',*files)})))
    with pytest.raises(ValueError,match='class policy'):Runtime(tmp_path)



def test_catalog_preserves_atomic_commands_and_baseline_wordings():
    m = importlib.import_module('scripts.runtime.mvp_static')
    a = {'intent': 'BRIGHTNESS', 'slots': {'percent': 20, 'light_id': 'light.all'}}
    b = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 60}}
    c = m.build_catalog([{'text': 'twenty', 'command': a}, {'text': '20', 'command': a}, {'text': 'sixty', 'command': b}])
    assert len(c) == 3 and c[0]['command'] is None
    assert set(c[1]['wordings'] + c[2]['wordings']) == {'twenty', '20', 'sixty'}
    assert {x['command']['slots']['percent'] for x in c[1:]} == {20, 60}
    with pytest.raises(ValueError):
        m.build_catalog([{'text': 'missing', 'command': {'intent': 'BRIGHTNESS', 'slots': {}}}])



def test_confidence_policy_preserves_rejection_and_unsupported_classes():
    m = importlib.import_module('scripts.runtime.mvp_static')
    catalog = m.build_catalog([{'text': 'pause', 'command': {'intent': 'PAUSE', 'slots': {}}}])
    p = {'threshold': .6, 'margin': .2, 'trained_classes': [0, 1]}
    assert m.decide([0., 4.], catalog, p)['command']['intent'] == 'PAUSE'
    assert m.decide([4., 0.], catalog, p)['command'] is None
    assert m.decide([0., .1], catalog, p)['command'] is None
    assert m.decide([0., 4.], catalog, {**p, 'trained_classes': [0]})['command'] is None
    with pytest.raises(ValueError):
        m.decide([float('nan'), 1.], catalog, p)



def test_calibration_never_uses_test_or_synthetic_as_human_evidence():
    m = importlib.import_module('scripts.runtime.mvp_static')
    rows = [dict(split='dev', synthetic=False, expected_index=1, logits=[0.,4.]),
            dict(split='calibration', synthetic=False, expected_index=0, logits=[4.,0.])]
    p = m.fit_policy(rows, [0, 1])
    assert p['fitting']['correct'] == 1 and p['fitting']['unknown_accepts'] == 0
    with pytest.raises(ValueError):
        m.fit_policy([{**rows[0], 'split': 'test'}], [0, 1])



@pytest.mark.parametrize('width', [128, 192])
def test_static_model_has_atomic_outputs_and_masks_padding(width):
    m = importlib.import_module('scripts.models.mvp_static')
    torch.manual_seed(53)
    net = m.StaticCommandNet(64, width=width).eval()
    x = torch.randn(1, 40, 31)
    with torch.no_grad():
        a = net(x, torch.tensor([31]))
        padded = torch.nn.functional.pad(x, (0, 20), value=300.)
        b = net(padded, torch.tensor([31]))
    assert a.shape == (1, 64)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert not any('ctc' in k or 'phones' in k for k in net.state_dict())
    assert sum(p.numel() for p in net.parameters()) < (1200000 if width == 128 else 2400000)



def test_export_has_variable_length_logits_and_ort_parity(tmp_path):
    from scripts.models.mvp_static import StaticCommandNet
    m = importlib.import_module('scripts.train.iris_export')
    net = StaticCommandNet(3, 128).eval()
    receipt = m.export_onnx(net, tmp_path / 'model.onnx')
    assert receipt['passed'] and receipt['max_absolute_error'] < 2e-5
    assert receipt['frames'] == [1, 3, 37, 128, 231]



def test_ncnn_preserves_selected_model_on_variable_lengths(tmp_path):
    from scripts.models.mvp_static import StaticCommandNet
    m=importlib.import_module('scripts.train.iris_export')
    net=StaticCommandNet(3,128).eval()
    m.export_onnx(net,tmp_path/'model.onnx')
    receipt=m.export_ncnn(net,tmp_path/'model.onnx',tmp_path/'ncnn')
    assert receipt['parity']['passed']
    assert receipt['parity']['max_absolute_error']<2e-5



def test_runtime_uses_finite_policy_and_rejects_bad_capture(tmp_path):
    from scripts.models.mvp_static import StaticCommandNet
    m=importlib.import_module('scripts.train.iris_export')
    runtime=importlib.import_module('scripts.runtime.mvp_static')
    net=StaticCommandNet(2,128).eval()
    with torch.no_grad():net.classifier.weight.zero_();net.classifier.bias.copy_(torch.tensor([0.,8.]))
    m.export_onnx(net,tmp_path/'model.onnx')
    catalog=[{'command_id':0,'command':None}, {'command_id':1,'command':{'intent':'PAUSE','slots':{}}}]
    for name,value in {'catalog.json':catalog,'normalization.json':{'mean':[0]*40,'std':[1]*40},
                       'policy.json':{'threshold':.8,'margin':.1,'trained_classes':[0,1],'enabled':True,'qualified':False}}.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'bundle.json').write_text(json.dumps({'format':'mvp-static-v1',
        'files':{name:runtime.digest(tmp_path/name) for name in ('model.onnx','catalog.json','normalization.json','policy.json')}}))
    engine=runtime.Runtime(tmp_path)
    pcm=np.random.default_rng(53).normal(0,.03,16000).astype('float32')
    assert engine.recognize(pcm)['output']['command']=={'intent':'PAUSE','slots':{}}
    assert engine.recognize(np.zeros(16000,dtype='float32'))['output']['reason']=='capture_fault'
    assert runtime.Runtime(tmp_path,mode='qualified').recognize(pcm)['output']['reason']=='unqualified_model'
    with pytest.raises(ValueError,match='Owner'):runtime.Runtime(tmp_path,profile='owner-protected')
    (tmp_path/'policy.json').write_text('{}')
    with pytest.raises(ValueError,match='identity'):runtime.Runtime(tmp_path)



def test_family_calibration_preserves_easy_commands_without_rescuing_confusions():
    m=importlib.import_module('scripts.runtime.mvp_static')
    catalog=m.build_catalog([{'text':'pause','command':{'intent':'PAUSE','slots':{}}},
                            {'text':'timer','command':{'intent':'TIMER','slots':{'duration_seconds':10}}}])
    rows=[dict(split='dev',synthetic=False,expected_index=1,logits=[0.,4.,0.]),
          dict(split='calibration',synthetic=False,expected_index=1,logits=[0.,0.,8.]),
          dict(split='dev',synthetic=False,expected_index=0,logits=[8.,0.,0.])]
    policy=m.fit_policy(rows,[0,1,2],catalog=catalog)
    assert m.decide(rows[0]['logits'],catalog,policy)['command']=={'intent':'PAUSE','slots':{}}
    assert m.decide(rows[1]['logits'],catalog,policy)['command'] is None
    assert m.decide(rows[2]['logits'],catalog,policy)['command'] is None
    unobserved=m.fit_policy([rows[0],rows[2]],[0,1,2],catalog=catalog)
    assert m.decide(rows[1]['logits'],catalog,unobserved)['command'] is None



def test_metrics_keep_unknown_ambiguity_and_wrong_accept_denominators():
    m = importlib.import_module('scripts.test.iris_metrics')
    records = [dict(expected_index=1, raw_index=1, accepted_index=1),
               dict(expected_index=1, raw_index=2, accepted_index=2),
               dict(expected_index=1, raw_index=1, accepted_index=None),
               dict(expected_index=0, raw_index=1, accepted_index=1),
               dict(expected_index=None, raw_index=1, accepted_index=1)]
    result = m.count_metrics(records)
    assert result['positive_n'] == 3 and result['negative_n'] == 1 and result['ambiguous_n'] == 1
    assert result['raw_correct'] == 2 and result['correct_accepts'] == 1
    assert result['wrong_accepted_positives'] == 1 and result['unknown_accepts'] == 1
    assert result['false_rejects'] == 1 and result['precision'] == pytest.approx(1/3)



def test_direct_worker_keeps_one_authoritative_decision():
    from scripts.runtime.speaker_worker import AudioService
    command = {'intent':'PAUSE', 'slots':{}}
    class Direct:
        decision_authority = 'atomic_command'
    events=[]
    service=AudioService(Direct(), events.append, settings={'inference':{'mode':'demo', 'intent_threshold':.95,'slot_threshold':.9},'actions':{'enabled':False}})
    event={'event':'command','decision_authority':'atomic_command',
           'output':{'accepted':True, 'command':command, 'reason':None},
           'diagnostic_candidate':{'candidate':command, 'class_probability':.8}}
    service.emit(event)
    assert events[-1]['application']['accepted'] is True
    assert events[-1]['application']['command'] == command
    service.emit({**event,'output':{'accepted':False,'command':None,'reason':'uncertain_or_unsupported'}})
    assert events[-1]['application']['accepted'] is False
    assert events[-1]['application']['command'] is None



def test_rejection_obeys_existing_typed_output_schema():
    m=importlib.import_module('scripts.runtime.mvp_static')
    from scripts.runtime.pcv_schema import output
    catalog=m.build_catalog([{'text':'pause','command':{'intent':'PAUSE','slots':{}}}])
    decision=m.decide([4.,0.],catalog,{'threshold':.9,'margin':0.,'trained_classes':[0,1]})
    assert output(decision['command'],decision['reason']) == {
        'schema_version':3,'accepted':False,'command':None,'reason':'OOV'}



def test_practical_policy_reports_known_errors_but_keeps_unknown_constraint():
    m=importlib.import_module('scripts.runtime.mvp_static')
    rows=[dict(split='dev',synthetic=False,expected_index=1,logits=[0.,4.,0.]) for _ in range(39)]
    rows += [dict(split='calibration',synthetic=False,expected_index=2,logits=[0.,4.,0.]),
             dict(split='calibration',synthetic=False,expected_index=0,logits=[0.,.1,0.])]
    policy=m.fit_policy(rows,[0,1,2],risk={'human':.03,'synthetic':.01})
    assert policy['fitting']['correct']==39 and policy['fitting']['wrong']==1
    assert policy['fitting']['unknown_accepts']==0
    assert m.fit_policy(rows,[0,1,2])['fitting']['correct']==0



def test_direct_worker_does_not_import_historical_decoder_or_torch():
    import subprocess,sys
    from pathlib import Path
    result=subprocess.run([sys.executable,'-c',
        'import sys; import scripts.runtime.speaker_worker; print("pcvcm.decoder" in sys.modules, "torch" in sys.modules)'],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='False False'



def test_slot_metrics_do_not_credit_wrong_target_as_complete_command():
    from scripts.test.iris_metrics import summarize
    catalog=[{'command':None},{'command':{'intent':'BRIGHTNESS','slots':{'light_id':'light.ceiling','percent':20}}},
        {'command':{'intent':'BRIGHTNESS','slots':{'light_id':'light.kitchen','percent':20}}}]
    rows=[dict(expected_index=1,raw_index=2,accepted_index=2,synthetic=False,speaker='a',source='human')]
    summary=summarize(rows,catalog)
    assert summary['overall']['correct_accepts']==0
    roles=summary['per_slot_role']
    assert roles['BRIGHTNESS/percent/human']['accepted_correct']==1
    assert roles['BRIGHTNESS/light_id/human']['accepted_correct']==0
    assert roles['BRIGHTNESS/light_id/human']['wrong_accepted']==1
