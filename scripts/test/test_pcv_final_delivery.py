"""Portable endpoint runtime guards and benchmark denominator regressions."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from scripts.runtime.pcv_final_runtime import Runtime, verify_bundle, decision, BINDINGS, sha
from scripts.runtime.pcv_pi_benchmark import summarize,summarize_session,write_report,run
from scripts.pcv_final_delivery import qualification_files,add_extra_files
from scripts.runtime.pcv_schema import DEFAULT_CUSTOM,INTENTS
from vcm.catalog import PHONES
from vcm.frontend import SPEC


def make_fixture_bundle(folder):
    """A generated signal-routing ONNX fixture, never a trained/qualified model.

    Dynamic input frames are subsampled by two and exposed as 40/19/2 raw
    heads. This exercises actual ONNX execution and time-layout guards without
    reading any experimental payload, recording, checkpoint or frozen receipt.
    """
    import onnx
    from onnx import helper,TensorProto
    root=Path(folder)
    constants=[helper.make_tensor(name,TensorProto.INT64,[1],[value]) for name,value in
               [('start',0),('end',2**63-1),('time_axis',1),('stride',2),
                ('class_axis',2),('one',1),('intent_end',19),('target_end',2)]]
    nodes=[helper.make_node('Transpose',['features'],['time_first'],perm=[0,2,1]),
           helper.make_node('Slice',['time_first','start','end','time_axis','stride'],['ctc']),
           helper.make_node('Slice',['ctc','start','intent_end','class_axis','one'],['intent']),
           helper.make_node('Slice',['ctc','start','target_end','class_axis','one'],['target'])]
    graph=helper.make_graph(nodes,'portable_runtime_test_fixture',
        [helper.make_tensor_value_info('features',TensorProto.FLOAT,[1,40,'N'])],
        [helper.make_tensor_value_info(name,TensorProto.FLOAT,[1,'T',width])
         for name,width in [('ctc',40),('intent',19),('target',2)]],initializer=constants)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
    onnx.checker.check_model(model);onnx.save(model,root/'model.onnx')
    def write(name,value): (root/name).write_text(json.dumps(value),encoding='utf-8')
    write('normalization.json',{'mean':[0.]*40,'std':[1.]*40,'frontend':SPEC})
    write('custom-slots.json',DEFAULT_CUSTOM)
    write('catalog.json',{'entries':[
        {'id':'fixture-time','command':{'intent':'TIME','slots':{}},'phones':[1,2],'use_case':'TIME'},
        {'id':'fixture-weather','command':{'intent':'WEATHER','slots':{}},'phones':[3,4],'use_case':'WEATHER'}]})
    write('output.schema.json',{'$schema':'https://json-schema.org/draft/2020-12/schema','type':'object',
        'required':['schema_version','accepted','command','reason'],'additionalProperties':False,
        'properties':{'schema_version':{'const':3},'accepted':{'type':'boolean'},
            'command':{'type':['object','null']},
            'reason':{'enum':[None,'OOV','invalid_argument','capture_fault','unqualified_model']}}})
    binding={key:sha(root/name) for key,name in BINDINGS.items()}
    write('policy.json',{'binding':binding,'enabled':False,'min_score':-100.,'min_margin':0.,'min_intent_probability':0.})
    write('recipe.json',{'inference_mode':'complete_utterance','state_shapes':[],
        'heads':['ctc','intent','target'],'output_hop_ms':20,'feature_hop_ms':10,
        'binding':binding,'policy_enabled':False,'status':'generated_software_fixture'})
    return root


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='pcv-runtime-test-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.bundle=make_fixture_bundle(cls.temporary.name)
        cls.runtime = Runtime(cls.bundle, diagnostic=True)

    @classmethod
    def tearDownClass(cls):
        # Release ONNX's file handles before Windows removes the temp directory.
        cls.runtime=None

    def test_disabled_policy_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'diagnostic'):
            Runtime(self.bundle)

    def test_corruption_and_state_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in self.bundle.iterdir():
                if path.is_file(): shutil.copyfile(path, root/path.name)
            (root/'catalog.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'identity'):
                verify_bundle(root, diagnostic=True)
            shutil.copyfile(self.bundle/'catalog.json', root/'catalog.json')
            recipe=json.loads((root/'recipe.json').read_text())
            recipe['state_shapes']=[[1,40,2]]
            (root/'recipe.json').write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, 'state'):
                verify_bundle(root, diagnostic=True)

    def test_bad_input_and_silence(self):
        for data,rate in [(np.zeros(16000),8000),(np.zeros((1,16000)),16000),
                          (np.array([np.nan]),16000),(np.array([2.]),16000)]:
            with self.assertRaises(ValueError): self.runtime.recognize(data,rate)
        event=self.runtime.recognize(np.zeros(16000,dtype='float32'),16000)
        self.assertFalse(event['output']['accepted'])
        self.assertEqual(event['output']['reason'],'capture_fault')
        self.assertIsNone(event['diagnostic_candidate'])

    def test_declared_head_labels_must_match_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for path in self.bundle.iterdir(): shutil.copyfile(path,root/path.name)
            recipe=json.loads((root/'recipe.json').read_text())
            recipe.update(architecture_class='PooledContextVCM',intent_labels=['OOV',*INTENTS],
                          acoustic_labels=['<blank>',*PHONES])
            for key in ('intent_labels','acoustic_labels'):
                wrong={**recipe,key:list(reversed(recipe[key]))}
                (root/'recipe.json').write_text(json.dumps(wrong))
                with patch('scripts.runtime.pcv_final_runtime.ort.InferenceSession') as session:
                    with self.assertRaisesRegex(ValueError,'labels'):Runtime(root,diagnostic=True)
                    session.assert_not_called()
                missing={k:v for k,v in recipe.items() if k!=key}
                (root/'recipe.json').write_text(json.dumps(missing))
                with self.assertRaisesRegex(ValueError,'labels'):verify_bundle(root,diagnostic=True)
            (root/'recipe.json').write_text(json.dumps(recipe))
            verify_bundle(root,diagnostic=True)

    def test_onnx_state_interface_rejected(self):
        fake=SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='features',type='tensor(float)',shape=[1,40,'N']),
                                                 SimpleNamespace(name='state',type='tensor(float)',shape=[1,40,2])],
                             get_outputs=lambda:[])
        with patch('scripts.runtime.pcv_final_runtime.ort.InferenceSession',return_value=fake):
            with self.assertRaisesRegex(ValueError,'state interface'):Runtime(self.bundle,diagnostic=True)

    def test_generated_model_executes_dynamic_three_head_contract(self):
        for frames in (1,3,16,37,80):
            arrays=self.runtime.session.run(None,{'features':np.zeros((1,40,frames),dtype='float32')})
            self.assertEqual([a.shape for a in arrays],[(1,(frames+1)//2,width) for width in (40,19,2)])
        pcm=(.1*np.sin(2*np.pi*440*np.arange(16000)/16000)).astype('float32')
        event=self.runtime.recognize(pcm)
        self.assertEqual(event['output']['reason'],'unqualified_model')
        self.assertFalse(event['output']['accepted'])
        self.assertIn(event['diagnostic_candidate']['candidate']['intent'],('TIME','WEATHER'))
        self.assertGreater(event['timing']['rtf'],0)

    def test_diagnostic_never_becomes_accepted(self):
        record={'candidate':{'intent':'TIME','slots':{}},'score':0.,'margin':1.,
                'intent_index':3,'intent_probability':1.,'top_intent_probability':1.,
                'target_named_probability':0.}
        self.assertEqual(decision(record,{'enabled':False})['reason'],'unqualified_model')
        enabled={'enabled':True,'min_score':-1.,'min_margin':0.,'min_intent_probability':.5}
        self.assertTrue(decision(record,enabled)['accepted'])

    def test_missing_recordings_are_untested(self):
        definitions=[{'id':'a','command':{'intent':'TIME','slots':{}}},
                     {'id':'b','command':None}]
        metrics=summarize([],definitions)
        self.assertEqual(metrics['untested_phrase_count'],2)
        self.assertIsNone(metrics['accepted']['recall'])
        self.assertEqual(metrics['accepted']['positive_n'],0)

    def test_cross_intent_false_predictions_count_in_precision(self):
        command={'intent':'TIME','slots':{}}
        rows=[{'id':'a','phrase_id':None,'expected':command,'accepted':command,'candidate':command,'event':{}},
              {'id':'b','phrase_id':None,'expected':{'intent':'WEATHER','slots':{}},'accepted':command,'candidate':command,'event':{}}]
        report=summarize(rows,[])
        self.assertEqual(report['per_intent']['TIME']['accepted']['precision'],.5)

    def test_rejection_reason_is_part_of_typed_schema_accuracy(self):
        expected={'schema_version':3,'accepted':False,'command':None,'reason':'invalid_argument'}
        row={'id':'negative','phrase_id':None,'expected':None,'expected_output':expected,
             'accepted':None,'candidate':None,'event':{'output':{**expected,'reason':'unqualified_model'}}}
        report=summarize([row],[])
        self.assertEqual(report['typed_output'],{'n':1,'exact':0,'reason_unannotated_n':0})

    def test_per_phrase_typed_output_distinguishes_negative_reasons(self):
        expected={'schema_version':3,'accepted':False,'command':None,'reason':'OOV'}
        base={'id':'negative','phrase_id':'negative','expected':None,'expected_output':expected,
              'accepted':None,'candidate':None}
        rows=[{**base,'event':{'output':expected}},
              {**base,'event':{'output':{**expected,'reason':'invalid_argument'}}},
              {**base,'expected_output':None,'event':{'output':expected}}]
        report=summarize(rows,[{'id':'negative'},{'id':'unrecorded'}])
        measured=report['phrases']['negative']
        self.assertEqual(measured['trials'],3)
        self.assertEqual(measured['typed_output'],{'n':2,'exact':1,'reason_unannotated_n':1})
        self.assertEqual(report['phrases']['unrecorded']['typed_output'],
                         {'n':0,'exact':0,'reason_unannotated_n':0})
        self.assertEqual(report['untested_phrase_count'],1)

    def test_replay_cannot_inflate_guided_live_metrics_or_latency(self):
        wanted={'intent':'TIME','slots':{}};wrong={'intent':'WEATHER','slots':{}}
        definitions=[{'id':'time','command':wanted},{'id':'weather','command':wrong}]
        def row(mode,actual,latency):
            return {'id':mode,'phrase_id':'time','expected':wanted,'accepted':actual,'candidate':actual,
                    'mode':mode,'capture_completed':True,'event':{'timing':{'total_ms':latency,'rtf':latency/1000}}}
        result=summarize_session([row('replay',wanted,1),row('guided_live',wrong,50)],definitions,guided=True)
        self.assertEqual((result['accepted']['exact'],result['accepted']['positive_n']),(0,1))
        self.assertEqual((result['strata']['replay']['accepted']['exact'],result['strata']['replay']['accepted']['positive_n']),(1,1))
        self.assertEqual(result['phrases']['time']['trials'],1)
        self.assertEqual(result['untested_phrase_count'],1)
        self.assertEqual(result['latency_ms']['p50'],50)
        self.assertEqual(result['strata']['replay']['latency_ms']['p50'],1)
        self.assertEqual(result['rtf']['p50'],.05)
        replay_only=summarize_session([row('replay',wanted,1),row('guided_live',wrong,50)],definitions,guided=False)
        self.assertEqual(replay_only['selected_metric_mode'],'replay')
        self.assertEqual(replay_only['accepted']['exact'],1)
        with tempfile.TemporaryDirectory() as folder:
            result.update(utc='fixture',hardware={'device_model':'fixture','os':'fixture','physical_pi_measured':False},
                mode='replay_and_guided_live',policy_enabled=True,session_load_ms=0,
                wake={'session_load_ms':[]},peak_rss_mb=None)
            write_report(Path(folder),result)
            report=(Path(folder)/'REPORT.md').read_text(encoding='utf-8')
            self.assertIn('Headline metric mode: **guided_live**',report)
            live_section=report.split('## Completed live captures')[1].split('## Prerecorded replay')[0]
            replay_section=report.split('## Prerecorded replay')[1].split('## Per-intent')[0]
            self.assertIn('| Accepted exact commands | 0/1 |',live_section)
            self.assertIn('| Accepted exact commands | 1/1 |',replay_section)

    def test_failed_live_capture_keeps_attempt_denominator_without_replay_coverage(self):
        wanted={'intent':'TIME','slots':{}}
        definitions=[{'id':'time','command':wanted}]
        replay={'id':'replay','phrase_id':'time','expected':wanted,'accepted':wanted,'candidate':wanted,
                'mode':'replay','capture_completed':True,'event':{}}
        live={**replay,'id':'live','mode':'guided_live','capture_completed':False,'accepted':None,'candidate':None,
              'event':{'output':{'accepted':False,'command':None,'reason':'capture_fault'},'capture_status':'no_completed_command'}}
        result=summarize_session([replay,live],definitions,guided=True)
        self.assertEqual(result['accepted']['n'],0)
        self.assertEqual(result['untested_phrase_count'],1)
        self.assertEqual(result['strata']['replay']['untested_phrase_count'],0)
        self.assertEqual(result['live_attempts']['positive_n'],1)
        self.assertEqual(result['live_attempts']['exact'],0)
        self.assertEqual(result['live_attempts']['completed_n'],0)
        self.assertEqual(result['live_attempts']['no_completed_command_n'],1)
        self.assertFalse(result['live_microphone_measured'])

    def test_capture_fault_is_not_successful_typed_negative(self):
        expected={'schema_version':3,'accepted':False,'command':None,'reason':'OOV'}
        row={'id':'negative','phrase_id':'negative','expected':None,'expected_output':expected,
             'accepted':None,'candidate':None,'mode':'guided_live','capture_completed':True,
             'event':{'output':{**expected,'reason':'capture_fault'}}}
        result=summarize_session([row],[{'id':'negative'}],guided=True)
        self.assertEqual(result['typed_output']['exact'],0)
        self.assertEqual(result['live_attempts']['typed_output_exact'],0)
        self.assertEqual(result['live_attempts']['capture_fault_n'],1)
        self.assertTrue(result['live_microphone_measured'])

    def test_guided_flow_distinguishes_capture_completion_without_microphone(self):
        wanted={'intent':'TIME','slots':{}}
        fault={'schema_version':3,'accepted':False,'command':None,'reason':'capture_fault'}
        events=[([],False),([{'event':'command','output':fault,'capture_status':'overflow_or_underflow'}],False),
                ([{'event':'command','output':fault,'audio_seconds':1.}],True)]
        fake=SimpleNamespace(engine='onnxruntime',root=self.bundle,policy={'enabled':False},recipe=self.runtime.recipe,load_ms=0,
                             session=SimpleNamespace(get_providers=lambda:['CPUExecutionProvider']))
        for emitted,completed in events:
            with self.subTest(emitted=emitted),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);(root/'verbatims').mkdir();(root/'samples').mkdir()
                (root/'verbatims/definitions.json').write_text(json.dumps([
                    {'id':'time','command':wanted,'spoken_text':'what time is it'},
                    {'id':'other','command':None}]))
                (root/'samples/manifest.json').write_text('[]')
                args=SimpleNamespace(root=root,output=root/'result',threads=1,seconds=1,no_wake=True,
                                     manifest=None,guided=True,case=['time'],mic=999)
                with patch('scripts.runtime.pcv_pi_benchmark.Runtime',return_value=fake), \
                     patch('scripts.runtime.pcv_pi_benchmark.capture',return_value=iter(emitted)), \
                     patch('builtins.input',return_value=''),patch('builtins.print'):
                    result=run(args)
                self.assertEqual(result['live_microphone_measured'],completed)
                self.assertEqual(result['live_attempts']['positive_n'],1)
                self.assertEqual(result['live_attempts']['exact'],0)
                self.assertEqual(result['untested_phrase_count'],1 if completed else 2)
                self.assertEqual(result['requested_live_case_ids'],['time'])
                self.assertEqual(result['strata']['replay']['accepted']['n'],0)

    def test_enabled_publication_needs_matching_qualification(self):
        from scripts.runtime.pcv_final_runtime import sha
        with self.assertRaisesRegex(ValueError,'requires'):qualification_files(None,{},'policy')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);result=root/'result.json';receipt=root/'receipt.json'
            result.write_text(json.dumps({'qualified':True,'model_binding':{'model':'abc'},'policy_sha256':'policy'}))
            data={'qualified':True,'model_binding':{'model':'abc'},'test_used_for_selection':False,
                  'results_path':'result.json','results_sha256':sha(result),'policy_sha256':'policy'}
            receipt.write_text(json.dumps(data))
            self.assertEqual(len(qualification_files(receipt,{'model':'abc'},'policy')),2)
            with self.assertRaisesRegex(ValueError,'match'):qualification_files(receipt,{'model':'changed'},'policy')
            with self.assertRaisesRegex(ValueError,'match'):qualification_files(receipt,{'model':'abc'},'changed')
            result.write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity'):qualification_files(receipt,{'model':'abc'},'policy')

    def test_extra_archive_paths_are_bounded(self):
        for name in ('../x.md','docs/../../x.md','scripts/x.py','docs/x/y.md','C:/x.md','docs\\x.md'):
            with self.assertRaises(ValueError):add_extra_files({}, {name:Path(__file__)})


if __name__=='__main__': unittest.main()
