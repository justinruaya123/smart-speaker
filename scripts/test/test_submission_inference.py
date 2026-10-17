"""Submission inference keeps compact decisions and separates recognition from actions."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def accepted_event():
    command = {'intent': 'TIMER', 'slots': {'duration_seconds': 30}}
    return {
        'event': 'command', 'decision_mode': 'demo',
        'output': {'schema_version': 3, 'accepted': True, 'command': command, 'reason': None},
        'qualified_output': {'accepted': False, 'command': None, 'reason': 'unqualified_model'},
        'application': {'accepted': True, 'command': command, 'intent_threshold': .7},
        'diagnostic_candidate': {'candidate': command, 'intent_probability': .81,
                                 'slot_confidence': {'duration_seconds': .95}},
    }


class InferenceTests(unittest.TestCase):
    def module(self, name='inference'):
        return importlib.import_module('scripts.submission.' + name)

    def test_root_help_is_standard_library_only(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT/'inference.py'), '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--mode', result.stdout)
        self.assertIn('--speaker', result.stdout)

    def test_submission_defaults_to_the_requested_full_model(self):
        self.assertEqual(self.module().parser().parse_args([]).model, ROOT/'submission/models/full')

    def test_debug_preserves_typed_slots_and_separate_qualification(self):
        value = self.module().debug_event(accepted_event())
        self.assertEqual(value['accepted_intent'], 'TIMER')
        self.assertEqual(value['slots'], {'duration_seconds': 30})
        self.assertIs(type(value['slots']['duration_seconds']), int)
        self.assertEqual(value['intent_confidence'], .81)
        self.assertEqual(value['slot_confidence'], {'duration_seconds': .95})
        self.assertFalse(value['qualified_accepted'])
        self.assertFalse(value['actions_enabled'])

    def test_rejected_candidate_is_never_reported_as_accepted(self):
        event = accepted_event()
        event['output'] = {'accepted': False, 'command': None, 'reason': 'OOV'}
        event['application'] = {'accepted': False, 'command': None, 'reason': 'intent_confidence'}
        value = self.module().debug_event(event)
        self.assertEqual(value['candidate']['intent'], 'TIMER')
        self.assertIsNone(value['accepted_intent'])
        self.assertEqual(value['slots'], {})

    def test_ui_environment_preserves_explicit_provider_flags(self):
        module = self.module()
        with patch.dict('os.environ', {'ACTIONS_ENABLED':'true', 'HA_ENABLED':'true',
                                       'SPOTIFY_ENABLED':'true', 'INFERENCE_ENGINE':'ncnn'}, clear=True):
            env = module.ui_environment(ROOT, ROOT/'submission/models/compact', 2, 3009, 4, 7)
        self.assertEqual(env['SMART_SPEAKER_WORKER_MODULE'], 'scripts.submission.worker')
        for key in ('ACTIONS_ENABLED', 'HA_ENABLED', 'SPOTIFY_ENABLED'):
            self.assertEqual(env[key], 'true')
        self.assertEqual(env['INFERENCE_ENGINE'], 'onnxruntime')
        self.assertEqual(env['SUBMISSION_SPEAKER_INDEX'], '7')
        self.assertEqual(env['SUBMISSION_MICROPHONE_INDEX'], '4')

    def test_ui_uses_requested_entity_allowlist_with_disabled_defaults(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict('os.environ', {}, clear=True):
            env = self.module().ui_environment(Path(temporary), ROOT/'submission/models/full', 1, 3000, None, None)
        self.assertEqual(env['HA_ENTITY_IDS'],
                         'light.hue_color_lamp_1,light.dining_light,camera.dining_area_main_camera_tc70_mainstream')
        for key in ('ACTIONS_ENABLED', 'HA_ENABLED', 'SPOTIFY_ENABLED'):
            self.assertEqual(env[key], 'false')
        self.assertEqual(json.loads(env['SMART_SPEAKER_BINDINGS'])['lights']['light.all'],'light.hue_color_lamp_1')

    def test_ui_loads_private_root_env_and_shell_overrides_it(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict('os.environ', {'HA_TOKEN':'shell-token'}, clear=True):
            root = Path(temporary)
            (root/'.env').write_text('HA_ENABLED=true\nHA_URL=http://ha.invalid\nHA_TOKEN=file-token\n'
                                    'ACTIONS_ENABLED=true\nSPOTIFY_ENABLED=true\n'
                                    'SPOTIFY_CLIENT_ID=local-client\nSPOTIFY_CLIENT_SECRET=local-secret\n')
            env = self.module().ui_environment(root, ROOT/'submission/models/full', 1, 3009, None, None)
        self.assertEqual(env['HA_TOKEN'], 'shell-token')
        self.assertEqual(env['HA_URL'], 'http://ha.invalid')
        self.assertEqual(env['SPOTIFY_CLIENT_SECRET'], 'local-secret')
        self.assertEqual(env['SPOTIFY_REDIRECT_URI'], 'http://127.0.0.1:3009/auth/spotify/callback')
        self.assertEqual(env['ACTIONS_ENABLED'], 'true')

    def test_ui_honors_configured_bindings_and_entity_override(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict('os.environ', {}, clear=True):
            root = Path(temporary)
            (root/'config.toml').write_text('[actions]\nenabled = true\n[providers]\n'
                                          'ha_enabled = true\nha_entity_ids = "light.custom"\n'
                                          '[bindings.lights]\n"light.kitchen" = "light.custom"\n')
            env = self.module().ui_environment(root, ROOT/'submission/models/full', 1, 3000, None, None)
        self.assertEqual(env['HA_ENTITY_IDS'], 'light.custom')
        self.assertEqual(json.loads(env['SMART_SPEAKER_BINDINGS'])['lights'], {'light.kitchen':'light.custom','light.all':'light.hue_color_lamp_1'})
        self.assertEqual(env['HA_ENABLED'], 'true')
        self.assertEqual(env['ACTIONS_ENABLED'], 'true')

    def test_audio_selection_rejects_wrong_direction_and_missing_device(self):
        devices = [{'index': 1, 'name':'Input', 'inputs':1, 'outputs':0},
                   {'index': 3, 'name':'Output', 'inputs':0, 'outputs':2}]
        module = self.module()
        self.assertEqual(module.select_device(devices, 'speaker', 3)['index'], 3)
        for direction, index in [('speaker', 1), ('microphone', 3), ('microphone', 9)]:
            with self.subTest(direction=direction,index=index), self.assertRaises(ValueError):
                module.select_device(devices, direction, index)

    def test_cli_does_not_resolve_or_play_response_assets(self):
        module = self.module()
        with patch.object(module, 'audio_devices', return_value=[]), \
             patch.object(module, 'load_runtime') as load, \
             patch.object(module, 'ensure_responses', side_effect=AssertionError('CLI response playback')):
            runtime = load.return_value
            runtime.recognize.return_value = accepted_event()
            with patch.object(module, 'read_waveform', return_value=[]), patch('builtins.print'):
                result = module.main(['--no-setup', '--mode','cli','--wav','fixture.wav'])
        self.assertEqual(result, 0)

    def test_prebuilt_ui_does_not_install_pnpm_or_build_on_target(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('dist/index.html','dist-server/server/main.js','node_modules/fastify/package.json'):
                path=root/'SpeakerUI'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('compiled')
            with patch.object(module,'runtime_node',return_value=('node',{'PATH':'local'})) as runtime, \
                 patch('scripts.runtime.speaker_tools.prepare_ui',side_effect=AssertionError('target attempted UI build')):
                self.assertEqual(module.prepare_frontend(root),('node',{'PATH':'local'}))
            runtime.assert_called_once_with(root)

    def test_source_build_exposes_local_pnpm_to_nested_package_scripts(self):
        module=self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); pnpm=root/'.runtime/tools/pnpm/node_modules/pnpm/bin/pnpm.mjs'
            expected=str(pnpm.parents[2]/'.bin')
            def prepare(*args,**kwargs):
                import os
                self.assertIn(expected,os.environ['PATH'].split(os.pathsep))
                return 'node',{'PATH':'local'}
            with patch('scripts.runtime.speaker_tools.node_tools',return_value=('node',pnpm,{})), \
                 patch('scripts.runtime.speaker_tools.prepare_ui',side_effect=prepare):
                self.assertEqual(module.prepare_frontend(root),('node',{'PATH':'local'}))

    def test_worker_preserves_fitted_policy_instead_of_fixed_threshold(self):
        module = self.module('worker')
        class Base:
            def __init__(self):
                self.sequence=0; self.session='session'; self.source='microphone'; self.settings={'actions':{'enabled':True}}
                self.delivered=[]; self.deliver=self.delivered.append
            def emit(self, value):
                raise AssertionError('Frozen emit would overwrite the fitted application decision')
        service = module.service_class(Base)()
        service.emit(accepted_event())
        event = service.delivered[0]
        self.assertTrue(event['application']['accepted'])
        self.assertEqual(event['application']['intent_threshold'], .7)
        self.assertFalse(event['actions_enabled'])

    def test_gui_worker_reports_actions_only_for_opted_in_microphone(self):
        module = self.module('worker')
        class Base:
            def __init__(self):
                self.sequence=0; self.session='session'; self.source='microphone'
                self.settings={'actions':{'enabled':True}}
                self.delivered=[]; self.deliver=self.delivered.append
        service = module.service_class(Base, actions=True)()
        service.emit(accepted_event())
        self.assertTrue(service.delivered[-1]['actions_enabled'])
        service.source='wav'
        service.emit(accepted_event())
        self.assertFalse(service.delivered[-1]['actions_enabled'])
        service.source='microphone'; service.settings['actions']['enabled']=False
        service.emit(accepted_event())
        self.assertFalse(service.delivered[-1]['actions_enabled'])

    def test_worker_resumes_capture_from_zero_after_recorded_reply(self):
        module = self.module('worker')
        class Base:
            def tick(self):
                # Frozen tick adds this block after forward() returns. Playback has
                # reset the gate inside forward(), so the next input must start at zero.
                self.offset = 0
                self._playback_reset = True
                self.offset += 512
        service=module.service_class(Base)()
        service.tick()
        self.assertEqual(service.offset, 0)

    def test_cli_service_never_plays_even_if_playback_environment_is_inherited(self):
        module=self.module('worker')
        class Base:
            source='microphone'
            def forward(self,event): pass
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true'}), \
             patch.object(module,'ensure_responses',side_effect=AssertionError('CLI attempted playback')):
            module.service_class(Base)().forward(accepted_event())

    def test_ui_rejection_uses_selected_host_speaker_and_restarts_capture(self):
        module=self.module('worker')
        class Base:
            def __init__(self):
                self.sequence=0;self.session='s';self.source='microphone';self.events=[]
                self.deliver=self.events.append;self.stream=Mock();self.gate=Mock();self.offset=99
            def forward(self,event): pass
            def set_state(self,state): pass
        play=Mock();read=Mock(return_value=([.1,.2],24000))
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true','SUBMISSION_SPEAKER_INDEX':'7'}), \
             patch.dict('sys.modules',{'sounddevice':SimpleNamespace(play=play),'soundfile':SimpleNamespace(read=read)}), \
             patch.object(module,'ensure_responses',return_value={'not_recognized':Path('timer.wav')}):
            service=module.service_class(Base,playback=True)()
            service.forward({'event':'command','output':{'accepted':False,'command':None,'reason':'OOV'}})
        read.assert_called_once_with(Path('timer.wav'),dtype='float32')
        play.assert_called_once_with([.1,.2],24000,device=7,blocking=True)
        service.stream.stop.assert_called_once();service.stream.start.assert_called_once()
        service.gate.reset.assert_called_once()
        self.assertEqual(service.events[0]['claim'],'recognition_only')
        self.assertFalse(service.events[0]['actions_enabled'])


class ResponseTests(unittest.TestCase):
    def module(self):
        return importlib.import_module('scripts.submission.response_runtime')

    def manifest(self, data=b'RIFFfixture'):
        return {'format':'smart-speaker-fixed-responses-v1',
                'hub':{'repo_id':'owner/model', 'revision':'a'*40, 'subdirectory':'responses'},
                'responses':[{'id':'recognized_timer', 'path':'audio/recognized_timer.wav',
                              'text':'I recognized a timer request.', 'sha256':hashlib.sha256(data).hexdigest(),
                              'bytes':len(data)}]}

    def test_response_acknowledges_recognition_without_claiming_device_success(self):
        module = self.module()
        self.assertEqual(module.response_id(accepted_event()), 'recognized_timer')
        for row in module.RESPONSE_TEXTS.values():
            self.assertNotIn('successfully', row.lower())
        self.assertIn('recognized', module.RESPONSE_TEXTS['recognized_timer'])

    def test_rejection_and_capture_fault_have_distinct_responses(self):
        module = self.module()
        event = accepted_event()
        event['output'] = {'accepted':False,'command':None,'reason':'OOV'}
        event['application'] = {'accepted':False,'command':None}
        self.assertEqual(module.response_id(event), 'not_recognized')
        event['output']['reason'] = 'capture_fault'
        self.assertEqual(module.response_id(event), 'capture_fault')
        self.assertIsNone(module.response_id({'event':'state','state':'wake_listening'}))

    def test_response_manifest_requires_pinned_revision_and_safe_paths(self):
        module = self.module()
        for field, value in [('path','../private.wav'), ('path','/etc/passwd'),
                             ('path','audio\\bad.wav'), ('sha256','wrong')]:
            manifest = self.manifest(); manifest['responses'][0][field] = value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                module.validate_manifest(manifest)
        manifest = self.manifest(); manifest['hub']['revision']='inference'
        with self.assertRaises(ValueError):
            module.validate_manifest(manifest)

    def test_corrupted_cached_response_is_rejected_without_download(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); folder=root/'submission/responses/audio'; folder.mkdir(parents=True)
            (folder/'recognized_timer.wav').write_bytes(b'corrupted')
            with patch.object(module, 'fetch', side_effect=AssertionError('must not hide corruption')):
                with self.assertRaises(ValueError):
                    module.ensure_responses(root, self.manifest())

    def test_packaged_response_works_offline_without_huggingface_library(self):
        module = self.module(); data=b'RIFFfixture'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); folder=root/'submission/responses/audio'; folder.mkdir(parents=True)
            path=folder/'recognized_timer.wav'; path.write_bytes(data)
            with patch.object(module, 'fetch', side_effect=AssertionError('offline package must not download')):
                result=module.ensure_responses(root,self.manifest(data))
            self.assertEqual(result['recognized_timer'],path)

    def test_download_url_is_exactly_one_pinned_response_not_model_snapshot(self):
        module = self.module(); data=b'RIFFfixture'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def fake_fetch(url,path,expected):
                self.assertEqual(url,'https://huggingface.co/owner/model/resolve/'+'a'*40+'/responses/audio/recognized_timer.wav')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
            with patch.object(module,'fetch',side_effect=fake_fetch):
                result=module.ensure_responses(root,self.manifest(data))
            self.assertEqual(result['recognized_timer'].read_bytes(),data)

    def test_public_dataset_response_url_uses_dataset_prefix(self):
        module=self.module();data=b'RIFFfixture';manifest=self.manifest(data)
        manifest['hub']['repo_type']='dataset'
        with tempfile.TemporaryDirectory() as temporary:
            def fake_fetch(url,path,expected):
                self.assertEqual(url,'https://huggingface.co/datasets/owner/model/resolve/'+'a'*40+'/responses/audio/recognized_timer.wav')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
            with patch.object(module,'fetch',side_effect=fake_fetch):
                result=module.ensure_responses(Path(temporary),manifest)
            self.assertEqual(result['recognized_timer'].read_bytes(),data)
        manifest['hub']['repo_type']='space'
        with self.assertRaises(ValueError):
            module.validate_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
