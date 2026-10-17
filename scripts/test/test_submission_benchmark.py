"""Portable benchmark closure and transport behavior; no acoustic fitting."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from types import SimpleNamespace

from scripts.submission import benchmark, common


class PortableBenchmarkTests(unittest.TestCase):
    def model_module(self):
        return importlib.import_module('scripts.submission.model')

    def test_shared_model_adapter_help_imports_without_site_packages(self):
        result = subprocess.run([sys.executable,'-S','-c',
            'import sys; from scripts.submission.model import load_runtime; '
            'assert "numpy" not in sys.modules; assert "torch" not in sys.modules'],
            cwd=common.ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_full_model_uses_fixed_demo_gate_and_retains_qualified_rejection(self):
        module = self.model_module()
        recipe = common.read(common.ROOT/'submission/models/full/recipe.json')
        command = {'intent':'PLAY_MUSIC','slots':{}}
        event = {'output':{'accepted':False,'command':None,'reason':'unqualified_model'},
                 'diagnostic_candidate':{'candidate':command,'intent_index':1,'intent_probability':.96,
                                         'slot_confidence':{}},
                 'audio_seconds':1.,'timing':{'total_ms':10.,'rtf':.01}}
        fake = type('Runtime',(),{'recipe':recipe,'root':common.ROOT/'submission/models/full',
                                 'recognize':lambda self,*a:event})()
        with patch('scripts.runtime.pcv_final_runtime.Runtime',return_value=fake):
            runtime = module.load_runtime(fake.root,threads=1)
            accepted = runtime.recognize([],16000)
            self.assertEqual(accepted['application']['command'],command)
            self.assertEqual(accepted['output']['command'],command)
            self.assertEqual(accepted['application']['intent_threshold'],.95)
            self.assertEqual(accepted['application']['slot_threshold'],.9)
            self.assertEqual(accepted['qualified_output']['reason'],'unqualified_model')
            self.assertGreaterEqual(accepted['timing']['total_ms'],10.)
            event['diagnostic_candidate']['intent_probability'] = .94
            rejected = runtime.recognize([],16000)
            self.assertIsNone(rejected['output']['command'])
            self.assertEqual(rejected['application']['reason'],'intent_confidence')
            self.assertEqual(event['output']['reason'],'unqualified_model')

    def test_compact_model_still_uses_its_bound_recall_policy(self):
        module = self.model_module()
        with patch('scripts.runtime.speaker_compact.CompactRuntime') as compact:
            result = module.load_runtime(common.ROOT/'submission/models/compact',threads=2)
            self.assertIs(result,compact.return_value)
            self.assertEqual(compact.call_args.kwargs['threads'],2)

    def test_cached_audio_does_not_require_parquet_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = b'original recording'
            target = root/'.cache/submission/holdout/0000.wav'
            target.parent.mkdir(parents=True)
            target.write_bytes(audio)
            spec = {'source': {}, 'rows': [{'audio': '0000.wav', 'sha256': hashlib.sha256(audio).hexdigest()}]}
            source = root/'submission/benchmark/holdout.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(spec))
            with patch.object(benchmark, 'fetch', side_effect=AssertionError('Unexpected download')):
                actual, directory = benchmark.prepare_samples(root)
            self.assertEqual(actual, spec)
            self.assertEqual(directory, target.parent)

    def test_cached_audio_tampering_fails_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root/'.cache/submission/holdout/0000.wav'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'changed')
            source = root/'submission/benchmark/holdout.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'source': {}, 'rows': [{'audio': '0000.wav', 'sha256': '0'*64}]}))
            with self.assertRaisesRegex(ValueError, 'Cached holdout changed'):
                benchmark.prepare_samples(root)

    def test_archive_closure_can_package_again_without_repository(self):
        entries = benchmark.runtime_entries()
        self.assertNotIn('scripts/pcv_final_delivery.py', entries)
        self.assertIn('submission/models/full/model.onnx', entries)
        self.assertNotIn('submission/models/compact/model.onnx', entries)
        self.assertIn('scripts/submission/model.py', entries)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            common.write_archive(tmp/'portable.zip', entries)
            common.extract_verified(tmp/'portable.zip', tmp/'work')
            result = subprocess.run([sys.executable, '-S', '-c',
                'import sys; from scripts.submission.benchmark import runtime_entries; '
                'e=runtime_entries(); assert "pcvcm/decoder.py" in e; '
                'assert "torch" not in sys.modules; print(len(e))'],
                cwd=tmp/'work', text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_ssh_argument_roundtrip_preserves_python_code(self):
        args = type('Args', (), {'ssh_port': 22, 'identity': Path('/tmp/key with spaces'), 'host': 'pi@pi.local'})()
        code = "print('hello'); x = 'a b'"
        command = benchmark.ssh_command(args, code, 'x; echo unsafe', 3)
        self.assertEqual(command[-2], 'pi@pi.local')
        self.assertEqual(shlex.split(command[-1]), ['python3', '-c', code, 'x; echo unsafe', '3'])

    def test_local_acquisition_uses_fresh_inference_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'result'
            samples = Path(tmp)/'audio'
            with patch.object(benchmark.subprocess, 'run') as run, patch.object(benchmark, 'worker',
                    side_effect=AssertionError('Parquet-acquisition process must not measure inference')):
                benchmark.main(['--target','pc','--no-setup','--samples',str(samples),
                                '--output',str(output),'--repeats','1'])
            command = run.call_args.args[0]
            self.assertIn('--worker', command)
            self.assertIn('--no-setup', command)
            self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')

    def test_safe_archive_names_reject_windows_and_dot_paths(self):
        for name in ('../escape', '/absolute', 'C:/drive', 'a\\b', '.', 'a//b', './a'):
            with self.subTest(name=name):
                self.assertFalse(common.safe_name(name))
        self.assertTrue(common.safe_name('.cache/submission/holdout/0000.wav'))

    def remote_fixture(self, root):
        source = root/'submission/benchmark/holdout.json'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'rows':[{'audio':'0000.wav'}]}))
        samples = root/'audio'; samples.mkdir(); (samples/'0000.wav').write_bytes(b'original bytes')
        args = SimpleNamespace(output=root/'local-results',model_choice='full',ssh_port=22,
                               identity=None,host='pi@test.local',threads=1,repeats=1,limit=None)
        return args,samples

    def test_simulated_ssh_roundtrip_verifies_and_retrieves_results(self):
        # Execute the exact transmitted Python against an isolated local HOME;
        # this checks the protocol, not SSH authentication or Raspberry Pi behavior.
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); args,samples=self.remote_fixture(root)
            home=root/'remote-home'; home.mkdir()
            runner = ("import json,pathlib\n"
                      "p=pathlib.Path('results');p.mkdir()\n"
                      "(p/'metrics.json').write_text(json.dumps({'actions_enabled':False,'simulated':True}))\n"
                      "[(p/n).write_text('') for n in ('telemetry.json','trials.jsonl','timing.jsonl','REPORT.md')]\n")
            calls=[]
            def transport(command, **kwargs):
                calls.append(command)
                parts=shlex.split(command[-1])
                env=os.environ.copy();env['HOME']=str(home)
                return real_run([sys.executable,*parts[1:]],env=env,**kwargs)
            with patch.object(benchmark,'ROOT',root),patch.object(benchmark,'run_id',return_value='test-roundtrip'),\
                 patch.object(benchmark,'runtime_entries',return_value={'benchmark.py':runner.encode()}),\
                 patch.object(benchmark.subprocess,'run',side_effect=transport):
                result=benchmark.remote(args,samples)
            self.assertTrue(result['simulated'])
            self.assertFalse(result['actions_enabled'])
            self.assertEqual(len(calls),3)
            receipt=json.loads((args.output/'ssh-receipt.json').read_text())
            self.assertEqual(receipt['host'],args.host)
            self.assertEqual((home/'.cache/smart-speaker-submission/test-roundtrip/work/.cache/submission/holdout/0000.wav').read_bytes(),b'original bytes')

    def test_simulated_ssh_rejects_changed_transfer_before_code_or_results(self):
        real_run = subprocess.run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args,samples=self.remote_fixture(root)
            home=root/'remote-home';home.mkdir();calls=[]
            def transport(command,**kwargs):
                calls.append(command)
                parts=shlex.split(command[-1]);env=os.environ.copy();env['HOME']=str(home)
                result=real_run([sys.executable,*parts[1:]],env=env,**kwargs)
                if len(calls)==1:
                    (home/'.cache/smart-speaker-submission/test-corrupt/payload.zip').write_bytes(b'changed transfer')
                return result
            with patch.object(benchmark,'ROOT',root),patch.object(benchmark,'run_id',return_value='test-corrupt'),\
                 patch.object(benchmark,'runtime_entries',return_value={'benchmark.py':b'raise AssertionError("must not execute")'}),\
                 patch.object(benchmark.subprocess,'run',side_effect=transport):
                with self.assertRaises(subprocess.CalledProcessError): benchmark.remote(args,samples)
            self.assertEqual(len(calls),2)
            self.assertFalse(args.output.exists())
            self.assertFalse((home/'.cache/smart-speaker-submission/test-corrupt/work').exists())


if __name__ == '__main__':
    unittest.main()
