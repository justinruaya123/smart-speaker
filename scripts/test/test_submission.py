"""Submission interfaces: preserve evidence and distinguish replay from devices."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class SubmissionTests(unittest.TestCase):
    def module(self, name):
        spec = importlib.util.find_spec('scripts.submission.' + name) if (ROOT/'scripts/submission').is_dir() else None
        self.assertIsNotNone(spec, 'Submission helper must exist: ' + name)
        return __import__('scripts.submission.' + name, fromlist=[name])

    def test_ssh_target_rejects_shell_options_and_injection(self):
        module = self.module('benchmark')
        for value in ('-oProxyCommand=evil', 'pi@host;touch x', 'pi@host$(id)', 'pi@host\nwhoami', 'a b', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.validate_host(value)
        self.assertEqual(module.validate_host('pi@raspberrypi.local'), 'pi@raspberrypi.local')

    def test_archive_does_not_write_traversal_or_unlisted_file(self):
        module = self.module('common')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('../escape', b'bad')
                archive.writestr('SUBMISSION-MANIFEST.json', json.dumps({'files': {}}))
            with self.assertRaises(ValueError):
                module.extract_verified(path, Path(temp)/'extract')
            self.assertFalse((Path(temp)/'escape').exists())

    def test_corrupt_archive_is_rejected_before_extraction(self):
        module = self.module('common')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'bad.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('model.onnx', b'wrong')
                archive.writestr('SUBMISSION-MANIFEST.json', json.dumps({'files': {'model.onnx': '0'*64}}))
            with self.assertRaises(ValueError):
                module.extract_verified(path, Path(temp)/'extract')
            self.assertFalse((Path(temp)/'extract/model.onnx').exists())

    def test_benchmark_output_cannot_overwrite_prior_evidence(self):
        module = self.module('benchmark')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'; path.mkdir(); (path/'metrics.json').write_text('prior')
            with self.assertRaises(FileExistsError):
                module.new_output(path)
            self.assertEqual((path/'metrics.json').read_text(), 'prior')

    def test_bootstrap_hashing_works_before_managed_python_is_installed(self):
        module = self.module('common')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'file'; path.write_bytes(b'abc')
            with mock.patch.object(module.hashlib, 'file_digest', None):
                self.assertEqual(module.sha(path), 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_arm_machine_name_alone_is_not_pi_evidence(self):
        module = self.module('benchmark')
        self.assertFalse(module.is_raspberry_pi('aarch64', 'Generic ARM server'))
        self.assertTrue(module.is_raspberry_pi('aarch64', 'Raspberry Pi 4 Model B Rev 1.5'))

    def test_three_entry_points_help_without_third_party_imports(self):
        for name in ('reproduce.py', 'inference.py', 'benchmark.py'):
            with self.subTest(name=name):
                result = subprocess.run([sys.executable, '-S', str(ROOT/name), '--help'], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
