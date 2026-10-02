"""Release setup/update checks; no microphone, training or installed-model writes."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources
register_sources()
from tinyvcm import launcher


class ReleaseSetupChecks(unittest.TestCase):
    def test_repeat_setup_preserves_environment_and_syncs_locked_release(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            python.parent.mkdir(parents=True)
            python.write_bytes(b'existing interpreter')
            (root / 'pyproject.toml').write_text('[project]\nname="fixture"\n')
            (root / 'uv.lock').write_text('version=1\n')
            (root / 'pipeline.default.toml').write_text('new defaults')
            (root / 'pipeline.toml').write_text('my settings')
            with patch.object(launcher, 'find_uv', return_value='uv', create=True), \
                    patch.object(launcher.subprocess, 'run') as run, \
                    patch.dict(os.environ, {'UV_PROJECT_ENVIRONMENT': '/unrelated', 'VIRTUAL_ENV': '/research'}):
                launcher.setup(root, [])
                launcher.setup(root, [])
            self.assertEqual(python.read_bytes(), b'existing interpreter')
            self.assertEqual((root / 'pipeline.toml').read_text(), 'my settings')
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[:2], ['uv', 'sync'])
                self.assertIn('--locked', command)
                self.assertIn('--no-build', command)
                self.assertNotIn('--upgrade', command)
                self.assertEqual(call.kwargs['cwd'], root)
                self.assertEqual(call.kwargs['env']['UV_PROJECT_ENVIRONMENT'], str(root / '.venv'))
                self.assertNotIn('VIRTUAL_ENV', call.kwargs['env'])

    def test_missing_lock_fails_before_installing_anything(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            with patch.object(launcher.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'uv.lock'):
                    launcher.setup(Path(temporary), [])
            run.assert_not_called()

    def test_sync_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            for name in ['pyproject.toml', 'uv.lock']:
                (root / name).write_text('fixture')
            with patch.object(launcher, 'find_uv', return_value='uv', create=True), \
                    patch.object(launcher.subprocess, 'run', side_effect=launcher.subprocess.CalledProcessError(1, ['uv'])):
                with self.assertRaises(launcher.subprocess.CalledProcessError):
                    launcher.setup(root, [])

    def test_first_setup_creates_editable_config_from_release_defaults(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            for name in ['pyproject.toml', 'uv.lock']:
                (root / name).write_text('fixture')
            (root / 'pipeline.default.toml').write_text('release defaults')
            with patch.object(launcher, 'find_uv', return_value='uv'), patch.object(launcher.subprocess, 'run'):
                launcher.setup(root, [])
            self.assertEqual((root / 'pipeline.toml').read_text(), 'release defaults')


if __name__ == '__main__':
    unittest.main(verbosity=2)
