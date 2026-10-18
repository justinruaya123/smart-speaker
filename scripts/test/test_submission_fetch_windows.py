"""Download failures must retain their cause and remove Windows tempfiles."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.submission.common import fetch


class FetchFailureTests(unittest.TestCase):
    def test_network_failure_preserves_error_and_cleans_tempfile(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'payload.wav'
            with patch('urllib.request.urlopen', side_effect=OSError('network-original')):
                with self.assertRaisesRegex(OSError, 'network-original'):
                    fetch('https://example.invalid/file', destination)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
