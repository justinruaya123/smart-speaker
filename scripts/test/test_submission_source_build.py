"""Source builds require checked-in, hash-bound assets and never fetch model/audio payloads."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.submission.build_inference import verify_local_responses


class SourceBuildTests(unittest.TestCase):
    def test_missing_source_clips_fail_even_if_a_download_url_exists(self):
        from scripts.submission.common import ROOT
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'submission/responses';folder.mkdir(parents=True)
            manifest=json.loads((ROOT/'submission/responses/manifest.json').read_text())
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(FileNotFoundError,'source audio'):
                verify_local_responses(root)

    def test_source_clips_match_the_frozen_manifest(self):
        from scripts.submission.common import ROOT
        result=verify_local_responses(ROOT)
        self.assertEqual(len(result),len(json.loads((ROOT/'submission/responses/manifest.json').read_text())['responses']))


if __name__=='__main__':unittest.main()
