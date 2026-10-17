"""Benchmark source fixtures are complete, immutable and usable without downloads."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.submission.build_benchmark import audio_entries


class BenchmarkPackageTests(unittest.TestCase):
    def test_original_wav_bytes_are_packaged_at_the_offline_runtime_paths(self):
        payload=b'original fixture WAV';spec={'rows':[{'audio':'0000.wav','sha256':hashlib.sha256(payload).hexdigest()}]}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'audio.zip'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('0000.wav',payload)
            self.assertEqual(audio_entries(path,spec),{'.cache/submission/holdout/0000.wav':payload})

    def test_changed_missing_extra_and_unsafe_samples_are_rejected(self):
        spec={'rows':[{'audio':'0000.wav','sha256':hashlib.sha256(b'original').hexdigest()}]}
        for files in ({'0000.wav':b'changed'},{},{'0000.wav':b'original','private.wav':b'private'},{'../0000.wav':b'original'}):
            with self.subTest(files=list(files)),tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'audio.zip'
                with zipfile.ZipFile(path,'w') as archive:
                    for name,data in files.items():archive.writestr(name,data)
                with self.assertRaises(ValueError):audio_entries(path,spec)


if __name__=='__main__':unittest.main()
