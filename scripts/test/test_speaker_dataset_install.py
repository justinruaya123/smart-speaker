"""Dataset preparation must protect held-out evidence and original bytes."""
import importlib
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class DatasetInstallationTests(unittest.TestCase):
    def module(self):
        name = 'scripts.train.speaker_dataset_install'
        self.assertIsNotNone(importlib.util.find_spec(name),
                             'Dataset installer implementation is missing')
        return importlib.import_module(name)

    def row(self, identity='a', **extra):
        row = dict(id=identity, split='train', kind='human',
                    speaker_id=identity, pair_id=identity,
                    pcm_sha256=identity, sha256=identity,
                    reference_status='supported', audio_flags=[],
                    qa_eligible=True,
                    command={'intent': 'STOP', 'slots': {}})
        return {**row, **extra}

    def test_failed_gap_qa_requires_explicit_rescue(self):
        m = self.module()
        self.assertFalse(m.qa_eligible('gap_fill',
                                      {'qa_passed': False, 'qa_rescued': False}))
        self.assertFalse(m.qa_eligible('gap_fill', {}))
        self.assertTrue(m.qa_eligible('gap_fill',
                                     {'qa_passed': 'False', 'qa_rescued': 'True'}))
        self.assertTrue(m.qa_eligible('default', {}))

    def test_source_identity_includes_config_directory(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'validate_sources'))
        files = [dict(path='.cache/source/' + config + '/train.parquet',
                      bytes=10, sha256=config) for config in ('gap', 'numerals')]
        records = m.validate_sources({'files': files}, {'files': files})
        self.assertEqual(len(records), 2)
        with self.assertRaisesRegex(ValueError, 'identity'):
            m.validate_sources({'files': [{**files[0], 'sha256': 'changed'},
                                         files[1]]}, {'files': files})

    def test_synthetic_supplement_is_train_only(self):
        m = self.module()
        with self.assertRaisesRegex(ValueError, 'train only'):
            m.build_views([self.row(kind='synthetic', split='test',
                                   supplemental=True)])

    def test_heldout_pcm_is_never_used_for_fitting(self):
        m = self.module()
        train = self.row('train')
        test = self.row('test', split='test')
        train['pcm_sha256'] = test['pcm_sha256']
        views, audit = m.build_views([train, test])
        self.assertEqual(views['human/train'], [])
        self.assertEqual(views['evaluation/test'], [test])
        self.assertEqual(audit['excluded']['heldout_pcm_overlap'], 1)

    def test_heldout_speaker_and_pair_are_excluded_from_fit(self):
        m = self.module()
        train = self.row('train')
        dev = self.row('dev', split='dev')
        holdout = self.row('holdout', split='holdout')
        train['speaker_id'] = holdout['speaker_id']
        dev['pair_id'] = holdout['pair_id']
        views, _ = m.build_views([train, dev, holdout])
        self.assertEqual(views['human/train'], [])
        self.assertEqual(views['human/dev'], [])
        self.assertEqual(len(views['evaluation/holdout']), 1)

    def test_conflicting_pcm_supervision_is_quarantined(self):
        m = self.module()
        a, b = self.row('a'), self.row('b')
        b['pcm_sha256'] = a['pcm_sha256']
        b['command'] = {'intent': 'PAUSE', 'slots': {}}
        views, audit = m.build_views([a, b])
        self.assertEqual(views['human/train'], [])
        self.assertEqual(audit['excluded']['conflicting_pcm_labels'], 2)

    def test_duplicate_fitting_audio_is_not_counted_twice(self):
        m = self.module()
        a, b = self.row('a'), self.row('b')
        b['pcm_sha256'] = a['pcm_sha256']
        views, audit = m.build_views([a, b])
        self.assertEqual(len(views['human/train']), 1)
        self.assertEqual(audit['excluded']['duplicate_pcm'], 1)

    def test_ambiguous_and_bad_qa_do_not_become_unknown(self):
        m = self.module()
        rows = [self.row('a', reference_status='ambiguous'),
                self.row('b', kind='synthetic', qa_eligible=False)]
        views, _ = m.build_views(rows)
        self.assertEqual(views['human/train'], [])
        self.assertEqual(views['synthetic/train'], [])

    def test_audio_reuse_refuses_changed_original_bytes(self):
        m = self.module()
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'audio.wav'
            m.write_audio(target, b'original')
            m.write_audio(target, b'original')
            with self.assertRaisesRegex(ValueError, 'existing audio'):
                m.write_audio(target, b'changed')
            self.assertEqual(target.read_bytes(), b'original')

    def test_concurrent_audio_writes_publish_only_complete_bytes(self):
        from concurrent.futures import ThreadPoolExecutor
        m = self.module()
        payload = b'recording' * (512 * 1024)
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / 'audio.wav'
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(m.write_audio, target, payload)
                           for _ in range(8)]
                errors = [str(f.exception()) for f in futures if f.exception()]
            self.assertEqual(errors, [])
            self.assertEqual(target.read_bytes(), payload)

    def test_source_receipt_cannot_silently_drop_a_shard(self):
        import hashlib
        import json
        m = self.module()
        self.assertTrue(hasattr(m, 'read_bound_json'))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'
            data = json.dumps({'files': ['train', 'test']}).encode()
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertEqual(len(m.read_bound_json(path, digest)['files']), 2)
            path.write_text(json.dumps({'files': ['train']}))
            with self.assertRaisesRegex(ValueError, 'source receipt'):
                m.read_bound_json(path, digest)

    def test_historical_crlf_receipt_recovery_is_hash_checked(self):
        import hashlib
        m = self.module()
        self.assertTrue(hasattr(m, 'verified_review'))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.json'
            original = b'{\r\n  "approved": true\r\n}\r\n'
            expected = hashlib.sha256(original).hexdigest()
            path.write_bytes(original.replace(b'\r\n', b'\n'))
            review, receipt = m.verified_review(path, expected)
            self.assertEqual(review, {'approved': True})
            self.assertEqual(receipt['verified_original_sha256'], expected)
            self.assertNotEqual(receipt['disk_sha256'], expected)
            path.write_bytes(b'{"approved": false}\n')
            with self.assertRaisesRegex(ValueError, 'Frozen annotation'):
                m.verified_review(path, expected)


if __name__ == '__main__':
    unittest.main()
