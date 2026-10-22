"""Human-first data safety checks; generated fixtures are not speech evidence."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest


def row(identity='positive', *, split='train', expected=None, metric=True,
        eligible=True, synthetic=False, speaker='speaker-a') -> dict:
    return dict(id=identity, split=split, expected=expected,
                metric_eligible=metric, training_eligible=eligible,
                synthetic=synthetic, kind='synthetic' if synthetic else 'human',
                speaker=speaker, sha256=identity, pcm_sha256=identity,
                phones=[1, 2], intent_index=1, target_index=-100,
                reference_reason='finite_definition')


class HumanPartitionTests(unittest.TestCase):
    def test_separates_real_negatives_ambiguity_and_synthetic(self):
        spec = importlib.util.find_spec('scripts.train.speaker_human_data')
        self.assertIsNotNone(spec, 'Human-only partitioner is not implemented')
        from scripts.train.speaker_human_data import partition_human_rows
        records = [
            row(expected={'intent': 'STOP', 'slots': {}}),
            row('unknown'),
            row('ambiguous', split='test', metric=False, eligible=False),
            row('tts', synthetic=True),
            row('ineligible', eligible=False),
        ]
        views, excluded = partition_human_rows(records)
        self.assertEqual([r['id'] for r in views['train']['STOP']], ['positive'])
        self.assertEqual([r['id'] for r in views['train']['UNKNOWN']], ['unknown'])
        self.assertEqual(views['test']['UNKNOWN'], [])
        self.assertEqual({r['id']: r['exclusion'] for r in excluded}, {
            'ambiguous': 'ambiguous_reference', 'tts': 'not_original_human',
            'ineligible': 'ineligible_training_row'})
        self.assertEqual(len(views['train']), 19)
        self.assertEqual(len(views), 5)


class HumanTrainingSafetyTests(unittest.TestCase):
    def test_full_readiness_cannot_ignore_extension_or_unknown_gaps(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'training_ready'), 'Full-catalog readiness guard is missing')
        coverage = {label: {'pilot_speaker_floors_met': True, 'catalog_collection_queue': []}
                    for label in data.LABELS}
        self.assertTrue(data.training_ready(coverage))
        coverage['VOLUME']['catalog_collection_queue'] = [{'operation': 'mute'}]
        self.assertFalse(data.training_ready(coverage))
        coverage['VOLUME']['catalog_collection_queue'] = []
        self.assertFalse(data.training_ready(coverage, overlaps=[{'speaker': 'same'}]))
        coverage.pop('UNKNOWN')
        self.assertFalse(data.training_ready(coverage))

    def test_target_index_convention_is_consistent_for_all_four_light_intents(self):
        from scripts.train.speaker_human_data import human_supervision
        for intent in ('LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'):
            for light_id, target_index in (('light.all', 0), ('light.kitchen', 1)):
                slots: dict[str, object] = {'light_id': light_id}
                if intent == 'BRIGHTNESS':
                    slots['percent'] = 20
                if intent == 'COLOR':
                    slots['color'] = 'red'
                example = row(expected={'intent': intent, 'slots': slots})
                example['target_index'] = 1 - target_index
                with self.subTest(intent=intent, light_id=light_id):
                    self.assertEqual(human_supervision(example)['target_index'], target_index)
                    self.assertEqual(example['target_index'], 1 - target_index)

    def test_cli_exposes_readiness_gate_and_explicit_plumbing_smoke(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'main'), 'Audit CLI is missing')
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as exit:
            data.main(['--help'])
        self.assertEqual(exit.exception.code, 0)
        self.assertIn('--require-ready', stream.getvalue())
        self.assertIn('--smoke-fit', stream.getvalue())

    def test_metrics_exclude_ambiguity_without_hiding_wrong_accepts_or_missing_intents(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'human_metrics'), 'Human complete-command metrics are missing')
        stop = {'intent': 'STOP', 'slots': {}}
        records = [row('correct', expected=stop), row('unknown'),
                   row('ambiguous', metric=False), row('rejected', expected=stop)]
        for record in records:
            record.update(candidate=stop, application_command=stop, application_reason=None)
        records[-1].update(application_command=None, application_reason='intent_confidence')
        result = data.human_metrics(records)
        self.assertEqual(result['positive_n'], 2)
        self.assertEqual(result['negative_n'], 1)
        self.assertEqual(result['ambiguous_n'], 1)
        self.assertEqual(result['accepted_correct'], 1)
        self.assertEqual(result['unknown_false_accepts'], 1)
        self.assertEqual(result['accepted_precision'], .5)
        self.assertEqual(result['per_intent']['MESSAGE']['positive_n'], 0)
        self.assertIsNone(result['per_intent']['MESSAGE']['accepted_recall'])
        self.assertEqual(result['per_intent']['STOP']['raw_correct'], 2)
        with self.assertRaises(ValueError):
            data.human_metrics([{**records[0], 'synthetic': True}])

    def test_normalization_is_refitted_on_exact_admitted_human_train(self):
        import numpy as np
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'fit_human_normalization'), 'Human normalization refit is missing')
        scratch = Path(__file__).resolve().parents[2] / '.cache/test-human-data'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            records = []
            for i in range(2):
                record = row(str(i), expected={'intent': 'STOP', 'slots': {}})
                path = Path(directory) / (str(i) + '.npy')
                np.save(path, np.full((40, 10 + i), i * 2, dtype='float32'), allow_pickle=False)
                record.update(features=str(path), features_sha256=data.file_sha256(path))
                records.append(record)
            result = data.fit_human_normalization(records)
            np.testing.assert_allclose(result['mean'], np.ones(40))
            np.testing.assert_allclose(result['std'], np.ones(40))
            self.assertEqual(result['source_ids'], ['0', '1'])
            for changes in ({'split': 'dev'}, {'synthetic': True}, {'training_eligible': False}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    data.fit_human_normalization([{**records[0], **changes}])

    def test_publisher_join_uses_full_pinned_location_not_short_id(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'join_publisher_metadata'), 'Pinned metadata join is missing')
        scratch = Path(__file__).resolve().parents[2] / '.cache/test-human-data'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            path = root / 'data/train-00000.parquet'
            path.parent.mkdir()
            pq.write_table(pa.Table.from_pylist([dict(transcript='Stop', source='fixture',
                speaker_id='fixture-a', file='stop.wav', is_synthetic=0, transcript_source='prompt',
                command='STOP', whisper_check='')]), path)
            record = row('repo/primary/train-00000/000000', expected={'intent': 'STOP', 'slots': {}})
            record['location'] = dict(repository='repo', revision='revision', file='data/train-00000.parquet',
                                      row=0, sha256=data.file_sha256(path))
            joined, receipt = data.join_publisher_metadata([record], root)
            self.assertEqual(joined[0]['text'], 'stop')
            self.assertEqual(joined[0]['transcript_source'], 'prompt')
            self.assertEqual(joined[0]['expected'], record['expected'])
            self.assertEqual(receipt['joined_rows'], 1)
            self.assertNotIn('text', record)
            wrong = {**record, 'location': {**record['location'], 'sha256': 'wrong'}}
            with self.assertRaisesRegex(ValueError, 'Publisher shard checksum'):
                data.join_publisher_metadata([wrong], root)

    def test_audit_writes_immutable_human_views_without_opening_test_audio(self):
        import numpy as np
        import soundfile as sf
        from scripts.train import speaker_human_data as data
        from scripts.test.speaker_demo_suite import file_digest, pcm_digest
        from vcm.frontend import features
        self.assertTrue(hasattr(data, 'audit_dataset'), 'Executable data audit is missing')
        scratch = Path(__file__).resolve().parents[2] / '.cache/test-human-data'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory)
            prepared = root / 'prepared'
            prepared.mkdir()
            positive = {'intent': 'STOP', 'slots': {}}
            for split in data.SPLITS:
                records = []
                for i in range(2 if split == 'train' else 1):
                    record = row(split + str(i), split=split, speaker=split + str(i),
                                 expected=positive if i == 0 else None, eligible=split == 'train')
                    if split in ('train', 'dev'):
                        pcm = np.random.default_rng(20 + len(split) + i).uniform(-.1, .1, 16000).astype('float32')
                        audio = root / (record['id'] + '.wav')
                        sf.write(audio, pcm, 16000, subtype='FLOAT')
                        path = root / (record['id'] + '.npy')
                        np.save(path, features(pcm), allow_pickle=False)
                        record.update(audio=str(audio), sha256=file_digest(audio), pcm_sha256=pcm_digest(pcm),
                                      features=str(path), features_sha256=file_digest(path))
                    else:
                        record['audio'] = str(root / 'must-not-open-evaluation-audio.wav')
                    records.append(record)
                (prepared / (split + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in records))
            output = root / 'audit'
            report = data.audit_dataset(prepared, output, baseline=[{'text': 'stop', 'command': positive}])
            self.assertEqual(report['audio_checked_splits'], ['train', 'dev'])
            self.assertEqual(report['audio_checked_rows'], 3)
            self.assertEqual(report['view_rows']['train'], 2)
            self.assertEqual(len(report['per_intent']), 19)
            self.assertFalse(report['ready_for_full_human_training'])
            self.assertTrue((output / 'views/calibration/MESSAGE.jsonl').exists())
            self.assertEqual((output / 'views/calibration/MESSAGE.jsonl').read_text(), '')
            self.assertTrue((output / 'audit.json').exists())
            self.assertTrue(hasattr(data, 'optimizer_smoke'), 'Human optimizer path is not exercised')
            smoke = data.optimizer_smoke(output, root / 'smoke',
                architecture=dict(width=8, layers=1, heads=2, feedforward=16, dropout=0.))
            self.assertTrue(smoke['weights_changed'])
            self.assertEqual(smoke['optimizer_updates'], 1)
            self.assertFalse(smoke['accuracy_gain_claimed'])
            self.assertEqual(smoke['normalization_human_rows'], 2)
            with self.assertRaises(FileExistsError):
                data.audit_dataset(prepared, output, baseline=[])
            bad = json.loads((prepared / 'train.jsonl').read_text().splitlines()[0])
            Path(bad['features']).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Feature checksum'):
                data.audit_dataset(prepared, root / 'tampered', baseline=[])

    def test_coverage_is_per_command_speaker_and_split_not_pooled(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'coverage_report'), 'Per-intent coverage is missing')
        low = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 20}}
        high = {'intent': 'BRIGHTNESS', 'slots': {'light_id': 'light.all', 'percent': 60}}
        records = [row('a', expected=low), row('b', expected=low), row('unknown')]
        report = data.coverage_report(records, [{'text': 'low', 'command': low},
                                               {'text': 'high', 'command': high}])
        self.assertEqual(set(report), set(data.LABELS))
        train = report['BRIGHTNESS']['splits']['train']
        self.assertEqual(train['rows'], 2)
        self.assertEqual(train['speakers'], 1)
        self.assertEqual(train['typed_commands'], 1)
        self.assertEqual(train['largest_speaker_fraction'], 1.)
        self.assertEqual(report['BRIGHTNESS']['baseline_commands']['train'], {'covered': 1, 'total': 2})
        self.assertTrue(any(x['command'] == high and x['split'] == 'train'
                            for x in report['BRIGHTNESS']['collection_queue']))
        self.assertEqual(report['MESSAGE']['splits']['train']['rows'], 0)
        self.assertFalse(report['MESSAGE']['ready_for_human_selection'])
        self.assertEqual(report['UNKNOWN']['splits']['train']['rows'], 1)
        catalog = [{'command': low}, {'command': high},
                   {'command': {'intent': 'VOLUME', 'slots': {'operation': 'mute'}}}]
        self.assertIn('catalog_commands', report['VOLUME'], 'Approved extensions must not disappear from coverage')
        expanded = data.coverage_report(records, [], catalog=catalog)
        self.assertEqual(expanded['VOLUME']['catalog_commands']['train'], {'covered': 0, 'total': 1})
        self.assertTrue(any(c['command']['slots'].get('operation') == 'mute'
                            for c in expanded['VOLUME']['catalog_collection_queue']))

    def test_boundary_diagnostics_preserve_short_utterance_and_internal_pause(self):
        import numpy as np
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'audio_diagnostics'), 'Acoustic audit is missing')
        pcm = np.zeros(16000, dtype=np.float32)
        pcm[3200:4800] = .1
        pcm[12000:13600] = .1
        before = pcm.copy()
        result = data.audio_diagnostics(pcm, 16000)
        self.assertAlmostEqual(result['duration_seconds'], 1.)
        self.assertGreaterEqual(result['longest_internal_low_energy_seconds'], .4)
        start, end = result['proposed_segment_samples']
        self.assertLess(start, 3200)
        self.assertGreater(end, 13600)
        np.testing.assert_array_equal(pcm, before)
        silent = data.audio_diagnostics(np.zeros(400, dtype=np.float32), 16000)
        self.assertIn('no_detectable_energy', silent['flags'])
        self.assertIsNone(silent['proposed_segment_samples'])
        short = data.audio_diagnostics(np.ones(800, dtype=np.float32) * .1, 16000)
        self.assertEqual(short['proposed_segment_samples'], [0, 800])
        for bad in (np.array([float('nan')]), np.ones((2, 400)), np.array([1.1])):
            with self.assertRaises(ValueError):
                data.audio_diagnostics(bad, 16000)

    def test_leakage_checks_cover_every_split_and_parent_identity(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'split_leakage'), 'Cross-split checks are missing')
        a = row('a', split='dev', speaker='same-speaker', eligible=False)
        b = row('b', split='calibration', speaker='same-speaker', eligible=False)
        c = row('c', split='train', speaker='other')
        a['parent_recording_id'] = c['parent_recording_id'] = 'one-recording'
        issues = data.split_leakage([a, b, c])
        self.assertTrue(any(x['field'] == 'speaker' and x['splits'] == ['calibration', 'dev'] for x in issues))
        self.assertTrue(any(x['field'] == 'parent_recording_id' for x in issues))
        b['speaker'] = 'different'
        a.pop('parent_recording_id')
        self.assertEqual(data.split_leakage([a, b, c]), [])

    def test_training_admission_rebuilds_indices_and_refuses_unsafe_rows(self):
        from scripts.train import speaker_human_data as data
        self.assertTrue(hasattr(data, 'HumanTrainingPools'), 'Safe human sampler is missing')
        positive = row(expected={'intent': 'LIGHT_ON', 'slots': {'light_id': 'light.kitchen'}})
        positive['intent_index'] = 0
        positive['target_index'] = 0
        negative = row('negative')
        pools = data.HumanTrainingPools([positive, negative])
        batch = pools.sample_batch(seed=53, size=4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(sum(r['expected'] is None for r in batch), 1)
        corrected = next(r for r in batch if r['expected'])
        self.assertEqual(corrected['intent_index'], data.INTENTS.index('LIGHT_ON') + 1)
        self.assertEqual(corrected['target_index'], 1)
        self.assertEqual(positive['target_index'], 0, 'Do not mutate the frozen input')
        self.assertEqual(batch, pools.sample_batch(seed=53, size=4))
        for changes in ({'split': 'dev'}, {'synthetic': True}, {'metric_eligible': False},
                        {'speaker': 'unknown'}, {'kind': 'constructed'}, {'phones': []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                data.HumanTrainingPools([{**positive, **changes}, negative])


if __name__ == '__main__':
    unittest.main()
