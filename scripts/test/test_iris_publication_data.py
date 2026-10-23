"""Publication checks run with unittest/stdlib, without training or providers."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / 'scripts' / 'iris_publication_data.py'


def module():
    spec = importlib.util.spec_from_file_location('iris_publication_data', GENERATOR)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class PublicationDataTests(unittest.TestCase):
    def test_verified_snapshot_becomes_complete_finite_publication(self):
        self.assertTrue(GENERATOR.is_file(), 'publication generator must exist')
        data = module().build_publication(ROOT)
        self.assertEqual(data['schema_version'], 1)
        self.assertEqual(data['model']['parameters'], 258912)
        self.assertEqual(data['model']['bundle_bytes'], 1088682)
        self.assertEqual(data['benchmarks']['reference_desktop']['label'], '2026-10-24 reference driver / under publication load')
        self.assertTrue(data['benchmarks']['reference_desktop']['measurement_not_quiesced'])
        self.assertEqual(data['model']['selected_epoch'], 44)
        self.assertEqual(data['model']['command_count'], 31)
        self.assertEqual(data['model']['definition_count'], 93)
        self.assertEqual(data['model']['intent_count'], 18)
        self.assertEqual(len(data['use_cases']), 10)
        assigned = [i for case in data['use_cases'] for i in case['class_indices']]
        self.assertEqual(sorted(assigned), list(range(1, 32)))
        self.assertEqual(data['datasets']['splits']['train'], {'human': 1889, 'synthetic': 21426})
        self.assertEqual(data['datasets']['normalization_human_train'], 1889)
        self.assertIsNone(data['benchmarks']['pi'])
        self.assertIsNotNone(data['benchmarks']['desktop'], 'attach actual current product receipt')
        self.assertTrue(data['benchmarks']['desktop']['independent_parity']['full_typed_outputs_match_frozen_gold'])
        self.assertEqual(data['benchmarks']['desktop']['original_recordings'], 1146)
        self.assertEqual(data['benchmarks']['desktop']['timed_calls'], 3438)
        self.assertFalse(data['model']['qualified'])
        commands = [r['command'] for r in data['commands']]
        for case in data['use_cases']:
            self.assertTrue(case['diagram'])
            self.assertTrue(case['output']['example_only'])
            for example in case['examples']:
                self.assertIn(example['command'], commands)
        json.dumps(data, allow_nan=False)

    def test_unknown_measurement_target_is_rejected(self):
        candidate = {'schema_version': 1, 'format': 'iris-publication-measurement-v1', 'target': 'phone',
                     'hashes': module().MODEL_HASHES, 'hardware': {'cpu': 'fixture', 'platform': 'fixture', 'python': 'fixture'},
                     'timing_scope': 'recognize_pcm', 'timing_excludes': ['fixture'],
                     'latency': {'p50_ms': 1, 'p95_ms': 2}, 'original_recordings': 1, 'timed_calls': 3,
                     'repeats': 3, 'receipt_path': 'fixture.json', 'measured_at': '2026-10-24T00:00:00Z'}
        with self.assertRaisesRegex(ValueError, 'unsupported measurement target'):
            module().validate_measurement(candidate, 'phone')

    def test_measurement_identity_counts_scope_and_pi_hardware_fail_closed(self):
        from copy import deepcopy
        implementation = module()
        fixture = {'schema_version': 1, 'format': 'iris-publication-measurement-v1', 'target': 'pi',
                   'hashes': dict(implementation.MODEL_HASHES),
                   'hardware': {'device': 'Raspberry Pi fixture only', 'cpu': 'fixture', 'platform': 'fixture', 'python': 'fixture'},
                   'timing_scope': 'recognize_pcm', 'timing_excludes': ['fixture'],
                   'latency': {'p50_ms': 1, 'p95_ms': 2}, 'original_recordings': 1, 'timed_calls': 3,
                   'repeats': 3, 'receipt_path': 'fixture-only.json', 'measured_at': '2026-10-24T00:00:00Z'}
        # Schema fixture only: these values are never exported as a measurement.
        self.assertEqual(implementation.validate_measurement(fixture, 'pi'), fixture)
        variants = []
        for name in ('model.onnx', 'policy.json'):
            value = deepcopy(fixture); value['hashes'][name] = '0' * 64; variants.append(value)
        value = deepcopy(fixture); value['hardware']['device'] = 'desktop'; variants.append(value)
        value = deepcopy(fixture); value['timing_scope'] = 'graph_only'; variants.append(value)
        value = deepcopy(fixture); value['timed_calls'] = 4; variants.append(value)
        for latency in (float('nan'), float('inf'), -1, 0, True):
            value = deepcopy(fixture); value['latency']['p50_ms'] = latency; variants.append(value)
        value = deepcopy(fixture); value['latency']['p95_ms'] = .5; variants.append(value)
        value = deepcopy(fixture); del value['measured_at']; variants.append(value)
        for index, value in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ValueError):
                implementation.validate_measurement(value, 'pi')

    def test_changed_payload_is_rejected_without_mutating_frozen_assets(self):
        import hashlib
        import os
        import tempfile
        implementation = module()
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as folder:
            path = Path(folder) / 'fixture.json'
            path.write_bytes(b'changed fixture, not a model')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                implementation.checked(path, hashlib.sha256(b'original fixture').hexdigest())

    def test_failed_wake_study_is_not_called_a_release_pass(self):
        data = module().build_publication(ROOT)
        self.assertTrue('wake' in data, 'actual failed wake study must be published')
        self.assertTrue(data['wake']['verdict']['release_blocker'])
        self.assertEqual(data['wake']['baseline']['detected'], 1)
        self.assertEqual(data['wake']['companion_test']['detected'], 21)
        self.assertEqual(data['wake']['companion_test']['false_wakes'], 2)
        self.assertFalse(data['wake']['verdict']['publish_companion'])


class AnimationAssetTests(unittest.TestCase):
    def test_real_render_and_slide_export_have_local_assets(self):
        for name in ('pipeline', 'model'):
            folder = ROOT / 'docs/site/animations'
            self.assertTrue((folder / f'{name}.mp4').is_file(), 'actual Manim MP4 is required')
            self.assertGreater((folder / f'{name}.mp4').stat().st_size, 10000)
            self.assertTrue((folder / f'{name}-poster.png').is_file())
            html = (folder / f'{name}.html').read_text(encoding='utf-8')
            self.assertIn('Reveal', html)
            self.assertTrue('<html lang="en">' in html, 'export must declare its language')
            self.assertNotIn('user-scalable=no', html)
            self.assertNotIn('href="model_assets' + chr(92), html)
            self.assertNotIn('href="pipeline_assets' + chr(92), html)
            self.assertIn('data-background-video', html)
            self.assertNotIn('src="https://', html)
            self.assertNotIn('href="https://', html)


class StaticPublicationTests(unittest.TestCase):
    def test_pages_is_main_only_and_separate_acoustic_gate_is_required(self):
        workflow = ROOT / '.github/workflows/iris-publication.yml'
        self.assertTrue(workflow.is_file(), 'main-only Pages workflow is required')
        body = workflow.read_text(encoding='utf-8')
        self.assertIn("github.event_name == 'push' && github.ref == 'refs/heads/main'", body)
        self.assertIn('needs: [validate, acoustic-release]', body)
        self.assertIn('path: docs/site', body)
        self.assertIn('presentation/validate.py --release-gate', body)
        self.assertNotIn('pull_request_target', body)
        import re
        refs = re.findall(r'uses: (\S+)', body)
        self.assertTrue(refs)
        self.assertTrue(all(re.fullmatch(r'actions/[a-z-]+@[0-9a-f]{40}', ref) for ref in refs))

    def test_acoustic_release_gate_fails_for_preserved_wake_study(self):
        spec = importlib.util.spec_from_file_location('iris_site_validate', ROOT / 'presentation/validate.py')
        assert spec is not None and spec.loader is not None
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        self.assertTrue(hasattr(value, 'release_gate'), 'static QA must not substitute for the acoustic gate')
        with self.assertRaisesRegex(ValueError, 'wake acceptance gate failed'):
            value.release_gate(ROOT)

    def test_static_site_and_readme_are_validated_without_scientific_dependencies(self):
        validator = ROOT / 'presentation/validate.py'
        self.assertTrue(validator.is_file(), 'stdlib asset/link validator is required')
        spec = importlib.util.spec_from_file_location('iris_site_validate', validator)
        assert spec is not None and spec.loader is not None
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        result = value.validate_site(ROOT)
        self.assertEqual(result['chapters'], 6)
        self.assertEqual(result['use_cases'], 10)
        self.assertEqual(result['readme_diagrams'], 12)
        self.assertGreater(result['assets_checked'], 20)
        self.assertEqual(result['missing_assets'], [])


if __name__ == '__main__':
    unittest.main()
