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
        self.assertIsNone(data['benchmarks']['desktop'])
        self.assertFalse(data['model']['qualified'])
        commands = [r['command'] for r in data['commands']]
        for case in data['use_cases']:
            self.assertTrue(case['diagram'])
            self.assertTrue(case['output']['example_only'])
            for example in case['examples']:
                self.assertIn(example['command'], commands)
        json.dumps(data, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
