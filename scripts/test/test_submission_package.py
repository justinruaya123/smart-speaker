"""Inference release must fail closed on training payloads or changed model bytes."""
import json
import unittest
from pathlib import Path
import tempfile

from scripts.submission import package
from scripts.submission.common import runtime_entries


class SubmissionPackageTests(unittest.TestCase):
    def test_training_payload_is_rejected(self):
        entries = runtime_entries()
        entries['submission/checkpoints/private.pt'] = b'not an inference artifact'
        with self.assertRaisesRegex(ValueError, 'inference-only'):
            package.validate_entries(entries)

    def test_changed_graph_is_rejected(self):
        entries = runtime_entries()
        entries[package.MODEL_PREFIX+'model.onnx'] += b'x'
        with self.assertRaisesRegex(ValueError, 'model identity'):
            package.validate_entries(entries)

    def test_minimal_project_contains_no_training_group(self):
        entries = runtime_entries()
        entries['pyproject.toml'] = entries['pyproject.toml'].replace(b'[dependency-groups]',b'[dependency-groups]\ntraining = ["torch"]')
        with self.assertRaisesRegex(ValueError, 'training dependency'):
            package.validate_entries(entries)

    def test_runtime_inventory_is_complete_and_manifest_bound(self):
        entries = runtime_entries()
        inventory = package.validate_entries(entries)
        self.assertEqual(len(inventory),len(entries))
        self.assertEqual(inventory[package.MODEL_PREFIX+'model.onnx'],
                         json.loads(entries[package.MODEL_PREFIX+'recipe.json'])['binding']['model_sha256'])

    def test_public_configuration_example_does_not_allow_private_env_files(self):
        entries = runtime_entries()
        entries['SpeakerUI/.env.example'] = b'HA_TOKEN=\n'
        package.validate_entries(entries)
        for name in ('.env','SpeakerUI/.env.production'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError,'inference-only'):
                package.validate_entries({**entries,name:b'HA_TOKEN=private-example'})

    def test_production_tree_keeps_notices_and_omits_dependency_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('index.js','LICENSE','test.js','fixtures/private.wav','example/index.html'):
                path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fixture')
            entries = {}
            package._tree(entries,root,'SpeakerUI/node_modules/example-package')
            self.assertEqual(set(entries),{'SpeakerUI/node_modules/example-package/index.js',
                                           'SpeakerUI/node_modules/example-package/LICENSE'})


if __name__ == '__main__':
    unittest.main()
