"""Small behavioral checks for E18 data isolation and iteration publishing."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources, digest, save_json
register_sources()
import publish
from tinyvcm.prototype_data import revise, sample_epoch, verify
from tinyvcm.family_contract import annotate


class PrototypeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / 'artifacts/bundle.json').read_text(encoding='utf-8'))['catalog']

    def row(self, text, family='UNKNOWN'):
        row = {'id': 'fixture', 'split': 'train', 'kind': 'human', 'source': 'fsc',
            'unusable_reasons': [], 'filtered_reasons': [], 'command': family, 'transcript': text}
        row['annotation'] = annotate(row, self.catalog)
        row['supervision'] = copy.deepcopy(row['annotation'])
        return row

    def test_plain_queries_and_qualified_requests_have_distinct_semantics(self):
        for text, expected in [('what date is it', 'DATE_QUERY'), ('tell me the current time', 'TIME_QUERY')]:
            revised, _ = revise(self.row(text), self.catalog)
            self.assertEqual(revised['command'], expected)
        for text, family in [('tell me the time in london', 'TIME_QUERY'), ('what date is next monday', 'DATE_QUERY')]:
            revised, _ = revise(self.row(text, family), self.catalog)
            self.assertEqual(revised['command'], 'UNKNOWN')
        revised, _ = revise(self.row('may i have the current time please', 'TIME_QUERY'), self.catalog)
        self.assertEqual(revised['command'], 'TIME_QUERY')

    def test_target_corrections_preserve_original(self):
        original = self.row('turn the lights off in my bed room', 'POWER_OFF')
        before = copy.deepcopy(original)
        revised, _ = revise(original, self.catalog)
        self.assertEqual(original, before)
        self.assertEqual(revised['annotation']['labels']['target'], 'bedroom')
        revised, _ = revise(self.row('turn on the porch lights', 'POWER_ON'), self.catalog)
        self.assertEqual(revised['annotation']['labels']['target'], 'OTHER')
        self.assertEqual(revised['annotation']['reference']['status'], 'needs_parameters')

    def test_source_balancing_changes_sampling_without_validation_leakage(self):
        rows = [self.row('what time is it', 'TIME_QUERY') for _ in range(100)]
        rows.append({**self.row('what time is it', 'TIME_QUERY'), 'source': 'slurp'})
        cfg = {'per_family_draws': 4000, 'slot_draws': 0}
        for balanced, lower, upper in [(True, .45, .55), (False, 0, .03)]:
            selected = sample_epoch(rows, cfg, random.Random(17), balanced)
            fraction = sum(rows[i]['source'] == 'slurp' for i in selected) / len(selected)
            self.assertTrue(lower <= fraction <= upper)
        rows[0]['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'eligible human training'):
            sample_epoch(rows, cfg, random.Random(17), True)

    def test_frozen_overlay_and_original_manifest(self):
        verify()

    @unittest.expectedFailure
    def test_known_frozen_annotation_gap_with_intervening_query_words(self):
        # Discovered after fitting started. Preserve v1 evidence; a follow-up
        # must version the overlay before correcting and repeating fitting.
        revised, _ = revise(self.row('what time is it in london', 'TIME_QUERY'), self.catalog)
        self.assertEqual(revised['command'], 'UNKNOWN')

    def release(self, root, name, when, complete=True):
        folder = root / '.cache' / name / 'uniform'; folder.mkdir(parents=True)
        (folder / 'vcm.fp32.onnx').write_bytes(b'test graph bytes, never executed')
        save_json(folder / 'bundle.json', {'runtime_components': ['vcm']})
        files = {n: {'sha256': digest(folder / n), 'bytes': (folder / n).stat().st_size} for n in ['bundle.json', 'vcm.fp32.onnx']}
        exp = root / 'experiments' / name; exp.mkdir(parents=True)
        selection = exp / 'selection.json'
        save_json(selection, {'files': {(folder / n).relative_to(root).as_posix(): f['sha256'] for n,f in files.items()}})
        result = exp / 'results.json'
        save_json(result, {'selection_sha256': digest(selection), 'selected_arm': 'uniform', 'fixed_joint_gate_passed': False})
        receipt = {'format_version': 1, 'experiment': name, 'complete': complete, 'completed_utc': when,
            'bundle_directory': folder.relative_to(root).as_posix(), 'files': files, 'arm': 'uniform',
            'selection_receipt': selection.relative_to(root).as_posix(), 'selection_sha256': digest(selection),
            'test_receipt': result.relative_to(root).as_posix(), 'test_sha256': digest(result),
            'fixed_joint_gate_passed': False, 'requirements_complete': False}
        save_json(exp / 'release.json', receipt)
        return folder, exp, receipt

    def test_latest_uses_completed_release_timestamp_and_requires_explicit_experimental(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            self.release(root, 'old', '2026-09-19T00:00:00+00:00')
            self.release(root, 'new', '2026-09-20T00:00:00+00:00')
            self.release(root, 'unfinished', '2026-09-21T00:00:00+00:00', complete=False)
            with patch.object(publish, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'unmet gates'):
                    publish.resolve_iteration('latest')
                _, result = publish.resolve_iteration('latest', experimental=True)
                self.assertEqual(result['experiment'], 'new')

    def test_publish_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            folder, exp, receipt = self.release(root, 'trial', '2026-09-20T00:00:00+00:00')
            with patch.object(publish, 'ROOT', root):
                with self.assertRaises(ValueError): publish.resolve_iteration('../trial', experimental=True)
                original = (folder / 'vcm.fp32.onnx').read_bytes()
                (folder / 'vcm.fp32.onnx').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'integrity'):
                    publish.resolve_iteration('trial', experimental=True)
                (folder / 'vcm.fp32.onnx').write_bytes(original)
                receipt['bundle_directory'] = 'artifacts'
                save_json(exp / 'release.json', receipt)
                with self.assertRaisesRegex(ValueError, 'experiment cache'):
                    publish.resolve_iteration('trial', experimental=True)

    def test_release_requires_test_of_exact_development_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.cache') as temporary:
            root = Path(temporary)
            _, exp, receipt = self.release(root, 'trial', '2026-09-20T00:00:00+00:00')
            save_json(exp / 'results.json', {'selection_sha256': 'another-selection', 'selected_arm': 'uniform', 'fixed_joint_gate_passed': False})
            receipt['test_sha256'] = digest(exp / 'results.json'); save_json(exp / 'release.json', receipt)
            with patch.object(publish, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'this selected iteration'):
                    publish.resolve_iteration('trial', experimental=True)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrototypeChecks))
    save_json(ROOT / 'experiments/vcm-prototype-v1/software-checks.json', {
        'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
        'known_expected_failures': len(result.expectedFailures),
        'known_gap': 'Frozen v1 rule misses what time is it in london; broader location qualification requires a new overlay version and refit.',
        'passed': result.wasSuccessful(), 'scope': 'Label semantics, training-only sampling, immutable inputs and release lineage; no acoustic/Pi/live quality claim'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
