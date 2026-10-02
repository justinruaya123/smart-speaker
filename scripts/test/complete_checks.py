"""Behavioral research checks that require no corpus fitting or microphone."""
from pathlib import Path
import json
import random
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources
register_sources()
from tinyvcm.complete_eval import summarize_fixed, custom_metrics
from tinyvcm.complete_training import mask_training_features, sample_epoch
from tinyvcm.complete_data import verify_frozen, read_rows, DATA, EXP, save


class CompleteChecks(unittest.TestCase):
    def test_multiple_endpoints_cannot_earn_a_correct_command(self):
        gold = {'action': 'POWER_ON', 'arguments': {'target': {'id': 'main_lights'}}}
        accepted = {'status': 'accepted', 'vcm_command': 'POWER_ON', 'command': gold, 'intent_accepted': True}
        row = {'gold': 'POWER_ON', 'reference': gold, 'outputs': [accepted, accepted],
            'result': {'status': 'rejected'}, 'raw_family': 'POWER_ON', 'raw_command': gold}
        metric = summarize_fixed([row])['totals']
        self.assertEqual(metric['emitted'], 2)
        self.assertEqual(metric['correct_complete'], 0)

    def test_custom_retrieval_cannot_rescue_rejected_family_or_generic_mode(self):
        row = {'known': True, 'correct': True, 'playlist': True, 'score': .99, 'margin': .2,
            'family_accepted': False, 'custom_mode': True, 'custom_probability': 1.}
        policy = {'score': .8, 'margin': .1, 'custom_probability': .9}
        self.assertEqual(custom_metrics([row], policy, False)['accepted'], 1)
        self.assertEqual(custom_metrics([row], policy, True)['accepted'], 0)
        row.update(family_accepted=True, custom_mode=False)
        self.assertEqual(custom_metrics([row], policy, True)['accepted'], 0)

    def test_masks_leave_padding_and_original_features_untouched(self):
        original = torch.ones(2, 40, 50)
        original[0, :, 30:] = -7.
        result = mask_training_features(original.clone(), torch.tensor([30, 50]), random.Random(17))
        self.assertTrue(torch.equal(result[0, :, 30:], original[0, :, 30:]))
        self.assertTrue(torch.all(original[1] == 1.))
        self.assertLess(result.eq(1).sum(), original.eq(1).sum())

    def test_sampler_rejects_heldout_and_reserved_rows(self):
        for row in [{'split': 'dev'}, {'split': 'train', 'unusable_reasons': ['reserved_enrollment_recording']}]:
            with self.assertRaises(ValueError):
                sample_epoch([row], {}, random.Random(17))

    def test_frozen_inputs_and_declared_synthetic_draw_fraction(self):
        verify_frozen()
        rows = [r for r in read_rows(DATA / 'manifest.jsonl') if r['split'] == 'train'
                and not r['unusable_reasons'] and not r['filtered_reasons']]
        cfg = json.loads((EXP / 'protocol.json').read_text())['training']
        chosen = sample_epoch(rows, cfg, random.Random(17))
        self.assertEqual(sum(rows[i]['kind'] == 'synthetic' for i in chosen), round(len(chosen) * .25))

    def test_enrollment_has_no_training_or_query_speaker_or_paired_overlap(self):
        rows = read_rows(DATA / 'manifest.jsonl')
        catalog = json.loads((EXP / 'custom-catalog-v3.json').read_text())['catalog']
        ids = {i for e in catalog for i in e['enrollment_ids']}
        enrollment = [r for r in rows if r['id'] in ids]
        queries = [r for r in rows if r['kind'] == 'human' and
            (r['split'] == 'test' or r['split'] == 'dev' and r['fold'] == 'selection')]
        training = [r for r in rows if r['kind'] == 'human' and r['split'] == 'train']
        self.assertEqual(len(enrollment), 12)
        self.assertFalse({r['speaker_id'] for r in enrollment} & {r['speaker_id'] for r in training + queries})
        self.assertFalse({r['paired_group'] for r in enrollment} & {r['paired_group'] for r in queries})

    def test_number_stress_covers_all_values_in_each_voice_and_context(self):
        plan = json.loads((EXP / 'number-stress-plan-v2.json').read_text())
        for voice in ['designed-10', 'designed-11']:
            for kind in ['bare', 'distractor']:
                subset = [r for r in plan['rows'] if r['voice'] == voice and r['kind'] == kind]
                self.assertEqual(sorted(r['value'] for r in subset), list(range(101)))
                self.assertTrue(all(r['expected_family'] == 'UNKNOWN' and r['use'] == 'test_only' for r in subset))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CompleteChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    save(EXP / 'software-checks.json', {'tests': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'passed': result.wasSuccessful(), 'scope': 'Research behavior and data isolation; no microphone/Pi/model quality claim'})
    raise SystemExit(0 if result.wasSuccessful() else 1)
