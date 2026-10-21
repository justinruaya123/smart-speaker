"""Unit tests for targeted human-first training pipeline."""
import unittest
from pathlib import Path
import numpy as np
import torch

from scripts.train import speaker_targeted_train as targeted


class TargetedTrainingTests(unittest.TestCase):
    def test_model_architecture_parameter_budget(self):
        model = targeted.PooledContextVCM(**targeted.ARCHITECTURE)
        params = sum(p.numel() for p in model.parameters())
        self.assertEqual(params, 935567)
        self.assertLess(params, 1000000)

    def test_label_corrections_loading(self):
        corrections = targeted.load_label_corrections()
        self.assertGreaterEqual(len(corrections), 43)

    def test_pools_initialization_and_sampling(self):
        dummy_records = [
            {'id': 'h1', 'speaker': 'spk1', 'training_eligible': True, 'synthetic': False,
             'expected': {'intent': 'STOP', 'slots': {}}, 'phones': [1, 2], 'intent_index': 1, 'target_index': -100},
            {'id': 's1', 'speaker': 'synth', 'training_eligible': True, 'synthetic': True,
             'expected': {'intent': 'MESSAGE', 'slots': {}}, 'phones': [1, 2], 'intent_index': 2, 'target_index': -100},
            {'id': 's2', 'speaker': 'synth', 'training_eligible': True, 'synthetic': True,
             'expected': {'intent': 'STOP', 'slots': {}}, 'phones': [1, 2], 'intent_index': 1, 'target_index': -100},
        ]
        pools = targeted.TargetedTrainingPools(dummy_records)
        self.assertEqual(len(pools.human_speakers), 1)
        self.assertEqual(len(pools.synthetic), 2)

        rng = np.random.default_rng(53)
        batch_h = targeted.sample_batch(pools, rng, size=4, stage='human')
        self.assertEqual(len(batch_h), 4)
        self.assertTrue(all(not r['synthetic'] for r in batch_h))

        batch_m = targeted.sample_batch(pools, rng, size=4, stage='mixed')
        self.assertEqual(len(batch_m), 4)
        self.assertEqual(sum(1 for r in batch_m if r['synthetic']), 2)
        self.assertEqual(sum(1 for r in batch_m if not r['synthetic']), 2)


if __name__ == '__main__':
    unittest.main()
