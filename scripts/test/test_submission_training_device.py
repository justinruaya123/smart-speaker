"""CPU-only hosts must execute the shared fitting path without CUDA calls."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from scripts.train import speaker_final_train as training


class TrainingDeviceTests(unittest.TestCase):
    def test_cpu_configuration_never_configures_cuda(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.object(torch.cuda, 'set_per_process_memory_fraction', side_effect=AssertionError('CUDA call')):
            self.assertEqual(training.configure_training_device(), 'cpu')

    def test_cuda_configuration_preserves_existing_memory_limit(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'set_per_process_memory_fraction') as limit:
            self.assertEqual(training.configure_training_device(), 'cuda')
            limit.assert_called_once_with(.45)

    def test_cpu_fit_stage_runs_real_optimizer_and_saves_checkpoint(self):
        architecture = dict(width=8, layers=1, heads=2, feedforward=16, dropout=0.)
        torch.manual_seed(53)
        initial = training.PooledContextVCM(**architecture)
        before = {key: value.detach().clone() for key, value in initial.state_dict().items()}
        rows = [dict(id='positive', split='train', training_eligible=True, kind='human', synthetic=False,
                     expected={'intent':'TIME_QUERY','slots':{}}, intent_index=1,
                     target_index=-100, speaker='train', phones=[1,2]),
                dict(id='negative', split='train', training_eligible=True, kind='human', synthetic=False,
                     expected=None, intent_index=0, target_index=-100,
                     speaker='train', phones=[1,2])]
        arrays = {r['id']:np.random.default_rng(53).normal(0,.1,(40,20)).astype('float32') for r in rows}
        config = dict(architecture=architecture, parameters=sum(p.numel() for p in initial.parameters()),
                      training={**training.TRAINING, 'human_epochs':1, 'steps_per_epoch':1,
                                'development_every':1})
        predictions = [dict(synthetic=False, expected=None, candidate=None,
                            application_command=None, speaker='dev')]
        scratch = Path(os.environ.get('TMPDIR', training.ROOT/'.cache'))
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            root = Path(directory); (root/'protocol.json').write_text('{}')
            with mock.patch.multiple(training, ROOT=root, HERE=root, CACHE=root/'cache'), \
                    mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                    mock.patch.object(torch.nn.Module, 'cuda', side_effect=AssertionError('CUDA model')), \
                    mock.patch.object(torch.cuda, 'empty_cache', side_effect=AssertionError('CUDA cleanup')), \
                    mock.patch.object(training, 'score_model', return_value=predictions) as score:
                result = training.fit_stage('human','shared',None,training.TrainingPools(rows),
                                            [rows[1]],arrays,[],None,config)
            checkpoint = torch.load(root/result['checkpoint'], map_location='cpu', weights_only=True)
            self.assertTrue(all(value.device.type=='cpu' for value in checkpoint['weights'].values()))
            self.assertTrue(any(not torch.equal(before[key], value) for key,value in checkpoint['weights'].items()))
            self.assertEqual(score.call_args.kwargs['device'], 'cpu')
            self.assertEqual(result['device'], 'cpu')
            self.assertEqual(checkpoint['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
