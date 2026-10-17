"""Reconstruction must bind exact data and preserve original evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


class ReproduceTests(unittest.TestCase):
    def module(self):
        spec = importlib.util.find_spec('scripts.submission.reproduce')
        self.assertIsNotNone(spec, 'Portable reproduction must be implemented')
        return __import__('scripts.submission.reproduce', fromlist=['reproduce'])

    def test_default_seed_and_full_recipe_preserve_original_budget(self):
        m = self.module(); args=m.parser().parse_args([])
        self.assertIsNone(args.seed)
        cfg=json.loads((ROOT/'submission/reproduction/recipe.json').read_text())
        self.assertEqual(cfg['original_seed'],53)
        self.assertEqual(cfg['training']['human_epochs'],24)
        self.assertEqual(cfg['training']['mixed_epochs'],32)
        self.assertEqual(len(cfg['views']['train']),23289)

    def test_existing_workspace_is_preserved(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run';path.mkdir();(path/'prior').write_text('evidence')
            with self.assertRaises(FileExistsError):m.copy_source(path)
            self.assertEqual((path/'prior').read_text(),'evidence')

    def test_public_source_inventory_has_no_caches_credentials_or_dataset_payloads(self):
        m=self.module();names=m.source_paths()
        self.assertIn('scripts/train/speaker_final_train.py',names)
        self.assertIn('scripts/models/pcv_release_pooling.py',names)
        for name in names:
            self.assertFalse(name.startswith(('.cache/','.venv/','data/audio/')))
            self.assertNotEqual(Path(name).name,'.env')

    def test_reconstruction_rejects_modified_row_map(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);(path/'rows.jsonl.gz').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.load_recipe_rows({'rows_file':'rows.jsonl.gz','rows_sha256':'0'*64},path)

    def test_seed_rejects_mixed_phase_overflow_before_creating_a_workspace(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'run'
            with mock.patch.object(m,'copy_source',side_effect=AssertionError('Seed passed preflight')), self.assertRaises(ValueError):
                m.main(['--seed',str(2**32-1),'--workspace',str(path),'--no-setup','--yes'])
            self.assertFalse(path.exists())

    def test_post_training_benchmark_targets_the_new_model(self):
        m=self.module()
        self.assertTrue(hasattr(m,'benchmark_command'),'The post-training benchmark must bind the regenerated model')
        root=Path('/example/new-run')
        command=m.benchmark_command(root,'python')
        self.assertEqual(command[command.index('--model')+1],str(root/'results/model'))
        self.assertEqual(command[command.index('--output')+1],str(root/'results/benchmark'))
        self.assertEqual(command[command.index('--target')+1],'pc')

    def test_successful_post_training_benchmark_refreshes_the_card(self):
        m=self.module();root=Path('/example/new-run')
        with mock.patch.object(m.subprocess,'call',return_value=0) as execute, \
                mock.patch('scripts.submission.report.render_reproduction') as render:
            self.assertEqual(m.benchmark_reproduction(root,'python'),0)
            execute.assert_called_once_with(m.benchmark_command(root,'python'),cwd=root)
            render.assert_called_once_with(root)
        with mock.patch.object(m.subprocess,'call',return_value=1), \
                mock.patch('scripts.submission.report.render_reproduction') as render:
            self.assertEqual(m.benchmark_reproduction(root,'python'),1)
            render.assert_not_called()

    def test_evaluation_uses_single_thread_frontend_and_preserves_fitting_features(self):
        import numpy as np
        import soundfile as sf
        from threadpoolctl import threadpool_limits,threadpool_info
        from vcm.frontend import features
        m=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            pcm=np.random.default_rng(53).normal(0,.05,16000).astype('float32')
            audio=root/'audio.wav';sf.write(audio,pcm,16000,subtype='FLOAT')
            cached=np.full((40,98),123,dtype='float32');path=root/'frozen-dev.npy';np.save(path,cached)
            prepared={key:dict(id=key,audio=str(audio),features=str(path),capture_fault=False,pcm_sha256=key)
                      for key in ('dev','test')}
            recipe={'views':{'dev':['dev'],'calibration':[],'test':['test'],'holdout':[]}}
            norm={'mean':[0]*40,'std':[1]*40}
            before=path.read_bytes()
            with threadpool_limits(limits=1,user_api='blas'):expected=features(pcm)
            with threadpool_limits(limits=4,user_api='blas'):
                arrays=m.evaluation_arrays(prepared,recipe,norm,root/'evaluation')
                self.assertTrue(all(p['num_threads']==4 for p in threadpool_info() if p['user_api']=='blas'))
            self.assertTrue(np.array_equal(arrays['test'],expected))
            self.assertTrue(np.array_equal(arrays['dev'],cached))
            self.assertEqual(path.read_bytes(),before)
            audit=json.loads((root/'evaluation/manifest.json').read_text())
            self.assertEqual(audit['blas_threads'],1)
            self.assertEqual(len(audit['files']),1)


if __name__ == '__main__':unittest.main()
