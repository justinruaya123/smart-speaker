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

    def test_frozen_feature_replay_returns_exact_bound_bytes(self):
        import hashlib
        import io
        import zipfile
        import numpy as np
        m=self.module()
        self.assertTrue(hasattr(m,'frozen_feature_payload'),
                        'Explicit frozen-feature replay must preserve original input bytes')
        value=np.asfortranarray(np.arange(80,dtype='float32').reshape(40,2))
        stream=io.BytesIO();np.save(stream,value,allow_pickle=False);payload=stream.getvalue()
        expected=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.zip'
            with zipfile.ZipFile(path,'w') as archive:archive.writestr('features/'+expected+'.npy',payload)
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(m.frozen_feature_payload(archive,{'id':'fixture','features_sha256':expected},560),payload)

    def test_frozen_feature_replay_rejects_invalid_arrays_even_with_bound_hashes(self):
        import hashlib
        import io
        import zipfile
        import numpy as np
        m=self.module()
        for value in (np.zeros((40,3),dtype='float32'),np.zeros((40,2),dtype='float64'),
                      np.full((40,2),np.nan,dtype='float32')):
            with self.subTest(shape=value.shape,dtype=value.dtype):
                payload_file=io.BytesIO();np.save(payload_file,value,allow_pickle=False);payload=payload_file.getvalue()
                expected=hashlib.sha256(payload).hexdigest();container=io.BytesIO()
                with zipfile.ZipFile(container,'w') as archive:archive.writestr('features/'+expected+'.npy',payload)
                with zipfile.ZipFile(container) as archive,self.assertRaisesRegex(ValueError,'Invalid frozen feature array'):
                    m.frozen_feature_payload(archive,{'id':'fixture','features_sha256':expected},560)

    def test_frozen_feature_replay_is_explicit_and_forwarded_to_worker(self):
        m=self.module();defaults=m.parser().parse_args([])
        self.assertTrue(hasattr(defaults,'frozen_features'),'Frozen input replay must be an explicit CLI mode')
        self.assertIsNone(defaults.frozen_features)
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'features.zip';archive.write_bytes(b'fixture')
            destination=Path(directory)/'new-run'
            with mock.patch.object(m,'copy_source'),mock.patch.object(m.subprocess,'run',return_value=mock.Mock(returncode=0)) as execute:
                self.assertEqual(m.main(['--frozen-features',str(archive),'--workspace',str(destination),'--yes','--no-setup']),0)
            command=execute.call_args.args[0]
            self.assertEqual(command[command.index('--frozen-features')+1],str(archive.resolve()))

    def test_reconstruction_replays_frozen_inputs_and_normalization_without_rederivation(self):
        import gzip
        import hashlib
        import io
        import types
        import zipfile
        import numpy as np
        import soundfile as sf
        from scripts.test.speaker_demo_suite import pcm_digest
        m=self.module()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);recipe_dir=root/'submission/reproduction-full';recipe_dir.mkdir(parents=True)
            pcm=np.full(560,.25,dtype='float32');wav=io.BytesIO();sf.write(wav,pcm,16000,format='WAV',subtype='PCM_16')
            payload=wav.getvalue();audio_sha=hashlib.sha256(payload).hexdigest()
            values=np.asfortranarray(np.arange(80,dtype='float32').reshape(40,2));blob=io.BytesIO();np.save(blob,values,allow_pickle=False)
            original=blob.getvalue();feature_sha=hashlib.sha256(original).hexdigest()
            row=dict(id='fixture',speaker='speaker-a',pcm_sha256=pcm_digest(pcm),sha256=audio_sha,
                     expected=None,features_sha256=feature_sha,location={'auxiliary_file':audio_sha+'.wav'})
            rows=gzip.compress((json.dumps(row)+'\n').encode());(recipe_dir/'rows.jsonl.gz').write_bytes(rows)
            (recipe_dir/'auxiliary-source.json').write_text(json.dumps(dict(repository='fixture',revision='pinned',prefix='audio')))
            downloads=root/'.cache/reproduction/downloads';downloads.mkdir(parents=True);(downloads/(audio_sha+'.wav')).write_bytes(payload)
            mean=values.astype('float64').mean(axis=1);square=(values.astype('float64')**2).mean(axis=1)
            (root/'normalization.json').write_text(json.dumps(dict(mean=mean.tolist(),std=np.sqrt(np.maximum(square-mean**2,1e-6)).tolist())))
            cache=root/'frozen.zip'
            with zipfile.ZipFile(cache,'w') as archive:archive.writestr('features/'+feature_sha+'.npy',original)
            views={key:[] for key in ('train','dev','calibration','test','holdout','normalization')}
            views.update(train=['fixture'],normalization=['fixture'])
            recipe=dict(rows_file='rows.jsonl.gz',rows_sha256=hashlib.sha256(rows).hexdigest(),views=views,normalization={'path':'normalization.json'})
            args=types.SimpleNamespace(model_choice='full',source_cache=None,frozen_features=cache)
            with mock.patch.object(m,'ROOT',root),mock.patch('vcm.frontend.features',side_effect=AssertionError('Frozen inputs were rederived')):
                prepared,target=m.reconstruct(args,recipe)
            self.assertEqual(Path(prepared['fixture']['features']).read_bytes(),original)
            audit=json.loads((target/'audit.json').read_text())
            self.assertEqual(audit['feature_input_mode'],'frozen_hash_bound_replay')
            self.assertEqual(audit['frozen_feature_rows'],1)
            self.assertEqual(audit['frozen_features_sha256'],hashlib.sha256(cache.read_bytes()).hexdigest())
            self.assertTrue(audit['normalization_reconstructed'])
            self.assertEqual(audit['feature_hash_mismatches'],[])
            self.assertFalse((root/'results').exists())

    def test_frozen_feature_replay_rejects_oversize_before_decompression(self):
        import hashlib
        import io
        import zipfile
        m=self.module();payload=b'oversized'*1000;expected=hashlib.sha256(payload).hexdigest()
        container=io.BytesIO()
        with zipfile.ZipFile(container,'w') as archive:archive.writestr('features/'+expected+'.npy',payload)
        with zipfile.ZipFile(container) as archive,mock.patch.object(archive,'read',side_effect=AssertionError('Oversized image was decompressed')):
            with self.assertRaisesRegex(ValueError,'Frozen feature image exceeds'):
                m.frozen_feature_payload(archive,{'id':'fixture','features_sha256':expected},560)

    def test_frozen_feature_replay_fails_closed_on_missing_or_corrupt_images(self):
        import io
        import zipfile
        m=self.module();expected='0'*64
        for entries in ({},{'features/'+expected+'.npy':b'changed'}):
            with self.subTest(entries=len(entries)):
                container=io.BytesIO()
                with zipfile.ZipFile(container,'w') as archive:
                    for name,payload in entries.items():archive.writestr(name,payload)
                with zipfile.ZipFile(container) as archive,self.assertRaises((KeyError,ValueError)):
                    m.frozen_feature_payload(archive,{'id':'fixture','features_sha256':expected},560)

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
