"""Protect public evidence identity and the distinction between replay and Pi tests."""
from pathlib import Path
import shutil
import tempfile
import unittest
import json

from scripts.submission import report


ROOT = Path(__file__).resolve().parents[2]


class SubmissionReportTests(unittest.TestCase):
    def test_comparison_keeps_policies_and_denominators(self):
        summary = report.build_summary(ROOT)
        compact = summary['models']['935k']
        full = summary['models']['2m']
        self.assertEqual(compact['test']['human']['correct_accepts'], 238)
        self.assertEqual(compact['test']['human']['positive_n'], 636)
        self.assertEqual(compact['test']['human']['unknown_false_accepts'], 1)
        self.assertEqual(full['test']['human']['correct_accepts'], 274)
        self.assertNotEqual(compact['demo_policy'], full['demo_policy'])
        self.assertFalse(summary['physical_pi_measured'])
        self.assertIsNone(summary['pi_measurements']['latency_p95_ms'])
        self.assertIn('Complete-utterance', summary['encoder'])

    def test_changed_public_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'submission/training', root/'submission/training')
            record = report.read(root/'submission/training/manifest.json')['histories'][0]
            path = root/record['path']
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'history identity'):
                report.load_histories(root)

    def test_desktop_benchmark_cannot_fill_pi_fields(self):
        summary = report.build_summary(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metrics.json'
            path.write_text(json.dumps({'hardware':{'physical_pi_measured':False,'machine':'x86_64','model':'PC'},
                                        'latency_ms':{'p95':1.0}}))
            report.attach_benchmark(summary,path)
        self.assertFalse(summary['physical_pi_measured'])
        self.assertIsNone(summary['pi_measurements']['latency_p95_ms'])

    def test_reproduction_benchmark_must_match_the_regenerated_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'results/benchmark/metrics.json'
            path.parent.mkdir(parents=True)
            metrics = {'hardware':{'physical_pi_measured':False,'machine':'x86_64','model':'PC'},
                       'model_binding':{'model_sha256':'new-graph'},'latency_ms':{'p95':12.5}}
            path.write_text(json.dumps(metrics))
            summary = {'onnx_sha256':'new-graph','physical_pi_measured':False}
            report.attach_reproduction_benchmark(summary,root)
            self.assertEqual(summary['desktop_benchmark']['latency_ms']['p95'],12.5)
            self.assertFalse(summary['physical_pi_measured'])
            self.assertEqual(summary['benchmark_receipt']['path'],'results/benchmark/metrics.json')
            self.assertEqual(summary['benchmark_receipt']['sha256'],report.sha(path))
            summary['onnx_sha256'] = 'different-graph'
            with self.assertRaisesRegex(ValueError,'regenerated graph'):
                report.attach_reproduction_benchmark(summary,root)

    def test_reproduction_pi_benchmark_requires_pi_hardware_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'results/benchmark/metrics.json'
            path.parent.mkdir(parents=True)
            metrics = {'hardware':{'physical_pi_measured':True,'machine':'x86_64','model':'PC'},
                       'model_binding':{'model_sha256':'new-graph'}}
            path.write_text(json.dumps(metrics))
            summary = {'onnx_sha256':'new-graph','physical_pi_measured':False}
            with self.assertRaisesRegex(ValueError,'hardware identity'):
                report.attach_reproduction_benchmark(summary,root)
            metrics['hardware'].update(machine='aarch64',model='Raspberry Pi 4 Model B')
            path.write_text(json.dumps(metrics))
            report.attach_reproduction_benchmark(summary,root)
            self.assertTrue(summary['physical_pi_measured'])
            self.assertIsNone(summary['keyword_accuracy'])
            self.assertFalse(summary['live_microphone_measured'])

    def test_reproduction_render_uses_only_new_run_and_portable_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ('human-shared','mixed-control'):
                path = root/'.cache/reproduction/training'/phase/'history.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps([{'epoch':1,'losses':[1.,.1,.01]}]))
            (root/'results').mkdir()
            result = {'seed':7,'original_seed':53,'weights_bitwise_equal':False,
                      'maximum_weight_absolute_difference':.2,'same_human_epoch':False,'same_mixed_epoch':False,
                      'strict_qualification_enabled':False,'physical_pi_measured':False,
                      'results':{'test':{'human':{'raw':{'exact':400,'positive_n':636},
                                                'demo':{'exact':180,'positive_n':636,'negative_errors':2,'negative_n':229}}}},
                      'original_human_test':{'raw_exact':489,'demo_exact':238,'positive_n':636,'unknown_accepts':1,'negative_n':229}}
            (root/'results/comparison.json').write_text(json.dumps(result))
            shutil.copytree(ROOT/'submission/reproduction',root/'submission/reproduction')
            report.render_reproduction(root)
            self.assertTrue((root/'results/figures/training-curves.svg').is_file())
            self.assertTrue((root/'results/figures/model-dataset-training-card.png').is_file())
            summary = report.read(root/'results/figures/metrics.json')
            self.assertEqual(summary['seed'],7)
            self.assertFalse(summary['physical_pi_measured'])


if __name__ == '__main__':
    unittest.main()
