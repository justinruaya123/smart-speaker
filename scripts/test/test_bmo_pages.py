"""Network-free checks for the BMO Pages artifact and its failure boundaries."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def implementation():
    spec = importlib.util.spec_from_file_location('bmo_validate', ROOT / 'presentation/bmo_validate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BmoPagesTests(unittest.TestCase):
    def setUp(self):
        self.validator = implementation()
        self.temp = tempfile.TemporaryDirectory(prefix='bmo-pages-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / 'docs/site'
        shutil.copytree(ROOT / 'docs/site/bmo', self.site / 'bmo')
        for name in ('index.html', 'reference.html'):
            shutil.copyfile(ROOT / 'docs/site' / name, self.site / name)
        self.validator.write_manifest(self.root)

    def change(self, name, old, new):
        path = self.site / name
        body = path.read_text(encoding='utf-8')
        self.assertIn(old, body, 'mutation must exercise a real input')
        path.write_text(body.replace(old, new, 1), encoding='utf-8')

    def change_json(self, name, mutate):
        path = self.site / 'bmo' / name
        data = json.loads(path.read_text(encoding='utf-8'))
        mutate(data)
        path.write_text(json.dumps(data), encoding='utf-8')

    def reject(self, message, refresh=True):
        if refresh:
            self.validator.write_manifest(self.root)
        with self.assertRaisesRegex(ValueError, message):
            self.validator.validate_bmo(self.root)

    def test_complete_site_is_self_contained_and_reports_evidence_scope(self):
        summary = self.validator.validate_bmo(self.root)
        self.assertEqual(summary['slides'], 4)
        self.assertEqual(summary['wordings'], 102)
        self.assertEqual(summary['commands'], 34)
        self.assertEqual(summary['training_epochs'], 48)
        self.assertEqual(summary['published_epoch'], 38)
        self.assertEqual(summary['refit_epoch'], 36)
        self.assertIsNone(summary['pi'])

    def test_root_relative_asset_is_rejected_even_when_file_exists(self):
        self.change('index.html', 'href="bmo/deck.css"', 'href="/bmo/deck.css"')
        self.reject('project.prefix')

    def test_missing_slide_is_not_publishable(self):
        self.change('index.html', 'data-slide="training"', 'data-removed="training"')
        self.reject('four.*slides')

    def test_wrong_page_identity_is_not_publishable(self):
        self.change('index.html', '<title>BMO', '<title>Iris')
        self.reject('identity')

    def test_missing_module_and_document_relative_fetch_are_rejected(self):
        self.change('bmo/deck.js', "'./model-scene.js'", "'./missing-model.js'")
        self.reject('missing.*missing-model')

    def test_fetch_does_not_accidentally_resolve_beside_index(self):
        path = self.site / 'bmo/deck.js'
        with path.open('a', encoding='utf-8') as stream:
            stream.write("\nfetch('./data/trace.json');\n")
        self.reject('missing.*trace.json')

    def test_external_runtime_resource_is_rejected_but_https_citations_work(self):
        self.change('index.html', 'src="bmo/deck.js"', 'src="https://cdn.example/deck.js"')
        self.reject('external.*resource')

    def test_css_import_cannot_add_an_external_font(self):
        path = self.site / 'bmo/deck.css'
        with path.open('a', encoding='utf-8') as stream:
            stream.write('\n@import "https://fonts.example/font.css";\n')
        self.reject('external.*resource')

    def test_local_modal_navigation_is_checked(self):
        self.change('bmo/sources.js', "'data/provenance.json'", "'data/missing-provenance.json'")
        self.reject('missing.*missing-provenance')

    def test_escape_and_encoded_root_paths_are_rejected(self):
        for ref in ('../outside.js', '%2fbmo/deck.css', '//cdn.example/deck.css'):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.validator.check_reference(self.site, self.site / 'index.html', ref, True)

    def test_manifest_detects_bytes_changed_after_inventory(self):
        with (self.site / 'bmo/deck.css').open('a', encoding='utf-8') as stream:
            stream.write('\n/* changed after manifest */\n')
        self.reject('manifest', refresh=False)

    def test_manifest_writer_never_changes_legacy_inventory(self):
        legacy = self.site / 'asset-manifest.json'
        legacy.write_bytes(b'legacy inventory sentinel\n')
        self.validator.write_manifest(self.root)
        self.assertEqual(legacy.read_bytes(), b'legacy inventory sentinel\n')
        inventory = json.loads((self.site / 'bmo/manifest.json').read_text())
        self.assertIn('data/sample.wav', inventory['files'])
        self.assertNotIn('manifest.json', inventory['files'])

    def test_raster_artwork_is_rejected(self):
        (self.site / 'bmo/picture.png').write_bytes(b'not an authorized code-rendered visual')
        self.change('index.html', '</main>', '<img src="bmo/picture.png" alt="art"></main>')
        self.reject('raster|imagery')

    def test_recorded_trace_cannot_drift_even_with_new_manifest(self):
        self.change_json('data/trace.json', lambda d: d['logits37'].__setitem__(0, 999))
        self.reject('provenance|trace')

    def test_audio_must_match_recorded_forward_pass(self):
        path = self.site / 'bmo/data/sample.wav'
        payload = bytearray(path.read_bytes())
        payload[-10] ^= 1
        path.write_bytes(payload)
        self.reject('provenance|audio|sample.wav')

    def test_training_curve_cannot_be_shortened_or_replaced(self):
        self.change_json('data/training-history.json', lambda d: d['epochs'].pop())
        self.reject('48|history|epoch')

    def test_training_loss_is_verified_against_frozen_extract(self):
        self.change_json('data/training-history.json', lambda d: d['epochs'][0].__setitem__('training_loss', 1))
        self.reject('history|curve')

    def test_refit_cannot_inherit_published_qualification(self):
        self.change_json('data/training-history.json', lambda d: d.__setitem__('candidate_test_evaluated', True))
        self.reject('refit|candidate')

    def test_wording_command_mapping_matches_csv(self):
        self.change('bmo/wordings.csv', 'Play music,Play music', 'Play another command,Play music')
        self.reject('wording|CSV')

    def test_source_specific_terms_are_required(self):
        self.change_json('evidence/facts.json', lambda d: d['dataset'].__setitem__('license_name', 'MIT'))
        self.reject('source.specific|terms')

    def test_unmeasured_pi_cannot_acquire_a_numeric_result(self):
        self.change_json('evidence/facts.json', lambda d: d['raspberry_pi'].__setitem__('latency_p50_ms', 1))
        self.reject('Pi|unmeasured')

    def test_missing_generation_receipt_is_not_hidden_by_manifest_refresh(self):
        (self.site / 'bmo/data/dataset-recipe.json').unlink()
        self.reject('dataset-recipe|missing')

    def test_generation_sources_must_reconcile_with_split_totals(self):
        self.change_json('data/dataset-recipe.json', lambda d: d['source_composition'][0].__setitem__('rows', 1))
        self.reject('composition')

    def test_unknown_citation_id_is_rejected(self):
        self.change_json('data/dataset-recipe.json', lambda d: d['generation_sources'][0]['sources'].append('missing_receipt'))
        self.reject('citation IDs')

    def test_benchmark_cannot_replace_pi_placeholder_with_desktop_latency(self):
        self.change_json('data/benchmark.json', lambda d: d['raspberry_pi'].__setitem__('p50_ms', 3.434))
        self.reject('Pi')

    def test_benchmark_percentages_must_use_stated_denominator(self):
        self.change_json('data/benchmark.json', lambda d: d['accuracy'][0].__setitem__('accuracy', 1))
        self.reject('denominator')

    def test_tts_generation_cannot_be_relabelled_as_teacher_distillation(self):
        self.change_json('data/dataset-recipe.json', lambda d: d['distillation'].__setitem__('teacher_model', 'invented-teacher'))
        self.reject('distillation teacher')

    def test_malformed_evidence_fails_with_validation_error(self):
        self.change_json('data/benchmark.json', lambda d: d['baseline'].pop('model_sha256'))
        self.reject('invalid or missing BMO evidence')


if __name__ == '__main__':
    unittest.main()
