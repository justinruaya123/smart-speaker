"""Validate the self-contained BMO documentation artifact using only the stdlib.

This is a static documentation gate, not an acoustic release or Pi qualification
gate. No network, scientific imports, model inference, or training is performed.
Only --write-manifest writes anything, and only docs/site/bmo/manifest.json.
"""
import argparse
from collections import Counter
import csv
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import wave

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = '9b98acefa8bad6610d98501ff0310c9b2a2ec1594619b4b5b7d2c7a80dc888a6'
# Exact 48-row extract of the completed receipt, after checking every value
# against training-curve.jsonl (raw SHA-256 below). Paths are deliberately excluded
# from the digest so public provenance can redact local workstation paths.
CURVE_SOURCE_SHA256 = 'dde1d8497fbecf64a19ad06658fb3cf0e2c97f921338a9abad519d0a027a6e18'
CURVE_ROWS_SHA256 = 'e90fd37afc96d4b96ab67bd7bd4cb46ef7cd379a7cfa3488e883811f2b9488a1'
RASTER_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.bmp'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    require(path.is_file(), f'missing evidence: {path.name}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid JSON: {path.name}') from exc


def check_reference(site, source, reference, resource=False):
    """Resolve browser paths without accepting origin-root or filesystem paths.

    The site is served at /smart-speaker/, not necessarily /. HTTPS navigation
    is allowed; runtime dependencies must ship within this site.
    """
    reference = unquote(reference.strip())
    if reference == 'data:,' and source.suffix == '.html':
        return None  # deliberately empty favicon
    require('\\' not in reference, f'nonportable reference: {reference}')
    parts = urlsplit(reference)
    if parts.scheme or parts.netloc:
        require(not resource, f'external runtime resource: {reference}')
        require(parts.scheme == 'https' and bool(parts.netloc), f'navigation must use HTTPS: {reference}')
        return None
    if not parts.path:
        return None
    require(not parts.path.startswith('/'), f'project-prefix unsafe reference: {reference}')
    target = (source.parent / parts.path).resolve()
    require(target.is_relative_to(site.resolve()), f'reference escapes site: {reference}')
    require(target.is_file(), f'missing local target: {source.name} -> {reference}')
    return target


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.slides = []
        self.ids = []
        self.images = 0
        self.title = ''
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'title':
            self.in_title = True
        if tag in {'img', 'picture', 'image'}:
            self.images += 1
        if 'data-slide' in attrs:
            self.slides.append(attrs['data-slide'])
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        for key in ('src', 'href', 'poster', 'data-src'):
            value = attrs.get(key)
            if value and '${' not in value:
                self.links.append((value, key != 'href' or tag in {'link', 'base', 'use'}))
        if attrs.get('srcset'):
            self.images += 1

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def inventory(bmo):
    files = {}
    for path in sorted(bmo.rglob('*')):
        require(not path.is_symlink(), f'site cannot publish a symlink: {path}')
        if path.is_file() and path != bmo / 'manifest.json':
            files[path.relative_to(bmo).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return files


def write_manifest(root=ROOT):
    bmo = Path(root) / 'docs/site/bmo'
    require(bmo.is_dir(), 'missing BMO asset directory')
    manifest = {'format': 'bmo-static-assets-v1', 'files': inventory(bmo)}
    (bmo / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def validate_references(site, bmo, html):
    parser = References()
    parser.feed(html)
    require(parser.slides == ['architecture', 'dataset', 'training', 'evidence'], 'exactly four ordered BMO slides required')
    require(parser.title.startswith('BMO') and 'Iris' not in parser.title, 'BMO page identity required')
    require(len(parser.ids) == len(set(parser.ids)), 'duplicate HTML IDs')
    require(not parser.images, 'BMO imagery must be code-rendered, not raster images')
    for reference, resource in parser.links:
        check_reference(site, site / 'index.html', reference, resource)
    for source in sorted(bmo.rglob('*')):
        if not source.is_file():
            continue
        require(source.suffix.lower() not in RASTER_SUFFIXES, f'raster imagery is not part of BMO: {source.name}')
        if source.suffix not in {'.js', '.css'}:
            continue
        body = source.read_text(encoding='utf-8')
        if source.suffix == '.css':
            refs = re.findall(r'url\(\s*[\'\"]?([^\)\'\"]+)', body)
            refs += re.findall(r'@import\s+[\'\"]([^\'\"]+)', body)
            for ref in refs:
                check_reference(site, source, ref.strip(), True)
            continue
        # Static ES imports, re-exports, import(), and module-relative URLs.
        patterns = [r'\b(?:import|export)\s+[^;]*?\bfrom\s*[\'\"]([^\'\"]+)[\'\"]',
                    r'\bimport\s*(?:\(\s*)?[\'\"]([^\'\"]+)[\'\"]',
                    r'new\s+URL\(\s*[\'\"]([^\'\"]+)[\'\"]\s*,\s*import\.meta\.url']
        for pattern in patterns:
            for ref in re.findall(pattern, body):
                check_reference(site, source, ref, True)
        # fetch('x') resolves against the document, unlike new URL('x', import.meta.url).
        for ref in re.findall(r'\bfetch\(\s*[\'\"]([^\'\"]+)[\'\"]', body):
            check_reference(site, site / 'index.html', ref, True)
        if source.parent == bmo:
            embedded = References()
            embedded.feed(body)
            require(not embedded.images, f'BMO imagery must be code-rendered: {source.name}')
            for ref, resource in embedded.links:
                check_reference(site, source, ref, resource)
            # sources.js link() resolves local receipts beside its module.
            for ref in re.findall(r'\blink\(\s*[\'\"]([^\'\"]+)[\'\"]\s*,', body):
                check_reference(site, source, ref, False)
    return parser


def validate_wordings(bmo):
    body = (bmo / 'wordings.js').read_text(encoding='utf-8')
    match = re.fullmatch(r'\s*window\.BMO_WORDINGS\s*=\s*(\[.*\]);?\s*', body, re.S)
    require(match is not None, 'wordings must be a literal BMO_WORDINGS array')
    rows = json.loads(match[1])
    require(len(rows) == 102 and len({r['definition_id'] for r in rows}) == 102, '102 unique wordings required')
    commands = Counter(json.dumps(r['command'], sort_keys=True) for r in rows)
    require(len(commands) == 34 and set(commands.values()) == {3}, 'three wordings per each of 34 commands required')
    require(len({r['intent'] for r in rows}) == 18, 'wordings require 18 intent families')
    with (bmo / 'wordings.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(rows), 'wordings CSV row count mismatch')
    for row, exported in zip(rows, csv_rows):
        expected = {'definition_id': row['definition_id'], 'class_index': str(row['class_index']),
                    'intent': row['intent'], 'slots': row['command']['slots'],
                    'verbatim': row['text'], 'spoken_form': row['spoken']}
        exported['slots'] = json.loads(exported['slots'])
        require(exported == expected and row['intent'] == row['command']['intent'], 'wordings CSV command mapping mismatch')
    return len(rows), len(commands)


def validate_trace(bmo):
    provenance = read_json(bmo / 'data/provenance.json')
    trace = read_json(bmo / 'data/trace.json')
    for name in ('sample.wav', 'trace.json', 'extract_trace.py'):
        path = bmo / 'data' / name
        record = provenance['files'].get(name, {})
        require(path.is_file() and record.get('bytes') == path.stat().st_size and record.get('sha256') == sha(path),
                f'recorded trace provenance mismatch: {name}')
    for record in (trace['model'], provenance['model']):
        require(record['sha256'] == MODEL_SHA256 and record['selected_epoch'] == 38, 'trace must use published epoch-38 graph')
    require(trace['model']['qualified'] is False and trace['model']['actions_enabled'] is False, 'trace is not model qualification')
    require(len(trace['logits37']) == 37 and trace['logits37'] == trace['original_graph_logits37'], 'trace logits differ from original graph')
    for record in (trace, provenance):
        require(record['evidence_kind'] == 'recorded_forward_pass_demo', 'recorded trace evidence required')
        require(record['sample']['sha256'] == sha(bmo / 'data/sample.wav'), 'audio hash mismatch')
        require(record['sample']['synthetic'] is True and record['sample']['previously_evaluated'] is True, 'sample provenance must disclose synthetic evaluated source')
        require(record['verification']['original_vs_trace_allclose'] is True
                and record['verification']['training_updates'] == 0, 'forward trace verification required')
    with wave.open(str(bmo / 'data/sample.wav'), 'rb') as audio:
        require((audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getnframes()) == (1, 16000, 2, 47616),
                'sample audio must be the recorded mono 16 kHz PCM')
    require(provenance['license']['license_name'] == 'source-specific-research-terms', 'sample source-specific terms required')


def validate_history(bmo):
    history = read_json(bmo / 'data/training-history.json')
    epochs = history['epochs']
    require(history['complete'] is True and history['epoch_count'] == 48 and len(epochs) == 48, 'complete 48-epoch history required')
    require(history['selected_epoch'] == 36 and history['reference_selected_epoch'] == 38, 'published and refit epoch identities must stay separate')
    require(history['candidate_test_evaluated'] is False and history['installed'] is False
            and history['exact_reference_match'] is False and history['result_status'] == 'exact_fit_comparison_failed',
            'refit candidate must retain failed comparison and no test/installation status')
    require(history['source']['sha256'] == CURVE_SOURCE_SHA256 and history['source']['hash_matches_receipt'] is True, 'history must cite the completed raw curve receipt')
    for epoch, row in enumerate(epochs, 1):
        require(row['epoch'] == epoch and row['phase'] == ('human' if epoch <= 6 else 'mixed'), 'history epoch/phase mismatch')
        require(row['optimizer_steps'] == epoch * 96 and math.isfinite(row['training_loss']) and row['training_loss'] >= 0,
                'invalid measured training history')
        require(math.isfinite(row['lr']) and 0.00005 <= row['lr'] <= 0.002, 'invalid history learning rate')
    digest = hashlib.sha256(json.dumps(epochs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(digest == CURVE_ROWS_SHA256, 'training history curve differs from frozen measured extract')
    return history


def validate_facts(bmo):
    facts = read_json(bmo / 'evidence/facts.json')
    architecture, dataset = facts['architecture'], facts['dataset']
    require(architecture['onnx_sha256'] == MODEL_SHA256 and architecture['published_selected_epoch'] == 38,
            'facts must identify published epoch-38 graph')
    require(all(architecture[k] is False for k in ('qualified', 'actions_enabled', 'installed', 'asr', 'llm', 'cloud_recognition')),
            'facts must preserve model qualification and runtime limits')
    require(dataset['license_name'] == 'source-specific-research-terms' and dataset['license'] == 'other', 'dataset source-specific terms required')
    require(dataset['views']['train']['rows'] == 25958 and dataset['combined_dev']['rows'] == 2132
            and dataset['views']['test']['rows'] == 440 and dataset['views']['official']['rows'] == 202,
            'frozen dataset split totals mismatch')
    require(dataset['audio_hours'] is None and dataset['independent_human_speaker_count'] is None,
            'unaudited dataset hours/speaker counts must remain unknown')
    require(facts['training']['foundation_quotas'] == {'human_negative': 8, 'human_positive': 24}
            and facts['training']['mixed_quotas'] == {'demo_slot_positive': 8, 'human_negative': 4,
                'human_positive': 8, 'synthetic_negative': 4, 'synthetic_positive': 8}, 'frozen source quotas mismatch')
    reproduction = facts['rtx4080_reproduction']
    require(reproduction['selected_epoch'] == 36 and reproduction['published_reference_epoch'] == 38
            and reproduction['candidate_test_evaluated'] is False and reproduction['exact_fit_comparison_passed'] is False,
            'published results and refit candidate must remain separate')
    pi = facts['raspberry_pi']
    require(pi['physical_device_tested'] is False and all(v is None for k, v in pi.items() if k not in {'status', 'physical_device_tested'}),
            'Pi measurements are unmeasured placeholders')
    for source in facts['sources']:
        if source.get('sha256') and source.get('local_file'):
            path = bmo / 'evidence' / source['local_file']
            require(path.is_file() and sha(path) == source['sha256'], f'fact source receipt hash mismatch: {source["local_file"]}')
    return facts


def validate_research_details(bmo):
    """Require substantive, cited machine-readable dataset and benchmark panels."""
    for filename, keys in [('dataset-recipe.json', ('dataset', 'splits', 'source_composition', 'generation_sources', 'citations', 'unknowns')),
                           ('benchmark.json', ('scope', 'accuracy', 'slot_cells', 'latency', 'comparison', 'caveats'))]:
        record = read_json(bmo / 'data' / filename)
        require(record.get('schema_version') == 1, f'invalid detail schema: {filename}')
        require(all(record.get(key) is not None for key in keys), f'missing research detail fields: {filename}')
        require(bool(record.get('sources')), f'citation sources required: {filename}')
        ids = set()
        for source in record['sources']:
            require(all(source.get(key) for key in ('id', 'title')), f'incomplete citation in {filename}')
            require(source['id'] not in ids, f'duplicate citation ID in {filename}')
            ids.add(source['id'])
            require(source.get('url') or source.get('site_path'), f'citation target required in {filename}')
            if source.get('url'):
                parts = urlsplit(source['url'])
                require(parts.scheme == 'https' and bool(parts.netloc), f'HTTPS citation required in {filename}')
            if source.get('site_path'):
                check_reference(bmo.parent, bmo / 'data' / filename, source['site_path'])

        def check_citation_ids(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == 'sources':
                        require(isinstance(child, list) and bool(child) and all(x in ids for x in child),
                                f'unknown or empty citation IDs in {filename}')
                    else:
                        check_citation_ids(child)
            elif isinstance(value, list):
                for child in value:
                    check_citation_ids(child)

        check_citation_ids({k: v for k, v in record.items() if k != 'sources'})
        baseline = record['baseline']
        require(baseline['model_sha256'] == MODEL_SHA256 and baseline['selected_epoch'] == 38
                and baseline['qualified'] is False and baseline['actions_enabled'] is False,
                f'published benchmark identity/qualification mismatch: {filename}')
        require(bool(record.get('generation_sources', True)), 'generation evidence must be present')
        if filename == 'dataset-recipe.json':
            require(record['dataset']['license_name'] == 'source-specific-research-terms', 'dataset detail must retain source-specific terms')
            require(record['dataset']['repository_url'] == 'https://huggingface.co/datasets/justinruaya123/ai231-smart-speaker',
                    'explicit Hugging Face dataset repository required')
            distillation = record['distillation']
            require(distillation['used'] is False and distillation['teacher_model'] is None,
                    'frozen BMO fit must not acquire a distillation teacher')
            for generator in record['generation_sources']:
                require(bool(generator['reference_links']) and bool(generator['citation_summary']),
                        'each TTS source requires explicit references')
                for reference in generator['reference_links']:
                    parsed = urlsplit(reference['url'])
                    require(parsed.scheme == 'https' and bool(parsed.netloc), 'HTTPS TTS source reference required')

            require(record['dataset']['audio_hours'] is None and record['dataset']['unique_human_speakers'] is None,
                    'unaudited dataset detail counts must remain null')
            expected = {'train': (25958, 1889, 24069), 'original-dev': (1692, 256, 1436),
                        'new-dev': (440, 0, 440), 'test': (440, 0, 440), 'official': (202, 96, 106)}
            require({r['id']: (r['total'], r['human'], r['synthetic']) for r in record['splits']} == expected,
                    'dataset detail split counts mismatch')
            for split, (total, human, synthetic) in expected.items():
                rows = [row for row in record['source_composition'] if row['split'] == split]
                require(sum(row['rows'] for row in rows) == total
                        and sum(row['rows'] for row in rows if row['kind'] == 'human') == human
                        and sum(row['rows'] for row in rows if row['kind'] == 'synthetic') == synthetic,
                        f'dataset source composition mismatch: {split}')
            generation = {row['id']: row for row in record['generation_sources']}
            require({'edge_speech', 'system_speech', 'legacy_qwen', 'martinnavs'}.issubset(generation),
                    'recorded generation sources required')
            require(generation['edge_speech']['total_rows'] == 2200
                    and generation['system_speech']['total_rows'] == 1764
                    and generation['legacy_qwen']['counts'] == {'train': 0, 'original_dev': 184, 'test': 0},
                    'generation evidence counts mismatch')
        else:
            for row in record['accuracy'] + record['slot_cells']:
                require(row['correct'] + row['wrong_accepted'] + row['rejected_supported'] == row['supported'],
                        f'benchmark supported outcomes do not add up: {row["id"]}')
                require(0 <= row['negative_false_accepts'] <= row['negative_rows'], 'invalid benchmark negative count')
                accuracy = row['correct'] / row['supported'] if row['supported'] else None
                require(row['accuracy'] == accuracy, f'benchmark accuracy denominator mismatch: {row["id"]}')
                rate = row['negative_false_accepts'] / row['negative_rows'] if row['negative_rows'] else None
                require(row['negative_false_accept_rate'] == rate, 'benchmark false accept denominator mismatch')
            pi = record['raspberry_pi']
            require(pi['qualification'] is False and all(value is None for key, value in pi.items() if key not in {'status', 'qualification'}),
                    'benchmark Pi measurements must remain unmeasured')
            require(record['latency']['raspberry_pi'] is False and record['latency']['re_measured_in_this_deck'] is False,
                    'historical desktop timing must not become a Pi/new measurement')
            refit = record['rtx4080_reproduction']
            require(refit['selected_epoch'] == 36 and refit['published_epoch'] == 38
                    and refit['exact_fit_comparison_passed'] is False and refit['candidate_test_evaluated'] is False,
                    'benchmark must separate published results from refit candidate')


def validate_pi5_replay(bmo, html):
    """Audit new actual-host evidence without promoting historical null placeholders."""
    record = read_json(bmo / 'data/pi5-replay.json')
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(record['schema_version'] == 1 and record['evidence_kind'] == 'physical_pi5_recorded_command_replay', 'Pi 5 replay schema')
    require(record['hardware']['physical_pi_measured'] is True and 'Raspberry Pi 5' in record['hardware']['device_model'], 'Pi 5 must identify actual hardware')
    scope = record['scope']
    require(scope['physical_pi_measured'] is True and all(scope[key] is False for key in
            ('physical_microphone_measured', 'wake_tested', 'owner_gating_tested', 'providers_called', 'audio_played', 'actions_enabled', 'qualified', 'pi4_measured', 'power_measured')),
            'Pi 5 replay cannot claim live/qualified/actions or Pi 4 measurements')
    require(scope['training_updates'] == 0 and scope['live_microphone_accuracy'] is None, 'Pi 5 live/training scope')
    refs = record['references']
    require(hashlib.sha256(canonical(refs).encode()).hexdigest() == '11a28f4888185ece5c3d14b09d1d8310007fdb79e4cbf14c81c0b0be001d7e09', 'Pi 5 original reference projection changed')
    require(len(refs) == 196 and len({row['id'] for row in refs}) == 196, 'Pi 5 recording identities')
    require(hashlib.sha256(canonical(record['catalogs']).encode()).hexdigest() == '3f55b6d38beb053bf7d52299f9f4ff9f5df5d7847bcb2bd2c0edf2bb24328a74', 'Pi 5 immutable catalogs changed')
    catalog = {key: {canonical(command) for command in values} for key, values in record['catalogs'].items()}
    wording_body = (bmo / 'wordings.js').read_text(encoding='utf-8')
    wording_rows = json.loads(re.fullmatch(r'\s*window\.BMO_WORDINGS\s*=\s*(\[.*\]);?\s*', wording_body, re.S)[1])
    require(catalog['baseline_a'] == {canonical(row['command']) for row in wording_rows} | {'null'}, 'Pi 5 catalog differs from frozen A wordings')
    common = catalog['baseline_a'] & catalog['epoch44']
    full = [i for i, row in enumerate(refs) if row['scored']]
    shared = [i for i in full if canonical(refs[i]['expected']) in common]
    require(len(full) == 194 and len(shared) == 167 and record['reference_contract']['outside_baseline_inventory_n'] == 27, 'Pi 5 denominator/coverage mismatch')

    def check_score(decisions, indices, score):
        positive = [i for i in indices if refs[i]['expected'] is not None]
        negative = [i for i in indices if refs[i]['expected'] is None]
        exact_positive = sum(decisions[i] == refs[i]['expected'] for i in positive)
        false = sum(decisions[i] is not None for i in negative)
        exact = exact_positive + len(negative) - false
        wrong = sum(decisions[i] is not None and decisions[i] != refs[i]['expected'] for i in positive)
        rejected = sum(decisions[i] is None for i in positive)
        require((score['scored_n'], score['positive_n'], score['negative_n'], score['exact_positive_n'], score['wrong_positive_n'], score['rejected_positive_n'], score['false_accept_n'], score['exact_n']) ==
                (len(indices), len(positive), len(negative), exact_positive, wrong, rejected, false, exact), 'Pi 5 score counts/denominator mismatch')
        require(score['complete_command_accuracy'] == exact / len(indices), 'Pi 5 score percentage mismatch')

    def quantile(values, fraction):
        values = sorted(values)
        index = (len(values) - 1) * fraction
        low = int(index)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)

    require(len(record['runs']) == 2, 'two preserved Pi 5 repeats required')
    for run in record['runs']:
        require(set(run['cases']) == {'baseline-a-mnn', 'baseline-a-onnx', 'epoch44-onnx'}, 'Pi 5 configuration matrix')
        parity = run['baseline_backend_parity']
        require(parity['decision_mismatch_trials'] == 0 and parity['logits_within_tolerance'] is True and parity['sources_with_logits'] == 196, 'Pi 5 native parity failed')
        for name, case in run['cases'].items():
            metrics = case['metrics']
            require(metrics['source_n'] == 196 and metrics['observation_n'] == 588 and metrics['threads'] == 2 and metrics['repeats'] == 3,
                    'Pi 5 timing/recording denominators')
            require(metrics['actions_enabled'] is False and metrics['qualified'] is False and metrics['physical_pi_measured'] is True, 'Pi 5 measurement/qualification scope')
            expected_hash = MODEL_SHA256 if name.startswith('baseline-a') else 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'
            require(metrics['model_sha256'] == expected_hash, 'Pi 5 model identity')
            first = case['first_pass']
            require(len(first) == 196 and all(row['raw_available'] for row in first), 'Pi 5 first-pass evidence')
            for mode, decisions in [('accepted', [row['output']['command'] for row in first]), ('raw', [row['raw'] for row in first])]:
                check_score(decisions, full, metrics[mode])
                check_score(decisions, shared, case['shared_inventory'][mode])
            times = case['end_to_end_ms']
            require(len(times) == 588 and all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times), 'Pi 5 finite timing observations')
            for key, fraction in [('p50', .5), ('p95', .95)]:
                require(math.isclose(metrics['latency_ms']['end_to_end'][key], quantile(times, fraction), rel_tol=1e-12, abs_tol=1e-9), 'Pi 5 timing quantile mismatch')
            require(case['offline_guard']['network'] == 0 and case['offline_guard']['setup'] == 0, 'Pi 5 offline guard')
        require(run['cases']['baseline-a-onnx']['first_pass'] == run['cases']['baseline-a-mnn']['first_pass'], 'Pi 5 typed decision parity mismatch')
    latest = next(run for run in record['runs'] if run['id'] == record['current_run_id'])['cases']['baseline-a-mnn']
    metrics, score = latest['metrics'], latest['shared_inventory']['accepted']
    holdout = metrics['accepted']
    latency = metrics['latency_ms']['end_to_end']
    displayed = {'latency': f"{latency['p50']:.2f} / {latency['p95']:.2f} ms", 'rtf': f"{metrics['rtf']['p50']:.5f}",
                 'rss': f"{metrics['peak_process_rss_mib']:.2f} MiB", 'shared': f"{score['exact_n']} / {score['scored_n']} · {100*score['complete_command_accuracy']:.2f}%",
                 'holdout': f"{holdout['exact_n']} / {holdout['scored_n']} · {100*holdout['complete_command_accuracy']:.2f}%"}
    for key, text in displayed.items():
        values = re.findall(r'data-pi5="' + key + r'">([^<]+)<', html)
        require(values and all(value == text for value in values), 'Pi 5 visible metric differs from receipt: ' + key)
    return {'device': 'Raspberry Pi 5', 'recorded_replay': True, 'qualified': False, 'latest_run': record['current_run_id']}


def _validate_bmo(root):
    root = Path(root)
    site = root / 'docs/site'
    bmo = site / 'bmo'
    require((site / 'index.html').is_file(), 'missing BMO index')
    require(bmo.is_dir(), 'missing BMO asset directory')
    require(not bmo.is_symlink() and not (site / 'index.html').is_symlink(), 'site cannot publish a symlink')
    require((bmo / 'vendor/THREE-LICENSE.txt').is_file(), 'vendored Three.js license must be preserved')
    html = (site / 'index.html').read_text(encoding='utf-8')
    parser = validate_references(site, bmo, html)
    require('Pi 4, power and live wake/owner acceptance remain unmeasured' in html and 'published' in html.lower() and 'refit differs and has no test score' in html,
            'visible Pi/published-versus-refit limitations required')
    wordings, commands = validate_wordings(bmo)
    validate_trace(bmo)
    history = validate_history(bmo)
    validate_facts(bmo)
    validate_research_details(bmo)
    pi5 = validate_pi5_replay(bmo, html)
    files = inventory(bmo)
    require(read_json(bmo / 'manifest.json') == {'format': 'bmo-static-assets-v1', 'files': files}, 'BMO asset manifest is stale')
    return {'slides': len(parser.slides), 'wordings': wordings, 'commands': commands,
            'training_epochs': history['epoch_count'], 'published_epoch': 38, 'refit_epoch': 36,
            'assets': len(files), 'pi': None, 'pi5_recorded_replay': pi5, 'scope': 'static documentation; no acoustic qualification'}


def validate_bmo(root=ROOT):
    """Return a static-evidence summary; malformed/missing content fails closed."""
    try:
        return _validate_bmo(root)
    except (KeyError, TypeError, OSError, wave.Error) as exc:
        raise ValueError(f'invalid or missing BMO evidence: {exc}') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    try:
        print(json.dumps(validate_bmo(), indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f'BMO validation failed: {exc}\n')


if __name__ == '__main__':
    main()
