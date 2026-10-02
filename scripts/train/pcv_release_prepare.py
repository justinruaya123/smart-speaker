"""Fresh HF release preparation with frozen metadata-only annotation corrections.

Old acoustic features may be reused by PCM identity; normalization and labels may
not. No recognizer, teacher output, generated audio or test-driven label design.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from einops import reduce

from scripts.runtime.pcv_schema import (
    INTENTS, DEFAULT_CUSTOM, baseline_cases, canonical, compile_inventory,
    validate_command,
)
from scripts.train.pcv_public_data import sha, read_rows, _words, audit_split_rows
from scripts.train.pcv_three_engine import immutable
from scripts.train.pcv_updated_prepare import (
    annotate as original_annotate, assign_split, pair_identity, spoken,
)
from scripts.test.pcv_public_qualification import verify_bindings
from vcm.frontend import SPEC, features

HERE = ROOT / 'experiments/pcv-release-train-v1'
CACHE = ROOT / '.cache/pcv-release-train-v1'
PREPARED = CACHE / 'prepared'
SOURCE = ROOT / '.cache/pcv-source-refresh-v1/audited-rows.jsonl'
SOURCE_SHA256 = '7017a509ebd480e9a24e81c38a4f8d26bf4d71047c7b45aa6a71ca8eac10ae52'
REVISION = '25111444af3adff7588d86ab27c304bb080895cf'
ACQUISITION = ROOT / 'experiments/pcv-source-refresh-v1/acquisition-complete.json'
PROPOSAL = HERE / 'training-label-review.json'
PROPOSAL_SHA256 = 'e1c0fba7d11b87983aa0499669ec2f10c3949e3fdf9304c4c32484f69490a80d'
PROMPTS = ROOT / 'experiments/pcv-b1-three-engine-v1/prompts.json'
OLD_HERE = ROOT / 'experiments/pcv-updated-voice-v1'
SPLITS = ('train', 'dev', 'calibration', 'test', 'holdout')

# These were proposed from fitting/development metadata but do not unambiguously
# distinguish changing color from selecting an already colored light, or playback
# from changing volume. Leave the frozen parent annotation in force.
REJECTED_PROPOSALS = {
    'please switch to the green light': 'May select an existing colored light rather than set its color.',
    'switch to red lights': 'May select existing colored lights rather than set their color.',
    'olly switch to red colored lights': 'May select existing colored lights rather than set their color.',
    'blue lights please': 'Noun phrase may request activating existing colored lights.',
    'give me red lights': 'May request activating existing colored lights.',
    'i should like to see red lights': 'Desire statement does not resolve activation versus color change.',
    'play it louder': 'May request playback and a volume change; no combined command is represented.',
}


@lru_cache(maxsize=1)
def annotation_policy():
    verify_bindings(ROOT, {str(PROPOSAL.relative_to(ROOT)): PROPOSAL_SHA256})
    proposal = json.loads(PROPOSAL.read_text(encoding='utf8'))
    if proposal['provenance']['source_sha256'] != SOURCE_SHA256:
        raise ValueError('Annotation proposal belongs to another source')
    supported = {}
    decisions = []
    for evidence in proposal['supported_correction_evidence']:
        text = evidence['text']
        expected = evidence['expected']
        validate_command(expected)
        approved = text not in REJECTED_PROPOSALS
        if approved:
            supported[text] = expected
        decisions.append({
            'text': text, 'approved': approved, 'expected': expected,
            'reason': REJECTED_PROPOSALS.get(text, evidence['rationale']),
            'review_source_ids': evidence['source_ids'],
        })
    return {
        'version': 'pcv-release-train-v1-annotation-1',
        'supported': supported,
        'ambiguous': proposal['text_to_ambiguous_corrections'],
        'decisions': decisions,
        'proposal_sha256': PROPOSAL_SHA256,
        'source_sha256': SOURCE_SHA256,
        'rules': 'Exact normalized-text corrections across every split; publisher out_of_scope and unintelligible flags take precedence; no generic rules added.',
        'evidence': 'Supplied transcript metadata review, not independent listening; no teacher or ASR generated ground truth.',
    }


def annotate(row, policy=None):
    result = original_annotate(row)
    # Preserve the publisher's explicit rejection and uncertainty boundaries.
    if (row.get('published_split') == 'numerals'
            or row.get('out_of_scope') or row.get('command') == 'OUT_OF_SCOPE'
            or 'not understood' in row.get('whisper_check', '')):
        return result
    policy = policy or annotation_policy()
    text = result['text']
    if text in policy['ambiguous']:
        return {**result, 'expected': None, 'reference_status': 'ambiguous',
                'reason': 'reviewed_unverified_or_incomplete_metadata'}
    if text in policy['supported']:
        expected = policy['supported'][text]
        coarse = {'VOLUME_UP': 'VOLUME', 'VOLUME_DOWN': 'VOLUME'}.get(row['command'], row['command'])
        return {**result, 'expected': expected, 'reference_status': 'supported',
                'reason': 'reviewed_exact_finite_metadata',
                'slot_metadata_conflict': result['slot_metadata_conflict'] or expected['intent'] != coarse}
    return result


def approved_definitions():
    definitions = json.loads(PROMPTS.read_text(encoding='utf8'))
    if len(definitions) != 415:
        raise ValueError('Approved phrase contract changed')
    return definitions


def catalog_definitions(rows, predefined):
    definitions = list(predefined)
    for row in rows:
        if (row['split'] == 'train' and row['training_eligible']
                and row['reference_status'] in ('supported', 'unsupported')):
            definitions.append({
                'id': 'HF-' + hashlib.sha256(row['text'].encode()).hexdigest()[:14],
                'text': row['text'], 'command': row['expected'], 'tier': 'hf_train',
            })
    return definitions


def fit_normalization(rows):
    means, squares, identities = [], [], []
    for row in rows:
        if (row['split'] != 'train' or row['synthetic'] or not row['training_eligible']
                or row['reference_status'] == 'auxiliary'):
            continue
        if sha(row['features']) != row['features_sha256']:
            raise ValueError('Normalization feature hash mismatch')
        x = np.load(row['features'], allow_pickle=False).astype('float64')
        if x.ndim != 2 or x.shape[0] != SPEC['bins'] or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Invalid normalization feature array')
        means.append(reduce(x, 'c t -> c', 'mean'))
        squares.append(reduce(x * x, 'c t -> c', 'mean'))
        identities.append(row['id'])
    if not means:
        raise ValueError('No eligible human training normalization rows')
    mean = reduce(means, 'n c -> c', 'mean')
    std = np.sqrt(np.maximum(reduce(squares, 'n c -> c', 'mean') - mean ** 2, 1e-6))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'frontend': SPEC,
            'human_training_utterances': len(identities), 'source_ids': identities,
            'weighting': 'equal weight per eligible human training utterance',
            'source_sha256': SOURCE_SHA256}


def reuse_feature(row, index):
    entry = index.get(row['pcm_sha256'])
    if entry is None:
        return None
    if entry['frontend'] != SPEC:
        raise ValueError('Raw frontend specification differs')
    path = Path(entry['features'])
    if sha(path) != entry['features_sha256']:
        raise ValueError('Raw feature hash mismatch')
    x = np.load(path, mmap_mode='r', allow_pickle=False)
    if x.ndim != 2 or x.shape[0] != SPEC['bins'] or x.shape[1] != entry['frames']:
        raise ValueError('Raw feature shape differs')
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError('Invalid raw feature values')
    return {key: entry[key] for key in ('features', 'features_sha256', 'frames')}


def old_feature_index():
    freeze_path = OLD_HERE / 'data-freeze.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf8'))
    normalization_path = ROOT / '.cache/pcv-updated-voice-v1/prepared/normalization.json'
    normalization_key = next(k for k in freeze['bindings'] if k.replace('\\', '/').endswith('/normalization.json'))
    verify_bindings(ROOT, {normalization_key: freeze['bindings'][normalization_key]})
    if json.loads(normalization_path.read_text(encoding='utf8'))['frontend'] != SPEC:
        raise ValueError('Old features have another frontend specification')
    training_path = OLD_HERE / 'training-protocol.json'
    training = json.loads(training_path.read_text(encoding='utf8'))
    frontend_hash = next(v for k, v in training['bindings'].items() if k.replace('\\', '/') == 'vcm/frontend.py')
    verify_bindings(ROOT, {'vcm/frontend.py': frontend_hash})
    index = {}
    bindings = {str(freeze_path.relative_to(ROOT)): sha(freeze_path),
                str(training_path.relative_to(ROOT)): sha(training_path),
                normalization_key: freeze['bindings'][normalization_key]}
    for spec in freeze['manifests'].values():
        verify_bindings(ROOT, {spec['path']: spec['sha256']})
        bindings[spec['path']] = spec['sha256']
        # Deliberately project only acoustic identity fields. Old text, labels,
        # predictions and normalization values never enter the new preparation.
        with (ROOT / spec['path']).open(encoding='utf8') as stream:
            for line in stream:
                source = json.loads(line)
                item = {k: source[k] for k in ('pcm_sha256', 'features', 'features_sha256', 'frames')}
                item['frontend'] = SPEC
                previous = index.get(item['pcm_sha256'])
                if previous and previous['features_sha256'] != item['features_sha256']:
                    raise ValueError('One PCM has conflicting raw feature hashes')
                index[item['pcm_sha256']] = item
    return index, bindings


def attach_features(row, receipt):
    row = {**row, **receipt}
    needed = len(row['phones']) + sum(a == b for a, b in zip(row['phones'], row['phones'][1:]))
    if row['frames'] < needed:
        row['training_eligible'] = False
        row['exclusion_reason'] = row['exclusion_reason'] or 'insufficient_CTC_frames'
    return row


def audit_annotations(rows):
    result = audit_split_rows(rows)
    commands = defaultdict(set)
    for row in rows:
        if row['reference_status'] in ('supported', 'unsupported'):
            commands[row['pcm_sha256']].add(canonical(row['expected']))
    if any(len(values) > 1 for values in commands.values()):
        raise ValueError('Conflicting typed references for identical PCM')
    result['within_split_pcm_duplicate_rows'] = len(rows) - len({r['pcm_sha256'] for r in rows})
    result['within_split_pair_duplicate_rows'] = len(rows) - len({r['pair'] for r in rows})
    result['conflicting_pcm_references'] = 0
    return result


def coverage(rows, definitions):
    result = {}
    base = baseline_cases()
    for split in SPLITS:
        for origin in ('human', 'synthetic'):
            subset = [r for r in rows if r['split'] == split and r['synthetic'] == (origin == 'synthetic')]
            result[split + '/' + origin] = {
                'rows': len(subset),
                'exact_baseline_cells': len({r['phrase_id'] for r in subset if r['phrase_id']}),
                'positive_intents': dict(Counter(r['intent'] for r in subset if r['expected'])),
                'baseline_denominators': {b['id']: sum(r['phrase_id'] == b['id'] for r in subset) for b in base},
                'approved_phrase_denominators': {
                    d['id']: sum(r['text'] == spoken(d['text']) and r['reference_status'] != 'ambiguous'
                                 and canonical(r['expected']) == canonical(d['command']) for r in subset)
                    for d in definitions},
            }
    return result


def prepare():
    import pyarrow.parquet as pq
    import soundfile as sf

    if (HERE / 'data-freeze.json').exists() or PREPARED.exists():
        raise FileExistsError('Preserve existing prepared release; use a new experiment version')
    verify_bindings(ROOT, {str(SOURCE.relative_to(ROOT)): SOURCE_SHA256})
    acquired = json.loads(ACQUISITION.read_text(encoding='utf8'))
    if acquired['revision'] != REVISION:
        raise ValueError('Acquisition revision mismatch')
    verify_bindings(ROOT, {r['path']: r['sha256'] for r in acquired['files']})
    policy = annotation_policy()
    immutable(HERE / 'annotation-approval.json', policy)
    words = _words()
    definitions = approved_definitions()
    base_ids = {spoken(r['text']): r['id'] for r in baseline_cases()}
    rows, exclusions, raw_integrity = [], [], []
    missing_words = Counter()
    published_counts = Counter()
    for source in read_rows(SOURCE):
        item = annotate(source, policy)
        split = assign_split(source)
        published_counts[source['published_split']] += 1
        row = {**source, **item, 'split': split,
               'speaker': str(source['source']) + '/' + str(source['speaker_id']),
               'pair': pair_identity(source), 'synthetic': bool(source['is_synthetic']),
               'phrase_id': base_ids.get(item['text']),
               'intent': item['expected']['intent'] if item['expected'] else 'OOV'}
        raw_integrity.append({k: row[k] for k in ('id', 'speaker', 'pair', 'pcm_sha256', 'split')})
        flags = source['waveform_flags']
        bad_audio = bool(flags) and not (source['transcript_source'] == 'silence' and set(flags) <= {'low_rms'})
        reason = 'waveform_quality' if bad_audio else None
        if item['reference_status'] == 'ambiguous':
            reason = 'ambiguous_reference'
        phones, missing = [], []
        tokens = item['text'].split()
        from vcm.catalog import PHONES
        for i, token in enumerate(tokens):
            ps = ['EY'] if token == 'a' and tokens[i + 1:i + 2] == ['m'] else words.get(token)
            if ps:
                phones.extend(PHONES.index(p) + 1 for p in ps)
            else:
                missing.append(token)
        if missing:
            reason = reason or 'missing_pronunciation'
            # Held-out lexical diagnostics are not emitted for development.
            if split == 'train':
                missing_words.update(missing)
        row.update(phones=phones,
                   intent_index=INTENTS.index(row['intent']) + 1 if row['expected'] else 0,
                   target_index=int(row['expected']['slots']['light_id'] != 'light.all')
                   if row['expected'] and 'light_id' in row['expected']['slots'] else -100,
                   training_eligible=reason is None, exclusion_reason=reason)
        if row['reference_status'] == 'auxiliary':
            row['intent_index'] = -100
        if reason:
            exclusions.append({'id': row['id'], 'split': split, 'reason': reason,
                               'reference_status': row['reference_status']})
        if split in ('test', 'holdout') or not reason:
            rows.append(row)
    # Audit even excluded records so exclusions cannot conceal speaker/PCM leaks.
    full_integrity = audit_split_rows(raw_integrity)
    integrity = audit_annotations(rows)
    print('ANNOTATED', len(rows), 'retained of', len(raw_integrity), flush=True)
    feature_index, reuse_bindings = old_feature_index()
    receipt_by_pcm = {}
    needed = sorted({r['pcm_sha256'] for r in rows})
    for i, pcm in enumerate(needed):
        receipt = reuse_feature({'pcm_sha256': pcm}, feature_index)
        if receipt:
            receipt_by_pcm[pcm] = receipt
        if i % 10000 == 0:
            print('VERIFIED_FEATURE_IDENTITIES', i, 'of', len(needed), flush=True)
    reused_count = len(receipt_by_pcm)
    PREPARED.mkdir(parents=True)
    missing_by_file = defaultdict(dict)
    for row in rows:
        if row['pcm_sha256'] not in receipt_by_pcm:
            missing_by_file[str(Path(row['parquet']).resolve())][row['parquet_row']] = row
    (PREPARED / 'features').mkdir()

    def extract(pair):
        source, payload = pair
        pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or hashlib.sha256(pcm.tobytes()).hexdigest() != source['pcm_sha256']:
            raise ValueError('Source PCM identity changed: ' + source['id'])
        value = features(pcm)
        path = PREPARED / 'features' / (source['pcm_sha256'] + '.npy')
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        return source['pcm_sha256'], {'features': str(path), 'features_sha256': sha(path), 'frames': value.shape[1]}

    with ThreadPoolExecutor(max_workers=6) as pool:
        queued_pcm = set()
        for file in acquired['files']:
            path = (ROOT / file['path']).resolve()
            wanted = missing_by_file.get(str(path), {})
            if not wanted:
                continue
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=['audio']):
                pairs = []
                for value in batch.to_pylist():
                    source = wanted.get(index)
                    index += 1
                    if source and source['pcm_sha256'] not in queued_pcm:
                        pairs.append((source, value['audio']['bytes']))
                        queued_pcm.add(source['pcm_sha256'])
                receipt_by_pcm.update(pool.map(extract, pairs))
            print('EXTRACTED_MISSING_FEATURES', path.name, len(receipt_by_pcm) - reused_count, flush=True)
    if set(needed) != set(receipt_by_pcm):
        raise ValueError('Missing source audio feature receipts')
    outputs = [attach_features(r, receipt_by_pcm[r['pcm_sha256']]) for r in rows]
    normalization = fit_normalization(outputs)
    immutable(PREPARED / 'normalization.json', normalization)
    catalog = compile_inventory(catalog_definitions(outputs, definitions), words)
    signatures = {(spoken(r.get('spoken_text', r['text'])), canonical(r['command'])) for r in catalog['entries']}
    if any((spoken(r.get('spoken_text', r['text'])), canonical(r['command'])) not in signatures for r in definitions):
        raise ValueError('Approved finite phrase coverage missing')
    for split in ('test', 'holdout'):
        if sum(r['split'] == split for r in outputs) != published_counts[split]:
            raise ValueError('Published held-out membership changed')
    manifests = {}
    for split in SPLITS:
        subset = [r for r in outputs if r['split'] == split]
        path = PREPARED / (split + '.jsonl')
        with path.open('x', encoding='utf8', newline='\n') as stream:
            for row in subset:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False) + '\n')
        manifests[split] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'rows': len(subset)}
    immutable(HERE / 'catalog.json', catalog)
    immutable(HERE / 'custom-slots.json', DEFAULT_CUSTOM)
    immutable(HERE / 'baseline.json', baseline_cases())
    immutable(HERE / 'annotation-exclusions.json', exclusions)
    coverage_data = coverage(outputs, definitions)
    immutable(HERE / 'coverage.json', coverage_data)
    counts = Counter((r['split'], 'synthetic' if r['synthetic'] else 'human', r['reference_status']) for r in outputs)
    summary = {
        'source_revision': REVISION, 'source_only': True, 'fallback_used': False,
        'distillation_used': False, 'test_inference_accessed': False,
        'integrity': integrity, 'full_source_integrity': full_integrity,
        'counts': {'/'.join(k): v for k, v in counts.items()},
        'exclusions': dict(Counter(r['exclusion_reason'] for r in outputs if r['exclusion_reason'])) ,
        'excluded_before_features': dict(Counter(r['reason'] for r in exclusions)),
        'missing_pronunciations_train_only': dict(missing_words),
        'slot_metadata_conflicts': sum(r['slot_metadata_conflict'] for r in outputs),
        'coverage': coverage_data, 'catalog_entries': len(catalog['entries']),
        'approved_definition_count': len(definitions), 'approved_definition_missing': 0,
        'catalog_built_from': 'predeclared 415 definitions and eligible training text only',
        'reused_raw_feature_identities': reused_count,
        'new_raw_feature_identities': len(receipt_by_pcm) - reused_count,
        'features_reused_by': 'exact PCM hash, verified feature file hash, frozen frontend SPEC and implementation hash; shared files referenced without copying',
        'normalization': {'path': str((PREPARED / 'normalization.json').relative_to(ROOT)),
                          'human_training_utterances': normalization['human_training_utterances'],
                          'fresh_fit': True, 'old_statistics_used': False},
        'annotation_review': {'proposed_positive_texts': len(policy['decisions']),
                              'approved_positive_texts': len(policy['supported']),
                              'rejected_positive_texts': len(REJECTED_PROPOSALS),
                              'ambiguous_texts': len(policy['ambiguous'])},
        'test_and_holdout_membership': 'all published rows retained; uncertain references remain separate',
        'heldout_design_exposure': 'No individual held-out transcripts or predictions inspected for design. A source audit header accidentally displayed aggregate holdout baseline bucket names; these were not used for label or catalog decisions.',
        'segmentation': 'Publisher utterance boundaries preserved; no cropping, splicing, synthesis, ASR or independent listening claim',
    }
    immutable(HERE / 'data-audit.json', summary)
    paths = [Path(__file__), ROOT / 'scripts/test/test_pcv_release_prepare.py',
             ROOT / 'scripts/train/pcv_updated_prepare.py', ROOT / 'scripts/train/pcv_public_data.py',
             ROOT / 'scripts/runtime/pcv_schema.py', ROOT / 'vcm/frontend.py',
             ROOT / 'data/raw/cmudict/cmudict.dict', ROOT / 'experiments/pcv-synthetic-v2/lexicon.json',
             PROPOSAL, SOURCE, ACQUISITION, PROMPTS,
             HERE / 'annotation-approval.json', HERE / 'catalog.json', HERE / 'baseline.json',
             HERE / 'custom-slots.json', HERE / 'annotation-exclusions.json', HERE / 'coverage.json',
             HERE / 'data-audit.json', HERE / 'preparation-protocol.json',
             HERE / 'preparation-verification.json', PREPARED / 'normalization.json']
    freeze = {'manifests': manifests,
              'bindings': {str(p.relative_to(ROOT)): sha(p) for p in paths},
              'raw_feature_reuse_bindings': reuse_bindings,
              'source_revision': REVISION, 'source_sha256': SOURCE_SHA256,
              'frontend': SPEC, 'acquisition_sha256': sha(ACQUISITION),
              'annotation_version': policy['version'], 'test_inference_accessed': False}
    immutable(HERE / 'data-freeze.json', freeze)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('coverage', 'missing_pronunciations_train_only')}, indent=2), flush=True)
    return freeze


if __name__ == '__main__':
    prepare()
