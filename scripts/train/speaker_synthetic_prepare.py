"""Prepare explicitly synthetic-assisted research without changing human gates."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl, write_new
from scripts.train.speaker_demo_v2_data import _load_inputs, _check_rows, _copy_new
from scripts.train.speaker_synthetic_data import (
    ROOT, HERE, CACHE, _path, _relative, _verify, audit_generated, audit_anchors,
)
from scripts.train.pcv_updated_prepare import spoken
from vcm.frontend import SPEC, features


def _receipt(path, **extra):
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path), **extra}


def _phones(row, words):
    text = row.get('spoken_text') or row.get('publisher_metadata', {}).get('transcript', '')
    text = spoken(text)
    labels, unknown = [], []
    tokens = text.split()
    for i, token in enumerate(tokens):
        value = [13] if token == 'a' and tokens[i + 1:i + 2] == ['m'] else words.get(token)
        if value is None:
            unknown.append(token)
        else:
            labels.extend(value)
    return text, labels, sorted(set(unknown))


def prepare_run(protocol: Path, installed: Path, generated: Path, output: Path, *, workers=8):
    protocol, output, generated = Path(protocol), Path(output).resolve(), Path(generated).resolve()
    if output.exists():
        raise FileExistsError('Use a new research preparation directory; preserve earlier evidence')
    cfg = read_json(protocol)
    _verify(cfg['bindings'])
    parent = _path(cfg['parent_protocol'])
    (_, audit, paths, views, _, definitions, _, _, source_bindings) = _load_inputs(parent, installed)
    complete_path = generated / 'generation-complete.json'
    complete = read_json(complete_path)
    if not complete['complete'] or complete['protocol_sha256'] != file_digest(protocol):
        raise ValueError('Generation is incomplete or belongs to a different protocol')
    manifest, anchors_path = _path(complete['manifest']), _path(complete['anchors'])
    if not manifest.resolve().is_relative_to(generated) or not anchors_path.resolve().is_relative_to(generated):
        raise ValueError('Generated receipt points outside its declared output')
    _verify({str(manifest): complete['manifest_sha256'], str(anchors_path): complete['anchors_sha256']})
    plan, voices = read_json(_path(cfg['generation_plan'])), read_json(_path(cfg['voices_file']))
    anchors, new = read_json(anchors_path), read_jsonl(manifest)
    if 'synthetic/diagnostic' in paths:
        diagnostic = read_jsonl(paths['synthetic/diagnostic'])
    else:
        diagnostic = []
    views['synthetic/dev'] = [r for r in diagnostic if r['split'] == 'dev']
    views['synthetic/calibration'] = [r for r in diagnostic if r['split'] == 'calibration']
    if len(diagnostic) != len(views['synthetic/dev']) + len(views['synthetic/calibration']):
        raise ValueError('Unexpected split in existing synthetic diagnostic view')
    existing = [r for rows in views.values() for r in rows]
    generation_audit = audit_generated(new, plan, existing)
    if not generation_audit['complete']:
        raise ValueError('Missing generated cells cannot be silently omitted')
    audit_anchors(anchors, voices)
    anchor_hashes = {r['id']: r['pcm_sha256'] for r in anchors}
    for row in [*anchors, *new]:
        read_audio(row, ROOT)
    for row in new:
        if row['anchor_pcm_sha256'] != anchor_hashes[row['voice_id']]:
            raise ValueError('Generated record has the wrong voice anchor')
        views['synthetic/' + row['split']].append(row)
    _check_rows(views)
    fitting_views = ('human/train', 'human/dev', 'human/calibration',
                     'synthetic/train', 'synthetic/dev', 'synthetic/calibration')
    catalog_path = _path(cfg['catalog_file'])
    catalog = read_json(catalog_path)
    commands = {canonical(r['command']) for r in catalog['entries'] if r['command'] is not None}
    outside = {}
    for name in fitting_views:
        unexpected = set()
        for row in views[name]:
            if (row['kind'] != name.split('/')[0] or row['role'] != 'command'
                    or row['qa_eligible'] is not True or row['audio_flags']
                    or row['reference_status'] not in {'supported', 'unsupported'}
                    or row['metric_eligible'] is not True):
                raise ValueError('Ineligible or mislabelled research fitting record')
            if (row['command'] is None) != (row['reference_status'] == 'unsupported'):
                raise ValueError('Ambiguous reference cannot become UNKNOWN')
            if row['command'] is not None:
                validate_command(row['command'])
                if canonical(row['command']) not in commands:
                    unexpected.add(canonical(row['command']))
        outside[name] = [json.loads(v) for v in sorted(unexpected)]
    if any(outside.values()):
        raise ValueError('Research catalog has unsupported typed supervision: ' + canonical(outside))
    coverage = {name: {'rows': len(views[name]),
        'intents': dict(Counter(r['command']['intent'] if r['command'] else 'UNKNOWN' for r in views[name])),
        'declared_speakers': len({r['speaker_id'] for r in views[name]})} for name in fitting_views}
    human_missing = {name: sorted(set(INTENTS) - coverage[name]['intents'].keys())
                     for name in fitting_views if name.startswith('human/')}
    for split in ('train', 'dev', 'calibration'):
        combined = set(coverage['human/' + split]['intents']) | set(coverage['synthetic/' + split]['intents'])
        if not set([*INTENTS, 'UNKNOWN']) <= combined:
            raise ValueError('Missing combined research intents or true UNKNOWN in ' + split)
    output.mkdir(parents=True, exist_ok=False)
    implementation = {str(p.relative_to(ROOT)): file_digest(p) for p in (
        Path(__file__), ROOT / 'scripts/train/speaker_synthetic_data.py',
        ROOT / 'scripts/train/speaker_demo_v2_data.py', ROOT / 'scripts/train/pcv_updated_prepare.py',
        ROOT / 'scripts/runtime/pcv_schema.py', ROOT / 'vcm/frontend.py',
        ROOT / 'vcm/catalog.py', ROOT / 'STT/alternatives/sources/cmudict.dict',
        ROOT / 'experiments/pcv-synthetic-v2/lexicon.json')}
    source_bindings.update({str(p.resolve()): file_digest(p) for p in (
        protocol, complete_path, manifest, anchors_path, catalog_path,
        _path(cfg['generation_plan']), _path(cfg['voices_file']))})
    write_new(output / 'preparation-protocol.json', {
        'source_bindings': source_bindings, 'implementation_bindings': implementation,
        'normalization': 'equal weight per eligible original human training utterance only',
        'features': 'human/synthetic train and dev only; no calibration/test/holdout features',
        'training_labels': 'existing source transcript metadata, deterministic number expansion and dictionary pronunciations; missing/too-short training labels excluded with reasons',
        'human_qualified': False})
    manifests = {}
    for name, rows in views.items():
        dest = output / 'views' / (name + '.jsonl')
        if name.startswith(('human/', 'evaluation/')):
            manifests[name] = _copy_new(paths[name], dest)
        else:
            write_new(dest, rows, jsonl=True)
            manifests[name] = _receipt(dest)
        manifests[name]['rows'] = len(rows)
    _copy_new(catalog_path, output / 'catalog.json')
    from vcm.catalog import dictionary, PHONES
    words = dictionary(ROOT / 'STT/alternatives/sources/cmudict.dict')
    words.update({word: [PHONES.index(p) + 1 for p in phones] for word, phones in
                  read_json(ROOT / 'experiments/pcv-synthetic-v2/lexicon.json')['words'].items()})
    feature_dir = output / 'features'
    feature_dir.mkdir()
    selected = [r for name in ('human/train', 'human/dev', 'synthetic/train', 'synthetic/dev') for r in views[name]]

    def extract(row):
        values = features(read_audio(row, ROOT))
        if not values.shape[1] or not np.isfinite(values).all():
            raise ValueError('Empty or nonfinite original features')
        target = feature_dir / (row['pcm_sha256'] + '.npy')
        with target.open('xb') as stream:
            np.save(stream, values, allow_pickle=False)
        text, labels, unknown = _phones(row, words)
        required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
        reason = ('missing_pronunciation_or_text' if not labels or unknown else
                  ('insufficient_ctc_frames' if (values.shape[1] + 1) // 2 < required else None))
        command = row['command']
        light = command['slots'].get('light_id') if command else None
        result = {**row, 'text': text, 'phones': labels, 'pronunciation_unknown': unknown,
                  'features': str(target), 'features_sha256': file_digest(target),
                  'synthetic': row['kind'] == 'synthetic', 'expected': command,
                  'intent': command['intent'] if command else 'OOV',
                  'intent_index': INTENTS.index(command['intent']) + 1 if command else 0,
                  'target_index': (0 if light == 'light.all' else 1) if light else -100,
                  'speaker': row['speaker_id'],
                  'training_eligible': row['split'] == 'train' and reason is None,
                  'training_exclusion': reason if row['split'] == 'train' else None}
        stats = None
        if row['kind'] == 'human' and result['training_eligible']:
            precise = values.astype('float64')
            stats = (precise.mean(axis=1), (precise ** 2).mean(axis=1), row['id'])
        return result, stats

    prepared, normalization_stats = [], []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, (row, stats) in enumerate(pool.map(extract, selected), 1):
            prepared.append(row)
            if stats is not None:
                normalization_stats.append(stats)
            if i % 2500 == 0:
                print('FEATURES', i, '/', len(selected), flush=True)
    if not normalization_stats:
        raise ValueError('No eligible original human training normalization rows')
    mean = np.mean([v[0] for v in normalization_stats], axis=0)
    std = np.sqrt(np.maximum(np.mean([v[1] for v in normalization_stats], axis=0) - mean ** 2, 1e-6))
    norm_path = output / 'normalization.json'
    write_new(norm_path, {'mean': mean.tolist(), 'std': std.tolist(), 'frontend': SPEC,
                         'source_ids': [v[2] for v in normalization_stats],
                         'human_training_utterances': len(normalization_stats),
                         'weighting': 'equal eligible human training utterance'})
    train = [r for r in prepared if r['training_eligible']]
    dev = [r for r in prepared if r['split'] == 'dev']
    excluded = [r for r in prepared if r['split'] == 'train' and not r['training_eligible']]
    for split, rows in (('train', train), ('dev', dev)):
        present = {r['intent_index'] for r in rows}
        if not set(range(len(INTENTS) + 1)) <= present:
            raise ValueError('Post-label-filter research coverage is incomplete in ' + split)
        write_new(output / (split + '.jsonl'), rows, jsonl=True)
    write_new(output / 'excluded-training.jsonl', excluded, jsonl=True)
    _verify(source_bindings); _verify(implementation)
    receipt = {'experiment_id': cfg['experiment_id'], 'status': 'synthetic_research_inputs_prepared',
        'research_ready_for_fitting': True, 'human_qualified': False, 'human_missing_intents': human_missing,
        'coverage': coverage, 'generation': generation_audit, 'manifests': manifests,
        'normalization': _receipt(norm_path), 'train': _receipt(output / 'train.jsonl', rows=len(train)),
        'dev': _receipt(output / 'dev.jsonl', rows=len(dev)),
        'excluded_training': _receipt(output / 'excluded-training.jsonl', rows=len(excluded)),
        'training_exclusion_reasons': dict(Counter(r['training_exclusion'] for r in excluded)),
        'features': len(prepared), 'catalog': _receipt(output / 'catalog.json', entries=len(catalog['entries'])),
        'source_bindings': source_bindings, 'implementation_bindings': implementation,
        'published_evaluation_audio_read': False, 'calibration_features_extracted': False,
        'generated_calibration_audio_integrity_checked': True,
        'limitations': ['Synthetic voices are not independent human people or verified accents',
                       'Original human calibration/qualification gaps remain; no new model gain or Pi claim',
                       'Labels are intended generated text or existing metadata, not independent listening']}
    write_new(output / 'preparation.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--installed', type=Path, default=ROOT / '.cache/speaker-dataset-v1/verified')
    parser.add_argument('--generated', type=Path, default=CACHE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    r = prepare_run(args.protocol, args.installed, args.generated, args.output, workers=args.workers)
    print(json.dumps({'status': r['status'], 'features': r['features'],
                      'train': r['train']['rows'], 'dev': r['dev']['rows'],
                      'excluded_training': r['excluded_training']['rows']}, indent=2))


if __name__ == '__main__':
    main()
