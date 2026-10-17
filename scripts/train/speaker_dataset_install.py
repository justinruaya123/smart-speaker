"""Install pinned AI231 recordings; prepare audited views without training.

Run from the repository root. Original WAV bytes and published evaluation rows
are immutable. Supplemental TTS is train-only, separately labelled, and cannot
enter the original speaker-demo-v1 human-only protocol.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_REVISION = '25111444af3adff7588d86ab27c304bb080895cf'
SUPPLEMENT_REVISION = '230c8b8d58caf78e4bc90886d9c6fa77fe6024b8'
PRIMARY_RECEIPT_SHA256 = '994798b431f9d376f70fc74f2e4758a2d5fa630d384200131f5481578955cb0a'
SUPPLEMENT_RECEIPT_SHA256 = '41bc968e3e90d299c1aca11bdfe1345a2b6fec60676d9c6c75f2276da0788a23'


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def qa_eligible(config, row):
    def passed(value):
        return value is True or (type(value) is int and value == 1) or (
            isinstance(value, str) and value.strip().lower() in {'true', '1'})
    return config != 'gap_fill' or passed(row.get('qa_passed')) or passed(
        row.get('qa_rescued'))


def write_audio(path, payload):
    """Atomically publish complete bytes; never replace shared audio."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Refusing to overwrite mismatched existing audio')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError('Refusing to overwrite mismatched existing audio')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_bound_json(path, expected):
    data = Path(path).read_bytes()
    lf = data.replace(b'\r\n', b'\n')
    if not any(hashlib.sha256(value).hexdigest() == expected
               for value in (data, lf, lf.replace(b'\n', b'\r\n'))):
        raise ValueError('Pinned source receipt identity mismatch')
    return json.loads(data)


def validate_sources(acquired, expected):
    """Config directories are part of shard identity; basenames may repeat."""
    def relative(record):
        return str(record.get('local_path', record.get('path'))).replace('\\', '/')
    pinned = {relative(r): r for r in expected['files']}
    records = {relative(r): r for r in acquired['files']}
    if (len(pinned) != len(expected['files']) or len(records) != len(acquired['files'])
            or records.keys() != pinned.keys()):
        raise ValueError('Acquired source inventory differs from pinned source')
    for path, record in records.items():
        if (record['bytes'], record['sha256']) != (pinned[path]['bytes'], pinned[path]['sha256']):
            raise ValueError('Source receipt identity differs from pinned source')
    return list(records.values())


def verified_review(path, expected):
    """Recover lost historical CRLF bytes only when their frozen hash matches."""
    data = Path(path).read_bytes()
    disk_digest = hashlib.sha256(data).hexdigest()
    representation = 'exact'
    if disk_digest != expected:
        recovered = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        if hashlib.sha256(recovered).hexdigest() != expected:
            raise ValueError('Frozen annotation identity mismatch')
        data, representation = recovered, 'verified CRLF recovery in memory'
    return json.loads(data), {'disk_sha256': disk_digest,
                             'verified_original_sha256': expected,
                             'representation': representation}


def load_policy():
    from scripts.train import pcv_release_prepare as parent
    proposal, recovery = verified_review(parent.PROPOSAL, parent.PROPOSAL_SHA256)
    if proposal['provenance']['source_sha256'] != parent.SOURCE_SHA256:
        raise ValueError('Annotation proposal belongs to another source')
    supported, decisions = {}, []
    for evidence in proposal['supported_correction_evidence']:
        text = evidence['text']
        approved = text not in parent.REJECTED_PROPOSALS
        if approved:
            supported[text] = evidence['expected']
        decisions.append({'text': text, 'approved': approved,
                          'expected': evidence['expected'],
                          'reason': parent.REJECTED_PROPOSALS.get(text, evidence['rationale']),
                          'review_source_ids': evidence['source_ids']})
    return {'version': 'pcv-release-train-v1-annotation-1',
            'supported': supported, 'ambiguous': proposal['text_to_ambiguous_corrections'],
            'decisions': decisions, 'proposal_sha256': parent.PROPOSAL_SHA256,
            'source_sha256': parent.SOURCE_SHA256,
            'rules': 'Exact normalized-text corrections across every split; publisher out_of_scope and unintelligible flags take precedence; no generic rules added.',
            'evidence': 'Supplied transcript metadata review, not independent listening; no teacher or ASR generated ground truth.'}, recovery


def build_views(rows):
    """Keep full evaluation membership; quarantine fitting contamination."""
    views = {name: [] for name in (
        'human/train', 'human/dev', 'human/calibration', 'synthetic/train',
        'evaluation/test', 'evaluation/holdout', 'synthetic/diagnostic',
        'auxiliary', 'excluded')}
    groups = {key: defaultdict(set) for key in (
        'speaker_id', 'pair_id', 'pcm_sha256')}
    evaluation = {key: set() for key in groups}
    labels = defaultdict(set)
    for row in rows:
        if row.get('supplemental') and (
                row['kind'] != 'synthetic' or row['split'] != 'train'):
            raise ValueError('Supplemental synthetic data is train only')
        for key in groups:
            if row.get(key):
                groups[key][row[key]].add(row['split'])
                if row['split'] in {'test', 'holdout'}:
                    evaluation[key].add(row[key])
        if row.get('pcm_sha256'):
            labels[row['pcm_sha256']].add(canonical(
                [row['command'], row['reference_status']]))
    conflicting = {key for key, values in labels.items() if len(values) > 1}
    excluded, used_pcm = Counter(), set()
    for row in rows:
        split = row['split']
        if split in {'test', 'holdout'}:
            views['evaluation/' + split].append(row)
            continue
        if row['reference_status'] == 'auxiliary':
            views['auxiliary'].append(row)
            continue
        reason = None
        if row['audio_flags']:
            reason = 'audio_integrity'
        elif not row['qa_eligible']:
            reason = 'failed_supplemental_qa'
        elif row['reference_status'] not in {'supported', 'unsupported'}:
            reason = 'ambiguous_reference'
        elif row['pcm_sha256'] in conflicting:
            reason = 'conflicting_pcm_labels'
        for key, label in (('pcm_sha256', 'pcm'), ('speaker_id', 'speaker'),
                           ('pair_id', 'pair')):
            value = row.get(key)
            if reason is None and (not value or str(value).endswith('/')):
                reason = 'missing_' + label
            if reason is None and value in evaluation[key]:
                reason = 'heldout_' + label + '_overlap'
            if reason is None and len(groups[key][value]) > 1:
                reason = 'cross_split_' + label + '_overlap'
        if reason is None and row['kind'] == 'synthetic' and split != 'train':
            views['synthetic/diagnostic'].append(row)
            continue
        if reason is None and row['pcm_sha256'] in used_pcm:
            reason = 'duplicate_pcm'
        if reason:
            excluded[reason] += 1
            views['excluded'].append({**row, 'exclusion_reason': reason})
        else:
            used_pcm.add(row['pcm_sha256'])
            views[row['kind'] + '/' + split].append(row)
    return views, {
        'excluded': dict(excluded),
        'raw_cross_split_groups': {key: sum(len(v) > 1 for v in values.values())
                                   for key, values in groups.items()},
        'conflicting_pcm_groups': len(conflicting),
        'evaluation_membership_preserved': True,
        'speaker_identity_scope': 'Declared source IDs only; aliases and voice identity unverified',
        'synthetic_reference_aliases': 'unresolved; not proof of unseen human speakers',
        'near_duplicate_audio': 'not audited', 'independent_listening': 'not performed',
    }


def install(output, workers=8):
    import numpy as np
    import pyarrow
    import pyarrow.parquet as pq
    import soundfile as sf
    from scripts.runtime.pcv_schema import validate_command
    from scripts.test.speaker_demo_suite import pcm_digest
    from scripts.train.pcv_release_prepare import annotate
    from scripts.train.pcv_updated_prepare import assign_split, pair_identity

    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / '.cache'):
        raise ValueError('Installation indexes must stay under root .cache')
    if output.exists():
        raise FileExistsError('Use a new output directory; preserve earlier audit')
    primary_path = ROOT / 'experiments/pcv-source-refresh-v1/acquisition-complete.json'
    supplement_path = ROOT / 'data/sources/fil-supplemental.source.json'
    primary = read_bound_json(primary_path, PRIMARY_RECEIPT_SHA256)
    supplement = read_bound_json(supplement_path, SUPPLEMENT_RECEIPT_SHA256)
    if primary['revision'] != PRIMARY_REVISION or supplement['revision'] != SUPPLEMENT_REVISION:
        raise ValueError('Dataset revision differs from installation contract')
    primary_expected = primary
    supplement_expected = supplement
    inputs = []
    for acquired, expected, extra in (
            (primary, primary_expected, False),
            (supplement, supplement_expected, True)):
        for record in validate_sources(acquired, expected):
            relative = str(record.get('local_path', record.get('path'))).replace('\\', '/')
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT / '.cache'):
                raise ValueError('Source payload must be under root .cache')
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('Source payload identity mismatch: ' + str(path))
            inputs.append((path, extra, acquired['repository'], acquired['revision']))
    pool = ROOT / 'data/audio/ai231-pool-v1'
    for bucket in range(256):
        (pool / f'{bucket:02x}').mkdir(parents=True, exist_ok=True)
    output.mkdir()
    policy, policy_recovery = load_policy()
    primary_license = 'Source-dependent research/education terms; pinned primary source card retained'
    supplement_license = supplement['license']

    def decode(item):
        source, location, extra, repo, revision, config, index = item
        source = dict(source)
        audio = source.pop('audio', None)
        original_split = location.name.split('-')[0]
        source['published_split'] = original_split
        typed = annotate(source, policy)
        if typed['expected'] is not None:
            validate_command(typed['expected'])
        flags, pcm_hash, audio_hash, relative_audio = [], None, None, None
        details = {}
        payload = audio.get('bytes') if audio else None
        if payload is None:
            flags.append('missing_embedded_audio')
        else:
            audio_hash = hashlib.sha256(payload).hexdigest()
            path = pool / audio_hash[:2] / (audio_hash + '.wav')
            write_audio(path, payload)
            relative_audio = path.relative_to(ROOT).as_posix()
            try:
                info = sf.info(io.BytesIO(payload))
                pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
                details = {'sample_rate': rate, 'channels': info.channels,
                           'subtype': info.subtype, 'samples': len(pcm),
                           'decoded_duration_seconds': len(pcm) / rate}
                if info.format != 'WAV' or info.subtype != 'PCM_16' or rate != 16000 or pcm.ndim != 1:
                    flags.append('unexpected_audio_format')
                if not pcm.size or not np.isfinite(pcm).all():
                    flags.append('empty_or_nonfinite_audio')
                elif pcm.ndim == 1 and rate == 16000:
                    pcm_hash = pcm_digest(pcm, rate)
                    details.update(rms=float(np.sqrt(np.mean(pcm * pcm))),
                                   clipped_fraction=float(np.mean(np.abs(pcm) >= .999)))
                    if not np.any(pcm):
                        flags.append('all_zero_audio')
            except (ValueError, RuntimeError) as exc:
                flags.append('unreadable_audio:' + type(exc).__name__)
        synthetic_flag = source.get('is_synthetic')
        if synthetic_flag not in (0, 1, False, True):
            raise ValueError('Unresolved human/synthetic provenance')
        if extra and not synthetic_flag:
            raise ValueError('Supplement unexpectedly contains human recording')
        reference = source.get('reference_voice')
        speaker = (('fil-reference/' + str(reference)) if extra and reference else
                   (str(source.get('source') or '') + '/' + str(source.get('speaker_id') or '')))
        kind = 'synthetic' if synthetic_flag else 'human'
        return {
            'id': f'{repo}/{config}/{location.stem}/{index:06d}',
            'audio': relative_audio, 'sha256': audio_hash, 'pcm_sha256': pcm_hash,
            'speaker_id': speaker, 'speaker_aliases_resolved': False,
            'pair_id': pair_identity(source), 'source': repo,
            'source_revision': revision, 'license': supplement_license if extra else primary_license,
            'kind': kind, 'role': 'command', 'command': typed['expected'],
            'split': 'train' if extra else assign_split(source),
            'published_split': original_split, 'config': config,
            'supplemental': extra, 'qa_eligible': qa_eligible(config, source),
            'reference_status': typed['reference_status'], 'reference_reason': typed['reason'],
            'slot_metadata_conflict': typed['slot_metadata_conflict'],
            'audio_flags': flags,
            'metric_eligible': not flags and typed['reference_status'] in {'supported', 'unsupported'},
            'parquet': location.relative_to(ROOT).as_posix(), 'parquet_row': index,
            'publisher_metadata': source, **details,
        }

    rows, source_counts = [], Counter()
    inventory = output / 'inventory.jsonl'
    with inventory.open('x', encoding='utf8') as stream, ThreadPoolExecutor(max_workers=workers) as executor:
        for path, extra, repo, revision in inputs:
            config = path.parent.name if extra else 'primary'
            if extra and config == 'data':
                config = 'default'
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                items = []
                for source in batch.to_pylist():
                    items.append((source, path, extra, repo, revision, config, index))
                    index += 1
                for row in executor.map(decode, items):
                    rows.append(row)
                    source_counts[repo + '/' + config + '/' + row['published_split']] += 1
                    stream.write(canonical(row) + '\n')
            print('INSTALLED', path.name, index, 'rows; total', len(rows), flush=True)
    views, audit = build_views(rows)
    manifests = {}
    for name, records in views.items():
        path = output / 'views' / (name + '.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf8') as stream:
            for row in records:
                stream.write(canonical(row) + '\n')
        manifests[name] = {'path': path.relative_to(ROOT).as_posix(), 'rows': len(records), 'sha256': sha(path)}
    counts = Counter(row['kind'] for row in rows)
    audit.update({
        'version': 'speaker-dataset-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'sources': dict(source_counts), 'recordings': len(rows), 'kinds': dict(counts),
        'unique_audio_files': len({r['sha256'] for r in rows if r['sha256']}),
        'unique_pcm': len({r['pcm_sha256'] for r in rows if r['pcm_sha256']}),
        'audio_flags': dict(Counter(flag for row in rows for flag in row['audio_flags'])),
        'reference_status': dict(Counter(r['reference_status'] for r in rows)),
        'supplemental_qa': dict(Counter(str(r['qa_eligible']) for r in rows if r['supplemental'])),
        'inventory': inventory.relative_to(ROOT).as_posix(), 'inventory_sha256': sha(inventory),
        'views': manifests, 'source_receipts': {primary_path.relative_to(ROOT).as_posix(): sha(primary_path),
            supplement_path.relative_to(ROOT).as_posix(): sha(supplement_path)},
        'source_payloads': {p.relative_to(ROOT).as_posix(): sha(p) for p, *_ in inputs},
        'implementation_sha256': sha(Path(__file__)),
        'annotation_policy_sha256': hashlib.sha256(canonical(policy).encode()).hexdigest(),
        'annotation_receipt_recovery': policy_recovery,
        'evaluation': 'Every published test/holdout row retained, including ambiguity/errors; filter metric_eligible explicitly for labelled metrics',
        'no_training_performed': True, 'features_extracted': False,
        'coverage': {name: {'intents': dict(Counter(r['command']['intent'] if r['command'] else 'UNKNOWN' for r in records)),
                           'declared_speakers': len({r['speaker_id'] for r in records})}
                    for name, records in views.items() if name.startswith(('human/', 'synthetic/train'))},
    })
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'pyarrow': pyarrow.__version__, 'soundfile': sf.__version__,
                   'purpose': 'Dataset installation only; project runtime lock unchanged',
                   'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()}
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('recordings', 'kinds', 'unique_audio_files', 'excluded', 'audio_flags')}, indent=2), flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/speaker-dataset-v1')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    install(args.output, args.workers)


if __name__ == '__main__':
    main()
