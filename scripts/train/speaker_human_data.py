"""Research-only human-data preparation; never mutates frozen models or inputs.

Original publisher utterances only. Ambiguous references are not UNKNOWN.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from scripts.runtime.pcv_schema import INTENTS, canonical, validate_command

SPLITS = ('train', 'dev', 'calibration', 'test', 'holdout')
LABELS = (*INTENTS, 'UNKNOWN')


def partition_human_rows(rows):
    """Segregate references, not audio, preserving every exclusion explicitly."""
    views = {split: {label: [] for label in LABELS} for split in SPLITS}
    excluded = []
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError('Duplicate example identity: ' + row['id'])
        seen.add(row['id'])
        if row['split'] not in views:
            raise ValueError('Unrecognized split: ' + row['split'])
        reason = None
        if row.get('synthetic') is not False or row.get('kind') != 'human':
            reason = 'not_original_human'
        elif row.get('metric_eligible') is not True or row.get('reference_status') in ('ambiguous', 'auxiliary'):
            reason = 'ambiguous_reference'
        elif row['split'] == 'train' and row.get('training_eligible') is not True:
            reason = 'ineligible_training_row'
        if reason:
            excluded.append({**row, 'exclusion': reason})
            continue
        expected = row['expected']
        if expected is not None:
            validate_command(expected)
        label = expected['intent'] if expected is not None else 'UNKNOWN'
        views[row['split']][label].append(dict(row))
    return views, excluded


def split_leakage(rows):
    """Check all partition pairs, including calibration vs dev and parents."""
    fields = ('id', 'speaker', 'sha256', 'pcm_sha256', 'pair', 'source_speaker_key',
              'session_id', 'parent_recording_id', 'speaker_alias_group')
    groups = {field: defaultdict(list) for field in fields}
    for row in rows:
        for field in fields:
            value = row.get(field)
            if value not in (None, '', 'unknown'):
                groups[field][str(value)].append(row)
    result = []
    for field, identities in groups.items():
        for identity, members in identities.items():
            splits = sorted({r['split'] for r in members})
            if len(splits) > 1:
                result.append(dict(field=field, identity=identity, splits=splits,
                                   ids=sorted({r['id'] for r in members})))
    return result


def coverage_report(rows, baseline, *, speaker_targets=None, catalog=()):
    """Audit typed command cells; targets are pilot floors, not qualification."""
    targets = speaker_targets or {'train': 5, 'dev': 3, 'calibration': 3}
    views, _ = partition_human_rows(rows)
    result = {}
    for label in LABELS:
        commands = {canonical(r['command']): r['command'] for r in baseline
                    if (r['command'] or {}).get('intent', 'UNKNOWN') == label}
        splits, queue, baseline_counts = {}, [], {}
        for split in SPLITS:
            values = views[split][label]
            speakers = Counter(r['speaker'] for r in values)
            by_command = defaultdict(list)
            by_slot = defaultdict(list)
            for r in values:
                by_command[canonical(r['expected'])].append(r)
                for key, value in (r['expected'] or {}).get('slots', {}).items():
                    by_slot[key + '=' + canonical(value)].append(r)
            cell = lambda rs: dict(rows=len(rs), speakers=len({r['speaker'] for r in rs}))
            splits[split] = dict(rows=len(values), speakers=len(speakers),
                typed_commands=len(by_command),
                largest_speaker_fraction=max(speakers.values()) / len(values) if values else None,
                sources=dict(Counter(r.get('publisher_source', r.get('source', 'unknown')) for r in values)),
                transcript_origins=dict(Counter(r.get('transcript_source', 'not_audited') for r in values)),
                unique_texts=len({r['text'] for r in values if r.get('text')}),
                commands={key: cell(rs) for key, rs in sorted(by_command.items())},
                slot_values={key: cell(rs) for key, rs in sorted(by_slot.items())})
            baseline_counts[split] = dict(covered=sum(k in by_command for k in commands), total=len(commands))
            if split in targets:
                for signature, command in commands.items():
                    available = len({r['speaker'] for r in by_command.get(signature, [])})
                    if available < targets[split]:
                        queue.append(dict(split=split, command=command, current_speakers=available,
                                          minimum_additional_speakers=targets[split] - available,
                                          target_speakers=targets[split]))
        ready = all(splits[s]['rows'] > 0 for s in ('train', 'dev', 'calibration'))
        approved = {canonical(r['command']): r['command'] for r in catalog
                    if r['command'] is not None and r['command']['intent'] == label}
        catalog_counts, catalog_queue = {}, []
        for split in SPLITS:
            present = splits[split]['commands']
            catalog_counts[split] = dict(covered=sum(k in present for k in approved), total=len(approved))
            if split in targets:
                for signature, command in approved.items():
                    available = present.get(signature, {}).get('speakers', 0)
                    if available < targets[split]:
                        catalog_queue.append(dict(split=split, command=command, current_speakers=available,
                            minimum_additional_speakers=targets[split] - available,
                            target_speakers=targets[split]))
        result[label] = dict(splits=splits, baseline_commands=baseline_counts,
                             collection_queue=queue, ready_for_human_selection=ready,
                             catalog_commands=catalog_counts, catalog_collection_queue=catalog_queue,
                             pilot_speaker_floors_met=ready and not queue)
    return result


def human_supervision(row):
    """Rebuild dependent semantic targets; never reinterpret publisher labels."""
    result = copy.deepcopy(row)
    expected = result['expected']
    if expected is not None:
        validate_command(expected)
    result['intent_index'] = INTENTS.index(expected['intent']) + 1 if expected else 0
    result['target_index'] = -100
    if expected and expected['intent'] in {'LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'}:
        result['target_index'] = int(expected['slots']['light_id'] != 'light.all')
    return result


def training_ready(coverage, *, overlaps=(), conflicts=(), unsafe=()):
    """Full scope includes approved extensions and UNKNOWN, not just baseline."""
    return (set(coverage) == set(LABELS) and not (overlaps or conflicts or unsafe)
            and all(v['pilot_speaker_floors_met'] and not v['catalog_collection_queue']
                    for v in coverage.values()))


def audio_diagnostics(pcm, sample_rate) -> dict:
    """Energy-only boundary screen, NOT VAD or proof of spoken annotation.

    Preserve original audio. Propose one continuous outer interval with 150 ms
    context; never splice at an internal pause or discard short commands.
    """
    pcm = np.asarray(pcm, dtype=np.float32)
    if (sample_rate != 16000 or pcm.ndim != 1 or not len(pcm)
            or not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1.):
        raise ValueError('Expected nonempty finite mono 16 kHz PCM in [-1, 1]')
    peak = float(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    frame = 320
    count = (len(pcm) + frame - 1) // frame
    values = np.pad(pcm, (0, count * frame - len(pcm))).reshape(count, frame)
    energy = np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1))
    threshold = max(1e-4, float(energy.max()) * .03)
    active = np.flatnonzero(energy >= threshold)
    flags = []
    if len(pcm) < 400:
        flags.append('below_frontend_window')
    if len(pcm) > 240000:
        flags.append('exceeds_runtime_15_seconds')
    clipping = float(np.mean(np.abs(pcm) >= .999))
    if clipping > .001:
        flags.append('possible_clipping')
    if abs(float(pcm.mean())) > .01:
        flags.append('dc_offset')
    longest = 0
    leading = trailing = None
    segment = None
    if len(active):
        first, last = int(active[0]), int(active[-1])
        leading = min(len(pcm), first * frame) / sample_rate
        trailing = max(0, len(pcm) - (last + 1) * frame) / sample_rate
        run = 0
        for quiet in energy[first:last + 1] < threshold:
            run = run + 1 if quiet else 0
            longest = max(longest, run)
        pad = int(.15 * sample_rate)
        segment = [max(0, first * frame - pad), min(len(pcm), (last + 1) * frame + pad)]
        if first == 0:
            flags.append('active_at_start_review_boundary')
        if last == count - 1:
            flags.append('active_at_end_review_boundary')
        if longest * frame / sample_rate >= .4:
            flags.append('internal_pause_endpoint_risk')
    else:
        flags.append('no_detectable_energy')
    return dict(duration_seconds=len(pcm) / sample_rate, peak=peak,
                rms_dbfs=20 * float(np.log10(max(rms, 1e-12))),
                dc_offset=float(pcm.mean()), clipping_fraction=clipping,
                leading_low_energy_seconds=leading, trailing_low_energy_seconds=trailing,
                longest_internal_low_energy_seconds=longest * frame / sample_rate,
                proposed_segment_samples=segment, flags=flags,
                method='20ms RMS >= max(1e-4, .03*maximum_frame_RMS); proposal_only')


class HumanTrainingPools:
    """Balance intent -> typed command/rejection -> source -> speaker -> row.

    Repetition is deliberate sampling, never additional human coverage. No
    synthetic, auxiliary, ambiguous or evaluation row can enter this sampler.
    """
    def __init__(self, rows):
        self.groups = {}
        self.ids = set()
        for row in rows:
            if (row.get('split') != 'train' or row.get('training_eligible') is not True
                    or row.get('synthetic') is not False or row.get('kind') != 'human'
                    or row.get('metric_eligible') is not True
                    or row.get('reference_status') in ('ambiguous', 'auxiliary')
                    or row.get('speaker') in (None, '', 'unknown')
                    or not row.get('phones') or row['id'] in self.ids):
                raise ValueError('Unsafe human fitting row: ' + row['id'])
            self.ids.add(row['id'])
            item = human_supervision(row)
            positive = item['expected'] is not None
            key = item['expected']['intent'] if positive else item.get('publisher_command', 'UNKNOWN')
            signature = canonical(item['expected']) if positive else item.get('reference_reason', 'unknown')
            node = self.groups.setdefault('positive' if positive else 'negative', {})
            for value in (key, signature, item.get('publisher_source', item.get('source', 'unknown')), item['speaker']):
                node = node.setdefault(value, {})
            node.setdefault('rows', []).append(item)
        if set(self.groups) != {'positive', 'negative'}:
            raise ValueError('Human positives and true UNKNOWN examples are both required')

    def sample_batch(self, *, seed, size=32):
        if type(size) is not int or size < 4:
            raise ValueError('Human batch size must be an integer >= 4')
        rng = random.Random(seed)
        negative_n = max(1, size // 4)
        result = []
        for group, count in (('positive', size - negative_n), ('negative', negative_n)):
            for _ in range(count):
                node = self.groups[group]
                for _ in range(4):
                    node = node[rng.choice(sorted(node))]
                result.append(copy.deepcopy(rng.choice(node['rows'])))
        rng.shuffle(result)
        return result


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(canonical(row) + '\n')


def fit_human_normalization(rows):
    """Equal utterance weighting on the exact admitted human training view."""
    from vcm.frontend import SPEC
    means, squares, identities = [], [], []
    for row in rows:
        if (row.get('split') != 'train' or row.get('synthetic') is not False
                or row.get('kind') != 'human' or row.get('training_eligible') is not True
                or row.get('metric_eligible') is not True or row['id'] in identities):
            raise ValueError('Only unique admitted human train rows may fit normalization')
        if file_sha256(row['features']) != row['features_sha256']:
            raise ValueError('Feature checksum mismatch: ' + row['id'])
        values = np.load(row['features'], allow_pickle=False).astype(np.float64)
        if values.ndim != 2 or values.shape[0] != 40 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError('Invalid raw features for human normalization')
        means.append(values.mean(axis=1))
        squares.append((values * values).mean(axis=1))
        identities.append(row['id'])
    if not means:
        raise ValueError('No admitted human training rows')
    mean = np.mean(means, axis=0)
    std = np.sqrt(np.maximum(np.mean(squares, axis=0) - mean ** 2, 1e-6))
    return dict(mean=mean.tolist(), std=std.tolist(), frontend=SPEC,
                human_training_utterances=len(identities), source_ids=identities,
                weighting='equal weight per admitted human training utterance',
                inference_change_allowed=False)


def human_metrics(rows):
    """Complete-command metrics with explicit missing and ambiguous evidence."""
    rows = list(rows)
    if any(r.get('synthetic') is not False or r.get('kind') != 'human' for r in rows):
        raise ValueError('Human metrics require explicitly original-human rows')
    scored = [r for r in rows if r.get('metric_eligible') is True]
    positive = [r for r in scored if r['expected'] is not None]
    negative = [r for r in scored if r['expected'] is None]
    exact = lambda r, key: r.get(key) is not None and r[key] == r['expected']
    ratio = lambda n, d: n / d if d else None
    accepted = [r for r in scored if r.get('application_command') is not None]
    correct = sum(exact(r, 'application_command') for r in positive)
    per_intent = {}
    for intent in INTENTS:
        examples = [r for r in positive if r['expected']['intent'] == intent]
        predictions = [r for r in accepted if r['application_command']['intent'] == intent]
        hit = sum(exact(r, 'application_command') for r in examples)
        per_intent[intent] = dict(positive_n=len(examples),
            raw_correct=sum(exact(r, 'candidate') for r in examples),
            accepted_correct=hit, accepted_recall=ratio(hit, len(examples)),
            accepted_precision=ratio(hit, len(predictions)), wrong_accepts=len(predictions) - hit,
            rejections=dict(Counter(r.get('application_reason', 'unspecified')
                for r in examples if r.get('application_command') is None)))
    speaker_recalls = []
    for speaker in sorted({r['speaker'] for r in positive}):
        examples = [r for r in positive if r['speaker'] == speaker]
        speaker_recalls.append(sum(exact(r, 'application_command') for r in examples) / len(examples))
    covered = [v['accepted_recall'] for v in per_intent.values() if v['positive_n']]
    return dict(positive_n=len(positive), negative_n=len(negative), ambiguous_n=len(rows) - len(scored),
        raw_correct=sum(exact(r, 'candidate') for r in positive), accepted_correct=correct,
        accepted_precision=ratio(correct, len(accepted)), accepted_recall=ratio(correct, len(positive)),
        wrong_positive_accepts=sum(not exact(r, 'application_command') and r.get('application_command') is not None for r in positive),
        unknown_false_accepts=sum(r.get('application_command') is not None for r in negative),
        covered_intents=len(covered), total_intents=len(INTENTS),
        macro_covered_intent_recall=float(np.mean(covered)) if covered else None,
        macro_speaker_recall=float(np.mean(speaker_recalls)) if speaker_recalls else None,
        per_intent=per_intent, qualification_granted=False)


def join_publisher_metadata(rows, directory):
    """Bind metadata by repository/revision/shard hash/row, excluding audio.

    ASR-derived transcript provenance is retained as unreviewed supervision,
    not silently upgraded to independent human verification.
    """
    import pyarrow.parquet as pq
    from scripts.train.pcv_updated_prepare import spoken, pair_identity

    directory = Path(directory).resolve()
    groups = defaultdict(list)
    for row in rows:
        loc = row['location']
        key = (loc['repository'], loc['revision'], loc['file'], loc['sha256'])
        groups[key].append(row)
    output, shards, omitted = [], [], []
    all_ids = {r['id'] for r in rows}
    for (repository, revision, name, digest), members in groups.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Publisher metadata path escapes source root')
        if file_sha256(path) != digest:
            raise ValueError('Publisher shard checksum mismatch: ' + name)
        parquet = pq.ParquetFile(path)
        columns = [c for c in parquet.schema_arrow.names if c not in ('audio', 'whisper_transcript')]
        metadata = parquet.read(columns=columns).to_pylist()
        prefix = members[0]['id'].rsplit('/', 1)[0]
        shards.append(dict(repository=repository, revision=revision, file=name,
                           sha256=digest, metadata_rows=len(metadata), audio_column_read=False))
        for row in members:
            index = row['location']['row']
            if (type(index) is not int or not 0 <= index < len(metadata)
                    or row['id'] != prefix + '/' + f'{index:06d}'):
                raise ValueError('Publisher row identity mismatch: ' + row['id'])
            meta = metadata[index]
            if meta['is_synthetic'] not in (0, False):
                raise ValueError('Human manifest contains publisher synthetic recording: ' + row['id'])
            item = dict(row)
            item.update(publisher_source=meta.get('source', 'unknown'),
                publisher_command=meta.get('command', 'UNKNOWN'),
                transcript_source=meta.get('transcript_source', 'unknown'),
                publisher_whisper_check=meta.get('whisper_check', ''),
                publisher_slot_value=meta.get('slot_value', ''),
                publisher_out_of_scope=meta.get('out_of_scope', 0),
                listening_status='not_independently_reviewed',
                accent_group=meta.get('accent_group', 'unknown'))
            if row['split'] in ('train', 'dev'):
                item['text'] = spoken(meta.get('transcript') or '')
                item['publisher_transcript'] = meta.get('transcript') or ''
            speaker = str(meta.get('speaker_id', ''))
            item['source_speaker_key'] = hashlib.sha256(canonical(
                [repository, revision, meta.get('source'), speaker]).encode()).hexdigest()
            item['pair'] = hashlib.sha256(canonical(
                [repository, revision, pair_identity(meta)]).encode()).hexdigest()
            output.append(item)
        if Path(name).name.startswith('train-'):
            for index, meta in enumerate(metadata):
                identity = prefix + '/' + f'{index:06d}'
                if meta['is_synthetic'] in (0, False) and identity not in all_ids:
                    omitted.append(dict(id=identity, publisher_command=meta.get('command'),
                        transcript_source=meta.get('transcript_source'),
                        publisher_whisper_check=meta.get('whisper_check'),
                        text=spoken(meta.get('transcript') or ''),
                        status='requires_frozen_split_eligibility_and_annotation_review'))
    receipt = dict(joined_rows=len(output), shards=shards,
                   omitted_human_training_shard_rows=len(omitted),
                   omitted_by_command=dict(Counter(r['publisher_command'] for r in omitted)),
                   omitted_rows=omitted,
                   independently_listened_rows=0,
                   annotation_policy='Parent expected labels unchanged; metadata is provenance only')
    return output, receipt


def audit_dataset(prepared, output, *, baseline, metadata_dir=None, protected_paths=(), catalog=()):
    """Execute a reference/feature audit; calibration/test audio is never opened."""
    import soundfile as sf
    from scripts.test.speaker_demo_suite import pcm_digest

    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protected = {str(p): file_sha256(p) for p in protected_paths}
    manifest_hashes = {s: file_sha256(prepared / (s + '.jsonl')) for s in SPLITS}
    rows = []
    for split in SPLITS:
        records = read_rows(prepared / (split + '.jsonl'))
        if any(r['split'] != split for r in records):
            raise ValueError('Manifest split disagrees with row split: ' + split)
        rows.extend(records)
    human = [r for r in rows if r.get('synthetic') is False and r.get('kind') == 'human']
    metadata_receipt = None
    if metadata_dir is not None:
        human, metadata_receipt = join_publisher_metadata(human, metadata_dir)
        enriched = {r['id']: r for r in human}
        rows = [enriched.get(r['id'], r) for r in rows]
    views, exclusions = partition_human_rows(rows)
    overlaps = split_leakage(human)
    audio_records = []
    dependent_repairs = []
    for split in ('train', 'dev'):
        for label in LABELS:
            for row in views[split][label]:
                if file_sha256(row['features']) != row['features_sha256']:
                    raise ValueError('Feature checksum mismatch: ' + row['id'])
                if file_sha256(row['audio']) != row['sha256']:
                    raise ValueError('Audio checksum mismatch: ' + row['id'])
                pcm, rate = sf.read(row['audio'], dtype='float32', always_2d=False)
                if pcm_digest(pcm, rate) != row['pcm_sha256']:
                    raise ValueError('PCM checksum mismatch: ' + row['id'])
                diagnostic = audio_diagnostics(pcm, rate)
                values = np.load(row['features'], allow_pickle=False)
                expected_frames = max(0, 1 + (len(pcm) - 400) // 160)
                if (values.dtype != np.dtype('float32') or values.shape != (40, expected_frames)
                        or not np.isfinite(values).all()):
                    raise ValueError('Malformed feature array: ' + row['id'])
                phones = row.get('phones') or []
                required = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
                sufficient = bool(phones) and (values.shape[1] + 1) // 2 >= required
                if not sufficient:
                    diagnostic['flags'].append('insufficient_or_missing_ctc_supervision')
                audio_records.append(dict(id=row['id'], split=split, intent=label,
                    source=row.get('publisher_source', row.get('source')),
                    features_sha256=row['features_sha256'],
                    feature_mean=float(values.mean()), feature_std=float(values.std()),
                    ctc_required_frames=required, ctc_available_frames=(values.shape[1] + 1) // 2,
                    **diagnostic))
                if split == 'train':
                    repaired = human_supervision(row)
                    if any(repaired[k] != row.get(k) for k in ('intent_index', 'target_index')):
                        dependent_repairs.append(dict(id=row['id'],
                            before={k: row.get(k) for k in ('intent_index', 'target_index')},
                            after={k: repaired[k] for k in ('intent_index', 'target_index')}))
                    row.update({k: repaired[k] for k in ('intent_index', 'target_index')})
    # A failed QA row is retained as evidence, never silently trained or deleted.
    unsafe = {r['id'] for r in audio_records if set(r['flags']) & {
        'no_detectable_energy', 'below_frontend_window', 'exceeds_runtime_15_seconds',
        'insufficient_or_missing_ctc_supervision'}}
    for split in ('train', 'dev'):
        for label in LABELS:
            kept = []
            for row in views[split][label]:
                if row['id'] in unsafe:
                    exclusions.append({**row, 'exclusion': 'failed_audio_or_ctc_QA'})
                else:
                    kept.append(row)
            views[split][label] = kept
    eligible = [r for split in SPLITS for group in views[split].values() for r in group]
    per_intent = coverage_report(eligible, baseline, catalog=catalog)
    conflicts = []
    by_pcm = defaultdict(list)
    for r in human:
        by_pcm[r['pcm_sha256']].append(r)
    for digest, members in by_pcm.items():
        if len({canonical(r['expected']) for r in members}) > 1:
            conflicts.append(dict(pcm_sha256=digest, ids=[r['id'] for r in members]))
    report = dict(format='speaker-human-first-v1',
        input_manifests=manifest_hashes, input_rows=len(rows), input_human_rows=len(human),
        view_rows={s: sum(len(rs) for rs in views[s].values()) for s in SPLITS},
        exclusions=dict(Counter(r['exclusion'] for r in exclusions)),
        ambiguous_by_split=dict(Counter(r['split'] for r in exclusions if r['exclusion'] == 'ambiguous_reference')),
        per_intent=per_intent, audio_checked_splits=['train', 'dev'],
        audio_checked_rows=len(audio_records),
        audio_flags=dict(Counter(flag for r in audio_records for flag in r['flags'])),
        split_leakage=overlaps, conflicting_pcm_labels=conflicts,
        dependent_label_repairs=dependent_repairs,
        metadata_receipt=metadata_receipt,
        ready_for_full_human_training=training_ready(per_intent, overlaps=overlaps,
                                                    conflicts=conflicts, unsafe=unsafe),
        blind_evaluation_status='Existing published test/holdout are regression only; new people/sessions required',
        quality_gain_measured=False, original_audio_modified=False, synthetic_injection=False)
    for split in SPLITS:
        for label in LABELS:
            write_rows(output / 'views' / split / (label + '.jsonl'), views[split][label])
        write_rows(output / (split + '.jsonl'), [r for label in LABELS for r in views[split][label]])
    write_rows(output / 'exclusions.jsonl', exclusions)
    write_rows(output / 'audio-audit.jsonl', audio_records)
    after = {str(p): file_sha256(p) for p in protected_paths}
    if after != protected:
        raise ValueError('Protected input changed during audit')
    if manifest_hashes != {s: file_sha256(prepared / (s + '.jsonl')) for s in SPLITS}:
        raise ValueError('Input manifest changed during audit')
    report['protected_files'] = protected
    report['outputs'] = {p.relative_to(output).as_posix(): file_sha256(p)
                         for p in output.rglob('*.jsonl')}
    write_json(output / 'audit.json', report)
    return report


def optimizer_smoke(audit_root, output, *, architecture: dict | None = None):
    """One random-init CPU update verifies plumbing, not recognition quality."""
    import torch
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.train.speaker_final_train import optimization_step

    audit_root, output = Path(audit_root), Path(output)
    audit = json.loads((audit_root / 'audit.json').read_text(encoding='utf-8'))
    if audit['split_leakage'] or audit['conflicting_pcm_labels']:
        raise ValueError('Unsafe split/label audit; optimizer smoke blocked')
    if file_sha256(audit_root / 'train.jsonl') != audit['outputs']['train.jsonl']:
        raise ValueError('Admitted training manifest checksum mismatch')
    rows = read_rows(audit_root / 'train.jsonl')
    pools = HumanTrainingPools(rows)
    normalization = fit_human_normalization(rows)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'normalization-human.json', normalization)
    batch = pools.sample_batch(seed=53, size=4)
    mean = np.asarray(normalization['mean'], dtype=np.float32)[:, None]
    std = np.asarray(normalization['std'], dtype=np.float32)[:, None]
    arrays = {r['id']: (np.load(r['features'], allow_pickle=False) - mean) / std for r in batch}
    architecture = architecture or dict(width=128, layers=6, heads=4, feedforward=256, dropout=.1)
    torch.set_num_threads(2)
    torch.manual_seed(53)
    model = PooledContextVCM(**architecture).cpu()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.001)
    model.train()
    losses = optimization_step(model, batch, arrays, optimizer, device='cpu', seed=53,
                               augment_training=False, microbatch=4)
    changed = any(not torch.equal(before[k], value) for k, value in model.state_dict().items())
    if not changed or not np.isfinite(losses).all():
        raise ValueError('Optimizer smoke did not produce a finite real weight update')
    receipt = dict(format='human-optimizer-plumbing-smoke-v1', device='cpu',
        parameters=sum(p.numel() for p in model.parameters()), architecture=architecture,
        seed=53, optimizer_updates=1, weights_changed=changed, losses=losses,
        original_human_batch_ids=[r['id'] for r in batch],
        normalization_human_rows=normalization['human_training_utterances'],
        normalization_sha256=file_sha256(output / 'normalization-human.json'),
        training_manifest_sha256=audit['outputs']['train.jsonl'],
        synthetic_rows=0, augmentation=False, pretrained_weights=False,
        evaluation_audio_opened=False, selected_model_modified=False,
        accuracy_gain_claimed=False, full_training_completed=False)
    write_json(output / 'receipt.json', receipt)
    return receipt


def main(argv=None):
    import argparse
    from scripts.runtime.pcv_schema import baseline_cases

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New immutable audit/cache directory')
    parser.add_argument('--metadata-root', type=Path, help='Pinned primary source root containing data/*.parquet')
    parser.add_argument('--protocol', type=Path,
                        default=root / 'experiments/speaker-human-first-v1/protocol.json')
    parser.add_argument('--require-ready', action='store_true',
                        help='Exit 2 for missing human coverage/QA; never imply full-model qualification')
    parser.add_argument('--smoke-fit', action='store_true',
                        help='One random-init CPU optimizer plumbing update; not a full training run')
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    fixes_path = args.protocol.parent / 'intent-fixes.json'
    fixes = json.loads(fixes_path.read_text(encoding='utf-8'))
    if set(fixes['intents']) != set(LABELS):
        raise ValueError('Every finite intent and UNKNOWN needs an explicit fix')
    protected = [root / name for name in protocol['baseline']['inputs']]
    for name, digest in protocol['baseline']['inputs'].items():
        if file_sha256(root / name) != digest:
            raise ValueError('Protocol input changed; freeze a new version: ' + name)
    catalog = json.loads((root / 'submission/models/compact/catalog.json').read_text(encoding='utf-8'))['entries']
    report = audit_dataset(args.prepared, args.output, baseline=baseline_cases(),
                           metadata_dir=args.metadata_root, protected_paths=protected, catalog=catalog)
    write_json(args.output / 'intent-fixes.json', fixes)
    receipt = dict(protocol_sha256=file_sha256(args.protocol),
        intent_fixes_sha256=file_sha256(fixes_path), audit_sha256=file_sha256(args.output / 'audit.json'),
        readiness_passed=report['ready_for_full_human_training'],
        full_training_run=False, installed_artifacts_touched=False)
    if args.smoke_fit and not (args.require_ready and not receipt['readiness_passed']):
        receipt['optimizer_smoke'] = optimizer_smoke(args.output, args.output / 'optimizer-smoke')
    write_json(args.output / 'preparation-receipt.json', receipt)
    print(json.dumps(dict(input_rows=report['input_rows'], view_rows=report['view_rows'],
        audio_checked_rows=report['audio_checked_rows'],
        ready_for_full_human_training=report['ready_for_full_human_training'],
        output=str(args.output)), indent=2))
    return 2 if args.require_ready and not receipt['readiness_passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
