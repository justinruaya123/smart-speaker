"""Versioned original-human recovery with head-specific supervision.

Missing pronunciations are not missing intent labels. Disputed references remain
untrained; no waveform augmentation, test relocation or synthetic speech.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from scripts.runtime.pcv_schema import canonical
from scripts.train.pcv_updated_prepare import annotate, assign_split, pair_identity
from scripts.train.speaker_human_data import (
    LABELS, SPLITS, audio_diagnostics, file_sha256, human_supervision, read_rows,
    split_leakage, write_json, write_rows,
)


def candidate_from_metadata(metadata, location, historical_exclusion=None):
    """Create a proposal, not an admission, using exact pinned row identity."""
    loc = copy.deepcopy(location)
    name = Path(loc['file']).name
    index = loc['row']
    if (not re.fullmatch(r'data/train-\d{5}-of-\d{5}\.parquet', loc['file'])
            or type(index) is not int or index < 0
            or not re.fullmatch(r'[0-9a-f]{64}', loc['sha256'])
            or not re.fullmatch(r'[0-9a-f]{40}', loc['revision'])):
        raise ValueError('Malformed pinned human training location')
    meta = copy.deepcopy(metadata)
    if type(meta.get('is_synthetic')) not in (int, bool) or meta['is_synthetic'] != 0:
        raise ValueError('Expected explicitly original human publisher metadata')
    meta['published_split'] = 'train'
    annotation = annotate(meta)
    split = assign_split(meta)
    identity = loc['repository'] + '/primary/' + Path(name).stem + '/' + f'{index:06d}'
    original_id = 'train/' + Path(name).stem + '/' + f'{index:06d}'
    old = historical_exclusion or {}
    if old and (old['id'] != original_id or old['split'] != split):
        raise ValueError('Historical exclusion does not match the exact source row/split')
    speaker = str(meta.get('source', '')) + '/' + str(meta.get('speaker_id', ''))
    review_intent = {'VOLUME_UP': 'VOLUME', 'VOLUME_DOWN': 'VOLUME',
                     'OUT_OF_SCOPE': 'UNKNOWN'}.get(meta.get('command'), meta.get('command'))
    row: dict = dict(id=identity, location=loc, split=split, kind='human', synthetic=False,
        speaker='speaker-' + hashlib.sha256(speaker.encode()).hexdigest()[:24],
        source=loc['repository'], publisher_source=meta.get('source', 'unknown'),
        publisher_command=meta.get('command'), publisher_transcript=meta.get('transcript', ''),
        review_intent=review_intent if review_intent in LABELS else 'UNKNOWN',
        publisher_slot_value=meta.get('slot_value', ''),
        publisher_whisper_check=meta.get('whisper_check', ''),
        publisher_out_of_scope=meta.get('out_of_scope', 0),
        transcript_source=meta.get('transcript_source', 'unknown'),
        source_speaker_key=hashlib.sha256(canonical([
            loc['repository'], loc['revision'], meta.get('source'), str(meta.get('speaker_id', '')),
        ]).encode()).hexdigest(),
        pair=hashlib.sha256(canonical([
            loc['repository'], loc['revision'], pair_identity(meta),
        ]).encode()).hexdigest(),
        text=annotation['text'], expected=annotation['expected'],
        reference_status=annotation['reference_status'], reference_reason=annotation['reason'],
        slot_metadata_conflict=annotation['slot_metadata_conflict'],
        historical_exclusion=copy.deepcopy(old), phones=[],
        metric_eligible=False, training_eligible=False,
        listening_status='not_independently_reviewed',
        recovery_status='review_required', loss_mask=dict(ctc=False, intent=False, target=False))
    # A documented lexical exclusion does not license a new semantic label.
    if (old.get('reason') == 'missing_pronunciation'
            and old.get('reference_status') == annotation['reference_status']
            and annotation['reference_status'] in ('supported', 'unsupported')
            and not annotation['slot_metadata_conflict']
            and meta.get('transcript_source') == 'dataset'
            and meta.get('is_synthetic') in (0, False)
            and meta.get('speaker_id') not in (None, '', 'unknown')
            and split == 'train'):
        row['recovery_status'] = 'metadata_semantic_candidate'
        row['loss_mask']['intent'] = True
    return human_supervision(row)


def recover_dataset(audit_root, source_root, historical_path, output, *, protected_paths=()):
    """Recover only documented lexical exclusions; retain all disputed rows.

    The source shard binds payload identity. Only train PCM is decoded; frozen
    development/calibration/test/holdout membership and bytes remain unchanged.
    """
    import io
    import pyarrow.parquet as pq
    import soundfile as sf
    from threadpoolctl import threadpool_limits
    from scripts.test.speaker_demo_suite import pcm_digest
    from vcm.frontend import features, SPEC

    audit_root, source_root, output = Path(audit_root), Path(source_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((audit_root / 'audit.json').read_text(encoding='utf-8'))
    if parent['split_leakage'] or parent['conflicting_pcm_labels']:
        raise ValueError('Unsafe parent split/annotation audit')
    paths = [Path(historical_path), audit_root / 'audit.json', *map(Path, protected_paths)]
    paths += [audit_root / name for name in parent['outputs']]
    if (audit_root / 'exclusions.jsonl').exists():
        paths.append(audit_root / 'exclusions.jsonl')
    protected = {str(p): file_sha256(p) for p in paths}
    for name, digest in parent['outputs'].items():
        if file_sha256(audit_root / name) != digest:
            raise ValueError('Parent manifest checksum mismatch: ' + name)
    views = {s: read_rows(audit_root / (s + '.jsonl')) for s in SPLITS}
    known = [r for rows in views.values() for r in rows]
    known += [r for r in read_rows(audit_root / 'exclusions.jsonl')
              if r.get('synthetic') is False and r.get('kind') == 'human']
    known_by_id = {r['id']: r for r in known}
    historical = {r['id']: r for r in json.loads(Path(historical_path).read_text(encoding='utf-8'))}
    expected_ids = {r['id'] for r in parent['metadata_receipt']['omitted_rows']}
    candidates, shards = [], []
    for spec in parent['metadata_receipt']['shards']:
        if not Path(spec['file']).name.startswith('train-'):
            continue
        path = (source_root / spec['file']).resolve()
        if not path.is_relative_to(source_root) or file_sha256(path) != spec['sha256']:
            raise ValueError('Publisher shard checksum/path mismatch: ' + spec['file'])
        parquet = pq.ParquetFile(path)
        metadata = parquet.read(columns=[c for c in parquet.schema_arrow.names
                                        if c not in ('audio', 'whisper_transcript')]).to_pylist()
        prefix = spec['repository'] + '/primary/' + path.stem + '/'
        wanted = {}
        for index, meta in enumerate(metadata):
            identity = prefix + f'{index:06d}'
            if meta.get('is_synthetic') not in (0, False) or identity in known_by_id:
                continue
            if identity not in expected_ids:
                raise ValueError('Omitted publisher inventory differs from parent audit')
            old_id = 'train/' + path.stem + '/' + f'{index:06d}'
            loc = {k: spec[k] for k in ('repository', 'revision', 'file', 'sha256')}
            row = candidate_from_metadata(meta, {**loc, 'row': index}, historical.get(old_id))
            row['review_reasons'] = [row['historical_exclusion'].get('reason', 'absent_from_fit_view')]
            if row['slot_metadata_conflict']:
                row['review_reasons'].append('conflicting_slot_metadata')
            if row['transcript_source'] == 'whisper':
                row['review_reasons'].append('ASR_transcript_not_independently_verified')
            candidates.append(row)
            wanted[index] = row
        shards.append((path, wanted, spec['sha256']))
    if len(candidates) != len(expected_ids) or {r['id'] for r in candidates} != expected_ids:
        raise ValueError('Omitted publisher count/identity mismatch')
    overlaps = split_leakage([*known, *candidates])
    reserved = {identity for overlap in overlaps for identity in overlap['ids']}
    for row in candidates:
        if row['id'] in reserved:
            row['recovery_status'] = 'review_required'
            row['loss_mask'] = dict(ctc=False, intent=False, target=False)
            row['review_reasons'].append('reserved_or_cross_split_identity')
    audio_checks = []
    for path, wanted, digest in shards:
        offset = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['audio']):
            for index, payload in enumerate(batch.column('audio').to_pylist(), start=offset):
                row = wanted.get(index)
                if row is None or row['split'] != 'train' or row['id'] in reserved:
                    continue
                raw = payload['bytes']
                row['sha256'] = hashlib.sha256(raw).hexdigest()
                audio = output / 'audio' / (row['sha256'] + '.wav')
                audio.parent.mkdir(parents=True, exist_ok=True)
                if not audio.exists():
                    with audio.open('xb') as stream:
                        stream.write(raw)
                row['audio'] = str(audio)
                try:
                    pcm, rate = sf.read(io.BytesIO(raw), dtype='float32', always_2d=False)
                    row['pcm_sha256'] = pcm_digest(pcm, rate)
                    diagnostic = audio_diagnostics(pcm, rate)
                    if set(diagnostic['flags']) & {'below_frontend_window', 'exceeds_runtime_15_seconds',
                                                  'no_detectable_energy', 'possible_clipping', 'dc_offset'}:
                        row['recovery_status'] = 'review_required'
                        row['loss_mask'] = dict(ctc=False, intent=False, target=False)
                        row['review_reasons'].append('audio_quality_review')
                    if row['recovery_status'] == 'metadata_semantic_candidate':
                        with threadpool_limits(limits=1, user_api='blas'):
                            values = features(pcm)
                        target = output / 'features' / (row['pcm_sha256'] + '.npy')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            with target.open('xb') as stream:
                                np.save(stream, values, allow_pickle=False)
                        row.update(features=str(target), features_sha256=file_sha256(target),
                                   frames=values.shape[1], feature_origin='source_rederived_one_thread')
                    audio_checks.append(dict(id=row['id'], split='train', **diagnostic))
                except (ValueError, RuntimeError) as error:
                    row['recovery_status'] = 'review_required'
                    row['loss_mask'] = dict(ctc=False, intent=False, target=False)
                    row['review_reasons'].append('invalid_audio')
                    audio_checks.append(dict(id=row['id'], split='train', error=str(error)))
            offset += len(batch)
        if file_sha256(path) != digest:
            raise ValueError('Publisher shard checksum changed during recovery')
    # Include new PCM identities and every excluded frozen human reference.
    overlaps = split_leakage([*known, *candidates])
    reserved = {identity for overlap in overlaps for identity in overlap['ids']}
    admitted, queue = [], []
    seen = {field: {r[field] for r in known if r.get(field)}
            for field in ('sha256', 'pcm_sha256', 'pair')}
    for row in candidates:
        safe = row['recovery_status'] == 'metadata_semantic_candidate' and row['id'] not in reserved
        if safe and any(row.get(field) in values for field, values in seen.items()):
            safe = False
            row['review_reasons'].append('duplicate_recording_or_pair')
        if safe:
            row.update(recovery_status='admitted_semantic_only', metric_eligible=True, training_eligible=True)
            admitted.append(row)
            for field in seen:
                seen[field].add(row[field])
        else:
            row.update(recovery_status='review_required', metric_eligible=False, training_eligible=False,
                       loss_mask=dict(ctc=False, intent=False, target=False))
            queue.append(row)
    write_rows(output / 'recovered-train.jsonl', admitted)
    write_rows(output / 'train.jsonl', [*views['train'], *admitted])
    for split in SPLITS[1:]:
        with (output / (split + '.jsonl')).open('xb') as stream:
            stream.write((audit_root / (split + '.jsonl')).read_bytes())
    write_rows(output / 'review-queue.jsonl', queue)
    write_rows(output / 'all-candidates.jsonl', candidates)
    write_rows(output / 'audio-audit.jsonl', audio_checks)
    result = dict(format='speaker-human-recovery-v1', frontend=SPEC,
        parent_audit_sha256=protected[str(audit_root / 'audit.json')],
        parent_train_rows=len(views['train']), omitted_rows=len(candidates),
        omitted_by_split=dict(Counter(r['split'] for r in candidates)),
        historical_exclusions=dict(Counter(r['historical_exclusion'].get('reason', 'absent_from_fit_view') for r in candidates)),
        admitted_semantic_rows=len(admitted), train_rows=len(views['train']) + len(admitted),
        review_rows=len(queue), review_by_reason=dict(Counter(reason for r in queue for reason in r['review_reasons'])),
        per_intent={label: dict(
            omitted=sum(r['review_intent'] == label for r in candidates),
            admitted=sum((r['expected'] or {}).get('intent', 'UNKNOWN') == label for r in admitted),
            review=sum(r['review_intent'] == label for r in queue)) for label in LABELS},
        audio_checked_rows=len(audio_checks), decoded_audio_splits=['train'],
        split_leakage=overlaps, independently_listened_rows=0,
        full_training_ready=False, qualification_granted=False, synthetic_rows_added=0,
        protected_files=protected, original_audio_modified=False,
        note='Metadata-supported semantic recovery only; no new CTC pronunciation or independently verified spoken evidence.')
    if protected != {name: file_sha256(name) for name in protected}:
        raise ValueError('Protected input changed during recovery')
    result['outputs'] = {p.name: file_sha256(p) for p in output.glob('*.jsonl')}
    write_json(output / 'recovery.json', result)
    return result


def masked_optimization_step(model, rows, arrays, optimizer, *, device='cpu', microbatch=8) -> dict:
    """Normalize each head by its own labelled count, not total batch size.

    Unavailable CTC is omitted, never treated as an empty/blank-only target.
    Validate every row before zeroing gradients or updating weights.
    """
    import torch

    if not rows or type(microbatch) is not int or microbatch < 1:
        raise ValueError('Nonempty human batch and positive microbatch required')
    masks = []
    for row in rows:
        semantic = human_supervision(row)
        mask = row.get('loss_mask', dict(ctc=True, intent=True, target=row['target_index'] != -100))
        values = arrays[row['id']]
        unsafe = (row.get('kind') != 'human' or row.get('synthetic') is not False
                  or row.get('split') != 'train' or row.get('training_eligible') is not True
                  or row.get('metric_eligible') is not True
                  or row.get('reference_status') in ('ambiguous', 'auxiliary')
                  or any(semantic[k] != row[k] for k in ('intent_index', 'target_index'))
                  or set(mask) != {'ctc', 'intent', 'target'}
                  or any(type(v) is not bool for v in mask.values()) or not any(mask.values())
                  or values.ndim != 2 or values.shape[0] != 40 or not values.shape[1]
                  or not np.isfinite(values).all())
        phones = row.get('phones') or []
        needed = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
        if (unsafe or (mask['target'] and row['target_index'] == -100)
                or (mask['ctc'] and (not phones or (values.shape[1] + 1) // 2 < needed
                                    or any(type(p) is not int or not 1 <= p < 40 for p in phones)))
                or (not mask['ctc'] and phones)):
            raise ValueError('Unsafe human head supervision: ' + row['id'])
        masks.append(mask)
    counts = {head: sum(mask[head] for mask in masks) for head in ('ctc', 'intent', 'target')}
    optimizer.zero_grad(set_to_none=True)
    losses = np.zeros(3)
    for offset in range(0, len(rows), microbatch):
        batch, bm = rows[offset:offset + microbatch], masks[offset:offset + microbatch]
        lengths = [arrays[r['id']].shape[1] for r in batch]
        values = np.zeros((len(batch), 40, max(lengths)), dtype=np.float32)
        for i, row in enumerate(batch):
            values[i, :, :lengths[i]] = arrays[row['id']]
        ctc, intent, target, _ = model(torch.from_numpy(values).to(device),
                                       torch.tensor(lengths, device=device))
        loss = ctc.sum() * 0
        for h, (head, logits, weight) in enumerate((('ctc', ctc, 1.), ('intent', intent, .3), ('target', target, .15))):
            chosen = [i for i, mask in enumerate(bm) if mask[head]]
            if not chosen:
                continue
            if head == 'ctc':
                labels = [batch[i]['phones'] for i in chosen]
                term = torch.nn.functional.ctc_loss(
                    logits[chosen].log_softmax(1).permute(2, 0, 1),
                    torch.tensor([p for label in labels for p in label], dtype=torch.long, device=device),
                    torch.tensor([(lengths[i] + 1) // 2 for i in chosen]),
                    torch.tensor([len(label) for label in labels]), blank=0, zero_infinity=False)
            else:
                term = torch.nn.functional.cross_entropy(logits[chosen, :, 0],
                    torch.tensor([batch[i][head + '_index'] for i in chosen], device=device))
            fraction = len(chosen) / counts[head]
            loss = loss + weight * term * fraction
            losses[h] += float(term.detach()) * fraction
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite head-masked human loss; optimizer not updated')
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return dict(losses=losses.tolist(), supervised_rows=counts)


def adjudicate_review(row, reviews):
    """Require consistent, separately identified listener attestations.

    A consensus creates only a semantic proposal. It never bypasses fresh audio,
    split, duplicate or input-identity checks and never fabricates CTC labels.
    """
    reviewers, signatures = set(), set()
    for review in reviews:
        if (not row.get('sha256') or review.get('id') != row['id']
                or review.get('sha256') != row['sha256']
                or review.get('location') != row['location']):
            raise ValueError('Review audio identity does not match the full pinned source')
        reviewer = review.get('reviewer_id')
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError('Explicit independent reviewer identity required')
        if reviewer in reviewers:
            raise ValueError('Duplicate reviewer is not an independent vote')
        if review.get('independent') is not True or review.get('listened') is not True:
            raise ValueError('Explicit independent listening attestation required')
        decision = review.get('decision')
        expected = review.get('expected')
        if (decision not in ('supported', 'unsupported', 'ambiguous')
                or (decision == 'supported') != (expected is not None)
                or not isinstance(review.get('transcript'), str)):
            raise ValueError('Incomplete or inconsistent review decision')
        human_supervision({**row, 'expected': expected})
        reviewers.add(reviewer)
        signatures.add(canonical({k: review.get(k) for k in
                                  ('decision', 'expected', 'transcript', 'boundary_status')}))
    result: dict = dict(status='awaiting_second_reviewer', candidate=None, reviewer_ids=sorted(reviewers))
    if len(reviewers) < 2:
        return result
    if (len(signatures) != 1 or reviews[0]['decision'] == 'ambiguous'
            or reviews[0].get('boundary_status') != 'complete'):
        result['status'] = 'disputed'
        return result
    candidate = human_supervision({**copy.deepcopy(row), 'expected': reviews[0]['expected']})
    candidate.update(phones=[], text=reviews[0]['transcript'],
        reference_status=reviews[0]['decision'], training_eligible=False, metric_eligible=False,
        loss_mask=dict(ctc=False, intent=True, target=False),
        listening_status='reviewer_attested_independent_consensus')
    result.update(status='consensus_semantic_proposal', candidate=candidate)
    return result


def masked_optimizer_smoke(recovery_root, output, *, architecture: dict | None = None):
    """One real human-only mixed-supervision update; not a trained candidate."""
    import torch
    from scripts.models.pcv_release_pooling import PooledContextVCM
    from scripts.train.speaker_human_data import fit_human_normalization

    recovery_root, output = Path(recovery_root), Path(output)
    receipt = json.loads((recovery_root / 'recovery.json').read_text(encoding='utf-8'))
    if receipt['split_leakage']:
        raise ValueError('Unsafe recovered split audit')
    for name, digest in receipt['outputs'].items():
        if file_sha256(recovery_root / name) != digest:
            raise ValueError('Recovered manifest checksum mismatch: ' + name)
    train = read_rows(recovery_root / 'train.jsonl')
    recovered = read_rows(recovery_root / 'recovered-train.jsonl')
    if not recovered:
        raise ValueError('No admitted semantic recovery to exercise')
    normalization = fit_human_normalization(train)
    recovered_ids = {r['id'] for r in recovered}
    batch = [r for r in train if r['id'] not in recovered_ids][:2]
    positive = [r for r in recovered if r['expected'] is not None]
    negative = [r for r in recovered if r['expected'] is None]
    batch += positive[:1] + negative[:1]
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
    result = masked_optimization_step(model, batch, arrays, optimizer, device='cpu', microbatch=2)
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    if not changed or not np.isfinite(result['losses']).all():
        raise ValueError('Head-masked smoke failed to produce a finite weight update')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'normalization-human.json', normalization)
    result.update(format='human-head-masked-optimizer-smoke-v1',
        device='cpu', parameters=sum(p.numel() for p in model.parameters()), architecture=architecture,
        seed=53, optimizer_updates=1, weights_changed=changed,
        batch_ids=[r['id'] for r in batch], normalization_human_rows=len(train),
        normalization_sha256=file_sha256(output / 'normalization-human.json'),
        training_manifest_sha256=receipt['outputs']['train.jsonl'],
        recovery_sha256=file_sha256(recovery_root / 'recovery.json'),
        synthetic_rows=0, augmentation=False, pretrained_weights=False,
        evaluation_audio_opened=False, selected_model_modified=False,
        accuracy_gain_claimed=False, full_training_completed=False)
    write_json(output / 'receipt.json', result)
    return result


def main(argv=None):
    import argparse

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='New immutable human-only research view')
    prepare.add_argument('--audit', type=Path, required=True)
    prepare.add_argument('--source-root', type=Path, required=True)
    prepare.add_argument('--historical-exclusions', type=Path,
                         default=root / 'experiments/pcv-release-train-v1/annotation-exclusions.json')
    prepare.add_argument('--protocol', type=Path,
                         default=root / 'experiments/speaker-human-recovery-v1/protocol.json')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--smoke-fit', action='store_true', help='One masked optimizer update, not full training')
    prepare.add_argument('--require-ready', action='store_true', help='Exit 2 while human qualification prerequisites are unmet')
    review = commands.add_parser('adjudicate', help='Validate independent votes; never edits a training manifest')
    review.add_argument('--queue', type=Path, required=True)
    review.add_argument('--reviews', type=Path, required=True)
    review.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'adjudicate':
        queue, votes = read_rows(args.queue), read_rows(args.reviews)
        ids = {r['id'] for r in queue}
        if len(ids) != len(queue) or any(v['id'] not in ids for v in votes):
            raise ValueError('Duplicate queue identity or review outside the exact queue')
        args.output.mkdir(parents=True, exist_ok=False)
        decisions = [dict(id=r['id'], **adjudicate_review(r, [v for v in votes if v['id'] == r['id']])) for r in queue]
        write_rows(args.output / 'adjudications.jsonl', decisions)
        summary = dict(queue_sha256=file_sha256(args.queue), reviews_sha256=file_sha256(args.reviews),
            rows=len(decisions), outcomes=dict(Counter(r['status'] for r in decisions)),
            training_manifest_written=False, automatic_admission=False)
        write_json(args.output / 'receipt.json', summary)
        print(json.dumps(summary, indent=2))
        return 0 if decisions and all(r['status'] == 'consensus_semantic_proposal' for r in decisions) else 2
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    protected = []
    for name, digest in protocol['bindings'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError('Frozen recovery protocol input changed: ' + name)
        protected.append(path)
    result = recover_dataset(args.audit, args.source_root, args.historical_exclusions,
                             args.output, protected_paths=protected)
    receipt: dict = dict(protocol_sha256=file_sha256(args.protocol),
        recovery_sha256=file_sha256(args.output / 'recovery.json'),
        readiness_passed=result['full_training_ready'], full_training_run=False,
        comparison_run=False, installed_artifacts_touched=False,
        comparison_blocked='Missing human command/calibration coverage and independent disputed-label review')
    if args.smoke_fit and not args.require_ready:
        receipt['optimizer_smoke'] = masked_optimizer_smoke(args.output, args.output / 'optimizer-smoke')
    write_json(args.output / 'preparation-receipt.json', receipt)
    print(json.dumps({k: result[k] for k in ('omitted_rows', 'admitted_semantic_rows', 'review_rows',
        'audio_checked_rows', 'train_rows', 'full_training_ready')}, indent=2))
    return 2 if args.require_ready and not result['full_training_ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
