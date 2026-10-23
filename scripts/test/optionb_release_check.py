"""Read-only epoch44/adapter parity and exposed human regression replay.

Never trains, selects a policy, edits a reference or qualifies a release.
"""
from pathlib import Path
import json
from typing import Any


def validate_inputs(protocol, *, root=Path('.')) -> dict[str, Any]:
    """Bind consumed paths and compare full cohorts to separately frozen originals."""
    from collections import Counter
    from scripts.test.speaker_demo_suite import read_jsonl, file_digest, canonical
    root = Path(root)
    splits = ('train', 'dev', 'calibration', 'test', 'holdout')
    manifests, originals = protocol['manifest_files'], protocol['canonical_manifests']
    if set(manifests) != set(splits) or set(originals) != set(splits):
        raise ValueError('Every fitting/selection/evaluation manifest needs a canonical binding')
    bindings = protocol['bindings']
    consumed = {*manifests.values(), *originals.values(), protocol['wording_metadata'], protocol['cached_predictions']}
    if consumed - bindings.keys():
        raise ValueError('Unbound consumed input: ' + str(sorted(consumed - bindings.keys())))
    for name, expected in bindings.items():
        if file_digest(root / name) != expected:
            raise ValueError('Frozen input identity mismatch: ' + name)
    data = {s: read_jsonl(root / manifests[s]) for s in splits}
    canonical_data = {s: read_jsonl(root / originals[s]) for s in splits}
    for split in splits:
        for cohort in (data[split], canonical_data[split]):
            keys = [r['id'] for r in cohort]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate canonical or consumed recording: ' + split)
            if any(r['split'] != split for r in cohort):
                raise ValueError('Manifest split differs from its canonical role')
        if Counter(map(canonical, data[split])) != Counter(map(canonical, canonical_data[split])):
            raise ValueError('Consumed manifest differs from frozen canonical cohort: ' + split)
    expected_keys = [(s, r['id']) for s in ('test', 'holdout') for r in canonical_data[s] if r['kind'] == 'human']
    for split in ('test', 'holdout'):
        count = sum(s == split for s, _ in expected_keys)
        expected_count = protocol['canonical_human_counts'][split]
        if type(expected_count) is not int or expected_count <= 0 or count != expected_count:
            raise ValueError('Empty or changed frozen canonical human cohort: ' + split)
    references = read_jsonl(root / protocol['wording_metadata'])
    metadata = {(r['split'], r['id']): r for r in references}
    if len(metadata) != len(references):
        raise ValueError('Duplicate full-reference metadata')
    predictions = [r for r in read_jsonl(root / protocol['cached_predictions'])
                   if r['variant'] == 'original' and r['kind'] == 'human']
    prediction_keys = [(r['split'], r['parent_id']) for r in predictions]
    if Counter(prediction_keys) != Counter(expected_keys):
        raise ValueError('Cached original predictions differ from canonical human cohort')
    cached = dict(zip(prediction_keys, predictions))
    for split in ('test', 'holdout'):
        for row in data[split]:
            if row['kind'] != 'human':
                continue
            key = (split, row['id'])
            if key not in metadata:
                raise ValueError('Missing full-reference metadata')
            for field in ('location', 'sha256', 'pcm_sha256', 'class_index', 'expected', 'metric_eligible'):
                if metadata[key][field] != row[field]:
                    raise ValueError('Wording metadata identity differs: ' + field)
    return dict(data=data, metadata=metadata, cached=cached, expected_human_keys=expected_keys)


def preflight(protocol, *, root=Path('.'), split='holdout'):
    """No inference: verify canonical membership, references, original WAV and PCM."""
    import hashlib
    from scripts.test.speaker_demo_suite import read_audio, canonical
    if split not in ('test', 'holdout'):
        raise ValueError('Preflight supports only frozen evaluation splits')
    checked = validate_inputs(protocol, root=root)
    expected_keys = [key for key in checked['expected_human_keys'] if key[0] == split]
    rows = []
    for row in checked['data'][split]:
        if row['kind'] != 'human':
            continue
        pcm = read_audio(row, root=Path(root))
        previous = checked['cached'][split, row['id']]['event']
        rows.append(dict(id=row['id'], split=split, audio=str((Path(root) / row['audio']).resolve()),
                         sha256=row['sha256'], pcm_sha256=row['pcm_sha256'], pcm_samples=len(pcm),
                         expected_output=previous['output'], expected_application=previous['application']))
    return dict(format='optionb-native-input-proof-v2', rows=rows,
                expected_keys=[list(key) for key in expected_keys], expected_count=len(expected_keys),
                expected_keys_sha256=hashlib.sha256(canonical(sorted(expected_keys)).encode()).hexdigest(),
                wav_and_decoded_pcm_verified=True, bindings_verified=len(protocol['bindings']),
                bundle=protocol.get('bundle'), source_commit=protocol.get('source_commit'))


def run_check(protocol_path, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Preserve all prior evidence; choose a new output directory')
    from collections import Counter
    import csv
    import numpy as np
    import torch
    from scripts.models.speaker_optionb_tiny import OptionBTinyVCM, CONFIG
    from scripts.models.vcm import VCM
    from scripts.runtime.optionb_worker import FrozenOptionBRuntime, OptionBAudioService, BUNDLE_SHA256
    from scripts.test.speaker_demo_suite import read_json, read_jsonl, read_audio, write_new, file_digest
    from scripts.test.speaker_optionb_tiny_demo_report import wording_rows
    from scripts.train.speaker_optionb_tiny import complete_class_metrics, check_bindings

    protocol_path = Path(protocol_path)
    protocol = read_json(protocol_path)
    check_bindings(protocol['bindings'])
    if (protocol['format'] != 'optionb-release-freeze-v2'
            or protocol['policy'] != {'enabled': True, 'min_probability': .50, 'min_margin': .10}
            or protocol['parameters'] != 258912 or protocol['selected_epoch'] != 44):
        raise ValueError('Frozen architecture/policy contract changed')
    checked = validate_inputs(protocol)
    destination.mkdir(parents=True)
    runtime = FrozenOptionBRuntime(protocol['bundle'], threads=1)
    torch.set_num_threads(1)
    checkpoint = torch.load(protocol['checkpoint'], map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != 44:
        raise ValueError('The selected parent must remain epoch44')
    model = OptionBTinyVCM().eval()
    model.load_state_dict(checkpoint['state_dict'])
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 258912:
        raise ValueError('Complete model parameter count changed')
    # Count only: no optimization, export or shipment of the width320 alternative.
    wider = VCM(**{**CONFIG, 'width': 320})
    wider_parameters = sum(p.numel() for p in wider.parameters())
    if wider_parameters != 927392:
        raise ValueError('The comparison parameter count differs')
    del wider
    rng = np.random.default_rng(53)
    parity = []
    for batch, frames in ((1, 1), (1, 97), (2, 258), (1, 401)):
        values = rng.normal(size=(batch, 40, frames)).astype(np.float32)
        lengths = np.array([frames - 13 * i for i in range(batch)], dtype=np.int64)
        with torch.inference_mode():
            expected = model(torch.from_numpy(values), torch.from_numpy(lengths)).numpy()
        actual = runtime.session.run(['logits'], {'features': values, 'lengths': lengths})[0]
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
        if not np.array_equal(actual.argmax(-1), expected.argmax(-1)):
            raise ValueError('Artifact parity changes a complete-command class')
        parity.append({'batch': batch, 'frames': frames, 'lengths': lengths.tolist(),
                       'max_abs_logit_error': float(np.max(np.abs(actual - expected)))})
    delivered = []
    service = OptionBAudioService(runtime, delivered.append,
        settings={'inference': {'intent_threshold': .99, 'slot_threshold': .99}, 'actions': {'enabled': True}})
    faults = [np.zeros(16000), np.full(16000, np.nan), np.ones(399), np.ones(240001),
              np.ones((400, 2)), np.full(400, 1.01)]
    for pcm in faults:
        service.forward(runtime.recognize(pcm))
        event = next(e for e in delivered if e['event'] == 'command')
        if event['application']['accepted'] or event['application']['reason'] != 'capture_fault' or event['actions_enabled']:
            raise ValueError('Capture rejection or action boundary changed')
        delivered.clear()

    data = checked['data']
    fitting = [r for r in data['train'] if r['kind'] == 'human']
    development = [r for s in ('dev', 'calibration') for r in data[s] if r['kind'] == 'human']
    holdout = [r for r in data['holdout'] if r['kind'] == 'human']
    disjoint = {}
    for field in ('id', 'speaker', 'sha256', 'pcm_sha256'):
        reserved = {r[field] for r in holdout}
        used = {r[field] for r in [*fitting, *development]}
        disjoint[field] = len(reserved & used)
        if disjoint[field]:
            raise ValueError('Holdout overlaps training/development: ' + field)
    metadata, cached = checked['metadata'], checked['cached']
    records = []
    predictions_path = destination / 'adapter-predictions.jsonl'
    with predictions_path.open('x', encoding='utf-8') as stream:
        for split in ('test', 'holdout'):
            for row in data[split]:
                if row['kind'] != 'human':
                    continue
                key = (split, row['id'])
                ref = metadata[key]
                for field in ('location', 'sha256', 'pcm_sha256', 'class_index', 'expected', 'metric_eligible'):
                    if ref[field] != row[field]:
                        raise ValueError('Wording metadata identity differs: ' + field)
                read_audio(row)  # Verify original WAV bytes AND decoded PCM before adapter replay.
                service.wav(Path(row['audio']).read_bytes())
                event = next(e for e in delivered if e['event'] == 'command')
                delivered.clear()
                previous = cached[key]['event']
                if event['output'] != previous['output'] or event['application'] != previous['application']:
                    raise ValueError('Adapter changed the frozen decision: ' + row['id'])
                if 'logits' in event:
                    np.testing.assert_allclose(event['logits'], previous['logits'], rtol=3e-5, atol=3e-5)
                if event['actions_enabled'] or event['qualified_output']['accepted']:
                    raise ValueError('Preview replay crossed the action/qualification boundary')
                record = dict(id=row['id'], parent_id=row['id'], split=split, kind='human', variant='original',
                    metric_eligible=row['metric_eligible'], class_index=row['class_index'], scope=row['scope'],
                    raw_index=int(event.get('command_index', 0)),
                    accepted_index=int(event.get('command_index', 0)) if event['output']['accepted'] else 0, event=event)
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
                if len(records) % 100 == 0:
                    stream.flush()
                    print('Verified original human adapter replays:', len(records), flush=True)
    expected_ids = Counter(checked['expected_human_keys'])
    if Counter((r['split'], r['id']) for r in records) != expected_ids:
        raise ValueError('Original human replay cohort changed')
    table = [r for r in wording_rows(records, metadata) if r['kind'] == 'human']
    with (destination / 'per-wording.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    summary = {}
    for split in ('test', 'holdout'):
        all_rows = [r for r in records if r['split'] == split]
        rows = [r for r in all_rows if r['metric_eligible']]
        wording = [r for r in table if r['split'] == split]
        summary[split] = dict(original_human_rows=len(all_rows), ambiguous_unscored=len(all_rows) - len(rows),
            metrics=complete_class_metrics(rows, [r['accepted_index'] for r in rows]),
            wording={k: sum(r[k] for r in wording) for k in
                     ('metadata_exact_references', 'correct_accepted', 'wrong_accepted', 'rejected')},
            holdout_mistakes=[{'id': r['id'], 'reference_text': metadata[split, r['id']]['reference_text'],
                              'reference_class': r['class_index'], 'accepted_class': r['accepted_index']}
                             for r in rows if split == 'holdout' and r['accepted_index'] > 0 and r['accepted_index'] != r['class_index']])
    check_bindings(protocol['bindings'])
    result = dict(format='optionb-release-parity-v2', source_commit=protocol['source_commit'],
        protocol_sha256=file_digest(protocol_path), parameters=parameters, width320_count_only=wider_parameters,
        selected_epoch=44, bundle_sha256=BUNDLE_SHA256, policy=runtime.policy,
        checkpoint_export_parity=parity, capture_fault_cases=len(faults),
        original_human_adapter_replays=len(records), exact_original_cohort_and_decisions_preserved=True,
        independently_pinned_canonical_cohort=True,
        consumed_manifest_bindings={p: protocol['bindings'][p] for p in protocol['manifest_files'].values()},
        canonical_manifest_bindings={p: protocol['bindings'][p] for p in protocol['canonical_manifests'].values()},
        holdout_declared_speakers=len({r['speaker'] for r in holdout}),
        holdout_train_development_overlap=disjoint, cohorts=summary,
        predictions_sha256=file_digest(predictions_path), wording_table_sha256=file_digest(destination / 'per-wording.csv'),
        optimizer_updates=0, selection_changes=0, exposed_regression_not_fresh_blind=True,
        independent_listening_completed=False, fresh_unseen_speaker_check_completed=False,
        physical_pi_completed=False, live_microphone_completed=False, promoted=False, actions_enabled=False)
    write_new(destination / 'results.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('experiments/speaker-optionb-release-v1/protocol-v3.json'))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        from scripts.test.speaker_demo_suite import read_json, file_digest
        value = preflight(read_json(args.protocol))
        value['protocol_sha256'] = file_digest(args.protocol)
        print(json.dumps(value, allow_nan=False))
    else:
        run_check(args.protocol, args.output)
