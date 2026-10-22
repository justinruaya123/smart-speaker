"""Reporting-only recovery of full publisher references omitted by tiny views.

The first wording table correctly found no text in the projected manifests;
it did not measure absence of original-corpus wording evidence. Preserve it
and its receipt; enrich only a separate, versioned reporting view.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.test.speaker_optionb_tiny_demo import STUDY
from scripts.test.speaker_optionb_tiny_demo_report import COHORTS, wording_rows
from scripts.test.speaker_demo_suite import read_json, read_jsonl, file_digest, write_new
from scripts.train.speaker_optionb_tiny import check_bindings

SOURCE = Path('.cache/frontend-repro-20261019/source-view/.cache/pcv-source-refresh-v1/source')
LOCATOR_FIELDS = ('repository', 'revision', 'file', 'sha256', 'row')


def enrich_references(references, metadata):
    """Join by full pinned source locator, never variation/bucket or ID suffix."""
    enriched = {}
    for key, ref in references.items():
        locator = tuple(ref['location'][k] for k in LOCATOR_FIELDS)
        meta = metadata.get(locator)
        if (meta is None or type(meta.get('is_synthetic')) not in (int, bool)
                or meta['is_synthetic'] not in (0, 1)
                or bool(meta['is_synthetic']) is not ref['synthetic']):
            raise ValueError('Full source locator or declared human/synthetic provenance differs')
        text = meta.get('transcript')
        if text is None:
            text = ''
        if not isinstance(text, str):
            raise ValueError('Publisher full transcript must be text, not a variation fallback')
        enriched[key] = dict(ref, reference_text=text, transcript_source=meta.get('transcript_source'),
            reference_text_provenance='publisher_full_transcript_not_independently_listened')
    return enriched


def verify_metadata_report(study=STUDY, *, write_receipt=False):
    """Verify the fixed decisions, then add the separately bound wording table."""
    import pyarrow.parquet as pq

    study = Path(study)
    receipt_path = study / 'verification-wording-v2.json'
    if write_receipt and receipt_path.exists():
        raise FileExistsError('Preserve the earlier metadata-enriched wording receipt')
    base = read_json(study / 'verification.json')
    protocol = read_json(study / 'protocol.json')
    freeze = read_json(study / 'data-freeze.json')
    lock = read_json(study / 'selection-before-test.json')
    for bindings in (base['bindings'], freeze['bindings'], freeze['historical_bindings'], lock['bindings']):
        check_bindings(bindings)
    run = Path(protocol['outputs']['run'])
    references = {(r['split'], r['id']): r for split in ('test', 'holdout') for r in read_jsonl(run / (split + '.jsonl'))}
    groups = defaultdict(list)
    for ref in references.values():
        location = ref['location']
        if (location['repository'], location['revision']) != ('airimonda/ai231-me2-voice-commands', '25111444af3adff7588d86ab27c304bb080895cf'):
            raise ValueError('This reporting adapter requires the full pinned primary source identity')
        groups[tuple(location[k] for k in LOCATOR_FIELDS[:-1])].append(ref)
    metadata, sources = {}, {}
    for identity, members in groups.items():
        path = SOURCE / identity[2]
        if not path.resolve().is_relative_to(SOURCE.resolve()) or file_digest(path) != identity[3]:
            raise ValueError('Pinned source metadata path/digest mismatch')
        sources[path.as_posix()] = identity[3]
        # Do not decode audio or use an alternative ASR/variation reference.
        table = pq.ParquetFile(path).read(columns=['transcript', 'transcript_source', 'is_synthetic']).to_pylist()
        for ref in members:
            row = ref['location']['row']
            if type(row) is not int or not 0 <= row < len(table):
                raise ValueError('Exact source row is outside its pinned shard')
            metadata[(*identity, row)] = table[row]
    enriched = enrich_references(references, metadata)
    records = read_jsonl(run / 'final-predictions.jsonl')
    table = wording_rows(records, enriched)
    output = study / 'per-verbatim-cohorts-v2.csv'
    if not output.exists():
        if not write_receipt:
            raise FileNotFoundError('Create the exclusive metadata wording report with --write-receipt')
        with output.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    with output.open(encoding='utf-8', newline='') as stream:
        assert list(csv.DictReader(stream)) == [{k: '' if v is None else str(v) for k, v in r.items()} for r in table]
    summary = {}
    for split, kind in COHORTS:
        members = [r for r in table if r['split'] == split and r['kind'] == kind]
        summary[split + '/' + kind] = dict(definitions=len(members), represented=sum(r['metadata_exact_references'] > 0 for r in members),
            metadata_reference_matches=sum(r['metadata_exact_references'] for r in members),
            raw_correct=sum(r['raw_correct'] for r in members), correct_accepted=sum(r['correct_accepted'] for r in members),
            wrong_accepted=sum(r['wrong_accepted'] for r in members), rejected=sum(r['rejected'] for r in members))
    extra = [output, study / 'verification.json', Path(__file__).resolve().relative_to(Path.cwd()),
             Path('scripts/test/test_speaker_optionb_tiny_demo_metadata.py')]
    bindings = {**sources, **{p.as_posix(): file_digest(p) for p in extra}}
    for original in (base['bindings'], freeze['bindings'], freeze['historical_bindings'], lock['bindings'], bindings):
        check_bindings(original)
    receipt = dict(status='verified', reporting_addendum_only=True, original_v1_zero_text_table_and_receipt_preserved=True,
        projected_manifest_full_text_fields='omitted, not absent corpus evidence',
        source_full_transcripts=summary, original_reference_rows=len(references), wording_rows=len(table),
        source_metadata_shards=len(sources), source_columns_read=['transcript', 'transcript_source', 'is_synthetic'],
        audio_decoded=False, inference_or_training_performed=False, labels_or_policy_changed=False,
        source_metadata_not_independent_listener_truth=True, no_inference_time_wording_lookup=True, bindings=bindings)
    if write_receipt:
        write_new(receipt_path, receipt)
    elif receipt_path.exists():
        stored = read_json(receipt_path)
        check_bindings(stored['bindings'])
        assert stored == receipt
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['verify'])
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    verify_metadata_report(write_receipt=args.write_receipt)
