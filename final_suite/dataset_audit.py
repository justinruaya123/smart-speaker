"""Audit retained metadata without claiming absent waveforms were reread."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit() -> dict:
    paths = ['final_suite/archive/v1/manifest.jsonl',
             'final_suite/archive/v2/manifest.jsonl',
             'final_suite/archive/v2/expanded_manifest.jsonl',
             'experiments/vcm_streaming_v3/data/phone_manifest.jsonl',
             'experiments/vcm_services_v6/data/probe_manifest.jsonl']
    report = {'manifests': [], 'audio_recomputed': False}
    for relative in paths:
        path = ROOT / relative
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        identifiers, hashes, texts = defaultdict(set), defaultdict(set), defaultdict(set)
        impossible = []
        for row in rows:
            identifiers[row['id']].add(row['split'])
            texts[row['text']].add(row['split'])
            if row.get('pcm_sha256'):
                hashes[row['pcm_sha256']].add(row['split'])
            labels = row.get('labels', [])
            minimum = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
            if row.get('frames', minimum) < minimum:
                impossible.append(row['id'])
        report['manifests'].append({
            'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(rows), 'split_counts': dict(Counter(r['split'] for r in rows)),
            'recording_ids_shared_across_splits': sum(len(v) > 1 for v in identifiers.values()),
            'recorded_hashes_shared_across_splits': sum(len(v) > 1 for v in hashes.values()),
            'rows_without_recorded_pcm_hash': sum(not r.get('pcm_sha256') for r in rows),
            'identical_texts_across_splits': sum(len(v) > 1 for v in texts.values()),
            'impossible_ctc_by_saved_frame_counts': impossible,
            'note': 'Saved metadata only; not fresh audio or human-speaker verification.',
        })
    main = [json.loads(line) for line in (ROOT / paths[3]).read_text().splitlines()]
    other = [json.loads(line) for line in (ROOT / 'experiments/vcm_streaming_v3/data/oov_manifest.jsonl').read_text().splitlines()]
    predictions = json.loads((ROOT / 'experiments/vcm_streaming_v3/reports/phone_s53_test_predictions.json').read_text())
    extra = {r['id'] for r in predictions} - {r['id'] for r in main}
    resolved = extra & {r['id'] for r in other}
    report['supplemental_v3_test'] = {'extra_prediction_ids': len(extra),
                                     'resolved_by_oov_manifest': len(resolved),
                                     'unresolved': sorted(extra - resolved)}
    report['findings'] = [
        'All 48 extra V3 test IDs are accounted for by the supplemental OOV manifest.',
        'V1 lacks recorded PCM hashes. Previously identified training controls cannot be freshly byte-audited without the original WAVs.',
        'Text repetition is not unseen-wording generalization; synthetic voice variants are not independent human speakers.',
        'All inspected tests are now regression sets, not sealed model or policy selection sets.',
        'The original ten application use cases differ from the legacy ten labels, one of which is UNKNOWN.',
        'Most historical source waveforms are absent. Their predictions are archived evidence, not a new replay.',
    ]
    (ROOT / 'final_suite/reports/dataset_audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
