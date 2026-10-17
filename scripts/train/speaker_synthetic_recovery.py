"""Journal interrupted synthetic attempts without changing the frozen generator.

Run only after its generation process has stopped. Preserved but unjournaled
waveforms remain unaccepted evidence; the original generator advances to its
next already-declared attempt. This helper never synthesizes or adds labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.runtime.pcv_schema import canonical
from scripts.test.speaker_demo_suite import file_digest, read_audio, read_json, read_jsonl
from scripts.train.speaker_dataset_install import write_audio as publish_bytes
from scripts.train import speaker_synthetic_data as generation


def write_new(path, value):
    """Publish a complete journal atomically, preserving any existing bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(path, (canonical(value) + '\n').encode('utf8'))


def recover_attempt(protocol, generated, voice_id, batch_offset, attempt):
    protocol, generated = Path(protocol).resolve(), Path(generated).resolve()
    cfg = read_json(protocol)
    generation._verify(cfg['bindings'])
    plan = read_json(generation._path(cfg['generation_plan']))
    settings = cfg['generation']
    voice_rows = [r for r in plan if r['voice_id'] == voice_id]
    if (not voice_rows or type(batch_offset) is not int or batch_offset < 0
            or batch_offset >= len(voice_rows) or batch_offset % settings['batch_size']
            or type(attempt) is not int or not 0 <= attempt < settings['attempts']):
        raise ValueError('Recovery must identify a frozen voice, batch and attempt')
    batch = voice_rows[batch_offset:batch_offset + settings['batch_size']]
    stem = f'{voice_id}-{batch_offset:04d}-{attempt}'
    attempt_path = generated / 'attempts' / (stem + '.json')
    journal_path = generated / 'recovery-v1' / (stem + '.json')
    manifest = generated / 'generated.jsonl'
    existing = read_jsonl(manifest) if manifest.exists() else []
    generation.audit_generated(existing, plan, [])
    done = {r['id']: r for r in existing}
    for row in existing:
        read_audio(row, generation.ROOT)

    if journal_path.exists():
        journal = read_json(journal_path)
        generation._verify(journal['preserved_files'])
        if (journal['protocol_sha256'] != file_digest(protocol)
                or journal['implementation_sha256'] != file_digest(Path(__file__))
                or any(key not in done or generation._hash(done[key]) != digest
                       for key, digest in journal['completed_cells_before'].items())):
            raise ValueError('Recovery identity mismatch; preserve the journal')
        receipt = journal['attempt_receipt']
    else:
        if attempt_path.exists():
            raise FileExistsError('Original attempt receipt already exists; do not replace it')
        if (generated / 'generation-complete.json').exists():
            raise ValueError('A complete generation does not need recovery')
        first_index = next(i for i, row in enumerate(plan) if row['id'] == batch[0]['id'])
        if any(row['id'] not in done for row in plan[:first_index]):
            raise ValueError('An earlier batch is incomplete; recover its interruption first')
        if any(not (generated / 'attempts' / f'{voice_id}-{batch_offset:04d}-{a}.json').exists()
               for a in range(attempt)):
            raise ValueError('An earlier attempt has no receipt; do not skip its budget')
        if any((generated / 'attempts' / f'{voice_id}-{batch_offset:04d}-{a}.json').exists()
               for a in range(attempt + 1, settings['attempts'])):
            raise ValueError('A later attempt already exists; preserve it for investigation')
        incomplete = [r for r in batch if r['id'] not in done]
        if not incomplete:
            raise ValueError('This batch is already complete')
        preserved, cells = {}, []
        for row in incomplete:
            paths = {
                'raw_failure': generated / 'failed' / f'{stem}-{row["prompt_id"]}.npy',
                'native_unaccepted': generated / 'native' / voice_id / f'{row["prompt_id"]}-a{attempt}.wav',
                'derived_unaccepted': generation.ROOT / 'data/audio/speaker-synthetic-completion-v1'
                                      / voice_id / f'{row["prompt_id"]}-a{attempt}.wav',
            }
            evidence = {role: {'path': str(path.resolve()), 'sha256': file_digest(path)}
                        for role, path in paths.items() if path.exists()}
            preserved.update({r['path']: r['sha256'] for r in evidence.values()})
            cells.append({'id': row['id'], 'reason': 'interrupted_attempt_unaccepted',
                          'preserved_evidence': evidence})
        if not preserved:
            raise ValueError('No preserved incomplete-attempt evidence; do not invent an interruption')
        seed = int(generation._hash([cfg['experiment_id'], voice_id, batch_offset, attempt])[:8], 16)
        receipt = {'seed': seed, 'batch_ids': [r['id'] for r in batch], 'attempt': attempt,
                   'seconds': None, 'failures': cells,
                   'recovery': {'version': 1, 'journal': str(journal_path),
                                'rule': 'Preserve all unjournaled outputs as unaccepted; consume the interrupted attempt'}}
        journal = {'version': 1, 'status': 'interrupted_attempt_journaled', 'accepted_rows_added': 0,
                   'protocol_sha256': file_digest(protocol),
                   'implementation_sha256': file_digest(Path(__file__)),
                   'manifest_sha256_before': file_digest(manifest) if manifest.exists() else None,
                   'completed_cells_before': {r['id']: generation._hash(r) for r in existing},
                   'preserved_files': preserved, 'attempt_receipt': receipt}
        write_new(journal_path, journal)
    if attempt_path.exists():
        if read_json(attempt_path) != receipt:
            raise ValueError('Existing recovery receipt differs from its frozen journal')
    else:
        write_new(attempt_path, receipt)
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=generation.HERE / 'protocol.json')
    parser.add_argument('--generated', type=Path, default=generation.CACHE)
    parser.add_argument('--voice', required=True)
    parser.add_argument('--batch-offset', type=int, required=True)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    result = recover_attempt(args.protocol, args.generated, args.voice, args.batch_offset, args.attempt)
    print(json.dumps({'status': result['status'], 'accepted_rows_added': result['accepted_rows_added']}))


if __name__ == '__main__':
    main()
