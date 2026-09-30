"""Original-waveform preparation with frozen, relocatable feature identities."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from .contract import Contract, file_hash, identity, pronounce, read_jsonl, write_json
from .evidence import audit_recording_identities, validate_reference


from .paths import below


def verified_features(row: dict, root: Path) -> np.ndarray:
    path = below(root, row['features'])
    if file_hash(path) != row['features_sha256']:
        raise ValueError('Frozen feature bytes changed')
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[0] != 40 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid frontend feature array')
    return value


def batch_arrays(arrays: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    lengths = [a.shape[1] for a in arrays]
    output = np.zeros((len(arrays), 40, max(lengths)), dtype=np.float32)
    for index, array in enumerate(arrays):
        output[index, :, :lengths[index]] = array
    return output, lengths


def prepare(manifest: Path, audio_root: Path, output: Path,
            contract: Contract, lexicon_path: Path,
            confirmation_commitment: Path) -> dict:
    """Run on training/development/calibration originals, never sealed audio."""
    import hashlib
    import soundfile as sf
    from vcm.frontend import SPEC, features

    if Path(output).exists():
        raise FileExistsError('Preparation is immutable; choose a new output')
    rows = read_jsonl(manifest)
    if not rows or any(r.get('split') not in ('train', 'dev', 'calibration') for r in rows):
        raise ValueError('Final confirmation audio must remain with its custodian')
    commitment = json.loads(Path(confirmation_commitment).read_text())
    if (commitment.get('contract_sha256') != contract.sha256
            or not commitment.get('confirmation_manifest_sha256')
            or commitment.get('positive_recordings', 0) < 11400):
        raise ValueError('Missing frozen confirmation-set commitment')
    words = json.loads(Path(lexicon_path).read_text())['words']
    groups = defaultdict(list)
    for row in rows:
        validate_reference(row, contract)
        if row.get('synthetic') is not False or row.get('human_reviewed') is not True:
            raise ValueError('Only consented, reviewed, original human/nonspeech recordings')
        for key in ['recording_id', 'speaker_id', 'utterance_parent_id', 'source_id',
                    'source_license', 'consent_reference', 'audio', 'audio_sha256',
                    'pcm_sha256', 'condition_id', 'reference_class']:
            if not row.get(key):
                raise ValueError(f'Missing required recording field: {key}')
        phrase = contract.by_id.get(row.get('phrase_id'))
        rejection = contract.negative_by_id.get(row.get('phrase_id'))
        reference = phrase or rejection
        if reference and (row.get('text') != reference['exact_text']
                          or identity(row.get('expected_command')) != identity(reference.get('expected_command'))):
            raise ValueError('Recording reference differs from its fixed phrase')
        if row['reference_class'] == 'supported_positive' and phrase is None:
            raise ValueError('Positive lacks a required phrase ID')
        groups[(row['split'], row.get('phrase_id'))].append(row)
    audit = audit_recording_identities(rows)
    for split, recordings, speakers in [('train', 24, 12), ('dev', 8, 4), ('calibration', 8, 4)]:
        for phrase in contract.positive:
            selected = groups[(split, phrase['phrase_id'])]
            if len(selected) < recordings or len({r['speaker_id'] for r in selected}) < speakers:
                raise ValueError(f'BLOCKED_DATA: {split} {phrase["phrase_id"]} lacks independent coverage')
    final_speakers = set(commitment.get('speaker_identity_hashes', []))
    if not final_speakers or final_speakers & {identity(r['speaker_id']) for r in rows}:
        raise ValueError('Confirmation speakers must be explicitly committed and disjoint')
    Path(output).mkdir(parents=True)
    totals, squares, frames = np.zeros(40), np.zeros(40), 0
    prepared = []
    for index, row in enumerate(rows):
        audio = below(audio_root, row['audio'])
        if file_hash(audio) != row['audio_sha256']:
            raise ValueError('Original audio file identity mismatch')
        pcm, rate = sf.read(audio, dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Requires real mono 16 kHz PCM; never relabel a source rate')
        if hashlib.sha256(pcm.tobytes()).hexdigest() != row['pcm_sha256']:
            raise ValueError('Decoded PCM identity mismatch')
        phones = [] if row.get('nonspeech') is True else pronounce(row.get('text', ''), words)
        array = features(pcm)
        needed = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
        if array.shape[1] < max(1, needed):
            raise ValueError('Truncated or CTC-infeasible recording')
        feature_path = Path(output) / 'features' / f'{index:07}.npy'
        feature_path.parent.mkdir(exist_ok=True)
        np.save(feature_path, array)
        record = {**row, 'features': feature_path.relative_to(output).as_posix(),
                  'features_sha256': file_hash(feature_path), 'phones': phones,
                  'frontend_sha256': identity(SPEC), 'samples': len(pcm)}
        prepared.append(record)
        if row['split'] == 'train':
            totals += array.sum(axis=1, dtype=np.float64)
            squares += np.square(array.astype(np.float64)).sum(axis=1)
            frames += array.shape[1]
    mean = totals / frames
    std = np.sqrt(np.maximum(squares / frames - mean ** 2, 1e-4))
    write_json(Path(output) / 'normalization.json', {'mean': mean.tolist(), 'std': std.tolist(),
                                                   'fit_split': 'train', 'frontend_spec': SPEC})
    for split in ('train', 'dev', 'calibration'):
        records = [r for r in prepared if r['split'] == split]
        (Path(output) / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    write_json(Path(output) / 'confirmation.commitment.json', commitment)
    receipt = {'audit': audit, 'contract_sha256': contract.sha256,
               'input_manifest_sha256': file_hash(manifest),
               'artifacts': {p.name: file_hash(p) for p in Path(output).glob('*.json*')},
               'near_duplicate_manual_audit': 'REQUIRES_EXTERNAL_AUDIT_RECEIPT',
               'rights_review': 'SUPPLIED_CONSENT_REFERENCES_NOT_LEGAL_VERIFICATION'}
    write_json(Path(output) / 'freeze.json', receipt)
    return receipt
