"""Prepare, fit, lock, evaluate and export Iris's independent speaker backend.

Only speaker identifiers, source/split metadata and waveforms are consumed.
Original short commands, constructed overlap/switches and optional synthetic
wake pairs are explicitly proxy evidence, never human wake qualification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tarfile
import time

import numpy as np
import soundfile as sf

from scripts.train.iris_audio import DEFAULT, features, load_audio
from scripts.train.iris_owner_reference import PCM_NORMALIZATION, continuity_spans, owner_features, voiced_seconds

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = 'iris-release-v1'
SEED = 20261017
CALIBRATION_KIND = 'short_utterance_proxy+synthetic_wake_proxy'


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def dump(path: Path, value, *, immutable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Frozen evidence already exists: {path}')
        return
    path.write_text(content, encoding='utf8')


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fraction(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32


def base_view_directory(root: Path) -> Path:
    """Prefer the pinned portable view copy; never mix partial directories."""
    for relative in ('.cache/iris-release-v1/inputs/base-views', '.cache/vcm-mvp-hey-iris-v2/views'):
        path = root / relative
        if path.is_dir():
            missing = [name for name in ('train', 'dev', 'calibration', 'test', 'personal-test')
                       if not (path / (name + '.jsonl')).is_file()]
            if missing:
                raise ValueError('Incomplete owner base views: ' + ', '.join(missing))
            return path
    raise FileNotFoundError('Missing owner base views; run the pinned reproduction input preflight')


def historical_installed_integrity(root: Path) -> dict:
    """Historical preservation metadata; installed model files are never inputs."""
    protocol = root / 'experiments/iris-release-v1/owner-protocol.json'
    if protocol.exists():
        return dict(json.loads(protocol.read_text()).get('installed_preserved', {}))
    preflight = root / '.cache/iris-release-v1/preflight.json'
    if preflight.exists():
        frozen = json.loads(preflight.read_text()).get('installed_artifacts', {})
        return dict(frozen) if isinstance(frozen, dict) else {
            item['path']: item['sha256'] for item in frozen
            if item.get('path') in ('artifacts/bundle.json', 'artifacts/manifest.json')}
    return {}


def _canonical_rows_digest(path):
    sha, count = hashlib.sha256(), 0
    with path.open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            sha.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            sha.update(b'\n')
            count += 1
    return sha.hexdigest(), count


def _base_view_receipts(root, protocol):
    views = base_view_directory(root)
    expected = {Path(item['path']).name: item['sha256'] for item in protocol['data']['vcm_view_receipts']}
    proof_path = root / 'experiments/iris-release-v1/owner-base-view-equivalence.json'
    proof = {item['name']: item for item in json.loads(proof_path.read_text())['views']} if proof_path.exists() else {}
    result = []
    for name in ('train', 'dev', 'calibration', 'test', 'personal-test'):
        path = views / (name + '.jsonl')
        actual = digest(path)
        if actual != expected.get(path.name):
            equivalence = proof.get(path.name, {})
            canonical, count = _canonical_rows_digest(path)
            if (equivalence.get('original_sha256') != expected.get(path.name)
                    or equivalence.get('portable_sha256') != actual
                    or equivalence.get('canonical_rows_sha256') != canonical
                    or equivalence.get('row_count') != count):
                raise ValueError('Owner base view differs from frozen protocol: ' + path.name)
        result.append({'path': path.relative_to(root).as_posix(), 'sha256': actual})
    return result


def _verify_prepared(root, cache, prepared):
    for filename, key in (('manifest.jsonl', 'manifest_sha256'), ('synthetic-proxies.jsonl', 'synthetic_proxy_manifest_sha256')):
        if digest(cache / filename) != prepared[key]:
            raise ValueError('Frozen owner manifest changed: ' + filename)
    protocol_path = root / 'experiments/iris-release-v1/owner-protocol.json'
    if digest(protocol_path) != prepared['protocol_sha256']:
        raise ValueError('Frozen owner protocol changed')
    protocol = json.loads(protocol_path.read_text())
    if protocol['frontend'] != asdict(DEFAULT) or protocol['pcm_normalization'] != PCM_NORMALIZATION:
        raise ValueError('Owner frontend differs from frozen protocol')
    if digest(root / 'scripts/models/iris_owner.py') != protocol['source_files']['scripts/models/iris_owner.py']:
        raise ValueError('Owner model architecture source changed; a new study version is required')
    return protocol, _base_view_receipts(root, protocol)


def _bind_implementation(root, cache, prepared, protocol, view_receipts):
    """Append a source lock without rewriting the original experimental lock."""
    names = ('scripts/models/iris_owner.py', 'scripts/train/iris_owner_reference.py', 'scripts/train/iris_owner.py',
             'scripts/train/iris_audio.py', 'scripts/train/iris_session.py', 'scripts/iris_release.py',
             'pipeline.toml', 'requirements.lock')
    current = {name: digest(root / name) for name in names}
    normalization = cache / 'normalization.json'
    norm_sha = digest(normalization) if normalization.exists() else None
    feature_audit = root / 'experiments/iris-release-v1/owner-feature-audit.json'
    if normalization.exists():
        norm = json.loads(normalization.read_text())
        if norm.get('manifest_sha256') != prepared['manifest_sha256'] or norm.get('frontend') != asdict(DEFAULT):
            raise ValueError('Owner normalization is not bound to frozen human training data')
        if feature_audit.exists() and norm_sha != json.loads(feature_audit.read_text())['normalization_sha256']:
            raise ValueError('Owner normalization changed from the frozen training-feature audit')
    reservation = next((root / name for name in ('.cache/iris-release-v1/inputs/owner-reservations.json',
                        'experiments/iris-release-v1/owner-reservations.json') if (root / name).exists()), None)
    binding = {'protocol_sha256': prepared['protocol_sha256'], 'manifest_sha256': prepared['manifest_sha256'],
               'synthetic_proxy_manifest_sha256': prepared['synthetic_proxy_manifest_sha256'],
               'normalization_sha256': norm_sha, 'source_files': current, 'base_views': view_receipts,
               'reservation_sha256': digest(reservation) if reservation else None}
    equivalence = root / 'experiments/iris-release-v1/owner-base-view-equivalence.json'
    binding['base_view_equivalence_sha256'] = digest(equivalence) if equivalence.exists() else None
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    relative = f'experiments/iris-release-v1/owner-implementation-before-fitting-{fingerprint[:16]}.json'
    destination = root / relative
    if not destination.exists() and any((cache / name).exists() for name in ('resume.pt', 'selected.pt', 'selection-before-test.json')):
        raise ValueError('Owner source/input binding changed after fitting began; preserve evidence and use a new study version')
    record = {'experiment_id': EXPERIMENT, 'component': 'owner', 'schema_version': 1,
              'classification': 'path resolution and pre-fitting execution controls only; no acoustic data/model/frontend/loss/threshold design change',
              'binding': binding, 'fingerprint': fingerprint,
              'historical_installed_integrity': historical_installed_integrity(root),
              'historical_installed_files_are_required_inputs': False,
              'changes_from_frozen_source_snapshot': {name: {'historical': protocol['source_files'].get(name), 'current': value}
                                                     for name, value in current.items() if protocol['source_files'].get(name) != value},
              'frozen_protocol_preserved': True, 'frozen_manifest_preserved': True,
              'source_policy': 'Current implementation is bound here; original protocol source hashes remain historical evidence.'}
    dump(destination, record, immutable=True)
    pointer = {'path': relative, 'sha256': digest(destination), 'fingerprint': fingerprint}
    dump(cache / 'implementation-before-fitting.json', pointer)
    return pointer


def historical_splits(root: Path):
    for relative in ('.cache/iris-release-v1/inputs/owner-reservations.json',
                     'experiments/iris-release-v1/owner-reservations.json'):
        snapshot = root / relative
        if snapshot.exists():
            frozen = json.loads(snapshot.read_text())
            return dict(frozen['speaker_splits']), [{'path': relative, 'sha256': digest(snapshot),
                                                    'kind': 'portable_frozen_reservations'}]
    result, receipts = {}, []
    ranks = {'train': 0, 'enroll': 1, 'dev': 2, 'calibration': 3, 'test': 4, 'holdout': 5}
    for relative in ('data/parameter-study/manifest.jsonl', 'data/phonetic-v2/manifest.jsonl', 'data/asr.jsonl'):
        path = root / relative
        if not path.exists():
            continue
        receipts.append({'path': relative, 'sha256': digest(path)})
        for row in rows(path):
            speaker = row.get('speaker') or row.get('speaker_id')
            split = row.get('split')
            if speaker and split in ranks and ranks[split] >= ranks.get(result.get(speaker, 'train'), 0):
                result[speaker] = split
    return result, receipts


def mix_waveforms(a, b, *, seed: int):
    """Constructed equal-duration overlap at deterministic -4..4 dB SNR."""
    rng = np.random.default_rng(seed)
    output = []
    size = max(len(a), len(b))
    for source in (a, b):
        source = np.asarray(source, dtype=np.float32)
        if len(source) == 0 or not np.isfinite(source).all():
            raise ValueError('Mixture requires two finite waveforms')
        # Whole utterance repeats are not used; unoccupied tails remain silence.
        source = np.pad(source, (0, size - len(source)))
        source = source / max(float(np.sqrt(np.mean(source ** 2))), 1e-5) * .035
        output.append(source)
    mixed = output[0] + output[1] * 10 ** (rng.uniform(-4, 4) / 20)
    mixed *= min(1., .95 / max(float(np.max(np.abs(mixed))), 1e-6))
    return mixed.astype(np.float32)


def _audio(root: Path, row, max_seconds: float | None = None):
    pcm, _ = load_audio(_resolve_audio(root, row))
    if max_seconds is not None and len(pcm) > int(max_seconds * 16000):
        size = int(max_seconds * 16000)
        offset = (len(pcm) - size) // 2
        pcm = pcm[offset:offset + size]
    return pcm


def _resolve_audio(root: Path, row) -> Path:
    """Use the release's hash-bound portable mapping when it is available."""
    try:
        from scripts.iris_release import resolve_audio
    except ImportError:
        return root / row['audio']
    return resolve_audio(row, root=root)


def _download(root: Path, cache: Path):
    source_path = root / 'data/sources/librispeech.source.json'
    source = json.loads(source_path.read_text())
    entry = next(item for item in source['files'] if item['url'].endswith('train-clean-100.tar.gz'))
    archive = cache / 'downloads/train-clean-100.tar.gz'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or archive.stat().st_size != entry['bytes']:
        _download_ranges(archive, entry['bytes'])
    if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
        raise ValueError('LibriSpeech archive SHA-256/size mismatch')
    return archive, {**entry, 'receipt_sha256': digest(source_path),
                     'local_path': '.cache/iris-release-v1/owner/downloads/train-clean-100.tar.gz',
                     'mirror': 'https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/2dd4206a992cd7a475e1ce2e82b708b014319152/train-clean-100.tar.gz',
                     'mirror_revision': '2dd4206a992cd7a475e1ce2e82b708b014319152',
                     'acquisition': 'one archive; prefix resumed across publisher mirrors; disjoint HTTP ranges; final full SHA-256 verified'}


def _download_ranges(archive: Path, total: int):
    """Resume disjoint ranges into one archive; portable per-worker handles.

    The ledger distinguishes verified range boundaries from sparse/incomplete
    file length after interruption. Final archive identity is checked by caller.
    """
    import requests
    from concurrent.futures import as_completed
    url = 'https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/2dd4206a992cd7a475e1ce2e82b708b014319152/train-clean-100.tar.gz?download=true'
    ledger_path = archive.with_name('archive-http-ranges.json')
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger['total'] != total or ledger['url'] != url:
            raise ValueError('Existing archive download ledger differs')
    else:
        ledger = {'total': total, 'url': url, 'prefix': archive.stat().st_size if archive.exists() else 0,
                  'completed': []}
        dump(ledger_path, ledger)
    archive.touch(exist_ok=True)
    completed = {tuple(item) for item in ledger['completed']}
    chunk = 64 << 20
    pending = [(a, min(a + chunk, total) - 1) for a in range(ledger['prefix'], total, chunk)]
    pending = [part for part in pending if part not in completed]
    def fetch(part):
        a, b = part
        for attempt in range(4):
            try:
                with requests.get(url, headers={'Range': f'bytes={a}-{b}'}, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {a}-{b}/'):
                        raise ValueError('Archive range identity mismatch')
                    offset = a
                    with archive.open('r+b') as output:
                        output.seek(a)
                        for block in response.iter_content(1 << 20):
                            output.write(block)
                            offset += len(block)
                    if offset != b + 1:
                        raise ValueError('Archive range incomplete')
                return part
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(fetch, part) for part in pending]
        for done in as_completed(jobs):
            completed.add(done.result())
            ledger['completed'] = sorted(completed)
            dump(ledger_path, ledger)


def _libri(root: Path, cache: Path, archive: Path, legacy, *, per_speaker: int, max_hours: float,
           historical_origins=None):
    extracted, counts, seconds = [], Counter(), 0.
    pool = cache / 'audio'
    # tar stream ordering is pinned by the verified archive. Text members are
    # never extracted or decoded; filenames alone provide speaker identities.
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            parts = Path(member.name).parts
            if (not member.isfile() or member.issym() or member.islnk() or '..' in parts
                    or len(parts) != 5 or parts[:2] != ('LibriSpeech', 'train-clean-100')
                    or not member.name.endswith('.flac')):
                continue
            speaker = 'libri/' + parts[2]
            if counts[speaker] >= per_speaker:
                continue
            payload = source.extractfile(member).read()
            info = sf.info(io.BytesIO(payload))
            if info.samplerate != 16000 or info.channels != 1 or not .8 <= info.duration <= 20:
                continue
            if seconds + info.duration > max_hours * 3600:
                continue
            original = legacy.get(speaker)
            if original == 'enroll':
                # Historical enrollment identities are reserved from fitting.
                split = 'calibration'
            elif original in ('train', 'dev', 'calibration', 'test', 'holdout'):
                split = 'test' if original == 'holdout' else original
            else:
                value = fraction('phonetic-v2-libri/' + speaker)
                split = 'train' if value < .8 else 'dev' if value < .9 else 'test'
            target = pool / Path(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if digest(target) != sha:
                    raise ValueError('Existing selected LibriSpeech payload changed')
            else:
                target.write_bytes(payload)
            extracted.append({'id': 'owner-libri/' + Path(member.name).stem,
                              'speaker': speaker, 'split': split, 'source': 'LibriSpeech-train-clean-100',
                              'kind': 'original_human', 'audio': str(target.relative_to(root)),
                              'sha256': sha, 'duration': info.duration, 'sample_rate': 16000,
                              'license': 'CC-BY-4.0',
                              'historical_split': (historical_origins if historical_origins is not None else legacy).get(speaker)})
            counts[speaker] += 1
            seconds += info.duration
    return extracted


def _ensure_audio_inputs(root: Path, cache: Path):
    """Hydrate missing selected Libri files without altering frozen row bytes."""
    manifest = rows(cache / 'manifest.jsonl')
    proxies = rows(cache / 'synthetic-proxies.jsonl')
    libri = [row for row in manifest if row['source'] == 'LibriSpeech-train-clean-100']
    common = [row for row in manifest if row['source'] != 'LibriSpeech-train-clean-100'] + proxies
    missing = []
    for row in libri:
        path = root / row['audio']
        if path.exists():
            if not path.is_file() or digest(path) != row['sha256']:
                raise ValueError('Selected LibriSpeech payload identity changed: ' + row['audio'])
        else:
            missing.append(row)
    restored = 0
    if missing:
        archive, _ = _download(root, cache)
        wanted = {}
        allowed = (root / '.cache/iris-release-v1/owner/audio').resolve()
        for row in missing:
            parts = Path(row['audio']).parts
            index = parts.index('LibriSpeech')
            member = Path(*parts[index:]).as_posix()
            destination = root / row['audio']
            if '..' in parts or not destination.resolve().is_relative_to(allowed):
                raise ValueError('LibriSpeech destination escapes the declared owner audio cache')
            wanted[member] = (row, destination)
        with tarfile.open(archive, 'r|gz') as source:
            for member in source:
                if member.name not in wanted:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError('Expected an original regular LibriSpeech payload')
                row, destination = wanted.pop(member.name)
                payload = source.extractfile(member).read()
                if hashlib.sha256(payload).hexdigest() != row['sha256']:
                    raise ValueError('Pinned selected LibriSpeech payload hash mismatch')
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if digest(destination) != row['sha256']:
                        raise ValueError('Owner payload changed during hydration')
                else:
                    with destination.open('xb') as output:
                        output.write(payload)
                    restored += 1
                if not wanted:
                    break
        if wanted:
            raise ValueError('Selected frozen LibriSpeech files missing from pinned source archive')
    # Common VCM/wake payloads are resolved through the central lawful upstream
    # input map; owner preparation never substitutes a different recording.
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda row: _resolve_audio(root, row), common))
    report = {'original_human_rows': len(manifest), 'synthetic_proxy_rows': len(proxies),
              'libri_files_verified': len(libri), 'libri_files_restored': restored,
              'frozen_manifest_sha256': digest(cache / 'manifest.jsonl'),
              'frozen_proxy_manifest_sha256': digest(cache / 'synthetic-proxies.jsonl'),
              'manifest_bytes_changed': False}
    dump(cache / 'audio-hydration.json', report)
    return report


def prepare(root: Path, cache: Path, *, per_speaker: int = 24, max_hours: float = 30.):
    receipt = cache / 'preparation.json'
    if receipt.exists():
        prepared = json.loads(receipt.read_text())
        protocol, input_receipts = _verify_prepared(root, cache, prepared)
        if (per_speaker, max_hours) != (protocol['data']['libri_per_speaker_cap'], protocol['data']['libri_max_hours']):
            raise ValueError('Owner subset settings differ from frozen protocol')
        _ensure_audio_inputs(root, cache)
        implementation = _bind_implementation(root, cache, prepared, protocol, input_receipts)
        print(json.dumps({'owner_preparation_verified': True, 'implementation': implementation}), flush=True)
        return prepared
    if not 1 <= per_speaker <= 60 or not 0 < max_hours <= 30:
        raise ValueError('Owner subset must be <=30 hours and <=60 recordings/speaker')
    cache.mkdir(parents=True, exist_ok=True)
    legacy, historical = historical_splits(root)
    historical_origins = None
    for relative in ('.cache/iris-release-v1/inputs/owner-reservations.json',
                     'experiments/iris-release-v1/owner-reservations.json'):
        if (root / relative).exists():
            historical_origins = json.loads((root / relative).read_text()).get('historical_manifest_splits', {})
            break
    original, input_receipts, excluded, synthetic = [], [], Counter(), []
    views = base_view_directory(root)
    for split in ('train', 'dev', 'calibration', 'test'):
        path = views / f'{split}.jsonl'
        input_receipts.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
        for row in rows(path):
            if row.get('kind') != 'human' or row.get('synthetic', False):
                continue
            speaker = row.get('speaker_id') or row.get('speaker')
            if not speaker:
                excluded['missing_speaker'] += 1
                continue
            path = _resolve_audio(root, row)
            info = sf.info(path)
            if not .4 <= info.duration <= 20:
                excluded['duration'] += 1
                continue
            sha = digest(path)
            if sha != row['sha256']:
                raise ValueError('Pinned original human payload changed: ' + str(path))
            original.append({'id': row['id'], 'speaker': speaker, 'split': split,
                             'source': 'VCM-original-human', 'kind': 'original_human',
                             'audio': row['audio'], 'sha256': sha, 'duration': info.duration,
                             'sample_rate': info.samplerate, 'license': row.get('license', 'unknown')})
    for name, split in (('dev', 'dev'), ('calibration', 'calibration'), ('personal-test', 'test')):
        path = views / (name + '.jsonl')
        if name == 'personal-test':
            input_receipts.append({'path': str(path.relative_to(root)), 'sha256': digest(path)})
        for row in rows(path):
            if row.get('source') != 'Qwen3-original-designed-iris-v1' or row.get('role') not in ('wake', 'command'):
                continue
            path = _resolve_audio(root, row)
            sha = digest(path)
            if sha != row['sha256']:
                raise ValueError('Existing synthetic proxy waveform changed')
            synthetic.append({'id': row['id'], 'speaker': row['speaker'], 'split': split,
                              'source': 'Synthetic-wake-command-proxy', 'kind': 'synthetic_proxy',
                              'audio': row['audio'], 'sha256': sha, 'duration': sf.info(path).duration,
                              'sample_rate': 16000, 'role': row['role'], 'listener_verified': False,
                              'independent_human_speaker': False})
    archive, source_receipt = _download(root, cache)
    selected = _libri(root, cache, archive, legacy, per_speaker=per_speaker, max_hours=max_hours,
                      historical_origins=historical_origins)
    manifest, hash_split, speaker_split = [], defaultdict(set), defaultdict(set)
    seen = set()
    for row in original + selected:
        hash_split[row['sha256']].add(row['split'])
        speaker_split[row['speaker']].add(row['split'])
        if row['sha256'] in seen:
            excluded['same_split_exact_duplicate'] += 1
            continue
        seen.add(row['sha256'])
        manifest.append(row)
    conflicts = {key: sorted(value) for key, value in speaker_split.items() if len(value) > 1}
    hashes = {key: sorted(value) for key, value in hash_split.items() if len(value) > 1}
    if conflicts or hashes:
        raise ValueError(f'Cross-split speaker/hash overlap: {conflicts} {hashes}')
    manifest.sort(key=lambda row: (row['split'], row['speaker'], row['id']))
    manifest_path = cache / 'manifest.jsonl'
    manifest_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in manifest))
    proxy_path = cache / 'synthetic-proxies.jsonl'
    proxy_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in synthetic))
    protocol = {'experiment_id': EXPERIMENT, 'component': 'independent-owner', 'seed': SEED,
                'requirements': ['ME-01', 'ME-02', 'ME-03', 'ME-04', 'ME-06', 'ME-07', 'ME-08'],
                'hypothesis': 'A random-init small speaker encoder can separate short same-speaker/impostor proxies and constructed overlap without command embeddings or transcripts.',
                'unsupported': ['fresh original-human wake-command pairing', 'real overlap mixtures', 'physical Raspberry Pi', 'live microphone', 'biometric identification'],
                'architecture': {'width': 64, 'temporal_depthwise_dilations': [1, 2, 4, 8, 1, 2],
                                 'pooling': 'masked_mean_std', 'speaker_embedding': 128, 'overlap_head': True,
                                 'random_initialization': True, 'pretrained_identity_model': False},
                'frontend': asdict(DEFAULT), 'pcm_normalization': PCM_NORMALIZATION,
                'normalization': 'global_training_frames_only',
                'data': {'manifest_sha256': digest(manifest_path), 'vcm_view_receipts': input_receipts,
                         'synthetic_proxy_manifest_sha256': digest(proxy_path),
                         'synthetic_proxy_use': 'existing held-out voices for wake-to-command development calibration/test only; never identity fitting or normalization; no text consumed',
                         'source': source_receipt, 'historical_reservations': historical,
                         'historical_manifests_missing': len(historical) == 0,
                         'fallback_speaker_split': 'sha256-first32bits(phonetic-v2-libri/libri/<speaker>), .8/.9',
                         'libri_per_speaker_cap': per_speaker, 'libri_max_hours': max_hours,
                         'libri_selection': 'first eligible audio members per speaker in hash-verified archive order; no text members read',
                         'constructed_overlap': 'distinct speakers, same split only; -4..4dB SNR; RMS normalized .035; original singles preserved',
                         'switch_proxy': 'two different same-split 1.2s utterances in sequence; actual coverage-checked windows; marked constructed'},
                'fit': {'epochs': 20, 'steps_per_epoch': 160, 'batch_speakers': 12, 'utterances_per_speaker': 3,
                        'crop_seconds': [.8, 3.0], 'loss': 'cosine_speaker_CE + .20 supervised_contrastive + .50 overlap_BCE',
                        'optimizer': 'AdamW lr=.0015 weight_decay=.001; cosine schedule; clip_grad_norm=5'},
                'selection': 'lowest development balanced same/impostor error plus overlap balanced error; no test feature extraction before lock',
                'calibration': {'kind': CALIBRATION_KIND, 'target_impostor_FAR': .05, 'target_clean_overlap_FPR': .05,
                                'speaker_threshold': 'lowest development threshold at <=5% FAR, then minimize FRR',
                                'overlap_threshold': 'lowest development threshold at <=5% clean false overlap, then maximize mixed recall'},
                'policy': {'min_reference_seconds': .8, 'min_command_seconds': .4, 'min_window_seconds': .25,
                           'timeout_seconds': 8., 'require_windows': True},
                'status': 'experimental; proxy-only calibration; not human wake-qualified',
                'baseline': 'no independent owner backend; command embeddings forbidden for identity',
                'source_revision': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or 'not_a_git_checkout',
                'source_files': {relative: digest(root / relative) for relative in
                                 ('scripts/models/iris_owner.py', 'scripts/train/iris_owner_reference.py',
                                  'scripts/train/iris_owner.py', 'scripts/train/iris_audio.py',
                                  'scripts/train/iris_session.py', 'pipeline.toml', 'requirements.lock')},
                'installed_preserved': historical_installed_integrity(root),
                'environment': {'python': platform.python_version(), 'platform': platform.platform()},
                'ignore_gate': 'committed /.cache and /data/audio match positive examples; owner source/compact JSON remain visible; no new rules needed'}
    audit = {'manifest_sha256': digest(manifest_path), 'recordings': len(manifest),
             'synthetic_proxy_manifest_sha256': digest(proxy_path), 'synthetic_proxy_recordings': len(synthetic),
             'by_split': dict(Counter(row['split'] for row in manifest)),
             'speakers': {split: len({row['speaker'] for row in manifest if row['split'] == split}) for split in ('train', 'dev', 'calibration', 'test')},
             'hours': {split: sum(row['duration'] for row in manifest if row['split'] == split) / 3600 for split in ('train', 'dev', 'calibration', 'test')},
             'by_source': dict(Counter(row['source'] for row in manifest)),
             'cross_split_speaker_overlap': conflicts, 'cross_split_hash_overlap': hashes,
             'excluded': dict(excluded), 'all_selected_files_hash_verified': True,
             'libri_official_benchmark': False, 'transcripts_used': False,
             'historical_reservation_limit': 'Missing old manifests prevent independent audit of their identity lists; the identical historical fallback rule is retained.' if not historical else None}
    protocol_path = root / 'experiments/iris-release-v1/owner-protocol.json'
    audit_path = root / 'experiments/iris-release-v1/owner-dataset-audit.json'
    if protocol_path.exists():
        frozen = json.loads(protocol_path.read_text())
        expected = frozen['data']
        if (protocol['data']['manifest_sha256'] != expected['manifest_sha256']
                or protocol['data']['synthetic_proxy_manifest_sha256'] != expected['synthetic_proxy_manifest_sha256']
                or protocol['frontend'] != frozen['frontend'] or protocol['pcm_normalization'] != frozen['pcm_normalization']
                or protocol['architecture'] != frozen['architecture'] or protocol['fit'] != frozen['fit']):
            raise ValueError('Reproduced owner data/model/frontend differ from frozen protocol')
        dump(cache / 'reproduction-preparation.json', {'frozen_protocol_sha256': digest(protocol_path),
                                                     'current_environment': protocol['environment'],
                                                     'current_source_files': protocol['source_files'],
                                                     'reservation_receipts': historical, 'audit': audit})
    else:
        dump(protocol_path, protocol, immutable=True)
    if audit_path.exists():
        frozen_audit = json.loads(audit_path.read_text())
        if frozen_audit['manifest_sha256'] != audit['manifest_sha256']:
            raise ValueError('Reproduced owner audit manifest differs')
    else:
        dump(audit_path, audit, immutable=True)
    prepared = {'manifest_sha256': audit['manifest_sha256'], 'protocol_sha256': digest(root / 'experiments/iris-release-v1/owner-protocol.json'), **audit}
    dump(receipt, prepared, immutable=True)
    frozen, input_receipts = _verify_prepared(root, cache, prepared)
    _ensure_audio_inputs(root, cache)
    _bind_implementation(root, cache, prepared, frozen, input_receipts)
    print(json.dumps(audit), flush=True)
    return prepared


def _feature_row(item):
    root, cache, row = item
    # This experiment version freezes DEFAULT and a six-second center crop;
    # the feature recipe identity is explicit in addition to the audio hash.
    recipe = json.dumps({'frontend': asdict(DEFAULT), 'pcm_normalization': PCM_NORMALIZATION, 'maximum_seconds': 6}, sort_keys=True)
    frontend_id = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    target = cache / 'features' / frontend_id / (row['sha256'] + '.npy')
    if target.exists():
        return np.load(target, allow_pickle=False)
    value = owner_features(_audio(root, row, max_seconds=6))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, value, allow_pickle=False)
    return value


def _normalization(root, cache, training_rows):
    path = cache / 'normalization.json'
    if path.exists():
        return json.loads(path.read_text())
    total, squares, frames = np.zeros(40), np.zeros(40), 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        for feature in executor.map(_feature_row, ((root, cache, row) for row in training_rows)):
            total += feature.sum(axis=1, dtype=np.float64)
            squares += (feature.astype(np.float64) ** 2).sum(axis=1)
            frames += feature.shape[1]
    mean = total / frames
    std = np.sqrt(np.maximum(squares / frames - mean**2, 1e-4))
    value = {'mean': mean.tolist(), 'std': std.tolist(), 'frames': frames,
             'training_recordings': len(training_rows), 'frontend': asdict(DEFAULT),
             'manifest_sha256': digest(cache / 'manifest.jsonl'), 'max_feature_seconds': 6}
    dump(path, value, immutable=True)
    return value


def _crop(pcm, rng, seconds: float):
    size = int(seconds * 16000)
    if len(pcm) <= size:
        return pcm
    start = int(rng.integers(0, len(pcm) - size + 1))
    return pcm[start:start + size]


def _tensor_batch(waves, mean, std, device):
    import torch
    values = [(owner_features(pcm) - mean[:, None]) / std[:, None] for pcm in waves]
    lengths = np.array([value.shape[1] for value in values], dtype=np.int64)
    batch = np.zeros((len(values), 40, int(lengths.max())), dtype=np.float32)
    for i, value in enumerate(values):
        batch[i, :, :value.shape[1]] = value
    return torch.from_numpy(batch).to(device), torch.from_numpy(lengths).to(device)


def _predict(model, waves, mean, std, device='cpu'):
    import torch
    batch, lengths = _tensor_batch(waves, mean, std, device)
    with torch.no_grad():
        _, embeddings, overlap = model(batch, lengths)
    return embeddings.detach().cpu().numpy(), torch.sigmoid(overlap).detach().cpu().numpy()


def _development_rows(manifest, cap=10):
    groups = defaultdict(list)
    for row in manifest:
        if row['split'] in ('dev', 'calibration'):
            groups[row['speaker']].append(row)
    return [row for group in groups.values() for row in sorted(group, key=lambda r: fraction('owner-dev/' + r['id']))[:cap]]


def proxy_scores(model, selected, root, mean, std, *, device='cpu', limit=1200, wake_pairs=False):
    """No embedding serialization; output scores/denominators only."""
    groups = defaultdict(list)
    for row in selected:
        groups[(row['source'], row['speaker'])].append(row)
    rng = np.random.default_rng(SEED + 3)
    # One deterministic 1.2-s crop per original utterance. Enrollment and query
    # always have different original audio hashes; no two crops of one file.
    eligible = sorted([row for group in groups.values() if len(group) >= 2 for row in group],
                      key=lambda row: fraction('owner-proxy-order/' + row['id']))
    waves, vectors, probabilities = [], [], []
    for start in range(0, len(eligible), 32):
        batch = [(_audio(root, row, 6) if wake_pairs else
                  _crop(_audio(root, row, 6), np.random.default_rng(int(row['sha256'][:8], 16)), 1.2))
                 for row in eligible[start:start + 32]]
        embedding, probability = _predict(model, batch, mean, std, device)
        waves.extend(batch)
        vectors.extend(embedding)
        probabilities.extend(probability.tolist())
    if not eligible:
        raise ValueError('No independent original-utterance same-speaker pairs')
    index = defaultdict(list)
    for i, row in enumerate(eligible):
        index[(row['source'], row['speaker'])].append(i)
    pairs, same, impostor, mixed, switches, clean = [], [], [], [], [], []
    domains = defaultdict(lambda: {'same': [], 'impostor': [], 'clean_overlap': [], 'mixed_overlap': [], 'switch': []})
    for i, row in enumerate(eligible):
        if wake_pairs and row.get('role') != 'wake':
            continue
        same_ids = [j for j in index[row['source'], row['speaker']] if j != i
                    and (not wake_pairs or eligible[j].get('role') == 'command')]
        if not same_ids:
            continue
        j = same_ids[int(rng.integers(len(same_ids)))]
        alternatives = [key for key in index if key[0] == row['source'] and key[1] != row['speaker']]
        if not alternatives:
            continue
        key = alternatives[int(rng.integers(len(alternatives)))]
        options = [n for n in index[key] if not wake_pairs or eligible[n].get('role') == 'command']
        if not options:
            continue
        k = options[int(rng.integers(len(options)))]
        # Explicitly ensure mixture/switch construction stays within one split.
        # Development and reserved calibration are distinct; never mix them.
        if eligible[k]['split'] != row['split']:
            alternatives = [key for key in alternatives if any(eligible[n]['split'] == row['split'] for n in index[key])]
            if not alternatives:
                continue
            key = alternatives[int(rng.integers(len(alternatives)))]
            valid = [n for n in index[key] if eligible[n]['split'] == row['split']
                     and (not wake_pairs or eligible[n].get('role') == 'command')]
            if not valid:
                continue
            k = valid[int(rng.integers(len(valid)))]
        pairs.append((i, j, k))
        score = float(np.dot(vectors[i], vectors[j]))
        negative = float(np.dot(vectors[i], vectors[k]))
        same.append(score)
        impostor.append(negative)
        clean.append(float(probabilities[i]))
        domains[row['source']]['same'].append(score)
        domains[row['source']]['impostor'].append(negative)
        domains[row['source']]['clean_overlap'].append(float(probabilities[i]))
        if len(pairs) >= limit:
            break
    for offset in range(0, len(pairs), 24):
        subset = pairs[offset:offset + 24]
        mixed_waves = [mix_waveforms(waves[j], waves[k], seed=SEED + i) for i, j, k in subset]
        _, probabilities_mix = _predict(model, mixed_waves, mean, std, device)
        # Sequential speaker-switch proxy. Continuity checks both halves instead
        # of trusting a single pooled whole-command embedding.
        switched_waves = [np.concatenate((waves[j], waves[k])) for i, j, k in subset]
        windows, window_ranges = [], []
        for waveform in switched_waves:
            spans = continuity_spans(waveform)
            begin = len(windows)
            if spans:
                windows.extend(waveform[a:b] for a, b in spans)
            window_ranges.append((begin, len(windows)))
        embedded_windows = []
        for start in range(0, len(windows), 32):
            embedded, _ = _predict(model, windows[start:start + 32], mean, std, device)
            embedded_windows.extend(embedded)
        for triple, prob, (a, b) in zip(subset, probabilities_mix, window_ranges):
            i, j, k = triple
            # Actual embeddings of coverage-checked windows, never the known
            # identity label or a fabricated policy embedding.
            switch = min((float(np.dot(vectors[i], vector)) for vector in embedded_windows[a:b]), default=-1.)
            mixed.append(float(prob))
            switches.append(switch)
            domain = domains[eligible[i]['source']]
            domain['mixed_overlap'].append(float(prob))
            domain['switch'].append(switch)
    return {'same': same, 'impostor': impostor, 'clean_overlap': clean,
            'mixed_overlap': mixed, 'switch': switches,
            'domains': dict(domains), 'pairs': len(pairs),
            'provenance': ('held-out synthetic wake/command utterances; full PCM <=6s; constructed overlap/switch; not independent human identities'
                           if wake_pairs else 'different original utterances, 1.2s acoustic crops; same-split constructed mixtures and switches; no human wake qualification')}


def combined_development(model, selected, root, cache, mean, std, *, device='cpu', limit=1600):
    human = proxy_scores(model, selected, root, mean, std, device=device, limit=limit)
    proxies = [row for row in rows(cache / 'synthetic-proxies.jsonl') if row['split'] in ('dev', 'calibration')]
    synthetic = proxy_scores(model, proxies, root, mean, std, device=device, wake_pairs=True)
    combined = {name: human[name] + synthetic[name]
                for name in ('same', 'impostor', 'clean_overlap', 'mixed_overlap', 'switch')}
    combined['domains'] = {**human['domains'], **synthetic['domains']}
    return combined


def _threshold(positive, negative, max_false=.05):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        raise ValueError('Calibration needs both positive and negative observations')
    values = np.unique(np.concatenate((positive, negative)))
    candidates = np.concatenate(([values[0] - 1e-6], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-6]))
    options = [(float(np.mean(positive < threshold)), float(np.mean(negative >= threshold)), float(threshold)) for threshold in candidates]
    safe = [entry for entry in options if entry[1] <= max_false]
    return min(safe, key=lambda entry: (entry[0], entry[2]))[2]


def calibrate(scores):
    return {'threshold': float(np.clip(_threshold(scores['same'], scores['impostor']), -1, 1)),
            'overlap_threshold': float(np.clip(_threshold(scores['mixed_overlap'], scores['clean_overlap']), 0, 1)),
            'calibration_kind': CALIBRATION_KIND, 'experimental': True}


def rates(scores, policy):
    threshold, overlap_threshold = policy['threshold'], policy['overlap_threshold']
    def count(values, predicate):
        return {'errors': int(sum(predicate(v) for v in values)), 'denominator': len(values),
                'rate': float(np.mean([predicate(v) for v in values])) if values else None}
    return {'same_speaker_FRR': count(scores.get('same', []), lambda score: score < threshold),
            'impostor_FAR': count(scores.get('impostor', []), lambda score: score >= threshold),
            'switch_FAR': count(scores.get('switch', []), lambda score: score >= threshold),
            'clean_false_overlap': count(scores.get('clean_overlap', []), lambda score: score >= overlap_threshold),
            'overlap_miss_rate': count(scores.get('mixed_overlap', []), lambda score: score < overlap_threshold)}


def _load_model(cache, device='cpu'):
    import torch
    from scripts.models.iris_owner import OwnerEncoder
    checkpoint = torch.load(cache / 'selected.pt', map_location='cpu', weights_only=False)
    model = OwnerEncoder(checkpoint['speakers'], checkpoint['width'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def configure_training_device(device, *, cuda, environ):
    """Select one logical GPU; physical device authorization is the caller's."""
    import re
    selector = environ.get('CUDA_VISIBLE_DEVICES')
    result = {'requested_device': device, 'cuda_visible_devices': selector, 'logical_gpu': None}
    if not device.startswith('cuda'):
        return result
    match = re.fullmatch(r'cuda(?::([0-9]+))?', device)
    if match is None:
        raise ValueError('Use a single CUDA logical device')
    index = int(match[1] or 0)
    if selector is not None:
        visible = selector.strip()
        if not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+)', visible):
            raise ValueError('CUDA_VISIBLE_DEVICES must name a single GPU index or UUID')
        if index != 0:
            raise ValueError('A single visible GPU uses logical cuda:0')
    cuda.set_device(index)
    cuda.set_per_process_memory_fraction(.12, index)
    result['logical_gpu'] = index
    return result


def save_training_state(path, model, optimizer, scheduler, rng, progress, bindings):
    """Atomic boundary-only checkpoint, including every training RNG stream."""
    import torch
    state = {'bindings': bindings, 'progress': progress,
             'model': {name: value.detach().cpu() for name, value in model.state_dict().items()},
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'generator': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state(next(model.parameters()).device) if next(model.parameters()).is_cuda else None,
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}
    temporary = path.with_name(path.name + '.partial')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, scheduler, rng, bindings):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['bindings'] != bindings:
        raise ValueError('Resume checkpoint binding mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    rng.bit_generator.state = state['generator']
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None and next(model.parameters()).is_cuda:
        cuda_rng = state['cuda_rng']
        if isinstance(cuda_rng, (list, tuple)):
            if len(cuda_rng) != 1:
                raise ValueError('Resume checkpoint must contain a single GPU RNG state')
            cuda_rng = cuda_rng[0]
        torch.cuda.set_rng_state(cuda_rng, next(model.parameters()).device)
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    return state['progress']


def train(root: Path, cache: Path, *, device='cpu', epochs=20, steps=160):
    import torch
    import torch.nn.functional as F
    from scripts.models.iris_owner import OwnerEncoder, speaker_metric_loss
    prepared = json.loads((cache / 'preparation.json').read_text())
    protocol, input_receipts = _verify_prepared(root, cache, prepared)
    if (cache / 'selection-before-test.json').exists():
        selected = json.loads((cache / 'selection-before-test.json').read_text())
        for name, key in (('selected.pt', 'checkpoint_sha256'), ('manifest.jsonl', 'manifest_sha256'),
                          ('normalization.json', 'normalization_sha256')):
            if digest(cache / name) != selected[key]:
                raise ValueError('Existing owner selection receipt no longer matches ' + name)
        if selected['protocol_sha256'] != prepared['protocol_sha256']:
            raise ValueError('Existing owner protocol changed')
        print('Owner training already complete; frozen hashes verified', flush=True)
        return selected
    if (epochs, steps) != (protocol['fit']['epochs'], protocol['fit']['steps_per_epoch']):
        raise ValueError('Training settings differ from frozen owner protocol')
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    stop_flag = cache / 'stop-requested'
    if stop_flag.exists():
        print('Owner stop requested before fitting; no GPU allocation', flush=True)
        return {'status': 'incomplete', 'reason': 'stop-requested'}
    device_selection = configure_training_device(device, cuda=torch.cuda, environ=__import__('os').environ)
    execution = {'experiment_id': EXPERIMENT, 'component': 'owner',
                 'protocol_sha256': prepared['protocol_sha256'], 'manifest_sha256': prepared['manifest_sha256'],
                 'source_files': {relative: digest(root / relative) for relative in
                                  ('scripts/models/iris_owner.py', 'scripts/train/iris_owner_reference.py', 'scripts/train/iris_owner.py')},
                 'amendment': 'Portable single-GPU selection, boundary stop/resume, portable reservation loading and conservative full-PCM continuity coverage added after data freeze, before fitting; acoustic data/model/frontend/loss/selection unchanged.',
                 'device_selection': device_selection, 'max_cuda_memory_fraction': .12,
                 'development_inference_device': 'separate CPU mirror; GPU used only for optimizer batches; optimizer parameter identities preserved',
                 'stop_flag': '.cache/iris-release-v1/owner/stop-requested',
                 'resume': 'atomic model/optimizer/scheduler/RNG/epoch/next-step checkpoint; binding verification; no partial selection'}
    execution_path = root / 'experiments/iris-release-v1/owner-execution.json'
    manifest = rows(cache / 'manifest.jsonl')
    fitting = [row for row in manifest if row['split'] == 'train']
    norm = _normalization(root, cache, fitting)
    implementation = _bind_implementation(root, cache, prepared, protocol, input_receipts)
    execution['implementation_before_fitting'] = implementation
    dump(execution_path, execution, immutable=True)
    mean, std = np.array(norm['mean'], dtype=np.float32), np.array(norm['std'], dtype=np.float32)
    grouped = defaultdict(list)
    for row in fitting:
        grouped[row['speaker']].append(row)
    # Single-utterance identities cannot establish within-speaker invariance.
    grouped = {key: group for key, group in grouped.items() if len(group) >= 2}
    identities = sorted(grouped)
    labels = {speaker: i for i, speaker in enumerate(identities)}
    if len(identities) < 12:
        raise ValueError('Need at least twelve training identities')
    print('Loading read-only original waveforms into bounded owner memory', flush=True)
    wanted = [row for group in grouped.values() for row in group]
    with ThreadPoolExecutor(max_workers=8) as executor:
        waves = dict(zip((row['id'] for row in wanted), executor.map(lambda row: _audio(root, row, 6), wanted)))
    model = OwnerEncoder(len(identities)).to(device)
    development_model = OwnerEncoder(len(identities)).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.default_rng(SEED)
    development = _development_rows(manifest, cap=6)
    history, best = [], float('inf')
    resume_path = cache / 'resume.pt'
    bindings = {'protocol': prepared['protocol_sha256'], 'manifest': prepared['manifest_sha256'],
                'normalization': digest(cache / 'normalization.json'), 'execution': digest(execution_path),
                'identities': hashlib.sha256(json.dumps(identities).encode()).hexdigest(),
                'epochs': epochs, 'steps': steps}
    progress = {'epoch': 1, 'next_step': 0, 'history': [], 'best': None,
                'losses': [], 'correct': 0, 'examples': 0, 'elapsed_seconds': 0.}
    if resume_path.exists():
        progress = restore_training_state(resume_path, model, optimizer, scheduler, rng, bindings)
        history = progress['history']
        best = progress['best'] if progress['best'] is not None else float('inf')
        print(f"Resuming owner epoch {progress['epoch']} step {progress['next_step']} after binding/RNG verification", flush=True)
    previous_elapsed = progress['elapsed_seconds']
    started = time.time()
    def checkpoint_boundary(epoch, next_step, losses, correct, examples):
        state = {'epoch': epoch, 'next_step': next_step, 'history': history,
                 'best': best if math.isfinite(best) else None, 'losses': losses,
                 'correct': correct, 'examples': examples,
                 'elapsed_seconds': previous_elapsed + time.time() - started}
        save_training_state(resume_path, model, optimizer, scheduler, rng, state, bindings)
        return state
    for epoch in range(progress['epoch'], epochs + 1):
        if stop_flag.exists():
            state = checkpoint_boundary(epoch, progress['next_step'] if epoch == progress['epoch'] else 0,
                                        progress['losses'] if epoch == progress['epoch'] else [],
                                        progress['correct'] if epoch == progress['epoch'] else 0,
                                        progress['examples'] if epoch == progress['epoch'] else 0)
            return {'status': 'incomplete', 'progress': state}
        model.to(device)
        model.train()
        continuing = epoch == progress['epoch']
        losses = progress['losses'][:] if continuing else []
        correct = progress['correct'] if continuing else 0
        examples = progress['examples'] if continuing else 0
        first_step = progress['next_step'] if continuing else 0
        for step in range(first_step, steps):
            if stop_flag.exists():
                state = checkpoint_boundary(epoch, step, losses, correct, examples)
                print(json.dumps({'status': 'incomplete', 'reason': 'stop-requested', 'epoch': epoch, 'next_step': step}), flush=True)
                return {'status': 'incomplete', 'progress': state}
            selected = rng.choice(identities, size=12, replace=False)
            originals, targets, speakers = [], [], []
            seconds = float(rng.uniform(.8, 3.))
            for speaker in selected:
                group = grouped[speaker]
                indices = rng.choice(len(group), size=3, replace=len(group) < 3)
                for index in indices:
                    row = group[int(index)]
                    originals.append(_crop(waves[row['id']], rng, seconds))
                    targets.append(labels[speaker])
                    speakers.append(speaker)
            mixtures = []
            for i in range(0, len(originals), 2):
                other = (i + 3) % len(originals)
                if speakers[i] == speakers[other]:
                    raise AssertionError('Overlap mixture must have distinct identities')
                mixtures.append(mix_waveforms(originals[i], originals[other], seed=int(rng.integers(2**31))))
            acoustic, lengths = _tensor_batch(originals + mixtures, mean, std, device)
            logits, embedding, overlap = model(acoustic, lengths)
            target = torch.tensor(targets, dtype=torch.long, device=device)
            overlap_target = torch.cat((torch.zeros(len(originals)), torch.ones(len(mixtures)))).to(device)
            loss = (F.cross_entropy(logits[:len(target)], target)
                    + .2 * speaker_metric_loss(embedding[:len(target)], target)
                    + .5 * F.binary_cross_entropy_with_logits(overlap, overlap_target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int((logits[:len(target)].argmax(1) == target).sum())
            examples += len(target)
            if stop_flag.exists():
                state = checkpoint_boundary(epoch, step + 1, losses, correct, examples)
                print(json.dumps({'status': 'incomplete', 'reason': 'stop-requested', 'epoch': epoch, 'next_step': step + 1}), flush=True)
                return {'status': 'incomplete', 'progress': state}
        scheduler.step()
        model.eval()
        development_model.load_state_dict(model.state_dict())
        development_model.eval()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        scores = proxy_scores(development_model, development, root, mean, std, device='cpu', limit=300)
        policy = calibrate(scores)
        metric = rates(scores, policy)
        objective = (metric['same_speaker_FRR']['rate'] + metric['impostor_FAR']['rate']
                     + metric['clean_false_overlap']['rate'] + metric['overlap_miss_rate']['rate']) / 4
        entry = {'epoch': epoch, 'loss': float(np.mean(losses)), 'train_identity_accuracy': correct / examples,
                 'development_objective': objective, 'development': metric,
                 'elapsed_seconds': previous_elapsed + time.time() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        dump(cache / 'training-history.json', history)
        if objective < best:
            best = objective
            torch.save({'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                        'epoch': epoch, 'speakers': len(identities),
                        'width': 64, 'seed': SEED, 'manifest_sha256': prepared['manifest_sha256'],
                        'protocol_sha256': prepared['protocol_sha256'], 'normalization_sha256': digest(cache / 'normalization.json'),
                        'execution_sha256': digest(execution_path),
                        'epochs_planned': epochs, 'steps_per_epoch': steps, 'development_objective': objective}, cache / 'selected.pt')
        checkpoint_boundary(epoch + 1, 0, [], 0, 0)
    selected, checkpoint = _load_model(cache, 'cpu')
    development = _development_rows(manifest, cap=12)
    scores = combined_development(selected, development, root, cache, mean, std, device='cpu')
    policy = calibrate(scores)
    selection = {'schema_version': 1, 'experiment_id': EXPERIMENT, 'component': 'owner',
                 'selection_completed_before_test': True, 'selected_epoch': checkpoint['epoch'],
                 'checkpoint_sha256': digest(cache / 'selected.pt'), 'manifest_sha256': prepared['manifest_sha256'],
                 'protocol_sha256': prepared['protocol_sha256'], 'normalization_sha256': digest(cache / 'normalization.json'),
                 'execution_sha256': digest(execution_path),
                 'policy': policy, 'development': rates(scores, policy),
                 'development_domains': {name: rates(value, policy) for name, value in scores['domains'].items()},
                 'human_wake_qualified': False, 'physical_pi_measured': False,
                 'training': {'epochs': epochs, 'steps_per_epoch': steps, 'random_seed': SEED,
                              'speaker_classes': len(identities), 'seconds': previous_elapsed + time.time() - started,
                              'torch': torch.__version__, 'device': device,
                              'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                              'gpu_name': torch.cuda.get_device_name(device_selection['logical_gpu']) if device.startswith('cuda') else None,
                              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device_selection['logical_gpu']) if device.startswith('cuda') else 0,
                              'command': list(__import__('sys').argv),
                              'dtype': 'float32', 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()},
                 'history_sha256': digest(cache / 'training-history.json')}
    dump(cache / 'selection-before-test.json', selection, immutable=True)
    dump(root / 'experiments/iris-release-v1/owner-selection-before-test.json', selection, immutable=True)
    print(json.dumps(selection), flush=True)
    return selection


def evaluate(root: Path, cache: Path):
    import torch
    torch.set_num_threads(4)
    selection = json.loads((cache / 'selection-before-test.json').read_text())
    if digest(cache / 'selected.pt') != selection['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed after lock')
    if digest(cache / 'normalization.json') != selection['normalization_sha256']:
        raise ValueError('Normalization changed after selection')
    if digest(root / 'experiments/iris-release-v1/owner-protocol.json') != selection['protocol_sha256']:
        raise ValueError('Protocol changed after selection')
    if (cache / 'test-results.json').exists():
        report = json.loads((cache / 'test-results.json').read_text())
        if report['selection_sha256'] != digest(cache / 'selection-before-test.json'):
            raise ValueError('Existing test receipt selection changed')
        print('Owner locked evaluation already complete; selection hash verified', flush=True)
        return report
    manifest = rows(cache / 'manifest.jsonl')
    if digest(cache / 'manifest.jsonl') != selection['manifest_sha256']:
        raise ValueError('Manifest changed after lock')
    model, _ = _load_model(cache)
    norm = json.loads((cache / 'normalization.json').read_text())
    mean, std = np.array(norm['mean'], dtype=np.float32), np.array(norm['std'], dtype=np.float32)
    grouped = defaultdict(list)
    for row in manifest:
        if row['split'] == 'test':
            grouped[row['speaker']].append(row)
    selected = [row for group in grouped.values() for row in sorted(group, key=lambda row: fraction('owner-test/' + row['id']))[:16]]
    scores = proxy_scores(model, selected, root, mean, std, limit=2400)
    synthetic_rows = [row for row in rows(cache / 'synthetic-proxies.jsonl') if row['split'] == 'test']
    synthetic = proxy_scores(model, synthetic_rows, root, mean, std, wake_pairs=True)
    policy = selection['policy']
    metrics = rates(scores, policy)
    report = {'experiment_id': EXPERIMENT, 'component': 'owner', 'status': 'experimental_proxy_only',
              'selection_sha256': digest(cache / 'selection-before-test.json'),
              'manifest_sha256': selection['manifest_sha256'], 'checkpoint_sha256': selection['checkpoint_sha256'],
              'calibration_kind': CALIBRATION_KIND, 'policy': policy, 'test': metrics,
              'test_domains': {name: rates(value, policy) for name, value in scores['domains'].items()},
              'synthetic_wake_command_proxy': rates(synthetic, policy),
              'test_recordings': len(selected), 'test_speakers': len(grouped),
              'construction': scores['provenance'], 'test_tuning': False,
              'gates': {'proxy_impostor_FAR_le_5_percent': metrics['impostor_FAR']['rate'] <= .05,
                        'proxy_same_speaker_FRR_le_35_percent': metrics['same_speaker_FRR']['rate'] <= .35,
                        'proxy_overlap_miss_le_20_percent': metrics['overlap_miss_rate']['rate'] <= .20,
                        'human_wake_qualified': False, 'physical_pi_measured': False},
              'remaining': ['fresh original-human wake-command recordings >=.8s wake', 'real simultaneous overlap',
                            'unseen microphones/environments', 'physical Pi CPU timing and live capture'],
              'embedding_logs_written': False, 'installed_artifacts_changed': False}
    dump(cache / 'test-results.json', report, immutable=True)
    dump(root / 'experiments/iris-release-v1/owner-results.json', report, immutable=True)
    print(json.dumps(report), flush=True)
    return report


def _runtime_proxy_report(root, cache, backend, *, limit=80):
    """Exercise real ORT evidence and the unchanged session policy after lock."""
    from scripts.train.iris_session import SpeakerSession
    manifest = [row for row in rows(cache / 'manifest.jsonl') if row['split'] == 'test']
    synthetic = [row for row in rows(cache / 'synthetic-proxies.jsonl') if row['split'] == 'test']
    summary = {}
    for track, selected in (('original_human_short_speech_proxy', manifest), ('synthetic_wake_command_proxy', synthetic)):
        grouped = defaultdict(list)
        for row in selected:
            grouped[row['source'], row['speaker']].append(row)
        keys = sorted((key for key in grouped if len(grouped[key]) >= 2),
                      key=lambda key: fraction('owner-runtime/' + key[1]))
        trials = {'same': [], 'impostor': [], 'switch': [], 'overlap': []}
        reasons = Counter()
        for n, key in enumerate(keys[:limit]):
            group = sorted(grouped[key], key=lambda row: fraction('owner-runtime-row/' + row['id']))
            references = [row for row in group if track != 'synthetic_wake_command_proxy' or row.get('role') == 'wake']
            commands = [row for row in group if track != 'synthetic_wake_command_proxy' or row.get('role') == 'command']
            if not references or not commands:
                continue
            reference_row = references[0]
            commands = [row for row in commands if row['sha256'] != reference_row['sha256']]
            alternatives = [other for other in keys if other[0] == key[0] and other[1] != key[1]]
            if not commands or not alternatives:
                continue
            alternate = grouped[alternatives[n % len(alternatives)]]
            alternate = [row for row in alternate if track != 'synthetic_wake_command_proxy' or row.get('role') == 'command']
            if not alternate:
                continue
            reference = _audio(root, reference_row, 3)
            if track == 'original_human_short_speech_proxy':
                reference = _crop(reference, np.random.default_rng(SEED + n), 1.2)
            command = _audio(root, commands[0], 3)
            impostor = _audio(root, alternate[0], 3)
            switched = np.concatenate((_crop(command, np.random.default_rng(SEED + n), 1.2),
                                       _crop(impostor, np.random.default_rng(SEED + n + 1), 1.2)))
            mixture = mix_waveforms(command, impostor, seed=SEED + n)
            reference_evidence = backend.evidence(reference)
            for kind, waveform in (('same', command), ('impostor', impostor), ('switch', switched), ('overlap', mixture)):
                session = SpeakerSession(backend.policy)
                armed = session.arm('proxy', reference_evidence, now=0.)
                if not armed.accepted:
                    verdict = armed
                else:
                    verdict = session.finish('proxy', backend.evidence(waveform),
                                             now=len(waveform) / 16000 + .1,
                                             windows=backend.continuity(waveform))
                trials[kind].append(verdict.accepted)
                if verdict.reason:
                    reasons[verdict.reason] += 1
        def metric(kind, expected):
            values = trials[kind]
            errors = sum(value != expected for value in values)
            return {'errors': errors, 'denominator': len(values), 'rate': errors / len(values) if values else None}
        summary[track] = {'same_speaker_FRR': metric('same', True),
                          'impostor_FAR': metric('impostor', False),
                          'switch_FAR': metric('switch', False),
                          'overlap_FAR': metric('overlap', False),
                          'rejection_reasons': dict(reasons),
                          'evidence': 'actual PCM -> shared frontend -> trained CPU ONNX -> coverage windows -> SpeakerSession',
                          'fresh_human_wake_qualified': False}
    return summary


def export(root: Path, cache: Path):
    import torch
    import onnx
    import onnxruntime as ort
    from scripts.models.iris_owner import OwnerExport
    torch.set_num_threads(1)
    selection = json.loads((cache / 'selection-before-test.json').read_text())
    if digest(cache / 'selected.pt') != selection['checkpoint_sha256']:
        raise ValueError('Checkpoint changed after selection')
    existing = cache / 'export-results.json'
    if existing.exists():
        result = json.loads(existing.read_text())
        if (digest(cache / 'export/owner.onnx') != result['model']['sha256']
                or digest(cache / 'export/owner.json') != result['metadata_sha256']):
            raise ValueError('Existing owner export receipt mismatch')
        print('Owner export already complete; graph and metadata hashes verified', flush=True)
        return result
    model, checkpoint = _load_model(cache)
    adapter = OwnerExport(model).eval()
    destination = cache / 'export'
    destination.mkdir(parents=True, exist_ok=True)
    graph = destination / 'owner.onnx'
    with torch.no_grad():
        torch.onnx.export(adapter, (torch.randn(1, 40, 120), torch.tensor([120])), str(graph),
                          input_names=['features', 'lengths'], output_names=['embedding', 'overlap'],
                          dynamic_axes={'features': {0: 'batch', 2: 'frames'}, 'lengths': {0: 'batch'},
                                        'embedding': {0: 'batch'}, 'overlap': {0: 'batch'}},
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(graph)))
    norm = json.loads((cache / 'normalization.json').read_text())
    policy = {**selection['policy']}
    policy.pop('calibration_kind')
    policy.pop('experimental')
    policy.update({'min_reference_seconds': .8, 'min_command_seconds': .4,
                   'min_window_seconds': .25, 'timeout_seconds': 8., 'require_windows': True})
    runtime_parameters = sum(parameter.numel() for name, parameter in model.named_parameters() if not name.startswith('identity.'))
    metadata = {'schema_version': 1, 'experiment_id': EXPERIMENT, 'component': 'owner',
                'experimental': True, 'human_wake_qualified': False,
                'calibration_kind': CALIBRATION_KIND,
                'identity_backend': 'independent_random_init_temporal_depthwise_speaker_encoder',
                'model': {'path': 'owner.onnx', 'sha256': digest(graph), 'bytes': graph.stat().st_size,
                          'runtime_parameters': runtime_parameters,
                          'training_parameters': sum(p.numel() for p in model.parameters()),
                          'embedding_dimensions': 128, 'selected_epoch': checkpoint['epoch']},
                'files': {'owner.onnx': digest(graph)},
                'frontend': asdict(DEFAULT), 'pcm_normalization': PCM_NORMALIZATION,
                'normalization': norm, 'policy': policy,
                'speech_mask': {'kind': 'energy_proxy', 'frame_samples': 400, 'hop_samples': 160,
                                'absolute_rms_floor': .001, 'peak_relative_floor': .08,
                                'continuity_window_seconds': 1.2, 'hop_seconds': .6,
                                'coverage': 'whole supplied PCM, including quiet edges, or fail closed'},
                'manifest_sha256': selection['manifest_sha256'], 'protocol_sha256': selection['protocol_sha256'],
                'selection_sha256': digest(cache / 'selection-before-test.json'),
                'no_transcripts': True, 'no_command_embeddings': True, 'no_pretrained_identity_model': True,
                'remaining': ['Original-human wake-command and real-overlap qualification missing', 'Physical Pi/live microphone not measured']}
    dump(destination / 'owner.json', metadata)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    started = time.perf_counter()
    session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    load_ms = (time.perf_counter() - started) * 1000
    mean, std = np.array(norm['mean'], dtype=np.float32), np.array(norm['std'], dtype=np.float32)
    selected_rows = [row for row in rows(cache / 'manifest.jsonl') if row['split'] == 'test'][:12]
    parity, timings = [], []
    for row in selected_rows:
        pcm = _audio(root, row, 4)
        acoustic = (owner_features(pcm) - mean[:, None]) / std[:, None]
        lengths = np.array([acoustic.shape[1]], dtype=np.int64)
        with torch.no_grad():
            expected = adapter(torch.from_numpy(acoustic[None]), torch.from_numpy(lengths))
        before = time.perf_counter()
        actual = session.run(None, {'features': acoustic[None], 'lengths': lengths})
        timings.append((time.perf_counter() - before) * 1000)
        errors = [float(np.max(np.abs(a - b.numpy()))) for a, b in zip(actual, expected)]
        parity.append({'audio_sha256': row['sha256'], 'samples': len(pcm), 'maximum_absolute_errors': errors})
        if max(errors) > 2e-5:
            raise ValueError('Real PCM Torch/ORT owner parity failed')
    from scripts.train.iris_owner_reference import OwnerBackend
    backend = OwnerBackend(destination, threads=1)
    runtime_proxies = _runtime_proxy_report(root, cache, backend)
    report = {'experiment_id': EXPERIMENT, 'component': 'owner', 'model': metadata['model'],
              'metadata_sha256': digest(destination / 'owner.json'), 'real_pcm_parity': parity,
              'parity_tolerance': 2e-5, 'parity_passed': True,
              'runtime_policy_proxies': runtime_proxies,
              'desktop_cpu': {'threads': 1, 'onnx_only_p50_ms': float(np.percentile(timings, 50)),
                              'onnx_only_p95_ms': float(np.percentile(timings, 95)),
                              'session_load_ms': load_ms, 'observations': len(timings),
                              'excludes': ['frontend', 'disk I/O', 'utterance duration', 'wake/VAD', 'continuity windows'],
                              'not_physical_pi': True},
              'portable_onnx': True, 'ncnn': 'not exported; CPU ONNX is authoritative',
              'bundle': '.cache/iris-release-v1/owner/export', 'installed_artifacts_changed': False}
    dump(cache / 'export-results.json', report)
    dump(root / 'experiments/iris-release-v1/owner-export.json', report)
    print(json.dumps(report), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'train', 'evaluate', 'export'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--libri-per-speaker', type=int, default=24)
    parser.add_argument('--libri-max-hours', type=float, default=30.)
    args = parser.parse_args(argv)
    root = args.root.absolute()
    cache = args.cache.absolute() if args.cache else root / '.cache/iris-release-v1/owner'
    if args.action == 'prepare':
        prepare(root, cache, per_speaker=args.libri_per_speaker, max_hours=args.libri_max_hours)
    elif args.action == 'train':
        result = train(root, cache, device=args.device, epochs=args.epochs, steps=args.steps)
        if result.get('status') == 'incomplete':
            raise SystemExit(75)
    elif args.action == 'evaluate':
        evaluate(root, cache)
    else:
        export(root, cache)


if __name__ == '__main__':
    main()
