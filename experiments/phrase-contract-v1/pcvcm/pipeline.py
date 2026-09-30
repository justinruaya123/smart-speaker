"""Verified WAV replay, immutable calibration, and diagnostic-only packaging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .bundle import ASSETS, RUNTIME_CODE, seal_bundle, verify_bundle
from .contract import Contract, file_hash, identity, read_jsonl, write_json
from .data import below
from .evidence import audit_partitions, validate_reference, write_reports
from .policy import calibrate
from .runtime import Runtime


def reserve_confirmation(manifest_hash, binding, policy_hash, ledger=None):
    ledger = Path(ledger or '.cache/phrase-contract-v1/confirmation-access')
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f'{manifest_hash}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'state': 'INFERENCE_RESERVED_NOT_UNTOUCHED',
            'manifest_sha256': manifest_hash, 'binding': binding,
            'policy_sha256': policy_hash,
            'recovery': 'A crash requires an explicit reviewed incident; never delete this receipt.'}, stream)
    return path


def verify_confirmation(manifest: Path, commitment: dict,
                        contract: Contract) -> None:
    if (commitment.get('contract_sha256') != contract.sha256
            or commitment.get('confirmation_manifest_sha256') != file_hash(manifest)
            or len(commitment.get('speaker_identity_hashes', [])) < 10):
        raise ValueError('Final manifest is not its frozen speaker-bound commitment')
    rows = read_jsonl(manifest)
    audit_partitions(rows)
    expected_speakers = set(commitment['speaker_identity_hashes'])
    if {identity(r['speaker_id']) for r in rows} != expected_speakers:
        raise ValueError('Confirmation speaker identities changed')
    if any(r['split'] != 'confirmation' for r in rows):
        raise ValueError('Confirmation manifest contains other partitions')
    for row in rows:
        validate_reference(row, contract)


def replay(bundle: Path, manifest: Path, audio_root: Path,
           contract: Contract, split: str, commitment: Path | None = None,
           threads: int = 1, diagnostic_development: bool = False, prediction_log: Path | None = None) -> list[dict]:
    import soundfile as sf

    rows = read_jsonl(manifest)
    if not rows or any(r.get('split') != split for r in rows):
        raise ValueError('Replay requires one explicitly declared partition')
    audit_partitions(rows)
    for row in rows:
        validate_reference(row, contract)
    if split == 'confirmation':
        if commitment is None or diagnostic_development:
            raise ValueError('Confirmation requires frozen, non-diagnostic policy')
        verify_confirmation(manifest, json.loads(commitment.read_text()), contract)
    elif split not in ('dev', 'calibration'):
        raise ValueError('Training audio is not evaluation evidence')
    runtime = Runtime.from_bundle(bundle, contract, threads)
    if split == 'confirmation':
        if runtime.policy.get('enabled') is not True:
            raise ValueError('Confirmation requires an enabled calibrated policy')
        reserve_confirmation(file_hash(manifest), runtime.binding, file_hash(Path(bundle) / 'policy.json'))
    if diagnostic_development:
        if split != 'dev':
            raise ValueError('Tentative endpoint selection is development-only')
        runtime.policy = {'enabled': True, 'binding': runtime.binding,
                          'thresholds': {'global': [-1e9, 0.]}}
    results = []
    manifest_hash = file_hash(manifest)
    for row in rows:
        started = time.perf_counter()
        runtime.reset()
        decision = {'schema_version': 2, 'accepted': False,
                    'command': None, 'reason': 'capture_fault'}
        verified, error = False, None
        try:
            audio = below(audio_root, row['audio'])
            if file_hash(audio) != row['audio_sha256']:
                raise ValueError('Original recording hash mismatch')
            pcm, rate = sf.read(audio, dtype='float32')
            if (rate != 16000 or pcm.ndim != 1
                    or not np.isfinite(pcm).all()):
                raise ValueError('Requires finite original mono 16 kHz PCM')
            pcm_hash = hashlib.sha256(pcm.tobytes()).hexdigest()
            if pcm_hash != row['pcm_sha256']:
                raise ValueError('Decoded PCM hash mismatch')
            verified = True
            for start in range(0, len(pcm), 640):
                decision = runtime.feed(pcm[start:start + 640], start)
                if runtime.finalized or runtime.endpoint.fault:
                    break
            if not runtime.finalized:
                decision = runtime.eof()
        except (ValueError, OSError, RuntimeError) as exc:
            error = f'{type(exc).__name__}: {exc}'
        diagnostic = dict(runtime.diagnostic)
        result = {**row, **decision, 'diagnostic': diagnostic,
                  'candidate': diagnostic.get('command'),
                  'endpoint_detected': runtime.finalized,
                  'capture_fault': error or runtime.endpoint.fault,
                  'replay_verified': verified and error is None,
                  'binding': runtime.binding, 'manifest_sha256': manifest_hash,
                  'policy_sha256': file_hash(Path(bundle) / 'policy.json'),
                  'evidence_type': ('human_waveform' if row.get('synthetic') is False
                                    else 'synthetic_or_unknown_origin'),
                  'processing_ms': (time.perf_counter() - started) * 1000,
                  'diagnostic_development': diagnostic_development}
        if error:
            result.update(accepted=False, command=None, reason='capture_fault')
        results.append(result)
        if prediction_log is not None:
            with Path(prediction_log).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                import os
                os.fsync(stream.fileno())
    return results


def evaluate(bundle: Path, manifest: Path, audio_root: Path, output: Path,
             contract: Contract, split: str,
             commitment: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError('Evaluation outputs are append-only')
    output.mkdir(parents=True)
    write_json(output / 'started.json', {'split': split, 'manifest_sha256': file_hash(manifest),
                                       'state': 'STARTED_PRESERVE_ON_FAILURE'})
    rows = replay(bundle, manifest, audio_root, contract, split, commitment,
                  prediction_log=output / 'predictions.jsonl')
    report = write_reports(rows, contract, output, existing=True)
    write_json(output / 'per_recording.json', rows)
    write_json(output / 'evaluation.json', {
        'split': split, 'binding': rows[0]['binding'],
        'manifest_sha256': file_hash(manifest),
        'rows_sha256': file_hash(output / 'per_recording.json'),
        'confirmation': split == 'confirmation', 'qualified': False})
    write_json(output / 'release.json', {'qualified': False,
        'status': 'ACOUSTIC_GATES_PASSED_PENDING_PHYSICAL' if report['acoustic_gates_passed'] else 'BLOCKED_ACOUSTIC',
        'failed_gates': report['failed_gates'], 'summary_sha256': file_hash(output / 'summary.json')})
    return report


def calibrate_bundle(bundle: Path, manifest: Path, audio_root: Path,
                     output: Path, contract: Contract) -> dict:
    if output.exists():
        raise FileExistsError('Calibration outputs are append-only')
    content = verify_bundle(bundle, contract)
    endpoint_receipt = json.loads((Path(bundle) / 'endpoint-selection.json').read_text())
    if (endpoint_receipt.get('status') != 'ENDPOINT_SELECTED'
            or endpoint_receipt.get('endpoint_sha256') != file_hash(Path(bundle) / 'endpoint.json')):
        raise ValueError('Endpoint must be selected and frozen before calibration')
    rows = replay(bundle, manifest, audio_root, contract, 'calibration')
    policy = calibrate(rows, contract, content['binding'])
    output.mkdir(parents=True)
    write_json(output / 'raw_calibration_rows.json', rows)
    write_json(output / 'calibration.json', policy)
    destination = output / 'bundle'
    destination.mkdir()
    for name in ASSETS:
        shutil.copyfile(bundle / name, destination / name)
    write_json(destination / 'policy.json', policy)
    seal_bundle(destination, contract, content['state_shapes'], policy['status'])
    evaluate(destination, manifest, audio_root, output / 'policy_replay',
             contract, 'calibration')
    return {'status': policy['status'], 'bundle': str(destination),
            'policy_sha256': file_hash(destination / 'policy.json'),
            'qualified': False}


def package_diagnostic(bundle: Path, output: Path, contract: Contract,
                       qualified: bool = False) -> dict:
    if qualified:
        raise ValueError('No qualified publisher until physical evidence is verified')
    content = verify_bundle(bundle, contract)
    if content['policy'].get('enabled') is not False:
        raise ValueError('Diagnostic publication must have acceptance disabled')
    if output.exists():
        raise FileExistsError('Never overwrite a previous runtime package')
    import vcm.frontend

    root = Path(__file__).parent
    files = {f'pcvcm/{name}': root / name for name in RUNTIME_CODE}
    files['pcvcm/__init__.py'] = root / '__init__.py'
    files['vcm/frontend.py'] = Path(vcm.frontend.__file__)
    entries = {name: path.read_bytes() for name, path in files.items()}
    entries['vcm/__init__.py'] = b'"""Finite VCM runtime frontend."""\n'
    for name in (*ASSETS, 'policy.json', 'recipe.json'):
        entries[f'bundle/{name}'] = (bundle / name).read_bytes()
    for name in contract.binding:
        entries[f'spec/{name}'] = (contract.root / name).read_bytes()
    entries['README.txt'] = (
        'DIAGNOSTIC ONLY; ACCEPTANCE DISABLED; NOT HARDWARE QUALIFIED.\n'
        'One own-trained ONNX graph. No external actions or transcription.\n'
        'Install separately audited target-architecture dependencies.\n').encode()
    manifest = {name: hashlib.sha256(value).hexdigest()
                for name, value in entries.items()}
    entries['MANIFEST.json'] = (json.dumps(manifest, sort_keys=True) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    return {'sha256': file_hash(output), 'qualified': False,
            'acceptance_enabled': False}
