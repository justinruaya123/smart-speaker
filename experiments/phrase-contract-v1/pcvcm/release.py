"""Fail-closed qualification derived from hash-bound recordings and raw traces.

No caller-supplied qualified/passed boolean grants publication. Physical
provenance still requires actual custodian/operator evidence; hashes cannot
prove that a recording or hardware measurement was honestly obtained.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .contract import canonical, file_hash, identity, read_jsonl
from .data import below


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_hardware(profile, traces, advertised, binding, policy, dependencies):
    require(advertised in ('pi4', 'pi5') and profile.get('profile') == advertised,
            'Physical profile differs from advertised Pi')
    require(f'Raspberry Pi {advertised[-1]}' in profile.get('model', ''), 'Wrong physical Pi model')
    for key in ('os', 'kernel', 'python', 'onnxruntime', 'microphone', 'gain', 'power', 'cooling', 'serial'):
        require(bool(profile.get(key)), f'Missing physical metadata: {key}')
    require(profile.get('machine') == 'aarch64' and profile['python'].startswith('3.12.'), 'Physical target mismatch')
    require(profile.get('binding') == binding and profile.get('policy_sha256') == policy
            and profile.get('dependency_sha256') == dependencies, 'Physical binding mismatch')
    require(profile.get('evidence_type') == 'physical_microphone', 'Not live physical evidence')
    require(isinstance(profile.get('ambient_c'), (int, float)) and math.isfinite(profile['ambient_c']), 'Ambient missing')
    require(0 < profile.get('graph_bytes', 0) < 1024 * 1024, 'Physical graph size failed')
    selected = profile.get('selected_threads')
    require(selected in (1, 2, 4), 'Missing selected thread profile')
    benchmark = traces.get('benchmark', [])
    require({r.get('threads') for r in benchmark} == {1, 2, 4}, 'Must measure 1/2/4 threads')
    components = ('frontend_ms', 'encoder_ms', 'decoder_ms', 'gate_ms', 'endpoint_ms')
    for kind in ('benchmark', 'sustained', 'negative'):
        rows = traces.get(kind, [])
        require(bool(rows), f'Missing {kind} trace')
        for row in rows:
            for field in (*components, 'processing_ms', 'rss_bytes', 'cpu_percent', 'temperature_c', 'elapsed_seconds'):
                value = row.get(field)
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f'Invalid {kind} {field}')
            require(row['processing_ms'] + 1e-6 >= sum(row[k] for k in components), 'Timing components do not reconcile')
            require(row.get('throttled') == 0 and row.get('queue_overruns') == 0
                    and row.get('capture_fault') is None, 'Capture/throttling gate failed')
            require(type(row.get('accepted')) is bool, 'Missing capture decision')
            require(row['rss_bytes'] < 256 * 1024 ** 2, 'RSS gate failed')
        if kind in ('sustained', 'negative'):
            next_sample, previous_time = 0, 0.
            for row in rows:
                require(row.get('start_sample') == next_sample and type(row.get('samples')) is int and row['samples'] > 0,
                        f'Noncontiguous {kind} sample sequence')
                require(row.get('threads') == selected and row['elapsed_seconds'] > previous_time,
                        'Capture thread/timestamp mismatch')
                next_sample += row['samples']; previous_time = row['elapsed_seconds']
            minimum = 1800 if kind == 'sustained' else 10800
            require(next_sample >= minimum * 16000 and previous_time >= minimum,
                    f'Insufficient {kind} duration')
            require(abs(previous_time - next_sample / 16000) <= max(1., previous_time * .01), 'Capture wall/audio duration mismatch')
            if kind == 'negative':
                require(not any(r['accepted'] for r in rows), 'Continuous negative acceptance')
        tested = rows if kind != 'benchmark' else [r for r in rows if r['threads'] == selected]
        require(np.quantile([r['processing_ms'] for r in tested], .95) <= 100, 'Processing p95 failed')
        total_audio = sum(r.get('samples', 0) for r in tested) / 16000
        require(total_audio > 0 and sum(r['processing_ms'] for r in tested) / 1000 / total_audio < 1,
                'Real-time factor failed')
    responses = [r['response_ms'] for r in benchmark if r['threads'] == selected and r.get('response_ms') is not None]
    require(responses and all(math.isfinite(v) and v >= 0 for v in responses)
            and np.quantile(responses, .95) <= 1000, 'Post-speech response p95 failed')
    return {'profile': advertised, 'passed': True, 'threads': selected,
            'sustained_seconds': sum(r['samples'] for r in traces['sustained']) / 16000,
            'negative_seconds': sum(r['samples'] for r in traces['negative']) / 16000}


def bound_files(evidence_path):
    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text())
    paths = {}
    for name, record in evidence.get('files', {}).items():
        path = below(evidence_path.parent, record['path'])
        require(file_hash(path) == record['sha256'], f'Evidence hash mismatch: {name}')
        paths[name] = path
    return evidence, paths


def verify_parity_binding(graph, parity, rows, checkpoint, selection, model, dev_hash, dev_rows):
    require(graph.get('checkpoint_sha256') == checkpoint and graph.get('selection_sha256') == selection
            and graph.get('graph_sha256') == model, 'Graph/checkpoint/selection binding mismatch')
    require(parity.get('manifest_sha256') == dev_hash, 'Waveform parity development manifest mismatch')
    require(len(rows) == len(dev_rows) and {r['recording_id'] for r in rows} == {r['recording_id'] for r in dev_rows},
            'Waveform parity rows do not reconcile with development')


def verify_clean_install(report, paths, content_hash):
    required = {'install', 'integrity', 'wav', 'list_mics', 'microphone'}
    commands = report.get('commands', [])
    require({c.get('id') for c in commands} == required and len(commands) == len(required), 'Clean-install commands incomplete')
    require(report.get('content_sha256') == content_hash, 'Clean-install payload identity mismatch')
    for command in commands:
        require(command.get('exit_code') == 0 and command.get('argv'), 'Clean-install command failed')
        require(command.get('stdout_key') in paths and command.get('stderr_key') in paths, 'Missing bound command output')
        stdout = paths[command['stdout_key']].read_text(encoding='utf-8')
        if command['id'] == 'integrity':
            result = json.loads(stdout)
            require(result.get('binding') == report['binding'] and result.get('integrity') == 'verified', 'Clean integrity output mismatch')
        elif command['id'] in ('wav', 'microphone'):
            result = json.loads(stdout)
            decision = result.get('decision', result)
            require(type(decision.get('accepted')) is bool and decision.get('schema_version') == 2,
                    'Clean command output missing typed decision')
        elif command['id'] == 'list_mics':
            require(bool(stdout.strip()), 'No microphone inventory')
    require(report.get('network_disabled') is True and report.get('research_unavailable') is True
            and report.get('operator') and report.get('tested_at')
            and report.get('isolation_evidence_key') in paths, 'Missing clean-install environment evidence')


def verify_capture_audio(sources, trace, paths):
    import soundfile as sf
    indexed = {r['recording_id']: r for r in sources}
    require(len(indexed) == len(sources) and sources, 'Repeated/missing capture sources')
    require({r.get('capture_id') for r in trace} == set(indexed), 'Capture sources disconnected from trace')
    for key, source in indexed.items():
        require(source.get('synthetic') is False and source.get('human_reviewed') is True
                and source.get('source_consent') and source.get('file_key') in paths, 'Missing original capture evidence')
        rows = [r for r in trace if r['capture_id'] == key]
        with sf.SoundFile(paths[source['file_key']]) as audio:
            require(audio.samplerate == 16000 and audio.channels == 1, 'Physical PCM format mismatch')
            for row in rows:
                require(row.get('capture_start_sample') == audio.tell(), 'Capture interval gap or duplication')
                pcm = audio.read(row['samples'], dtype='float32')
                require(len(pcm) == row['samples'] and hashlib.sha256(pcm.tobytes()).hexdigest() == row.get('pcm_sha256'),
                        'Physical PCM trace identity mismatch')
            require(audio.tell() == len(audio), 'Physical capture not fully accounted')


def reconcile_replay(rows, manifest, manifest_hash):
    original = {r['recording_id']: r for r in manifest}
    require(len(original) == len(manifest) == len(rows) and {r['recording_id'] for r in rows} == set(original), 'Replay rows do not reconcile')
    for row in rows:
        require(row.get('manifest_sha256') == manifest_hash, 'Replay manifest binding mismatch')
        for key, value in original[row['recording_id']].items():
            require(canonical(row.get(key)) == canonical(value), f'Replay reference changed: {key}')
        require(canonical(row.get('candidate')) == canonical(row.get('diagnostic', {}).get('command')),
                'Replay raw candidate differs from verified diagnostic')


def verify_frozen_commitment(freeze, commitment_hash):
    require(freeze['artifacts'].get('confirmation.commitment.json') == commitment_hash,
            'Confirmation commitment differs from pretraining freeze')


def verify_condition_replay(rows, sources, contract, policy, binding, policy_hash):
    from .evidence import validate_reference
    from .policy import decide
    references = {r['recording_id']: r for r in sources}
    require(rows and {r.get('condition_id') for r in rows} == {'C1', 'C2', 'C3', 'C4'}, 'Physical C1-C4 captures missing')
    for row in rows:
        require(row.get('capture_id') in references, 'Physical condition reference missing')
        source = references[row['capture_id']]
        for key in ('phrase_id', 'expected_command', 'reference_class', 'use_case', 'condition_id'):
            require(key in row and key in source and canonical(row[key]) == canonical(source[key]), 'Physical reference changed')
        validate_reference(row, contract)
        require(row.get('binding') == binding and row.get('policy_sha256') == policy_hash
                and row.get('replay_verified') is True and not row.get('capture_fault'), 'Physical condition replay binding/fault')
        diagnostic = row.get('diagnostic', {})
        require(canonical(row.get('candidate')) == canonical(diagnostic.get('command')), 'Physical raw candidate mismatch')
        decision = decide(diagnostic, policy, binding, endpoint=row.get('endpoint_detected') is True)
        require(all(k in row and row[k] == v for k, v in decision.items()), 'Physical decision differs from runtime')
    for condition in ('C1', 'C2', 'C3', 'C4'):
        positive = [r for r in rows if r['condition_id'] == condition and r['reference_class'] == 'supported_positive']
        emitted = [r for r in rows if r['condition_id'] == condition and r['accepted']]
        require(positive and emitted, 'Physical all-reject/missing positives')
        exact = lambda r: r.get('expected_command') is not None and canonical(r['command']) == canonical(r['expected_command'])
        raw = sum(canonical(r['candidate']) == canonical(r['expected_command']) for r in positive) / len(positive)
        recall = sum(r['accepted'] and exact(r) for r in positive) / len(positive)
        precision = sum(exact(r) for r in emitted) / len(emitted)
        require(raw >= (.95 if condition in ('C1', 'C2') else .9) and recall >= .9 and precision >= .99,
                'Physical condition recognition gate failed')


def verify_release(bundle, evidence_path, contract, lock, wheels):
    from .bundle import verify_bundle
    from .dependencies import verify_lock
    from .evidence import audit_recording_identities, summarize
    from .pipeline import verify_confirmation
    from .policy import calibrate, decide
    from .cv import cv_passed, speaker_folds
    from .training import prediction_metrics
    evidence, paths = bound_files(evidence_path)
    required = {'freeze', 'rights_review', 'training_protocol', 'selection', 'cv',
        'endpoint_selection', 'calibration_rows', 'confirmation_rows', 'confirmation_manifest',
        'confirmation_commitment', 'graph_parity', 'waveform_parity', 'waveform_parity_rows',
        'clean_install', 'checkpoint', 'train_manifest', 'dev_manifest', 'calibration_manifest',
        'development_predictions', 'cv_folds', 'cv0_predictions', 'cv1_predictions', 'cv2_predictions',
        'cv0_checkpoint', 'cv1_checkpoint', 'cv2_checkpoint'}
    require(required <= paths.keys(), 'Missing bound evidence: ' + ', '.join(sorted(required - paths.keys())))
    content = verify_bundle(bundle, contract)
    binding, policy = content['binding'], content['policy']
    policy_hash = file_hash(Path(bundle) / 'policy.json')
    dependencies = verify_lock(lock, wheels)
    dependency_hash = file_hash(Path(lock) / 'dependencies.json')
    load = lambda key: json.loads(paths[key].read_text())
    freeze, protocol, selection = load('freeze'), load('training_protocol'), load('selection')
    require(freeze['contract_sha256'] == contract.sha256, 'Data contract mismatch')
    require(protocol['freeze_sha256'] == file_hash(paths['freeze']), 'Training/data binding mismatch')
    require(selection['protocol_sha256'] == identity(protocol) and selection['status'] == 'DEVELOPMENT_SELECTED', 'Development selection failed')
    require(file_hash(paths['checkpoint']) == selection['winner']['checkpoint_sha256'], 'Selected checkpoint mismatch')
    require(selection['winner']['phrases_passing'] == 570, 'Development phrase gate failed')
    required_ucs = {f'UC-{i:02}' for i in range(1, 11)}
    development_predictions = load('development_predictions')
    require(file_hash(paths['development_predictions']) == selection['winner']['predictions_sha256'], 'Development predictions changed')
    actual_dev = prediction_metrics(development_predictions, set(contract.by_id), required_ucs)
    dev_original = {r['recording_id']: r for r in read_jsonl(paths['dev_manifest'])}
    require(len(development_predictions) == len(dev_original) and {r['recording_id'] for r in development_predictions} == set(dev_original), 'Development coverage changed')
    for row in development_predictions:
        original = dev_original[row['recording_id']]
        require(row['expected'] == original['expected_command'] and row['phrase_id'] == original['phrase_id']
                and row['use_case'] == original['use_case'] and row['reference_class'] == original['reference_class'], 'Development reference changed')
    require(actual_dev['phrases_passing'] == 570, 'Recomputed development phrase gate failed')
    cv = load('cv')
    folds = speaker_folds(read_jsonl(paths['train_manifest']))
    fold_receipt = load('cv_folds')
    require(fold_receipt['folds'] == folds and cv['folds_sha256'] == file_hash(paths['cv_folds']), 'CV speaker fold binding mismatch')
    actual_cv = []
    for index, fold in enumerate(folds):
        predictions = load(f'cv{index}_predictions')
        original = {r['recording_id']: r for r in read_jsonl(paths['train_manifest']) if r['speaker_id'] in fold['held_speakers']}
        require(len(predictions) == len(original) and {r['recording_id'] for r in predictions} == set(original), 'CV held speaker rows differ')
        for row in predictions:
            require(row['expected'] == original[row['recording_id']]['expected_command'] and row['phrase_id'] == original[row['recording_id']]['phrase_id'], 'CV reference changed')
        require(file_hash(paths[f'cv{index}_predictions']) == cv['folds'][index]['predictions_sha256'] and
                file_hash(paths[f'cv{index}_checkpoint']) == cv['folds'][index]['checkpoint_sha256'], 'CV checkpoint/prediction binding changed')
        actual_cv.append(prediction_metrics(predictions, set(contract.by_id), required_ucs))
    require(cv_passed(actual_cv) and cv['freeze_sha256'] == file_hash(paths['freeze']), 'CV gate or data binding failed')
    audit = load('rights_review')
    require(audit.get('frozen_data_sha256') == file_hash(paths['freeze']) and audit.get('passed') is True,
            'Missing human rights/duplicate review')
    for key in ('reviewer', 'reviewed_at', 'source_consent_evidence', 'manual_label_evidence', 'near_duplicate_pairs'):
        require(bool(audit.get(key)), f'Missing reviewed evidence: {key}')
    endpoint = load('endpoint_selection')
    require(endpoint.get('status') == 'ENDPOINT_SELECTED' and endpoint['winner']['misses'] == 0
            and endpoint['winner']['premature_pauses'] == 0, 'Endpoint selection failed')
    require(endpoint['endpoint_sha256'] == file_hash(Path(bundle) / 'endpoint.json') and
            endpoint['checkpoint_sha256'] == file_hash(paths['checkpoint']), 'Endpoint binding mismatch')
    for split in ('train', 'dev', 'calibration'):
        require(freeze['artifacts'][f'{split}.jsonl'] == file_hash(paths[f'{split}_manifest']), 'Prepared manifest changed')
    manifests = [r for s in ('train', 'dev', 'calibration', 'confirmation') for r in read_jsonl(paths[f'{s}_manifest'])]
    audit_recording_identities(manifests)
    require('audio_root' in evidence and 'prepared_root' in evidence, 'Missing original audio/features root')
    audio_root = below(Path(evidence_path).parent, evidence['audio_root'])
    prepared_root = below(Path(evidence_path).parent, evidence['prepared_root'])
    import soundfile as sf
    from .data import verified_features
    for row in manifests:
        require(row.get('synthetic') is False and row.get('human_reviewed') is True, 'Unreviewed original recording')
        audio = below(audio_root, row['audio'])
        require(file_hash(audio) == row['audio_sha256'], 'Original recording changed')
        pcm, rate = sf.read(audio, dtype='float32')
        require(rate == 16000 and pcm.ndim == 1 and hashlib.sha256(pcm.tobytes()).hexdigest() == row['pcm_sha256'], 'Original PCM changed')
        if row['split'] != 'confirmation':
            verified_features(row, prepared_root)
    verify_confirmation(paths['confirmation_manifest'], load('confirmation_commitment'), contract)
    verify_frozen_commitment(freeze, file_hash(paths['confirmation_commitment']))
    calibration = load('calibration_rows')
    reconcile_replay(calibration, read_jsonl(paths['calibration_manifest']), file_hash(paths['calibration_manifest']))
    expected_policy = calibrate(calibration, contract, binding)
    require(expected_policy['enabled'] and canonical(expected_policy) == canonical(policy), 'Calibrated policy mismatch/failure')
    confirmation = load('confirmation_rows')
    reconcile_replay(confirmation, read_jsonl(paths['confirmation_manifest']), file_hash(paths['confirmation_manifest']))
    final_manifest = {r['recording_id']: r for r in read_jsonl(paths['confirmation_manifest'])}
    require(len(confirmation) == len(final_manifest) and {r['recording_id'] for r in confirmation} == final_manifest.keys(), 'Confirmation rows do not reconcile')
    for row in confirmation:
        original = final_manifest[row['recording_id']]
        for key, value in original.items():
            require(canonical(row.get(key)) == canonical(value), f'Confirmation reference changed: {key}')
        require(row.get('binding') == binding and row.get('policy_sha256') == policy_hash, 'Confirmation binding mismatch')
        actual = decide(row['diagnostic'], policy, binding, endpoint=row.get('endpoint_detected') is True, fault=row.get('capture_fault'))
        require(all(row.get(k) == v for k, v in actual.items()), 'Confirmation decision differs from shipped predicate')
    report = summarize(confirmation, contract)
    require(report['acoustic_gates_passed'], 'Confirmation acoustic gates failed: ' + ', '.join(report['failed_gates']))
    graph = load('graph_parity')
    require(graph['graph_sha256'] == file_hash(Path(bundle) / 'model.onnx') and graph['max_absolute_error'] < 2e-5
            and graph['precision'] == 'FP32' and [t['frames'] for t in graph['trials']] == [1, 3, 4, 8, 17], 'Graph parity failed')
    parity = load('waveform_parity'); parity_rows = load('waveform_parity_rows')
    verify_parity_binding(graph, parity, parity_rows, file_hash(paths['checkpoint']),
        file_hash(paths['selection']), file_hash(Path(bundle) / 'model.onnx'),
        file_hash(paths['dev_manifest']), read_jsonl(paths['dev_manifest']))
    require(parity['binding'] == binding and parity['policy_sha256'] == policy_hash and parity['rows_sha256'] == file_hash(paths['waveform_parity_rows']), 'Waveform parity binding mismatch')
    require(parity_rows and all(r['decisions'][0] == r['decisions'][1] and r['candidates'][0] == r['candidates'][1] and r['endpoints'][0] == r['endpoints'][1] for r in parity_rows), 'Waveform parity failed')
    clean = load('clean_install')
    require(clean.get('binding') == binding and clean.get('dependency_sha256') == dependency_hash
            and clean.get('policy_sha256') == policy_hash, 'Clean-install binding mismatch')
    from .packaging import runtime_content_hash
    verify_clean_install(clean, paths, runtime_content_hash(bundle, contract, lock, wheels))
    profiles = evidence.get('advertised_profiles', [])
    require(profiles and set(profiles) <= {'pi4', 'pi5'}, 'Explicit advertised profiles required')
    hardware = []
    for name in profiles:
        require({name, f'{name}_benchmark', f'{name}_sustained', f'{name}_negative', f'{name}_negative_sources',
                 f'{name}_condition_rows', f'{name}_condition_sources', f'{name}_sustained_sources'} <= paths.keys(), 'Missing bound hardware evidence')
        profile = load(name)
        require(profile['onnxruntime'] == dependencies['wheels']['onnxruntime']['version'], 'Physical ORT differs from package')
        sources = load(f'{name}_negative_sources')
        require(sources and len({r.get('category') for r in sources}) >= 3, 'Diverse real negative source evidence missing')
        traces = {kind: read_jsonl(paths[f'{name}_{kind}']) for kind in ('benchmark', 'sustained', 'negative')}
        verify_capture_audio(sources, traces['negative'], paths)
        verify_capture_audio(load(f'{name}_sustained_sources'), traces['sustained'], paths)
        condition_rows = load(f'{name}_condition_rows')
        verify_condition_replay(condition_rows, load(f'{name}_condition_sources'), contract, policy, binding, policy_hash)
        verify_capture_audio(load(f'{name}_condition_sources'), condition_rows, paths)
        hardware.append(verify_hardware(profile, traces, name, binding, policy_hash, dependency_hash))
    return {'status': 'QUALIFIED_PC_V1', 'binding': binding, 'policy_sha256': policy_hash,
            'dependency_sha256': dependency_hash, 'evidence_sha256': file_hash(evidence_path),
            'hardware': hardware, 'acoustic_summary': report, 'qualified': True}
