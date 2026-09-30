import copy
import pytest


def profile():
    return {'profile': 'pi4', 'model': 'Raspberry Pi 4 Model B', 'os': 'Debian',
        'kernel': '6.1', 'python': '3.12.13', 'onnxruntime': '1.30.0', 'machine': 'aarch64',
        'microphone': 'test fixture', 'gain': 'fixed', 'power': 'test fixture',
        'cooling': 'test fixture', 'ambient_c': 23, 'serial': 'test fixture',
        'binding': 'binding', 'policy_sha256': 'policy', 'dependency_sha256': 'dependencies',
        'selected_threads': 1, 'evidence_type': 'physical_microphone', 'graph_bytes': 500000}


def trace(seconds, thread=1):
    # Software fixtures, never physical receipts.
    return [dict(start_sample=i*16000, samples=16000, elapsed_seconds=i+1,
        threads=thread, frontend_ms=1., encoder_ms=2., decoder_ms=3., gate_ms=.1,
        endpoint_ms=.1, processing_ms=6.2, response_ms=710., rss_bytes=100000000,
        cpu_percent=50., temperature_c=50., throttled=0, queue_overruns=0,
        capture_fault=None, accepted=False) for i in range(seconds)]


def test_hardware_is_derived_from_contiguous_measurements_not_pass_boolean():
    from pcvcm.release import verify_hardware
    p = profile(); p['qualified'] = True
    measured = {'benchmark': sum((trace(10, t) for t in (1, 2, 4)), []),
                'sustained': trace(1800), 'negative': trace(10800)}
    result = verify_hardware(p, measured, 'pi4', 'binding', 'policy', 'dependencies')
    assert result['passed']
    measured['negative'][-1]['accepted'] = True
    with pytest.raises(ValueError, match='negative'):
        verify_hardware(p, measured, 'pi4', 'binding', 'policy', 'dependencies')


def test_pi5_cannot_qualify_pi4_and_gaps_nan_or_short_duration_fail():
    from pcvcm.release import verify_hardware
    good = {'benchmark': sum((trace(10, t) for t in (1, 2, 4)), []),
            'sustained': trace(1800), 'negative': trace(10800)}
    for change in ['profile', 'gap', 'nan', 'short']:
        p, m = profile(), copy.deepcopy(good)
        if change == 'profile': p['profile'] = 'pi5'
        if change == 'gap': m['sustained'][3]['start_sample'] += 1
        if change == 'nan': m['benchmark'][0]['processing_ms'] = float('nan')
        if change == 'short': m['negative'].pop()
        with pytest.raises(ValueError):
            verify_hardware(p, m, 'pi4', 'binding', 'policy', 'dependencies')


def test_qualified_publication_refuses_missing_evidence_even_with_boolean(tmp_path):
    from pcvcm.release import verify_release
    from pcvcm.contract import Contract, write_json
    from pathlib import Path
    evidence = tmp_path / 'release.json'
    write_json(evidence, {'qualified': True, 'files': {}, 'advertised_profiles': ['pi4', 'pi5']})
    with pytest.raises(ValueError, match='Missing bound evidence'):
        verify_release(tmp_path / 'bundle', evidence, Contract(Path(__file__).resolve().parents[1] / 'spec'), tmp_path, tmp_path)


def test_clean_install_needs_all_bound_command_outputs(tmp_path):
    from pcvcm.release import verify_clean_install
    with pytest.raises(ValueError, match='commands'):
        verify_clean_install({'commands': [{'exit_code': 0}], 'network_disabled': True,
            'research_unavailable': True, 'content_sha256': 'content'}, {}, 'content')


def test_parity_rejects_checkpoint_or_development_manifest_mixup():
    from pcvcm.release import verify_parity_binding
    graph = {'checkpoint_sha256': 'wrong', 'selection_sha256': 'selection', 'graph_sha256': 'graph'}
    with pytest.raises(ValueError, match='checkpoint'):
        verify_parity_binding(graph, {}, [], 'checkpoint', 'selection', 'graph', 'dev', [])


def test_physical_capture_trace_requires_original_pcm_identity(tmp_path):
    import hashlib
    import soundfile as sf
    import numpy as np
    from pcvcm.release import verify_capture_audio
    pcm = np.full(320, .1, dtype='float32')
    path = tmp_path / 'capture.wav'; sf.write(path, pcm, 16000, subtype='FLOAT')
    source = dict(recording_id='capture', file_key='capture', source_consent='receipt',
                  synthetic=False, human_reviewed=True, category='conversation')
    rows = [dict(capture_id='capture', capture_start_sample=0, samples=320,
                 pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())]
    verify_capture_audio([source], rows, {'capture': path})
    rows[0]['pcm_sha256'] = 'different'
    with pytest.raises(ValueError, match='PCM'):
        verify_capture_audio([source], rows, {'capture': path})


def test_replay_metadata_and_raw_candidate_cannot_be_substituted():
    from pcvcm.release import reconcile_replay
    manifest = [dict(recording_id='r', split='calibration', audio_sha256='a', expected_command={'value': 37})]
    row = {**manifest[0], 'manifest_sha256': 'm', 'candidate': {'value': 37},
           'diagnostic': {'command': {'value': 73}}}
    with pytest.raises(ValueError, match='candidate'):
        reconcile_replay([row], manifest, 'm')
    row['candidate'] = row['diagnostic']['command']
    reconcile_replay([row], manifest, 'm')
    row['audio_sha256'] = 'substituted'
    with pytest.raises(ValueError, match='reference'):
        reconcile_replay([row], manifest, 'm')


def test_pretraining_confirmation_commitment_cannot_be_replaced():
    from pcvcm.release import verify_frozen_commitment
    with pytest.raises(ValueError, match='pretraining'):
        verify_frozen_commitment({'artifacts': {'confirmation.commitment.json': 'original'}}, 'new')


def test_physical_conditions_without_recognition_or_all_reject_fail():
    from pcvcm.release import verify_condition_replay
    from pcvcm.contract import Contract
    from pathlib import Path
    contract = Contract(Path(__file__).resolve().parents[1] / 'spec')
    rows = [{'condition_id': c} for c in ['C1', 'C2', 'C3', 'C4']]
    with pytest.raises(ValueError):
        verify_condition_replay(rows, [], contract, {}, 'b', 'p')
