"""Evidence and release gates may not be passed with placeholders or averages."""
from pathlib import Path
import pytest
from pcvcm.contract import Contract
from pcvcm.evidence import summarize, audit_partitions, recording_plan
from pcvcm.policy import decide, calibrate

ROOT = Path(__file__).resolve().parents[1]


def test_empty_and_all_reject_never_pass():
    contract = Contract(ROOT / 'spec')
    assert not summarize([], contract)['acoustic_gates_passed']
    assert summarize([], contract)['precision'] is None
    row = contract.positive[0]
    rows = [dict(recording_id='r', phrase_id=row['phrase_id'], use_case=row['use_case'],
                 expected_command=row['expected_command'], command=None, accepted=False,
                 speaker_id='s', condition_id='C1', reference_class='supported_positive',
                 evidence_type='human_waveform')]
    result = summarize(rows, contract)
    assert result['recall'] == 0
    assert not result['acoustic_gates_passed']
    assert len(result['per_phrase']) == 570


def test_cross_family_errors_pollute_predicted_family_precision():
    contract = Contract(ROOT / 'spec')
    positive = contract.positive[0]
    wrong = next(r for r in contract.positive if r['use_case'] == 'UC-02')
    row = dict(recording_id='r', phrase_id=positive['phrase_id'],
               use_case=positive['use_case'], expected_command=positive['expected_command'],
               command=wrong['expected_command'], accepted=True, speaker_id='s',
               condition_id='C1', reference_class='supported_positive',
               evidence_type='human_waveform')
    report = summarize([row], contract)
    assert report['per_use_case']['UC-02']['predicted_emissions'] == 1
    assert report['per_use_case']['UC-02']['precision'] == 0


def test_synthetic_evidence_cannot_qualify_even_when_perfect():
    contract = Contract(ROOT / 'spec')
    rows = []
    for phrase in contract.positive:
        for i in range(20):
            rows.append(dict(recording_id=f"{phrase['phrase_id']}-{i}",
                             phrase_id=phrase['phrase_id'], use_case=phrase['use_case'],
                             expected_command=phrase['expected_command'],
                             command=phrase['expected_command'], candidate=phrase['expected_command'],
                             accepted=True, speaker_id=f's{i % 10}', condition_id=f'C{i // 5 + 1}',
                             reference_class='supported_positive', evidence_type='software_fixture'))
    report = summarize(rows, contract)
    assert report['recall'] == 1
    assert all(r['passed'] for r in report['per_phrase'])
    assert not report['acoustic_gates_passed']
    assert 'nonhuman_or_unverified_evidence' in report['failed_gates']
    removed = [r for r in rows if r['phrase_id'] != contract.positive[0]['phrase_id']]
    assert not summarize(removed, contract)['per_phrase'][0]['passed']


def test_leakage_detected_for_speaker_parent_pcm_and_recording_id():
    for field in ['speaker_id', 'utterance_parent_id', 'pcm_sha256', 'recording_id']:
        a = dict(split='train', speaker_id='s1', utterance_parent_id='a1',
                 pcm_sha256='h1', recording_id='r1')
        b = dict(split='dev', speaker_id='s2', utterance_parent_id='a2',
                 pcm_sha256='h2', recording_id='r2')
        b[field] = a[field]
        with pytest.raises(ValueError):
            audit_partitions([a, b])


def test_recording_plan_is_expanded_and_speaker_disjoint():
    contract = Contract(ROOT / 'spec')
    rows = recording_plan(contract)
    positives = [r for r in rows if r['reference_class'] == 'supported_positive']
    assert len(positives) == 34200
    final = [r for r in positives if r['split'] == 'confirmation']
    assert len(final) == 11400
    assert len({r['recording_id'] for r in rows}) == len(rows)
    pools = {s: {r['speaker_slot'] for r in rows if r['split'] == s}
             for s in ['train', 'dev', 'calibration', 'confirmation']}
    assert all(not pools[a] & pools[b] for a in pools for b in pools if a != b)


def test_one_shared_decision_function_fail_closed():
    command = {'intent': 'TIME_QUERY', 'timezone': 'Asia/Manila'}
    candidate = dict(command=command, score=-.1, margin=.8, use_case='UC-02')
    policy = dict(enabled=True, binding='abc', thresholds={'global': [-.5, .2]})
    assert decide(candidate, policy, 'abc', endpoint=True)['accepted']
    for kwargs in [{'endpoint': False}, {'endpoint': True, 'fault': 'capture_gap'}]:
        assert not decide(candidate, policy, 'abc', **kwargs)['accepted']
    assert not decide(candidate, policy, 'changed', endpoint=True)['accepted']
    for key, value in [('score', float('nan')), ('margin', 0), ('command', None)]:
        assert not decide({**candidate, key: value}, policy, 'abc', endpoint=True)['accepted']


def test_calibration_refuses_noncalibration_data():
    contract = Contract(ROOT / 'spec')
    with pytest.raises(ValueError):
        calibrate([{'split': 'confirmation'}], contract, 'binding')


def test_repeated_parent_or_pcm_inside_split_is_not_independent():
    a = dict(split='train', speaker_id='s1', utterance_parent_id='a1',
             pcm_sha256='h1', recording_id='r1')
    for field in ['utterance_parent_id', 'pcm_sha256']:
        b = dict(split='train', speaker_id='s1', utterance_parent_id='a2',
                 pcm_sha256='h2', recording_id='r2')
        b[field] = a[field]
        with pytest.raises(ValueError):
            audit_partitions([a, b])


def test_reference_cannot_be_relabelled_to_match_a_wrong_prediction():
    contract = Contract(ROOT / 'spec')
    phrase = contract.positive[0]
    other = contract.positive[2]
    row = dict(recording_id='r', phrase_id=phrase['phrase_id'],
               use_case=phrase['use_case'], reference_class='supported_positive',
               expected_command=other['expected_command'],
               command=other['expected_command'], accepted=True)
    with pytest.raises(ValueError):
        summarize([row], contract)


def test_missing_positive_reference_cannot_be_called_unknown():
    contract = Contract(ROOT / 'spec')
    phrase = contract.positive[0]
    row = dict(recording_id='r', phrase_id=phrase['phrase_id'],
               use_case=phrase['use_case'], reference_class='unknown_no_action',
               expected_command=None, command=None, accepted=False)
    with pytest.raises(ValueError):
        summarize([row], contract)


def test_natural_unknown_far_cannot_hide_in_prompted_denominator():
    contract = Contract(ROOT / 'spec')
    command = contract.positive[0]['expected_command']
    rows = [dict(recording_id=f'n{i}', phrase_id=None, natural_unknown=True,
                 reference_class='unknown_no_action', use_case='UC-01', expected_command=None,
                 accepted=i < 11, command=command if i < 11 else None) for i in range(1000)]
    report = summarize(rows, contract)
    assert report['negative_classes']['natural_unknown']['far'] == .011
    assert 'separate_negative_class_far' in report['failed_gates']


def test_strict_identity_audit_requires_sessions_and_original_hashes():
    from pcvcm.evidence import audit_recording_identities
    a = dict(split='train', speaker_id='s1', utterance_parent_id='p1', pcm_sha256='h1',
             audio_sha256='raw1', session_id='session1', recording_id='r1')
    b = dict(split='dev', speaker_id='s2', utterance_parent_id='p2', pcm_sha256='h2',
             audio_sha256='raw2', session_id='session2', recording_id='r2')
    assert audit_recording_identities([a, b])['rows'] == 2
    for field in ['session_id', 'audio_sha256']:
        bad = {**b, field: a[field]}
        with pytest.raises(ValueError):
            audit_recording_identities([a, bad])


def test_each_condition_raw_gate_cannot_be_rescued_by_other_condition():
    contract = Contract(ROOT / 'spec'); phrase = contract.positive[0]
    rows = []
    for i in range(20):
        rows.append(dict(recording_id=str(i), phrase_id=phrase['phrase_id'],
            expected_command=phrase['expected_command'], use_case=phrase['use_case'],
            reference_class='supported_positive', condition_id=f'C{i // 5 + 1}',
            candidate=None if i == 10 else phrase['expected_command'],
            command=None if i == 10 else phrase['expected_command'], accepted=i != 10))
    report = summarize(rows, contract)
    assert report['per_use_case'][phrase['use_case']]['acoustic_slices']['C3']['passed'] is False
