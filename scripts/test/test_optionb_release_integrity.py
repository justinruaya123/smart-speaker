"""Tamper guards for the actual consumed release inputs; no model fitting."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.test import optionb_release_check as checker
from scripts.test.speaker_demo_suite import pcm_digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def frozen(tmp_path):
    bindings = {}
    def save(name, value):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r) + '\n' for r in value), encoding='utf8')
        bindings[name] = sha(path)
        return name
    audio = tmp_path / 'audio.wav'
    sf.write(audio, np.full(1600, .1, dtype=np.float32), 16000, subtype='PCM_16')
    pcm, rate = sf.read(audio, dtype='float32')
    base = dict(kind='human', synthetic=False, metric_eligible=True, training_eligible=False,
                class_index=0, expected=None, scope='original_unknown', audio=str(audio),
                sha256=sha(audio), pcm_sha256=pcm_digest(pcm, rate), location={'row': 1})
    rows = {s: [{**base, 'id': s + '-one', 'split': s, 'speaker': s + '-speaker'}]
            for s in ('train', 'dev', 'calibration', 'test', 'holdout')}
    files = {s: save('parent/' + s + '.jsonl', r) for s, r in rows.items()}
    canonical = {s: save('canonical/' + s + '.jsonl', r) for s, r in rows.items()}
    references = [r for s in ('test', 'holdout') for r in rows[s]]
    metadata = save('metadata.jsonl', references)
    predictions = save('predictions.jsonl', [{**r, 'parent_id': r['id'], 'variant': 'original',
            'event': {'output': {'accepted': False}, 'application': {'accepted': False}}} for r in references])
    protocol = dict(format='optionb-release-freeze-v2', parameters=258912, selected_epoch=44,
        policy={'enabled': True, 'min_probability': .5, 'min_margin': .1},
        manifest_files=files, canonical_manifests=canonical, wording_metadata=metadata,
        cached_predictions=predictions, canonical_human_counts={'test': 1, 'holdout': 1}, bindings=bindings)
    return tmp_path, protocol


def test_every_consumed_manifest_requires_its_own_binding(frozen):
    root, protocol = frozen
    del protocol['bindings'][protocol['manifest_files']['dev']]
    with pytest.raises(ValueError, match='Unbound consumed input'):
        checker.validate_inputs(protocol, root=root)


def test_run_check_rejects_unbound_inputs_before_loading_any_model(frozen, monkeypatch):
    root, protocol = frozen
    del protocol['bindings'][protocol['manifest_files']['dev']]
    path = root / 'protocol.json'; path.write_text(json.dumps(protocol))
    monkeypatch.chdir(root)
    with pytest.raises(ValueError, match='Unbound consumed input'):
        checker.run_check(path, root / 'never-verified')
    assert not (root / 'never-verified/results.json').exists()


def test_validated_inputs_use_independent_canonical_human_membership(frozen):
    root, protocol = frozen
    checked = checker.validate_inputs(protocol, root=root)
    assert checked['expected_human_keys'] == [('test', 'test-one'), ('holdout', 'holdout-one')]


@pytest.mark.parametrize('mutation', ['drop', 'duplicate', 'empty_humans', 'substitute'])
def test_rehashing_a_changed_consumed_cohort_does_not_make_it_canonical(frozen, mutation):
    root, protocol = frozen
    name = protocol['manifest_files']['holdout']; path = root / name
    row = json.loads(path.read_text())
    changed = {'drop': [], 'duplicate': [row, row], 'empty_humans': [{**row, 'kind': 'synthetic'}],
               'substitute': [{**row, 'id': 'wrong-recording'}]}[mutation]
    path.write_text(''.join(json.dumps(r) + '\n' for r in changed))
    protocol['bindings'][name] = sha(path)
    with pytest.raises(ValueError, match='canonical|Duplicate'):
        checker.validate_inputs(protocol, root=root)


@pytest.mark.parametrize('key', ['cached_predictions', 'wording_metadata'])
def test_changed_reference_bytes_fail_their_binding(frozen, key):
    root, protocol = frozen
    path = root / protocol[key]
    with path.open('a') as stream: stream.write('{}\n')
    with pytest.raises(ValueError, match='identity'):
        checker.validate_inputs(protocol, root=root)


@pytest.mark.parametrize('field', ['sha256', 'pcm_sha256'])
def test_preflight_rejects_changed_wav_or_pcm_identity(frozen, field):
    root, protocol = frozen
    # Coherent manifest metadata is still not proof that the actual audio matches.
    for name in (protocol['manifest_files']['holdout'], protocol['canonical_manifests']['holdout']):
        path = root / name; row = json.loads(path.read_text()); row[field] = '0' * 64
        path.write_text(json.dumps(row) + '\n'); protocol['bindings'][name] = sha(path)
    for key in ('wording_metadata', 'cached_predictions'):
        path = root / protocol[key]; rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            if row['split'] == 'holdout': row[field] = '0' * 64
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows)); protocol['bindings'][protocol[key]] = sha(path)
    with pytest.raises(ValueError, match='Audio file hash|PCM hash'):
        checker.preflight(protocol, root=root, split='holdout')
