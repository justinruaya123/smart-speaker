"""Expanded wake cohorts never borrow another voice split."""
import importlib


def test_wake_training_keeps_designed_test_voices_out_of_fitting():
    m=importlib.import_module('scripts.train.iris_wake')
    generated=[dict(id=s,split=s,wake_positive=True,synthetic=True) for s in
               ('train','dev','calibration','voice-test')]
    base=[dict(id='human-'+s,split=s,synthetic=False,text='tell me the time') for s in
          ('train','dev','calibration','test')]
    result=m.wake_rows(generated,base,'train')
    assert {r['id'] for r in result}=={'train','human-train'}
    assert next(r for r in result if not r['synthetic'])['wake_positive'] is False


def test_wake_test_is_the_frozen_new_voice_cohort():
    m=importlib.import_module('scripts.train.iris_wake')
    generated=[dict(id=s,split=s,wake_positive=True,synthetic=True) for s in ('calibration','voice-test')]
    rows=m.wake_rows(generated,[],'test')
    assert len(rows)==1 and rows[0]['id']=='voice-test'


def test_false_wake_upper_limit_does_not_treat_zero_events_as_certainty():
    m=importlib.import_module('scripts.train.iris_wake')
    assert 25 < m.zero_event_upper_per_hour(421.53) < 26


def test_wake_preparation_does_not_require_command_model_initializers(tmp_path,monkeypatch):
    import json,hashlib
    m=importlib.import_module('scripts.train.iris_wake')
    here=tmp_path/'experiments/iris-release-v1';here.mkdir(parents=True)
    inputs=tmp_path/'.cache/iris-release-v1/inputs';(inputs/'base-views').mkdir(parents=True)
    cache=tmp_path/'.cache/iris-release-v1/diversity';cache.mkdir(parents=True)
    (inputs/'normalization.json').write_text('{"mean":[],"std":[]}')
    (inputs/'wake-baseline.pt').write_bytes(b'project wake baseline placeholder; not loaded by data preparation')
    bindings={}
    for split in ('train','dev','calibration','test'):
        p=inputs/'base-views'/f'{split}.jsonl';p.write_text('')
        bindings[p.relative_to(tmp_path).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    # This nonexistent VCM initializer is deliberately irrelevant to wake data.
    bindings['.cache/iris-release-v1/inputs/compact-initializer.pt']='0'*64
    manifest=cache/'generated.jsonl';manifest.write_text('')
    (here/'generation-complete.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(b'').hexdigest()}))
    (here/'protocol.json').write_text(json.dumps({'input_bindings':bindings}))
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'HERE',here)
    monkeypatch.setattr(m,'INPUTS',inputs);monkeypatch.setattr(m,'CACHE',tmp_path/'.cache/iris-release-v1/wake')
    cfg=m.prepare()
    assert cfg['original_human_wake_positive_n']==0
    assert (here/'wake-protocol.json').is_file()
