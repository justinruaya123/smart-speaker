"""Release preparation boundaries; fixtures never read held-out transcripts."""
import importlib
import importlib.util
import json

import numpy as np
import pytest

from scripts.runtime.pcv_schema import command
from scripts.train.pcv_public_data import sha
from vcm.frontend import SPEC


def module():
    name = 'scripts.train.pcv_release_prepare'
    assert importlib.util.find_spec(name), 'Release preparation implementation is missing'
    return importlib.import_module(name)


def row(text='make light color red', **extra):
    result = dict(id='fixture', transcript=text, command='COLOR', slot_value='',
                  published_split='train', source='fixture', speaker_id='speaker',
                  is_synthetic=0, transcript_source='prompt', whisper_check='',
                  file='fixture.wav', out_of_scope=0, waveform_flags=[])
    return {**result, **extra}


def test_exact_overrides_preserve_typed_target_across_splits():
    m = module()
    for split in ('train', 'dev', 'calibration', 'test', 'holdout'):
        r = row(published_split=split)
        assert m.annotate(r)['expected'] == command('COLOR', light_id='light.all', color='red')
    for text in ('turn the lights red in the bed room', 'change the color of the lights in my bedroom to red'):
        assert m.annotate(row(text))['expected']['slots']['light_id'] == 'light.bedroom'


def test_publisher_flags_cannot_be_rescued_by_override():
    m = module()
    assert m.annotate(row(out_of_scope=1))['reference_status'] == 'unsupported'
    assert m.annotate(row(command='OUT_OF_SCOPE'))['reference_status'] == 'unsupported'
    assert m.annotate(row(whisper_check='not understood'))['reference_status'] == 'ambiguous'


def test_uncertain_transcript_is_not_negative_ground_truth():
    r = module().annotate(row('the dim lights to eighty percent', command='BRIGHTNESS'))
    assert r['expected'] is None and r['reference_status'] == 'ambiguous'


@pytest.mark.parametrize('text,family', [
    ('make the attic light red', 'COLOR'), ('call Jane', 'CALL'),
    ('play Queen', 'PLAY_MUSIC'), ('turn volume down by twenty percent', 'VOLUME_DOWN'),
    ('set an alarm tomorrow at six a m', 'ALARM'),
    ('please switch to the green light', 'COLOR'), ('play it louder', 'VOLUME_UP'),
])
def test_unsupported_or_conflated_requests_never_promoted(text, family):
    assert module().annotate(row(text, command=family))['expected'] is None


def test_catalog_does_not_learn_dev_or_test_text():
    m = module()
    rows = [dict(text='train fixture', expected=command('TIME'), split='train',
                 training_eligible=True, reference_status='supported'),
            dict(text='dev fixture', expected=command('TIME'), split='dev',
                 training_eligible=True, reference_status='supported'),
            dict(text='test fixture', expected=command('TIME'), split='test',
                 training_eligible=True, reference_status='supported'),
            dict(text='aux fixture', expected=None, split='train',
                 training_eligible=True, reference_status='auxiliary'),
            dict(text='bad fixture', expected=None, split='train',
                 training_eligible=False, reference_status='ambiguous')]
    definitions = m.catalog_definitions(rows, [])
    assert [r['text'] for r in definitions] == ['train fixture']


def test_predeclared_catalog_includes_all_415_and_mother():
    m = module()
    prompts = m.approved_definitions()
    assert len(prompts) == 415
    mother = [r for r in prompts if m.spoken(r['text']) == 'call mother']
    assert len(mother) == 1 and mother[0]['command'] == command('CALL', contact_id='contact.mom')


def test_normalization_uses_only_fresh_eligible_human_train(tmp_path):
    m = module()
    source = []
    for i, (split, synthetic, eligible, status, value) in enumerate([
        ('train', False, True, 'supported', 2.),
        ('train', False, True, 'unsupported', 4.),
        ('test', False, True, 'supported', 100.),
        ('train', True, True, 'supported', 200.),
        ('train', False, False, 'supported', 300.),
        ('train', False, True, 'auxiliary', 400.),
    ]):
        path = tmp_path / f'{i}.npy'
        np.save(path, np.full((40, 3), value, dtype='float32'))
        source.append(dict(id=str(i), split=split, synthetic=synthetic,
                           training_eligible=eligible, reference_status=status,
                           features=str(path), features_sha256=sha(path)))
    result = m.fit_normalization(source)
    assert result['human_training_utterances'] == 2
    assert result['mean'] == [3.] * 40 and result['std'] == [1.] * 40
    assert result['source_ids'] == ['0', '1']


def test_reuse_requires_matching_pcm_spec_and_exact_feature_bytes(tmp_path):
    m = module()
    path = tmp_path / 'features.npy'
    np.save(path, np.zeros((40, 12), dtype='float32'))
    entry = dict(pcm_sha256='pcm', features=str(path), features_sha256=sha(path),
                 frontend=SPEC, frames=12)
    index = {'pcm': entry}
    assert m.reuse_feature({'pcm_sha256': 'different'}, index) is None
    assert m.reuse_feature({'pcm_sha256': 'pcm'}, index)['features'] == str(path)
    with pytest.raises(ValueError, match='frontend'):
        m.reuse_feature({'pcm_sha256': 'pcm'}, {'pcm': {**entry, 'frontend': {**SPEC, 'bins': 80}}})
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        m.reuse_feature({'pcm_sha256': 'pcm'}, index)


def test_cross_split_pcm_and_speaker_duplicates_fail_closed():
    m = module()
    a = dict(id='a', speaker='speaker-a', pair='pair-a', pcm_sha256='pcm-a', split='train')
    with pytest.raises(ValueError, match='pcm'):
        m.audit_split_rows([a, {**a, 'id': 'b', 'speaker': 'b', 'pair': 'b', 'split': 'test'}])
    with pytest.raises(ValueError, match='speaker'):
        m.audit_split_rows([a, {**a, 'id': 'b', 'pair': 'b', 'pcm_sha256': 'b', 'split': 'dev'}])


def test_one_pcm_cannot_carry_conflicting_training_commands():
    m = module()
    assert hasattr(m, 'audit_annotations'), 'Duplicate label audit is missing'
    a = dict(id='a', speaker='same', pair='same', pcm_sha256='same', split='train',
             reference_status='supported', expected=command('TIME'))
    with pytest.raises(ValueError, match='Conflicting'):
        m.audit_annotations([a, {**a, 'id': 'b', 'expected': command('WEATHER')}])


@pytest.mark.parametrize('split', ['test', 'holdout'])
def test_published_evaluation_membership_is_immutable(split):
    assert module().assign_split(row(published_split=split)) == split
