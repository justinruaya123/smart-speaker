"""Synthetic completion must not erase human gaps or mix voice partitions."""
import copy
import importlib

import pytest


def adapter():
    name = 'scripts.train.speaker_synthetic_data'
    assert importlib.util.find_spec(name) is not None, 'synthetic completion adapter is missing'
    return importlib.import_module(name)


def voices():
    return [dict(id=name, split=split, prompt='Original designed voice ' + name,
                 design_seed=seed) for seed, (name, split) in enumerate(
                     [('train-1', 'train'), ('dev-1', 'dev'), ('cal-1', 'calibration')], 53)]


def prompts():
    return [dict(id='message', text='Send a message', spoken_text='Send a message',
                 command={'intent': 'MESSAGE', 'slots': {}}, case_ids=['message']),
            dict(id='negative', text='Tell me a joke', spoken_text='Tell me a joke',
                 command=None, case_ids=['negative'])]


def completed(plan):
    return [dict(row, kind='synthetic', qa_eligible=True, audio_flags=[],
                 sha256=f'{i + 1:064x}', pcm_sha256=f'{i + 100:064x}',
                 speaker_id='qwen-completion/' + row['voice_id'],
                 pair_id=row['id'], listener_verified=False)
            for i, row in enumerate(plan)]


def test_color_completion_covers_declared_matrix_instead_of_one_observed_error():
    m = adapter()
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM
    rows = m.build_prompts(prompts(), DEFAULT_CUSTOM)
    colors = {'red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple',
              'pink', 'cyan', 'magenta', 'warm white', 'cool white'}
    pairs = {(r['command']['slots']['light_id'], r['command']['slots']['color'])
             for r in rows if r['command'] and r['command']['intent'] == 'COLOR'}
    assert pairs == {(light, color) for light in (
        'light.all', 'light.kitchen', 'light.bedroom', 'light.bathroom') for color in colors}
    assert any(r['command'] is None for r in rows)


def test_repeated_spoken_prompt_is_generated_once_with_both_case_ids():
    m = adapter()
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM
    cases = prompts()
    cases.append({**cases[0], 'id': 'message-alias', 'text': 'Send a message.'})
    rows = m.build_prompts(cases, DEFAULT_CUSTOM)
    message = [r for r in rows if r['command'] == {'intent': 'MESSAGE', 'slots': {}}]
    assert len(message) == 1
    assert message[0]['case_ids'] == ['message', 'message-alias']


def test_conflicting_prompt_does_not_silently_choose_one_supervision():
    m = adapter()
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM
    cases = prompts()
    cases.append({**cases[0], 'id': 'conflict', 'command': None})
    with pytest.raises(ValueError, match='conflicting'):
        m.build_prompts(cases, DEFAULT_CUSTOM)


def test_every_prompt_has_each_voice_and_fixed_split():
    m = adapter()
    rows = m.build_plan(prompts(), voices())
    assert len(rows) == 6
    assert {(r['voice_id'], r['split']) for r in rows} == {
        ('train-1', 'train'), ('dev-1', 'dev'), ('cal-1', 'calibration')}
    assert all(r['kind'] == 'synthetic' for r in rows)


def test_renaming_the_same_designed_voice_cannot_cross_partitions():
    m = adapter()
    conditions = voices()
    conditions[1].update(prompt=conditions[0]['prompt'], design_seed=conditions[0]['design_seed'])
    with pytest.raises(ValueError, match='voice'):
        m.build_plan(prompts(), conditions)


def test_generation_plan_cannot_add_an_evaluation_split():
    m = adapter()
    conditions = voices()
    conditions[2]['split'] = 'test'
    with pytest.raises(ValueError, match='split'):
        m.build_plan(prompts(), conditions)


def test_generated_records_cannot_be_relabelled_human():
    m = adapter()
    plan = m.build_plan(prompts(), voices())
    rows = completed(plan)
    rows[0]['kind'] = 'human'
    with pytest.raises(ValueError, match='synthetic'):
        m.audit_generated(rows, plan, [])


@pytest.mark.parametrize('change', [dict(qa_eligible=False), dict(audio_flags=['silence']),
                                   dict(split='test'), dict(command={'intent': 'CALL', 'slots': {}})])
def test_failed_or_changed_generation_never_enters_preparation(change):
    m = adapter()
    plan = m.build_plan(prompts(), voices())
    rows = completed(plan)
    rows[0].update(change)
    with pytest.raises(ValueError):
        m.audit_generated(rows, plan, [])


def test_generated_pcm_cannot_duplicate_published_evaluation():
    m = adapter()
    plan = m.build_plan(prompts(), voices())
    rows = completed(plan)
    existing = [dict(id='published-evaluation', split='test',
                     pcm_sha256=rows[0]['pcm_sha256'], sha256='e' * 64,
                     speaker_id='other-human', pair_id='other-pair')]
    with pytest.raises(ValueError, match='overlap|duplicate'):
        m.audit_generated(rows, plan, existing)


def test_missing_and_repeated_generation_cells_are_not_hidden():
    m = adapter()
    plan = m.build_plan(prompts(), voices())
    rows = completed(plan)
    receipt = m.audit_generated(rows[:-1], plan, [])
    assert receipt['complete'] is False
    assert receipt['missing'] == [rows[-1]['id']]
    with pytest.raises(ValueError, match='duplicate'):
        m.audit_generated(rows + [copy.deepcopy(rows[0])], plan, [])


def test_duplicate_actual_anchor_cannot_be_claimed_as_distinct_voices():
    m = adapter()
    anchors = [dict(id=v['id'], split=v['split'], pcm_sha256='a' * 64) for v in voices()]
    with pytest.raises(ValueError, match='anchor'):
        m.audit_anchors(anchors, voices())


def test_whole_waveform_conversion_produces_valid_sixteen_khz_audio():
    import numpy as np
    m = adapter()
    source = .1 * np.sin(2 * np.pi * 400 * np.arange(24000) / 24000)
    values, quality = m.convert_waveform(source, 24000, phone_count=8)
    assert values.shape == (16000,)
    assert quality['sample_rate'] == 16000
    assert quality['ctc_frames'] == 49
    assert quality['rms'] > .05


@pytest.mark.parametrize('kind', ['silence', 'too_short', 'nonfinite'])
def test_bad_generated_audio_fails_waveform_checks(kind):
    import numpy as np
    m = adapter()
    values = {'silence': np.zeros(24000), 'too_short': np.ones(50) * .1,
              'nonfinite': np.full(24000, np.nan)}[kind]
    with pytest.raises(ValueError):
        m.convert_waveform(values, 24000, phone_count=12)


def test_catalog_extension_preserves_old_acoustic_sequences():
    m = adapter()
    from scripts.runtime.pcv_schema import DEFAULT_CUSTOM
    original = {'schema_version': 3, 'entries': [dict(text='send a message',
                command={'intent': 'MESSAGE', 'slots': {}}, phones=[7, 2])]}
    catalog = m.extend_catalog(original, m.build_prompts(prompts(), DEFAULT_CUSTOM))
    assert catalog['entries'][0]['phones'] == [7, 2]
    assert len(original['entries']) == 1
    commands = [r['command'] for r in catalog['entries']]
    assert {'intent': 'COLOR', 'slots': {'light_id': 'light.bathroom', 'color': 'red'}} in commands
    assert catalog['acoustic_qualification'] is False
