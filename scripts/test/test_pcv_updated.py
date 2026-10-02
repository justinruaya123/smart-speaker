"""Protect source-only supervision, exact slots and held-out membership."""
import pytest
from scripts.runtime.pcv_schema import baseline_cases, canonical
from scripts.train.pcv_updated_prepare import annotate, spoken, assign_split, pair_identity


def example(text, intent, slot='', **extra):
    return dict(transcript=text, command=intent, slot_value=slot, published_split='train',
                source='real_voice', speaker_id='fixture', is_synthetic=0,
                transcript_source='prompt', whisper_check='', file='fixture.wav',
                out_of_scope=0, waveform_flags=[], **extra)


def test_all_baseline_targets_preserved():
    for case in baseline_cases():
        result = annotate(example(case['text'], case['command']['intent']))
        assert canonical(result['expected']) == canonical(case['command']), case['text']


def test_explicit_arguments_are_never_defaulted_away():
    assert annotate(example('Call Mom', 'CALL'))['expected']['slots']['contact_id'] == 'contact.mom'
    for text, intent in [('Call Jane', 'CALL'), ('Text Jane', 'MESSAGE'),
                         ('play music by Queen', 'PLAY_MUSIC'), ('turn on the attic light', 'LIGHT_ON')]:
        result = annotate(example(text, intent))
        assert result['expected'] is None
        assert result['reference_status'] == 'unsupported'


def test_compound_duration_overrides_lossy_variation_metadata():
    result = annotate(example('start a timer for 1 hour and 1 minute', 'TIMER', '1 minute'))
    assert result['expected']['slots']['duration_seconds'] == 3660
    assert result['slot_metadata_conflict']


def test_ambiguous_or_relative_brightness_not_absolute():
    assert annotate(example('increase brightness by twenty percent', 'BRIGHTNESS', '20 percent'))['expected'] is None
    assert annotate(example('Green lights to 20%.', 'BRIGHTNESS', '20 percent'))['reference_status'] == 'ambiguous'
    result = annotate(example('Set brightness of kitchen lamp to 60 percent', 'BRIGHTNESS'))
    assert result['expected']['slots'] == {'light_id':'light.kitchen', 'percent':60}


def test_unsupported_multi_action_and_wrong_intent_not_positive():
    assert annotate(example('turn off one light on porch', 'LIGHT_ON'))['expected'] is None
    assert annotate(example('confirm I have set an alarm for six am', 'ALARM'))['expected'] is None
    assert annotate(example('turn volume down and tell me the title', 'VOLUME_DOWN'))['expected'] is None


def test_times_and_percent_have_spoken_supervision():
    assert spoken('Alarm 6:00 AM') == 'alarm six a m'
    assert spoken('Dim lights to 20%.') == 'dim lights to twenty percent'
    assert spoken('Set an alarm at 9pm') == 'set an alarm at nine p m'


@pytest.mark.parametrize('split', ['test','holdout'])
def test_official_human_test_and_holdout_membership_never_reassigned(split):
    row=example('Call', 'CALL'); row['published_split']=split
    assert assign_split(row)==split


def test_paired_capture_variants_stay_together():
    a=example('Call','CALL'); a.update(source='SNIPS',file='audio/train_000001_snips_smart-speaker_3924_close.wav')
    b={**a,'file':'audio/test_999999_snips_smart-speaker_3924_far.wav'}
    assert pair_identity(a)==pair_identity(b)


def test_source_guard_rejects_external_audio_and_cross_split_training():
    from scripts.train.pcv_updated_training import validate_training_rows
    valid={'id':'train/x/1','published_split':'train','split':'train','synthetic':False,
           'parquet':'.cache/pcv-updated-voice-v1/source/data/train-00000-of-00002.parquet'}
    validate_training_rows([valid], human_only=True)
    with pytest.raises(ValueError):validate_training_rows([{**valid,'published_split':'test'}],human_only=True)
    with pytest.raises(ValueError):validate_training_rows([{**valid,'synthetic':True}],human_only=True)
    with pytest.raises(ValueError):validate_training_rows([{**valid,'parquet':'.cache/other-study/audio.parquet'}],human_only=False)


def test_balanced_sampling_does_not_let_numeral_pool_dominate():
    import numpy as np
    from scripts.train.pcv_updated_training import BalancedRows
    rows=[{'intent':'CALL','id':'one'}]+[{'intent':'TIME','id':str(i)} for i in range(100)]
    selected=BalancedRows(rows,np.random.default_rng(22)).take(20)
    assert sum(r['intent']=='CALL' for r in selected)==10
