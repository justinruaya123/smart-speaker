import pytest
from scripts.train.pcv_release_training import validate_rows,valid_run_id,completed_stage,select_arm


def row(**changes):
    return dict({'parquet':'.cache/source/train.parquet','published_split':'train','split':'train',
                 'synthetic':False,'training_eligible':True},**changes)


def test_source_and_split_guards():
    allowed={'.cache/source/train.parquet'}
    validate_rows([row()],allowed,fitting=True,human_only=True)
    for bad in (row(parquet='elsewhere.wav'),row(split='test'),row(published_split='holdout'),row(synthetic=True)):
        with pytest.raises(ValueError):validate_rows([bad],allowed,fitting=True,human_only=True)
    validate_rows([row(split='dev')],allowed)
    with pytest.raises(ValueError):validate_rows([row(split='calibration')],allowed)


def test_run_id_cannot_escape_workspace():
    assert valid_run_id('release-v1')=='release-v1'
    for bad in ('../old','A/B','', 'cafe/..'):
        with pytest.raises(ValueError):valid_run_id(bad)


def test_incomplete_training_is_not_claimed_completed(tmp_path):
    assert completed_stage(tmp_path/'missing',4) is None
    (tmp_path/'human').mkdir()
    with pytest.raises(ValueError,match='incomplete'):
        completed_stage(tmp_path/'human',4)


def test_matched_arm_tie_retains_hard_even_if_teacher_selected_earlier_epoch():
    assert select_arm({'hard':{'key':[.94,-5,-40]},'distilled':{'key':[.94,-5,-4]}})=='hard'
