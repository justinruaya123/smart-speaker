import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fitting import fit_normalization, require_training_split


def test_normalization_uses_only_training_frames():
    rows=[dict(split='train',recording_id='a'),dict(split='dev',recording_id='b')]
    arrays={'a':np.full((40,3),2,dtype=np.float32),'b':np.full((40,5),900,dtype=np.float32)}
    norm=fit_normalization(rows,arrays)
    assert norm['mean']==[2.0]*40
    assert norm['fit_split']=='train'


def test_training_cannot_load_final_or_calibration_audio():
    with pytest.raises(ValueError):require_training_split([dict(split='test')])
    with pytest.raises(ValueError):require_training_split([dict(split='calibration')])
    require_training_split([dict(split='train'),dict(split='dev')])
