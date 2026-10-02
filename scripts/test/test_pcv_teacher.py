"""Guard the source boundary and time alignment of the offline teacher."""
import numpy as np
import pytest
import torch


def test_teacher_frames_share_student_window_centres():
    from scripts.train.pcv_teacher_data import teacher_frames, student_frames
    for samples in (400, 560, 720, 16000, 57600, 192000):
        assert teacher_frames(samples) == (student_frames(samples) + 1) // 2
    assert teacher_frames(399) == 0


def test_teacher_rejects_reserved_or_external_training_audio():
    from scripts.train.pcv_teacher_data import check_rows, PARENT_CACHE
    row = {'parquet': str(PARENT_CACHE/'source/data/train-00000-of-00002.parquet'),
           'published_split': 'train', 'split': 'train', 'synthetic': False}
    check_rows([row], fitting=True, human_only=True)
    for change in ({'split': 'test'}, {'published_split': 'holdout'},
                   {'parquet': 'data/tts/unrelated.parquet'}, {'synthetic': True}):
        with pytest.raises(ValueError):
            check_rows([{**row, **change}], fitting=True, human_only=True)


def test_teacher_head_padding_does_not_change_valid_outputs():
    from scripts.models.pcv_teacher import FiniteTeacherHead
    torch.manual_seed(731)
    model = FiniteTeacherHead(width=16, dilations=(1, 2)).eval()
    sample = torch.randn(1, 768, 15)
    with torch.no_grad():
        alone = model(sample)
        padded = model(torch.nn.functional.pad(sample, (0, 19)))
    assert [a.shape[1] for a in alone[:3]] == [40, 19, 2]
    for a, b in zip(alone[:3], padded[:3]):
        torch.testing.assert_close(a, b[:, :, :15], atol=2e-6, rtol=2e-6)


def test_teacher_pcm_is_verified_before_inference():
    import io
    import hashlib
    import soundfile as sf
    from scripts.train.pcv_teacher_data import verified_pcm
    pcm = np.zeros(1600, dtype='float32')
    stream = io.BytesIO()
    sf.write(stream, pcm, 16000, format='WAV', subtype='PCM_16')
    payload = stream.getvalue()
    row = {'id': 'fixture', 'samples': 1600,
           'pcm_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
           'audio_sha256': hashlib.sha256(payload).hexdigest()}
    np.testing.assert_array_equal(verified_pcm(row, payload), pcm)
    with pytest.raises(ValueError):
        verified_pcm({**row, 'pcm_sha256': 'changed'}, payload)
