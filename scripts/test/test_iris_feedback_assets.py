"""Build-time cues and committed exact inventory; never synthesize at inference."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
CUES = ('wake_yes','wake_hmm','wake_yeah','wake_here','intent','success','failure')


def test_low_amplitude_beep_design_is_distinct_and_click_free():
    from scripts.build_iris_feedback import tone_pcm
    lengths = {'intent':int(.08*16000),'success':int((.08*2+.03)*16000),'failure':int((.1*2+.03)*16000)}
    waveforms = {}
    for cue, length in lengths.items():
        pcm = np.asarray(tone_pcm(cue),dtype='<i2')
        assert len(pcm) == length
        assert pcm[0] == 0 and pcm[-1] == 0
        assert 0 < np.max(np.abs(pcm.astype('int32'))) < 5000
        waveforms[cue] = hashlib.sha256(pcm.tobytes()).hexdigest()
    assert len(set(waveforms.values())) == 3
    with pytest.raises(ValueError):tone_pcm('any-text')


def test_builder_preserves_existing_output(tmp_path):
    from scripts.build_iris_feedback import build_feedback
    with pytest.raises(FileExistsError):
        build_feedback(tmp_path)


def test_shipped_inventory_pcm_and_file_hashes(tmp_path):
    folder = ROOT/'submission/feedback'
    manifest = json.loads((folder/'manifest.json').read_bytes())
    assert set(manifest['cues']) == set(CUES)
    assert sorted(p.name for p in folder.iterdir()) == sorted(['manifest.json',*[c+ext for c in CUES for ext in ('.wav','.mp3')]])
    decoded_hashes = []
    texts = dict(wake_yes='Yes',wake_hmm='Hmm',wake_yeah='Yeah',wake_here='Here')
    for cue, record in manifest['cues'].items():
        for kind in ('mp3','wav'):
            path = folder/record[kind]['path']
            assert path.name == cue+'.'+kind
            assert path.stat().st_size == record[kind]['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record[kind]['sha256']
        pcm, rate = sf.read(folder/record['wav']['path'],dtype='float32')
        assert rate == 16000 and pcm.ndim == 1 and np.isfinite(pcm).all()
        assert .05 <= len(pcm)/rate <= 2.5
        assert np.max(np.abs(pcm)) < .7
        assert len(pcm) == record['samples']
        assert len(pcm)/rate == record['duration_seconds']
        assert record['generator']['name'] == ('edge-tts' if cue in texts else 'iris_tone')
        if cue in texts:
            assert record['text'] == texts[cue]
            assert record['generator']['version'] == '7.3.1'
            assert record['voice'] == 'en-US-AriaNeural'
        decoded_hashes.append(hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest())
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            decoded=tmp_path/(cue+'.wav')
            subprocess.run([ffmpeg,'-nostdin','-v','error','-i',str(folder/record['mp3']['path']),
                '-ac','1','-ar','16000','-c:a','pcm_s16le',str(decoded)],check=True,capture_output=True)
            actual,_ = sf.read(decoded,dtype='float32')
            np.testing.assert_array_equal(actual,pcm)
    assert len(set(decoded_hashes)) == 7
