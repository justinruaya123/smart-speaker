"""Closed-ID, predecoded host feedback; no TTS, URL, text or shell execution."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import threading
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MANIFEST_SHA256='4649d644b7a77d5ed75c22de40a20b63059b953e081446815a8a22afe7d22b4a'
WAKE_CUES=('wake_yes','wake_hmm','wake_yeah','wake_here')
CONTROL_CUES=('intent','success','failure')
CUE_IDS=(*WAKE_CUES,*CONTROL_CUES)


class SounddevicePlayback:
    """One private blocking output stream; return only after its buffer drains."""
    def __call__(self,pcm,sample_rate,speaker_index):
        import sounddevice as sd
        stream=sd.OutputStream(device=speaker_index,samplerate=sample_rate,channels=1,dtype='float32')
        try:
            stream.start()
            stream.write(pcm[:,None])
            stream.stop()  # PortAudio stop drains pending output before acknowledging.
        except BaseException:
            stream.abort()
            raise
        finally:
            stream.close()


class FeedbackPlayer:
    """Verify and preload every shipped asset before accepting a control."""
    def __init__(self,folder=ROOT/'submission/feedback',*,speaker_index=None,enabled=False,playback=None):
        if type(enabled) is not bool:
            raise ValueError('Feedback enabled must be boolean')
        if enabled and (type(speaker_index) is not int or not 0<=speaker_index<=65535):
            raise ValueError('Enabled feedback requires an explicit host speaker index')
        self.enabled=enabled;self.speaker_index=speaker_index
        self.playback=playback or SounddevicePlayback();self.lock=threading.Lock()
        folder=Path(folder).resolve()
        path=folder/'manifest.json'
        if path.is_symlink() or path.stat().st_size>65536:
            raise ValueError('Invalid feedback manifest')
        payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('Feedback manifest identity mismatch')
        manifest=json.loads(payload)
        if (manifest.get('schema_version')!=1 or manifest.get('sample_rate')!=16000
                or manifest.get('channels')!=1 or set(manifest.get('cues',{}))!=set(CUE_IDS)):
            raise ValueError('Invalid closed feedback inventory')
        self.assets={};self.manifest=manifest
        for cue in CUE_IDS:
            record=manifest['cues'][cue]
            data={}
            for kind in ('wav','mp3'):
                expected=record[kind];target=folder/(cue+'.'+kind)
                if (expected['path']!=target.name or target.is_symlink()
                        or target.resolve().parent!=folder or not 0<target.stat().st_size<=512000):
                    raise ValueError('Invalid feedback asset path')
                raw=target.read_bytes()
                if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
                    raise ValueError('Feedback asset hash/identity mismatch: '+cue+'.'+kind)
                data[kind]=raw
            with wave.open(io.BytesIO(data['wav']),'rb') as stream:
                if (stream.getnchannels()!=1 or stream.getframerate()!=16000 or stream.getsampwidth()!=2
                        or not 800<=stream.getnframes()<=40000):
                    raise ValueError('Invalid predecoded feedback PCM')
                pcm_bytes=stream.readframes(stream.getnframes())
            if hashlib.sha256(pcm_bytes).hexdigest()!=record['pcm_sha256']:
                raise ValueError('Feedback PCM identity mismatch: '+cue)
            pcm=np.frombuffer(pcm_bytes,dtype='<i2').astype('float32')/32768
            if len(pcm)!=record['samples'] or not np.isfinite(pcm).all() or np.max(np.abs(pcm))>=.7:
                raise ValueError('Invalid feedback waveform bounds')
            pcm.setflags(write=False);self.assets[cue]=pcm

    def play(self,cue_id):
        if not isinstance(cue_id,str) or cue_id not in CUE_IDS:
            raise ValueError('Unknown closed feedback cue ID')
        if not self.enabled:
            return {'cue_id':cue_id,'played':False,'reason':'feedback_disabled'}
        with self.lock:
            self.playback(self.assets[cue_id],16000,self.speaker_index)
        return {'cue_id':cue_id,'played':True,'duration_seconds':len(self.assets[cue_id])/16000}
