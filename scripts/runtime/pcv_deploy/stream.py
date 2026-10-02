"""One active bounded PCM stream, pinned to one validated model generation."""
import base64
import binascii
import re
import time
import numpy as np


class StreamManager:
    def __init__(self,snapshot):
        self.current=snapshot;self.previous=None;self.active=None

    def stage(self,snapshot):
        if not snapshot.get('binding') or 'gate' not in snapshot:raise ValueError('Invalid validated snapshot')
        # Caller validates/loads before this atomic assignment. A live stream
        # continues holding its previous snapshot until end/cancel/disconnect.
        self.previous=self.current;self.current=snapshot

    def rollback(self):
        if self.previous is None:raise ValueError('No previous bundle')
        self.current,self.previous=self.previous,self.current
        return self.current['binding']

    def start(self,stream_id):
        if self.active is not None:raise ValueError('VCM busy: one active stream permitted')
        if not isinstance(stream_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',stream_id):raise ValueError('Invalid stream ID')
        gate=self.current['gate'];gate.reset()
        self.active=dict(id=stream_id,snapshot=self.current,sequence=0)
        return dict(type='started',stream_id=stream_id,binding=self.current['binding'],sample_rate=16000,
            channels=1,encoding='pcm_s16le',maximum_chunk_samples=1600,recommended_chunk_samples=640,
            wake_required=True,startup_warmup_samples=32000)

    def _require(self,stream_id):
        if self.active is None or self.active['id']!=stream_id:raise ValueError('Unknown stream')
        return self.active

    def audio(self,stream_id,message):
        active=self._require(stream_id);gate=active['snapshot']['gate'];start=time.perf_counter()
        try:
            sequence=message.get('sequence');encoded=message.get('pcm_s16le')
            if type(sequence) is not int or sequence!=active['sequence'] or not isinstance(encoded,str) or len(encoded)>4268:
                raise ValueError('Invalid PCM sequence or payload')
            data=base64.b64decode(encoded,validate=True)
            if not 0<len(data)<=3200 or len(data)%2 or message.get('overflow',False) is not False:raise ValueError('Invalid PCM chunk')
            pcm=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768.
            events=gate.feed(pcm,sequence)
            if any(e.get('reason')=='capture_fault' for e in events):raise ValueError('Capture fault')
            active['sequence']+=len(pcm)
        except (ValueError,TypeError,binascii.Error):
            self.cancel(stream_id)
            events=[dict(stage='stream',accepted=False,command=None,reason='capture_fault')]
        return dict(type='audio_result',stream_id=stream_id,binding=active['snapshot']['binding'],
            next_sequence=active['sequence'],events=events,processing_ms=(time.perf_counter()-start)*1000)

    def end(self,stream_id):
        active=self._require(stream_id)
        result=dict(type='ended',stream_id=stream_id,binding=active['snapshot']['binding'],final=active['snapshot']['gate'].eof())
        self.cancel(stream_id);return result

    def cancel(self,stream_id):
        if self.active is not None and self.active['id']==stream_id:
            self.active['snapshot']['gate'].reset();self.active=None
