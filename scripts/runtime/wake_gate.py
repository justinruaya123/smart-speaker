"""One wake arms one finite VCM command; capture failures always re-arm."""
from collections import deque
import math
import numpy as np
from einops import pack


class WakeGate:
    def __init__(self,command,wake,voice,*,threshold=.5,release_ms=320,
                 release_timeout_seconds=3.,command_timeout_seconds=5.,capture_timeout_seconds=8.):
        if (not 0<threshold<=1 or not 0<release_ms<release_timeout_seconds*1000 or
                command_timeout_seconds<=0 or capture_timeout_seconds<=0):
            raise ValueError('Invalid wake gate settings')
        self.command,self.wake,self.voice=command,wake,voice
        self.threshold=threshold;self.release_samples=round(16*release_ms)
        self.release_timeout=round(16000*release_timeout_seconds)
        self.command_timeout=round(16000*command_timeout_seconds)
        self.capture_timeout=round(16000*capture_timeout_seconds)
        self.reset()

    def _rearm(self):
        self.state='wake';self.elapsed=self.silence=0;self.started=False;self.command_sequence=0
        self.recent=deque(maxlen=10);self.preroll=deque(maxlen=4)
        self.command.reset();self.wake.reset();self.voice.reset()
        callback=getattr(self,'on_reset',None)
        if callback is not None:callback()

    def reset(self):
        self.pending=np.empty(0,dtype=np.float32);self.next_sequence=0
        self._rearm()

    @staticmethod
    def reject(reason,**extra):
        return dict(schema_version=2,stage='wake',accepted=False,command=None,reason=reason,**extra)

    def feed(self,pcm,sequence,*,overflow=False):
        x=np.asarray(pcm,dtype=np.float32)
        if overflow or type(sequence) is not int or sequence!=self.next_sequence or x.ndim!=1 or not np.isfinite(x).all():
            self._rearm();self.pending=np.empty(0,dtype=np.float32)
            if type(sequence) is int and x.ndim==1:self.next_sequence=sequence+len(x)
            return [self.reject('capture_fault')]
        self.next_sequence+=len(x)
        self.pending,_=pack([self.pending,x],'*');events=[]
        while len(self.pending)>=512:
            frame,self.pending=self.pending[:512],self.pending[512:]
            probability=float(self.voice(frame))
            if not math.isfinite(probability) or not 0<=probability<=1:
                self._rearm();events.append(self.reject('capture_fault'));continue
            if self.state=='wake':
                self.recent.append(probability);score=self.wake.push(frame)
                if score is not None and (not math.isfinite(score) or not 0<=score<=1):
                    self._rearm();events.append(self.reject('capture_fault'));continue
                if score is not None and score>=self.threshold and max(self.recent)>=.5:
                    self.state='release';self.elapsed=self.silence=0
                    callback=getattr(self,'on_wake',None)
                    if callback is not None:
                        callback(np.asarray(getattr(self.wake,'audio',[]),dtype='float32').copy())
                    events.append(dict(stage='wake',status='detected',wake_phrase='hey jarvis',score=score))
            elif self.state=='release':
                self.elapsed+=512;self.silence=self.silence+512 if probability<.35 else 0
                if self.silence>=self.release_samples:
                    self.state='command';self.elapsed=0;self.preroll.clear()
                    events.append(dict(stage='wake',status='command_ready'))
                elif self.elapsed>=self.release_timeout:
                    self._rearm();events.append(self.reject('wake_word_not_separated'))
            else:
                self.elapsed+=512
                if self.started and self.command_sequence>=self.capture_timeout:
                    self._rearm();events.append(self.reject('command_capture_timeout'));continue
                if not self.started:
                    self.preroll.append(frame.copy())
                    if probability<.5:
                        if self.elapsed>=self.command_timeout:
                            self._rearm();events.append(self.reject('command_start_timeout'))
                        continue
                    self.started=True;blocks=list(self.preroll);self.preroll.clear()
                else:blocks=[frame]
                for block in blocks:
                    output=self.command.feed(block,self.command_sequence);self.command_sequence+=len(block)
                    if self.command.finalized or output.get('reason')=='capture_fault':
                        events.append({**output,'stage':'command','wake_required':True,'wake_phrase':'hey jarvis'})
                        self._rearm();break
        return events

    def eof(self):
        return self.reject('wake_required' if self.state=='wake' else 'incomplete')


def iter_replay_pcm(gate,pcm):
    """Deliver bounded file warmup/flush while forwarding events immediately."""
    pcm=np.asarray(pcm,dtype=np.float32)
    if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Requires finite mono PCM')
    delivered,_=pack([np.zeros(32000,dtype=np.float32),pcm,np.zeros(96000,dtype=np.float32)],'*')
    gate.reset()
    for offset in range(0,len(delivered),640):
        yield from gate.feed(delivered[offset:offset+640],offset)


def replay_pcm(gate,pcm):
    """File replay uses the same detector warmup and timeout flush as acceptance."""
    return list(iter_replay_pcm(gate,pcm))
