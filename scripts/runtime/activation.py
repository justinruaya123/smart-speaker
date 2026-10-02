"""Wake once, endpoint one command, then re-arm. All clocks use audio samples."""
from collections import deque
import numpy as np
from einops import pack

from .wake_models import HeyJarvis, SileroVAD


class WakeCommandStream:
    def __init__(self, pipeline, wake=None, vad=None):
        self.pipeline = pipeline
        self.settings = pipeline.config['wake']
        self.enabled = self.settings['enabled']
        self.wake = wake if wake is not None else HeyJarvis(pipeline.bundle) if self.enabled else None
        self.vad = vad if vad is not None else SileroVAD(pipeline.bundle, pipeline.config['audio'])
        self.pending = np.empty(0, dtype=np.float32)
        self.reset()

    def reset(self):
        self.pending = np.empty(0, dtype=np.float32)
        self.state = 'wake' if self.enabled else 'command'
        self.wait_samples = self.release_silence = self.cooldown = 0
        self.recent_speech = deque(maxlen=10)
        self.vad.reset()
        if self.wake:
            self.wake.reset()

    def _rearm(self):
        pending = self.pending
        self.reset()
        self.pending = pending
        self.cooldown = round(16 * self.settings['cooldown_ms']) if self.enabled else 0

    def ready(self):
        return {'stage': 'wake' if self.enabled else 'vad', 'status': 'listening',
                'wake_phrase': 'hey jarvis' if self.enabled else None,
                'message': 'Say Hey Jarvis, pause, then speak after command_ready.' if self.enabled else 'Speak a command.'}

    def push(self, pcm):
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Live input must be finite mono 16 kHz PCM')
        self.pending, _ = pack([self.pending, pcm], '*')
        events = []
        while len(self.pending) >= 512:
            frame, self.pending = self.pending[:512], self.pending[512:]
            if self.state == 'wake':
                probability = self.vad.model(frame)
                self.vad.probability = probability
                self.recent_speech.append(probability)
                score = self.wake.push(frame)
                self.cooldown = max(0, self.cooldown - 512)
                if (not self.cooldown and score is not None and score >= self.settings['threshold']
                        and max(self.recent_speech) >= self.vad.threshold):
                    self.state = 'wake_release'
                    self.wait_samples = self.release_silence = 0
                    events.append({'stage': 'wake', 'status': 'detected', 'wake_phrase': 'hey jarvis',
                                   'confidence': score, 'message': 'Pause briefly before the command.'})
            elif self.state == 'wake_release':
                probability = self.vad.model(frame)
                self.vad.probability = probability
                self.wait_samples += 512
                self.release_silence = self.release_silence + 512 if probability < self.vad.threshold - .15 else 0
                if self.release_silence >= 16 * self.settings['release_silence_ms']:
                    self.vad.reset()
                    self.state, self.wait_samples = 'command', 0
                    events.append({'stage': 'wake', 'status': 'command_ready'})
                elif self.wait_samples >= 16000 * self.settings['release_timeout_seconds']:
                    events.append({'stage': 'wake', 'status': 'rejected', 'reason': 'wake_word_not_separated'})
                    self._rearm()
            else:
                self.wait_samples += 512
                was_active = self.vad.active
                endpoint = self.vad.push(frame)
                if not was_active and self.vad.active:
                    events.append({'stage': 'vad', 'status': 'speech_started'})
                if endpoint is not None:
                    audio, timeout = endpoint
                    outcome = self.pipeline.utterance(audio, timeout)
                    events.append({**outcome, 'wake_required': self.enabled, 'vad': self.vad.last_endpoint})
                    self._rearm()
                    events.append(self.ready())
                elif (self.enabled and not self.vad.active
                      and self.wait_samples >= 16000 * self.settings['command_timeout_seconds']):
                    events.append({'stage': 'wake', 'status': 'rejected', 'reason': 'command_start_timeout'})
                    self._rearm()
                    events.append(self.ready())
        return events

    def replay(self, pcm):
        self.reset()
        events = [self.ready()]
        for offset in range(0, len(pcm), 640):
            events.extend(self.push(pcm[offset:offset + 640]))
        # Flush endpoint silence, not an unbounded wait for an absent command.
        events.extend(self.push(np.zeros(16000, dtype=np.float32)))
        return events
