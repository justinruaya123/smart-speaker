"""Submission worker; the GUI gateway owns explicitly configured actions."""
from __future__ import annotations

import copy
import os
import tempfile

from scripts.submission.common import ROOT,limit_math_threads
from scripts.submission.response_runtime import RESPONSE_TEXTS, ensure_responses, response_id


def service_class(base, *, playback=False, actions=False):
    class SubmissionAudioService(base):
        def new_gate(self):
            from scripts.submission.wake import make_gate
            service=self
            class Observed:
                root=service.runtime.root
                def recognize(self,pcm):return service.recognize(pcm)
            gate=make_gate(self.runtime,Observed(),getattr(self,'wake_mode',os.environ.get('SUBMISSION_WAKE_MODE','iris_fallback')))
            self.wake_config=gate.wake_config
            self.emit({'event':'wake_config','wake':self.wake_config})
            return gate

        def select_wake(self,mode):
            from scripts.submission.wake import MODES
            if mode not in MODES:raise ValueError('Unknown wake mode')
            previous=getattr(self,'wake_mode',os.environ.get('SUBMISSION_WAKE_MODE','iris_fallback'))
            stream=self.stream;state=self.state
            if stream is not None:stream.stop()
            try:
                self.wake_mode=mode;gate=self.new_gate()
                self.gate=gate;self.offset=0
                return {'wake':self.wake_config}
            except Exception:
                self.wake_mode=previous
                raise
            finally:
                if stream is not None:
                    # Resuming capture always starts a fresh contiguous sequence.
                    if self.gate:self.gate.reset()
                    self.offset=0;stream.start();self.set_state('wake_listening')
                else:self.set_state(state)

        def speak(self, text):
            if (not playback or self.source != 'microphone'
                    or os.environ.get('SUBMISSION_RESPONSE_PLAYBACK') != 'true'):
                return {'spoken':False}
            from scripts.submission.speech import render
            import sounddevice as sd
            import soundfile as sf
            stream = self.stream
            previous_state = self.state
            try:
                if stream is not None:
                    stream.stop()
                self.set_state('responding')
                with tempfile.TemporaryDirectory(prefix='smart-speaker-speech-') as folder:
                    audio = render(text, folder)
                    pcm, rate = sf.read(audio, dtype='float32')
                    self.emit({'event':'response', 'text':text, 'synthesized_at_runtime':True, 'claim':'tool_result'})
                    sd.play(pcm, rate, device=int(os.environ['SUBMISSION_SPEAKER_INDEX']), blocking=True)
                return {'spoken':True}
            finally:
                if self.gate:
                    # Tool-result speech runs in the control loop, outside tick.
                    self.gate.reset(); self.offset=0; self._playback_reset=False
                if stream is not None:
                    stream.start()
                self.set_state('wake_listening' if stream is not None else previous_state)

        def tick(self):
            super().tick()
            if getattr(self, '_playback_reset', False):
                self.offset = 0
                self._playback_reset = False

        def emit(self, value):
            # The shared submission runtime already applied its bound decision;
            # preserve it rather than letting the historical worker recompute it.
            self.sequence += 1
            value = copy.deepcopy(value)
            if value.get('event') == 'ready':
                value['wake']=getattr(self,'wake_config',{'selected':os.environ.get('SUBMISSION_WAKE_MODE','iris_fallback'),'active':'pending','iris_experimental':True})
                if hasattr(self.runtime, 'recall_policy'):
                    value['application_config'] = {
                        'mode':'demo', 'policy':'speaker-compact-recall-v1', 'slot_threshold':.9,
                        'intent_thresholds':self.runtime.recall_policy['intent_thresholds']}
                else:
                    value['application_config'] = {
                        'mode':'demo', 'policy':'submission-fixed-threshold-v1',
                        'intent_threshold':.95, 'slot_threshold':.9}
                value['decision_mode'] = 'demo'
                value['recorded_responses'] = os.environ.get('SUBMISSION_RESPONSE_PLAYBACK') == 'true'
            settings = getattr(self, 'settings', None)
            enabled = bool(actions and settings and settings['actions']['enabled']
                           and self.source == 'microphone')
            self.deliver({**value, 'session_id':self.session, 'sequence':self.sequence,
                          'source':self.source, 'actions_enabled':enabled})
            microphone = os.environ.get('SUBMISSION_MICROPHONE_INDEX')
            if value.get('event') == 'ready' and microphone:
                self.listen(int(microphone))

        def forward(self, event):
            super().forward(event)
            # Accepted commands are answered by the gateway using real tool
            # results. Never race those answers with a recognition-only clip.
            decision = event.get('application') or event.get('output', {})
            if decision.get('accepted') and decision.get('command'):
                return
            if (not playback or 'output' not in event or self.source != 'microphone'
                    or os.environ.get('SUBMISSION_RESPONSE_PLAYBACK') != 'true'):
                return
            key = response_id(event)
            if key is None:
                return
            stream = self.stream
            try:
                if stream is not None:
                    stream.stop()
                clips = ensure_responses(ROOT)
                import sounddevice as sd
                import soundfile as sf
                pcm, rate = sf.read(clips[key], dtype='float32')
                self.set_state('responding')
                self.emit({'event':'response', 'response_id':key, 'text':RESPONSE_TEXTS[key],
                           'synthesized_at_runtime':False, 'claim':'recognition_only'})
                sd.play(pcm, rate, device=int(os.environ['SUBMISSION_SPEAKER_INDEX']), blocking=True)
            except Exception as error:
                self.emit({'event':'capture_warning', 'reason':'response_playback_failed', 'message':str(error)})
            finally:
                if self.gate:
                    self.gate.reset(); self.offset=0; self._playback_reset=True
                if stream is not None:
                    stream.start()
                    self.set_state('wake_listening')
    return SubmissionAudioService


def main():
    # Monkeypatch only this dedicated worker process. Frozen runtime source and
    # qualified policy files remain unchanged.
    limit_math_threads()
    os.environ.pop('PCV_BENCHMARK_LOG', None)
    from scripts.runtime import speaker_worker, speaker_config
    from scripts.submission.model import load_runtime
    original_resolve = speaker_config.resolve_config
    def resolve(*args, **kwargs):
        settings, env = original_resolve(*args, **kwargs)
        settings['inference']['engine'] = 'onnxruntime'
        settings['inference']['mode'] = 'demo'
        env['INFERENCE_ENGINE'] = 'onnxruntime'
        return settings, env
    def runtime(*args, **kwargs):
        kwargs.update(engine='onnxruntime', mode='demo')
        return load_runtime(*args, **kwargs)
    speaker_config.resolve_config = resolve
    speaker_worker.Runtime = runtime
    speaker_worker.AudioService = service_class(speaker_worker.AudioService, playback=True, actions=True)
    speaker_worker.main()


if __name__ == '__main__':
    main()
