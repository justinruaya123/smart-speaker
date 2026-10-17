"""Submission worker; the GUI gateway owns explicitly configured actions."""
from __future__ import annotations

import copy
import os

from scripts.submission.common import ROOT,limit_math_threads
from scripts.submission.response_runtime import RESPONSE_TEXTS, ensure_responses, response_id


def service_class(base, *, playback=False, actions=False):
    class SubmissionAudioService(base):
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
