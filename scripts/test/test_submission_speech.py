"""Tool responses use local speech, never a shell command or a recognition clip."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.submission import speech, worker


class SpeechTests(unittest.TestCase):
    def test_platform_renderers_keep_text_out_of_executable_code(self):
        text = 'It is 7:13 PM. $(do not execute)'
        for system, filename in [('Darwin', 'speech.aiff'), ('Windows', 'speech.wav'), ('Linux', 'speech.wav')]:
            with self.subTest(system=system), tempfile.TemporaryDirectory() as folder:
                def run(command, **kwargs):
                    self.assertNotIn('shell', kwargs)
                    if system == 'Darwin':
                        self.assertEqual(command[-1], text)
                    else:
                        self.assertEqual(kwargs['input'], text)
                        self.assertNotIn(text, ' '.join(command))
                    (Path(folder)/filename).write_bytes(b'audio fixture')
                with patch.object(speech.platform, 'system', return_value=system), patch.object(speech.shutil, 'which', return_value='/usr/bin/espeak-ng'), patch.object(speech.subprocess, 'run', side_effect=run):
                    self.assertEqual(speech.render(text, folder), Path(folder)/filename)

    def test_silent_replay_does_not_render_or_play_speech(self):
        class Base:
            source='wav';stream=Mock()
        service=worker.service_class(Base,playback=True)()
        with patch.object(speech,'render',side_effect=AssertionError('silent replay')):
            self.assertEqual(service.speak('The time is 7 PM.'),{'spoken':False})

    def test_actual_answer_pauses_capture_and_resets_wake_even_on_speech_failure(self):
        class Base:
            source='microphone'
            state='wake_listening'
            def __init__(self):self.stream=Mock();self.gate=Mock();self.set_state=Mock();self.emit=Mock()
        service=worker.service_class(Base,playback=True)()
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true','SUBMISSION_SPEAKER_INDEX':'3'}), patch.dict('sys.modules',{'sounddevice':SimpleNamespace(),'soundfile':SimpleNamespace()}), patch.object(speech,'render',side_effect=RuntimeError('TTS unavailable')):
            with self.assertRaisesRegex(RuntimeError,'TTS unavailable'):service.speak('Actual weather answer')
        service.stream.stop.assert_called_once();service.stream.start.assert_called_once();service.gate.reset.assert_called_once()

    def test_tool_answer_uses_selected_speaker(self):
        class Base:
            source='microphone'
            state='wake_listening'
            def __init__(self):self.stream=Mock();self.gate=Mock();self.set_state=Mock();self.emit=Mock()
        service=worker.service_class(Base,playback=True)();play=Mock()
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true','SUBMISSION_SPEAKER_INDEX':'7'}), patch.dict('sys.modules',{'sounddevice':SimpleNamespace(play=play),'soundfile':SimpleNamespace(read=Mock(return_value=([.1,.2],22050)))}), patch.object(speech,'render',return_value=Path('speech.wav')):
            self.assertEqual(service.speak('The time is 7:13 PM in Manila.'),{'spoken':True})
        play.assert_called_once_with([.1,.2],22050,device=7,blocking=True)
        self.assertTrue(service.emit.call_args.args[0]['synthesized_at_runtime'])

    def test_accepted_commands_skip_recognition_only_clips(self):
        class Base:
            def forward(self,event):pass
        service=worker.service_class(Base,playback=True)()
        with patch.object(worker,'response_id',side_effect=AssertionError('generic acknowledgment')):
            service.forward({'output':{'accepted':True,'command':{'intent':'TIME','slots':{}}}})

    def test_muted_microphone_does_not_disable_spoken_alerts_or_restart_capture(self):
        class Base:
            source='microphone';state='muted';stream=None;gate=None
            def __init__(self):self.set_state=Mock();self.emit=Mock()
        service=worker.service_class(Base,playback=True)();play=Mock()
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true','SUBMISSION_SPEAKER_INDEX':'7'}), patch.dict('sys.modules',{'sounddevice':SimpleNamespace(play=play),'soundfile':SimpleNamespace(read=Mock(return_value=([.1],22050)))}), patch.object(speech,'render',return_value=Path('speech.wav')):
            self.assertEqual(service.speak('Your timer is finished.'),{'spoken':True})
        service.set_state.assert_called_with('muted');self.assertIsNone(service.stream);play.assert_called_once()

    def test_spoken_tool_result_does_not_reset_next_audio_sequence_twice(self):
        class Base:
            source='microphone';state='wake_listening'
            def __init__(self):
                self.stream=Mock();self.gate=Mock();self.set_state=Mock();self.emit=Mock();self.offset=999;self.sequences=[]
            def tick(self):self.sequences.append(self.offset);self.offset+=512
        service=worker.service_class(Base,playback=True)()
        with patch.dict('os.environ',{'SUBMISSION_RESPONSE_PLAYBACK':'true','SUBMISSION_SPEAKER_INDEX':'7'}), patch.dict('sys.modules',{'sounddevice':SimpleNamespace(play=Mock()),'soundfile':SimpleNamespace(read=Mock(return_value=([.1],22050)))}), patch.object(speech,'render',return_value=Path('speech.wav')):
            service.speak('Actual answer');service.tick();service.tick()
        self.assertEqual(service.sequences,[0,512])


if __name__=='__main__':unittest.main()
