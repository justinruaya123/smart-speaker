"""Render local OS speech to audio; playback uses the explicitly selected speaker."""
from pathlib import Path
import platform
import shutil
import subprocess


def render(text, folder):
    if not isinstance(text, str) or not text.strip() or len(text) > 1000:
        raise ValueError('Speech requires 1–1000 characters')
    folder = Path(folder)
    system = platform.system()
    if system == 'Darwin':
        target = folder/'speech.aiff'
        command = ['/usr/bin/say', '-o', str(target), '--', text]
        kwargs = {}
    elif system == 'Windows':
        target = folder/'speech.wav'
        # Text arrives on stdin, never interpolated into executable code.
        script = ('Add-Type -AssemblyName System.Speech; '
                  '$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                  '$voice.SetOutputToWaveFile("speech.wav"); '
                  '$voice.Speak([Console]::In.ReadToEnd()); $voice.Dispose()')
        command = ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script]
        kwargs = {'input': text, 'text': True}
    else:
        executable = shutil.which('espeak-ng') or shutil.which('espeak')
        if not executable:
            raise RuntimeError('Install espeak-ng for local spoken responses on Linux / Raspberry Pi OS.')
        target = folder/'speech.wav'
        command = [executable, '-w', str(target), '--stdin']
        kwargs = {'input': text, 'text': True}
    subprocess.run(command, cwd=folder, check=True, capture_output=True, timeout=40, **kwargs)
    if not target.is_file():
        raise RuntimeError('Local TTS did not produce audio')
    return target
