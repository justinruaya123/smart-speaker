"""CLI diagnostics or local SpeakerUI using the selected model's demo policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tarfile
import urllib.request
import webbrowser
import zipfile

from scripts.submission.common import ROOT, fetch, reexec, safe_name, limit_math_threads
from scripts.submission.response_runtime import ensure_responses


HA_ENTITY_IDS = ('light.hue_color_lamp_1,light.dining_light,'
                 'camera.dining_area_main_camera_tc70_mainstream')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('cli', 'ui'), help='1: text diagnostics; 2: local website and recorded replies')
    p.add_argument('--model', type=Path, default=ROOT/'submission/models/full')
    p.add_argument('--mic', type=int, help='Actual host microphone index; listed by --list-devices')
    p.add_argument('--speaker', type=int, help='Actual host output index; CLI selects it without playing audio')
    p.add_argument('--list-devices', action='store_true', help='List host audio inputs and outputs, then exit')
    p.add_argument('--wav', type=Path, help='Replay mono 16 kHz command WAV silently in CLI mode')
    p.add_argument('--wake-wav', action='store_true', help='Replay a WAV through pretrained wake/VAD')
    p.add_argument('--check', action='store_true', help='Verify and load selected model without using audio hardware')
    p.add_argument('--threads', type=int, choices=range(1, 9), default=1)
    p.add_argument('--port', type=int, default=3000)
    p.add_argument('--rebuild-ui', action='store_true', help='Rebuild the local UI from its locked sources')
    p.add_argument('--no-audio', action='store_true', help='UI replay only: no host capture or recorded playback')
    p.add_argument('--no-browser', action='store_true', help='Print the local website URL without opening a browser')
    p.add_argument('--no-setup', action='store_true', help='Use the existing root environment')
    return p


def debug_event(event):
    record = event.get('diagnostic_candidate') or {}
    result = event.get('output') or {}
    command = result.get('command') if result.get('accepted') else None
    return {
        'event': event.get('event', 'command'), 'candidate': record.get('candidate'),
        'accepted_intent': command.get('intent') if command else None,
        'slots': command.get('slots', {}) if command else {},
        'intent_confidence': record.get('intent_probability'),
        'slot_confidence': record.get('slot_confidence', {}),
        'target_named_confidence': record.get('target_named_probability'),
        'accepted': bool(result.get('accepted')), 'reason': result.get('reason'),
        'application': event.get('application'),
        'qualified_accepted': bool((event.get('qualified_output') or {}).get('accepted')),
        'decision_mode': event.get('decision_mode', 'demo'), 'actions_enabled': False,
        'timing': event.get('timing'),
    }


def load_runtime(model, threads):
    from scripts.submission.model import load_runtime as selected_runtime
    return selected_runtime(model, diagnostic=True, threads=threads, engine='onnxruntime', mode='demo')


def read_waveform(path):
    from scripts.runtime.pcv_final_runtime import read_wav
    return read_wav(path)


def audio_devices():
    try:
        import sounddevice as sd
        hosts = sd.query_hostapis()
        return [{'index': i, 'name': d['name'], 'hostapi': hosts[d['hostapi']]['name'],
                 'inputs': d['max_input_channels'], 'outputs': d['max_output_channels'],
                 'default_sample_rate': d['default_samplerate']}
                for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] or d['max_output_channels']]
    except (OSError, RuntimeError) as error:
        raise RuntimeError('Audio discovery failed. Install the OS PortAudio library and connect audio devices: '
                           + str(error)) from error


def select_device(devices, direction, index=None):
    field = {'microphone': 'inputs', 'speaker': 'outputs'}[direction]
    available = [d for d in devices if d[field] > 0]
    if not available:
        raise ValueError('No ' + direction + ' devices on this host. Use --wav or UI --no-audio for replay.')
    if index is None:
        print('\n' + direction.capitalize() + ' devices:')
        for d in available:
            print(f"  [{d['index']}] {d['name']} ({d.get('hostapi', 'audio')})")
        index = int(input('Choose the ' + direction + ' index: ').strip())
    selected = next((d for d in available if d['index'] == index), None)
    if selected is None:
        raise ValueError('Choose a listed ' + direction + ' index: ' + str(index))
    return selected


def ui_environment(root, model, threads, port, microphone, speaker):
    from scripts.runtime.speaker_config import resolve_config
    # Reuse the private root .env/config loader. The standalone profile defaults
    # actions off, while explicit settings continue to enable the user's GUI.
    local = {'HOST':'127.0.0.1', 'PORT':str(port), 'APP_ORIGIN':f'http://127.0.0.1:{port}',
             'INFERENCE_ENGINE':'onnxruntime', 'EXECUTION_MODE':'demo'}
    _, env = resolve_config(root, profile='standalone', environ={**os.environ, **local})
    env.update({
        'INFERENCE_MODE': 'onnx', 'INFERENCE_ENGINE': 'onnxruntime', 'EXECUTION_MODE': 'demo',
        'SMART_SPEAKER_PYTHON': sys.executable, 'SMART_SPEAKER_ROOT': str(Path(root).resolve()),
        'SMART_SPEAKER_BUNDLE': str(Path(model).resolve()), 'SMART_SPEAKER_THREADS': str(threads),
        'SMART_SPEAKER_TOKEN': secrets.token_urlsafe(32),
        'SMART_SPEAKER_WORKER_MODULE': 'scripts.submission.worker',
        **local, 'HA_ENTITY_IDS': env.get('HA_ENTITY_IDS') or HA_ENTITY_IDS,
        'SERVE_UI': 'true', 'SUBMISSION_RESPONSE_PLAYBACK': 'true' if speaker is not None else 'false',
        'SUBMISSION_MICROPHONE_INDEX': '' if microphone is None else str(microphone),
        'SUBMISSION_SPEAKER_INDEX': '' if speaker is None else str(speaker),
    })
    # This launcher serves loopback HTTP; provider credentials stay server-side.
    env.pop('TLS_CERT_PATH', None); env.pop('TLS_KEY_PATH', None)
    return env


def runtime_node(root):
    """Production exports need Node alone; pnpm/build tools stay on the source host."""
    from scripts.runtime.speaker_tools import NODE_VERSION
    root=Path(root)
    system={'Windows':'win','Linux':'linux','Darwin':'darwin'}.get(platform.system())
    arch={'amd64':'x64','x86_64':'x64','aarch64':'arm64','arm64':'arm64'}.get(platform.machine().lower())
    if not system or not arch:
        raise RuntimeError('Use a supported 64-bit Windows, macOS or Linux host; Pi requires 64-bit OS')
    tools=root/'.runtime/tools';tools.mkdir(parents=True,exist_ok=True)
    name=f'node-v{NODE_VERSION}-{system}-{arch}'
    binary=tools/name/('node.exe' if system=='win' else 'bin/node')
    if not binary.exists():
        filename=name+('.zip' if system=='win' else '.tar.gz')
        base=f'https://nodejs.org/dist/v{NODE_VERSION}/'
        with urllib.request.urlopen(base+'SHASUMS256.txt',timeout=60) as response:
            checks={line.split()[1].lstrip('*'):line.split()[0] for line in response.read().decode().splitlines()}
        archive=fetch(base+filename,root/'.cache/tools'/filename,checks[filename])
        if system=='win':
            with zipfile.ZipFile(archive) as packed:
                if not all(safe_name(member) for member in packed.namelist()):
                    raise ValueError('Unsafe Node archive member')
                packed.extractall(tools)
        else:
            with tarfile.open(archive) as packed:
                packed.extractall(tools,filter='data')
    env=os.environ.copy();env['PATH']=str(binary.parent)+os.pathsep+env.get('PATH','')
    return binary,env


def prepare_frontend(root, rebuild=False):
    from scripts.runtime.speaker_tools import node_tools, prepare_ui
    ui = Path(root)/'SpeakerUI'
    compiled = (ui/'dist/index.html').exists() and (ui/'dist-server/server/main.js').exists()
    if compiled and (ui/'node_modules/fastify/package.json').exists() and not rebuild:
        return runtime_node(root)
    # pnpm invokes itself from nested package scripts. The canonical helper adds
    # Node to PATH but assumes pnpm was already globally available. Keep its
    # local package executable available only during this source-build call.
    _, pnpm, _ = node_tools(root)
    previous = os.environ.get('PATH', '')
    os.environ['PATH'] = str(pnpm.parents[2]/'.bin') + os.pathsep + previous
    try:
        return prepare_ui(root, build=rebuild or not compiled)
    finally:
        os.environ['PATH'] = previous


def serve(args, microphone, speaker):
    if speaker is not None:
        assets = ensure_responses(ROOT)
        print(f'Verified {len(assets)} recorded recognition replies. Runtime synthesis is not required.')
    node, tool_env = prepare_frontend(ROOT, args.rebuild_ui)
    env = ui_environment(ROOT, args.model, args.threads, args.port, microphone, speaker)
    env['PATH'] = tool_env['PATH']
    url = f'http://127.0.0.1:{args.port}'
    print('Local SpeakerUI:', url, flush=True)
    command = [str(node), 'dist-server/server/main.js']
    process = subprocess.Popen(command, cwd=ROOT/'SpeakerUI', env=env)
    try:
        if not args.no_browser:
            webbrowser.open(url)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError('Port must be 1..65535')
    if args.wake_wav and not args.wav:
        raise ValueError('--wake-wav requires --wav')
    if args.wav and args.mode == 'ui':
        raise ValueError('--wav is the silent CLI replay path; use the UI WAV upload for website replay')
    if not args.no_setup:
        return reexec('inference.py', list(sys.argv[1:] if argv is None else argv), group=None)
    limit_math_threads()
    if args.list_devices:
        print(json.dumps({'devices': audio_devices()}, indent=2)); return 0
    if args.check:
        runtime = load_runtime(args.model, args.threads)
        print(json.dumps({'event':'ready', 'model_sha256':runtime.recipe['binding']['model_sha256'],
                          'parameters':runtime.recipe.get('parameters'), 'engine':runtime.engine,
                          'qualified_policy_enabled':runtime.policy['enabled'], 'decision_mode':'demo',
                          'actions_enabled':False, 'session_load_ms':runtime.load_ms})); return 0
    if not args.mode:
        if args.wav:
            args.mode = 'cli'
        else:
            print('\n[1] CLI: recognized candidate, accepted intent, typed slots and confidence\n'
                  '[2] SpeakerUI: local website and recorded recognition replies')
            selected = input('Choose [1/2]: ').strip()
            if selected not in ('1', '2'):
                raise ValueError('Choose 1 or 2')
            args.mode = {'1':'cli', '2':'ui'}[selected]
    microphone = speaker = None
    if not args.wav and not (args.mode == 'ui' and args.no_audio):
        devices = audio_devices()
        microphone = select_device(devices, 'microphone', args.mic)['index']
        speaker = select_device(devices, 'speaker', args.speaker)['index']
        print(f'Selected microphone {microphone}; speaker {speaker}.')
    if args.mode == 'ui':
        return serve(args, microphone, speaker)
    runtime = load_runtime(args.model, args.threads)
    if args.wav and not args.wake_wav:
        print(json.dumps(debug_event(runtime.recognize(read_waveform(args.wav))), allow_nan=False)); return 0
    from scripts.runtime.speaker_worker import AudioService
    from scripts.submission.worker import service_class
    def emit(event):
        print(json.dumps(debug_event(event) if event.get('event') == 'command' else event,
                         allow_nan=False), flush=True)
    service = service_class(AudioService)(runtime, emit)
    if args.wav:
        service.wav(args.wav.read_bytes(), wake=True); return 0
    print('Say "Hey Jarvis", pause briefly, then say a command. Ctrl+C stops. CLI audio playback is disabled.', flush=True)
    try:
        service.listen(microphone)
        while True:
            service.tick()
    except KeyboardInterrupt:
        return 0
    finally:
        service.stop(announce=False)


if __name__ == '__main__':
    raise SystemExit(main())
