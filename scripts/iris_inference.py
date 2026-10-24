"""Iris source inference launcher: frozen epoch44, local CPU, no fitting.

Check/WAV/CLI always suppress actions and host feedback, regardless of inherited
flags. UI uses the existing combined gateway and explicit private action settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
import uuid
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = 'c1cd773b92337fd2a453554b047fa555c692fb981b5975700a32260388d4b9ce'
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def make_parser(root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(root)/'submission/models/optionb-tiny')
    parser.add_argument('--check',action='store_true',help='Verify/load VCM, wake/VAD and owner; no capture, actions or playback')
    parser.add_argument('--wav',type=Path,help='Silent action-free command WAV replay')
    parser.add_argument('--wake-wav',action='store_true',help='--wav includes Hey Iris; use actual wake/VAD path')
    parser.add_argument('--wake-root',type=Path,help='Explicit separately evidenced Hey Iris companion (or IRIS_WAKE_ROOT)')
    parser.add_argument('--list-devices',action='store_true',help='List host microphone and speaker indices; no capture')
    parser.add_argument('--mode',choices=['cli','ui'],default='cli')
    parser.add_argument('--mic',type=int,help='Explicit host input index; UI can select one later')
    parser.add_argument('--speaker',type=int,help='Host feedback output index; UI defaults to the host default output')
    parser.add_argument('--threads',type=int,choices=range(1,9),default=1)
    parser.add_argument('--no-audio',action='store_true',help='Suppress feedback AND actions, including inherited enable flags')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--no-setup',action='store_true',help='Use this interpreter and existing pinned Node/compiled UI; never install')
    parser.add_argument('--port',type=int)
    return parser


def resolve_settings(root,args,*,environ=None):
    from scripts.runtime.speaker_config import resolve_config,dotenv
    root = Path(root).resolve()
    actual_env = dict(os.environ if environ is None else environ)
    private_env = {**dotenv(root/'.env'),**actual_env}
    supplied = tomllib.loads((root/'config.toml').read_text(encoding='utf-8-sig')) if (root/'config.toml').exists() else {}
    explicit_actions = 'enabled' in supplied.get('actions',{}) or bool(private_env.get('ACTIONS_ENABLED'))
    settings, env = resolve_config(root,environ=actual_env,overrides={
        'engine':'onnxruntime','threads':args.threads,'port':args.port,'mic':args.mic})
    if not explicit_actions or args.no_audio or args.mode != 'ui' or args.check or args.wav or args.list_devices:
        settings['actions']['enabled'] = False
    env['ACTIONS_ENABLED'] = str(settings['actions']['enabled']).lower()
    return settings,env


def launch_environment(root,python,args,settings,runtime_env,*,token):
    env = {**os.environ,**runtime_env}
    env.pop('PYTHONPATH',None)
    ui = args.mode=='ui' and not (args.check or args.wav or args.list_devices)
    audible = ui and not args.no_audio
    env.update(SMART_SPEAKER_ROOT=str(Path(root).resolve()),SMART_SPEAKER_PYTHON=str(python),
        SMART_SPEAKER_BUNDLE=str(args.bundle.resolve()),SMART_SPEAKER_THREADS=str(args.threads),
        SMART_SPEAKER_TOKEN=token,INFERENCE_MODE='onnx',INFERENCE_ENGINE='onnxruntime',
        SERVE_UI='true' if ui else 'false',IRIS_PUBLICATION='true' if ui else 'false',
        RECOGNITION_PROFILE='owner-protected' if ui or args.check else 'benchmark-compatible',
        IRIS_FEEDBACK_ENABLED='true' if audible else 'false',
        ACTIONS_ENABLED=str(bool(ui and settings['actions']['enabled'] and not args.no_audio)).lower())
    env.pop('IRIS_SPEAKER_INDEX',None)
    if audible and args.speaker is not None:
        env['IRIS_SPEAKER_INDEX']=str(args.speaker)
    if args.wake_root is not None:
        env['IRIS_WAKE_ROOT']=str(args.wake_root.resolve())
    for key in THREAD_KEYS:
        env.setdefault(key,'1')
    return env


def ready_matches(health,token):
    runtime = health.get('runtime',{})
    return bool(health.get('instance')==token[:12] and health.get('inference')=='onnx'
        and runtime.get('loaded') is True and runtime.get('parameters')==258912
        and runtime.get('selected_epoch')==44 and runtime.get('model_sha256')==MODEL_SHA256)


def existing_ui(root):
    """Offline path only: do not call a bootstrap that can install implicitly."""
    from scripts.runtime.speaker_tools import NODE_VERSION
    system = {'Windows':'win','Linux':'linux','Darwin':'darwin'}.get(platform.system())
    arch = {'amd64':'x64','x86_64':'x64','aarch64':'arm64','arm64':'arm64'}.get(platform.machine().lower())
    directory=Path(root)/'.runtime/tools'/f'node-v{NODE_VERSION}-{system}-{arch}'
    node=directory/('node.exe' if system=='win' else 'bin/node')
    ui=Path(root)/'SpeakerUI'
    if not node.is_file() or not (ui/'dist/index.html').is_file() or not (ui/'dist-server/server/main.js').is_file():
        raise RuntimeError('Pinned Node/compiled combined SpeakerUI missing. Run inference.py --mode ui once without --no-setup.')
    env=os.environ.copy();env['PATH']=str(node.parent)+os.pathsep+env.get('PATH','')
    return node,env


def select_speaker(args,env):
    import sounddevice as sd
    index=args.speaker
    if index is None and env.get('IRIS_SPEAKER_INDEX'):
        index=int(env['IRIS_SPEAKER_INDEX'])
    if index is None:
        index=int(sd.default.device[1])
    if index<0:
        raise ValueError('No default host speaker. Use --speaker INDEX from --list-devices, or --no-audio.')
    sd.check_output_settings(device=index,channels=1,dtype='float32',samplerate=16000)
    args.speaker=index


def serve_ui(root,python,args,settings,runtime_env):
    from scripts.runtime.speaker_start import InstallLock,service_connection,service_request
    from scripts.runtime.speaker_tools import prepare_ui
    if not args.no_audio:
        select_speaker(args,runtime_env)
    node,tools_env=existing_ui(root) if args.no_setup else prepare_ui(root,build=True)
    lock=InstallLock(root)
    if not lock.acquire():
        raise RuntimeError('Another instance owns this installation; stop it explicitly before launching Iris')
    child=None
    try:
        token=uuid.uuid4().hex+uuid.uuid4().hex
        env={**tools_env,**launch_environment(root,python,args,settings,runtime_env,token=token)}
        child=subprocess.Popen([str(node),'dist-server/server/main.js'],cwd=Path(root)/'SpeakerUI',env=env)
        connection_args=argparse.Namespace(settings=settings,runtime_env=runtime_env,port=settings['server']['port'])
        connection=service_connection(connection_args)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if child.poll() is not None:
                raise RuntimeError('Combined SpeakerUI stopped during startup')
            try:
                with service_request(connection,'/api/health',timeout=1) as response:
                    health=json.load(response)
                if ready_matches(health,token):
                    break
            except (OSError,ValueError):
                pass
            time.sleep(.2)
        else:
            raise RuntimeError('Readiness failed: active runtime must identify frozen epoch44/258912 and its exact graph hash')
        print('Iris epoch44:',settings['server']['origin'],'Actions:',settings['actions']['enabled'],flush=True)
        if args.mic is not None:
            with service_request(connection,'/api/runtime/listen',data=json.dumps({'device':args.mic}).encode(),
                headers={'Content-Type':'application/json'},timeout=30) as response:
                print(response.read().decode(),flush=True)
        if not args.no_browser:
            webbrowser.open(settings['server']['origin'])
        return child.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        lock.close()


def microphone_cli(command,root,env,mic):
    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.PIPE,text=True)
    try:
        child.stdin.write(json.dumps({'id':'cli-listen','op':'listen','device':mic})+'\n')
        child.stdin.flush()
        return child.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if child.poll() is None:
            try:
                child.stdin.write('{"op":"shutdown"}\n');child.stdin.flush()
                child.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if child.stdin:child.stdin.close()


def main(argv=None,*,root=ROOT):
    root=Path(root).resolve();parser=make_parser(root);args=parser.parse_args(argv)
    if args.wake_wav and args.wav is None:
        parser.error('--wake-wav requires --wav PATH')
    if any(index is not None and index<0 for index in (args.mic,args.speaker)):
        parser.error('Microphone/speaker indices must be nonnegative')
    if not (args.check or args.wav or args.list_devices or args.mode=='ui' or args.mic is not None):
        parser.error('Choose --check, --wav PATH, --mode ui, --list-devices, or --mic INDEX for silent nonactuating CLI')
    if not args.list_devices:
        recipe=args.bundle/'recipe.json'
        if (not recipe.is_file() or hashlib.sha256(recipe.read_bytes()).hexdigest()!=
                '020c3deccb0f8cece0f23b8c016b6a63bdd0359321bb06cbd00e091966d46621'):
            parser.error('inference.py requires the pinned epoch44 bundle; use start.py for explicitly legacy inference')
    settings,runtime_env=resolve_settings(root,args)
    python=Path(sys.executable)
    if not args.no_setup:
        from scripts.runtime.speaker_tools import sync_python
        python=sync_python(root,source=True)  # Existing frozen runtime lock, never training requirements.
    if args.mode=='ui' and not (args.check or args.wav or args.list_devices):
        return serve_ui(root,python,args,settings,runtime_env)
    env=launch_environment(root,python,args,settings,runtime_env,token='')
    command=[str(python),'-B','-m','scripts.runtime.speaker_worker','--bundle',str(args.bundle.resolve()),
             '--engine','onnxruntime','--threads',str(args.threads),'--no-audio']
    if args.wake_root is not None:command+=['--wake-root',str(args.wake_root.resolve())]
    if args.check:command+=['--check','--profile','owner-protected']
    elif args.list_devices:command+=['--list-devices']
    elif args.wav:
        command+=['--wav',str(args.wav.resolve()),'--profile','benchmark-compatible']
        if args.wake_wav:command+=['--wake']
    else:
        command+=['--profile','benchmark-compatible']
        return microphone_cli(command,root,env,args.mic)
    return subprocess.call(command,cwd=root,env=env)


if __name__ == '__main__':
    raise SystemExit(main())
