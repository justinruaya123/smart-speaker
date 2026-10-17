"""Shared standard-library launcher for the checkout and exported application."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import ssl
from urllib.parse import urlsplit
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

from scripts.runtime.speaker_package import read,write,digest,inspect_archive,verify_install,installed_bundle,apply_update
from scripts.runtime.speaker_tools import sync_python,python_path,prepare_ui,find_uv,run


def make_parser(deployed):
    p=argparse.ArgumentParser(description='Smart speaker: setup, build/publish, WAV inference, microphone and benchmarks.')
    p.add_argument('--mode',choices=['setup','publish','serve','benchmark'])
    p.add_argument('--skip-train','--publish-only',action='store_true',help='Publish selected models without acquiring training data or fitting')
    p.add_argument('--benchmarks',choices=['include','skip','only'],help='Run benchmarks before serving/publishing, skip them, or run only benchmarks')
    p.add_argument('--benchmark-only',action='store_true',help='Benchmark selected models and exit; no training, UI, server or microphone')
    p.add_argument('--target',choices=['inference','benchmark','full','all','pi','standalone','both'],default='all')
    p.add_argument('--benchmark-manifest',type=Path,help='Pinned full benchmark manifest for benchmark/full publication')
    p.add_argument('--engine',choices=['ncnn','onnxruntime'],help='CPU inference engine (config default: ncnn)')
    p.add_argument('--profile',choices=['owner-protected','benchmark-compatible'],help='Owner-protected microphone or action-free command benchmark')
    p.add_argument('--run-id',default='release-v1');p.add_argument('--output',type=Path)
    p.add_argument('--wav',type=Path);p.add_argument('--wake-wav',action='store_true',help='WAV includes wake word and silence; replay through wake/VAD')
    p.add_argument('--list-mics',action='store_true');p.add_argument('--mic',type=int)
    p.add_argument('--threads',type=int,choices=range(1,9));p.add_argument('--port',type=int)
    p.add_argument('--cpu',type=int,help='Logical CPU for full-corpus single-core benchmark')
    p.add_argument('--update',type=Path,help='Apply a compatible model ZIP; default watches ~/smart-speaker-pi.zip')
    p.add_argument('--system-packages',action='store_true',help='Install missing audio libraries with apt on Debian/Pi OS')
    p.add_argument('--no-setup',action='store_true',help='Use the existing configured environment (offline/advanced)')
    p.add_argument('--yes',action='store_true',help='Use noninteractive defaults; microphone selection remains explicit')
    return p


def resolve_mode(args,*,deployed,interactive):
    if args.wav or args.list_mics:return 'infer','skip'
    if args.benchmark_only or args.benchmarks=='only':return 'benchmark','only'
    mode=args.mode or ('serve' if deployed else 'publish' if args.skip_train else None)
    if mode is None:
        if interactive:
            print('\n1. Setup: acquire/train selected workflow and publish\n2. Publish existing models (no training)\n3. Serve locally\n4. Benchmark only')
            answer=input('Choose [2]: ').strip() or '2'
            if answer not in ('1','2','3','4'):raise ValueError('Choose 1, 2, 3 or 4')
            mode={'1':'setup','2':'publish','3':'serve','4':'benchmark'}[answer]
        else:mode='publish' if args.skip_train else 'setup'
    benchmarks=args.benchmarks or ('only' if mode=='benchmark' else None)
    if benchmarks is None and interactive:
        answer=input('Benchmarks: 1 include, 2 skip, 3 benchmark only [1]: ').strip() or '1'
        if answer not in ('1','2','3'):raise ValueError('Choose 1, 2 or 3')
        benchmarks={'1':'include','2':'skip','3':'only'}[answer]
        if benchmarks=='only':mode='benchmark'
    return mode,benchmarks or 'skip'


def local_package(root,args):
    iris=root/'experiments/iris-release-v1/reproduction-release.json'
    if iris.exists():
        # Use only receipts bound to this release, never an older model cache.
        for record_path in sorted((root/'experiments/iris-release-v1').glob('reproduction-run-*.json'),reverse=True):
            record=read(record_path);package=record.get('package',{})
            if (record.get('release')!=iris.relative_to(root).as_posix() or
                    record.get('release_sha256')!=digest(iris)):continue
            path=root/package.get('path','missing')
            if record.get('status')=='complete' and path.is_file() and digest(path)==package.get('sha256'):
                return extract_local(root,path)
        run([python_path(root),root/'train.py','--skip-train'],cwd=root)
        return local_package(root,args)
    raise ValueError('Release inputs are not prepared. Run python train.py --skip-train first.')


def extract_local(root,path):
    manifest=inspect_archive(path);destination=root/'.runtime/packages'/digest(path)
    if not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:archive.extractall(destination)
    verify_install(destination)
    return destination,manifest


def benchmark(root,python,args):
    output=(args.output or root/args.settings['benchmark']['output'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    command=[python,'-m','scripts.runtime.iris_pi_benchmark','--output',output,
        '--engine',args.settings['inference']['engine'],'--threads',str(args.threads),
        '--bundle',installed_bundle(root)]
    if args.benchmark_manifest:command+=['--manifest',args.benchmark_manifest.resolve()]
    run(command,cwd=root)
    return read(output/'metrics.json')


class InstallLock:
    """OS lock releases on crashes; PID alone is never authority to kill a process."""
    def __init__(self,root):self.path=root/'state/service.lock';self.stream=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('a+b');self.stream.seek(0)
        if self.path.stat().st_size==0:self.stream.write(b'0');self.stream.flush();self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:self.stream.close();self.stream=None;return False
        return True
    def close(self):
        if self.stream:self.stream.close();self.stream=None


def service_connection(args):
    host=args.settings['server']['host']
    host={'0.0.0.0':'127.0.0.1','::':'::1'}.get(host,host)
    if ':' in host and not host.startswith('['):host='['+host+']'
    tls=bool(args.runtime_env.get('TLS_CERT_PATH'))
    return {'url':f"{'https' if tls else 'http'}://{host}:{args.port}",
            'origin':args.settings['server']['origin'],'tls_cert':args.runtime_env.get('TLS_CERT_PATH')}


def service_request(connection,path,*,data=None,headers=None,timeout=3):
    headers={'Host':urlsplit(connection['origin']).netloc,'Origin':connection['origin'],**(headers or {})}
    context=None
    if connection.get('tls_cert'):
        context=ssl.create_default_context(cafile=connection['tls_cert'])
        # Connect via the local bind address; authenticate the explicitly configured certificate.
        context.check_hostname=False
    request=urllib.request.Request(connection['url']+path,data=data,headers=headers)
    return urllib.request.urlopen(request,timeout=timeout,context=context)


def stop_previous(root):
    receipt=root/'state/service.json'
    if not receipt.exists():return
    info=read(receipt)
    if type(info.get('port')) is not int or not 1<=info['port']<=65535:raise ValueError('Invalid service receipt')
    connection=info.get('connection') or {'url':f"http://127.0.0.1:{info['port']}",'origin':f"http://127.0.0.1:{info['port']}"}
    try:
        with service_request(connection,'/api/runtime/shutdown',data=b'{}',headers={'Content-Type':'application/json','X-Launch-Token':info['token']}) as response:response.read()
    except urllib.error.URLError:pass


def serve(root,python,args,manifest,*,ui_root=None,benchmarks='skip'):
    app_root=ui_root or root
    node,env=prepare_ui(app_root,build=ui_root is not None)
    lock=InstallLock(root)
    if not lock.acquire():
        stop_previous(root)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if lock.acquire():break
            time.sleep(.2)
        else:raise RuntimeError('Another instance is still active; stop it before updating')
    state_path=root/'state/install.json';previous=read(state_path) if state_path.exists() else None
    child=None
    try:
        archive=args.update or Path.home()/'smart-speaker-pi.zip'
        if archive.exists():
            def load(folder):run([python,'-m','scripts.runtime.speaker_worker','--bundle',folder,'--profile','owner-protected','--check'],cwd=root)
            try:apply_update(root,archive,manifest,load)
            except (ValueError,OSError,subprocess.CalledProcessError) as error:
                print('Model update rejected; starting the previous bundle:',error,file=sys.stderr)
        elif args.update:raise FileNotFoundError(archive)
        bundle=installed_bundle(root)
        if benchmarks=='include':benchmark(root,python,args)
        token=uuid.uuid4().hex+uuid.uuid4().hex
        env.update(args.runtime_env)
        env.update({'SMART_SPEAKER_ROOT':str(root),'SMART_SPEAKER_PYTHON':str(python),'SMART_SPEAKER_BUNDLE':str(bundle),
            'SMART_SPEAKER_THREADS':str(args.threads),'SMART_SPEAKER_TOKEN':token,'INFERENCE_MODE':'onnx',
            'SERVE_UI':'true'})
        child=subprocess.Popen([str(node),'dist-server/server/main.js'],cwd=app_root/'SpeakerUI',env=env)
        deadline=time.monotonic()+60;connection=service_connection(args)
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError('SpeakerUI stopped during startup')
            try:
                with service_request(connection,'/api/health',timeout=1) as response:health=json.load(response)
                if health.get('runtime',{}).get('loaded') and health.get('inference')=='onnx' and health.get('instance')==token[:12]:break
            except (OSError,ValueError):pass
            time.sleep(.2)
        else:raise RuntimeError('Runtime readiness timed out')
        write(root/'state/service.json',{'port':args.port,'token':token,'connection':connection})
        print('Iris:',args.settings['server']['origin'],'\nChoose a microphone on the main screen. Actions:',args.settings['actions']['enabled'],flush=True)
        if args.mic is not None:
            data=json.dumps({'device':args.mic}).encode()
            with service_request(connection,'/api/runtime/listen',data=data,headers={'Content-Type':'application/json'},timeout=30) as response:print(response.read().decode())
        return child.wait()
    except KeyboardInterrupt:return 0
    except Exception:
        if previous is None:state_path.unlink(missing_ok=True)
        else:write(state_path,previous)
        raise
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        (root/'state/service.json').unlink(missing_ok=True);lock.close()


def main(root,argv=None):
    root=Path(root).resolve();deployed=(root/'deployment.json').exists()
    args=make_parser(deployed).parse_args(argv)
    if args.wake_wav and not args.wav:raise ValueError('--wake-wav requires --wav')
    config=read(root/'deployment.json') if deployed else {}
    from scripts.runtime.speaker_config import initialize_config,resolve_config
    initialize_config(root)
    args.settings,args.runtime_env=resolve_config(root,profile=config.get('target','inference'),
        overrides={'engine':args.engine,'threads':args.threads,'port':args.port,'cpu':args.cpu,'mic':args.mic,'profile':args.profile})
    args.threads=args.settings['inference']['threads'];args.port=args.settings['server']['port'];args.cpu=args.settings['benchmark']['cpu']
    args.mic=None if args.settings['audio']['microphone']==-1 else args.settings['audio']['microphone']
    if deployed and args.mode is None:args.mode=config.get('default_mode','serve')
    if deployed and args.benchmarks is None:args.benchmarks=config.get('benchmarks','skip')
    mode,benchmarks=resolve_mode(args,deployed=deployed,interactive=sys.stdin.isatty() and not args.yes and not deployed)
    if deployed and mode in ('setup','publish'):raise ValueError('Build/publish runs in the source checkout; this ZIP is ready to run')
    manifest=verify_install(root) if deployed else None
    executable=python_path(root)
    if not args.no_setup:
        executable=sync_python(root,source=not deployed)
        # Re-exec after bootstrap. Preserve the resolved menu choices and never reprompt.
        forwarded=list(sys.argv[1:] if argv is None else argv)
        forwarded+=['--no-setup','--yes','--mode',mode if mode!='infer' else 'serve','--benchmarks',benchmarks]
        return subprocess.call([str(executable),str(root/'start.py'),*forwarded],cwd=root)
    if not executable.exists():raise RuntimeError('Environment missing. Run start.py without --no-setup')
    if args.system_packages:
        if not sys.platform.startswith('linux') or not shutil.which('apt-get'):raise ValueError('--system-packages requires Debian/Pi OS')
        prefix=[] if hasattr(os,'geteuid') and os.geteuid()==0 else ['sudo']
        run([*prefix,'apt-get','update']);run([*prefix,'apt-get','install','-y','libportaudio2','libsndfile1'])
    if not deployed and mode in ('setup','publish'):
        command=[executable,root/'train.py']
        if args.skip_train or mode=='publish':command+=['--skip-train']
        return run(command,cwd=root).returncode
    if not deployed:runtime_root,manifest=local_package(root,args)
    else:runtime_root=root
    if mode=='infer':
        command=[executable,'-m','scripts.runtime.speaker_worker','--bundle',installed_bundle(runtime_root),'--diagnostic','--threads',str(args.threads)]
        if args.wav:command+=['--wav',args.wav.resolve()]
        if args.wake_wav:command+=['--wake']
        if args.list_mics:command+=['--list-mics']
        return run(command,cwd=runtime_root,env={**os.environ,**args.runtime_env}).returncode
    if mode=='benchmark' or benchmarks=='only':
        benchmark(runtime_root,executable,args)
        return 0
    if deployed and config.get('target') in ('benchmark','standalone'):raise ValueError('This package has no UI; use --wav, --list-mics or --benchmark-only. Extract the inference/full ZIP for SpeakerUI.')
    return serve(runtime_root,executable,args,manifest,ui_root=None if deployed else root,benchmarks=benchmarks)
