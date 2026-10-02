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


def source_release(root,run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):raise ValueError('Invalid run ID')
    from scripts.pcv_release import validate_release
    folder=root/'experiments/pcv-release-train-v1/runs'/run_id
    return validate_release(root,folder/'release.json',folder/'selection-before-calibration.json')


def source_path(root,name):
    return root/Path(name.replace('\\','/'))


def publish(root,args,benchmarks):
    from scripts.pcv_final_delivery import package
    release=source_release(root,args.run_id)
    target={'pi':'full','standalone':'benchmark','both':'all'}.get(args.target,args.target)
    targets=['inference','benchmark','full'] if target=='all' else [target]
    corpus=args.benchmark_manifest or root/'.cache/pcv-full-benchmark-v1/corpus/full-manifest.json'
    if any(t!='inference' for t in targets) and not corpus.is_file():raise ValueError('Benchmark data missing. Supply --benchmark-manifest or publish --target inference.')
    if any(t!='benchmark' for t in targets):prepare_ui(root,build=True)
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    output=args.output or root/'dist/speaker'/token
    output.mkdir(parents=True,exist_ok=True)
    packages={}
    from scripts.speaker_ncnn_export import prepare_ncnn
    ncnn_folder=prepare_ncnn(root,release,args.run_id)
    for target in targets:
        path=output/f'smart-speaker-{target}.zip'
        receipt=package(source_path(root,release['bundle']),path,diagnostic=not release['qualified'],
            sample_manifest=source_path(root,release['sample_manifest']),source_receipt=source_path(root,release['source_receipt']),
            source_card=source_path(root,release['source_card']),
            qualification_receipt=source_path(root,release['qualification_receipt']) if release.get('qualification_receipt') else None,
            app_target=target,benchmarks=benchmarks,
            benchmark_manifest=corpus if target!='inference' else None,ncnn_folder=ncnn_folder)
        inspect_archive(path);packages[target]=str(path.resolve())
        print('Published',path,'acceptance_enabled=',receipt['acceptance_enabled'])
    write(root/'.runtime/latest.json',{'packages':packages,'run_id':args.run_id})
    write(output/'release.json',{'run_id':args.run_id,'packages':{key:{'file':Path(path).name,'bytes':Path(path).stat().st_size,'sha256':digest(path)} for key,path in packages.items()}})
    return packages


def local_package(root,args):
    latest=root/'.runtime/latest.json'
    if latest.exists():
        value=read(latest)
        if value.get('run_id')==args.run_id:
            paths=value['packages'];path=Path(paths.get('full') or paths.get('inference') or paths.get('benchmark') or paths.get('standalone') or paths['pi'])
            if path.exists():return extract_local(root,path)
    # Reuse canonical packaging when local inference has not been prepared yet.
    from scripts.source_setup import ensure_dependencies
    ensure_dependencies(root)
    old_target=args.target;old_output=args.output
    args.target='inference';args.output=None
    try:packages=publish(root,args,'skip')
    finally:args.target=old_target;args.output=old_output
    return extract_local(root,Path(packages['inference']))


def extract_local(root,path):
    manifest=inspect_archive(path);destination=root/'.runtime/packages'/digest(path)
    if not destination.exists():
        destination.mkdir(parents=True)
        with zipfile.ZipFile(path) as archive:archive.extractall(destination)
    verify_install(destination)
    return destination,manifest


def benchmark(root,python,args):
    output=(args.output or root/args.settings['benchmark']['output'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    if (root/'benchmark/manifest.json').exists():
        if args.threads!=1:raise ValueError('Full benchmark requires --threads 1')
        # Fresh process sets affinity and native thread limits before any numerical import.
        # It also prevents benchmark settings from leaking into subsequent serving.
        run([python,'-m','scripts.runtime.pcv_full_benchmark','--output',output,
            '--engine',args.settings['inference']['engine'],
            '--cpu',str(args.cpu),'--bundle',installed_bundle(root),
            '--manifest',root/'benchmark/manifest.json','--definitions',root/'benchmark/definitions.json'],cwd=root)
        return read(output/'metrics.json')
    if not (root/'scripts/runtime/pcv_pi_benchmark.py').exists():raise ValueError('Benchmark data is not in the inference package. Use the benchmark or full ZIP.')
    from scripts.runtime.pcv_pi_benchmark import run as evaluate
    namespace=argparse.Namespace(root=root,output=output,threads=args.threads,engine=args.settings['inference']['engine'],manifest=None,guided=False,
        mic=None,case=None,seconds=None,no_wake=False,bundle=installed_bundle(root))
    result=evaluate(namespace)
    print('Acceptance enabled:',result['policy_enabled'],'Exact accepted commands:',result['accepted']['exact'],'/',result['accepted']['positive_n'])
    return result


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
            def load(folder):run([python,'-m','scripts.runtime.pcv_final_runtime','--bundle',folder,'--diagnostic','--check'],cwd=root)
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
        print('SpeakerUI:',args.settings['server']['origin'],'\nSelect a microphone in Debug > Inference. Actions:',args.settings['actions']['enabled'],flush=True)
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
    if deployed:initialize_config(root)
    args.settings,args.runtime_env=resolve_config(root,profile=config.get('target','inference'),
        overrides={'engine':args.engine,'threads':args.threads,'port':args.port,'cpu':args.cpu,'mic':args.mic})
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
        from scripts.source_setup import ensure_dependencies
        ensure_dependencies(root,training=mode=='setup' and not args.skip_train)
        if mode=='setup' and not args.skip_train:
            from scripts.pcv_release import run_workflow
            run_workflow(root=root,run_id=args.run_id,python=executable,release_only=True)
        packages=publish(root,args,benchmarks)
        if benchmarks in ('include','only'):
            local,_=extract_local(root,Path(packages.get('benchmark') or packages.get('full') or packages['inference']))
            benchmark(local,executable,args)
        return 0
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
