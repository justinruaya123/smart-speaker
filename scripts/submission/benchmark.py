"""Silent recorded inference on this computer or an SSH-connected Raspberry Pi."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile

from .common import ROOT, extract_verified, fetch, read, reexec, run_id, save, sha, write_archive


def validate_host(value):
    if not re.fullmatch(r'(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
        raise ValueError('Enter an SSH alias or user@hostname/IP, without shell options. Use --ssh-port/--identity separately.')
    return value


def new_output(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=False); return path


def is_raspberry_pi(machine, model):
    return machine.lower() in ('aarch64', 'arm64', 'armv7l') and 'raspberry pi' in model.lower()


def runtime_entries(model_choice='full'):
    """Use the canonical publisher's runtime closure; no trainers or datasets."""
    from .common import runtime_entries as shared_runtime_entries
    entries = shared_runtime_entries(model_choice=model_choice)
    for name in ('benchmark.py', 'scripts/submission/benchmark.py'):
        entries[name] = (ROOT/name).read_bytes()
    entries['submission/benchmark/holdout.json'] = (ROOT/'submission/benchmark/holdout.json').read_bytes()
    if (ROOT/'docs/BENCHMARK-HANDOFF.md').exists():
        entries['docs/BENCHMARK-HANDOFF.md'] = (ROOT/'docs/BENCHMARK-HANDOFF.md').read_bytes()
    return entries


def prepare_samples(root=ROOT):
    """Download only the pinned 15 MB held-out shard, never training data."""
    spec = read(Path(root)/'submission/benchmark/holdout.json')
    directory = Path(root)/'.cache/submission/holdout'
    for row in spec['rows']:
        target = directory/row['audio']
        if target.exists() and sha(target) != row['sha256']:
            raise ValueError('Cached holdout changed: ' + row.get('id', row['audio']))
    if all((directory/row['audio']).is_file() for row in spec['rows']):
        return spec, directory
    import pyarrow.parquet as pq
    source = spec['source']
    shard = fetch(f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{source['file']}",
                  Path(root)/'.cache/submission/downloads/holdout.parquet', source['sha256'])
    audio = pq.read_table(shard, columns=['audio']).column('audio')
    directory.mkdir(parents=True, exist_ok=True)
    for row in spec['rows']:
        target = directory/row['audio']
        payload = audio[row['parquet_row']].as_py()['bytes']
        if __import__('hashlib').sha256(payload).hexdigest() != row['sha256']:
            raise ValueError('Pinned holdout audio changed: ' + row['id'])
        if target.exists():
            if sha(target) != row['sha256']: raise ValueError('Cached holdout changed')
        else:
            with target.open('xb') as stream: stream.write(payload)
    return spec, directory


def worker(args):
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    from scripts.runtime.pcv_schema import canonical
    from scripts.submission.model import load_runtime
    output = new_output(args.output)
    spec = read(ROOT/'submission/benchmark/holdout.json')
    sample_root = args.samples or ROOT/'.cache/submission/holdout'
    rows = spec['rows'][:args.limit] if args.limit else spec['rows']
    if not rows: raise ValueError('No benchmark recordings')
    if hasattr(os, 'sched_getaffinity'):
        available = sorted(os.sched_getaffinity(0)); cpu = available[0] if args.cpu is None else args.cpu
        if cpu not in available: raise ValueError('Selected CPU is not available')
        os.sched_setaffinity(0, {cpu})
    else: cpu = None
    model = 'unavailable'
    for name in ('/proc/device-tree/model', '/sys/firmware/devicetree/base/model'):
        if Path(name).exists(): model = Path(name).read_bytes().decode(errors='replace').rstrip('\0'); break
    runtime = load_runtime(args.model or ROOT/'submission/models'/args.model_choice, threads=args.threads, mode='demo')
    wall_start = time.perf_counter(); cpu_start = time.process_time(); trials = []; timings = []; resources = []
    # Warm-up is recorded separately, then every evaluation recording is timed.
    first = rows[0]
    if sha(sample_root/first['audio']) != first['sha256']: raise ValueError('Warm-up audio checksum mismatch')
    first_pcm, first_rate = sf.read(sample_root/first['audio'], dtype='float32')
    runtime.recognize(first_pcm, first_rate)
    for i,row in enumerate(rows, 1):
        path = sample_root/row['audio']
        if sha(path) != row['sha256']: raise ValueError('Evaluation audio checksum mismatch')
        pcm, rate = sf.read(path, dtype='float32')
        event = runtime.recognize(pcm, rate)
        candidate = (event.get('diagnostic_candidate') or {}).get('candidate')
        trial = dict(id=row['id'], kind=row['kind'], expected=row['expected'],
                     metric_eligible=row['metric_eligible'], reference_reason=row.get('reference_reason'),
                     raw=candidate, demo=event['application']['command'],
                     qualified=event['qualified_output']['command'], event=event)
        trials.append(trial)
        for _ in range(args.repeats):
            measured = runtime.recognize(pcm, rate)
            timings.append({'id':row['id'], **measured['timing']})
        telemetry = {'after_recording':i}
        thermal = Path('/sys/class/thermal/thermal_zone0/temp')
        if thermal.exists(): telemetry['temperature_c'] = int(thermal.read_text()) / 1000
        frequency = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
        if frequency.exists(): telemetry['cpu_frequency_khz'] = int(frequency.read_text())
        if shutil.which('vcgencmd'):
            telemetry['throttled'] = subprocess.run(['vcgencmd','get_throttled'],text=True,capture_output=True).stdout.strip()
        resources.append(telemetry)
        if i % 25 == 0: print(f'REPLAY {i}/{len(rows)}', flush=True)
    def metrics(selected, key):
        selected = [r for r in selected if r['metric_eligible']]
        positive = [r for r in selected if r['expected'] is not None]
        negative = [r for r in selected if r['expected'] is None]
        correct = sum(r[key] is not None and canonical(r[key]) == canonical(r['expected']) for r in positive)
        wrong = sum(r[key] is not None and canonical(r[key]) != canonical(r['expected']) for r in positive)
        accepted = sum(r[key] is not None for r in selected)
        fa = sum(r[key] is not None for r in negative)
        intent = sum(r[key] is not None and r[key]['intent'] == r['expected']['intent'] for r in positive)
        return dict(n=len(selected), positive_n=len(positive), negative_n=len(negative),
                    complete_correct=correct, correct_accepts=correct, wrong_accepts=wrong,
                    accepted=accepted, unknown_false_accepts=fa,
                    complete_recall=correct/len(positive) if positive else None,
                    intent_correct=intent, intent_recall=intent/len(positive) if positive else None,
                    precision=correct/accepted if accepted else None, false_accept_rate=fa/len(negative) if negative else None)
    def q(values):
        return {'n':len(values), 'p50':float(np.percentile(values,50)), 'p95':float(np.percentile(values,95))} if values else {'n':0,'p50':None,'p95':None}
    from collections import Counter
    rejected_correct = {kind:dict(Counter(r['event']['application']['reason'] for r in trials
        if r['kind']==kind and r['metric_eligible'] and r['expected'] is not None
        and r['raw'] is not None and canonical(r['raw'])==canonical(r['expected']) and r['demo'] is None))
        for kind in ('human','synthetic')}
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024 if sys.platform == 'darwin' else 1024)
    except ImportError: rss = None
    elapsed = time.perf_counter() - wall_start
    cpu_seconds = time.process_time()-cpu_start
    result = dict(format='smart-speaker-benchmark-v1', mode='silent_recorded_replay',
        hardware=dict(model=model, machine=platform.machine(), os=platform.platform(), python=platform.python_version(),
                      physical_pi_measured=is_raspberry_pi(platform.machine(), model), cpu_affinity=[cpu] if cpu is not None else None),
        live_microphone_measured=False, wake_vad_measured=False, actions_enabled=False,
        source=spec['source'], published_rows=len(rows), excluded_reference_rows=sum(not r['metric_eligible'] for r in rows),
        model_binding=runtime.recipe['binding'], graph_bytes=(runtime.root/'model.onnx').stat().st_size,
        model_bundle_bytes=sum(p.stat().st_size for p in runtime.root.rglob('*') if p.is_file()),
        shipped_wake_vad_bytes=sum(p.stat().st_size for p in (runtime.root/'wake').rglob('*') if p.is_file()),
        parameters=runtime.recipe.get('parameters'), qualification_enabled=runtime.policy['enabled'],
        recall_policy_sha256=sha(runtime.root/'recall-policy.json') if (runtime.root/'recall-policy.json').is_file() else None,
        demo_policy='bound_per_intent_recall' if runtime.recipe.get('recall_policy_sha256') else 'fixed_intent_.95_slots_.90',
        model_choice=args.model_choice if not args.model else 'custom', threads=args.threads,frontend_blas_threads=1,
        raw_correct_but_demo_rejected_by_reason=rejected_correct,
        session_load_ms=runtime.load_ms, peak_rss_mib=rss,
        wall_seconds=elapsed, cpu_seconds=cpu_seconds, cpu_percent_one_core=100*cpu_seconds/elapsed,
        peak_rss_scope='fresh inference worker; excludes Parquet acquisition and SSH processes',
        timing_includes=['frontend','ONNX CPU','finite catalog decoder','typed validation','demo policy'],
        timing_excludes=['file IO','speech duration','wake listening','400 ms endpoint wait','SSH transfer','setup'],
        warmup_recordings=1, timing_repeats_per_recording=args.repeats,
        latency_ms=q([r['total_ms'] for r in timings]), rtf=q([r['rtf'] for r in timings if r['rtf'] is not None]),
        versions={n:importlib.metadata.version(n) for n in ('numpy','onnxruntime','soundfile','einops')},
        policies={key:{kind:metrics([r for r in trials if kind == 'all' or r['kind'] == kind],key)
                       for kind in ('all','human','synthetic')} for key in ('raw','demo','qualified')},
        per_intent={intent:metrics([r for r in trials if r['expected'] and r['expected']['intent'] == intent], 'demo')
                    for intent in sorted({r['expected']['intent'] for r in trials if r['expected']})},
        per_intent_by_kind={kind:{intent:metrics([r for r in trials if r['kind']==kind and r['expected'] and r['expected']['intent']==intent], 'demo')
                    for intent in sorted({r['expected']['intent'] for r in trials if r['expected']})}
                    for kind in ('human','synthetic')},
        limitations=['Previously examined holdout; thresholds are frozen.',
                    'Typed scoring excludes uncertain references and retains their predictions.',
                    'Recorded replay does not establish acoustic, wake or microphone performance.'])
    save(output/'metrics.json', result)
    save(output/'telemetry.json', resources)
    for name,values in [('trials.jsonl',trials),('timing.jsonl',timings)]:
        with (output/name).open('x',encoding='utf-8') as stream:
            for row in values: stream.write(json.dumps(row,allow_nan=False)+'\n')
    lines = ['# Silent recorded benchmark', '', f"Physical Pi measured: {result['hardware']['physical_pi_measured']}; microphone/wake measured: false.", '',
             'Raw = finite candidate recognition; demo = confidence-gated acceptance; qualified = frozen release policy.', '',
             '| Policy / group | Correct / positives | UNKNOWN accepts / negatives |', '|---|---:|---:|']
    for policy,groups in result['policies'].items():
        for kind,m in groups.items(): lines.append(f"| {policy} / {kind} | {m['complete_correct']}/{m['positive_n']} | {m['unknown_false_accepts']}/{m['negative_n']} |")
    lines += ['', f"Inference p50/p95: {result['latency_ms']['p50']:.2f}/{result['latency_ms']['p95']:.2f} ms; RTF p95: {result['rtf']['p95']}.",
              f"Correct raw human commands rejected by demo gates: {rejected_correct['human']}.",
              f"Excluded uncertain references: {result['excluded_reference_rows']}; all predictions retained.",
              'Includes frontend, model, finite decoding and validation; excludes disk, speech, wake, endpoint wait and SSH.',
              f"Runtime peak RSS: {rss} MiB; session loading: {runtime.load_ms:.2f} ms. Qualification remains disabled."]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


def ssh_command(args, code, *arguments):
    command = ['ssh', '-T', '-p', str(args.ssh_port), '-o', 'ConnectTimeout=20', '-o', 'StrictHostKeyChecking=accept-new']
    if args.identity: command += ['-i', str(args.identity)]
    command += [validate_host(args.host), ' '.join(shlex.quote(s) for s in ['python3','-c',code,*map(str,arguments)])]
    return command


def remote(args, samples):
    token = run_id(); output = Path(args.output)
    if output.exists(): raise FileExistsError('Use a new local result directory')
    entries = runtime_entries(args.model_choice)
    for row in read(ROOT/'submission/benchmark/holdout.json')['rows']:
        entries['.cache/submission/holdout/'+row['audio']] = (samples/row['audio']).read_bytes()
    package = ROOT/'.cache/submission/transfers'/f'{token}.zip'
    receipt = write_archive(package, entries, purpose='silent-benchmark', actions_enabled=False)
    # Binary stdin transport avoids scp shell/path differences on Windows and macOS.
    upload = "import pathlib,sys; p=pathlib.Path.home()/'.cache/smart-speaker-submission'/sys.argv[1]; p.mkdir(parents=True,exist_ok=False); (p/'payload.zip').write_bytes(sys.stdin.buffer.read())"
    print('Transferring inference runtime and held-out audio over SSH…', flush=True)
    with package.open('rb') as stream:
        subprocess.run(ssh_command(args, upload, token), stdin=stream, check=True)
    # Verify the source archive using only stdlib before importing any shipped code.
    execute = """import pathlib,sys,zipfile,hashlib,json,os,subprocess
p=pathlib.Path.home()/'.cache/smart-speaker-submission'/sys.argv[1]
z=p/'payload.zip'
if hashlib.sha256(z.read_bytes()).hexdigest()!=sys.argv[2]: raise ValueError('Transfer checksum mismatch')
with zipfile.ZipFile(z) as archive:
    names=archive.namelist(); manifest=json.loads(archive.read('SUBMISSION-MANIFEST.json'))
    if len(names)!=len(set(names)) or set(names)!={*manifest['files'],'SUBMISSION-MANIFEST.json'}: raise ValueError('Archive inventory mismatch')
    for name in names:
        path=pathlib.PurePosixPath(name)
        if not name or not path.parts or path.as_posix()!=name or name=='.' or path.is_absolute() or '..' in path.parts or chr(92) in name or ':' in name: raise ValueError('Unsafe archive member')
    for name,digest in manifest['files'].items():
        if hashlib.sha256(archive.read(name)).hexdigest()!=digest: raise ValueError('Archive checksum mismatch')
    (p/'work').mkdir(exist_ok=False); archive.extractall(p/'work')
os.chdir(p/'work')
subprocess.run([sys.executable,'benchmark.py','--target','pc','--worker','--output','results','--threads',sys.argv[3],'--repeats',sys.argv[4],'--model-choice',sys.argv[6],*(['--limit',sys.argv[5]] if sys.argv[5]!='0' else [])],check=True)
print('REMOTE_RESULTS_READY',flush=True)
"""
    subprocess.run(ssh_command(args, execute, token, receipt['sha256'], args.threads, args.repeats, args.limit or 0, args.model_choice), check=True)
    collect = "import pathlib,sys,zipfile,io; p=pathlib.Path.home()/'.cache/smart-speaker-submission'/sys.argv[1]/'work/results'; b=io.BytesIO(); z=zipfile.ZipFile(b,'w',compression=zipfile.ZIP_DEFLATED); [(z.write(f,f.relative_to(p).as_posix())) for f in p.rglob('*') if f.is_file()]; z.close(); sys.stdout.buffer.write(b.getvalue())"
    result = subprocess.run(ssh_command(args, collect, token), check=True, stdout=subprocess.PIPE)
    with zipfile.ZipFile(io.BytesIO(result.stdout)) as archive:
        from .common import safe_name
        names = archive.namelist()
        required = {'metrics.json','telemetry.json','trials.jsonl','timing.jsonl','REPORT.md'}
        if len(names) != len(set(names)) or not all(safe_name(n) for n in names) or not required <= set(names):
            raise ValueError('Unsafe or incomplete remote results')
        new_output(output); archive.extractall(output)
    save(output/'ssh-receipt.json', dict(host=args.host, remote_directory=f'~/.cache/smart-speaker-submission/{token}', package=receipt))
    print(f'Pi results retrieved: {output.resolve()}', flush=True)
    return read(output/'metrics.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['pc','pi'])
    parser.add_argument('--host', help='SSH alias or user@host; prompted for Pi')
    parser.add_argument('--ssh-port', type=int, default=22); parser.add_argument('--identity', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cpu', type=int); parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Smoke subset only; omit for the full held-out set')
    parser.add_argument('--model-choice', choices=['full','compact'], default='full', help='Full 2M shipping model (default) or optional compact comparison')
    parser.add_argument('--model', type=Path); parser.add_argument('--samples', type=Path)
    parser.add_argument('--no-setup', action='store_true'); parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8 or args.repeats < 1 or args.limit is not None and args.limit < 1:
        parser.error('threads must be 1..8; repeats/limit must be positive')
    if not 1 <= args.ssh_port <= 65535: parser.error('Invalid SSH port')
    if not args.target:
        if not sys.stdin.isatty(): parser.error('Use --target pc or --target pi in noninteractive runs')
        args.target = 'pi' if input('[1] Raspberry Pi over SSH  [2] This PC (silent replay): ').strip() != '2' else 'pc'
    if args.target == 'pi':
        args.host = validate_host(args.host or input('Raspberry Pi SSH address (user@host): ').strip())
        if args.model or args.samples or args.cpu is not None: parser.error('--model, --samples and --cpu apply to local PC runs')
    if not args.output: args.output = ROOT/'.cache/submission/benchmarks'/run_id()
    args.output = args.output.resolve()
    if args.model: args.model = args.model.resolve()
    if args.samples: args.samples = args.samples.resolve()
    if args.identity: args.identity = args.identity.resolve()
    if args.output.exists(): raise FileExistsError('Use a new output directory: ' + str(args.output))
    if not args.no_setup:
        forwarded = ['--target',args.target,'--output',str(args.output),
                     '--threads',str(args.threads),'--repeats',str(args.repeats),'--ssh-port',str(args.ssh_port),
                     '--model-choice',args.model_choice]
        if args.host: forwarded += ['--host',args.host]
        for name in ('identity','model','samples','cpu','limit'):
            value = getattr(args,name)
            if value is not None: forwarded += ['--'+name,str(value)]
        if args.worker: forwarded += ['--worker']
        return reexec('benchmark.py', forwarded, group=None if args.worker else 'benchmark')
    if args.worker:
        worker(args); return 0
    if args.samples:
        samples = args.samples
    else:
        _, samples = prepare_samples()
    if args.target == 'pi': remote(args, samples)
    else:
        command = [sys.executable, str(ROOT/'benchmark.py'), '--target', 'pc', '--worker', '--no-setup',
                   '--samples', str(samples), '--output', str(args.output),
                   '--threads', str(args.threads), '--repeats', str(args.repeats),'--model-choice',args.model_choice]
        for name in ('model','cpu','limit'):
            value = getattr(args,name)
            if value is not None: command += ['--'+name,str(value)]
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        # Acquisition can load large Parquet buffers; measure RSS in a clean process.
        subprocess.run(command, check=True, env=env)
    print('Saved:', args.output.resolve(), flush=True)
    return 0
