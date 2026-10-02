"""Portable full-corpus CPU benchmark. No microphone, network or training."""
import argparse
from collections import Counter, defaultdict
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


def get_affinity():
    if hasattr(os, 'sched_getaffinity'):return sorted(os.sched_getaffinity(0))
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        mask, system = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(mask), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [i for i in range(64) if mask.value & (1 << i)]
    raise RuntimeError('Verified CPU affinity requires Windows or Linux')


def set_affinity(cpus):
    if hasattr(os, 'sched_setaffinity'):os.sched_setaffinity(0, cpus);return
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << i for i in cpus)):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    raise RuntimeError('CPU affinity is unavailable')


def pin_cpu(cpu):
    before = get_affinity()
    if cpu not in before:raise ValueError('CPU is outside allowed affinity: '+str(before))
    set_affinity([cpu]);observed = get_affinity()
    if observed != [cpu]:raise RuntimeError('CPU affinity did not take effect')
    return {'requested':cpu, 'previous':before, 'observed':observed, 'verified':True,
            'meaning':'One logical processor; no exclusive core reservation or fixed frequency'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False);stream.write('\n')


def create_output(folder):
    folder.mkdir(parents=True, exist_ok=False)


def load_trials(manifest):
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text(encoding='utf8'))
    if not isinstance(rows, list) or not rows:raise ValueError('Empty or invalid corpus')
    ids = set()
    for row in rows:
        if row['id'] in ids:raise ValueError('Duplicate trial ID')
        ids.add(row['id'])
        path = (manifest.parent/row['audio']).resolve()
        if not path.is_relative_to(manifest.parent):raise ValueError('Audio escapes manifest directory')
        if sha(path) != row['sha256']:raise ValueError('Audio hash mismatch: '+row['id'])
        if row.get('reference_status') not in ('supported', 'unsupported', 'ambiguous'):
            raise ValueError('Unclassified reference: '+row['id'])
    return rows


def typed_metrics(rows, key):
    from scripts.runtime.pcv_schema import canonical
    known = [r for r in rows if r['reference_status'] != 'ambiguous']
    pos = [r for r in known if r['expected'] is not None]
    neg = [r for r in known if r['expected'] is None]
    exact = sum(r[key] is not None and canonical(r[key]) == canonical(r['expected']) for r in pos)
    emitted = sum(r[key] is not None for r in known)
    false_accepts = sum(r[key] is not None for r in neg)
    return {'source_n':len(rows), 'scored_n':len(known), 'ambiguous_n':len(rows)-len(known),
        'positive_n':len(pos), 'negative_n':len(neg), 'exact':exact, 'emitted':emitted,
        'recall':exact/len(pos) if pos else None, 'precision':exact/emitted if emitted else None,
        'false_accepts':false_accepts, 'false_accept_rate':false_accepts/len(neg) if neg else None}


def quantiles(values):
    import numpy as np
    return {'n':len(values), **{key:float(np.percentile(values, q)) if values else None
        for key, q in [('p50',50), ('p95',95), ('p99',99)]},
        'mean':float(np.mean(values)) if values else None, 'max':max(values) if values else None}


def timing_summary(events):
    return {key:quantiles([e['timing'][key] for e in events if e['timing'].get(key) is not None])
            for key in ('frontend_ms', 'model_ms', 'finite_decoder_ms', 'total_ms', 'rtf')}


def peak_rss_mib():
    if os.name == 'nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                *[(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                  'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
        kernel = ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
        info = Counters();info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):return None
        return info.PeakWorkingSetSize/1024**2
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024)


def telemetry():
    result = {}
    for name, path, divisor in [('temperature_c','/sys/class/thermal/thermal_zone0/temp',1000),
                               ('cpu0_mhz','/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq',1000)]:
        try:result[name] = float(Path(path).read_text())/divisor
        except (OSError, ValueError):result[name] = None
    if Path('/usr/bin/vcgencmd').exists():
        run = subprocess.run(['/usr/bin/vcgencmd','get_throttled'], capture_output=True, text=True, timeout=2)
        result['throttled'] = run.stdout.strip() if run.returncode == 0 else None
    else:result['throttled'] = None
    return result


def summarize(rows, definitions):
    from scripts.runtime.pcv_schema import INTENTS, canonical
    out = {key:typed_metrics(rows, key) for key in ('accepted','candidate')}
    out['per_intent'] = {}
    for intent in INTENTS:
        references = [r for r in rows if r['expected'] and r['expected']['intent'] == intent]
        out['per_intent'][intent] = {}
        for key in ('accepted','candidate'):
            item = typed_metrics(references, key)
            emitted = sum(r['reference_status'] != 'ambiguous' and r[key] is not None and r[key]['intent'] == intent for r in rows)
            item.update(predicted_n=emitted, precision=item['exact']/emitted if emitted else None)
            out['per_intent'][intent][key] = item
    out['phrases'] = {d['id']:{key:typed_metrics([r for r in rows if r.get('phrase_id') == d['id']], key)
                             for key in ('accepted','candidate')} for d in definitions}
    out['unrecorded_phrases'] = [i for i, v in out['phrases'].items() if not v['candidate']['source_n']]
    out['slots'] = {}
    for key in ('accepted','candidate'):
        slots = defaultdict(lambda:{'n':0,'correct':0})
        for row in rows:
            if row['reference_status'] != 'supported' or not row['expected']:continue
            for name, value in row['expected']['slots'].items():
                item = slots[row['expected']['intent']+'/'+name];item['n'] += 1
                actual = row[key]
                item['correct'] += int(actual is not None and actual['intent'] == row['expected']['intent']
                                       and canonical(actual['slots'].get(name)) == canonical(value))
        out['slots'][key] = dict(slots)
    return out


def run(args):
    before = get_affinity()
    keys = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
    previous = {key:os.environ.get(key) for key in keys}
    try:
        return _run_pinned(args)
    finally:
        set_affinity(before)
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key] = value


def _run_pinned(args):
    affinity = pin_cpu(args.cpu)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    # Numerical imports follow affinity and thread settings, including in exported CLI.
    from scripts.runtime.pcv_final_runtime import Runtime, read_wav
    from scripts.runtime.pcv_pi_benchmark import hardware
    import numpy as np
    import onnxruntime as ort
    create_output(args.output)
    write(args.output/'protocol.json', {'utc':datetime.now(timezone.utc).isoformat(), 'affinity':affinity,
        'manifest_sha256':sha(args.manifest), 'definitions_sha256':sha(args.definitions),
        'bundle_files':{p.name:sha(p) for p in sorted(args.bundle.iterdir()) if p.is_file()},
        'warmup':args.warmup, 'repeat':args.repeat, 'timing_samples':args.timing_samples,
        'seed':args.seed, 'threads':1, 'quality_selection':'No fitting or tuning; existing policy unchanged'})
    rows = load_trials(args.manifest)
    definitions = json.loads(args.definitions.read_text())
    rss_import = peak_rss_mib()
    started = time.perf_counter();runtime = Runtime(args.bundle, diagnostic=True, threads=1,engine=getattr(args,'engine','onnxruntime'))
    loaded_ms = (time.perf_counter()-started)*1000;rss_loaded = peak_rss_mib()
    if runtime.engine=='onnxruntime':
        options = runtime.session.get_session_options()
        if options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1:
            raise RuntimeError('Unexpected ORT threading')
    elif runtime.session.net.opt.num_threads!=1:raise RuntimeError('Unexpected ncnn threading')
    cold_pcm = read_wav(args.manifest.parent/rows[0]['audio'])
    cold = runtime.recognize(cold_pcm)
    for _ in range(args.warmup):runtime.recognize(cold_pcm)
    # One recording resident at a time; file reading/hash checking is outside inference timing.
    rng = random.Random(args.seed);order = list(range(len(rows)));rng.shuffle(order)
    trials = [];observations = [];thermal = [telemetry()];infer_cpu = 0.;infer_wall = 0.
    loop_start = time.perf_counter()
    with (args.output/'trials.jsonl').open('x', encoding='utf8') as stream:
        for j, index in enumerate(order):
            row = rows[index];pcm = read_wav(args.manifest.parent/row['audio'])
            cpu = time.process_time();wall = time.perf_counter();event = runtime.recognize(pcm)
            infer_wall += time.perf_counter()-wall;infer_cpu += time.process_time()-cpu
            trial = {**row, 'accepted':event['output']['command'],
                     'candidate':(event['diagnostic_candidate'] or {}).get('candidate'), 'event':event}
            trials.append(trial);stream.write(json.dumps(trial, allow_nan=False)+'\n')
            if (j+1)%250 == 0:
                thermal.append(telemetry());print('REPLAY', j+1, '/',len(rows),flush=True)
    # Balanced deterministic repeats are latency observations, never new quality trials.
    groups = defaultdict(list)
    for row in rows:groups[(row['track'], row.get('publisher_intent'))].append(row)
    for group in groups.values():rng.shuffle(group)
    selected = []
    while len(selected)<min(args.timing_samples,len(rows)):
        for group in groups.values():
            if group and len(selected)<args.timing_samples:selected.append(group.pop())
    with (args.output/'timing.jsonl').open('x', encoding='utf8') as stream:
        for row in selected:
            pcm = read_wav(args.manifest.parent/row['audio'])
            for repeat in range(args.repeat):
                event = runtime.recognize(pcm);observations.append(event)
                stream.write(json.dumps({'id':row['id'],'repeat':repeat,'timing':event['timing']})+'\n')
    result = {'utc':datetime.now(timezone.utc).isoformat(), 'hardware':hardware(), 'affinity':affinity,
        'versions':{'numpy':np.__version__,'onnxruntime':ort.__version__,'python':platform.python_version()},
        'providers':runtime.session.get_providers(), 'engine':runtime.engine, 'threads':1, 'inference_mode':runtime.recipe['inference_mode'],
        'model_binding':runtime.recipe['binding'], 'architecture':runtime.recipe.get('architecture_class'),
        'model_config':runtime.recipe.get('model_config'), 'model_bytes':sum(p.stat().st_size for p in (args.bundle/'ncnn').glob('model.ncnn.*')) if runtime.engine=='ncnn' else (args.bundle/'model.onnx').stat().st_size, 'engine_model_sha256':runtime.engine_model_sha256,
        'catalog_entries':len(runtime.entries), 'policy_enabled':runtime.policy['enabled'],
        'policy_sha256':sha(args.bundle/'policy.json'), 'session_and_decoder_load_ms':loaded_ms,
        'cold_first_inference':cold['timing'], 'warmup_n':args.warmup,
        'full_corpus_timing':timing_summary([r['event'] for r in trials]),
        'repeated_timing':timing_summary(observations), 'repeated_recording_n':len(selected),
        'inference_cpu_seconds':infer_cpu, 'inference_wall_seconds':infer_wall,
        'inference_cpu_percent_one_core':100*infer_cpu/infer_wall,
        'replay_and_repeats_wall_seconds':time.perf_counter()-loop_start,
        'peak_rss_mib':{'after_import':rss_import,'after_load':rss_loaded,'whole_process':peak_rss_mib()},
        'telemetry_samples':thermal+[telemetry()], 'source_n':len(trials),
        'tracks':{}, 'live_microphone_measured':False, 'qualification_changed':False,
        'timing_scope':'Complete preloaded PCM validation, frontend, selected engine, finite decoding and policy. Excludes WAV/disk, wake/VAD, spoken duration, 400ms endpoint and UI/transport. Cold first call separate.',
        'memory_scope':'Process high-water RSS includes Python, runtime, one PCM clip, catalog, corpus metadata and retained result records; not model-only RAM.',
        'runtime_source_sha256':sha(Path(__file__).with_name('pcv_final_runtime.py')),
        'benchmark_source_sha256':sha(Path(__file__)), 'manifest_sha256':sha(args.manifest)}
    for track in sorted({r['track'] for r in trials}):
        subset = [r for r in trials if r['track'] == track]
        result['tracks'][track] = summarize(subset, definitions)
        result['tracks'][track]['provenance'] = {origin:summarize([r for r in subset if r['synthetic']==synthetic], [])
            for origin, synthetic in [('human',False),('supplied_synthetic',True)]}
        result['tracks'][track]['no_prior_fitting_pcm'] = {key:typed_metrics([r for r in subset
            if not {'train','dev','calibration'} & set(r.get('prior_pcm_splits',[]))],key) for key in ('accepted','candidate')}
    write(args.output/'metrics.json', result)
    write_summary(args.output,result)
    print('COMPLETE', args.output/'metrics.json', flush=True)
    return result


def write_summary(folder,result):
    import csv
    lines=['# Raspberry Pi recorded benchmark','',f"Engine: {result['engine']}",f"Hardware: {result['hardware']}",
           f"Frozen acceptance enabled: {result['policy_enabled']}",'',
           'Recorded replay only; live microphone evidence is generated separately with guided_test.py.','',
           '| Track | Exact raw commands | Exact accepted commands | Raw false accepts |',
           '| --- | --- | --- | --- |']
    for track,values in result['tracks'].items():
        raw=values['candidate'];accepted=values['accepted']
        lines.append(f"| {track} | {raw['exact']}/{raw['positive_n']} | {accepted['exact']}/{accepted['positive_n']} | {raw['false_accepts']}/{raw['negative_n']} |")
    timing=result['full_corpus_timing']['total_ms']
    lines.extend(['',f"Full inference p50/p95: {timing['p50']:.3f}/{timing['p95']:.3f} ms.",
                  f"Peak process RSS: {result['peak_rss_mib']['whole_process']} MiB.",
                  f"Session load: {result['session_and_decoder_load_ms']} ms.",
                  '',result['timing_scope'],result['memory_scope'],'',
                  'ME-03/04: per-intent, per-slot and per-phrase results are in metrics.json and CSVs.',
                  'ME-05: physical Pi identity is recorded; live microphone demo requires the separate guided test.',
                  'ME-06: CPU, latency, memory and thermal measurements are recorded in metrics.json.',
                  'ME-07/08: local execution is measured; this run does not claim an OS-level network isolation test.',
                  'Missing phrase recordings retain zero denominators. No qualification result is overridden.'])
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for filename,section in [('phrases.csv','phrases'),('slots.csv','slots')]:
        with (folder/filename).open('x',newline='',encoding='utf8') as stream:
            writer=csv.writer(stream);writer.writerow(['track','item','mode','correct','denominator'])
            for track,values in result['tracks'].items():
                if section=='phrases':
                    for item,modes in values[section].items():
                        for mode,row in modes.items():writer.writerow([track,item,mode,row['exact'],row['positive_n']])
                else:
                    for mode,items in values[section].items():
                        for item,row in items.items():writer.writerow([track,item,mode,row['correct'],row['n']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('models/default'))
    parser.add_argument('--engine',choices=['ncnn','onnxruntime'])
    parser.add_argument('--manifest', type=Path, default=Path('benchmark/manifest.json'))
    parser.add_argument('--definitions', type=Path, default=Path('benchmark/definitions.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--timing-samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261002)
    args = parser.parse_args()
    from scripts.runtime.speaker_config import resolve_config
    cfg,_=resolve_config(Path.cwd(),overrides={'engine':args.engine,'cpu':args.cpu},profile='benchmark');args.engine=cfg['inference']['engine'];args.cpu=cfg['benchmark']['cpu']
    if args.warmup<0 or args.repeat<1 or args.timing_samples<1:parser.error('Invalid warmup/repeat/sample count')
    run(args)


if __name__ == '__main__':main()
