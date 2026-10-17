"""Action-free direct-VCM replay and resource measurements on the actual host.

No microphone, wake gate, owner session, UI gateway or action provider is started.
Accuracy uses the first pass per recording; all repeats contribute to timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RATE = 16000
MAX_SAMPLES = 240000


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _validate_output(value):
    from scripts.runtime.pcv_schema import validate_command
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'accepted', 'command', 'reason'}
            or type(value['schema_version']) is not int or value['schema_version'] != 3
            or type(value['accepted']) is not bool):
        raise ValueError('Invalid schema-v3 runtime output')
    if value['accepted']:
        validate_command(value['command'])
        if value['reason'] is not None:
            raise ValueError('Accepted output cannot carry a rejection reason')
    elif value['command'] is not None or not isinstance(value['reason'], str) or not value['reason']:
        raise ValueError('Abstention requires a null command and explicit reason')


def _audio_path(row, manifest, root):
    original = row.get('audio')
    if not isinstance(original, str) or not original:
        raise ValueError('Each recording requires a relative audio path')
    pure = PurePosixPath(original)
    if pure.is_absolute() or '..' in pure.parts or ':' in original or '\\' in original:
        raise ValueError('Use a safe relative audio path: ' + original)
    candidates = list(dict.fromkeys(path.resolve() for path in
        (manifest.parent / original, root / original) if path.is_file()))
    if len(candidates) > 1:
        raise ValueError('Ambiguous audio path exists under both manifest and installation: ' + original)
    if candidates:
        path = candidates[0]
    else:
        # The research checkout may have a pinned portable payload map. Reuse its
        # resolver only when present; ordinary installed packages need none of it.
        receipt = root / 'experiments/iris-release-v1/reproduction-inputs.json'
        if receipt.is_file() and importlib.util.find_spec('scripts.iris_release') is not None:
            from scripts.iris_release import resolve_audio
            path = resolve_audio({**row, 'sha256': row['sha256']}, root)
        else:
            raise FileNotFoundError('Missing recording ' + original + '; place it beside the manifest '
                                    'or under the installation root, preserving its declared SHA-256.')
    if digest(path) != row['sha256']:
        raise ValueError('Recording hash/identity mismatch: ' + original)
    return path


def load_manifest(path, root):
    from scripts.runtime.pcv_schema import validate_command
    path, root = Path(path).resolve(), Path(root).resolve()
    rows, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or type(value.get('id')) not in (str, int):
                    raise ValueError('Each recording requires an id')
                identity = str(value['id'])
                if not identity or identity in seen:
                    raise ValueError('Empty or duplicate recording id')
                if 'expected' not in value and 'command' not in value:
                    raise ValueError('Explicit expected command or expected:null is required')
                expected = value['expected'] if 'expected' in value else value['command']
                if expected is not None:
                    validate_command(expected)
                if 'expected' in value and 'command' in value and canonical(value['expected']) != canonical(value['command']):
                    raise ValueError('Conflicting expected and command references')
                expected_output = value.get('expected_output')
                if expected_output is not None:
                    _validate_output(expected_output)
                    if canonical(expected_output['command']) != canonical(expected):
                        raise ValueError('Typed expected_output disagrees with expected command')
                sha = value.get('sha256', value.get('audio_sha256'))
                if not isinstance(sha, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', sha):
                    raise ValueError('A recording requires sha256 (or audio_sha256)')
                alternate = value.get('audio_sha256', sha)
                if not isinstance(alternate, str) or alternate.lower() != sha.lower():
                    raise ValueError('Conflicting recording hashes')
                row = dict(id=identity, audio=value.get('audio'), sha256=sha.lower(), expected=expected,
                    expected_output=expected_output, scored=value.get('metric_eligible') is not False
                    and value.get('reference_status') != 'ambiguous',
                    reference_status=value.get('reference_status', 'explicit_typed_reference'),
                    provenance=value.get('provenance', value.get('kind', 'not_audited')))
                row['path'] = _audio_path(row, path, root)
                # Validate the original format before loading any model; do not
                # manufacture mono/16 kHz inputs by mixing or resampling them.
                _wav_info(row['path'])
                rows.append(row)
                seen.add(identity)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Invalid manifest reference at line {number}: {error}') from error
    if not rows:
        raise ValueError('Manifest has no recording rows; supply a hash-bound JSONL command manifest.')
    return rows


def _wav_info(path):
    import soundfile as sf
    info = sf.info(path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1 or info.format not in {'WAV', 'WAVEX'}:
        raise ValueError('Replay requires original mono 16000 Hz WAV PCM; no mixing or resampling is performed.')
    if not 1 <= info.frames <= MAX_SAMPLES:
        raise ValueError('Command WAV must contain 1..240000 samples (at most 15 seconds).')
    if Path(path).stat().st_size > 32 * 1024**2:
        raise ValueError('Command WAV exceeds the 32 MiB file bound.')
    return info


def read_pcm(path, expected_sha256=None):
    import soundfile as sf
    _wav_info(path)
    payload = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Recording identity changed after preflight: ' + str(path))
    with sf.SoundFile(io.BytesIO(payload)) as stream:
        if stream.samplerate != SAMPLE_RATE or stream.channels != 1 or not 1 <= len(stream) <= MAX_SAMPLES:
            raise ValueError('Expected original mono 16000 Hz command PCM')
        pcm = stream.read(dtype='float32', always_2d=False)
    if pcm.ndim != 1 or not np.isfinite(pcm).all() or np.max(np.abs(pcm), initial=0) > 1:
        raise ValueError('Expected finite mono PCM bounded to [-1, 1]')
    return np.ascontiguousarray(pcm)


def device_model():
    for name in ('/sys/firmware/devicetree/base/model', '/proc/device-tree/model'):
        try:
            return Path(name).read_bytes().decode(errors='replace').strip('\0\n ')
        except OSError:
            pass
    return None


def hardware():
    model, machine = device_model(), platform.machine()
    pi = bool(model and 'raspberry pi' in model.lower()
              and (machine.lower().startswith('arm') or machine.lower() == 'aarch64'))
    desktop = machine.lower() in {'x86_64', 'amd64', 'i386', 'i686'} or platform.system() in {'Windows', 'Darwin'}
    return dict(measurement_host='raspberry_pi_host' if pi else 'desktop_host' if desktop else 'non_pi_host',
        physical_pi_measured=pi, device_model=model, platform=platform.platform(), machine=machine,
        processor=platform.processor(), python=platform.python_version(), cpu_count=os.cpu_count())


def memory_usage():
    current, peak, method = None, None, 'unavailable'
    try:
        values = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, amount, _unit = line.split()
                values[key] = int(amount) / 1024
        current, peak = values.get('VmRSS:'), values.get('VmHWM:')
        method = '/proc/self/status (KiB converted to MiB)'
    except (OSError, ValueError):
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
            method = 'getrusage lifetime peak RSS'
        except ImportError:
            if os.name == 'nt':
                try:
                    import ctypes
                    from ctypes import wintypes
                    class Counters(ctypes.Structure):
                        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                            *[(key, ctypes.c_size_t) for key in ('PeakWorkingSetSize', 'WorkingSetSize',
                              'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                              'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
                    kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
                    kernel.GetCurrentProcess.restype = ctypes.c_void_p
                    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
                    info = Counters(); info.cb = ctypes.sizeof(info)
                    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
                        current, peak = info.WorkingSetSize / 1024**2, info.PeakWorkingSetSize / 1024**2
                        method = 'GetProcessMemoryInfo working set'
                except (OSError, AttributeError):
                    pass
    return dict(rss_mib=current, peak_rss_mib=peak, method=method)


def telemetry(*, pi=False):
    values = {}
    for field, name, scale in [('temperature_c', '/sys/class/thermal/thermal_zone0/temp', 1000),
                               ('cpu0_mhz', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 1000)]:
        try:
            number = float(Path(name).read_text()) / scale
            values[field] = number if math.isfinite(number) else None
        except (OSError, ValueError):
            values[field] = None
    values['throttled'] = None
    tool = shutil.which('vcgencmd') if pi else None
    if tool:
        try:
            result = subprocess.run([tool, 'get_throttled'], capture_output=True, text=True, timeout=2, check=False)
            if result.returncode == 0:
                values['throttled'] = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return values


def quantiles(values):
    return dict(n=len(values), p50=float(np.percentile(values, 50)) if values else None,
                p95=float(np.percentile(values, 95)) if values else None,
                mean=float(np.mean(values)) if values else None)


def accuracy(rows, key):
    references = [row for row in rows if row['scored']]
    measured = [row for row in references if key != 'raw' or row['raw_available']]
    positives = [row for row in measured if row['expected'] is not None]
    negatives = [row for row in measured if row['expected'] is None]
    exact = sum(row[key] is not None and canonical(row[key]) == canonical(row['expected']) for row in positives)
    emitted = sum(row[key] is not None for row in measured)
    false = sum(row[key] is not None for row in negatives)
    return dict(reference_n=len(references), scored_n=len(measured), unavailable_n=len(references)-len(measured),
        positive_n=len(positives), negative_n=len(negatives), exact_positive_n=exact, emitted_n=emitted,
        negative_abstention_n=len(negatives)-false, false_accept_n=false,
        complete_command_recall=exact/len(positives) if positives else None,
        precision=exact/emitted if emitted else None, false_accept_rate=false/len(negatives) if negatives else None)


def _trial(reference, event, repeat, pcm, elapsed, cpu):
    from scripts.runtime.pcv_schema import validate_command
    _validate_output(event['output'])
    diagnostic = event.get('diagnostic_candidate')
    available = isinstance(diagnostic, dict) and 'candidate' in diagnostic
    raw = diagnostic['candidate'] if available else None
    if raw is not None:
        validate_command(raw)
    front = event.get('frontend_ms', event.get('timing', {}).get('frontend_ms'))
    if front is not None and (type(front) not in (int, float) or not math.isfinite(front) or front < 0 or front > elapsed + .1):
        raise ValueError('Runtime reported an invalid frontend timing')
    return dict(id=reference['id'], repeat=repeat, scored=reference['scored'], expected=reference['expected'],
        expected_output=reference['expected_output'], provenance=reference['provenance'],
        audio=reference['audio'], sha256=reference['sha256'], pcm_sha256=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
        sample_rate=SAMPLE_RATE, audio_seconds=len(pcm)/SAMPLE_RATE, raw_available=available, raw=raw,
        accepted=event['output']['command'], output=event['output'], actions_enabled=False,
        timing=dict(end_to_end_ms=elapsed, frontend_ms=front,
                    after_frontend_ms=elapsed-front if front is not None else None,
                    rtf=elapsed/1000/(len(pcm)/SAMPLE_RATE), cpu_ms=cpu))


def benchmark(bundle, manifest, output, *, threads=1, engine='ncnn', repeats=3, root=None, runtime_factory=None):
    if type(threads) is not int or not 1 <= threads <= 8 or type(repeats) is not int or repeats < 1:
        raise ValueError('Use threads=1..8 and a positive integer repeat count')
    if engine not in {'ncnn', 'onnxruntime'}:
        raise ValueError('Engine must be ncnn or onnxruntime')
    bundle, manifest, folder = Path(bundle).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if folder.exists():
        raise FileExistsError('Benchmark output already exists; choose a new --output directory: ' + str(folder))
    preflight = time.perf_counter()
    manifest_hash = digest(manifest)
    references = load_manifest(manifest, Path(root) if root is not None else ROOT)
    if digest(manifest) != manifest_hash:
        raise ValueError('Manifest changed during preflight')
    preflight_ms = (time.perf_counter()-preflight)*1000
    if folder == bundle or folder.is_relative_to(bundle):
        raise ValueError('Benchmark output must be outside the immutable model bundle')
    if runtime_factory is None:
        from scripts.runtime.mvp_static import Runtime
        runtime_factory = Runtime
    host = hardware()
    resources = [dict(stage='before_load', **memory_usage(), **telemetry(pi=host['physical_pi_measured']))]
    started = time.perf_counter()
    runtime = runtime_factory(bundle, threads=threads, engine=engine, mode='demo', profile='benchmark-compatible')
    load_ms = (time.perf_counter()-started)*1000
    resources.append(dict(stage='after_load', **memory_usage(), **telemetry(pi=host['physical_pi_measured'])))
    folder.mkdir(parents=True, exist_ok=False)
    observations = []
    with (folder/'trials.jsonl').open('x', encoding='utf-8') as stream:
        for index, reference in enumerate(references):
            pcm = read_pcm(reference['path'], reference['sha256'])
            for repeat in range(repeats):
                cpu, wall = time.process_time(), time.perf_counter()
                event = runtime.recognize(pcm, sample_rate=SAMPLE_RATE)
                elapsed = (time.perf_counter()-wall)*1000
                used_cpu = (time.process_time()-cpu)*1000
                row = _trial(reference, event, repeat, pcm, elapsed, used_cpu)
                observations.append(row)
                stream.write(canonical(row)+'\n')
            if index % max(1, len(references)//20) == 0:
                resources.append(dict(stage='replay', completed_sources=index+1, **memory_usage(),
                                      **telemetry(pi=host['physical_pi_measured'])))
    sources = [row for row in observations if row['repeat'] == 0]
    signatures = {}
    for row in observations:
        signatures.setdefault(row['id'], set()).add(canonical(row['output']))
    if digest(manifest) != manifest_hash:
        raise ValueError('Manifest changed during replay; partial trials are preserved in ' + str(folder))
    from scripts.runtime.pcv_schema import INTENTS
    per_intent = {}
    for intent in [*INTENTS, 'UNKNOWN']:
        group = [row for row in sources if (row['expected'] or {}).get('intent', 'UNKNOWN') == intent]
        per_intent[intent] = {key: accuracy(group, key) for key in ('raw', 'accepted')}
        for key in ('raw', 'accepted'):
            predicted = sum(row['scored'] and (key != 'raw' or row['raw_available']) and row[key] is not None
                            and row[key]['intent'] == intent for row in sources)
            per_intent[intent][key].update(predicted_n=predicted,
                precision=per_intent[intent][key]['exact_positive_n']/predicted if predicted else None)
    latency = {name: quantiles([row['timing'][field] for row in observations if row['timing'][field] is not None])
               for name, field in [('end_to_end', 'end_to_end_ms'), ('frontend', 'frontend_ms'), ('after_frontend', 'after_frontend_ms')]}
    wall = sum(row['timing']['end_to_end_ms'] for row in observations)
    cpu = sum(row['timing']['cpu_ms'] for row in observations)
    resources.append(dict(stage='after_replay', **memory_usage(), **telemetry(pi=host['physical_pi_measured'])))
    active_files = ['model.onnx'] if engine == 'onnxruntime' else ['ncnn/model.ncnn.param', 'ncnn/model.ncnn.bin', 'ncnn/position-frequency.json']
    result = dict(schema_version=1, utc=datetime.now(timezone.utc).isoformat(), evidence_kind='command_only_recorded_replay',
        hardware=host, physical_pi_measured=host['physical_pi_measured'], physical_microphone_measured=False,
        actions_enabled=False, profile='benchmark-compatible', owner_gating_enabled=False, wake_tested=False, qualified=False,
        model_scope='Direct VCM session only; wake/VAD, owner sessions, UI and action providers are excluded.',
        bundle=str(bundle), bundle_sha256=digest(bundle/'bundle.json'), model_sha256=runtime.meta['files']['model.onnx'],
        engine_model_sha256=runtime.engine_model_sha256, engine=engine, threads=threads, sample_rate=SAMPLE_RATE,
        manifest=str(manifest), manifest_sha256=manifest_hash, preflight_ms=preflight_ms,
        repeats=repeats, source_n=len(sources), excluded_reference_n=sum(not row['scored'] for row in sources),
        observation_n=len(observations), accuracy_unit='First measured pass per source recording; repeats do not increase accuracy denominators.',
        raw=accuracy(sources, 'raw'), accepted=accuracy(sources, 'accepted'), per_intent=per_intent,
        capture_fault_source_n=sum(row['output']['reason']=='capture_fault' for row in sources),
        typed_output=dict(n=sum(row['expected_output'] is not None and row['scored'] for row in sources),
            exact_n=sum(row['expected_output'] is not None and row['scored'] and canonical(row['output'])==canonical(row['expected_output']) for row in sources)),
        unstable_output_source_n=sum(len(values)>1 for values in signatures.values()),
        latency_ms=latency, rtf=quantiles([row['timing']['rtf'] for row in observations]),
        timing_includes=['PCM validation/conversion', 'normalization/frontend', 'CPU engine inference', 'finite decision and typed output'],
        timing_excludes=['file reads and hashes', 'audio decoding', 'utterance duration', 'wake/VAD and endpoint waits', 'telemetry and report writing'],
        timing_components='after_frontend includes engine inference, finite decision and wrapper overhead; no unmeasured model/decision split is invented.',
        first_inference_included=True, session_load_ms=load_ms, runtime_reported_load_ms=getattr(runtime, 'load_ms', None),
        active_engine_bytes=sum((bundle/name).stat().st_size for name in active_files),
        all_bundle_file_bytes=sum(path.stat().st_size for path in bundle.rglob('*') if path.is_file()),
        policy_enabled=bool(runtime.policy.get('enabled', True)), policy_sha256=runtime.meta['files']['policy.json'],
        runtime_cpu_seconds=cpu/1000, runtime_wall_seconds=wall/1000,
        cpu_percent_one_core=100*cpu/wall if wall else None,
        peak_process_rss_mib=max((sample['peak_rss_mib'] for sample in resources if sample['peak_rss_mib'] is not None), default=None),
        resource_scope='Whole benchmark Python process, including libraries, result storage and runtime; lifetime RSS peak is not model-only memory.',
        resource_samples=resources, software_versions={},
        engine_threads=threads, frontend_thread_environment={key:os.environ.get(key) for key in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
        thread_scope='--threads controls the inference engine; unset frontend BLAS variables mean library defaults.',
        benchmark_source_sha256=digest(Path(__file__)),
        runtime_source_sha256=digest(Path(__file__).with_name('mvp_static.py')))
    from importlib.metadata import PackageNotFoundError, version
    for package in ('numpy', 'soundfile', engine):
        try:
            result['software_versions'][package] = version(package)
        except PackageNotFoundError:
            result['software_versions'][package] = 'not_available'
    (folder/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main(argv=None, *, root=None):
    root = Path(root or ROOT).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Default: the installed bundle pointer under this installation')
    parser.add_argument('--manifest', type=Path, help='Hash-bound JSONL; default samples/commands.jsonl')
    parser.add_argument('--output', type=Path, help='New result directory; default .runtime/benchmarks/iris-UTC')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--engine', choices=['ncnn', 'onnxruntime'], default='ncnn')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    manifest = args.manifest or root/'samples/commands.jsonl'
    if not manifest.is_file():
        parser.error(f'Missing replay manifest: {manifest}. Supply --manifest PATH to a JSONL file with '
                     'id, audio, sha256 and explicit expected:{intent,slots} or expected:null; no audio is downloaded.')
    from scripts.runtime.speaker_package import installed_bundle
    bundle = args.bundle or installed_bundle(root)
    output = args.output or root/'.runtime/benchmarks'/datetime.now(timezone.utc).strftime('iris-%Y%m%dT%H%M%S.%fZ')
    try:
        result = benchmark(bundle, manifest, output, threads=args.threads, engine=args.engine, repeats=args.repeats, root=root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(output), measurement_host=result['hardware']['measurement_host'],
        physical_pi_measured=result['physical_pi_measured'], actions_enabled=False, source_n=result['source_n'],
        accepted=result['accepted'], latency_ms=result['latency_ms']['end_to_end']), allow_nan=False))
    return result


if __name__ == '__main__':
    main()
