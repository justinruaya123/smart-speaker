"""Bounded offline resource probe of the shipped CPU runtime, without microphone access."""
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np


def peak_rss_mib():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                *[(name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                  'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                  'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]]
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        value = Counters(); value.cb = ctypes.sizeof(value)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise OSError('Process memory measurement failed')
        return value.PeakWorkingSetSize / (1024 ** 2)
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def benchmark(bundle, config, sample_directory):
    from .pipeline import Pipeline
    from .wake_models import HeyJarvis
    from .audio import load_audio
    config = copy.deepcopy(config)
    config['validation']['threads'] = 2
    if not all(config['runtime'].values()) or config['audio']['vad_backend'] != 'silero':
        raise ValueError('Benchmark requires the complete VCM/Silero pipeline')
    sample_directory = Path(sample_directory).resolve()
    rows = []
    with (sample_directory / 'samples.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            rows.append(json.loads(line))
            if len(rows) == 20: break
    if not rows: raise ValueError('Benchmark requires packaged audio samples')
    start = time.perf_counter()
    pipeline = Pipeline(bundle, config)
    wake = HeyJarvis(bundle)
    load_ms = (time.perf_counter() - start) * 1000
    timings, durations, replay_times, replay_rtfs = [], [], [], []
    for repeat in range(4):
        for row in rows:
            path = (sample_directory / row['audio']).resolve()
            if not path.is_relative_to(sample_directory): raise ValueError('Invalid benchmark sample path')
            pcm, _ = load_audio(path)
            if len(pcm) > 16000 * 15: raise ValueError('Benchmark sample exceeds 15-second bound')
            start = time.perf_counter(); pipeline.recognizer.predict(pcm, stream=False)
            elapsed = (time.perf_counter() - start) * 1000
            start = time.perf_counter(); pipeline.process(pcm)
            replay_ms = (time.perf_counter() - start) * 1000
            if repeat:
                timings.append(elapsed); durations.append(len(pcm) / 16000)
                replay_times.append(replay_ms); replay_rtfs.append(replay_ms / 1000 / (len(pcm) / 16000))
    wake_times, vad_times = [], []
    wake_frame, vad_frame = np.zeros(1280, dtype=np.float32), np.zeros(512, dtype=np.float32)
    for index in range(80):
        start = time.perf_counter(); wake.push(wake_frame)
        wake_elapsed = (time.perf_counter() - start) * 1000
        start = time.perf_counter(); pipeline.vad.model(vad_frame)
        vad_elapsed = (time.perf_counter() - start) * 1000
        if index >= 20: wake_times.append(wake_elapsed); vad_times.append(vad_elapsed)
    def stats(values):
        return {'observations': len(values), 'p50_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95))}
    model_path = Path('/proc/device-tree/model')
    hardware = model_path.read_text(encoding='utf-8').strip('\x00') if model_path.exists() else platform.platform()
    total_ram = None
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if line.startswith('MemTotal:'): total_ram = int(line.split()[1]) / 1024
    memory = peak_rss_mib()
    recognition, wake_stats, vad_stats = stats(timings), stats(wake_times), stats(vad_times)
    frame_compute_fraction = wake_stats['p95_ms'] / 80 + vad_stats['p95_ms'] / 32
    passed = memory <= 512 and recognition['p95_ms'] <= 100 and frame_compute_fraction < 1 and max(replay_rtfs) < 1
    return {'hardware': hardware, 'machine': platform.machine(), 'python': platform.python_version(),
        'physical_pi_measurement': hardware.startswith('Raspberry Pi'), 'live_microphone_tested': False,
        'physical_ram_mib': total_ram,
        'target_pi4_2gb_arm64': hardware.startswith('Raspberry Pi 4') and platform.machine() == 'aarch64' and total_ram is not None and 1536 <= total_ram <= 2253,
        'threads': {'vcm': 2, 'wake_and_vad': 1}, 'all_sessions_load_ms': load_ms,
        'recognition_including_frontend': recognition, 'wake_80ms_frame': wake_stats, 'vad_32ms_frame': vad_stats,
        'endpointed_command_replay': {**stats(replay_times), 'p95_real_time_factor': float(np.quantile(replay_rtfs, .95)), 'max_real_time_factor': max(replay_rtfs)},
        'estimated_wake_plus_vad_compute_fraction_from_p95': frame_compute_fraction,
        'sample_duration_seconds': {'min': min(durations), 'max': max(durations)},
        'peak_process_rss_mib': memory, 'budgets': {'peak_rss_mib': 512, 'recognition_p95_ms': 100},
        'resource_budget_passed_on_this_host': passed,
        'torch_imported': 'torch' in sys.modules,
        'scope': 'Bounded prerecorded CPU benchmark with all neural sessions resident. Excludes disk loading from timed recognition, microphone, playback echo, utterance duration and endpoint wait. Silence frames measure compute, not wake/VAD accuracy. Budgets are proposed prototype targets.'}
