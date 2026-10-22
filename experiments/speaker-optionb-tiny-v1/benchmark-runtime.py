"""Fresh-process Windows CPU timing/RSS probe; no Torch and no device actions."""
import ctypes
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.runtime.speaker_optionb_tiny import Runtime
from scripts.test.speaker_demo_suite import read_json,read_jsonl,read_audio,write_new,file_digest


def memory_counters():
    if sys.platform!='win32':
        raise RuntimeError('This native memory probe is verified on Windows only')
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_uint32),('PageFaultCount',ctypes.c_uint32),
                  *[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
    counter=Counters();counter.cb=ctypes.sizeof(counter)
    kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
    function=ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_uint32]
    function.restype=ctypes.c_int
    if not function(kernel.GetCurrentProcess(),ctypes.byref(counter),counter.cb):
        raise ctypes.WinError()
    return dict(working_set_bytes=counter.WorkingSetSize,peak_working_set_bytes=counter.PeakWorkingSetSize,
                private_commit_bytes=counter.PagefileUsage,peak_private_commit_bytes=counter.PeakPagefileUsage)


def main():
    study=ROOT/'experiments/speaker-optionb-tiny-v1'
    protocol=read_json(study/'protocol.json');run=ROOT/protocol['outputs']['run']
    before=memory_counters();runtime=Runtime(run/'bundle',threads=1)
    samples=[r for r in read_jsonl(run/'dev.jsonl') if r['kind']=='human' and r['metric_eligible']]
    times=[]
    for row in samples:
        event=runtime.recognize(read_audio(row))
        times.append(event['latency_ms'])
        assert event['actions_enabled'] is False
    assert 'torch' not in sys.modules
    receipt=dict(platform=platform.platform(),python=platform.python_version(),logical_cpu_count=os.cpu_count(),
        threads=1,blas_threads=1,measured_original_human_dev_utterances=len(times),
        p50_ms=float(np.percentile(times,50)),p95_ms=float(np.percentile(times,95)),session_load_ms=runtime.load_ms,
        memory_before_model=before,memory_after_replay=memory_counters(),torch_imported=False,
        excludes=['audio file loading','utterance speaking time','wake/VAD/endpoint wait','UI/TTS/device providers'],
        physical_pi_evidence=False,model_sha256=file_digest(run/'bundle/model.onnx'))
    write_new(study/'desktop-resource-results.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
