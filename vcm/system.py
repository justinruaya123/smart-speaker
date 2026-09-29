"""Small stdlib-only host memory/CPU measurement helpers."""
import ctypes
import os
from pathlib import Path


def memory_usage():
    if os.name=='nt':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
                ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        counters=Counters(); counters.cb=ctypes.sizeof(counters)
        handle=ctypes.windll.kernel32.GetCurrentProcess
        handle.restype=wintypes.HANDLE
        query=ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        if not query(handle(),ctypes.byref(counters),counters.cb): raise OSError('Process memory query failed')
        return {'rss_bytes':int(counters.WorkingSetSize),'peak_rss_bytes':int(counters.PeakWorkingSetSize)}
    values={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/self/status').read_text().splitlines() if ':' in line}
    return {'rss_bytes':int(values['VmRSS'].split()[0])*1024,'peak_rss_bytes':int(values['VmHWM'].split()[0])*1024}


def cpu_seconds():
    value=os.times(); return value.user+value.system
