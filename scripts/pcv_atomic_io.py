"""Bounded Windows atomic-replace retries; accounting/gates remain authoritative."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def replace_with_retry(source,target,attempts=20):
    if not 1<=attempts<=20:raise ValueError('Retry count must be 1..20')
    for attempt in range(attempts):
        try:
            os.replace(source,target)
            if attempt:print('ATOMIC_REPLACE_RETRIES',Path(target).name,attempt,flush=True)
            return
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt+1==attempts:raise
            time.sleep(.05)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.pending')
    if temporary.exists():raise FileExistsError('Preserve orphan receipt: '+str(temporary))
    path.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    replace_with_retry(temporary,path)


@contextmanager
def reliable_budget_io(recovery):
    from pcvcm import compute
    original=compute.ComputeBudget;old_json=recovery.atomic_json
    class ReliableComputeBudget(original):
        def _save(self):
            self.state['seconds']=self.base_seconds+time.monotonic()-self.started
            temporary=self.path.with_suffix('.pending')
            compute.write_json(temporary,self.state)
            replace_with_retry(temporary,self.path)
        def check(self):
            self._save()
            # Include retry time when enforcing the existing wall-time limit.
            if self.base_seconds+time.monotonic()-self.started>=self.limits['seconds']:
                raise RuntimeError('Global compute budget exhausted (time)')
    try:
        compute.ComputeBudget=ReliableComputeBudget;recovery.atomic_json=atomic_json
        yield
    finally:compute.ComputeBudget=original;recovery.atomic_json=old_json
