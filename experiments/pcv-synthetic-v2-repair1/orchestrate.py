"""Single-use child controller; never restarts or overwrites the failed parent."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import workflow as repair


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--job',type=Path,required=True)
    job = parser.parse_args().job.resolve(); repair.validate()
    if not job.is_relative_to(repair.CACHE.resolve()): raise ValueError('Job escapes child cache')
    # One controller ever owns these immutable outputs, even after a failure.
    repair.CACHE.mkdir(parents=True,exist_ok=True)
    lock = repair.CACHE/'controller-started.json'
    with lock.open('x',encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(),job=str(job)),stream); stream.flush(); os.fsync(stream.fileno())
    job.mkdir(parents=True,exist_ok=False)
    state = dict(status='RUNNING',pid=os.getpid(),steps=[],
        protocol_sha256=repair.file_hash(repair.HERE/'protocol.json'),
        parent_failure=str(repair.FAILED_JOB))
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        pending = job/'status.pending'
        repair.rev.write_json(pending,state); pending.replace(job/'status.json')
    save()
    try:
        for stage in ('generate-repair','assemble','prepare','train','export','parity','calibrate',
                'wake-dev','wake-calibration','evaluate','wake-evaluate','report'):
            repair.validate()
            env = os.environ.copy(); env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
            if stage == 'generate-repair':
                env['PYTHONPATH'] = str(repair.rev.OLD_CACHE/'parler-site')+os.pathsep+env.get('PYTHONPATH','')
            command = [sys.executable,'-u',str(repair.HERE/'run.py'),stage]
            with (job/f'{stage}.log').open('x',encoding='utf-8') as log:
                process = subprocess.Popen(command,cwd=repair.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                row = dict(name=stage,pid=process.pid,command=command,log=str(job/f'{stage}.log'),
                    started=time.time(),status='RUNNING')
                state['steps'].append(row); save()
                code = process.wait()
            row.update(exit_code=code,seconds=time.time()-row['started'],status='SUCCEEDED' if code==0 else 'FAILED'); save()
            if code: raise RuntimeError(f'{stage} failed; evidence preserved')
        final = json.loads((repair.CACHE/'report/finalization.json').read_text())
        state.update(status='COMPLETE',synthetic_qualified=final['synthetic_qualified'],production_ready=False)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc)); raise
    finally: save()


if __name__ == '__main__': main()
