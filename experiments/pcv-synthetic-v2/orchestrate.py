"""Guarded v2 continuation; starts only after the approved controller handoff."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import revision as rev
from corpus import generation_complete


def require_transition():
    path=rev.CACHE/'controller-transition-complete.json'
    if not path.exists():raise RuntimeError('Awaiting explicit approved replacement of the old controller; no duplicate job launched')
    transition=json.loads(path.read_text(encoding='utf-8-sig'))
    if transition.get('controller_stopped') is not True or transition.get('user_authorized') is not True:
        raise ValueError('Missing approved controller transition receipt')
    return transition


def original_ready(engine,pid,alive=None):
    if generation_complete(rev.OLD_CACHE,engine,13860):return True
    if pid is not None:
        if alive is None:
            import psutil
            alive=psutil.pid_exists
        if not alive(pid):
            # Re-read after observing exit, covering the final-receipt write race.
            if generation_complete(rev.OLD_CACHE,engine,13860):return True
            raise RuntimeError(f'Original {engine} generator exited without a complete receipt; inspect its preserved log')
    return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True);args=parser.parse_args()
    transition=require_transition();rev.validate();job=args.job.resolve()
    old_pids=transition.get('old_generators',{})
    if any(type(old_pids.get(e)) is not int for e in ('qwen','parler')):raise ValueError('Original generator PIDs missing from handoff')
    if not job.is_relative_to(rev.CACHE.resolve()):raise ValueError('Job escapes v2 cache')
    job.mkdir(parents=True,exist_ok=False)
    sources={str(p.relative_to(rev.ROOT)):rev.file_hash(p) for p in rev.HERE.glob('*.py')}
    state=dict(status='RUNNING',pid=os.getpid(),steps=[],source_hashes=sources,protocol_sha256=rev.file_hash(rev.HERE/'protocol.json'))
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=job/'status.pending';rev.write_json(temporary,state);temporary.replace(job/'status.json')
    def launch(name,engine=None):
        rev.validate()
        if any(rev.file_hash(rev.ROOT/p)!=h for p,h in sources.items()):raise ValueError('Executing v2 source changed')
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        if engine=='parler':env['PYTHONPATH']=str(rev.OLD_CACHE/'parler-site')+os.pathsep+env.get('PYTHONPATH','')
        command=[sys.executable,'-u',str(rev.HERE/'run.py'),name]+(['--engine',engine] if engine else [])
        label=f'{name}-{engine}' if engine else name;log=(job/f'{label}.log').open('x',encoding='utf-8')
        process=subprocess.Popen(command,cwd=rev.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        item=dict(name=label,pid=process.pid,status='RUNNING',command=command,log=str(job/f'{label}.log'),started=time.time())
        state['steps'].append(item);save();return process,log,item
    def finish(item):
        process,log,row=item;code=process.wait();log.close()
        row.update(status='SUCCEEDED' if code==0 else 'FAILED',exit_code=code,seconds=time.time()-row['started']);save()
        if code:raise RuntimeError(f'{row["name"]} failed; evidence preserved')
    save()
    try:
        pending={'qwen','parler'};running={}
        state['status']='WAITING_FOR_EXISTING_QWEN_PARLER';save()
        while pending or running:
            for engine in sorted(pending.copy()):
                if original_ready(engine,old_pids[engine]):
                    running[engine]=launch('generate',engine);pending.remove(engine)
            for engine,item in list(running.items()):
                if item[0].poll() is not None:finish(item);del running[engine]
            if pending or running:time.sleep(10)
        state['status']='RUNNING';save();finish(launch('generate','dia'))
        for stage in ('assemble','prepare','train','export','parity','calibrate','wake-dev','wake-calibration','evaluate','wake-evaluate','report'):
            finish(launch(stage))
        result=json.loads((rev.CACHE/'report/finalization.json').read_text())
        state.update(status='COMPLETE',synthetic_qualified=result['synthetic_qualified'],production_ready=False)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));raise
    finally:save()


if __name__=='__main__':main()
