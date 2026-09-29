"""Measure actual host cost. Desktop timings are not Raspberry Pi evidence."""
import argparse
import json
import platform
from pathlib import Path
import time
import numpy as np
from vcm.integrity import save, digest


def benchmark(bundle,manifest,output,threads=(1,2,4),limit=20):
    from vcm.runtime import Runtime
    from vcm.tui import read_wav
    from vcm.system import memory_usage,cpu_seconds
    output=Path(output)
    if output.exists(): raise FileExistsError('Benchmark output exists')
    rows=[json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()][:limit]
    if not rows: raise ValueError('Benchmark requires recordings')
    reports=[]
    for count in threads:
        start=time.perf_counter(); runtime=Runtime(bundle,count); load_ms=(time.perf_counter()-start)*1000
        observations=[]
        for r in rows:
            if digest(r['audio'])!=r['audio_sha256']: raise ValueError('Benchmark PCM source changed')
            pcm=read_wav(r['audio']); runtime.predict(pcm)
            for _ in range(3):
                cpu=cpu_seconds(); start=time.perf_counter(); runtime.predict(pcm); elapsed=time.perf_counter()-start
                observations.append({'id':r['id'],'duration_seconds':len(pcm)/16000,'processing_ms':elapsed*1000,
                                     'rtf':elapsed/(len(pcm)/16000),'cpu_percent_one_core':100*(cpu_seconds()-cpu)/elapsed,
                                     **memory_usage(),**runtime.timing})
        reports.append({'threads':count,'load_ms':load_ms,'observations':observations,
                        'p50_ms':float(np.percentile([x['processing_ms'] for x in observations],50)),
                        'p95_ms':float(np.percentile([x['processing_ms'] for x in observations],95))})
        del runtime
    result={'host':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
            'manifest_sha256':digest(manifest),'recipe_sha256':digest(Path(bundle)/'recipe.json'), 'runs':reports,
            'timing_excludes':['disk/audio decoding','utterance duration','human endpoint waiting'],
            'physical_microphone_tested':False,'temperature_throttling':'not measured','sustained_far_field':'not measured',
            'physical_pi_evidence':platform.machine().lower() in {'aarch64','arm64'} and Path('/proc/device-tree/model').exists()}
    save(output/'benchmark.json',result); return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--bundle',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--threads',type=int,nargs='+',default=[1,2,4]); p.add_argument('--limit',type=int,default=20)
    a=p.parse_args(); benchmark(a.bundle,a.manifest,a.output,a.threads,a.limit)
