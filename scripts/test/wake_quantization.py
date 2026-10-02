"""Development-only dense INT8 experiment; never automatically replaces artifacts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from train import register_sources,save_json,digest
register_sources()
import copy,json,time
from collections import defaultdict
import numpy as np
from onnxruntime.quantization import quantize_dynamic,QuantType
from tinyvcm.inference import Recognizer
from tinyvcm.audio import load_audio


def main():
    output=ROOT/'.cache/wake-pipeline-v1/dense-int8'
    output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    with (ROOT/'data/vcm.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['split']=='dev':groups[row['command']].append(row)
    rows=[row for key in sorted(groups) for row in sorted(groups[key],key=lambda r:r['id'])[:15]]
    ids={r['id'] for r in rows};refs={}
    with (ROOT/'data/references.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['id'] in ids:refs[row['id']]=row.get('reference')
    protocol={'experiment':'wake-pipeline-v1/dense-int8','data':'First 15 sorted dev IDs per family; historical test not used.',
              'ids':[r['id'] for r in rows],'source_sha256':digest(ROOT/'artifacts/vcm.fp32.onnx'),
              'quantized_ops':['MatMul','Gemm'],'criteria':'No lost correct command or added wrong accept, at least 10 percent smaller graph and 10 percent faster p95 on this development workload. Any promotion also requires a separate regression run.',
              'thresholds_retuned':False,'automatic_install':False}
    save_json(ROOT/'experiments/wake-pipeline-v1/quantization-protocol.json',protocol)
    graph=output/'vcm.fp32.onnx'
    quantize_dynamic(str(ROOT/'artifacts/vcm.fp32.onnx'),str(graph),weight_type=QuantType.QInt8,
                     per_channel=True,op_types_to_quantize=['MatMul','Gemm'])
    metadata=json.loads((ROOT/'artifacts/bundle.json').read_text())
    metadata['files']={'vcm.fp32.onnx':{'sha256':digest(graph),'bytes':graph.stat().st_size}}
    save_json(output/'bundle.json',metadata)
    models={'fp32':Recognizer(ROOT/'artifacts',threads=2),'dense_int8':Recognizer(output,threads=2)}
    counters={k:{'valid':0,'correct':0,'accepted':0,'wrong_accepts':0} for k in models}
    timings={k:[] for k in models};differences=[]
    for i,row in enumerate(rows):
        pcm,_=load_audio(ROOT/row['audio']);reference=refs.get(row['id'])
        valid=reference and reference.get('status')=='accepted' and not (set(reference['command']['arguments'])&{'title','artist','selection_text'})
        results={}
        for name in (list(models) if i%2==0 else list(reversed(models))):
            model=models[name];model.predict(pcm,stream=False)
            measurements=[]
            for _ in range(3):
                t=time.perf_counter();result=model.predict(pcm,stream=False);measurements.append((time.perf_counter()-t)*1000)
            timings[name].append(float(np.median(measurements)))
            accepted=result.get('status')=='accepted' and 'command' in result
            correct=bool(valid and accepted and result.get('command')==reference['command'] and result.get('vcm_command')==row['command'])
            counters[name]['valid']+=bool(valid);counters[name]['correct']+=correct
            counters[name]['accepted']+=bool(accepted);counters[name]['wrong_accepts']+=bool(accepted and not correct)
            results[name]={'correct':correct,'wrong_accept':bool(accepted and not correct),'status':result.get('status'),'command':result.get('command')}
        if results['fp32']!=results['dense_int8']:differences.append({'id':row['id'],**results})
    summaries={name:{**counters[name],'p50_ms':float(np.percentile(timings[name],50)),'p95_ms':float(np.percentile(timings[name],95))} for name in models}
    base=ROOT/'artifacts/vcm.fp32.onnx'
    lost=sum(x['fp32']['correct'] and not x['dense_int8']['correct'] for x in differences)
    added=sum(x['dense_int8']['wrong_accept'] and not x['fp32']['wrong_accept'] for x in differences)
    passes=not lost and not added and graph.stat().st_size<=.9*base.stat().st_size and summaries['dense_int8']['p95_ms']<=.9*summaries['fp32']['p95_ms']
    report={'scope':'Bounded balanced development comparison, desktop CPU, no Pi and no test tuning. Timing excludes endpoint/capture.',
            'examples':len(rows),'results':summaries,'fp32_bytes':base.stat().st_size,'dense_int8_bytes':graph.stat().st_size,
            'lost_correct':lost,'added_wrong_accepts':added,'differences':differences,'criteria_passed':passes,
            'installed':False,'decision':'Retain FP32 unless all predeclared improvement criteria pass and a separate regression is completed.'}
    save_json(ROOT/'experiments/wake-pipeline-v1/quantization-results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},indent=2))


if __name__=='__main__':main()
