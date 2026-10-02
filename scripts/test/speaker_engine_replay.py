"""Recorded engine comparison and repeat stability; no microphone or actions."""
import argparse
import json
from pathlib import Path
from scripts.runtime.pcv_final_runtime import Runtime,read_wav
from scripts.runtime.speaker_confidence import application_decision
from scripts.runtime.speaker_config import DEFAULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();observations={};rows=[]
    samples=sorted((args.package/'samples').glob('*.wav'))
    if not samples:raise ValueError('No recorded smoke samples')
    for engine in ('ncnn','onnxruntime'):
        runtime=Runtime(args.package/'models/default',diagnostic=True,engine=engine)
        baseline={}
        for repeat in range(3):
            for path in (samples if repeat%2==0 else list(reversed(samples))):
                event=runtime.recognize(read_wav(path));decision=application_decision(event,DEFAULTS['inference'])
                record={'candidate':event['diagnostic_candidate'],'application':decision,'output':event['output']}
                if repeat==0:baseline[path.name]=record
                elif baseline[path.name]!=record:raise AssertionError(f'Unstable inference: {engine} {path.name} repeat {repeat}')
        observations[engine]=baseline
    for path in samples:rows.append({'file':path.name,**{engine:values[path.name] for engine,values in observations.items()}})
    result={'passed':True,'samples':len(samples),'repeats':3,'inferences':len(samples)*6,'exact_repeat_stability':True,
            'candidate_agreement':sum(r['ncnn']['candidate']['candidate']==r['onnxruntime']['candidate']['candidate'] for r in rows),
            'rows':rows,'physical_pi':False,'live_microphone':False}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':main()
