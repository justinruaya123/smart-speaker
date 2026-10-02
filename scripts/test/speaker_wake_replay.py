"""Construct explicitly synthetic wake/silence fixtures; never mutate research audio."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from einops import pack
import soundfile as sf

from scripts.runtime.pcv_final_runtime import Runtime
from scripts.runtime.speaker_worker import AudioService


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[2];source=root/'.cache/pcv-synthetic-v2'
    rows=[json.loads(line) for line in (source/'wake-parler-manifest.jsonl').read_text().splitlines()]
    protocol=json.loads((root/'experiments/pcv-synthetic-v2/protocol.json').read_text())
    voices={v['id']:v for v in protocol['voices']}
    args.output.mkdir(parents=True,exist_ok=False)
    runtime=Runtime(args.bundle,diagnostic=True,threads=1);cases=[]
    for voice in ('test-gb-m','test-in-f'):
        reference=next(r for r in rows if r['voice_group']==voice and r['verbatim_id']=='WAKE-ONLY')
        path=Path(reference['audio']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=reference['audio_sha256']:raise ValueError('Source wake audio identity changed')
        pcm,rate=sf.read(path,dtype='float32')
        if rate!=16000 or pcm.ndim!=1:raise ValueError('Unexpected original format')
        # Two seconds for detector warm-up; eight seconds after wake includes command timeout.
        padded,_=pack([np.zeros(32000,dtype='float32'),pcm,np.zeros(128000,dtype='float32')],'*')
        output=args.output/(voice+'-silence-wake-silence.wav');sf.write(output,padded,16000,subtype='PCM_16')
        events=[];service=AudioService(runtime,events.append);result=service.wav(output.read_bytes(),wake=True)
        cases.append({'id':voice,'gender_requested':voices[voice]['gender'],'voice':voices[voice]['parler'],
            'synthetic':True,'text_verified_by_listener':False,'source':reference['recording_id'],
            'source_sha256':reference['audio_sha256'],'source_manifest':str((source/'wake-parler-manifest.jsonl').relative_to(root)),
            'audio':output.name,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'leading_silence_seconds':2,'trailing_silence_seconds':8,'expected_wake_detected':True,
            'expected_accepted_commands':0,'observed':result,'wake_passed':result['wake_detected'],
            'no_false_command':not any(e.get('output',{}).get('accepted') for e in events),'events':events})
    silent=args.output/'dead-air.wav';sf.write(silent,np.zeros(160000,dtype='float32'),16000,subtype='PCM_16')
    events=[];result=AudioService(runtime,events.append).wav(silent.read_bytes(),wake=True)
    report={'scope':'Constructed synthetic male/female voice checks; no human/live-mic/Pi qualification',
        'model_sha256':runtime.recipe['binding']['model_sha256'],'policy_enabled':runtime.policy['enabled'],
        'cases':cases,'dead_air':{'observed':result,'passed':not result['wake_detected'] and result['commands']==0},
        'passed':all(c['wake_passed'] and c['no_false_command'] for c in cases) and not result['wake_detected']}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in report.items() if k!='cases'},'cases':[{k:c[k] for k in ('id','gender_requested','observed','wake_passed','no_false_command')} for c in cases]},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
