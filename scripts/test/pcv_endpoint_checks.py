"""Verify endpoint-buffered PCM matches cached evaluation and neural chunks."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/phrase-contract-v1'))
import numpy as np
import onnxruntime as ort
import soundfile as sf
from einops import rearrange,pack
from pcvcm.decoder import PrefixCTC,rank_commands
from vcm.frontend import features
from scripts.train.pcv_public_data import CACHE,read_rows,save,sha
from scripts.train.pcv_calibration import CHILD,WORK
from scripts.test.pcv_public_eval import normalized
from scripts.test.pcv_public_qualification import verify_parent,verify_bindings
from scripts.runtime.pcv_schema import canonical


def main():
    protocol=json.loads((CHILD/'endpoint-check-protocol.json').read_text())
    assert sha(__file__)==protocol['source_sha256']
    assert sha(CACHE/'prepared/dev.jsonl')==protocol['manifest_sha256']
    target=CHILD/'endpoint-check-results.json'
    if target.exists():raise FileExistsError('Preserve previous receipt')
    verify_parent()
    rows={r['id']:r for r in read_rows(CACHE/'prepared/dev.jsonl')}
    results=[]
    for arm in ('human','mixed'):
        bundle=WORK/'evaluation'/arm/'bundle'
        recipe=json.loads((bundle/'recipe.json').read_text())
        norm=json.loads((bundle/'normalization.json').read_text())
        entries=json.loads((bundle/'catalog.json').read_text())['entries']
        verify_bindings(bundle,{'model.onnx':recipe['binding']['model_sha256'],
            'normalization.json':recipe['binding']['normalization_sha256'],'catalog.json':recipe['binding']['catalog_sha256']})
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        start=time.perf_counter()
        session=ort.InferenceSession(str(bundle/'model.onnx'),options,providers=['CPUExecutionProvider'])
        decoder=PrefixCTC([e['phones'] for e in entries])
        load_seconds=time.perf_counter()-start
        states=recipe['state_shapes']
        arm_rows=[]
        for rid in protocol['ids']:
            row=rows[rid]
            verify_bindings(ROOT,{row['audio']:row['audio_sha256'],row['features']:row['features_sha256']})
            pcm,rate=sf.read(row['audio'],dtype='float32');assert rate==16000 and pcm.ndim==1
            tick=time.perf_counter();x=features(pcm,norm['mean'],norm['std']);frontend=time.perf_counter()-tick
            feature_error=float(np.max(np.abs(x-normalized(row,norm))))
            feed={'features':rearrange(x,'c t -> 1 c t'),**{f'state_{i}':np.zeros(s,dtype='float32') for i,s in enumerate(states)}}
            tick=time.perf_counter();full=session.run(None,feed);onnx=time.perf_counter()-tick
            tick=time.perf_counter();full_scores=decoder.score(rearrange(full[0],'1 t c -> t c'));decode=time.perf_counter()-tick
            expected=rank_commands(full_scores,entries)
            saved=[np.zeros(s,dtype='float32') for s in states];parts=[[],[],[]]
            decoder.reset()
            for start in range(0,x.shape[1],7):
                feed={'features':rearrange(x[:,start:start+7],'c t -> 1 c t'),**{f'state_{i}':s for i,s in enumerate(saved)}}
                out=session.run(None,feed);saved=out[3:]
                for head,pieces in zip(out[:3],parts,strict=True):pieces.append(head)
                decoder.feed(rearrange(out[0],'1 t c -> t c'))
            heads=[pack(p,'b * c')[0] for p in parts]
            head_error=max(float(np.max(np.abs(a-b))) for a,b in zip(heads,full[:3],strict=True))
            state_error=max(float(np.max(np.abs(a-b))) for a,b in zip(saved,full[3:],strict=True))
            scores=decoder.scores();finite=np.isfinite(full_scores)
            score_error=float(np.max(np.abs(scores[finite]-full_scores[finite])))
            same=canonical(expected['command'])==canonical(rank_commands(scores,entries)['command'])
            arm_rows.append({'id':rid,'feature_error':feature_error,'head_error':head_error,'state_error':state_error,
                'score_error':score_error,'same_raw_command':same,'pass':max(feature_error,head_error,state_error,score_error)<2e-5 and same,
                'seconds':{'frontend':frontend,'onnx':onnx,'decoder':decode},'audio_seconds':len(pcm)/rate})
        results.append({'arm':arm,'rows':arm_rows,'pass':all(r['pass'] for r in arm_rows),'load_seconds':load_seconds,'model_sha256':sha(bundle/'model.onnx')})
    verify_parent()
    save(target,{'protocol_sha256':sha(CHILD/'endpoint-check-protocol.json'),'arms':results,'pass':all(a['pass'] for a in results),
        'scope':'12 fixed development utterances per model, original PCM, complete-utterance features then 7-frame neural chunks',
        'limitations':['No wake/VAD/capture/endpoint delay or disk time in timing','Desktop CPU under concurrent generation; not Pi','Separate from preserved failed streaming-frontend check']})
    print(json.dumps({'pass':all(a['pass'] for a in results),'receipt':str(target)}),flush=True)


if __name__=='__main__':main()
