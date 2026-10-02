"""Verified execution-only worker override; frozen scoring semantics unchanged."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from scripts.train.pcv_public_data import STUDY,CACHE,sha,save,read_rows
from scripts.train.pcv_calibration import CHILD,WORK,freeze_child
import scripts.test.pcv_public_qualification as qualification


def proof():
    import torch
    import onnxruntime as ort
    from scripts.models.pcv_residual import ResidualVCM,export_onnx
    from scripts.runtime.pcv_schema import canonical
    qualification.verify_parent()
    freeze_child()
    selection=json.loads((STUDY/'selection-before-test.json').read_text())
    chosen=selection['arms'][selection['winner']]
    qualification.verify_bindings(ROOT,{chosen['checkpoint']:chosen['checkpoint_sha256']})
    directory=WORK/'execution-proof'
    directory.mkdir(exist_ok=False)
    rows=read_rows(CACHE/'prepared/dev.jsonl')
    rows=sorted(rows,key=lambda r:hashlib.sha256(('execution-proof/'+r['id']).encode()).hexdigest())[:240]
    protocol={'purpose':'execution-only equivalence and speed; no weights, catalog, calibration or test changes',
        'source_sha256':sha(Path(__file__)),'qualification_protocol_sha256':sha(CHILD/'protocol.json'),
        'checkpoint':chosen,'development_manifest_sha256':sha(CACHE/'prepared/dev.jsonl'),
        'ids':[r['id'] for r in rows],'worker_counts':[6,12],'selection':'use 12 workers only if exactly equal and faster; otherwise keep 6',
        'test_inference_accessed':False}
    save(CHILD/'execution-protocol.json',protocol)
    torch.set_num_threads(2)
    payload=torch.load(ROOT/chosen['checkpoint'],map_location='cpu',weights_only=True)
    model=ResidualVCM(**payload['model_config'])
    model.load_state_dict(payload['weights'])
    export=export_onnx(model,directory/'model.onnx')
    save(directory/'export.json',export)
    if not export['pass']:
        raise ValueError('Preflight trained export failed; preserve receipt')
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(directory/'model.onnx'),options,providers=['CPUExecutionProvider'])
    norm=json.loads((CACHE/'prepared/normalization.json').read_text())
    entries=json.loads((CHILD/'catalog.json').read_text())['entries']
    runs={}
    for workers in (6,12):
        started=time.perf_counter()
        predictions=qualification.score(session,rows,norm,entries,workers=workers)
        runs[workers]={'seconds':time.perf_counter()-started,'predictions':predictions}
        save(directory/f'workers-{workers}.json',runs[workers])
    equal=canonical(runs[6]['predictions'])==canonical(runs[12]['predictions'])
    workers=12 if equal and runs[12]['seconds']<runs[6]['seconds'] else 6
    result={'pass':equal,'selected_workers':workers,'protocol_sha256':sha(CHILD/'execution-protocol.json'),
        'source_sha256':sha(Path(__file__)),'records':len(rows),'seconds':{k:v['seconds'] for k,v in runs.items()},
        'exact_equality':'all saved score, margin, intent/target probability and candidate fields',
        'test_inference_accessed':False,'export':export}
    save(CHILD/'execution-proof.json',result)
    print(json.dumps(result),flush=True)
    if not equal:
        raise ValueError('Parallel outputs differ; do not use override')


def evaluate():
    proof_path=CHILD/'execution-proof.json'
    result=json.loads(proof_path.read_text())
    assert result['pass'] and result['source_sha256']==sha(Path(__file__))
    assert result['protocol_sha256']==sha(CHILD/'execution-protocol.json')
    protocol=json.loads((CHILD/'execution-protocol.json').read_text())
    assert protocol['qualification_protocol_sha256']==sha(CHILD/'protocol.json')
    assert protocol['development_manifest_sha256']==sha(CACHE/'prepared/dev.jsonl')
    original=qualification.score
    def score(*args,**kwargs):
        if 'workers' in kwargs:
            raise ValueError('Conflicting worker override')
        return original(*args,**kwargs,workers=result['selected_workers'])
    qualification.score=score
    print('VERIFIED_EVALUATION_WORKERS',result['selected_workers'],flush=True)
    qualification.evaluate()


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in {'proof','evaluate'}:
        raise SystemExit('Use proof or evaluate')
    globals()[sys.argv[1]]()
