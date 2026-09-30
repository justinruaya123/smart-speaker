"""Selected synthetic graph export and calibration, separate from human release."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from study import HERE, PCV, ROOT, Contract, file_hash, identity, read_rows, write_json
from synthbench import evaluate, load_runtime
from pcvcm.decision import candidate_allowed
from pcvcm.contract import canonical


def export(run, output):
    import torch
    from vcm.model import CausalVCM, export_model
    from pcvcm.graph_validation import check_graph
    from pcvcm.training import native_cpu_convolution
    from pcvcm.endpoint import EndpointConfig
    run,output=Path(run),Path(output)
    if output.exists():
        raise FileExistsError('Synthetic bundle is immutable')
    selection=json.loads((run/'selection.json').read_text())
    selected=selection['winner']
    checkpoint=Path(selected['checkpoint'])
    if file_hash(checkpoint)!=selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    model=CausalVCM();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights'])
    output.mkdir(parents=True)
    with native_cpu_convolution():
        export_model(model,output/'model.onnx')
        parity=check_graph(model,output/'model.onnx')
    for name in ('catalog.json','normalization.json'):
        shutil.copyfile(run/name,output/name)
    from study import CACHE
    shutil.copyfile(CACHE/'prepared/global-audit.json',output/'global-audit.json')
    write_json(output/'endpoint.json',asdict(EndpointConfig(rms_threshold=10**(-45/20))))
    source={str(p.relative_to(ROOT)):file_hash(p) for p in [
        *HERE.glob('*.py'),ROOT/'vcm/frontend.py',ROOT/'vcm/model.py',
        PCV/'pcvcm/runtime.py',PCV/'pcvcm/decoder.py',PCV/'pcvcm/endpoint.py',PCV/'pcvcm/decision.py']}
    write_json(output/'source-identity.json',source)
    write_json(output/'selection.json',selection)
    shutil.copyfile(HERE/'protocol.json',output/'protocol.json')
    files={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}
    binding=identity(files)
    write_json(output/'policy.json',dict(enabled=False,binding=binding,thresholds={'global':[0,1]},
        status='SYNTHETIC_CALIBRATION_PENDING'))
    write_json(output/'synthetic-bundle.json',dict(files=files,binding=binding,
        policy_sha256=file_hash(output/'policy.json'),status='SYNTHETIC_UNQUALIFIED',human_qualified=False))
    write_json(output/'onnx-parity.json',parity)
    load_runtime(output)
    return dict(binding=binding,parity=parity)


def calibrate(bundle, output, parity):
    bundle,output=Path(bundle),Path(output)
    if output.exists():
        raise FileExistsError('Calibration output must be new')
    from synthbench import validate_parity_receipt
    runtime=load_runtime(bundle)
    validate_parity_receipt(parity,runtime.binding)
    predictions=output/'replay'
    evaluate(bundle,'calibration',predictions)
    rows=read_rows(predictions/'predictions.jsonl')
    binding=rows[0]['binding']
    scores=np.array([r['diagnostic'].get('score') if r['diagnostic'].get('score') is not None else np.nan for r in rows])
    margins=np.array([r['diagnostic'].get('margin') if r['diagnostic'].get('margin') is not None else np.nan for r in rows])
    good=np.array([r['expected_command'] is not None and canonical(r['candidate'])==canonical(r['expected_command']) for r in rows])
    positive=np.array([r['expected_command'] is not None for r in rows])
    critical=np.array([r.get('critical_zero_accept',False) for r in rows])
    base=np.array([candidate_allowed(r['diagnostic'],[-float('inf'),0],r['endpoint_detected'],r['capture_fault']) for r in rows])
    finite=base & np.isfinite(scores) & np.isfinite(margins)
    groups={}
    for r in rows:
        key=(r['phase'],r['engine'],r['use_case'])
        if key not in groups:
            groups[key]=np.array([(q['phase'],q['engine'],q['use_case'])==key for q in rows])
    contract=Contract(PCV/'spec')
    predicted={uc:np.array([contract.intent_use_case.get((r['candidate'] or {}).get('intent'))==uc for r in rows]) for uc in contract.intent_use_case.values()}
    emitted_groups={key:predicted[key[2]] & np.array([(r['phase'],r['engine'])==key[:2] for r in rows]) for key in groups}
    score_grid=np.unique(np.quantile(scores[finite],np.linspace(0,1,31))) if finite.any() else []
    margin_grid=np.unique(np.quantile(margins[finite],np.linspace(0,1,21))) if finite.any() else []
    history=[];best=None
    for score in score_grid:
        for margin in margin_grid:
            accepted=finite & (scores>=score) & (margins>=margin)
            feasible=bool(accepted.any()) and not bool((accepted & critical).any()) and all(
                r['endpoint_detected'] and not r['capture_fault'] for r in rows)
            for group in groups.values():
                n=int((positive & group).sum());neg=int((~positive & group).sum())
                count=int((accepted & group).sum());correct=int((accepted & good & group).sum())
                feasible &= n>0 and neg>0 and correct/n>=.90 and count>0 and correct/count>=.99 and int((accepted & ~positive & group).sum())/neg<=.01
            for group in predicted.values():
                emitted=int((accepted & group).sum())
                feasible &= emitted>0 and int((accepted & group & good).sum())/emitted>=.99
            for group in emitted_groups.values():
                emitted=int((accepted & group).sum())
                feasible &= emitted>0 and int((accepted & group & good).sum())/emitted>=.99
            record=dict(score=float(score),margin=float(margin),correct=int((accepted & good).sum()),accepted=int(accepted.sum()),feasible=bool(feasible))
            history.append(record)
            if feasible and (best is None or record['correct']>best['correct']):
                best=record
    policy=dict(binding=binding,enabled=best is not None,
        waveform_parity=dict(path=str(Path(parity).resolve()),sha256=file_hash(parity)),
        thresholds={'global':[best['score'],best['margin']] if best else [0.,1.]},
        status='SYNTHETIC_CALIBRATED' if best else 'SYNTHETIC_CALIBRATION_FAILED',
        calibration_predictions_sha256=file_hash(predictions/'predictions.jsonl'))
    write_json(output/'selection.json',dict(policy=policy,history=history,test_accessed=False))
    # Preserve the uncalibrated export; the calibrated candidate is a new directory.
    candidate=output/'bundle';shutil.copytree(bundle,candidate)
    write_json(candidate/'policy.json',policy)
    metadata=json.loads((candidate/'synthetic-bundle.json').read_text())
    metadata['policy_sha256']=file_hash(candidate/'policy.json')
    write_json(candidate/'synthetic-bundle.json',metadata)
    load_runtime(candidate)
    return policy


def waveform_parity(run, bundle, output):
    import torch
    from vcm.model import CausalVCM
    from pcvcm.runtime import StreamEngine
    from pcvcm.endpoint import EndpointConfig
    from pcvcm.endpoint_selection import TorchEncoder
    from pcvcm.training import native_cpu_convolution
    from synthbench import load_split, replay
    run,bundle,output=Path(run),Path(bundle),Path(output)
    if output.exists():
        raise FileExistsError('Parity evidence is immutable')
    exported=load_runtime(bundle)
    selected=json.loads((run/'selection.json').read_text())['winner']
    if file_hash(Path(selected['checkpoint']))!=selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint identity changed')
    if file_hash(run/'selection.json')!=file_hash(bundle/'selection.json'):
        raise ValueError('Bundle belongs to another selected checkpoint')
    model=CausalVCM();model.load_state_dict(torch.load(selected['checkpoint'],map_location='cpu',weights_only=True)['weights'])
    # A fixed diagnostic predicate exercises decisions before calibration.
    policy=dict(enabled=True,binding=exported.binding,thresholds={'global':[-1000.,0.]})
    exported.policy=policy
    native=StreamEngine(Contract(PCV/'spec'),json.loads((bundle/'catalog.json').read_text()),
        TorchEncoder(model),json.loads((bundle/'normalization.json').read_text()),
        EndpointConfig(**json.loads((bundle/'endpoint.json').read_text())),policy,exported.binding)
    rows=load_split('dev')
    output.mkdir(parents=True)
    maximum=0.;count=0
    torch.set_num_threads(2)
    with native_cpu_convolution(), (output/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for row in rows:
            a,b=replay(native,row),replay(exported,row)
            if a['output']!=b['output'] or a['endpoint_detected']!=b['endpoint_detected'] or canonical(a['candidate'])!=canonical(b['candidate']):
                raise ValueError(f'Waveform decision parity failed: {row["recording_id"]}')
            errors=[]
            for key in ('score','margin'):
                x,y=a['diagnostic'].get(key),b['diagnostic'].get(key)
                if (x is None)!=(y is None):
                    raise ValueError('Waveform score availability differs')
                if x is not None:
                    errors.append(abs(x-y))
            error=max(errors,default=0.)
            maximum=max(maximum,error)
            stream.write(json.dumps(dict(recording_id=row['recording_id'],max_error=error,outputs_equal=True))+'\n');stream.flush()
            if error>2e-5:
                raise ValueError(f'Waveform score parity exceeds fixed tolerance: {error}')
            count+=1
            if count%100==0:print('WAVEFORM_PARITY',count,'/',len(rows),flush=True)
    report=dict(binding=exported.binding,recordings=count,max_error=maximum,
        source='synthetic_development',complete=count==len(rows),checkpoint_sha256=selected['checkpoint_sha256'],
        rows_sha256=file_hash(output/'rows.jsonl'),human_evidence=False)
    write_json(output/'results.json',report)
    return report
