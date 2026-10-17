"""Expanded independent Hey Iris fitting and streaming calibration."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts.train.iris_diversity import ROOT,HERE,INPUTS,read_rows,write,digest,capture_device_rng,restore_device_rng

CACHE=ROOT/'.cache/iris-release-v1/wake'


def wake_rows(generated,base,split):
    target='voice-test' if split=='test' else split
    result=[dict(r) for r in generated if r['split']==target]
    rng=np.random.default_rng(61)
    for kind in (False,True):
        candidates=[r for r in base if r['split']==split and r['synthetic']==kind and
                    'iris' not in str(r.get('text','')).lower()]
        order=rng.permutation(len(candidates))[:(512 if split=='train' else 64)]
        result.extend({**candidates[i],'wake_positive':False} for i in order)
    return result


def zero_event_upper_per_hour(seconds):
    return -math.log(.05)*3600/seconds if seconds>0 else None


def read_pcm(row):
    import soundfile as sf
    from scripts.iris_release import resolve_audio
    path=resolve_audio(row,root=ROOT)
    if digest(path)!=row['sha256']:raise ValueError('Changed wake audio '+row['id'])
    pcm,rate=sf.read(path,dtype='float32')
    if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Invalid wake PCM')
    return pcm


def prepare():
    # Wake data depends on the frozen corpus, not VCM checkpoint availability.
    # This also permits the separately declared random-init reconstruction path.
    parent=json.loads((HERE/'protocol.json').read_text())
    for relative,expected in parent['input_bindings'].items():
        if relative.endswith('-initializer.pt'):continue
        if digest(ROOT/relative)!=expected:raise ValueError('Changed wake corpus input: '+relative)
    manifest=ROOT/'.cache/iris-release-v1/diversity/generated.jsonl'
    complete=json.loads((HERE/'generation-complete.json').read_text())
    if digest(manifest)!=complete['manifest_sha256']:raise ValueError('Changed completed wake generation')
    CACHE.mkdir(exist_ok=True,parents=True)
    generated=read_rows(manifest)
    bindings={};counts={}
    for split in ('train','dev','calibration','test'):
        base=read_rows(INPUTS/'base-views'/f'{split}.jsonl')
        rows=wake_rows(generated,base,split)
        target=CACHE/'views'/f'{split}.jsonl';target.parent.mkdir(exist_ok=True)
        data=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows)
        if target.exists() and target.read_text()!=data:raise ValueError('Changed wake source rows')
        target.write_text(data);bindings[str(target.relative_to(ROOT))]=digest(target)
        counts[split]=dict(total=len(rows),positive=sum(r['wake_positive'] for r in rows),
                          original_human_positive=0)
    baseline=INPUTS/'wake-baseline.pt'
    if not baseline.exists():
        import shutil
        shutil.copyfile(ROOT/'.cache/vcm-mvp-hey-iris-v1/wake-v1/training/best.pt',baseline)
    cfg=dict(experiment='iris-release-v1-wake',parent_protocol_sha256=digest(HERE/'protocol.json'),
        seed=61,initialization='random; separate older own-weight baseline compared on new calibration only',
        training=dict(epochs=20,steps_per_epoch=96,batch=64,learning_rate=.001,weight_decay=.001),
        input_bindings=bindings,normalization_sha256=digest(INPUTS/'normalization.json'),
        baseline_checkpoint_sha256=digest(baseline),split_counts=counts,
        calibration='maximize detected synthetic calibration wakes with zero observed negative/duplicate wakes; no test tuning',
        threshold_grid=[.5,.6,.7,.8,.85,.9,.95,.98,.995,.999,.9999,1.],
        window_samples=25600,step_samples=1280,original_human_wake_positive_n=0,
        source_sha256=digest(Path(__file__)),physical_pi_measured=False,qualified=False)
    write(HERE/'wake-protocol.json',cfg);return cfg


def train(device='cuda'):
    import torch
    from scripts.models.mvp_wake import WakeNet
    from scripts.train.iris_wake_helpers import training_arrays
    cfg=prepare();hp=cfg['training'];out=CACHE/'training';out.mkdir(exist_ok=True)
    from scripts.train.iris_diversity import completed_selection
    binding=dict(protocol_sha256=digest(HERE/'wake-protocol.json'),trainer_sha256=digest(Path(__file__)),
        model_source_sha256=digest(ROOT/'scripts/models/mvp_wake.py'),
        helper_sha256=digest(ROOT/'scripts/train/iris_wake_helpers.py'),frontend_sha256=digest(ROOT/'vcm/frontend.py'),
        normalization_sha256=digest(INPUTS/'normalization.json'))
    cached=completed_selection(out,binding)
    if cached is not None:return cached
    write(out/'run-inputs.json',binding)
    torch.set_num_threads(4);torch.manual_seed(61)
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.12)
    norm=json.loads((INPUTS/'normalization.json').read_text())
    x,y,excluded=training_arrays(read_rows(CACHE/'views/train.jsonl'),norm,training=True,hard=True,reader=read_pcm)
    dx,dy,dev_excluded=training_arrays(read_rows(CACHE/'views/dev.jsonl'),norm,training=False,reader=read_pcm)
    write(out/'data-audit.json',dict(training_windows=len(y),positive_windows=int(y.sum()),
        development_windows=len(dy),excluded_training=excluded,excluded_development=dev_excluded,
        source_protocol_sha256=digest(HERE/'wake-protocol.json'),human_positive_n=0))
    net=WakeNet().to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(61);groups=[np.flatnonzero(y==label) for label in (0,1)]
    history=[];best=None;began=time.perf_counter();first=1
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location=device,weights_only=True)
        if saved['protocol_sha256']!=digest(HERE/'wake-protocol.json'):raise ValueError('Changed wake resume recipe')
        net.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        restore_device_rng(saved['cuda_rng'],device)
        rng.bit_generator.state=saved['numpy_rng'];history=saved['history'];best=saved['best'];first=saved['epoch']+1
    for epoch in range(first,hp['epochs']+1):
        net.train();losses=[]
        for step in range(hp['steps_per_epoch']):
            ids=np.concatenate([rng.choice(group,32) for group in groups])
            batch=torch.from_numpy(x[ids]).to(device);target=torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(net(batch),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite wake loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        net.eval();pred=[]
        with torch.no_grad():
            for offset in range(0,len(dy),64):pred.extend(net(torch.from_numpy(dx[offset:offset+64]).to(device)).argmax(1).cpu().tolist())
        pred=np.asarray(pred);recall=float(np.mean(pred[dy==1]==1));specificity=float(np.mean(pred[dy==0]==0))
        key=(recall+specificity,specificity,recall)
        record=dict(epoch=epoch,loss=float(np.mean(losses)),dev_recall=recall,dev_specificity=specificity,seconds=time.perf_counter()-began)
        history.append(record);print('WAKE',json.dumps(record),flush=True)
        if best is None or key>tuple(best['key']):
            torch.save(dict(weights={k:v.detach().cpu() for k,v in net.state_dict().items()},config=net.config,epoch=epoch),out/'best.pt')
            best=dict(epoch=epoch,key=list(key),checkpoint_sha256=digest(out/'best.pt'))
        torch.save(dict(weights=net.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=capture_device_rng(device),numpy_rng=rng.bit_generator.state,
            protocol_sha256=digest(HERE/'wake-protocol.json'),history=history,best=best,epoch=epoch),out/'last.tmp')
        (out/'last.tmp').replace(out/'last.pt');(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    result={**best,'parameters':sum(p.numel() for p in net.parameters()),'seed':61,
        'training_seconds':time.perf_counter()-began,'test_used_for_selection':False,'original_human_positive_n':0}
    write(out/'selection.json',result);return result


def _load_model(path):
    import torch
    from scripts.models.mvp_wake import WakeNet
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=WakeNet(**saved['config']);model.load_state_dict(saved['weights']);model.eval();torch.set_num_threads(1)
    return model


def select():
    cfg=prepare();path=HERE/'wake-calibration.json'
    if path.exists():return json.loads(path.read_text())
    from scripts.train.iris_wake_helpers import streaming_scores,stream_metrics
    norm=json.loads((INPUTS/'normalization.json').read_text());rows=read_rows(CACHE/'views/calibration.jsonl')
    candidates=[('expanded',CACHE/'training/best.pt'),('baseline',INPUTS/'wake-baseline.pt')]
    choices=[]
    for name,checkpoint in candidates:
        records=streaming_scores(rows,_load_model(checkpoint),norm,reader=read_pcm)
        write(CACHE/f'{name}-calibration-scores.json',records)
        best=None
        for threshold in cfg['threshold_grid']:
            metric=stream_metrics(records,threshold)
            if metric['false_wakes'] or metric['duplicate_detections']:continue
            key=(metric['detected'],-threshold)
            if best is None or key>best[0]:best=(key,threshold,metric)
        if best is None:raise ValueError('No feasible wake policy')
        choices.append(dict(name=name,checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=digest(checkpoint),threshold=best[1],metrics=best[2]))
    chosen=max(choices,key=lambda c:(c['metrics']['detected'],-c['threshold'],c['name']=='expanded'))
    result=dict(**chosen,candidates=choices,protocol_sha256=digest(HERE/'wake-protocol.json'),test_used_for_selection=False)
    write(path,result);return result


def export():
    from scripts.train.iris_wake_helpers import export_engines
    selected=select();out=CACHE/'export';out.mkdir(exist_ok=True)
    if (out/'iris.json').exists():return str(out)
    checkpoint=ROOT/selected['checkpoint']
    if digest(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Changed wake checkpoint')
    model=_load_model(checkpoint);engines=export_engines(model,out)
    norm=json.loads((INPUTS/'normalization.json').read_text())
    write(out/'iris.json',dict(files=engines['files'],normalization=norm,
        threshold=selected['threshold'],release=.3,consecutive=2,cooldown_samples=16000,
        qualified=False,initialization='random',parameters=sum(p.numel() for p in model.parameters()),
        real_wake_positive_recordings=0,calibration=selected['metrics'],engine_parity_max_error=engines['max_absolute_error']))
    write(HERE/'wake-selection-before-test.json',dict(checkpoint_sha256=digest(checkpoint),
        model_sha256=digest(out/'iris.onnx'),policy_sha256=digest(out/'iris.json'),
        views_sha256={s:digest(CACHE/'views'/f'{s}.jsonl') for s in ('train','dev','calibration','test')},
        test_used_for_selection=False))
    return str(out)


def evaluate():
    from scripts.train.iris_wake_helpers import streaming_scores,stream_metrics
    selected=select();lock=json.loads((HERE/'wake-selection-before-test.json').read_text())
    if digest(CACHE/'export/iris.json')!=lock['policy_sha256']:raise ValueError('Changed locked wake policy')
    path=HERE/'wake-results.json'
    if path.exists():return json.loads(path.read_text())
    rows=read_rows(CACHE/'views/test.jsonl');norm=json.loads((INPUTS/'normalization.json').read_text())
    records=streaming_scores(rows,_load_model(ROOT/selected['checkpoint']),norm,reader=read_pcm)
    write(CACHE/'test-stream-scores.json',records);metric=stream_metrics(records,selected['threshold'])
    result=dict(selected=selected['name'],test=metric,original_human_positive_n=0,
        physical_pi_measured=False,qualified=False,
        zero_event_false_wakes_per_hour_upper_95=zero_event_upper_per_hour(metric['negative_seconds']) if metric['false_wakes']==0 else None)
    write(path,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train','select','export','evaluate'])
    p.add_argument('--device',default='cuda');args=p.parse_args()
    result=train(args.device) if args.stage=='train' else globals()[args.stage]()
    print(json.dumps(result,default=str,allow_nan=False),flush=True)


if __name__=='__main__':main()
