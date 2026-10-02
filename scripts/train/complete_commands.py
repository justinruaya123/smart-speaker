"""Matched joint VCM fitting; all selection uses original human development."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import copy
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .complete_data import ROOT, DATA, EXP, CACHE, CUES, read_rows, save, digest, verify_frozen, feature, cue_labels
from .slot_model import SlotVCM
from .train_helpers import seed_everything, batches
from .optimization import head_weights, balanced_losses, fit_intent_policy
from .finite_policy import provisional_slots, command_metrics, slot_records
from .metrics import policy_rows, metrics
from .candidate import head_metrics


def experiment_for(arm):
    if arm.startswith(('v2_', 'v3_', 'v4_', 'v5_')):
        return EXP.parent / ('vcm-complete-' + arm.split('_')[0])
    return EXP


def mask_training_features(x, lengths, rng):
    """Small train-only time/frequency masks; no evaluation or payload mutation."""
    for i, length in enumerate(lengths.tolist()):
        width = rng.randint(0, 4)
        start = rng.randint(0, 40 - width)
        x[i, start:start + width, :length] = 0
        width = rng.randint(0, min(12, length // 10))
        start = rng.randint(0, length - width)
        x[i, :, start:start + width] = 0
    return x


def load_features(rows):
    result, used, last = {}, 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, value in zip(rows, pool.map(feature, rows)):
            used += value.nbytes
            if used > 4 * 1024**3:raise RuntimeError('Declared 4 GiB feature RAM limit exceeded')
            result[row['id']] = value
            row['frames'] = value.shape[-1]
            if time.monotonic()-last > 20:
                print(f'FEATURES {len(result)}/{len(rows)} RAM={used/2**20:.0f} MiB',flush=True)
                last=time.monotonic()
    print(f'FEATURES_READY recordings={len(result)} RAM={used/2**20:.0f} MiB',flush=True)
    return result


def make_batch(rows, ids, cache, stats):
    lengths=torch.tensor([rows[i]['frames'] for i in ids],dtype=torch.int64)
    size=int(math.ceil(int(lengths.max())/32)*32)
    x=np.zeros((len(ids),40,size),dtype=np.float32)
    mean=rearrange(np.array(stats['mean'],dtype=np.float32),'mel -> mel 1')
    std=rearrange(np.array(stats['std'],dtype=np.float32),'mel -> mel 1')
    for j,i in enumerate(ids):
        array=cache[rows[i]['id']]
        x[j,:,:array.shape[-1]]=(array-mean)/std
    return torch.from_numpy(x), lengths


def sample_epoch(rows, cfg, rng):
    if any(r['split'] != 'train' or r.get('unusable_reasons') for r in rows):
        raise ValueError('Only eligible training rows may enter the sampler')
    human=[i for i,r in enumerate(rows) if r['kind']=='human']
    synthetic=[i for i,r in enumerate(rows) if r['kind']=='synthetic']
    by_family=defaultdict(list)
    for i in human:by_family[rows[i]['command']].append(i)
    chosen=[rng.choice(group) for family, group in by_family.items()
            for _ in range(cfg['per_family_draws'] * (cfg.get('unknown_multiplier', 1) if family == 'UNKNOWN' else 1))]
    by_slot=defaultdict(list)
    for i in human:
        for name in ['operation','amount','target','color']:
            value=rows[i]['supervision']['labels'].get(name)
            if value not in {None,'NONE','OTHER'}:by_slot[(name,value)].append(i)
    keys=sorted(by_slot)
    chosen.extend(rng.choice(by_slot[rng.choice(keys)]) for _ in range(cfg['slot_draws']))
    if synthetic:
        # Same update budget; replace a declared fraction with balanced TTS.
        group=defaultdict(list)
        for i in synthetic:
            a=rows[i]['supervision']['labels']
            key=(rows[i]['command'],a.get('operation'),a.get('amount'),a.get('color'))
            group[key].append(i)
        keys=list(group)
        for pos in rng.sample(range(len(chosen)),round(len(chosen)*cfg['synthetic_max_fraction'])):
            chosen[pos]=rng.choice(group[rng.choice(keys)])
    rng.shuffle(chosen)
    return chosen


def score_development(rows, outputs, metadata, fit_policy=True):
    vocabulary=metadata['acoustic_slots']['vocabulary']
    fit=[i for i,r in enumerate(rows) if r['fold']=='policy_fit']
    selected=[i for i,r in enumerate(rows) if r['fold']=='selection']
    family=torch.softmax(outputs[0],-1).tolist()
    records=[{'id':r['id'],'gold':r['command'],'probabilities':p} for r,p in zip(rows,family)]
    policy=fit_intent_policy([records[i] for i in fit],metadata['vcm']['labels'],1.) if fit_policy else metadata['routing']['variants']['fp32']
    accepted=policy_rows(records,metadata['vcm']['labels'],policy)
    slots=slot_records(outputs,vocabulary)
    slot_policy=provisional_slots([rows[i] for i in fit],[rows[i]['annotation'] for i in fit],
        [accepted[i] for i in fit],[slots[i] for i in fit],metadata['acoustic_slots'],metadata['catalog']) if fit_policy else metadata['acoustic_slots']['variants']['fp32']
    schema=copy.deepcopy(metadata['acoustic_slots']);schema['variants']={'fp32':slot_policy}
    command,details=command_metrics([rows[i] for i in selected],[rows[i]['annotation'] for i in selected],
        [accepted[i] for i in selected],[slots[i] for i in selected],schema,metadata['catalog'])
    intent=metrics([accepted[i] for i in selected],metadata['vcm']['labels'])
    heads=head_metrics(tuple(v[selected] for v in outputs),[rows[i]['annotation'] for i in selected],vocabulary)
    unknown=intent['unknown_false_accept_rate']
    valid_gate=command['emitted']>0 and command['accepted_command_precision']>=.95 and unknown<=.01
    recalls=[c['correct_complete']/c['resolvable'] for c in command['per_family'].values() if c['resolvable']]
    rank=(valid_gate,command['exact_command_success'],min(recalls),command['accepted_command_precision'])
    return {'commands':command,'intent':intent,'heads':heads,'rank':rank},policy,slot_policy


def train(arm):
    verify_frozen()
    experiment = experiment_for(arm)
    protocol=json.loads((experiment/'protocol.json').read_text())
    cfg,variant=protocol['training'],protocol['arms'][arm]
    if digest(ROOT/'artifacts/vcm.pt')!=protocol['baseline']['checkpoint_sha256']:
        raise ValueError('Installed source checkpoint changed')
    folder=CACHE/arm
    if folder.exists():raise ValueError('Run already exists; use a separately declared retry')
    folder.mkdir(parents=True)
    rows=read_rows(DATA/'manifest.jsonl')
    training=[r for r in rows if r['split']=='train' and not r['unusable_reasons']
        and (not variant['filtered'] or not r['filtered_reasons']) and (r['kind']=='human' or variant['synthetic'])]
    dev=[r for r in rows if r['kind']=='human' and r['split']=='dev']
    if not torch.cuda.is_available():raise RuntimeError('Declared CUDA device is unavailable')
    seed_everything(cfg['seed'])
    metadata=json.loads((ROOT/'artifacts/bundle.json').read_text())
    ck=torch.load(ROOT/'artifacts/vcm.pt',map_location='cpu',weights_only=False)
    model=SlotVCM(**ck['config']).to('cuda')
    model.load_state_dict(ck['model'],strict=True)
    for p in model.backbone.parameters():p.requires_grad_(True)
    width=model.heads.project[0].normalized_shape[0]
    keywords=nn.Linear(width,len(CUES)).to('cuda') if variant['keywords'] else None
    seed_everything(cfg['seed'])
    all_rows=training+dev
    cache=load_features(all_rows)
    vocab=ck['config']['vocabulary']
    supervision=[r['supervision'] for r in training]
    targets={name:torch.tensor([values.index(a['labels'][name]) if a['labels'].get(name) is not None else 0 for a in supervision],device='cuda') for name,values in vocab.items()}
    masks={name:torch.tensor([a['labels'].get(name) is not None for a in supervision],device='cuda') for name in vocab}
    families=torch.tensor([ck['labels'].index(r['command']) for r in training],device='cuda')
    cues=torch.tensor([cue_labels(r) for r in training],dtype=torch.float32,device='cuda')
    weights=head_weights(supervision,vocab,'cuda')
    groups=[{'params':list(model.backbone.parameters()),'lr':cfg['encoder_lr']},
            {'params':[p for p in model.heads.parameters() if p.requires_grad],'lr':cfg['heads_lr']}]
    if keywords is not None:groups.append({'params':list(keywords.parameters()),'lr':cfg['heads_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs'],eta_min=1e-5)
    def infer():
        outputs=[torch.empty(len(dev),n) for n in [len(ck['labels']),*map(len,vocab.values())]]
        model.eval()
        with torch.inference_mode():
            for ids in batches(dev,cfg['batch_size']):
                x,lengths=make_batch(dev,ids,cache,ck['normalization'])
                values=model(x.to('cuda'),lengths.to('cuda'))
                for out,val in zip(outputs,values):out[ids]=val.cpu()
        return tuple(outputs)
    baseline,_,_=score_development(dev,infer(),metadata,fit_policy=False)
    save(folder/'development-baseline.json',baseline)
    run={'arm':arm,'training_rows':len(training),'human_rows':sum(r['kind']=='human' for r in training),
         'synthetic_rows':sum(r['kind']=='synthetic' for r in training),'development_rows':len(dev),
         'protocol_sha256':digest(experiment/'protocol.json'),'data_lock_sha256':digest(DATA/'frozen.json'),
         'source_checkpoint_sha256':protocol['baseline']['checkpoint_sha256'],'source_sha256':digest(__file__),
         'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':cfg['seed'],'installed':False}
    save(folder/'run.json',run)
    history,best,stale=[],None,0
    started=time.perf_counter()
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        if keywords is not None:keywords.train()
        indices=sample_epoch(training,cfg,random.Random(cfg['seed']*1000+epoch))
        losses=[]
        augmentation_rng = random.Random(cfg['seed'] * 10000 + epoch)
        for ids in batches(training,cfg['batch_size'],rng=random.Random(17000+epoch),indices=indices,frame_budget=22000):
            x,lengths=make_batch(training,ids,cache,ck['normalization'])
            if variant.get('feature_masks'):
                x = mask_training_features(x, lengths, augmentation_rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                encoded=model.encode(x.to('cuda'),lengths.to('cuda'))
                values=model.heads(*encoded)
                per_head=balanced_losses(dict(zip(vocab,values[1:])),{k:v[ids] for k,v in targets.items()},
                    {k:v[ids] for k,v in masks.items()},weights)
                loss=cfg.get('family_loss_weight', 2.)*F.cross_entropy(values[0],families[ids],label_smoothing=cfg.get('family_label_smoothing', 0.))+sum(cfg.get('slot_loss_weight', .5)*v for v in per_head.values())
                if keywords is not None:loss=loss+cfg['keyword_loss_weight']*F.binary_cross_entropy_with_logits(keywords(encoded[0]),cues[ids])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.);optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        result,intent_policy,slot_policy=score_development(dev,infer(),metadata)
        entry={'epoch':epoch,'loss':float(np.mean(losses)),'elapsed_seconds':time.perf_counter()-started,**result}
        history.append(entry)
        rank=tuple(result['rank'])
        if best is None or rank>best:
            best,stale=rank,0
            selected={**ck,'config':model.config,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'epoch':epoch,'seed':cfg['seed'],'intent_policy':intent_policy,'slot_policy':slot_policy,'metrics':entry,
                'training_mode':'E13_joint_'+arm}
            torch.save(selected,folder/'vcm.pt')
            if keywords is not None:torch.save(keywords.state_dict(),folder/'keyword-training-only.pt')
        else:stale+=1
        save(folder/'training-history.json',history)
        print(json.dumps({'arm':arm,'epoch':epoch,'loss':entry['loss'],'dev_complete':result['commands']['exact_command_success'],
            'dev_precision':result['commands']['accepted_command_precision'],'dev_unknown':result['intent']['unknown_false_accept_rate'],
            'best':list(best),'seconds':round(entry['elapsed_seconds'],1)}),flush=True)
        if epoch>=cfg['minimum_epochs'] and stale>=cfg['patience']:break
    run.update(complete=True,epochs=len(history),selected_epoch=selected['epoch'],checkpoint_sha256=digest(folder/'vcm.pt'),
               seconds=time.perf_counter()-started,test_used=False)
    save(folder/'run.json',run)
    save(experiment/(arm+'-training.json'),run)
    print('TRAINING_COMPLETE '+arm,flush=True)


class ExportModel(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model
    def forward(self,x,lengths):
        encoded=self.model.encode(x,lengths)
        return (*self.model.heads(*encoded),F.normalize(encoded[0],dim=-1))


def finalize(arm):
    """Fresh CPU process: export and calibrate on deployment-matched dev calls."""
    import onnx
    from .inference import Recognizer
    from .audio import load_audio,features
    folder=CACHE/arm
    ck=torch.load(folder/'vcm.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(4)
    model=SlotVCM(**ck['config']).eval();model.load_state_dict(ck['model'])
    wrapper=ExportModel(model).eval()
    graph=folder/'vcm.fp32.onnx'
    names=['logits',*['slot_'+name for name in ck['config']['vocabulary']],'embedding']
    torch.onnx.export(wrapper,(torch.randn(1,40,160),torch.tensor([160])),str(graph),
        input_names=['features','lengths'],output_names=names,opset_version=17,dynamo=False,
        dynamic_axes={'features':{0:'batch',2:'frames'},'lengths':{0:'batch'},**{n:{0:'batch'} for n in names}})
    metadata=json.loads((ROOT/'artifacts/bundle.json').read_text())
    metadata['files']={'vcm.fp32.onnx':{'sha256':digest(graph),'bytes':graph.stat().st_size},
                       'vcm.pt':{'sha256':digest(folder/'vcm.pt'),'bytes':(folder/'vcm.pt').stat().st_size}}
    metadata['installation']={'status':'research_candidate_not_installed','experiment':experiment_for(arm).name,'arm':arm}
    metadata['experiment'].update(status='E13_development_selected',seed=ck['seed'],epoch=ck['epoch'],checkpoint_sha256=digest(folder/'vcm.pt'))
    metadata['vcm']['checkpoint_sha256']=digest(folder/'vcm.pt')
    m=onnx.load(graph)
    for key,value in [('vcm_labels',ck['labels']),('slot_vocabulary',ck['config']['vocabulary'])]:
        prop=m.metadata_props.add();prop.key=key;prop.value=json.dumps(value)
    onnx.save(m,graph);metadata['files']['vcm.fp32.onnx'].update(sha256=digest(graph),bytes=graph.stat().st_size)
    save(folder/'bundle.json',metadata)
    recognizer=Recognizer(folder,threads=2)
    rows=[r for r in read_rows(DATA/'manifest.jsonl') if r['kind']=='human' and r['split']=='dev']
    outputs=[torch.empty(len(rows),n) for n in [len(ck['labels']),*map(len,ck['config']['vocabulary'].values())]]
    parity_error=0.;started=last=time.monotonic()
    for i,row in enumerate(rows):
        pcm,_=load_audio(ROOT/row['audio'])
        x=recognizer.normalize(features(pcm));lengths=np.array([x.shape[-1]],dtype=np.int64)
        values=recognizer.vcm.run(names[:-1],{'features':x,'lengths':lengths})
        for output,value in zip(outputs,values):output[i]=torch.from_numpy(value[0])
        if i<20:
            with torch.inference_mode():expected=wrapper(torch.from_numpy(x),torch.from_numpy(lengths))
            parity_error=max(parity_error,max(float(np.max(np.abs(a.numpy()-b))) for a,b in zip(expected,values)))
        if time.monotonic()-last>20:
            print(f'CPU_DEV {arm} {i+1}/{len(rows)}',flush=True);last=time.monotonic()
    assert parity_error<1e-4
    result,intent_policy,slot_policy=score_development(rows,tuple(outputs),metadata)
    metadata['routing']['variants']['fp32']=intent_policy
    metadata['acoustic_slots']['variants']['fp32']=slot_policy
    save(folder/'bundle.json',metadata)
    result.update(arm=arm,graph_sha256=digest(graph),graph_bytes=graph.stat().st_size,
        source_checkpoint_sha256=digest(folder/'vcm.pt'),parity_max_abs_error=parity_error,
        single_recording_cpu_onnx=True,test_used=False,seconds=time.monotonic()-started)
    save(experiment_for(arm)/(arm+'-development.json'),result)
    torch.save(tuple(outputs),folder/'development-logits.pt')
    print(json.dumps({'arm':arm,'rank':result['rank'],'parity_max_abs_error':parity_error}),flush=True)
