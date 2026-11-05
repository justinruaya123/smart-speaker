"""Matched shared-encoder fits with complete typed DEV-only selection."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from scripts.runtime.demo_slots_contract import COMMANDS, decode_scores
from scripts.models.demo_slots_v1 import DemoSlotVCM, supervised_loss
from scripts.train.optionb_synthetic_data import SourceVoiceSampler
from scripts.train.speaker_optionb_tiny import augmented, normalized, tensor_batch
from scripts.train.speaker_optionb_tiny_accelerated import _DeviceAdamW, _configure_cuda
from scripts.test.speaker_demo_suite import write_new, file_digest


def metrics(rows, predicted):
    truth = np.asarray([r['class_index'] for r in rows],dtype=np.int64)
    pred = np.asarray(predicted)
    if len(pred)==0:
        pred = pred.astype(np.int64)
    if (pred.shape != truth.shape or not np.issubdtype(pred.dtype,np.integer)
            or np.any(pred<0) or np.any(pred>=len(COMMANDS)) or len({r['id'] for r in rows}) != len(rows)):
        raise ValueError('One valid demo prediction per unique recording is required')
    positive = truth>0
    per_class = []
    for c in range(1,len(COMMANDS)):
        guesses = pred[truth==c]
        n = len(guesses)
        correct = int(np.sum(guesses==c))
        rejected = int(np.sum(guesses==0))
        per_class.append(dict(class_index=c,command=COMMANDS[c],rows=n,correct=correct,rejected=rejected,
            wrong_accepted=n-correct-rejected,accuracy=correct/n if n else None,
            wrong_predictions=dict(Counter(str(int(v)) for v in guesses if v>0 and v!=c))))
    covered = [v['accuracy'] for v in per_class if v['rows']]
    wrong_slots = sum(t>0 and p>0 and t!=p and COMMANDS[t]['intent']==COMMANDS[p]['intent'] for t,p in zip(truth,pred))
    return dict(rows=len(rows),supported=int(positive.sum()),correct=int(np.sum(positive & (truth==pred))),
        rejected=int(np.sum(positive & (pred==0))),wrong_accepted=int(np.sum(positive & (pred>0) & (pred!=truth))),
        same_intent_wrong_slots=int(wrong_slots),supported_macro=float(np.mean(covered)) if covered else None,
        covered_classes=len(covered),negative_rows=int(np.sum(~positive)),
        negative_false_accepts=int(np.sum(~positive & (pred>0))),per_class=per_class)


def summarize(rows,predicted):
    predicates = dict(all=lambda r:True,human=lambda r:r['kind']=='human',
        parent_synthetic=lambda r:r['origin']=='parent' and r['kind']=='synthetic',
        demo=lambda r:r['origin']=='demo',
        demo_slots=lambda r:r['origin']=='demo' and (r.get('percent',0)>0 or r.get('alarm',0)>0))
    out = {}
    for key,predicate in predicates.items():
        indices = [i for i,r in enumerate(rows) if r.get('metric_eligible',True) and predicate(r)]
        out[key] = metrics([rows[i] for i in indices],np.asarray(predicted,dtype=np.int64)[indices])
    return out


def selection_rank(summary):
    human = summary['human']
    h = human['correct']/human['supported'] if human['supported'] else 0.
    slot = summary['demo_slots']['supported_macro'] or 0.
    return ((h+slot)/2,slot,human['correct'],summary['demo']['supported_macro'] or 0.,
            -summary['all']['same_intent_wrong_slots'])


def infer_logits(model,rows,arrays,norm,device):
    result = np.zeros((len(rows),model.output_size),dtype=np.float32)
    order = sorted(range(len(rows)),key=lambda i:arrays[rows[i]['id']].shape[1])
    model.eval()
    with torch.inference_mode(),torch.autocast(device.type,enabled=False):
        for start in range(0,len(order),32):
            indices = order[start:start+32]
            x,n = tensor_batch([normalized(arrays[rows[i]['id']],norm) for i in indices])
            scores = model(x.to(device),n.to(device))
            if scores.shape != (len(indices),model.output_size) or not scores.isfinite().all():
                raise ValueError('Invalid shared-demo logits')
            result[indices] = scores.detach().cpu().numpy()
    return result


def predictions(scores,mode):
    return np.asarray([decode_scores(s,mode)['class_index'] for s in scores],dtype=np.int64)


def fit(train,dev,arrays,norm,output,config):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing shared-demo fit')
    if not train or any(r['split']!='train' for r in train) or not dev or any(r['split']!='dev' for r in dev):
        raise ValueError('Only TRAIN may optimize and DEV may select')
    if {r['id'] for r in train} & {r['id'] for r in dev} or set(arrays)!={r['id'] for r in train+dev}:
        raise ValueError('Missing/reserved/colliding feature identity')
    total_epochs = config['foundation_epochs']+config['mixed_epochs']
    if min(config['foundation_epochs'],config['mixed_epochs'],config['steps_per_epoch'])<1:
        raise ValueError('Both declared phases require real optimizer steps')
    device = torch.device(config['device'])
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable; no implicit fallback')
        _configure_cuda()
    amp = config['amp_bfloat16']
    if amp and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('Declared BF16 CUDA unavailable')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    model = DemoSlotVCM(config['mode']).to(device)
    encoder_digest = hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if 'head' not in key:
            encoder_digest.update(key.encode()); encoder_digest.update(value.detach().cpu().numpy().tobytes())
    # Equal dropout RNG after the differently sized heads consume init draws.
    torch.manual_seed(config['seed']+1000)
    initial = next(model.parameters()).detach().cpu().clone()
    sampler = SourceVoiceSampler(train)
    for quotas in (config['foundation_quotas'],config['mixed_quotas']):
        sampler.batch(quotas,np.random.default_rng(0))
    optimizer = _DeviceAdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    output.mkdir(parents=True,exist_ok=False)
    write_new(output/'config.json',config)
    write_new(output/'normalization.json',norm)
    counts,sources = Counter(),Counter()
    sequence = hashlib.sha256()
    steps,diag_rank,eligible_rank = 0,None,None
    selected_diag,selected_eligible = None,None
    started = time.perf_counter()
    with (output/'training-curve.jsonl').open('x',encoding='utf-8') as stream:
        for epoch in range(1,total_epochs+1):
            mixed = epoch>config['foundation_epochs']
            position = (epoch-config['foundation_epochs']-1)/max(1,config['mixed_epochs']-1)
            end_lr = config['learning_rate']*config['lr_end_fraction']
            lr = end_lr+.5*(config['learning_rate']-end_lr)*(1+math.cos(math.pi*position)) if mixed else config['learning_rate']
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            losses=[]
            for _ in range(config['steps_per_epoch']):
                chosen = sampler.batch(config['mixed_quotas'] if mixed else config['foundation_quotas'],rng)
                batch = [train[i] for i in chosen]
                values = [augmented(arrays[r['id']],norm,rng,r['kind'],config['context_max_frames']) for r in batch]
                x,n = (v.to(device) for v in tensor_batch(values))
                labels = {k:torch.tensor([r[k] for r in batch],device=device) for k in ('class_index','action','percent','alarm')}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type,dtype=torch.bfloat16,enabled=amp):
                    loss = supervised_loss(model(x,n),labels,config['mode'],config['head_auxiliary_weight'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite shared-demo loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                counts.update(r['id'] for r in batch)
                sources.update(r['sampling_group'] for r in batch)
                sequence.update(('\n'.join(r['id'] for r in batch)+'\n').encode())
                steps+=1
            summary = summarize(dev,predictions(infer_logits(model,dev,arrays,norm,device),config['mode']))
            rank = selection_rank(summary)
            eligible = summary['human']['correct']>=config['human_floor']
            checkpoint = dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                epoch=epoch,optimizer_steps=steps,config=config,model_config=model.config,dev=summary,
                sample_counts=dict(counts),source_draws=dict(sources),sequence_sha256=sequence.hexdigest(),
                initial_encoder_sha256=encoder_digest.hexdigest(),retention_eligible=eligible)
            if mixed and (diag_rank is None or rank>diag_rank):
                diag_rank,selected_diag=rank,epoch
                torch.save(checkpoint,output/'best-diagnostic.pt')
            if mixed and eligible and (eligible_rank is None or rank>eligible_rank):
                eligible_rank,selected_eligible=rank,epoch
                torch.save(checkpoint,output/'best.pt')
            torch.save({**checkpoint,'optimizer':optimizer.state_dict(),'numpy_rng':rng.bit_generator.state,
                'torch_rng':torch.random.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},output/'last.pt')
            event = dict(epoch=epoch,phase='mixed' if mixed else 'human',optimizer_steps=steps,learning_rate=lr,rank=list(rank),
                loss=float(np.mean(losses)),human_correct=summary['human']['correct'],human_supported=summary['human']['supported'],
                demo_slot_correct=summary['demo_slots']['correct'],demo_slot_supported=summary['demo_slots']['supported'],
                demo_slot_macro=summary['demo_slots']['supported_macro'],demo_correct=summary['demo']['correct'],
                demo_supported=summary['demo']['supported'],wrong_slots=summary['all']['same_intent_wrong_slots'],
                negative_false_accepts=summary['all']['negative_false_accepts'],retention_eligible=eligible,
                selected_epoch=selected_eligible,diagnostic_epoch=selected_diag,elapsed_seconds=time.perf_counter()-started)
            stream.write(json.dumps(event)+'\n');stream.flush()
            print(json.dumps(event),flush=True)
    if steps != total_epochs*config['steps_per_epoch']:
        raise ValueError('Incomplete declared optimizer budget')
    path = output/('best.pt' if selected_eligible is not None else 'best-diagnostic.pt')
    result = dict(mode=config['mode'],optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
        selected_epoch=selected_eligible if selected_eligible is not None else selected_diag,
        checkpoint=str(path.resolve()),retention_eligible=selected_eligible is not None,
        weights_changed=not torch.equal(initial,next(model.parameters()).detach().cpu()),
        draws=sum(counts.values()),sample_counts=dict(counts),source_draws=dict(sources),
        sequence_sha256=sequence.hexdigest(),initial_encoder_sha256=encoder_digest.hexdigest(),
        checkpoint_sha256=file_digest(path),selection_split='dev',fitting_splits=['train'],test_inferred=False,
        device=str(device),amp_bfloat16=amp,elapsed_seconds=time.perf_counter()-started,actions_enabled=False,installed=False)
    write_new(output/'training-result.json',result)
    return result
