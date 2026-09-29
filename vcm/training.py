"""Bounded random-initialization fitting and development-only selection."""
import json
from pathlib import Path
import random
import time
import numpy as np
from einops import rearrange, pack
from .integrity import load, save, digest, identity


def select_epoch(previous_correct, candidate_correct):
    return set(previous_correct) <= set(candidate_correct) and len(candidate_correct)>len(previous_correct)


def train(config_path,output):
    import torch
    from .model import CausalVCM, export_model, STATE_SHAPES
    from .data import read_manifest, load_verified_features
    from .decoder import CTCRanker
    from .frontend import SPEC
    config=load(config_path); output=Path(output)
    if output.exists(): raise FileExistsError('Training output exists; choose a new run ID')
    prepared=Path(config['prepared']); norm=load(prepared/'normalization.json'); catalog=load(prepared/'catalog.json')
    train_rows=read_manifest(prepared/'train.jsonl'); dev_rows=read_manifest(prepared/'dev.jsonl')
    # Identity-based deterministic sample; selection never sees test/calibration.
    selection=sorted(dev_rows,key=lambda r:identity(r['id']))[:config['selection_examples']]
    output.mkdir(parents=True)
    protocol={'config':config,'config_sha256':digest(config_path),'manifests':{s:digest(prepared/f'{s}.jsonl') for s in ('train','dev','calibration','test')},
              'catalog':identity(catalog),'normalization':digest(prepared/'normalization.json'),
              'selection_ids':[r['id'] for r in selection],'training_examples':len(train_rows),
              'initialization':'random; no pretrained neural weights','source_files':{str(p):digest(p) for p in sorted(Path('vcm').glob('*.py'))}}
    save(output/'protocol.json',protocol)
    torch.set_num_threads(2)
    device=torch.device(config['device'])
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; freeze a CPU config explicitly')
    mean=np.asarray(norm['mean'],dtype='float32')[:,None]; std=np.asarray(norm['std'],dtype='float32')[:,None]
    arrays={r['id']:(load_verified_features(r)-mean)/std for r in train_rows+selection}
    histories=[]; winners=[]
    for seed in config['seeds']:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=CausalVCM().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
        prior=set(); best_count=-1; best_epoch=None; seed_dir=output/f'seed-{seed}'; seed_dir.mkdir()
        ranker=CTCRanker([r['phones'] for r in catalog['commands']])
        for epoch in range(1,config['epochs']+1):
            started=time.perf_counter(); model.train(); order=list(range(len(train_rows))); random.shuffle(order); losses=[]
            for start in range(0,len(order),config['batch_size']):
                rows=[train_rows[i] for i in order[start:start+config['batch_size']]]
                lengths=[arrays[r['id']].shape[1] for r in rows]
                x=np.zeros((len(rows),40,max(lengths)),dtype='float32')
                for i,r in enumerate(rows): x[i,:,:lengths[i]]=arrays[r['id']]
                targets=[p for r in rows for p in r['phones']]
                logits,_=model(torch.from_numpy(x).to(device))
                loss=criterion(rearrange(logits.log_softmax(1),'b c t -> t b c'),torch.tensor(targets,dtype=torch.long,device=device),
                               torch.tensor(lengths,dtype=torch.long),torch.tensor([len(r['phones']) for r in rows],dtype=torch.long))
                if not torch.isfinite(loss): raise ValueError('Nonfinite CTC loss')
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip']); optimizer.step()
                losses.append(float(loss.detach()))
            record={'seed':seed,'epoch':epoch,'ctc_loss':float(np.mean(losses)),'seconds':time.perf_counter()-started}
            if epoch in config['selection_epochs']:
                model.eval(); correct=set(); predictions=[]
                with torch.no_grad():
                    for r in selection:
                        logits,_=model(torch.from_numpy(arrays[r['id']][None]).to(device))
                        scores=ranker.score(rearrange(logits[0].cpu().numpy(),'c t -> t c'))
                        command=catalog['commands'][int(np.argmax(scores))]['command']
                        exact=r['expected'] is not None and command==r['expected']
                        if exact: correct.add(r['id'])
                        predictions.append({'id':r['id'],'expected':r['expected'],'candidate':command,'exact':exact})
                chosen=best_count<0 or select_epoch(prior,correct)
                record.update(dev_correct=len(correct),dev_total=len(selection),lost_previous_correct=sorted(prior-correct),selected=chosen)
                save(seed_dir/f'dev-epoch-{epoch}.json',predictions)
                if chosen:
                    prior=correct; best_count=len(correct); best_epoch=epoch
                    torch.save({'weights':model.state_dict(),'seed':seed,'epoch':epoch,'initialization':'random','protocol_hash':identity(protocol)},seed_dir/'selected.pt')
            histories.append(record); save(output/'history.json',histories); print(json.dumps(record),flush=True)
        winners.append({'seed':seed,'correct':best_count,'epoch':best_epoch,'checkpoint':str(seed_dir/'selected.pt')})
    winner=sorted(winners,key=lambda w:(-w['correct'],w['seed']))[0]
    checkpoint=torch.load(winner['checkpoint'],map_location='cpu',weights_only=True)
    model=CausalVCM(); model.load_state_dict(checkpoint['weights'])
    bundle=output/'bundle'; bundle.mkdir()
    export=export_model(model,bundle/'model.onnx'); save(output/'export.json',export)
    frontend={'spec':SPEC,'mean':norm['mean'],'std':norm['std']}
    binding={'model':digest(bundle/'model.onnx'),'catalog':identity(catalog),'frontend':identity(frontend)}
    save(bundle/'catalog.json',catalog)
    save(bundle/'policy.json',{'enabled':False,'binding':binding,'reason':'calibration pending','min_score':0.,'min_margin':0.})
    recipe={'format_version':1,'status':'diagnostic_unqualified','experiment':config['experiment_id'],'frontend':frontend,
            'state_shapes':STATE_SHAPES,'chunk_frames':4,'logits':'raw','opset':17,'provider':'CPUExecutionProvider',
            'training_protocol':identity(protocol),'selection':winner,'files':{p.name:digest(p) for p in bundle.iterdir()}}
    save(bundle/'recipe.json',recipe)
    save(output/'selection-before-test.json',{'protocol_hash':identity(protocol),'candidates':winners,'winner':winner,
                                             'binding':binding,'test_accessed':False,'calibration_accessed':False})
    print('BUNDLE',bundle,flush=True)
    return bundle
