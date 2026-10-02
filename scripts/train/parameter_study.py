"""Train/evaluate four small parameter approaches on original public human audio."""
from collections import Counter, defaultdict
import argparse
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
from einops import rearrange, pack
from .parameter_study_data import ROOT, OUT, DATA, prepare, dump, digest, pcm, lexicon, phones
from .parameter_models import KeywordEncoder
from .stt import BrightnessSTT
from .stt_runtime import stt_features


def edit(a,b):
    previous=list(range(len(b)+1))
    for i,aa in enumerate(a):
        current=[i+1]
        for j,bb in enumerate(b): current.append(min(previous[j+1]+1,current[j]+1,previous[j]+(aa!=bb)))
        previous=current
    return previous[-1]


def cached_features(rows):
    fingerprint=digest(DATA/'manifest.jsonl')
    from . import stt_runtime
    frontend_sha=digest(__import__('pathlib').Path(stt_runtime.__file__))
    path=OUT/f'features-{fingerprint[:12]}-{frontend_sha[:12]}.pt'
    if path.exists():return torch.load(path,weights_only=False)['features']
    cached=[]
    for i,r in enumerate(rows):
        cached.append(stt_features(pcm(r)))
        if i%1000==0:print('Human PCM features',i,'/',len(rows),flush=True)
    torch.save({'features':cached,'manifest_sha256':fingerprint,'frontend_sha256':frontend_sha},path)
    return cached


def collate(indices, features):
    indices=sorted(indices,key=lambda i:len(features[i]),reverse=True)
    lengths=torch.tensor([len(features[i]) for i in indices],dtype=torch.long)
    x=torch.zeros(len(indices),int(lengths[0]),80)
    for j,i in enumerate(indices):x[j,:lengths[j]]=torch.from_numpy(features[i])
    return x,lengths,indices


def decode(logits, length, vocab):
    best=logits[:length].argmax(-1).tolist();tokens=[];previous=-1
    for item in best:
        if item and item!=previous:tokens.append(vocab[item])
        previous=item
    return tokens


def export(model, name, output_dir=None, parity_lengths=None):
    model=model.cpu().eval();path=(output_dir or OUT)/f'{name}.onnx'
    is_ctc=name in ('character','phoneme')
    names=['logits','output_lengths'] if is_ctc else ['logits','embedding']
    axes={'features':{0:'batch',1:'time'},'lengths':{0:'batch'},'logits':{0:'batch'}}
    axes[names[1]]={0:'batch'}
    if is_ctc:axes['logits'][1]='output_time'
    torch.onnx.export(model,(torch.zeros(1,201,80),torch.tensor([201])),path,dynamo=False,opset_version=17,
        input_names=['features','lengths'],output_names=names,dynamic_axes=axes)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    errors=[]
    for sizes in (parity_lengths or ([95],[201],[419],[301,177,95])):
        x=torch.randn(len(sizes),max(sizes),80);lengths=torch.tensor(sizes)
        for j,size in enumerate(sizes):x[j,size:]=0
        with torch.inference_mode():expected=model(x,lengths)
        actual=session.run(None,{'features':x.numpy(),'lengths':lengths.numpy()})
        error=max(float(np.max(np.abs(a-e.numpy()))) for a,e in zip(actual,expected))
        if error>1e-4:raise AssertionError((name,'ONNX parity',error))
        for j,size in enumerate(sizes):
            single=session.run(None,{'features':x[j:j+1,:size].numpy(),'lengths':np.array([size],dtype=np.int64)})
            for a,b in zip(single,actual):
                target=b[j,:a.shape[1]] if is_ctc and a.ndim==3 else b[j]
                difference=float(np.max(np.abs(a[0]-target)))
                if difference>1e-4:raise AssertionError((name,'padding parity',difference))
        errors.append(dict(lengths=sizes,max_abs_error=error))
    return errors


def train_one(name, rows, features, epochs=28):
    if epochs < (4 if name in ('phoneme','character') else 1):
        raise ValueError('Training needs at least one epoch, or four including CTC warmup')
    seed={'fixed':31,'prototype':32,'phoneme':33,'character':34}[name]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('This recorded protocol requests CUDA training')
    device='cuda';is_ctc=name in ('phoneme','character')
    dictionary=lexicon();labels={}
    classes=sorted({t['role']+'::'+t['value'] for r in rows if r['split']=='train' for t in r['targets']})
    classes=['__other__']+classes;class_ids={k:i for i,k in enumerate(classes)}
    for i,r in enumerate(rows):
        labels[i]=[class_ids[t['role']+'::'+t['value']] for t in r['targets'] if t['role']+'::'+t['value'] in class_ids] or [0]
    if name=='character':vocab=['<blank>']+list(" abcdefghijklmnopqrstuvwxyz'")
    elif name=='phoneme':vocab=['<blank>']+sorted({p for values in dictionary.values() for p in values}|{'|'})
    else:vocab=classes
    lookup={w:i for i,w in enumerate(vocab)};targets={};excluded=Counter()
    for i,r in enumerate(rows):
        sequence=phones(r['text'],dictionary) if name=='phoneme' else list(r['text'])
        if sequence is None or any(w not in lookup for w in sequence):
            if is_ctc:excluded[r['split']]+=1
            continue
        targets[i]=[lookup[w] for w in sequence]
    train=[i for i,r in enumerate(rows) if r['split']=='train' and (not is_ctc or i in targets)]
    dev=[i for i,r in enumerate(rows) if r['split']=='dev' and (not is_ctc or i in targets)]
    model=BrightnessSTT(len(vocab),width=64,layers=2) if is_ctc else KeywordEncoder(len(classes))
    model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(epochs,1),eta_min=.00015)
    ctc=nn.CTCLoss(blank=0,zero_infinity=True)
    counts=Counter(k for i in train for k in labels[i]);weights=torch.tensor([1/math.sqrt(max(1,counts[k])) for k in range(len(classes))],device=device)
    weights=weights/weights.mean();weights=weights.clamp(.1,5.)
    by_label=defaultdict(list)
    for i in train:
        for k in labels[i]:
            if k:by_label[k].append(i)
    checkpoint=OUT/f'{name}.pt';history=[];best=float('inf');started=time.perf_counter();bad=0
    for epoch in range(1,epochs+1):
        model.train();begin=time.perf_counter();order=list(train)
        if is_ctc and epoch<=3:order=[i for i in train if rows[i]['kind']=='word']
        random.shuffle(order);loss_total=0.;steps=0
        for offset in range(0,len(order),64):
            indices=order[offset:offset+64]
            if name=='prototype':
                chosen=random.sample(list(by_label),min(16,len(by_label)))
                indices=[random.choice(by_label[k]) for k in chosen for _ in range(4)]
            x,l,ii=collate(indices,features);optimizer.zero_grad(set_to_none=True)
            logits,aux=model(x.to(device),l.to(device))
            if is_ctc:
                y,_=pack([torch.tensor(targets[i],dtype=torch.long) for i in ii],'*')
                yl=torch.tensor([len(targets[i]) for i in ii],dtype=torch.long)
                loss=ctc(rearrange(logits.log_softmax(-1),'b t v -> t b v'),y.to(device),aux.cpu(),yl)
            else:
                logp=logits.log_softmax(-1)
                losses,_=pack([-logp[j,labels[i]].mean()*weights[labels[i]].mean() for j,i in enumerate(ii)],'*')
                loss=losses.mean()
                if name=='prototype':
                    primary=torch.tensor([labels[i][0] for i in ii],device=device)
                    same=rearrange(primary,'b -> b 1')==rearrange(primary,'b -> 1 b')
                    diagonal=torch.eye(len(ii),device=device,dtype=torch.bool)
                    positive=same & ~diagonal
                    similarity=aux@rearrange(aux,'b e -> e b')/.12
                    logprob=similarity.masked_fill(diagonal,-1e4).log_softmax(-1)
                    denominator=positive.sum(-1).clamp(min=1)
                    loss=loss+.3*(-(logprob*positive).sum(-1)/denominator).mean()
            if not torch.isfinite(loss):raise RuntimeError((name,'nonfinite loss'))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_total+=float(loss.detach());steps+=1
        model.eval();errors=0;units=0;class_scores=defaultdict(lambda:[0,0])
        with torch.inference_mode():
            for offset in range(0,len(dev),96):
                x,l,ii=collate(dev[offset:offset+96],features);logits,aux=model(x.to(device),l.to(device))
                logits=logits.cpu().numpy();aux=aux.cpu().numpy()
                for j,i in enumerate(ii):
                    if is_ctc:
                        pred=decode(logits[j],int(aux[j]),vocab);ref=[vocab[t] for t in targets[i]]
                        errors+=edit(ref,pred);units+=len(ref)
                    else:
                        prediction=int(logits[j].argmax())
                        for label in labels[i]:class_scores[label][0]+=int(prediction in labels[i]);class_scores[label][1]+=1
        score=errors/max(1,units) if is_ctc else 1-sum(a/n for a,n in class_scores.values())/max(1,len(class_scores))
        entry=dict(epoch=epoch,loss=loss_total/max(1,steps),development_error=score,seconds=time.perf_counter()-begin)
        history.append(entry);print(name,json.dumps(entry),flush=True);dump(OUT/f'{name}-history.json',history)
        if score<best and (not is_ctc or epoch>3):
            best=score;bad=0
            torch.save(dict(name=name,state_dict=model.state_dict(),config=model.config,vocabulary=vocab,epoch=epoch,
                            initialization='random_from_scratch',seed=seed,manifest_sha256=digest(DATA/'manifest.jsonl'),
                            development_error=score,train_examples=len(train),dev_examples=len(dev),excluded_targets=dict(excluded)),checkpoint)
        else:bad+=1
        scheduler.step()
        if bad>=9 and epoch>10:break
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']);parity=export(model,name)
    result={k:v for k,v in saved.items() if k!='state_dict'}
    result.update(seconds=time.perf_counter()-started,epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),
                  graph_bytes=(OUT/f'{name}.onnx').stat().st_size,graph_sha256=digest(OUT/f'{name}.onnx'),onnx_parity=parity)
    dump(OUT/f'{name}-training.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','scale','all'])
    parser.add_argument('--models',nargs='+',choices=['fixed','prototype','phoneme','character'],default=['fixed','prototype','phoneme','character'],help='models to train; evaluate compares all four frozen candidates')
    parser.add_argument('--epochs',type=int,default=28)
    args=parser.parse_args();rows=prepare()
    if args.action=='prepare':return
    if args.action=='scale':
        from .parameter_study_eval import scale_catalog
        print(json.dumps(scale_catalog(rows)),flush=True);return
    features=cached_features(rows)
    if args.action in ('train','all'):
        for name in args.models:train_one(name,rows,features,args.epochs)
    if args.action in ('evaluate','all'):
        from .parameter_study_eval import evaluate
        evaluate(rows,features)
