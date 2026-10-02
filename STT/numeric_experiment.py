"""Separate randomly initialized finite-vocabulary CTC numeral experiment."""
from __future__ import annotations
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from einops import pack,rearrange
from common import ROOT,ONES,TENS,NUM_WORDS,read_jsonl,write_json,write_jsonl,say_number,parse_number,extract_number
from model import AcousticCTC
from prepare import cache_one
from train import batches,SEED

VOCAB=['<blank>','<unknown>']+ONES+TENS[2:]+['hundred']
INDEX={w:i for i,w in enumerate(VOCAB)}

def collate(items,augment=False):
    features=[r['x'] for r in items]
    x=pad_sequence(features,batch_first=True)
    lengths=torch.tensor([len(f) for f in features])
    if augment:
        for i,l in enumerate(lengths.tolist()):
            f0=random.randrange(72);x[i,:,f0:f0+random.randrange(1,9)]=0
            t0=random.randrange(max(1,l-10));x[i,t0:t0+random.randrange(1,10)]=0
    labels=[torch.tensor([INDEX[w] for w in r['target'].split()]) for r in items]
    return x,lengths,pad_sequence(labels,batch_first=True),torch.tensor([len(y) for y in labels])

def make_compositions(records,split,count,seed):
    rng=random.Random(seed);words=defaultdict(list)
    for r in records:
        if r['split']==split and r['kind']=='human_word' and r['target'] in NUM_WORDS:
            words[r['target']].append(r)
    out=[]
    for i in range(count):
        value=i%101
        target='hundred' if value==100 else say_number(value)
        if any(not words[w] for w in target.split()):continue
        pieces=[rng.choice(words[w]) for w in target.split()]
        x,_=pack([r['x'] for r in pieces],'* mel')
        out.append({'id':f'composed/{split}/{i}','x':x,'target':target,'value':value,'split':split,
            'source':'Concatenated human words','kind':'constructed_number','components':[r['id'] for r in pieces]})
    return out

@torch.inference_mode()
def predict(model,items,device):
    model.eval();out=[]
    for start in range(0,len(items),64):
        batch=items[start:start+64];x,l,_,_=collate(batch)
        logits,ll=model(x.to(device),l);probs=logits.softmax(-1)
        conf,ids=probs.max(-1)
        ids=ids.cpu().tolist();conf=conf.cpu().tolist()
        for row,tokens,scores,length in zip(batch,ids,conf,ll.tolist()):
            words=[];confidence=[];last=-1
            for token,score in zip(tokens[:length],scores[:length]):
                if token and token!=last:words.append(VOCAB[token]);confidence.append(score)
                last=token
            hyp=' '.join(words)
            value=None if '<unknown>' in words else parse_number(hyp)
            out.append({k:v for k,v in row.items() if k!='x'}|{'hypothesis':hyp,'prediction':value,
                'confidence':min(confidence) if confidence else 0})
    return out

def metrics(rows,threshold=0):
    valid=[r for r in rows if r['value'] is not None]
    accepted=[r for r in rows if r['prediction'] is not None and r['confidence']>=threshold]
    correct=sum(r['prediction']==r['value'] for r in accepted)
    ood=[r for r in rows if r['value'] is None]
    return {'n':len(rows),'valid':len(valid),'ood':len(ood),'accepted':len(accepted),'correct_accepted':correct,
        'raw_exact_value_accuracy':sum(r['prediction']==r['value'] for r in valid)/max(1,len(valid)),
        'accepted_precision':correct/max(1,len(accepted)),'valid_value_recall':correct/max(1,len(valid)),
        'ood_false_accepts':sum(r['value'] is None for r in accepted),
        'ood_false_accept_rate':sum(r['value'] is None for r in accepted)/max(1,len(ood))}

def main():
    while not (ROOT/'evidence/mlend_acquisition.json').exists():
        print('Waiting for completed MLEnd download',flush=True);time.sleep(15)
    torch.set_num_threads(4);torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device='cuda' if torch.cuda.is_available() else 'cpu';rng=random.Random(SEED)
    main_rows=read_jsonl(ROOT/'data/manifest.jsonl')
    rows=read_jsonl(ROOT/'data/mlend.jsonl')+[r for r in main_rows if r['source']=='FSDD']
    # Completed, decodable MSWC clips can be retained even if an upstream
    # archive stream was interrupted; per-clip evidence is saved below.
    selection_path=ROOT/'evidence/mswc_selected.json'
    if selection_path.exists():
        for r in json.loads(selection_path.read_text()).values():
            if r['text'] in NUM_WORDS and (ROOT/r['audio']).exists():rows.append(r)
    print('Numeric human word candidates',len(rows),flush=True)
    (ROOT/'data/features').mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=[r for r in pool.map(cache_one,rows) if r is not None]
    # Cap extra MSWC classes to avoid an incomplete download dominating numerals.
    kept=[];counts=Counter()
    for r in rows:
        key=(r['source'],r['split'],r['text'])
        if r['source']=='MSWC-English' and counts[key]>=150:continue
        counts[key]+=1;kept.append(r)
    rows=kept
    for split in ['train','dev','test']:
        negative=[r for r in main_rows if r['split']==split and r['kind']!='human_word' and extract_number(r['text']) is None]
        rng.shuffle(negative)
        for r in negative[:1500 if split=='train' else 250]:
            r=dict(r);r['text']='<unknown>';r['value']=None;r['kind']='human_non_numeric'
            rows.append(r)
    frozen_manifest=ROOT/'data/numeric_manifest.jsonl'
    if frozen_manifest.exists():
        rows=read_jsonl(frozen_manifest)
    else:
        write_jsonl(frozen_manifest,rows)
    speaker_sets={s:{r['speaker'] for r in rows if r['split']==s} for s in ['train','dev','test']}
    assert not (speaker_sets['train']&speaker_sets['dev'] or speaker_sets['train']&speaker_sets['test'] or speaker_sets['dev']&speaker_sets['test'])
    records=[]
    for r in rows:
        x=torch.from_numpy(np.load(ROOT/r['feature']).astype(np.float32))
        records.append({'id':r['id'],'x':x,'target':r['text'],'value':parse_number(r['text']),
            'split':r['split'],'source':r['source'],'kind':r['kind'],'speaker':r['speaker']})
    train_human=[r for r in records if r['split']=='train']
    dev_human=[r for r in records if r['split']=='dev']
    test_human=[r for r in records if r['split']=='test']
    composed={s:make_compositions(records,s,n,SEED+i) for i,(s,n) in enumerate([('train',5050),('dev',505),('test',1010)])}
    model=AcousticCTC(len(VOCAB)).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    out=ROOT/'runs/numeric_ctc';out.mkdir(parents=True,exist_ok=True)
    config={'vocabulary':VOCAB,'initialization':'Random initialization; no pretrained weights; independent of character CTC',
        'seed':SEED,'parameters':sum(p.numel() for p in model.parameters()),
        'initial_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest(),
        'human_split_counts':dict(Counter(r['source']+'/'+r['split'] for r in records)),
        'composed_split_counts':{s:len(v) for s,v in composed.items()},'speaker_overlap':0,
        'manifest_sha256':hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()}
    write_json(out/'config.json',config)
    history=[];steps=0;started=time.monotonic()
    for stage,items,epochs,lr in [('words',train_human,12,.0015),('compositions',train_human+composed['train'],24,.0008)]:
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        best=-1
        for epoch in range(1,epochs+1):
            start=time.monotonic();model.train();total=n=0
            rate=lr*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,epochs-1))))
            for group in optimizer.param_groups:group['lr']=rate
            for ids in batches(range(len(items)),[len(r['x']) for r in items],64,rng):
                batch=[items[i] for i in ids];x,l,y,yl=collate(batch,True)
                logits,ll=model(x.to(device),l)
                loss=nn.functional.ctc_loss(rearrange(logits.log_softmax(-1),'batch time classes -> time batch classes'),y.to(device),ll,yl,blank=0,zero_infinity=False)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite numeric CTC loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                total+=float(loss.detach())*len(ids);n+=len(ids);steps+=1
            dev=predict(model,dev_human+(composed['dev'] if stage=='compositions' else []),device)
            m=metrics(dev)
            score=m['raw_exact_value_accuracy']-m['ood_false_accept_rate']*.25
            h={'stage':stage,'epoch':epoch,'train_loss':total/n,'dev':m,'seconds':time.monotonic()-start,'steps':steps};history.append(h)
            if score>best:
                best=score
                torch.save({'state':model.state_dict(),'config':config,'stage':stage,'epoch':epoch,'score':score},out/(stage+'_best.pt'))
            write_json(out/'history.json',history);print(json.dumps(h),flush=True)
        chosen=torch.load(out/(stage+'_best.pt'),map_location=device,weights_only=False);model.load_state_dict(chosen['state'])
    dev=predict(model,dev_human+composed['dev'],device)
    trials=[{'threshold':t,**metrics(dev,t)} for t in [0,.3,.5,.6,.7,.8,.9,.95,.98,.99]]
    eligible=[r for r in trials if r['accepted_precision']>=.95 and r['correct_accepted']>=20]
    chosen_threshold=max(eligible,key=lambda x:x['valid_value_recall'])['threshold'] if eligible else 0
    test=predict(model,test_human+composed['test'],device)
    per_value={str(v):{'human_n':sum(r['value']==v and r['kind']=='human_word' for r in test),
        'human_correct':sum(r['value']==v and r['prediction']==v and r['kind']=='human_word' for r in test),
        'composed_n':sum(r['value']==v and r['kind']=='constructed_number' for r in test),
        'composed_correct':sum(r['value']==v and r['prediction']==v and r['kind']=='constructed_number' for r in test)} for v in range(101)}
    results={'threshold':chosen_threshold,'development_precision_gate_met':bool(eligible),'calibration':trials,
        'overall':metrics(test,chosen_threshold),
        'human_isolated':metrics([r for r in test if r['kind']=='human_word'],chosen_threshold),
        'constructed_sequences':metrics([r for r in test if r['kind']=='constructed_number'],chosen_threshold),
        'natural_non_numeric':metrics([r for r in test if r['kind']=='human_non_numeric'],chosen_threshold),
        'by_source':{s:metrics([r for r in test if r['source']==s],chosen_threshold) for s in sorted({r['source'] for r in test})},
        'per_value':per_value,'missing_natural_values':[v for v in range(101) if per_value[str(v)]['human_n']==0],
        'seconds':time.monotonic()-started,'optimizer_steps':steps,'selected_epoch':chosen['epoch'],
        'parameter_l2_change_from_initial':math.sqrt(sum(float(((v.detach().cpu()-initial[k])**2).sum()) for k,v in model.state_dict().items())),
        'checkpoint_sha256':hashlib.sha256((out/'compositions_best.pt').read_bytes()).hexdigest(),
        'test_selection':'No test examples used for fitting, epoch selection or threshold tuning.',
        'limitations':'Constructed sequences concatenate held-out human word features. They are not natural connected speech or complete volume commands.'}
    write_jsonl(out/'predictions.jsonl',test)
    write_json(out/'evaluation.json',results)
    write_json(out/'completed.json',{'completed':True,'seconds':results['seconds'],'optimizer_steps':steps})
    print(json.dumps({k:v for k,v in results.items() if k not in {'per_value','calibration'}},indent=2),flush=True)

if __name__=='__main__':main()
