"""Reproducible vocabulary pretraining and command CTC training from random weights."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange

from common import ROOT, CHARS, NUM_WORDS, say_number, digest, read_jsonl, write_json, edit_distance
from model import AcousticCTC, greedy_decode

SEED = 20260919
TOKEN = {c: i for i,c in enumerate(CHARS)}

def load_features(rows):
    values = []
    for i,r in enumerate(rows):
        values.append(torch.from_numpy(np.load(ROOT/r['feature']).astype(np.float32)))
        if i % 20000 == 0:
            print('Load features',i,'/',len(rows),flush=True)
    return values

def collate(items, augment=False):
    features, texts = zip(*items)
    lengths = torch.tensor([len(x) for x in features], dtype=torch.long)
    x = pad_sequence(features, batch_first=True)
    if augment:
        for i, length in enumerate(lengths.tolist()):
            f = random.randint(0,8)
            f0 = random.randint(0,80-f)
            x[i,:,f0:f0+f] = 0
            t = random.randint(0,min(12,length//8))
            t0 = random.randint(0,length-t)
            x[i,t0:t0+t,:] = 0
    targets = [torch.tensor([TOKEN[c] for c in t], dtype=torch.long) for t in texts]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return x, lengths, pad_sequence(targets, batch_first=True), target_lengths

def batches(indices, lengths, batch_size, rng):
    indices = list(indices)
    rng.shuffle(indices)
    out = []
    for start in range(0,len(indices),batch_size*30):
        chunk = sorted(indices[start:start+batch_size*30], key=lambda i:lengths[i])
        out.extend(chunk[j:j+batch_size] for j in range(0,len(chunk),batch_size))
    rng.shuffle(out)
    return out

def number_compositions(rows, features, split, count, seed):
    """Concatenations of real word crops, explicitly NOT natural connected speech."""
    rng = random.Random(seed)
    words = defaultdict(list)
    for r,x in zip(rows,features):
        if r['split']==split and r['kind']=='human_word' and r['text'] in NUM_WORDS:
            words[r['text']].append(x)
    made = []
    for i in range(count):
        value = i % 101
        text = say_number(value)
        if not all(words[w] for w in text.split()):
            continue
        pieces = [rng.choice(words[w]) for w in text.split()]
        # einops packs variable-time tensors with explicit axes.
        from einops import pack
        joined, _ = pack(pieces, '* mel')
        made.append((joined,text))
    return made

@torch.inference_mode()
def evaluate(model, items, device, batch_size=64):
    model.eval()
    word_edits=words=char_edits=chars=exact=0
    loss_total=0
    loss_fn=nn.CTCLoss(blank=0,zero_infinity=False)
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]
        x,l,y,yl=collate(batch)
        logits,ll=model(x.to(device),l)
        loss=loss_fn(rearrange(logits.log_softmax(-1),'batch time chars -> time batch chars'),y.to(device),ll,yl)
        loss_total+=float(loss)*len(batch)
        predicted=greedy_decode(logits,ll)
        for (_,ref),hyp in zip(batch,predicted):
            word_edits+=edit_distance(ref.split(),hyp.split());words+=len(ref.split())
            char_edits+=edit_distance(ref,hyp);chars+=len(ref)
            exact+=ref==hyp
    return {'n':len(items),'loss':loss_total/max(1,len(items)),'wer':word_edits/max(1,words),'cer':char_edits/max(1,chars),'exact':exact/max(1,len(items))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--word-epochs',type=int,default=10)
    parser.add_argument('--command-epochs',type=int,default=28)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=read_jsonl(ROOT/'data/manifest.jsonl')
    # Training caps are fixed before evaluation; all selected song commands are retained.
    rng=random.Random(SEED)
    capped=[];by_word=defaultdict(list)
    for r in rows:
        if r['kind']=='human_word':
            by_word[r['source']+'/'+r['split']+'/'+r['text']].append(r)
        else:
            capped.append(r)
    for key,rr in by_word.items():
        rng.shuffle(rr)
        capped.extend(rr[:800 if rr[0]['split']=='train' else 150])
    rows=capped
    features=load_features(rows)
    train_words=[(x,r['text']) for r,x in zip(rows,features) if r['split']=='train' and r['kind']=='human_word']
    train_commands=[(x,r['text']) for r,x in zip(rows,features) if r['split']=='train' and r['kind']!='human_word']
    dev_words=[(x,r['text']) for r,x in zip(rows,features) if r['split']=='dev' and r['kind']=='human_word']
    dev_commands=[(x,r['text']) for r,x in zip(rows,features) if r['split']=='dev' and r['kind']!='human_word']
    rng.shuffle(dev_words);dev_words=dev_words[:1000]
    rng.shuffle(dev_commands);dev_commands=dev_commands[:1800]
    compositions=number_compositions(rows,features,'train',6060,SEED)
    word_training=train_words+compositions
    replay=rng.sample(train_words,min(6000,len(train_words)))
    command_training=train_commands+replay+compositions
    out=ROOT/'runs/scratch_ctc'
    out.mkdir(parents=True,exist_ok=True)
    model=AcousticCTC().to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    init_digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    config={'seed':SEED,'initialization':'PyTorch random weights; no external checkpoint loaded',
            'initial_weights_sha256':init_digest,'parameters':sum(p.numel() for p in model.parameters()),
            'torch':torch.__version__,'numpy':np.__version__,'device':device,
            'gpu':torch.cuda.get_device_name() if device=='cuda' else None,'python':os.sys.version,
            'word_pretraining_examples':len(word_training),'command_training_examples':len(command_training),
            'train_human_words':len(train_words),'train_human_commands':len(train_commands),
            'constructed_number_sequences':len(compositions),'dev_words':len(dev_words),'dev_commands':len(dev_commands),
            'args':vars(args),'manifest_sha256':hashlib.sha256((ROOT/'data/manifest.jsonl').read_bytes()).hexdigest(),
            'characters':CHARS,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(out/'config.json',config)
    print(json.dumps(config,indent=2),flush=True)
    loss_fn=nn.CTCLoss(blank=0,zero_infinity=False)
    all_history=[];steps=0;started=time.monotonic()
    stages=[('words',word_training,dev_words,args.word_epochs,.0015),('commands',command_training,dev_commands,args.command_epochs,.001)]
    for stage,items,dev,epochs,lr in stages:
        if not items:
            continue
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        best=float('inf');best_epoch=0
        lengths=[len(x) for x,_ in items]
        for epoch in range(1,epochs+1):
            epoch_start=time.monotonic()
            model.train();total=0;n=0
            epoch_lr=lr*(.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/max(1,epochs-1))))
            for group in optimizer.param_groups:group['lr']=epoch_lr
            for ids in batches(range(len(items)),lengths,args.batch_size,rng):
                batch=[items[i] for i in ids]
                x,l,y,yl=collate(batch,augment=True)
                optimizer.zero_grad(set_to_none=True)
                logits,ll=model(x.to(device),l)
                loss=loss_fn(rearrange(logits.log_softmax(-1),'batch time chars -> time batch chars'),y.to(device),ll,yl)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite CTC loss at {stage} epoch {epoch}')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step();steps+=1
                total+=float(loss.detach())*len(ids);n+=len(ids)
            metrics=evaluate(model,dev,device)
            history={'stage':stage,'epoch':epoch,'train_loss':total/n,'dev':metrics,'lr':epoch_lr,'seconds':time.monotonic()-epoch_start,'steps':steps}
            all_history.append(history)
            if metrics['cer']<best:
                best=metrics['cer'];best_epoch=epoch
                torch.save({'state':model.state_dict(),'config':config,'stage':stage,'epoch':epoch,'dev':metrics,'steps':steps},out/(stage+'_best.pt'))
            write_json(out/'history.json',all_history)
            print(json.dumps(history),flush=True)
        checkpoint=torch.load(out/(stage+'_best.pt'),map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['state'])
        print(f'Selected {stage} epoch {best_epoch}, development CER {best:.4f}',flush=True)
    final={k:v.detach().cpu() for k,v in model.state_dict().items()}
    delta=math.sqrt(sum(float(((v-initial[k])**2).sum()) for k,v in final.items()))
    write_json(out/'completed.json',{'completed':True,'seconds':time.monotonic()-started,'optimizer_steps':steps,
        'parameter_l2_change_from_random_initialization':delta,'final_selected_stage':checkpoint['stage'],
        'final_selected_epoch':checkpoint['epoch'],'test_audio_used_for_selection':False,
        'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

if __name__=='__main__':main()
