"""Fit isolated speech/name embedding candidates and export CPU ONNX graphs."""
from collections import Counter
import argparse
import json
import math
import random
import time
import numpy as np
import torch
from torch import nn
from einops import rearrange, reduce
from .speech_embedding_data import ROOT, DATA, OUT, prepare, token_data, preserved_check
from .parameter_study_data import dump, digest
from .parameter_study import cached_features, collate
from .speech_embedding import SpeechEmbedding, contrastive


def token_batch(sequences, device='cpu'):
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)
    values = torch.zeros(len(sequences), max(map(len,sequences)), dtype=torch.long, device=device)
    for j,s in enumerate(sequences):
        values[j,:len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return values, lengths


def development(model, rows, features, encode, device):
    """Select solely on dev; no test features are forwarded here."""
    catalog = json.loads((DATA/'catalogs.json').read_text())['expanded']
    catalog['isolated_word'] = sorted({r['text'] for r in rows if r['split']=='train' and r['kind']=='word'})
    vectors = {}
    with torch.inference_mode():
        for role,values in catalog.items():
            usable = [(v,encode(v)) for v in values]
            if role == 'number':
                from .levels import spoken_number
                usable = [(v,encode(spoken_number(int(v)))) for v in values]
            usable = [(v,s) for v,s in usable if s]
            if usable:
                t,l = token_batch([s for _,s in usable],device)
                vectors[role] = ([v for v,_ in usable], model.text(t,l))
        counts = {role:[0,0] for role in ('artist','song','isolated_word')}
        indices = [i for i,r in enumerate(rows) if r['split']=='dev' and
                   (r['kind']=='word' or any(t['role'] in ('artist','song') for t in r['targets']))]
        for first in range(0,len(indices),64):
            x,l,ii = collate(indices[first:first+64],features)
            windows,_ = model.audio(x.to(device),l.to(device))
            for role,(values,queries) in vectors.items():
                if role not in counts: continue
                scores = torch.einsum('bwe,qe->bwq', windows,queries)
                best = reduce(scores,'b w q -> b q','max').argmax(-1).cpu().tolist()
                for j,i in enumerate(ii):
                    refs = ([rows[i]['text']] if rows[i]['kind']=='word' else []) if role=='isolated_word' else [
                        t['value'] for t in rows[i]['targets'] if t['role']==role]
                    for ref in refs:
                        counts[role][1] += 1
                        counts[role][0] += values[best[j]] == ref
    score = sum(weight*counts[role][0]/max(1,counts[role][1]) for role,weight in
                [('artist',.4),('song',.4),('isolated_word',.2)])
    return score, counts


def train_one(name, rows, protocol, features):
    path = OUT/f'{name}.pt'
    if path.exists():
        raise ValueError('Candidate exists; preserve this version and use export/evaluate, or a new study version')
    seed = protocol['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab, full, targets, encode = token_data(rows)
    classes = ['__other__']+sorted({t['role']+'::'+t['value'] for r in rows if r['split']=='train' for t in r['targets']})
    lookup = {k:i for i,k in enumerate(classes)}
    labels = {i:[lookup[t['role']+'::'+t['value']] for t in r['targets']
                 if t['role']+'::'+t['value'] in lookup] or [0] for i,r in enumerate(rows)}
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    entities = [i for i in train if any(t[0] in ('artist','song','device') for t in targets[i])]
    counts = Counter(k for i in train for k in labels[i])
    weights = torch.tensor([1/math.sqrt(max(1,counts[k])) for k in range(len(classes))],device=device)
    weights = (weights/weights.mean()).clamp(.1,5.)
    model = SpeechEmbedding(len(classes),len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,protocol['epochs'],eta_min=.00015)
    best=-1.;bad=0;history=[];started=time.perf_counter()
    for epoch in range(1,protocol['epochs']+1):
        model.train();order=list(train);random.shuffle(order);total=0.;steps=0;begin=time.perf_counter()
        # Balanced sampling changes example frequency, never the recorded waveform.
        for first in range(0,len(order),48):
            indices=order[first:first+48]+random.choices(entities,k=protocol['entity_examples_per_batch'])
            x,l,ii=collate(indices,features)
            optimizer.zero_grad(set_to_none=True)
            windows, logits=model.audio(x.to(device),l.to(device))
            full_positions=[j for j,i in enumerate(ii) if i in full]
            loss=windows.sum()*0.
            if full_positions:
                sequences=[full[ii[j]] for j in full_positions]
                t,tl=token_batch(sequences,device);text=model.text(t,tl)
                audio=windows[full_positions,0]
                scores=audio@rearrange(text,'q e -> e q')/protocol['temperature']
                keys=[tuple(s) for s in sequences]
                same=torch.tensor([[a==b for b in keys] for a in keys],device=device)
                loss=reduce(contrastive(scores,same),'b ->','mean')
                loss=(loss+reduce(contrastive(rearrange(scores,'b q -> q b'),rearrange(same,'b q -> q b')),'b ->','mean'))/2
            phrases={tuple(s) for i in ii for role,value,s in targets[i]}
            phrases=sorted(phrases)
            if phrases:
                t,tl=token_batch(phrases,device);text=model.text(t,tl)
                scores=reduce(torch.einsum('bwe,qe->bwq',windows,text),'b w q -> b q','max')/protocol['temperature']
                positive=torch.tensor([[any(tuple(s)==p for _,_,s in targets[i]) for p in phrases] for i in ii],device=device)
                active=positive.any(-1)
                loss=loss+protocol['entity_loss_weight']*reduce(contrastive(scores[active],positive[active]),'b ->','mean')
            if name=='dual_kws':
                logp=logits.log_softmax(-1)
                losses=[-reduce(logp[j,labels[i]],'k ->','mean')*reduce(weights[labels[i]],'k ->','mean') for j,i in enumerate(ii)]
                loss=loss+protocol['kws_loss_weight']*reduce(rearrange(losses,'b -> b'),'b ->','mean')
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach());steps+=1
        model.eval();score,devcounts=development(model,rows,features,encode,device)
        entry=dict(epoch=epoch,loss=total/steps,development_score=score,development_counts=devcounts,seconds=time.perf_counter()-begin)
        history.append(entry);dump(OUT/f'{name}-history.json',history);print(name,json.dumps(entry),flush=True)
        if score>best:
            best=score;bad=0
            torch.save(dict(name=name,config=model.config,state_dict=model.state_dict(),classes=classes,vocabulary=vocab,
                            epoch=epoch,development_score=score,seed=seed,initialization='random',
                            protocol_sha256=digest(DATA/'protocol.json'),manifest_sha256=digest(DATA/'manifest.jsonl'),
                            training_full_transcripts=sum(i in full for i in train),training_rows=len(train)),path)
        else:bad+=1
        scheduler.step()
        if bad>=protocol['patience'] and epoch>=12:break
    dump(OUT/f'{name}-fitting.json',dict(seconds=time.perf_counter()-started,epochs_run=len(history),device=device,
         cuda_name=torch.cuda.get_device_name(0) if device=='cuda' else None,torch_version=torch.__version__,
         pretrained_weights=False,synthetic_audio=False,checkpoint_sha256=digest(path)))
    return export_one(name)


def load_model(name):
    saved=torch.load(OUT/f'{name}.pt',map_location='cpu',weights_only=False)
    if saved['protocol_sha256']!=digest(DATA/'protocol.json'):
        raise ValueError('Checkpoint protocol mismatch')
    model=SpeechEmbedding(**saved['config']);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved


def ort_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def export_one(name):
    torch.set_num_threads(4)
    model,saved=load_model(name);results={}
    for kind,module,inputs,input_names,output_names in [
        ('audio',model.audio,(torch.zeros(1,201,80),torch.tensor([201])),['features','lengths'],['embeddings','logits']),
        ('text',model.text,(torch.ones(1,21,dtype=torch.long),torch.tensor([21])),['tokens','lengths'],['embedding'])]:
        path=OUT/f'{name}-{kind}.onnx'
        axes={input_names[0]:{0:'batch',1:'time'},'lengths':{0:'batch'}}
        axes.update({n:{0:'batch'} for n in output_names})
        torch.onnx.export(module,inputs,path,dynamo=False,opset_version=17,input_names=input_names,output_names=output_names,dynamic_axes=axes)
        session=ort_session(path);checks=[]
        sizesets=([1],[13],[95],[201],[799],[301,177,95]) if kind=='audio' else ([1],[5],[21],[55],[35,17,5])
        for sizes in sizesets:
            values=torch.randn(len(sizes),max(sizes),80) if kind=='audio' else torch.randint(1,len(saved['vocabulary']),(len(sizes),max(sizes)))
            lengths=torch.tensor(sizes)
            with torch.inference_mode():expected=module(values,lengths)
            if not isinstance(expected,tuple):expected=(expected,)
            actual=session.run(None,{input_names[0]:values.numpy(),'lengths':lengths.numpy()})
            error=max(float(np.max(np.abs(a-e.numpy()))) for a,e in zip(actual,expected))
            if error>1e-4:raise AssertionError((kind,sizes,error))
            # Verify each padded member equals its own unpadded run.
            padding=[]
            for j,n in enumerate(sizes):
                single=session.run(None,{input_names[0]:values[j:j+1,:n].numpy(),'lengths':np.array([n],dtype='int64')})
                padding.append(max(float(np.max(np.abs(a[j]-s[0]))) for a,s in zip(actual,single)))
            if max(padding)>1e-4:raise AssertionError(('padding',kind,sizes,padding))
            checks.append(dict(lengths=sizes,torch_onnx_max_abs=error,batch_single_max_abs=max(padding)))
        results[kind]=dict(bytes=path.stat().st_size,sha256=digest(path),parameters=sum(p.numel() for p in module.parameters()),parity=checks)
    result={k:v for k,v in saved.items() if k!='state_dict'}
    result.update(graphs=results,total_parameters=sum(p.numel() for p in model.parameters()),fitting=json.loads((OUT/f'{name}-fitting.json').read_text()))
    dump(OUT/f'{name}-training.json',result);print('Exported',name,results['audio']['bytes'],results['text']['bytes'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','export','check','evaluate','benchmark','report','all','query'])
    parser.add_argument('--models',nargs='+',choices=['dual','dual_kws'],default=['dual','dual_kws'])
    parser.add_argument('--audio');parser.add_argument('--catalog',default='expanded');parser.add_argument('--role',default='artist')
    parser.add_argument('--intent',default='UNKNOWN',help='Already accepted VCM family; UNKNOWN refuses parameter resolution')
    args=parser.parse_args();rows,protocol=prepare()
    if args.action=='prepare':print(json.dumps({k:protocol[k] for k in ('version','counts','role_counts')}));return
    if args.action in ('train','all'):
        cache=cached_features(rows)
        for name in args.models:train_one(name,rows,protocol,cache)
    if args.action=='export':
        for name in args.models:export_one(name)
    if args.action in ('check','all'):
        from .speech_embedding_eval import checks
        print(json.dumps(checks(rows,protocol)),flush=True)
    if args.action in ('evaluate','all'):
        from .speech_embedding_eval import evaluate
        evaluate(rows,protocol,cached_features(rows))
    if args.action in ('benchmark','all'):
        from .speech_embedding_eval import benchmark
        benchmark(rows,cached_features(rows))
    if args.action in ('report','all'):
        from .speech_embedding_eval import report
        report(rows,protocol)
    if args.action=='query':
        from .speech_embedding_eval import query_file
        print(json.dumps(query_file(args),indent=2),flush=True)
    preserved_check(protocol)
