"""Joint human/TTS command fitting with fixed-word CTC supervision."""
from collections import Counter
import json
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, pack
from .keyword_data import DATA,CACHE,REPORTS,HEADS,VOCAB,INDEX,prepare,dataset,feature_list,digest,dump
from .keyword_vcm import KeywordVCM
from .keyword_runtime import decode_record


def batch(data,arrays,ids,device):
    lengths=[arrays[i].shape[1] for i in ids];x=np.zeros((len(ids),80,max(lengths)),dtype=np.float32)
    for j,i in enumerate(ids):x[j,:,:lengths[j]]=arrays[i]
    target=torch.tensor([INDEX[w] for i in ids for w in data[i]['keyword_tokens']],dtype=torch.long,device=device)
    target_lengths=torch.tensor([len(data[i]['keyword_tokens']) for i in ids],dtype=torch.long)
    intent=torch.tensor([HEADS['intent'].index(data[i]['labels']['intent']) if data[i]['labels'] else -100 for i in ids],device=device)
    ready=torch.tensor([int(data[i]['labels']['ready']=='READY') if data[i]['task']=='command' else -100 for i in ids],device=device)
    return torch.from_numpy(x).to(device),torch.tensor(lengths,device=device),target,target_lengths,intent,ready


def semantic_metrics(outputs,data):
    groups={c:[] for c in HEADS['intent']};families={c:[] for c in HEADS['intent']};token_correct=0
    for o,r in zip(outputs,data):
        y=r['labels'];c=y['intent'];families[c].append(o['family']==c)
        correct=(o['family']==c and o['valid'] and o['labels']==y) if y['ready']=='READY' else (not o['valid'])
        groups[c].append(correct)
        token_correct += [x['word'] for x in o['cues']]==r['keyword_tokens']
    macro_family=float(np.mean([np.mean(v) for v in families.values() if v]));macro_exact=float(np.mean([np.mean(v) for v in groups.values() if v]))
    return {'score':.5*(macro_family+macro_exact),'macro_family':macro_family,'macro_exact':macro_exact,'keyword_sequence_exact':token_correct/len(data),
        'per_intent':{c:{'n':len(groups[c]),'family':float(np.mean(families[c])) if groups[c] else None,'exact':float(np.mean(groups[c])) if groups[c] else None} for c in groups}}


@torch.inference_mode()
def predict(model,data,arrays,device='cuda'):
    model.eval();results=[]
    for offset in range(0,len(data),64):
        ids=list(range(offset,min(offset+64,len(data))));x,lengths,*_=batch(data,arrays,ids,device)
        outputs=[v.float().cpu().numpy() for v in model(x,lengths)]
        for i in range(len(ids)):results.append(decode_record(outputs[0][i],outputs[1][i],outputs[2][i],int(outputs[3][i])))
    return results


def export(mode):
    out=CACHE/mode;saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    if saved['protocol_sha256']!=digest(DATA/'protocol.json'):raise ValueError('Protocol changed')
    model=KeywordVCM(saved['width']).eval();model.load_state_dict(saved['state'])
    torch.set_num_threads(4)
    torch.onnx.export(model,(torch.randn(2,80,201),torch.tensor([201,137])),str(out/'model.onnx'),input_names=['features','lengths'],
        output_names=['intent','ready','keywords','output_lengths'],dynamic_axes={'features':{0:'batch',2:'frames'},'lengths':{0:'batch'},'intent':{0:'batch'},'ready':{0:'batch'},'keywords':{0:'batch',1:'output_frames'},'output_lengths':{0:'batch'}},opset_version=17,dynamo=False)
    receipt={'selected_epoch':saved['epoch'],'development':saved['development'],'width':saved['width'],'parameters':sum(p.numel() for p in model.parameters()),
        'onnx_bytes':(out/'model.onnx').stat().st_size,'model_sha256':digest(out/'model.onnx'),'checkpoint_sha256':digest(out/'best.pt'),'random_initialization':True,'test_used':False}
    dump(out/'export.json',receipt);dump(REPORTS[mode]/'training.json',receipt)
    print('EXPORTED',mode,json.dumps(receipt),flush=True)


def fit(mode):
    protocol=prepare();out=CACHE/mode;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Fit already complete; preserve the study version')
    torch.set_num_threads(4);torch.manual_seed(23);np.random.seed(23);random.seed(23)
    torch.backends.cuda.matmul.allow_tf32=True
    data=dataset(mode,'train');dev=dataset(mode,'dev')
    print('FEATURES',mode,len(data),len(dev),flush=True);arrays=feature_list(data);dev_arrays=feature_list(dev)
    model=KeywordVCM(protocol['width']).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'],eta_min=.00004)
    weights=[]
    strata=[('word',r['transcript']) if r['task']=='word' else ('intent',r['labels']['intent']) if r['task']=='intent' else ('context',r['source']) if r['task']=='context' else (r['kind'],r['labels']['intent']) for r in data]
    counts=Counter(strata);kinds=Counter(s[0] for s in counts)
    fractions={'human':.45,'synthetic':.25,'word':.1,'intent':.15,'context':.05} if mode=='mixed' else {'synthetic':1.}
    for stratum in strata:weights.append(fractions[stratum[0]]/kinds[stratum[0]]/counts[stratum])
    weights=torch.tensor(weights,dtype=torch.double);history=[];best=-1.;started=time.perf_counter()
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=True,reduction='none')
    for epoch in range(protocol['epochs']):
        model.train();loss_sum=0.;steps=0;indices=torch.multinomial(weights,len(data),replacement=True).tolist()
        # Length buckets reduce zero padding; recordings are not cropped or augmented.
        batches=[]
        for j in range(0,len(indices),512):
            group=sorted(indices[j:j+512],key=lambda i:arrays[i].shape[1])
            batches.extend(group[k:k+protocol['batch_size']] for k in range(0,len(group),protocol['batch_size']))
        random.shuffle(batches)
        for ids in batches:
            x,lengths,target,target_lengths,intent,ready=batch(data,arrays,ids,'cuda');optimizer.zero_grad(set_to_none=True)
            fi,fr,kw,ol=model(x,lengths)
            ctc=criterion(rearrange(kw.log_softmax(-1),'b t v -> t b v'),target,ol,target_lengths.to('cuda'))
            ctc=(ctc/target_lengths.to('cuda').clamp_min(1)).mean()
            supervised=intent!=-100
            loss=ctc
            if supervised.any():loss=loss+F.cross_entropy(fi[supervised],intent[supervised])
            complete=ready!=-100
            if complete.any():loss=loss+.5*F.cross_entropy(fr[complete],ready[complete])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();loss_sum+=float(loss.detach());steps+=1
        schedule.step();metrics=semantic_metrics(predict(model,dev,dev_arrays),dev)
        entry={'epoch':epoch+1,'loss':loss_sum/steps,'development':metrics,'seconds':time.perf_counter()-started};history.append(entry)
        dump(out/'history.json',history)
        if metrics['score']>best:
            best=metrics['score'];torch.save({'state':model.state_dict(),'width':protocol['width'],'epoch':epoch+1,'development':metrics,'protocol_sha256':digest(DATA/'protocol.json')},out/'best.pt')
        print('EPOCH',mode,epoch+1,'loss',round(loss_sum/steps,4),'dev_family',round(metrics['macro_family'],4),'dev_exact',round(metrics['macro_exact'],4),'token_exact',round(metrics['keyword_sequence_exact'],4),'seconds',round(time.perf_counter()-started),flush=True)
    dump(out/'complete.json',{'epochs':protocol['epochs'],'seconds':time.perf_counter()-started,'n_train':len(data),'n_dev':len(dev),'history_sha256':digest(out/'history.json')})
    dump(REPORTS[mode]/'history.json',history);export(mode)


def fit_intent():
    from .keyword_intent import MicroIntent
    out=CACHE/'micro-intent';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Micro intent candidate already fitted')
    torch.set_num_threads(4);torch.manual_seed(29);np.random.seed(29);random.seed(29)
    torch.backends.cuda.matmul.allow_tf32=True
    data=[r for r in dataset('mixed','train') if r['labels'] is not None];dev=dataset('mixed','dev')
    arrays=feature_list(data);dev_arrays=feature_list(dev);model=MicroIntent().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.005);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40,eta_min=.00003)
    counts=Counter((r['kind'],r['labels']['intent']) for r in data)
    weights=torch.tensor([(.75 if r['kind']=='human' else .25)/counts[(r['kind'],r['labels']['intent'])] for r in data],dtype=torch.double)
    history=[];best=-1.;started=time.perf_counter()
    spec={'version':'keyword-v4-micro-intent','epochs':40,'seed':29,'selection':'Human development macro family accuracy; ready accuracy breaks ties. No test selection.',
        'architecture':'Five small 2D convolutions, global max pooling, 64-unit projection, family/readiness/auxiliary keyword-presence heads.',
        'reference':'https://github.com/AIWintermuteAI/Speech-to-Intent-Micro','random_initialization':True,'data_protocol_sha256':digest(DATA/'protocol.json')}
    dump(REPORTS['mixed']/'micro-intent-protocol.json',spec)
    for epoch in range(40):
        model.train();indices=torch.multinomial(weights,len(data),replacement=True).tolist();batches=[]
        for j in range(0,len(indices),384):
            ordered=sorted(indices[j:j+384],key=lambda i:arrays[i].shape[1]);batches.extend(ordered[k:k+48] for k in range(0,len(ordered),48))
        random.shuffle(batches);total=0.
        for ids in batches:
            x,l,_,_,y,ready=batch(data,arrays,ids,'cuda');fi,fr,words=model(x,l);keyword=torch.zeros_like(words)
            for j,i in enumerate(ids):keyword[j,[INDEX[w] for w in data[i]['keyword_tokens'] if w not in ('<blank>','<other>')]]=1.
            loss=F.cross_entropy(fi,y)+.5*F.binary_cross_entropy_with_logits(words,keyword,pos_weight=torch.full((len(VOCAB),),3.,device='cuda'))
            mask=ready!=-100
            if mask.any():loss=loss+.5*F.cross_entropy(fr[mask],ready[mask])
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();total+=float(loss.detach())
        scheduler.step();model.eval();family=[];readiness=[]
        with torch.inference_mode():
            for j in range(0,len(dev),48):
                ids=list(range(j,min(j+48,len(dev))));x,l,*_=batch(dev,dev_arrays,ids,'cuda');a,b,_=model(x,l)
                family.extend(a.argmax(-1).cpu().tolist());readiness.extend(b.argmax(-1).cpu().tolist())
        per={c:float(np.mean([family[i]==HEADS['intent'].index(c) for i,r in enumerate(dev) if r['labels']['intent']==c])) for c in HEADS['intent']}
        accuracy=float(np.mean(list(per.values())));ra=float(np.mean([v==int(r['labels']['ready']=='READY') for v,r in zip(readiness,dev)]))
        entry={'epoch':epoch+1,'loss':total/len(batches),'macro_family':accuracy,'ready_accuracy':ra,'per_intent':per,'seconds':time.perf_counter()-started};history.append(entry);dump(out/'history.json',history)
        score=accuracy+.001*ra
        if score>best:
            best=score;torch.save({'state':model.state_dict(),'epoch':epoch+1,'development':entry},out/'best.pt')
        print('MICRO EPOCH',epoch+1,'macro family',round(accuracy,4),'ready',round(ra,4),'seconds',round(time.perf_counter()-started),flush=True)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model=MicroIntent().eval();model.load_state_dict(saved['state'])
    torch.onnx.export(model,(torch.randn(2,80,201),torch.tensor([201,137])),str(out/'model.onnx'),input_names=['features','lengths'],output_names=['intent','ready','words'],
        dynamic_axes={'features':{0:'batch',2:'frames'},'lengths':{0:'batch'},**{k:{0:'batch'} for k in ('intent','ready','words')}},opset_version=17,dynamo=False)
    receipt={'selected_epoch':saved['epoch'],'development':saved['development'],'parameters':sum(p.numel() for p in model.parameters()),'model_sha256':digest(out/'model.onnx'),'onnx_bytes':(out/'model.onnx').stat().st_size,'epochs':40,'seconds':time.perf_counter()-started}
    dump(out/'complete.json',receipt);dump(REPORTS['mixed']/'micro-intent-training.json',receipt);dump(REPORTS['mixed']/'micro-intent-history.json',history)
