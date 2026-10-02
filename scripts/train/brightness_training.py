"""Reproducible from-scratch brightness STT training and ONNX export."""
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch import nn
from einops import rearrange, pack
from .brightness_data import prepare, load_pcm, write_json, sha
from .stt import BrightnessSTT
from .stt_runtime import stt_features, decode_words
from .parameter_backend import parse_brightness


def collate(indices, cached, rows, lookup):
    indices=sorted(indices,key=lambda i:len(cached[i]),reverse=True)
    lengths=torch.tensor([len(cached[i]) for i in indices],dtype=torch.long)
    x=torch.zeros(len(indices),int(lengths[0]),80)
    targets=[];target_lengths=[]
    for j,i in enumerate(indices):
        x[j,:lengths[j]]=torch.from_numpy(cached[i])
        tokens=[lookup.get(w,1) for w in rows[i]['text'].split()]
        targets.append(torch.tensor(tokens,dtype=torch.long));target_lengths.append(len(tokens))
    y,_=pack(targets,'*')
    return x,lengths,y,torch.tensor(target_lengths,dtype=torch.long),indices


def fit_brightness(root,config,*,extra_manifest=None,refresh=False):
    root=Path(root);out=root/'.cache/stt-human';out.mkdir(parents=True,exist_ok=True)
    settings=config['training'];seed=settings['seed']
    limited=bool(settings.get('max_train_examples',0) or settings.get('max_dev_examples',0))
    if limited and settings.get('update_artifacts',False):
        raise ValueError('Limited STT debug training cannot replace artifacts; set update_artifacts=false')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    device=torch.device(settings['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    rows=prepare(root,extra_manifest,refresh)
    vocabulary=json.loads((root/'data/stt-human/vocabulary.json').read_text())
    lookup={w:i for i,w in enumerate(vocabulary)}
    fingerprint=sha(root/'data/stt-human/brightness.jsonl')
    from . import stt_runtime
    frontend_sha=sha(stt_runtime.__file__)
    feature_path=out/f'features-{fingerprint[:16]}-{frontend_sha[:12]}.pt'
    if feature_path.exists():
        cached=torch.load(feature_path,weights_only=False)['features']
    else:
        cached=[];audio_cache={}
        for i,row in enumerate(rows):
            pcm=load_pcm(row,root,audio_cache)
            cached.append(stt_features(pcm))
            if i%1000==0:print('Features',i,'/',len(rows),flush=True)
        torch.save({'features':cached,'manifest_sha256':fingerprint,'frontend_sha256':frontend_sha},feature_path)
    allowed=[i for i,x in enumerate(cached) if 10<=len(x)<=801]
    train=[i for i in allowed if rows[i]['split']=='train']
    dev=[i for i in allowed if rows[i]['split']=='dev']
    if settings.get('max_train_examples',0):
        train=random.Random(seed).sample(train,min(len(train),settings['max_train_examples']))
    if settings.get('max_dev_examples',0):
        dev=random.Random(seed+1).sample(dev,min(len(dev),settings['max_dev_examples']))
    warmup=[i for i in train if rows[i]['kind']=='word']
    if not train or not dev or not warmup:raise ValueError('Training requires human words and distinct train/dev data')
    model=BrightnessSTT(len(vocabulary),settings.get('stt_width',128),settings.get('stt_layers',2)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.get('stt_learning_rate',.001),weight_decay=.001)
    ctc=nn.CTCLoss(blank=0,zero_infinity=True)
    batch_size=settings.get('stt_batch_size',64)
    warmup_epochs=settings.get('stt_word_epochs',8);epochs=settings.get('stt_epochs',36)
    catalog=json.loads((root/'artifacts/bundle.json').read_text())['catalog']
    best=-1.;best_epoch=0;started=time.perf_counter();history=[]
    checkpoint=out/'stt.pt'
    log=(out/'training.jsonl').open('w',encoding='utf-8')
    for epoch in range(1,warmup_epochs+epochs+1):
        t=time.perf_counter();model.train()
        indices=list(warmup if epoch<=warmup_epochs else train)
        random.shuffle(indices);loss_total=0.;batches=0
        for offset in range(0,len(indices),batch_size):
            x,lengths,y,yl,_=collate(indices[offset:offset+batch_size],cached,rows,lookup)
            optimizer.zero_grad(set_to_none=True)
            logits,output_lengths=model(x.to(device),lengths.to(device))
            loss=ctc(rearrange(logits.log_softmax(-1),'batch time token -> time batch token'),y.to(device),output_lengths.cpu(),yl)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTC training loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_total+=float(loss.detach());batches+=1
        entry={'epoch':epoch,'phase':'words' if epoch<=warmup_epochs else 'commands',
               'loss':loss_total/max(1,batches),'seconds':time.perf_counter()-t}
        if epoch>warmup_epochs:
            model.eval();groups=defaultdict(lambda:[0,0])
            with torch.inference_mode():
                for offset in range(0,len(dev),batch_size):
                    x,l,_,_,ii=collate(dev[offset:offset+batch_size],cached,rows,lookup)
                    logits,lengths=model(x.to(device),l.to(device));logits=logits.cpu().numpy()
                    for j,i in enumerate(ii):
                        row=rows[i];text=decode_words(logits[j,:int(lengths[j])],vocabulary)['text']
                        if row['kind']=='word':correct=text==row['text']
                        else:
                            parsed=parse_brightness(text,catalog)
                            value=parsed.get('command',{}).get('arguments',{}).get('percent') if parsed.get('status')=='accepted' else None
                            correct=value==row['value']
                        groups[row['kind']][0]+=int(correct);groups[row['kind']][1]+=1
            entry['dev']={k:{'correct':c,'n':n,'accuracy':c/n} for k,(c,n) in groups.items()}
            # Equal weighting prevents large constructed pools hiding natural/negative failures.
            score=sum(v['accuracy'] for v in entry['dev'].values())/len(groups)
            entry['selection_score']=score
            if score>best:
                best=score;best_epoch=epoch
                torch.save({'state_dict':model.state_dict(),'config':model.config,'vocabulary':vocabulary,
                    'epoch':epoch,'manifest_sha256':fingerprint,'seed':seed,'initialization':'random_from_scratch',
                    'development':entry['dev'],'limited':limited},checkpoint)
            if epoch-best_epoch>=settings.get('stt_patience',10):
                history.append(entry);log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps(entry),flush=True);break
        history.append(entry);log.write(json.dumps(entry)+'\n');log.flush();print(json.dumps(entry),flush=True)
    log.close()
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=BrightnessSTT(**payload['config']);model.load_state_dict(payload['state_dict']);model.eval()
    candidate=out/'candidate';candidate.mkdir(exist_ok=True)
    shutil.copy2(checkpoint,candidate/'stt.pt')
    dummy=torch.zeros(1,201,80);length=torch.tensor([201],dtype=torch.long)
    torch.onnx.export(model,(dummy,length),candidate/'stt.fp32.onnx',dynamo=False,opset_version=17,
        input_names=['features','lengths'],output_names=['logits','output_lengths'],
        dynamic_axes={'features':{0:'batch',1:'time'},'lengths':{0:'batch'},'logits':{0:'batch',1:'output_time'},'output_lengths':{0:'batch'}})
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(candidate/'stt.fp32.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    parity=[]
    for sizes in ([100],[201],[411],[301,173,100]):
        x=torch.randn(len(sizes),max(sizes),80);l=torch.tensor(sizes,dtype=torch.long)
        for j,size in enumerate(sizes):x[j,size:]=0
        with torch.no_grad():expected,expected_lengths=model(x,l)
        actual,actual_lengths=session.run(None,{'features':x.numpy(),'lengths':l.numpy()})
        error=float(np.max(np.abs(actual-expected.numpy())))
        assert error<1e-4 and np.array_equal(actual_lengths,expected_lengths.numpy()),(sizes,error)
        # Padding and batch membership must not change a valid utterance.
        for j,size in enumerate(sizes):
            single,_=session.run(None,{'features':x[j:j+1,:size].numpy(),'lengths':np.array([size],dtype=np.int64)})
            assert np.max(np.abs(single[0]-actual[j,:len(single[0])]))<1e-4
        parity.append({'input_lengths':sizes,'max_abs_error':error})
    metadata=json.loads((root/'artifacts/bundle.json').read_text())
    metadata['stt']={'architecture':'word_ctc_cnn_bigru','vocabulary':vocabulary,
        'model_config':payload['config'],'parameters':sum(p.numel() for p in model.parameters()),
        'frontend':{'sample_rate':16000,'mel_bins':80,'fft':400,'hop':160,'normalization':'utterance_cmvn'},
        'training':{k:payload[k] for k in ('epoch','manifest_sha256','seed','initialization','development','limited')}}
    metadata['runtime_components']=list(dict.fromkeys(metadata.get('runtime_components',['vcm'])+['stt']))
    metadata['brightness_parameters']={'scope':'absolute_brightness_only','default_backend':'stt',
        'stt':{'threshold':0.,'calibrated':False}}
    for name in ('vcm.fp32.onnx',):
        shutil.copy2(root/'artifacts'/name,candidate/name)
    for name in ('stt.pt','stt.fp32.onnx'):
        path=candidate/name;metadata['files'][name]={'sha256':sha(path),'bytes':path.stat().st_size}
    write_json(candidate/'bundle.json',metadata)
    result={'candidate':str(candidate),'training_seconds':time.perf_counter()-started,'best_epoch':best_epoch,
        'epochs_run':len(history),'parameters':metadata['stt']['parameters'],'manifest_sha256':fingerprint,
        'train_examples':len(train),'dev_examples':len(dev),'skipped_duration':len(rows)-len(allowed),
        'onnx_parity':parity,'initialization':'random_from_scratch','artifacts_updated':False,'frontend_sha256':frontend_sha,'limited':limited}
    write_json(out/'training_result.json',result)
    print(json.dumps(result),flush=True)
    return result
