"""Bounded semantic-head correction on the same HF audio; frozen acoustic trunk."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from einops import rearrange
from scripts.runtime.pcv_schema import INTENTS,canonical
from scripts.models.pcv_residual import ResidualVCM
from scripts.train.pcv_updated_data import HERE as PARENT,CACHE as PARENT_CACHE
from scripts.train.pcv_updated_training import development_subset,validate_training_rows
from scripts.train.pcv_public_training import pooled_logits
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import normalized,probabilities
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings
from scripts.pcv_parallel import ParallelScorer
from pcvcm.decoder import rank_commands

HERE=ROOT/'experiments/pcv-updated-voice-v2'
CACHE=ROOT/'.cache/pcv-updated-voice-v2'


def semantic_label(row):
    if row['expected']:return INTENTS.index(row['expected']['intent'])+1
    coarse={'VOLUME_UP':'VOLUME','VOLUME_DOWN':'VOLUME'}.get(row['command'],row['command'])
    if row['reference_status']=='unsupported' and coarse in INTENTS and row['reason'] not in ('publisher_out_of_scope','extra_action_or_unrepresented_date'):
        return INTENTS.index(coarse)+1
    return row['intent_index']


def target_label(row):
    if row['target_index']!=-100:return row['target_index']
    if row['reason'] in ('explicit_light_target','unknown_light') and row['expected'] is None:return 1
    return -100


def joint_rank(scores,entries,probability,alpha):
    winner=int(np.argmax(probability))
    if winner==0:
        return {'command':None,'phrase_id':None,'score':None,'margin':0.,'use_case':'OOV'}
    labels=np.asarray([INTENTS.index(e['command']['intent'])+1 if e['command'] else 0 for e in entries])
    values=np.asarray(scores,dtype='float64')+alpha*np.log(np.maximum(probability[labels],1e-10))
    values[(labels!=winner)&(labels!=0)]=-np.inf
    return rank_commands(values,entries)


def extract(model,rows,norm,entries=None):
    model.cuda().eval();vectors=[];ctcs=[];latest=[]
    handle=model.final_norm.register_forward_hook(lambda module,inputs,output:latest.append(output))
    with torch.no_grad():
        for offset in range(0,len(rows),24):
            batch=rows[offset:offset+24];values=[normalized(r,norm) for r in batch];lengths=[v.shape[1] for v in values]
            x=np.zeros((len(batch),40,max(lengths)),dtype='float32')
            for i,v in enumerate(values):x[i,:,:v.shape[1]]=v
            out=model(torch.from_numpy(x).cuda())
            pooled=pooled_logits(latest.pop(),torch.tensor(lengths,device='cuda')).cpu().numpy()
            vectors.extend(pooled)
            if entries is not None:
                ctcs.extend([rearrange(out[0][i,:,:n],'c t -> t c').cpu().numpy() for i,n in enumerate(lengths)])
            if (offset+24)%1200==0:print('HEAD_FEATURES',offset+24,'/',len(rows),flush=True)
    handle.remove()
    vectors=np.asarray(vectors,dtype='float32')
    if entries is None:return vectors,None
    with ParallelScorer(entries,workers=4,return_scores=True) as scorer:
        scores=[r['scores'] for r in scorer.map(ctcs)]
    return vectors,np.asarray(scores)


def weight_rows(rows,labels):
    buckets=Counter((r['synthetic'],int(y)) for r,y in zip(rows,labels) if y!=-100)
    sizes=Counter(origin for origin,label in buckets)
    return np.asarray([0. if y==-100 else 1/(len(sizes)*sizes[r['synthetic']]*buckets[(r['synthetic'],int(y))]) for r,y in zip(rows,labels)],dtype='float32')


def run():
    torch.set_num_threads(2);torch.manual_seed(442);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.3)
    parent=json.loads((PARENT/'training-protocol.json').read_text());verify_bindings(ROOT,parent['bindings'])
    verify_bindings(ROOT,parent['installed_artifact_hashes'])
    selection=json.loads((PARENT/'selection-before-calibration.json').read_text())['arms']['mixed']
    verify_bindings(ROOT,{selection['checkpoint']:selection['checkpoint_sha256']})
    data=json.loads((PARENT/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in data['manifests'].values()})
    train=[r for r in read_rows(PARENT_CACHE/'prepared/train.jsonl') if r['training_eligible'] and r['reference_status']!='auxiliary']
    validate_training_rows(train)
    dev=development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    paths=[Path(__file__),ROOT/'scripts/test/test_pcv_updated_heads.py',PARENT/'data-freeze.json',PARENT/'selection-before-calibration.json']
    protocol={'version':'pcv-updated-voice-v2','parent':'pcv-updated-voice-v1','initial':selection,
        'recording_source':'identical pinned HF recordings; no additional data, external teacher or synthetic generation',
        'change':'retain recognizable intent for unsupported arguments; refit only existing intent and target heads on frozen acoustic features',
        'decoder':'intent head owns family; OOV is never rescued; finite CTC competes within that family and against negative candidates',
        'fit':{'epochs':80,'optimizer':'AdamW','lr':.01,'weight_decay':.0001,'full_batch':len(train),'seed':442,
            'weights':'equal human/synthetic origin, then equal semantic classes within origin','development_every':10},
        'alphas':[0.,.05,.1,.25,.5],'selection':'maximize (human positive exact recall + synthetic positive exact recall + human negative rejection)/3; then positive recall sum; then earliest epoch and smallest alpha',
        'test_inference_accessed':False,'freeze_acoustic_trunk':True,'source_revision':parent['source_revision'],
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths},'installed_artifact_hashes':parent['installed_artifact_hashes']}
    immutable(HERE/'protocol.json',protocol);CACHE.mkdir(parents=True,exist_ok=True)
    labels=np.asarray([semantic_label(r) for r in train]);targets=np.asarray([target_label(r) for r in train])
    immutable(HERE/'label-corrections.json',{'intent_changed':sum(y!=r['intent_index'] for y,r in zip(labels,train)),
        'target_changed':sum(y!=r['target_index'] for y,r in zip(targets,train)),
        'rows':[{'id':r['id'],'intent_before':r['intent_index'],'intent_after':int(y),'target_before':r['target_index'],'target_after':int(t)}
            for r,y,t in zip(train,labels,targets) if y!=r['intent_index'] or t!=r['target_index']]})
    payload=torch.load(ROOT/selection['checkpoint'],map_location='cpu',weights_only=True)
    model=ResidualVCM(**payload['model_config']);model.load_state_dict(payload['weights'])
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text());entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    train_x,_=extract(model,train,norm);dev_x,dev_scores=extract(model,dev,norm,entries)
    model.cpu();x=torch.from_numpy(train_x);dx=torch.from_numpy(dev_x)
    for p in model.parameters():p.requires_grad_(False)
    for module in (model.intent,model.target):
        for p in module.parameters():p.requires_grad_(True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=.0001)
    y=torch.tensor(labels);t=torch.tensor(targets);iw=torch.from_numpy(weight_rows(train,labels));tw=torch.from_numpy(weight_rows(train,targets))
    best=None;best_key=None;history=[]
    for epoch in range(81):
        if epoch:
            model.train();optimizer.zero_grad(set_to_none=True)
            logits=torch.nn.functional.linear(x,model.intent.weight[:,:,0],model.intent.bias)
            target_logits=torch.nn.functional.linear(x,model.target.weight[:,:,0],model.target.bias)
            loss=(torch.nn.functional.cross_entropy(logits,y,reduction='none')*iw).sum()+.5*(torch.nn.functional.cross_entropy(target_logits,t,ignore_index=-100,reduction='none')*tw).sum()
            loss.backward();optimizer.step()
        if epoch%10:continue
        model.eval()
        with torch.no_grad():
            ip=torch.softmax(torch.nn.functional.linear(dx,model.intent.weight[:,:,0],model.intent.bias),1).numpy()
            tp=torch.softmax(torch.nn.functional.linear(dx,model.target.weight[:,:,0],model.target.bias),1).numpy()
        for alpha in protocol['alphas']:
            records=[]
            for row,scores,intent,target in zip(dev,dev_scores,ip,tp):
                pred=joint_rank(scores,entries,intent,alpha);c=pred['command'];ci=INTENTS.index(c['intent'])+1 if c else 0;winner=int(intent.argmax())
                records.append({'id':row['id'],'source':row['source'],'speaker':row['speaker'],'synthetic':row['synthetic'],'text':row['text'],
                    'expected':row['expected'],'candidate':c,'score':pred['score'],'margin':pred['margin'],'intent_index':winner,
                    'intent_probability':float(intent[ci]),'top_intent_probability':float(intent[winner]),'target_named_probability':float(target[1]),
                    'intent_correct':winner==semantic_label(row),'target_expected':target_label(row),'candidate_phrase_id':pred['phrase_id'],
                    'phrase_id':row['phrase_id'],'reference_status':row['reference_status']})
            human=precise_metrics([r for r in records if not r['synthetic']])['overall'];synth=precise_metrics([r for r in records if r['synthetic']])['overall']
            h=human['recall'] or 0;s=synth['recall'] or 0;rejection=1-(human['false_accept_rate'] or 0)
            key=((h+s+rejection)/3,h+s,-epoch,-alpha)
            item={'epoch':epoch,'alpha':alpha,'human':human,'synthetic':synth,'key':list(key)};history.append(item)
            if best_key is None or key>best_key:
                checkpoint=CACHE/f'heads-{epoch:03d}.pt'
                if not checkpoint.exists():torch.save({'weights':model.state_dict(),'model_config':model.config,'epoch':epoch},checkpoint)
                best_key=key;best={**item,'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha(checkpoint),'test_inference_accessed':False}
                save(CACHE/'selected-development.json',records)
        print('HEAD_DEVELOPMENT',epoch,json.dumps(best),flush=True)
    save(CACHE/'history.json',history)
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,parent['installed_artifact_hashes'])
    immutable(HERE/'selection-before-calibration.json',{**best,'protocol_sha256':sha(HERE/'protocol.json'),
        'parent_test_manifest_sha256':data['manifests']['test']['sha256'],'development_rows':len(dev)})
    print('HEADS_COMPLETE',flush=True)


if __name__=='__main__':run()
