"""Longer source-only acoustic continuation with corrected intent supervision."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
import random
import time
import numpy as np
import torch
from einops import rearrange
from scripts.models.pcv_residual import ResidualVCM
from scripts.train.pcv_updated_data import HERE as PARENT,CACHE as PARENT_CACHE
from scripts.train.pcv_updated_training import BalancedRows,development_subset,validate_training_rows
from scripts.train.pcv_public_training import CyclingRows,pooled_logits,load_arrays
from scripts.train.pcv_updated_heads import semantic_label,target_label
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import score
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings

HERE=ROOT/'experiments/pcv-updated-voice-v3'
CACHE=ROOT/'.cache/pcv-updated-voice-v3'


def main():
    seed=543;torch.set_num_threads(2);torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.35);generator=np.random.default_rng(seed)
    parent=json.loads((PARENT/'training-protocol.json').read_text());verify_bindings(ROOT,parent['bindings'])
    verify_bindings(ROOT,parent['installed_artifact_hashes'])
    data=json.loads((PARENT/'data-freeze.json').read_text());verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in data['manifests'].values()})
    initial=json.loads((PARENT/'selection-before-calibration.json').read_text())['arms']['mixed']
    verify_bindings(ROOT,{initial['checkpoint']:initial['checkpoint_sha256']})
    files=[Path(__file__),ROOT/'scripts/train/pcv_updated_heads.py',PARENT/'training-protocol.json',PARENT/'data-freeze.json',PARENT/'selection-before-calibration.json']
    protocol={'version':'pcv-updated-voice-v3','initial':initial,'recordings':'pinned HF source only; same split, features and catalog as v1',
        'epochs':64,'steps_per_epoch':96,'batch_size':32,'microbatch':8,'seed':seed,'optimizer':'AdamW',
        'lr':.0005,'minimum_lr':.00002,'weight_decay':.0001,'intent_loss_weight':.3,'target_loss_weight':.15,
        'batch':{'human_positive':8,'human_negative':4,'numeral':4,'supplied_synthetic':16},
        'supervision':'recognizable intents with unsupported arguments retain their semantic class; explicit unknown light names train the named-target head',
        'selection':'every four epochs; require human positive recall >= 0.7080952380952381 (v1 human minus 0.03), then maximize mean human/synthetic positive recall, then minimize human negative errors, then earliest epoch',
        'human_noninferiority_floor':.7080952380952381,'test_inference_accessed':False,'external_audio_used':False,
        'bindings':{str(p.relative_to(ROOT)):sha(p) for p in files},'installed_artifact_hashes':parent['installed_artifact_hashes']}
    immutable(HERE/'protocol.json',protocol)
    if CACHE.exists():raise FileExistsError('Preserve previous continuation output')
    CACHE.mkdir(parents=True)
    rows=[r for r in read_rows(PARENT_CACHE/'prepared/train.jsonl') if r['training_eligible']]
    validate_training_rows(rows)
    changes=[]
    for r in rows:
        semantic=semantic_label(r);target=target_label(r)
        if semantic!=r['intent_index'] or target!=r['target_index']:
            changes.append({'id':r['id'],'intent_before':r['intent_index'],'intent_after':semantic,'target_before':r['target_index'],'target_after':target})
        r['intent_index']=semantic;r['target_index']=target
    immutable(HERE/'supervision-corrections.json',{'rows':changes,'recordings_added':0,'full_command_references_changed':0})
    dev=development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text());entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    arrays=load_arrays(rows+dev,norm)
    state=torch.load(ROOT/initial['checkpoint'],map_location='cpu',weights_only=True)
    model=ResidualVCM(**state['model_config']).cuda();model.load_state_dict(state['weights'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['epochs'],eta_min=protocol['minimum_lr'])
    human=[r for r in rows if not r['synthetic'] and r['reference_status']!='auxiliary']
    pools={'human_positive':BalancedRows([r for r in human if r['expected']],generator),
        'human_negative':CyclingRows([r for r in human if not r['expected']],generator),
        'numeral':CyclingRows([r for r in rows if r['reference_status']=='auxiliary'],generator),
        'supplied_synthetic':BalancedRows([r for r in rows if r['synthetic']],generator)}
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);history=[];best=None;best_key=None;seen=set()
    for epoch in range(1,protocol['epochs']+1):
        began=time.perf_counter();model.train();losses=np.zeros(3)
        for step in range(protocol['steps_per_epoch']):
            batch=[]
            for key,n in protocol['batch'].items():batch.extend(pools[key].take(n))
            generator.shuffle(batch);seen.update(r['id'] for r in batch);optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),protocol['microbatch']):
                micro=batch[offset:offset+protocol['microbatch']];lengths=[arrays[r['id']].shape[1] for r in micro]
                x=np.zeros((len(micro),40,max(lengths)),dtype='float32')
                for i,r in enumerate(micro):x[i,:,:lengths[i]]=arrays[r['id']]
                ctc,intent,target,_=model(torch.from_numpy(x).cuda());lengths_gpu=torch.tensor(lengths,device='cuda')
                acoustic=criterion(rearrange(ctc.log_softmax(1),'b c t -> t b c'),
                    torch.tensor([p for r in micro for p in r['phones']],device='cuda',dtype=torch.long),
                    torch.tensor(lengths),torch.tensor([len(r['phones']) for r in micro]))
                il=torch.tensor([r['intent_index'] for r in micro],device='cuda');tl=torch.tensor([r['target_index'] for r in micro],device='cuda')
                iloss=torch.nn.functional.cross_entropy(pooled_logits(intent,lengths_gpu),il,ignore_index=-100) if torch.any(il!=-100) else intent.sum()*0
                tloss=torch.nn.functional.cross_entropy(pooled_logits(target,lengths_gpu),tl,ignore_index=-100) if torch.any(tl!=-100) else target.sum()*0
                fraction=len(micro)/len(batch);loss=(acoustic+.3*iloss+.15*tloss)*fraction
                if not torch.isfinite(loss):raise ValueError('Nonfinite continuation loss')
                loss.backward();losses+=np.asarray([acoustic.item(),iloss.item(),tloss.item()])*fraction
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        scheduler.step();result={'epoch':epoch,'seconds':time.perf_counter()-began,'losses':(losses/96).tolist(),'unique_recordings_seen':len(seen)}
        checkpoint={'weights':model.state_dict(),'model_config':model.config,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'protocol_sha256':sha(HERE/'protocol.json')}
        torch.save(checkpoint,CACHE/'last.pt')
        if epoch%4==0:
            records=score(model,dev,norm,entries,'cuda',workers=4,arrays=arrays)
            h=precise_metrics([r for r in records if not r['synthetic']])['overall'];s=precise_metrics([r for r in records if r['synthetic']])['overall']
            key=(int(h['recall']>=protocol['human_noninferiority_floor']),(h['recall']+s['recall'])/2,-h['negative_errors'],-epoch)
            result['development']={'human':h,'synthetic':s};result['selected']=best_key is None or key>best_key
            save(CACHE/f'development-{epoch:03d}.json',records)
            if result['selected']:
                path=CACHE/f'epoch-{epoch:03d}.pt';torch.save(checkpoint,path);best_key=key
                best={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'epoch':epoch,
                    'development':result['development'],'development_key':list(key),'test_inference_accessed':False}
                save(CACHE/'selection.json',best)
        history.append(result);save(CACHE/'history.json',history);print(json.dumps(result),flush=True)
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,parent['installed_artifact_hashes'])
    immutable(HERE/'selection-before-calibration.json',{**best,'parent_data_freeze_sha256':sha(PARENT/'data-freeze.json'),
        'protocol_sha256':sha(HERE/'protocol.json'),'unique_recordings_seen':len(seen)})
    print('CONTINUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
