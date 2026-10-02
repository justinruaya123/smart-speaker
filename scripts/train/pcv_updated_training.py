"""Source-only, matched human then supplied-synthetic VCM experiment."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from einops import rearrange
from scripts.models.pcv_residual import ResidualVCM
from scripts.train.pcv_updated_data import HERE,CACHE,REVISION
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_public_training import CyclingRows,pooled_logits,load_arrays
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import score
from scripts.test.pcv_public_qualification import verify_bindings,precise_metrics


def validate_training_rows(rows,human_only=False):
    for row in rows:
        path=(ROOT/row['parquet']).resolve()
        if not path.is_relative_to((CACHE/'source/data').resolve()):
            raise ValueError('External recording is not authorized for this source-only experiment')
        if row['published_split'] not in ('train','numerals') or row['split']!='train':
            raise ValueError('Held-out recording cannot enter fitting')
        if human_only and row['synthetic']:
            raise ValueError('Synthetic recording cannot enter human foundation/control')


class BalancedRows:
    """Uniform round-robin intent exposure, shuffled recordings within each intent."""
    def __init__(self,rows,generator):
        groups=defaultdict(list)
        for row in rows:groups[row['intent']].append(row)
        self.pools=[CyclingRows(groups[k],generator) for k in sorted(groups)]
        self.cursor=0

    def take(self,n):
        if not self.pools and n:raise ValueError('Required source pool is empty')
        result=[]
        for _ in range(n):
            result.extend(self.pools[self.cursor%len(self.pools)].take(1));self.cursor+=1
        return result


def development_subset(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['reference_status'] in ('supported','unsupported') and row['training_eligible']:
            groups[(row['synthetic'],row['source'],row['intent'])].append(row)
    result=[]
    for key, values in sorted(groups.items()):
        # Deterministic metadata-only selection, never evaluation-dependent.
        values=sorted(values,key=lambda r:sha_text(r['id']))
        limit=12 if not key[0] else 6
        if key[2]=='OOV':limit=24
        result.extend(values[:limit])
    return result


def sha_text(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def freeze_training():
    data=json.loads((HERE/'data-freeze.json').read_text())
    verify_bindings(ROOT,data['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in data['manifests'].values()})
    if (HERE/'training-protocol.json').exists():
        frozen=json.loads((HERE/'training-protocol.json').read_text())
        verify_bindings(ROOT,frozen['bindings']);verify_bindings(ROOT,frozen['installed_artifact_hashes'])
        return frozen
    source=['scripts/train/pcv_updated_training.py','scripts/train/pcv_updated_prepare.py',
        'scripts/models/pcv_residual.py','scripts/train/pcv_public_training.py','scripts/test/pcv_public_eval.py',
        'scripts/test/pcv_public_qualification.py','scripts/runtime/pcv_schema.py','scripts/pcv_parallel.py',
        'experiments/phrase-contract-v1/pcvcm/decoder.py','vcm/frontend.py','data/raw/cmudict/cmudict.dict',
        'experiments/pcv-synthetic-v2/lexicon.json','experiments/pcv-updated-voice-v1/data-freeze.json']
    protocol={'experiment':'pcv-updated-voice-v1','source_revision':REVISION,'recording_source':'HF pinned revision only',
        'sequence':['human foundation','matched human-only continuation','matched supplied-HF-synthetic continuation'],
        'fallback_policy':'none in this experiment; separate version only when no suitable source recording matches a documented required input',
        'distillation':'not used; no outside-data teacher or inherited checkpoint',
        'initialization':'random weights; both continuation arms start from the same selected human foundation',
        'seed':241,'model':{'width':128,'dilations':[1,2,4,8,16,1,2,4,8,16]},
        'training':{'foundation_epochs':32,'continuation_epochs':16,'steps_per_epoch':96,'batch_size':32,
            'microbatch_size':8,'learning_rate':.001,'continuation_learning_rate':.0003,
            'weight_decay':.0001,'intent_loss_weight':.3,'target_loss_weight':.15,'gradient_clip':5.,
            'development_every_epochs':4,'early_stopping':False,
            'human_pool_counts':{'human_positive':24,'human_negative':4,'numeral':4,'synthetic':0},
            'mixed_pool_counts':{'human_positive':16,'human_negative':4,'numeral':4,'synthetic':8},
            'positive_sampling':'equal intent; shuffled recordings within intent',
            'augmentation':'none; preserve complete source waveform; source synthetic includes publisher clean/noisy variants'},
        'selection':'maximize human dev complete recall, then minimize human negative errors, then maximize synthetic dev recall, then earliest epoch; foundation ignores synthetic development',
        'calibration':{'precision':.99,'recall':.90,'negative_false_accept_rate':.01,'requires_all_18_intents':True,
            'human_and_synthetic_positive_recall_separate':True,'missing_negative_stratum':'unmeasured, never zero by assumption'},
        'test':'official human and synthetic test unchanged; no test-driven training/threshold/curriculum choices',
        'holdout':'preserved for locked independent replay; never used for weights or thresholds; not Pi evidence',
        'bindings':{p:sha(ROOT/p) for p in source},
        'installed_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()},
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
        'test_inference_accessed':False,'installation_authorized':False}
    immutable(HERE/'training-protocol.json',protocol)
    return protocol


def stage(name,initial,rows,development,arrays,norm,entries,protocol,epochs):
    cfg=protocol['training'];directory=CACHE/'training'/name
    if directory.exists():raise FileExistsError('Preserve existing run: '+str(directory))
    directory.mkdir(parents=True)
    seed=protocol['seed']+(0 if name=='foundation' else 100)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    generator=np.random.default_rng(seed)
    model=ResidualVCM(**protocol['model']).cuda()
    if initial:model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['weights'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'] if name=='foundation' else cfg['continuation_learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.00002)
    human=[r for r in rows if not r['synthetic'] and r['reference_status']!='auxiliary']
    groups={'human_positive':[r for r in human if r['expected']],
        'human_negative':[r for r in human if not r['expected']],
        'numeral':[r for r in rows if r['reference_status']=='auxiliary'],
        'synthetic':[r for r in rows if r['synthetic']]}
    for key, values in groups.items():validate_training_rows(values,human_only=key!='synthetic')
    sources={k:(BalancedRows(v,generator) if k in ('human_positive','synthetic') else CyclingRows(v,generator)) for k,v in groups.items()}
    plan=cfg['mixed_pool_counts'] if name=='mixed' else cfg['human_pool_counts']
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False)
    history=[];best=None;best_key=None
    for epoch in range(1,epochs+1):
        began=time.perf_counter();model.train();losses=np.zeros(3)
        for step in range(cfg['steps_per_epoch']):
            batch=[]
            for key,n in plan.items():
                if n:batch.extend(sources[key].take(n))
            generator.shuffle(batch);opt.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),cfg['microbatch_size']):
                micro=batch[offset:offset+cfg['microbatch_size']]
                lengths=[arrays[r['id']].shape[1] for r in micro]
                x=np.zeros((len(micro),40,max(lengths)),dtype='float32')
                for i,r in enumerate(micro):x[i,:,:lengths[i]]=arrays[r['id']]
                ctc,intents,targets,_=model(torch.from_numpy(x).cuda())
                length_tensor=torch.tensor(lengths,device='cuda')
                ctc_term=criterion(rearrange(ctc.log_softmax(1),'b c t -> t b c'),
                    torch.tensor([p for r in micro for p in r['phones']],dtype=torch.long,device='cuda'),
                    torch.tensor(lengths),torch.tensor([len(r['phones']) for r in micro]))
                intent_labels=torch.tensor([r['intent_index'] for r in micro],device='cuda')
                target_labels=torch.tensor([r['target_index'] for r in micro],device='cuda')
                intent_term=torch.nn.functional.cross_entropy(pooled_logits(intents,length_tensor),intent_labels,ignore_index=-100) if torch.any(intent_labels!=-100) else intents.sum()*0
                target_term=torch.nn.functional.cross_entropy(pooled_logits(targets,length_tensor),target_labels,ignore_index=-100) if torch.any(target_labels!=-100) else targets.sum()*0
                fraction=len(micro)/len(batch)
                loss=(ctc_term+cfg['intent_loss_weight']*intent_term+cfg['target_loss_weight']*target_term)*fraction
                if not torch.isfinite(loss):raise ValueError('Nonfinite fitting loss')
                loss.backward();losses+=np.asarray([ctc_term.item(),intent_term.item(),target_term.item()])*fraction
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
        scheduler.step()
        metric={'stage':name,'epoch':epoch,'steps':cfg['steps_per_epoch'],'seconds':time.perf_counter()-began,
            'losses':(losses/cfg['steps_per_epoch']).tolist(),'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
        checkpoint={'weights':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'model_config':model.config,'epoch':epoch,'stage':name,'seed':seed,'protocol_sha256':sha(HERE/'training-protocol.json')}
        torch.save(checkpoint,directory/'last.pt')
        if epoch%cfg['development_every_epochs']==0:
            subset=[r for r in development if not r['synthetic']] if name=='foundation' else development
            records=score(model,subset,norm,entries,'cuda',workers=4,arrays=arrays)
            hm=precise_metrics(r for r in records if not r['synthetic'])['overall']
            sm=precise_metrics(r for r in records if r['synthetic'])['overall']
            key=(hm['recall'] or 0,-hm['negative_errors'],sm['recall'] or 0,-epoch)
            metric['development']={'human':hm,'synthetic':sm};metric['selected']=best_key is None or key>best_key
            save(directory/f'dev-{epoch:03d}.json',records)
            if metric['selected']:
                path=directory/f'epoch-{epoch:03d}.pt';torch.save(checkpoint,path)
                best_key=key;best={'stage':name,'epoch':epoch,'checkpoint':str(path.relative_to(ROOT)),
                    'checkpoint_sha256':sha(path),'development_key':list(key),'development':metric['development'],
                    'initial_checkpoint':str(initial) if initial else None,'test_inference_accessed':False}
                save(directory/'selection.json',best)
        history.append(metric);save(directory/'history.json',history)
        print(json.dumps(metric),flush=True)
    return best


def train():
    protocol=freeze_training();cfg=protocol['training']
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.35)
    rows=[r for r in read_rows(CACHE/'prepared/train.jsonl') if r['training_eligible']]
    validate_training_rows(rows)
    development=development_subset(read_rows(CACHE/'prepared/dev.jsonl'))
    immutable(HERE/'development-selection.json',{'ids':[r['id'] for r in development],'rule':'source/intent/positive stratified deterministic hash selection'})
    norm=json.loads((CACHE/'prepared/normalization.json').read_text());entries=json.loads((HERE/'catalog.json').read_text())['entries']
    arrays=load_arrays(rows+development,norm)
    human=[r for r in rows if not r['synthetic']]
    base=stage('foundation',None,human,development,arrays,norm,entries,protocol,cfg['foundation_epochs'])
    arms={}
    for name in ('human','mixed'):
        arms[name]=stage(name,ROOT/base['checkpoint'],human if name=='human' else rows,development,arrays,norm,entries,protocol,cfg['continuation_epochs'])
    winner=max(arms,key=lambda k:tuple(arms[k]['development_key']))
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    immutable(HERE/'selection-before-calibration.json',{'foundation':base,'arms':arms,'winner':winner,
        'training_protocol_sha256':sha(HERE/'training-protocol.json'),'test_inference_accessed':False})
    print('TRAINING_COMPLETE',winner,flush=True)


if __name__=='__main__':train()
