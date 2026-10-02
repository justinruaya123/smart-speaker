"""HF-only human-first endpoint VCM, then offline finite-teacher distillation."""
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from einops import rearrange,reduce
from scripts.models.pcv_contextual import ContextualVCM
from scripts.train.pcv_teacher_data import PARENT,PARENT_CACHE,HERE as TEACHER,CACHE as TEACHER_CACHE
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_public_training import CyclingRows,pooled_logits,load_arrays
from scripts.train.pcv_updated_training import BalancedRows,validate_training_rows,development_subset
from scripts.train.pcv_updated_heads import semantic_label,target_label
from scripts.train.pcv_teacher_train import ctc_eligible
from scripts.train.pcv_distillation import finite_scores,distribution_loss
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import probabilities
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings
from scripts.runtime.pcv_schema import INTENTS
from scripts.pcv_parallel import ParallelScorer

HERE=ROOT/'experiments/pcv-updated-context-v1'
CACHE=ROOT/'.cache/pcv-updated-context-v1'


def score_context(model,rows,arrays,entries):
    is_onnx=hasattr(model,'get_inputs');records=[];heads=[];began=time.perf_counter()
    if not is_onnx:model.eval()
    def outputs():
        size=1 if is_onnx else 16
        for offset in range(0,len(rows),size):
            batch=rows[offset:offset+size];lengths=[arrays[r['id']].shape[1] for r in batch]
            x=np.zeros((len(batch),40,max(lengths)),dtype='float32')
            for i,r in enumerate(batch):x[i,:,:lengths[i]]=arrays[r['id']]
            if is_onnx:
                ctc,intent,target=[rearrange(a,'b t c -> b c t') for a in model.run(None,{'features':x})]
            else:
                with torch.no_grad():
                    ctc,intent,target,_=model(torch.from_numpy(x).cuda(),torch.tensor(lengths,device='cuda'))
                    ctc,intent,target=ctc.cpu().numpy(),intent.cpu().numpy(),target.cpu().numpy()
            for i,length in enumerate(lengths):
                n=(length+1)//2
                heads.append((probabilities(reduce(intent[i,:,:n],'c t -> c','mean')),
                              probabilities(reduce(target[i,:,:n],'c t -> c','mean'))))
                yield rearrange(ctc[i,:,:n],'c t -> t c')
    with ParallelScorer(entries,workers=4) as scorer:
        for i,(row,item) in enumerate(zip(rows,scorer.map(outputs()),strict=True)):
            pred=item['prediction'];ip,tp=heads[i];winner=int(ip.argmax());candidate=pred['command']
            ci=INTENTS.index(candidate['intent'])+1 if candidate else 0
            records.append({'id':row['id'],'source':row['source'],'speaker':row['speaker'],
                'synthetic':row['synthetic'],'phrase_id':row.get('phrase_id'),'text':row['text'],
                'expected':row['expected'],'candidate':candidate,'score':pred['score'],'margin':pred['margin'],
                'intent_index':winner,'intent_probability':float(ip[ci]),'top_intent_probability':float(ip[winner]),
                'target_named_probability':float(tp[1]),'candidate_phrase_id':pred['phrase_id']})
        print('CONTEXT_EVALUATION',len(records),'seconds',round(time.perf_counter()-began,1),flush=True)
    return records


def freeze(rows,excluded):
    source=json.loads((PARENT/'data-freeze.json').read_text());verify_bindings(ROOT,source['bindings'])
    verify_bindings(ROOT,{v['path']:v['sha256'] for v in source['manifests'].values()})
    teacher=json.loads((TEACHER/'teacher-selection.json').read_text())
    if not teacher['distillation_gate_pass']:raise ValueError('Unqualified teacher-use gate')
    path=HERE/'protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text());verify_bindings(ROOT,protocol['bindings']);return protocol
    files=['scripts/models/pcv_contextual.py','scripts/train/pcv_context_training.py',
        'scripts/test/test_pcv_contextual.py','scripts/train/pcv_distillation.py',
        'scripts/train/pcv_updated_training.py','scripts/train/pcv_updated_heads.py',
        'scripts/train/pcv_public_training.py','scripts/test/pcv_public_eval.py',
        'scripts/test/pcv_public_qualification.py','scripts/pcv_parallel.py',
        'experiments/phrase-contract-v1/pcvcm/decoder.py','experiments/pcv-updated-voice-v1/catalog.json',
        'experiments/pcv-updated-voice-v1/data-freeze.json','experiments/pcv-updated-context-v1/model-preflight-fixed.json',
        'experiments/pcv-updated-distill-v1/teacher-selection.json','experiments/pcv-updated-distill-v1/targets-freeze.json']
    protocol={'experiment':'pcv-updated-context-v1','owner':'current PCV finalization task',
        'requirements':['ME-02','ME-03','ME-04','ME-06','ME-07','ME-08'],
        'hypothesis':'Endpoint bidirectional context and a 20ms output grid improve the tiny student ability to use offline teacher knowledge.',
        'causal_limitation':'Hypothesis only; not an isolated causal ablation because architecture and fitting schedule also differ.',
        'architecture':{'width':80,'layers':6,'heads':4,'feedforward':160,'dropout':.1},
        'inference':'Whole utterance after wake/VAD endpoint. No KV cache or acoustic-streaming claim; finite catalog only, no transcript decoder.',
        'seed':902,'initialization':'random weights; original human-only foundation before supplied-synthetic plus teacher continuation',
        'training':{'human_epochs':24,'mixed_epochs':40,'steps_per_epoch':96,'batch_size':32,'microbatch_size':8,
            'human_learning_rate':.0008,'mixed_learning_rate':.0003,'minimum_learning_rate':.00002,'weight_decay':.001,
            'gradient_clip':5.,'intent_weight':.3,'target_weight':.15,'sequence_weight':.3,'frame_weight':.1,
            'semantic_weight':.1,'sequence_temperature':.03,'head_temperature':2.,'development_every_epochs':4,
            'human_batch':{'human_positive':24,'human_negative':4,'numeral':4,'synthetic':0},
            'mixed_batch':{'human_positive':12,'human_negative':4,'numeral':4,'synthetic':12}},
        'waveform_augmentation':'none; model dropout is internal regularization, not generated audio',
        'source':'Pinned supplied HF recordings only; offline pretrained WavLM is the user-authorized weight exception.',
        'ctc_frame_exclusions':excluded,'fit_ids':[r['id'] for r in rows],
        'selection':'Human stage: human raw exact recall, fewer negative errors, earlier epoch. Mixed: mean human/synthetic raw exact recall, fewer human negative errors, earlier epoch. Compare parent v3 before qualification.',
        'test':'No new test access during fitting. Prior v1/v3 aggregate test results already exposed; no per-test-error tuning.',
        'baseline':json.loads((ROOT/'experiments/pcv-updated-voice-v3/selection-before-calibration.json').read_text()),
        'bindings':{p:sha(ROOT/p) for p in files},
        'installed_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()},
        'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda},
        'installation_authorized':False,'external_recordings_used':False}
    immutable(path,protocol);return protocol


def stage(name,initial,rows,dev,arrays,targets,frames,norm,entries,protocol):
    cfg=protocol['training'];directory=CACHE/name
    if directory.exists():raise FileExistsError('Preserve existing contextual run')
    directory.mkdir(parents=True);seed=protocol['seed']+(100 if name=='mixed' else 0)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);generator=np.random.default_rng(seed)
    model=ContextualVCM(**protocol['architecture']).cuda()
    if initial:model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['weights'])
    epochs=cfg[name+'_epochs'];opt=torch.optim.AdamW(model.parameters(),lr=cfg[name+'_learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=cfg['minimum_learning_rate'])
    groups={'human_positive':[r for r in rows if not r['synthetic'] and r['expected']],
        'human_negative':[r for r in rows if not r['synthetic'] and r['reference_status']=='unsupported'],
        'numeral':[r for r in rows if r['reference_status']=='auxiliary'],'synthetic':[r for r in rows if r['synthetic']]}
    plan=cfg[name+'_batch'];pools={k:(BalancedRows(v,generator) if k in ('human_positive','synthetic') else CyclingRows(v,generator)) for k,v in groups.items()}
    validate_training_rows(rows,human_only=name=='human')
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);history=[];best=None;best_key=None
    for epoch in range(1,epochs+1):
        model.train();began=time.perf_counter();losses=np.zeros(6)
        for step in range(cfg['steps_per_epoch']):
            batch=[]
            for key,n in plan.items():
                if n:batch.extend(pools[key].take(n))
            generator.shuffle(batch);opt.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),cfg['microbatch_size']):
                micro=batch[offset:offset+cfg['microbatch_size']];input_lengths=[arrays[r['id']].shape[1] for r in micro]
                lengths=[(n+1)//2 for n in input_lengths];lt=torch.tensor(lengths,device='cuda')
                x=np.zeros((len(micro),40,max(input_lengths)),dtype='float32')
                for i,r in enumerate(micro):x[i,:,:input_lengths[i]]=arrays[r['id']]
                ctc,intent,target,_=model(torch.from_numpy(x).cuda(),torch.tensor(input_lengths,device='cuda'))
                ct=criterion(rearrange(ctc.log_softmax(1),'b c t -> t b c'),
                    torch.tensor([p for r in micro for p in r['phones']],device='cuda',dtype=torch.long),
                    torch.tensor(lengths),torch.tensor([len(r['phones']) for r in micro]))
                il=torch.tensor([semantic_label(r) for r in micro],device='cuda');tl=torch.tensor([target_label(r) for r in micro],device='cuda')
                pi=pooled_logits(intent,lt);pt=pooled_logits(target,lt)
                it=torch.nn.functional.cross_entropy(pi,il,ignore_index=-100) if torch.any(il!=-100) else intent.sum()*0
                tt=torch.nn.functional.cross_entropy(pt,tl,ignore_index=-100) if torch.any(tl!=-100) else target.sum()*0
                sq=ctc.sum()*0;fr=ctc.sum()*0;sem=ctc.sum()*0
                if name=='mixed':
                    positions=[i for i,r in enumerate(micro) if r['id'] in targets]
                    if positions:
                        items=[targets[micro[i]['id']] for i in positions]
                        paths=[[entries[j]['phones'] for j in t['indices']] for t in items]
                        values=finite_scores(ctc[positions],[lengths[i] for i in positions],paths)
                        sq=distribution_loss(values,torch.tensor([t['scores'] for t in items],device='cuda'),
                            torch.tensor([t['valid'] for t in items],device='cuda'),cfg['sequence_temperature'])
                        temperature=cfg['head_temperature'];parts=[]
                        for i in positions:
                            teacher=torch.from_numpy(frames[micro[i]['id']]).cuda()
                            if teacher.shape[1]!=lengths[i]:raise ValueError('Teacher and contextual student grids differ')
                            parts.append(torch.nn.functional.kl_div((ctc[i,:,:lengths[i]]/temperature).log_softmax(0),
                                (teacher/temperature).softmax(0),reduction='sum')/lengths[i]*temperature**2)
                        fr=reduce(parts,'b ->','mean')
                        for student,key in ((pi,'intent_logits'),(pt,'target_logits')):
                            teacher=torch.tensor([t[key] for t in items],device='cuda')
                            sem=sem+torch.nn.functional.kl_div((student[positions]/temperature).log_softmax(1),
                                (teacher/temperature).softmax(1),reduction='batchmean')*temperature**2
                fraction=len(micro)/len(batch)
                loss=(ct+cfg['intent_weight']*it+cfg['target_weight']*tt+cfg['sequence_weight']*sq+
                      cfg['frame_weight']*fr+cfg['semantic_weight']*sem)*fraction
                if not torch.isfinite(loss):raise ValueError('Nonfinite contextual fitting loss')
                loss.backward();losses+=fraction*np.asarray([v.item() for v in (ct,it,tt,sq,fr,sem)])
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
        scheduler.step();metric={'stage':name,'epoch':epoch,'seconds':time.perf_counter()-began,'losses':(losses/cfg['steps_per_epoch']).tolist()}
        checkpoint={'weights':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'model_config':model.config,'epoch':epoch,'stage':name,'seed':seed,'protocol_sha256':sha(HERE/'protocol.json')}
        torch.save(checkpoint,directory/'last.pt')
        if epoch%cfg['development_every_epochs']==0:
            subset=dev if name=='mixed' else [r for r in dev if not r['synthetic']]
            records=score_context(model,subset,arrays,entries)
            hm=precise_metrics(r for r in records if not r['synthetic'])['overall'];sm=precise_metrics(r for r in records if r['synthetic'])['overall']
            main=hm['recall'] if name=='human' else (hm['recall']+sm['recall'])/2
            key=(main,-hm['negative_errors'],-epoch);metric['development']={'human':hm,'synthetic':sm}
            metric['selected']=best_key is None or key>best_key;save(directory/f'dev-{epoch:03d}.json',records)
            if metric['selected']:
                path=directory/f'epoch-{epoch:03d}.pt';torch.save(checkpoint,path);best_key=key
                best={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'epoch':epoch,
                    'stage':name,'development':metric['development'],'key':list(key),'test_inference_accessed':False}
                save(directory/'selection.json',best)
        history.append(metric);save(directory/'history.json',history);print('CONTEXT_TRAIN',json.dumps(metric),flush=True)
    return best


def run():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    rows=[r for r in read_rows(PARENT_CACHE/'prepared/train.jsonl') if r['training_eligible']]
    validate_training_rows(rows);excluded=[{'id':r['id'],'reason':'insufficient_20ms_CTC_frames'} for r in rows if not ctc_eligible(r,(r['frames']+1)//2)]
    rows=[r for r in rows if ctc_eligible(r,(r['frames']+1)//2)];protocol=freeze(rows,excluded)
    target_freeze=json.loads((TEACHER/'targets-freeze.json').read_text());verify_bindings(ROOT,{target_freeze['path']:target_freeze['sha256']})
    targets={r['id']:r for r in read_rows(ROOT/target_freeze['path'])};frames={};by_id={r['id']:r for r in rows}
    for identity,item in targets.items():
        if identity not in by_id or by_id[identity]['pcm_sha256']!=item['pcm_sha256']:raise ValueError('Teacher target source differs')
        if sha(ROOT/item['frame_path'])!=item['frame_sha256']:raise ValueError('Teacher target identity changed')
        frames[identity]=np.load(ROOT/item['frame_path'],allow_pickle=False).astype('float32')
    dev=development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text());arrays=load_arrays(rows+dev,norm)
    entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    human=stage('human',None,[r for r in rows if not r['synthetic']],dev,arrays,targets,frames,norm,entries,protocol)
    mixed=stage('mixed',ROOT/human['checkpoint'],rows,dev,arrays,targets,frames,norm,entries,protocol)
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    immutable(HERE/'selection-before-calibration.json',{'human':human,'selected':mixed,
        'protocol_sha256':sha(HERE/'protocol.json'),'test_inference_accessed':False})
    print('CONTEXT_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':run()
