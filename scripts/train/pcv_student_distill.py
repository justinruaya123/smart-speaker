"""Matched tiny-student continuation with finite-command teacher guidance."""
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from einops import rearrange, reduce
from scripts.models.pcv_residual import ResidualVCM
from scripts.models.pcv_teacher import FiniteTeacherHead
from scripts.train.pcv_teacher_data import HERE,CACHE,PARENT,PARENT_CACHE,rows_for_probe,check_rows
from scripts.train.pcv_teacher_train import feature_arrays
from scripts.train.pcv_distillation import finite_scores,top_distinct,distribution_loss,aligned_teacher_logits
from scripts.train.pcv_updated_training import BalancedRows,validate_training_rows,development_subset
from scripts.train.pcv_updated_heads import semantic_label,target_label
from scripts.train.pcv_public_training import CyclingRows,pooled_logits,load_arrays
from scripts.train.pcv_public_data import sha,save,read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import score
from scripts.test.pcv_public_qualification import precise_metrics,verify_bindings
from scripts.pcv_parallel import ParallelScorer


def configure():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def freeze():
    teacher=json.loads((HERE/'teacher-selection.json').read_text())
    if not teacher['distillation_gate_pass']:raise ValueError('Teacher did not pass development-use gate')
    selected=teacher['mixed'];verify_bindings(ROOT,{selected['checkpoint']:selected['checkpoint_sha256']})
    source=json.loads((HERE/'teacher-training-protocol.json').read_text());verify_bindings(ROOT,source['bindings'])
    baseline_path=ROOT/'experiments/pcv-updated-voice-v3/selection-before-calibration.json'
    baseline=json.loads(baseline_path.read_text());verify_bindings(ROOT,{baseline['checkpoint']:baseline['checkpoint_sha256']})
    path=HERE/'student-protocol.json'
    if path.exists():
        protocol=json.loads(path.read_text());verify_bindings(ROOT,protocol['bindings']);return protocol
    names=['scripts/train/pcv_student_distill.py','scripts/train/pcv_distillation.py',
        'scripts/test/test_pcv_distillation.py','scripts/test/test_pcv_distill_frames.py',
        'scripts/models/pcv_residual.py','scripts/train/pcv_updated_heads.py',
        'scripts/train/pcv_public_training.py','scripts/train/pcv_updated_training.py',
        'scripts/test/pcv_public_eval.py','scripts/test/pcv_public_qualification.py',
        'scripts/pcv_parallel.py','experiments/phrase-contract-v1/pcvcm/decoder.py',
        'experiments/pcv-updated-distill-v1/teacher-selection.json',
        'experiments/pcv-updated-voice-v1/data-freeze.json','experiments/pcv-updated-voice-v1/catalog.json']
    protocol={'experiment':'pcv-updated-distill-v1','phase':'matched tiny-student distillation',
        'requirements':['ME-02','ME-03','ME-04','ME-06','ME-08'],'seed':841,
        'hypothesis':'Finite command competition and audio-head guidance improve the tiny VCM over matched hard-label continuation.',
        'teacher':selected,'initial_student':baseline,
        'initialization':'Both arms start from the same HF-only v3 checkpoint; its lineage and the teacher both trained on human recordings before supplied synthetic speech.',
        'model':{'width':128,'dilations':[1,2,4,8,16,1,2,4,8,16]},
        'training':{'epochs':32,'steps_per_epoch':96,'batch_size':32,'microbatch_size':8,
            'learning_rate':.0003,'minimum_learning_rate':.00002,'weight_decay':.0001,
            'gradient_clip':5.,'intent_weight':.3,'target_weight':.15,'sequence_weight':.3,
            'frame_weight':.2,'semantic_weight':.1,'candidate_count':8,
            'sequence_temperature':.03,'head_temperature':2.,'development_every_epochs':4,
            'batch':{'human_positive':8,'human_negative':4,'numeral':4,'synthetic':16}},
        'teacher_targets':'Training recordings only. Exact finite-catalog scores, one strongest phrase per distinct command; no unconstrained transcript or teacher label replacement.',
        'frame_alignment':'Same 400-sample receptive-field centres; teacher stride320 versus student160. Zero-order interpolation, valid frames only. Hard CTC remains primary supervision.',
        'selection':'Maximize mean human and synthetic raw exact development recall, then fewer human negative errors, then earlier epoch. Arm selection uses the same key; keep parent as another development-only candidate.',
        'test_access':'No test inference in this training phase; parent aggregate test results were already exposed.',
        'augmentation':'none','external_recordings':False,'installation_authorized':False,
        'bindings':{n:sha(ROOT/n) for n in names},'installed_artifact_hashes':source['installed_artifact_hashes']}
    immutable(path,protocol);return protocol


def make_targets():
    configure();protocol=freeze();cfg=protocol['training']
    rows,_=rows_for_probe();check_rows(rows,fitting=True);arrays=feature_arrays(rows)
    entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    selected=protocol['teacher'];checkpoint=torch.load(ROOT/selected['checkpoint'],map_location='cpu',weights_only=True)
    model=FiniteTeacherHead(**checkpoint['model_config']).cuda().eval();model.load_state_dict(checkpoint['weights'])
    directory=CACHE/'targets'
    if directory.exists():raise FileExistsError('Preserve existing target output')
    directory.mkdir(parents=True);pending=[];metadata=[];began=time.perf_counter()
    def outputs():
        for offset in range(0,len(rows),16):
            batch=rows[offset:offset+16];lengths=[arrays[r['id']].shape[1] for r in batch]
            x=np.zeros((len(batch),768,max(lengths)),dtype='float32')
            for i,r in enumerate(batch):x[i,:,:lengths[i]]=arrays[r['id']]
            with torch.no_grad():
                ctc,intent,target,_=model(torch.from_numpy(x).cuda())
                ctc,intent,target=ctc.cpu().numpy(),intent.cpu().numpy(),target.cpu().numpy()
            for i,(row,n) in enumerate(zip(batch,lengths)):
                path=directory/(row['pcm_sha256']+'.npy');np.save(path,ctc[i,:,:n].astype('float16'),allow_pickle=False)
                pending.append({'frame_path':str(path.relative_to(ROOT)),'frame_sha256':sha(path),
                    'intent_logits':reduce(intent[i,:,:n],'c t -> c','mean').tolist(),
                    'target_logits':reduce(target[i,:,:n],'c t -> c','mean').tolist()})
                yield rearrange(ctc[i,:,:n],'c t -> t c')
    with ParallelScorer(entries,workers=4,return_scores=True) as scorer:
        for i,(row,item) in enumerate(zip(rows,scorer.map(outputs()),strict=True)):
            indices,values,valid=top_distinct(item['scores'],entries,cfg['candidate_count'])
            metadata.append({'id':row['id'],'pcm_sha256':row['pcm_sha256'],**pending[i],
                'indices':indices,'scores':values,'valid':valid})
            if (i+1)%400==0:print('DISTILL_TARGETS',i+1,'/',len(rows),'seconds',round(time.perf_counter()-began,1),flush=True)
    path=CACHE/'targets.jsonl'
    with path.open('x',encoding='utf8',newline='\n') as stream:
        for item in metadata:stream.write(json.dumps(item)+'\n')
    immutable(HERE/'targets-freeze.json',{'path':str(path.relative_to(ROOT)),'sha256':sha(path),
        'rows':len(metadata),'student_protocol_sha256':sha(HERE/'student-protocol.json'),
        'teacher_checkpoint_sha256':selected['checkpoint_sha256'],'test_inference_accessed':False})
    print('DISTILL_TARGETS_COMPLETE',len(metadata),flush=True)


def stage(name,rows,dev,arrays,targets,frames,entries,norm,protocol):
    cfg=protocol['training'];directory=CACHE/('student-'+name)
    if directory.exists():raise FileExistsError('Preserve existing student run')
    directory.mkdir(parents=True);seed=protocol['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    generator=np.random.default_rng(seed);model=ResidualVCM(**protocol['model']).cuda()
    model.load_state_dict(torch.load(ROOT/protocol['initial_student']['checkpoint'],map_location='cpu',weights_only=True)['weights'])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'],eta_min=cfg['minimum_learning_rate'])
    groups={'human_positive':[r for r in rows if not r['synthetic'] and r['expected']],
        'human_negative':[r for r in rows if not r['synthetic'] and r['reference_status']=='unsupported'],
        'numeral':[r for r in rows if r['reference_status']=='auxiliary'],
        'synthetic':[r for r in rows if r['synthetic']]}
    pools={k:(BalancedRows(v,generator) if k in ('human_positive','synthetic') else CyclingRows(v,generator)) for k,v in groups.items()}
    criterion=torch.nn.CTCLoss(blank=0,zero_infinity=False);history=[];best=None;best_key=None
    for epoch in range(1,cfg['epochs']+1):
        model.train();began=time.perf_counter();losses=np.zeros(6)
        for step in range(cfg['steps_per_epoch']):
            batch=[]
            for key,n in cfg['batch'].items():batch.extend(pools[key].take(n))
            generator.shuffle(batch);opt.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),cfg['microbatch_size']):
                micro=batch[offset:offset+cfg['microbatch_size']];lengths=[arrays[r['id']].shape[1] for r in micro]
                x=np.zeros((len(micro),40,max(lengths)),dtype='float32')
                for i,r in enumerate(micro):x[i,:,:lengths[i]]=arrays[r['id']]
                ctc,intent,target,_=model(torch.from_numpy(x).cuda());lt=torch.tensor(lengths,device='cuda')
                ct=criterion(rearrange(ctc.log_softmax(1),'b c t -> t b c'),
                    torch.tensor([p for r in micro for p in r['phones']],device='cuda',dtype=torch.long),
                    torch.tensor(lengths),torch.tensor([len(r['phones']) for r in micro]))
                il=torch.tensor([semantic_label(r) for r in micro],device='cuda')
                tl=torch.tensor([target_label(r) for r in micro],device='cuda')
                pooled_i=pooled_logits(intent,lt);pooled_t=pooled_logits(target,lt)
                it=torch.nn.functional.cross_entropy(pooled_i,il,ignore_index=-100) if torch.any(il!=-100) else intent.sum()*0
                tt=torch.nn.functional.cross_entropy(pooled_t,tl,ignore_index=-100) if torch.any(tl!=-100) else target.sum()*0
                sq=ctc.sum()*0;fr=ctc.sum()*0;sem=ctc.sum()*0
                if name=='distilled':
                    positions=[i for i,r in enumerate(micro) if r['id'] in targets]
                    if positions:
                        items=[targets[micro[i]['id']] for i in positions]
                        paths=[[entries[j]['phones'] for j in t['indices']] for t in items]
                        values=finite_scores(ctc[positions],[lengths[i] for i in positions],paths)
                        sq=distribution_loss(values,torch.tensor([t['scores'] for t in items],device='cuda'),
                            torch.tensor([t['valid'] for t in items],device='cuda'),cfg['sequence_temperature'])
                        temperature=cfg['head_temperature'];frame_terms=[]
                        for i in positions:
                            teacher=aligned_teacher_logits(torch.from_numpy(frames[micro[i]['id']]).cuda(),lengths[i])
                            logp=(ctc[i,:,:lengths[i]]/temperature).log_softmax(0)
                            p=(teacher/temperature).softmax(0)
                            frame_terms.append(torch.nn.functional.kl_div(logp,p,reduction='sum')/lengths[i]*temperature**2)
                        fr=reduce(frame_terms,'b ->','mean')
                        for student,key in ((pooled_i,'intent_logits'),(pooled_t,'target_logits')):
                            teacher=torch.tensor([t[key] for t in items],device='cuda')
                            sem=sem+torch.nn.functional.kl_div((student[positions]/temperature).log_softmax(1),
                                (teacher/temperature).softmax(1),reduction='batchmean')*temperature**2
                fraction=len(micro)/len(batch)
                loss=(ct+cfg['intent_weight']*it+cfg['target_weight']*tt+cfg['sequence_weight']*sq+
                    cfg['frame_weight']*fr+cfg['semantic_weight']*sem)*fraction
                if not torch.isfinite(loss):raise ValueError('Nonfinite student distillation loss')
                loss.backward();losses+=fraction*np.asarray([v.item() for v in (ct,it,tt,sq,fr,sem)])
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
        scheduler.step();metric={'arm':name,'epoch':epoch,'seconds':time.perf_counter()-began,'losses':(losses/cfg['steps_per_epoch']).tolist()}
        checkpoint={'weights':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'model_config':model.config,'epoch':epoch,'stage':name,'seed':seed,'protocol_sha256':sha(HERE/'student-protocol.json')}
        torch.save(checkpoint,directory/'last.pt')
        if epoch%cfg['development_every_epochs']==0:
            records=score(model,dev,norm,entries,'cuda',workers=4,arrays=arrays)
            hm=precise_metrics(r for r in records if not r['synthetic'])['overall']
            sm=precise_metrics(r for r in records if r['synthetic'])['overall']
            key=((hm['recall']+sm['recall'])/2,-hm['negative_errors'],-epoch)
            metric['development']={'human':hm,'synthetic':sm};metric['selected']=best_key is None or key>best_key
            save(directory/f'dev-{epoch:03d}.json',records)
            if metric['selected']:
                path=directory/f'epoch-{epoch:03d}.pt';torch.save(checkpoint,path);best_key=key
                best={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'epoch':epoch,
                    'arm':name,'development':metric['development'],'key':list(key),'test_inference_accessed':False}
                save(directory/'selection.json',best)
        history.append(metric);save(directory/'history.json',history)
        print('STUDENT_TRAIN',json.dumps(metric),flush=True)
    return best


def run():
    configure();protocol=freeze();target_freeze=json.loads((HERE/'targets-freeze.json').read_text())
    verify_bindings(ROOT,{target_freeze['path']:target_freeze['sha256']})
    targets={r['id']:r for r in read_rows(ROOT/target_freeze['path'])};frames={}
    for identity,item in targets.items():
        if sha(ROOT/item['frame_path'])!=item['frame_sha256']:raise ValueError('Teacher posterior identity changed')
        frames[identity]=np.load(ROOT/item['frame_path'],allow_pickle=False).astype('float32')
    rows=[r for r in read_rows(PARENT_CACHE/'prepared/train.jsonl') if r['training_eligible']]
    validate_training_rows(rows);by_id={r['id']:r for r in rows}
    for identity,item in targets.items():
        if identity not in by_id or by_id[identity]['pcm_sha256']!=item['pcm_sha256']:
            raise ValueError('Distillation targets do not belong to frozen training audio')
    dev=development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    norm=json.loads((PARENT_CACHE/'prepared/normalization.json').read_text());arrays=load_arrays(rows+dev,norm)
    entries=json.loads((PARENT/'catalog.json').read_text())['entries']
    arms={}
    for name in ('hard','distilled'):arms[name]=stage(name,rows,dev,arrays,targets,frames,entries,norm,protocol)
    winner=max(arms,key=lambda name:arms[name]['key'])
    verify_bindings(ROOT,protocol['bindings']);verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    immutable(HERE/'student-selection-before-calibration.json',{'arms':arms,'winner':winner,
        'protocol_sha256':sha(HERE/'student-protocol.json'),'test_inference_accessed':False})
    print('STUDENT_DISTILLATION_COMPLETE',winner,flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='targets':make_targets()
    else:run()
