"""One-GPU v2 fitting, preserving the model, matched recipe and human-dev selector."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from scripts.models.vcm_intent_slots import IntentSlotsVCM, joint_loss
from scripts.runtime.vcm_intent_slots_contract import digest
from scripts.runtime.vcm_intent_slots_entities import enroll_examples
from scripts.runtime.vcm_intent_slots_entities import catalog_ids
from scripts.train.vcm_intent_slots_data import rows, sha, write_new
from scripts.train.vcm_intent_slots_mixed_data import eligible, prepare_features, validate_protocol
from scripts.train.vcm_intent_slots_train import development_score, make_targets, sample_indices
from scripts.test.speaker_demo_suite import read_audio
from vcm.frontend import features

ROOT=Path(__file__).resolve().parents[2]
RECIPE={'human_only':False, 'new_synthesis':True, 'optimizer':'AdamW', 'learning_rate':.001,
        'weight_decay':.0001, 'gradient_norm_limit':1., 'batch_size':64, 'epochs':30,
        'steps_per_epoch':96, 'whole_command_contrastive_weight':0., 'auxiliary_ctc_weight':0.,
        'update_artifacts':False}


def validate_training(protocol, arm, seed, device):
    if (protocol.get('experiment_id')!='vcm-mixed-final-v2' or arm not in ('uniform','balanced')
            or seed not in (53,71,89) or device!='cuda'
            or any(protocol['training'].get(k)!=v for k,v in RECIPE.items())
            or protocol.get('installation_allowed') is not False
            or protocol.get('provider_actions_allowed') is not False):
        raise ValueError('Unapproved recipe, seed, arm, CPU fitting or installation')


def targets(batch, contract, device):
    value=make_targets(batch,contract,device)
    # Defaults and unobserved roles never become invented acoustic entity labels.
    for n,row in enumerate(batch):
        for r,role in enumerate(contract['entity_roles']):
            if row.get('role_labels',{}).get(role)!='present':
                value['entity_ids'][n][r]=None
    return value


def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_feature_cache(receipt):
    """Load verified raw features once; normalization and batch arithmetic stay identical."""
    cache={}
    for identifier,binding in receipt.items():
        if sha(binding['path'])!=binding['sha256']:
            raise ValueError('Changed feature hash before training')
        x=np.load(binding['path'],allow_pickle=False)
        if x.shape!=(40,binding['frames']) or x.dtype!=np.float32 or not np.isfinite(x).all():
            raise ValueError('Invalid raw training/development feature array')
        x.setflags(write=False);cache[identifier]=x
    return cache


def choose_candidate(results):
    # The fixed arm/seed iteration order is the last fallback after metric/epoch ties.
    return max(results,key=lambda r:(*r['development_key'],-r['epoch']))


def require_enrollment(data, protocol, contract):
    required=catalog_ids(contract); required['light'].discard('light.all')
    coverage=defaultdict(set); voices=defaultdict(set)
    query=[r for s,rs in data.items() if s!='enrollment' for r in rs]
    people={r['speaker_id'] for r in query}; pcm={r['pcm_sha256'] for r in query}
    for row in data['enrollment']:
        eligible(row,protocol,contract)
        role,entity=row['entity_role'],row['entity_id']
        if (row['role_labels'].get(role)!='present' or row['speaker_id'] in people
                or row['pcm_sha256'] in pcm or entity not in required.get(role,set())):
            raise ValueError('Unobserved, overlapping or invalid named enrollment')
        coverage[(role,entity)].add(row['pcm_sha256'])
        voices[(role,entity)].add(row['voice_id'])
    if any(len(coverage[(role,entity)])!=3 or len(voices[(role,entity)])!=3
           for role,names in required.items() for entity in names):
        raise ValueError('Every named entity needs three distinct separated enrollment takes')


def enrollment_policy(protocol):
    return {'study':protocol['experiment_id'], 'source':protocol['source']['repository'],
            'source_revision':protocol['source']['revision'], 'voice_ids':protocol['voices']['enrollment']}


def memory_for_model(model, enrollment, protocol, contract, norm, audio_root, binding=None):
    """Development uses state identity; selected export explicitly supplies file identity."""
    binding=binding or {'encoder':state_hash(model), 'frontend':sha(ROOT/'vcm/frontend.py'),
                       'normalization':digest(norm)}
    examples=[]
    device=next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for row in enrollment:
            eligible(row,protocol,contract)
            x=features(read_audio(row,Path(audio_root)),norm['mean'],norm['std'])
            out=model(torch.tensor(x[None],device=device),torch.tensor([x.shape[1]],device=device))
            role=row['entity_role']
            vector=out['entity_embeddings'][0,contract['entity_roles'].index(role)].cpu().numpy()
            examples.append({**row,'role':role,'embedding':vector.tolist()})
    memory=enroll_examples(examples,binding,contract,provenance_policy=enrollment_policy(protocol))
    policies={}
    for role in contract['entity_roles']:
        count=len({ex['entity_id'] for ex in memory['examples'] if ex['role']==role})
        policies[role]={**protocol['enrollment']['development_policy'],'encoder_binding':binding,
                        'catalog_hash':contract['catalog_hash'],'catalog_sizes':[count]}
    return memory,{'entities':policies,'require_calibration':False,
                   'status':'predeclared_raw_development_only','contract_hash':contract['contract_hash']}


def fit(protocol, inputs, contract, output, *, arm, seed, receipt, normalization,feature_cache=None):
    validate_training(protocol,arm,seed,'cuda')
    validate_protocol(protocol,contract)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one actual CUDA GPU must be visible')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    data={s:rows(b['path']) for s,b in inputs['views'].items() if s in ('train','dev','enrollment')}
    for subset in data.values():
        for row in subset:
            eligible(row,protocol,contract)
    require_enrollment(data,protocol,contract)
    human_dev=[r for r in data['dev'] if r['kind']=='human']
    synthetic_dev=[r for r in data['dev'] if r['kind']=='synthetic']
    if not human_dev:
        raise ValueError('Human development selection cannot be replaced with synthetic scores')
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    model=IntentSlotsVCM(contract).to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    initial=state_hash(model)
    write_new(output/'run-identity.json',{'pid':os.getpid(),'arm':arm,'seed':seed,'initial_state_sha256':initial,
              'initialization':'random','device':torch.cuda.get_device_name(0),'visible_device_count':1,
              'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,
              'cuda_runtime':torch.version.cuda,'protocol':protocol,'inputs':inputs,
              'model_source_sha256':sha(ROOT/'scripts/models/vcm_intent_slots.py'),
              'training_source_sha256':sha(__file__),'parameters':model.parameter_counts(),
              'feature_cache':'verified raw host arrays' if feature_cache is not None else 'disk reads',
              'selection':'original human development only','final_tts_accessed':False})
    best_key,selected=None,None
    counts_total,mass=defaultdict(int),defaultdict(int)
    history=[]; start_run=time.perf_counter()
    for epoch in range(30):
        start_epoch=time.perf_counter(); model.train()
        old={k:v.detach().clone() for k,v in model.named_parameters()}
        indices=sample_indices(data['train'],arm,seed+epoch*1009,64*96)
        total_loss=0.; gradients=[]
        for offset in range(0,len(indices),64):
            batch=[data['train'][i] for i in indices[offset:offset+64]]
            from scripts.train.vcm_intent_slots_train import feature_batch
            x,lengths=feature_batch(batch,receipt,normalization,'cuda',feature_cache)
            optimizer.zero_grad(set_to_none=True)
            loss,counts=joint_loss(model(x,lengths),targets(batch,contract,'cuda'))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; failed run retained')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            gradients.append(float(norm.detach()))
            optimizer.step()
            total_loss+=float(loss.detach())
            for k,v in counts.items(): counts_total[k]+=v
            for row in batch: mass[row['kind']]+=1
        delta=float(torch.sqrt(sum((v-old[k]).square().sum() for k,v in model.named_parameters())).detach())
        if not np.isfinite(delta) or delta<=0:
            raise ValueError('No finite actual parameter updates; failed run retained')
        memory,policy=memory_for_model(model,data['enrollment'],protocol,contract,normalization,inputs['audio_root'])
        key,metrics=development_score(model,human_dev,receipt,normalization,contract,memory,policy,feature_cache)
        _,synthetic_metrics=development_score(model,synthetic_dev,receipt,normalization,contract,memory,policy,feature_cache)
        record={'epoch':epoch+1,'optimizer_updates':96,'cumulative_updates':(epoch+1)*96,
                'loss':total_loss/96,'gradient_norm_min':min(gradients),'gradient_norm_max':max(gradients),
                'finite_losses_and_gradients':True,'actual_parameter_delta_l2':delta,
                'selection_key':key,'human_development':metrics,'synthetic_development':synthetic_metrics,
                'synthetic_development_used_for_selection':False,'seconds':time.perf_counter()-start_epoch}
        history.append(record); write_new(output/f'epoch-{epoch+1:02}.json',record)
        if best_key is None or key>best_key:
            best_key,selected=key,epoch+1
            path=output/f'checkpoint-{selected:02}.pt'
            torch.save({'weights':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'contract':contract,'normalization':normalization,'training':protocol['training'],
                        'seed':seed,'arm':arm,'epoch':selected,'status':'mixed_silver_unqualified_candidate',
                        'inputs':inputs},path)
        print(json.dumps({'arm':arm,'seed':seed,'epoch':epoch+1,'loss':record['loss'],
                          'human_dev_recall':metrics['all']['recall'],'selected_epoch':selected,
                          'seconds':record['seconds']}),flush=True)
    result={'status':'mixed_silver_unqualified_candidate','arm':arm,'seed':seed,'epoch':selected,
            'checkpoint':str(output/f'checkpoint-{selected:02}.pt'),
            'checkpoint_sha256':sha(output/f'checkpoint-{selected:02}.pt'),'development_key':best_key,
            'optimizer_updates':2880,'sampling_mass':dict(mass),'supervision_denominators':dict(counts_total),
            'initial_state_sha256':initial,'final_state_sha256':state_hash(model),
            'seconds':time.perf_counter()-start_run,'test_accessed':False,'final_tts_accessed':False}
    write_new(output/'history.json',history); write_new(output/'selection-before-calibration.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=ROOT/'experiments/vcm-mixed-final-v2/protocol.json')
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); protocol=json.loads(a.protocol.read_text()); inputs=json.loads(a.inputs.read_text())
    if sha(a.protocol)!=inputs['protocol']['sha256']:
        raise ValueError('Frozen protocol changed')
    for b in inputs['views'].values():
        if sha(b['path'])!=b['sha256']: raise ValueError('Frozen data manifest changed')
    if sha(inputs['contract']['path'])!=inputs['contract']['sha256']:
        raise ValueError('Frozen slot contract changed')
    contract=json.loads(Path(inputs['contract']['path']).read_text())
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('Verify exactly one CUDA training device before feature preparation')
    from scripts.train.vcm_intent_slots_mixed_data import audit_views
    audit=audit_views({s:b['path'] for s,b in inputs['views'].items()}, {'views':inputs['views']},
                      a.output/'preflight-audit',protocol=protocol,contract=contract,
                      audio_root=inputs['audio_root'])
    if not audit['fit_ready']: raise ValueError('Missing full fitting intent/role coverage')
    require_enrollment({s:rows(b['path']) for s,b in inputs['views'].items()},protocol,contract)
    receipt,norm=prepare_features(rows(inputs['views']['train']['path'])+rows(inputs['views']['dev']['path']),
                                  inputs['audio_root'],a.output/'features',protocol=protocol,contract=contract)
    cache=load_feature_cache(receipt)
    write_new(a.output/'feature-loading-receipt.json',{'feature_receipt_sha256':sha(a.output/'features/features.json'),
              'normalization_sha256':sha(a.output/'features/normalization.json'),'arrays':len(cache),
              'bytes':sum(x.nbytes for x in cache.values()),'raw_feature_hashes_verified':True,
              'normalization_and_batch_arithmetic':'unchanged from default disk feature_batch',
              'final_TTS_or_evaluation_features_loaded':False})
    results=[]
    for arm in ('uniform','balanced'):
        for seed in (53,71,89):
            results.append(fit(protocol,inputs,contract,a.output/f'{arm}-{seed}',arm=arm,seed=seed,
                               receipt=receipt,normalization=norm,feature_cache=cache))
    winner=choose_candidate(results)
    write_new(a.output/'grid.json',results)
    write_new(a.output/'selection-before-calibration.json',
              {'selected':winner,'candidate_count':len(results),'selection':'human development only',
               'protocol_sha256':sha(a.protocol),'inputs_sha256':sha(a.inputs),
               'calibration_accessed':False,'test_accessed':False,'final_tts_accessed':False})


if __name__=='__main__': main()
