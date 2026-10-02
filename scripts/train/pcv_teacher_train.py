"""Human-first finite teacher probe; no external downstream speech or ASR."""
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from einops import rearrange, reduce
from scripts.models.pcv_teacher import FiniteTeacherHead
from scripts.runtime.pcv_schema import INTENTS
from scripts.train.pcv_teacher_data import HERE, CACHE, PARENT, rows_for_probe, check_rows, freeze_acquisition
from scripts.train.pcv_public_data import sha, save, read_rows
from scripts.train.pcv_public_training import CyclingRows, pooled_logits
from scripts.train.pcv_updated_training import BalancedRows
from scripts.train.pcv_updated_heads import semantic_label, target_label
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_eval import probabilities
from scripts.test.pcv_public_qualification import precise_metrics, verify_bindings
from scripts.pcv_parallel import ParallelScorer


def ctc_eligible(row, frames):
    phones = row['phones']
    return frames >= len(phones)+sum(a == b for a, b in zip(phones, phones[1:]))


def feature_arrays(rows):
    frozen = json.loads((HERE/'teacher-feature-freeze.json').read_text())
    verify_bindings(ROOT, {frozen['index']: frozen['sha256']})
    index = {r['id']: r for r in read_rows(ROOT/frozen['index'])}
    arrays = {}
    for row in rows:
        item = index[row['id']]
        if item['pcm_sha256'] != row['pcm_sha256'] or sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Feature/source identity mismatch: '+row['id'])
        arrays[row['id']] = np.load(ROOT/item['path'], allow_pickle=False).astype('float32')
    return arrays


def evaluate(model, rows, arrays, entries):
    records, heads = [], []
    model.eval()
    def outputs():
        for offset in range(0, len(rows), 16):
            batch = rows[offset:offset+16]
            lengths = [arrays[r['id']].shape[1] for r in batch]
            x = np.zeros((len(batch), 768, max(lengths)), dtype='float32')
            for i, row in enumerate(batch): x[i, :, :lengths[i]] = arrays[row['id']]
            with torch.no_grad():
                ctc, intent, target, _ = model(torch.from_numpy(x).cuda())
                ctc, intent, target = ctc.cpu().numpy(), intent.cpu().numpy(), target.cpu().numpy()
            for i, n in enumerate(lengths):
                heads.append((probabilities(reduce(intent[i,:,:n], 'c t -> c', 'mean')),
                              probabilities(reduce(target[i,:,:n], 'c t -> c', 'mean'))))
                yield rearrange(ctc[i,:,:n], 'c t -> t c')
    with ParallelScorer(entries, workers=4) as scorer:
        for i, (row, item) in enumerate(zip(rows, scorer.map(outputs()), strict=True)):
            pred = item['prediction']; ip, tp = heads[i]; winner = int(ip.argmax())
            candidate = pred['command']; ci = INTENTS.index(candidate['intent'])+1 if candidate else 0
            records.append({'id': row['id'], 'source': row['source'], 'speaker': row['speaker'],
                'synthetic': row['synthetic'], 'expected': row['expected'], 'text': row['text'],
                'candidate': candidate, 'score': pred['score'], 'margin': pred['margin'],
                'intent_index': winner, 'intent_probability': float(ip[ci]),
                'top_intent_probability': float(ip[winner]), 'target_named_probability': float(tp[1]),
                'candidate_phrase_id': pred['phrase_id']})
    return records


def freeze_training(rows, excluded):
    path = HERE/'teacher-training-protocol.json'
    if path.exists():
        protocol = json.loads(path.read_text()); verify_bindings(ROOT, protocol['bindings'])
        return protocol
    files = ['scripts/train/pcv_teacher_train.py','scripts/train/pcv_teacher_data.py',
        'scripts/test/test_pcv_teacher_train.py',
        'scripts/models/pcv_teacher.py','scripts/models/pcv_residual.py',
        'scripts/train/pcv_updated_heads.py','scripts/train/pcv_public_training.py',
        'scripts/train/pcv_updated_training.py','scripts/test/pcv_public_qualification.py',
        'scripts/test/pcv_public_eval.py','scripts/pcv_parallel.py',
        'experiments/phrase-contract-v1/pcvcm/decoder.py',
        'experiments/pcv-updated-voice-v1/catalog.json','scripts/runtime/pcv_schema.py',
        'experiments/pcv-updated-distill-v1/teacher-source-protocol.json',
        'experiments/pcv-updated-distill-v1/teacher-feature-freeze.json',
        'experiments/pcv-updated-voice-v3/selection-before-calibration.json']
    protocol = {'experiment': 'pcv-updated-distill-v1', 'stage': 'frozen-encoder finite-head teacher probe',
        'seed': 731, 'model': {'width':256, 'dilations':[1,2,4,8,1,2]},
        'training': {'human_epochs':32, 'mixed_epochs':24, 'steps_per_epoch':96,
            'batch_size':32, 'microbatch_size':8, 'human_learning_rate':.001,
            'mixed_learning_rate':.0003, 'minimum_learning_rate':.00002,
            'weight_decay':.0001, 'gradient_clip':5., 'intent_weight':.3, 'target_weight':.15,
            'human_batch': {'human_positive':28,'human_negative':4,'synthetic':0},
            'mixed_batch': {'human_positive':12,'human_negative':4,'synthetic':16},
            'development_every_epochs':4, 'augmentation':'none', 'positive_sampling':'uniform intent'},
        'fit_ids':[r['id'] for r in rows], 'ctc_frame_exclusions':excluded,
        'numerals':'No isolated numeral fitting in this bounded teacher probe; complete supplied command recordings only.',
        'selection': 'Human stage: human raw complete recall, then fewer human negative errors, then earlier epoch. Mixed stage: mean human and supplied-synthetic raw complete recall, then fewer human negative errors, then earlier epoch.',
        'distillation_gate': {'human_recall_min':.8038095238095238, 'synthetic_recall_min':.9522222222222222,
            'explanation':'At least 3 percentage points above v3 human development recall, at most 2 points below its synthetic recall. This is a teacher-use gate, not deployment acceptance.'},
        'test_access':'None in this probe. Prior parent aggregate test results were seen; no test-error-driven sample selection.',
        'bindings':{p:sha(ROOT/p) for p in files},
        'installed_artifact_hashes':{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()},
        'hardware':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda},
        'runtime':False, 'installation_authorized':False}
    immutable(path, protocol)
    return protocol


def stage(name, initial, rows, dev, arrays, entries, protocol):
    cfg = protocol['training']; directory = CACHE/name
    if directory.exists(): raise FileExistsError('Preserve existing teacher run: '+str(directory))
    directory.mkdir(parents=True)
    seed = protocol['seed']+(100 if name == 'mixed' else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = FiniteTeacherHead(**protocol['model']).cuda()
    if initial: model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=True)['weights'])
    human = [r for r in rows if not r['synthetic']]
    groups = {'human_positive':[r for r in human if r['expected']],
              'human_negative':[r for r in human if not r['expected']],
              'synthetic':[r for r in rows if r['synthetic']]}
    plan = cfg[name+'_batch']; epochs = cfg[name+'_epochs']
    pools = {k:(CyclingRows(v,generator) if k == 'human_negative' else BalancedRows(v,generator))
             for k,v in groups.items()}
    for key,n in plan.items():
        if n: check_rows(groups[key], fitting=True, human_only=name=='human')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg[name+'_learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg['minimum_learning_rate'])
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    best = None; best_key = None; history = []; seen = set()
    for epoch in range(1, epochs+1):
        model.train(); began = time.perf_counter(); losses = np.zeros(3)
        for step in range(cfg['steps_per_epoch']):
            batch = []
            for key,n in plan.items():
                if n: batch.extend(pools[key].take(n))
            generator.shuffle(batch); seen.update(r['id'] for r in batch); opt.zero_grad(set_to_none=True)
            for offset in range(0,len(batch),cfg['microbatch_size']):
                micro = batch[offset:offset+cfg['microbatch_size']]
                lengths = [arrays[r['id']].shape[1] for r in micro]
                x = np.zeros((len(micro),768,max(lengths)),dtype='float32')
                for i,r in enumerate(micro): x[i,:,:lengths[i]] = arrays[r['id']]
                ctc,intent,target,_ = model(torch.from_numpy(x).cuda())
                lt = torch.tensor(lengths,device='cuda')
                ct = criterion(rearrange(ctc.log_softmax(1),'b c t -> t b c'),
                    torch.tensor([p for r in micro for p in r['phones']],device='cuda',dtype=torch.long),
                    torch.tensor(lengths),torch.tensor([len(r['phones']) for r in micro]))
                il = torch.tensor([semantic_label(r) for r in micro],device='cuda')
                tl = torch.tensor([target_label(r) for r in micro],device='cuda')
                it = torch.nn.functional.cross_entropy(pooled_logits(intent,lt),il,ignore_index=-100)
                tt = torch.nn.functional.cross_entropy(pooled_logits(target,lt),tl,ignore_index=-100) if torch.any(tl!=-100) else target.sum()*0
                fraction = len(micro)/len(batch)
                loss = fraction*(ct+cfg['intent_weight']*it+cfg['target_weight']*tt)
                if not torch.isfinite(loss): raise ValueError('Nonfinite teacher loss')
                loss.backward(); losses += fraction*np.asarray([ct.item(),it.item(),tt.item()])
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']); opt.step()
        scheduler.step()
        metric = {'stage':name,'epoch':epoch,'seconds':time.perf_counter()-began,
            'losses':(losses/cfg['steps_per_epoch']).tolist(),'unique_recordings_seen':len(seen)}
        checkpoint = {'weights':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
            'model_config':model.config,'epoch':epoch,'stage':name,'seed':seed,
            'protocol_sha256':sha(HERE/'teacher-training-protocol.json')}
        torch.save(checkpoint,directory/'last.pt')
        if epoch%cfg['development_every_epochs'] == 0:
            selected_dev = [r for r in dev if not r['synthetic']] if name=='human' else dev
            records = evaluate(model,selected_dev,arrays,entries)
            hm = precise_metrics(r for r in records if not r['synthetic'])['overall']
            sm = precise_metrics(r for r in records if r['synthetic'])['overall']
            main = hm['recall'] if name=='human' else ((hm['recall'] or 0)+(sm['recall'] or 0))/2
            key = (main,-hm['negative_errors'],-epoch)
            metric['development'] = {'human':hm,'synthetic':sm}; metric['selected'] = best_key is None or key>best_key
            save(directory/f'dev-{epoch:03d}.json',records)
            if metric['selected']:
                path = directory/f'epoch-{epoch:03d}.pt'; torch.save(checkpoint,path)
                best_key = key; best = {'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),
                    'stage':name,'epoch':epoch,'development':metric['development'],'key':list(key),
                    'test_inference_accessed':False}
                save(directory/'selection.json',best)
        history.append(metric); save(directory/'history.json',history)
        print('TEACHER_TRAIN',json.dumps(metric),flush=True)
    immutable(directory/'seen-recordings.json', {'ids':sorted(seen)})
    return best


def run():
    source = freeze_acquisition()
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    rows,dev = rows_for_probe(); arrays = feature_arrays(rows+dev)
    excluded = [{'id':r['id'],'reason':'insufficient_teacher_CTC_frames','frames':arrays[r['id']].shape[1]}
                for r in rows if not ctc_eligible(r,arrays[r['id']].shape[1])]
    rows = [r for r in rows if ctc_eligible(r,arrays[r['id']].shape[1])]
    protocol = freeze_training(rows,excluded)
    entries = json.loads((PARENT/'catalog.json').read_text())['entries']
    base = stage('human',None,[r for r in rows if not r['synthetic']],dev,arrays,entries,protocol)
    mixed = stage('mixed',ROOT/base['checkpoint'],rows,dev,arrays,entries,protocol)
    gate = protocol['distillation_gate']; hm=mixed['development']['human']; sm=mixed['development']['synthetic']
    permitted = hm['recall']>=gate['human_recall_min'] and sm['recall']>=gate['synthetic_recall_min']
    verify_bindings(ROOT,source['bindings']); verify_bindings(ROOT,protocol['bindings'])
    verify_bindings(ROOT,protocol['installed_artifact_hashes'])
    immutable(HERE/'teacher-selection.json', {'human':base,'mixed':mixed,'distillation_gate_pass':permitted,
        'acceptance_qualified':False,'test_inference_accessed':False,
        'teacher_training_protocol_sha256':sha(HERE/'teacher-training-protocol.json')})
    print('TEACHER_PROBE_COMPLETE',permitted,flush=True)


if __name__=='__main__': run()
