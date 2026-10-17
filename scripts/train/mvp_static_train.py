"""Bounded direct-command continuation of our own compact and full encoders."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import random
import re
import platform
import socket
import time
import numpy as np
import torch
from scripts.models.mvp_static import StaticCommandNet, ExportStatic
from scripts.runtime.mvp_static import build_catalog, decide, fit_policy, digest
from scripts.runtime.pcv_schema import baseline_cases, canonical

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'experiments/vcm-mvp-hey-iris-v1'
CACHE=ROOT/'.cache/vcm-mvp-hey-iris-v1'
PREP=ROOT/'.cache/speaker-synthetic-completion-v1/prepared-v1'
SEED=53


def select_study(version):
    global HERE,CACHE
    if version not in ('vcm-mvp-hey-iris-v1','vcm-mvp-hey-iris-v2'):raise ValueError('Unknown MVP study version')
    HERE=ROOT/'experiments'/version;CACHE=ROOT/'.cache'/version


def eligible(row):
    return (row['split']=='train' and row.get('qa_eligible', True) and
            row.get('reference_status') in ('supported','unsupported') and
            not row.get('slot_metadata_conflict',False))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Preserve existing evidence: '+str(path))
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare():
    target=HERE/'protocol.json'
    if target.exists():
        cfg=json.loads(target.read_text())
        verify_protocol(cfg)
        return cfg
    CACHE.mkdir(parents=True,exist_ok=True)
    entries=json.loads((PREP/'catalog.json').read_text())['entries']
    # Additive, user-specified fixed names. Zero acoustic evidence stays explicit.
    songs=[('Talk that Talk','twice'),('Queen of Hearts','twice'),('Basics','twice'),
           ('Trouble','twice'),('Brave','twice'),('Gone','twice'),('When We Were Kids','twice'),('Enemy','imagine_dragons')]
    for title,artist in songs:
        song='song.'+'_'.join(title.lower().split())
        for explicit in (False,True):
            slots={'song_id':song}
            if explicit: slots['artist_id']='artist.'+artist
            text='play '+title+(' by '+artist.replace('_',' ') if explicit else '')
            entries.append({'text':text,'command':{'intent':'PLAY_MUSIC','slots':slots}})
    for intent,verb in [('LIGHT_ON','turn on'),('LIGHT_OFF','turn off')]:
        entries.append({'text':verb+' the ceiling light','command':{'intent':intent,'slots':{'light_id':'light.ceiling'}}})
    catalog=build_catalog(entries);lookup={canonical(c['command']):c['command_id'] for c in catalog}
    baseline=baseline_cases()
    assert len(baseline)==93 and len({canonical(c['command']) for c in baseline})==31
    assert all(canonical(c['command']) in lookup for c in baseline)
    inputs={};views={}
    for split in ('train','dev','calibration','test','holdout'):
        if split in ('test','holdout'):
            paths=[PREP/'views/evaluation'/f'{split}.jsonl']
        else: paths=[PREP/'views'/kind/f'{split}.jsonl' for kind in ('human','synthetic')]
        rows=[]
        for path in paths:
            inputs[str(path.relative_to(ROOT))]=digest(path)
            for original in read_rows(path):
                row=copy.deepcopy(original)
                row['expected']=row.get('expected',row.get('command'))
                row['synthetic']=row.get('kind')=='synthetic'
                row['speaker']=row.get('speaker',row.get('speaker_id',''))
                if not row['speaker']: raise ValueError('Missing speaker identity')
                if split=='train' and not eligible(row): continue
                if row.get('reference_status')=='ambiguous' or row.get('metric_eligible') is False:
                    row['class_index']=None
                else:
                    key=canonical(row['expected'])
                    if key not in lookup: raise ValueError('A valid command is absent from the frozen inventory')
                    row['class_index']=lookup[key]
                rows.append(row)
        views[split]=rows
        path=CACHE/'views'/f'{split}.jsonl';path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(''.join(canonical(r)+'\n' for r in rows))
    overlaps={}
    for i,a in enumerate(views):
        for b in list(views)[i+1:]:
            cells={}
            for field in ('id','speaker','pcm_sha256','pair_id'):
                x={r.get(field) for r in views[a] if r.get(field)};y={r.get(field) for r in views[b] if r.get(field)}
                cells[field]=len(x&y)
            overlaps[a+'/'+b]=cells
            if any(cells.values()): raise ValueError('Cross-split identity overlap: '+a+'/'+b+str(cells))
    coverage={}
    for split,rows in views.items():
        coverage[split]={kind:dict(rows=sum(r['synthetic']==(kind=='synthetic') for r in rows),
             commands=dict(Counter(str(r['class_index']) for r in rows if r['synthetic']==(kind=='synthetic'))),
             speakers=len({r['speaker'] for r in rows if r['synthetic']==(kind=='synthetic')})) for kind in ('human','synthetic')}
    train_counts=Counter(r['class_index'] for r in views['train'])
    for c in catalog:c['training_recordings']=train_counts[c['command_id']]
    write(HERE/'catalog.json',catalog)
    write(HERE/'data-audit.json',dict(coverage=coverage,cross_split_overlap=overlaps,
          human_gender_and_accent_independently_verified=False,
          zero_evidence_commands=[c['command_id'] for c in catalog if not c['training_recordings']],
          phonetic_or_role_annotations_required=False,baseline_wordings=93,baseline_commands=31))
    checkpoints={'compact':ROOT/'.cache/speaker-compact-recall-v1/training/mixed-control/epoch-028.pt',
                 'full':ROOT/'.cache/speaker-final-refinement-v1/training/mixed-control/epoch-032.pt'}
    bench=ROOT/'.cache/vcm-benchmark/ab39857cf73dd067d068fed388c321ab160dc191'
    binding={str(p.relative_to(ROOT)):digest(p) for p in bench.rglob('*') if p.is_file() and p.suffix in ('.py','.csv','.md')}
    cfg=dict(experiment='vcm-mvp-hey-iris-v1',seed=SEED,inputs=inputs,benchmark_bindings=binding,
       benchmark_commit='ab39857cf73dd067d068fed388c321ab160dc191',benchmark_rows=196,
       primary_revision='25111444af3adff7588d86ab27c304bb080895cf',
       supplemental_revision='230c8b8d58caf78e4bc90886d9c6fa77fe6024b8',
       catalog_sha256=digest(HERE/'catalog.json'),classes=len(catalog),
       checkpoints={k:dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for k,p in checkpoints.items()},
       architecture='our existing 6-block contextual encoder, masked mean, atomic command CE',
       initialization='declared continuation from our own encoder; random direct classifier',
       epochs=16,steps_per_epoch=64,batch=32,microbatch=8,learning_rate=.0003,
       selection='raw human macro-speaker exact command accuracy, then human count, synthetic count; development only',
       calibration='joint command probability and margin; maximize correct human accepts with zero observed fitting wrong/unknown accepts',
       augmentations={'frequency_mask':'0..4 bins, probability .5','time_mask':'0..min(10,5% frames), probability .5',
                      'citation':'https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html',
                      'new_waveform_augmentation':'none in this encoder-continuation comparison'},
       benchmark_fixed=True,test_used_for_selection=False,actions_enabled=False,
       physical_pi_measured=False,normalization_sha256=digest(PREP/'normalization.json'))
    write(target,cfg)
    print(json.dumps({'prepared':{k:len(v) for k,v in views.items()},'classes':len(catalog),'overlap_pairs':len(overlaps)}),flush=True)
    return cfg


def verify_protocol(cfg):
    bindings={**cfg['inputs'], **cfg['benchmark_bindings']}
    bindings.update({v['path']:v['sha256'] for v in cfg['checkpoints'].values()})
    for path, expected in bindings.items():
        if digest(ROOT/path)!=expected: raise ValueError('Input identity mismatch: '+path)
    for path, expected in ((HERE/'catalog.json', cfg['catalog_sha256']),
                           (PREP/'normalization.json', cfg['normalization_sha256'])):
        if digest(path)!=expected: raise ValueError('Input identity mismatch: '+str(path))


def create_run(name, run_id=None):
    if run_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',run_id):
        raise ValueError('Invalid run ID')
    out=CACHE/name if run_id is None else CACHE/'runs'/run_id/name
    out.mkdir(parents=True,exist_ok=False)
    return out


def apply_recipe(cfg, recipe, protocol):
    if recipe is None:return dict(cfg)
    extension=json.loads(Path(recipe).read_text())
    if extension['parent_protocol_sha256']!=digest(protocol):raise ValueError('Changed parent protocol')
    overrides=extension['overrides']
    if not overrides.keys() <= {'epochs','steps_per_epoch','min_epochs','patience'}:
        raise ValueError('Training budget extension cannot change frozen inputs or architecture')
    if any(type(v) is not int or v<=0 for v in overrides.values()):raise ValueError('Invalid training budget')
    return {**cfg,**overrides}


def load_features(rows,norm):
    from vcm.frontend import features
    import soundfile as sf
    # einops lazily registers backends without a lock; initialize before workers.
    features(np.zeros(400,dtype=np.float32),norm['mean'],norm['std'])
    previous={r['id']:r for name in ('train.jsonl','dev.jsonl') for r in read_rows(PREP/name)}
    cache=CACHE/'features';cache.mkdir(exist_ok=True)
    mean=np.asarray(norm['mean'],dtype=np.float32)[:,None];std=np.asarray(norm['std'],dtype=np.float32)[:,None]
    def one(row):
        old=previous.get(row['id'],{})
        if old.get('features'):
            p=Path(old['features'])
            if digest(p)!=old['features_sha256']:raise ValueError('Feature hash mismatch')
            x=(np.load(p,allow_pickle=False).astype(np.float32)-mean)/std
        else:
            p=cache/(row['pcm_sha256']+'.npy')
            if p.exists():x=np.load(p,allow_pickle=False)
            else:
                pcm,rate=sf.read(ROOT/row['audio'],dtype='float32')
                if rate!=16000 or pcm.ndim!=1:raise ValueError('Unexpected audio format')
                if digest(ROOT/row['audio'])!=row['sha256']:raise ValueError('Audio hash mismatch')
                x=features(pcm,norm['mean'],norm['std'])
                with p.open('wb') as stream:np.save(stream,x,allow_pickle=False)
        if x.shape[0]!=40 or not np.isfinite(x).all():raise ValueError('Malformed features')
        return row['id'],x
    arrays={}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,(key,x) in enumerate(pool.map(one,rows),1):
            arrays[key]=x
            if i%5000==0:print('FEATURES',i,len(rows),flush=True)
    return arrays


def batch_data(rows,arrays,device,rng=None):
    from scripts.train.speaker_final_train import augment
    lengths=[arrays[r['id']].shape[1] for r in rows]
    x=np.zeros((len(rows),40,max(lengths)),dtype=np.float32)
    for i,row in enumerate(rows):
        a=arrays[row['id']]
        x[i,:,:len(a[0])]=augment(a,rng) if rng is not None else a
    return torch.from_numpy(x).to(device),torch.tensor(lengths,device=device),torch.tensor([r['class_index'] for r in rows],device=device)


def predict(model,rows,arrays,device):
    result=[];model.eval()
    ordered=sorted(rows,key=lambda r:arrays[r['id']].shape[1])
    with torch.no_grad():
        for start in range(0,len(ordered),32):
            batch=ordered[start:start+32];x,l,y=batch_data(batch,arrays,device)
            logits=model(x,l).cpu().numpy()
            result.extend(dict(id=r['id'],speaker=r['speaker'],split=r['split'],synthetic=r['synthetic'],
                  expected_index=r['class_index'],logits=v.tolist()) for r,v in zip(batch,logits))
    return result


def raw_key(records):
    human=[r for r in records if not r['synthetic'] and r['expected_index'] not in (None,0)]
    speakers=defaultdict(list)
    for r in human:speakers[r['speaker']].append(int(np.argmax(r['logits']))==r['expected_index'])
    count=sum(sum(x) for x in speakers.values())
    macro=float(np.mean([np.mean(v) for v in speakers.values()])) if speakers else 0.
    syn=sum(r['synthetic'] and r['expected_index'] is not None and int(np.argmax(r['logits']))==r['expected_index'] for r in records)
    return (macro,count,syn)


def train(name,device,run_id=None,recipe=None):
    cfg=apply_recipe(prepare(),recipe,HERE/'protocol.json');out=create_run(name,run_id)
    write(out/'effective-protocol.json',cfg)
    write(out/'run-inputs.json',dict(protocol_sha256=digest(HERE/'protocol.json'),
        views={s:digest(CACHE/'views'/f'{s}.jsonl') for s in ('train','dev','calibration','test','holdout')},
        trainer_sha256=digest(Path(__file__)),model_sha256=digest(ROOT/'scripts/models/mvp_static.py'),
        host=socket.gethostname(),python=platform.python_version(),torch=torch.__version__,
        cuda=torch.version.cuda,run_id=run_id,restart_from_parent=True,exact_interrupted_resume=False))
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this process')
        torch.cuda.set_per_process_memory_fraction(.12);torch.backends.cuda.matmul.allow_tf32=True
    catalog=json.loads((HERE/'catalog.json').read_text());norm=json.loads((PREP/'normalization.json').read_text())
    train_rows=read_rows(CACHE/'views/train.jsonl');dev=[r for r in read_rows(CACHE/'views/dev.jsonl') if r['class_index'] is not None]
    arrays=load_features(train_rows+dev,norm)
    width=128 if name=='compact' else 192
    initial=ROOT/cfg['checkpoints'][name]['path']
    if digest(initial)!=cfg['checkpoints'][name]['sha256']:raise ValueError('Changed parent weights')
    def make():
        m=StaticCommandNet(len(catalog),width);state=torch.load(initial,map_location='cpu',weights_only=True)
        if 'config' in state:m.load_state_dict(state['weights'])
        else:m.initialize_our_encoder(state)
        return m.to(device)
    # Small-slice overfit is disposable and does not initialize the selected fit.
    byclass={r['class_index']:r for r in train_rows if not r['synthetic']}
    small=list(byclass.values())[:24]; probe=make();opt=torch.optim.AdamW(probe.parameters(),lr=.002)
    x,l,y=batch_data(small,arrays,device); began=time.perf_counter();probe_accuracy=0.
    for step in range(300):
        probe.train();opt.zero_grad();loss=torch.nn.functional.cross_entropy(probe(x,l),y);loss.backward();opt.step()
        probe.eval()
        with torch.no_grad():probe_accuracy=float((probe(x,l).argmax(1)==y).float().mean())
        if probe_accuracy>=.99:break
    write(out/'overfit.json',dict(rows=len(small),steps=step+1,accuracy=probe_accuracy,passed=probe_accuracy>=.99,seconds=time.perf_counter()-began))
    if probe_accuracy<.99:raise RuntimeError('Small-slice overfit failed; do not launch full fitting')
    del probe,opt;torch.manual_seed(SEED);model=make();rng=np.random.default_rng(SEED)
    groups=defaultdict(lambda:defaultdict(list))
    for r in train_rows:groups['synthetic' if r['synthetic'] else 'human'][r['class_index']].append(r)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.001)
    history=[];best=None;began=time.perf_counter();stale=0
    for epoch in range(1,cfg['epochs']+1):
        model.train();losses=[]
        lr=cfg['learning_rate']*(.05+.95*.5*(1+math.cos(math.pi*(epoch-1)/cfg['epochs'])))
        for group in optimizer.param_groups:group['lr']=lr
        for step in range(cfg['steps_per_epoch']):
            rows=[]
            for kind in ('human','synthetic'):
                labels=list(groups[kind])
                priority=[i for i in cfg.get('priority_classes',[]) if i in labels and i!=0]
                for _ in range(16):
                    if rng.random()<.20 and 0 in labels:label=0
                    elif priority and rng.random()<cfg.get('priority_probability',0.):label=int(rng.choice(priority))
                    else:label=int(rng.choice(labels))
                    candidates=groups[kind][label]
                    speakers=sorted({r['speaker'] for r in candidates});speaker=str(rng.choice(speakers))
                    candidates=[r for r in candidates if r['speaker']==speaker]
                    rows.append(candidates[int(rng.integers(len(candidates)))])
            optimizer.zero_grad(set_to_none=True);total=0.
            for start in range(0,len(rows),8):
                x,l,y=batch_data(rows[start:start+8],arrays,device,rng)
                loss=torch.nn.functional.cross_entropy(model(x,l),y,label_smoothing=.02)/4
                loss.backward();total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(total)
        records=predict(model,dev,arrays,device);key=raw_key(records)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),raw_selection=list(key),seconds=time.perf_counter()-began)
        history.append(row);print(name,json.dumps(row),flush=True)
        if best is None or key>tuple(best['key']):
            torch.save(dict(weights=model.cpu().state_dict(),config=model.config,epoch=epoch),out/'best.pt');model.to(device)
            best=dict(epoch=epoch,key=list(key),checkpoint_sha256=digest(out/'best.pt'));stale=0
        else:stale+=1
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if epoch>=cfg.get('min_epochs',8) and stale>=cfg.get('patience',4):break
    write(out/'selection.json',dict(**best,parameters=sum(p.numel() for p in model.parameters()),
          training_seconds=time.perf_counter()-began,device=device,gpu=torch.cuda.get_device_name() if device=='cuda' else None,
          seed=SEED,training_rows=len(train_rows),test_used_for_selection=False))
    print('TRAINING_COMPLETE',name,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','train']);p.add_argument('--model',choices=['compact','full'],default='compact');p.add_argument('--device',default='cpu')
    p.add_argument('--run-id',help='New output directory; preserves interrupted runs')
    p.add_argument('--recipe',type=Path,help='Hash-bound, separately versioned training-budget extension')
    p.add_argument('--study',choices=['vcm-mvp-hey-iris-v1','vcm-mvp-hey-iris-v2'],default='vcm-mvp-hey-iris-v1')
    a=p.parse_args();select_study(a.study);prepare() if a.stage=='prepare' else train(a.model,a.device,a.run_id,a.recipe)


if __name__=='__main__':main()
