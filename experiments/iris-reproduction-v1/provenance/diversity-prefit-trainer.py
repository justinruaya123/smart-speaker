"""Frozen, bounded direct-command diversity comparison and release selection.

All runtime outputs are finite typed commands. Synthesis is an offline data
stage, never an inference dependency. Historical datasets remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import shutil
import socket
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/iris-release-v1'
CACHE = ROOT / '.cache/iris-release-v1/diversity'
INPUTS = ROOT / '.cache/iris-release-v1/inputs'
BASE = ROOT / '.cache/vcm-mvp-hey-iris-v2'
PARENT_RUN = BASE / 'runs/augmented-20261017'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != data:
            raise FileExistsError('Preserve existing evidence: ' + str(path))
    else:
        path.write_text(data)


def designed_voices():
    accents = ['Philippine English', 'American English', 'British English',
               'Indian English', 'Australian English', 'Irish English',
               'South African English', 'Canadian English']
    styles = ['an adult woman with a medium pitch', 'an adult man with a lower pitch',
              'an adult with an androgynous higher voice']
    voices = []
    for index in range(24):
        split = 'train' if index < 12 else 'dev' if index < 16 else 'calibration' if index < 20 else 'voice-test'
        voices.append(dict(id=f'iris-release-voice-{index:02d}', split=split,
            prompt=f'A clear natural voice of {styles[index//8]} speaking {accents[index%8]}. '
                   'Comfortable conversational delivery. Clean dry speech, no music.',
            design_seed=83000+index, identity_source='original_synthetic_voice_design',
            accent_verified=False, demographic_labels='requested design conditions, not verified people'))
    return voices


def generation_plan(catalog):
    from scripts.runtime.pcv_schema import baseline_cases, canonical
    lookup = {canonical(c['command']): c['command_id'] for c in catalog}
    personal = [c for c in catalog if c['command'] and
        (c['command']['slots'].get('light_id') == 'light.ceiling' or
         c['command']['slots'].get('song_id') in {'song.talk_that_talk','song.queen_of_hearts',
             'song.basics','song.trouble','song.brave','song.gone','song.when_we_were_kids','song.enemy'})]
    prompts = [(r.get('spoken_text', r['text']), r['command'], 'command', False) for r in baseline_cases()]
    prompts += [(c['wordings'][0], c['command'], 'command', False) for c in personal]
    near = ['Hey Siri.', 'Hey Irish.', 'Iris.', 'Hey.', 'Hey Jarvis.', 'Hey Alice.',
            'Hey Paris.', 'They hire us.', 'A virus.', 'Here it is.', 'Good morning.', 'Are you ready?']
    negative = ['Play Mystery Planet.', 'Turn on the attic light.', 'Set brightness to 140 percent.',
                'Set a timer for minus five seconds.', 'Call a stranger.', 'Set a timer.',
                'Turn off the outdoor camera.', 'Set brightness to twenty or sixty percent.',
                'Play a song whose name you do not know.', 'Set the temperature to one hundred degrees.']
    prompts += [(text, None, 'wake_negative', False) for text in near]
    prompts += [(text, None, 'command_negative', False) for text in negative]
    prompts += [('Hey Iris.', None, 'wake', True) for _ in range(8)]
    result = []
    for voice in designed_voices():
        for index, (text, command, role, wake) in enumerate(prompts):
            key = hashlib.sha256(f"{voice['id']}/{index}/{text}".encode()).hexdigest()[:20]
            result.append(dict(id='iris-release/'+key, voice_id=voice['id'], speaker=voice['id'],
                split=voice['split'], text=text, command=command, expected=command,
                class_index=lookup[canonical(command)], role=role, wake_positive=wake,
                pair_id='iris-release/'+key))
    return result


def freeze():
    HERE.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    path = HERE/'protocol.json'
    if path.exists():
        cfg = json.loads(path.read_text())
        for rel, expected in cfg['input_bindings'].items():
            if digest(ROOT/rel) != expected:
                raise ValueError('Changed frozen input: '+rel)
        return cfg
    INPUTS.mkdir(exist_ok=True, parents=True)
    source_catalog = ROOT/'experiments/vcm-mvp-hey-iris-v2/catalog.json'
    shutil.copyfile(source_catalog, INPUTS/'catalog-parent.json')
    normalization = ROOT/'.cache/speaker-synthetic-completion-v1/prepared-v1/normalization.json'
    shutil.copyfile(normalization, INPUTS/'normalization.json')
    bindings = {}
    for split in ('train','dev','calibration','test','holdout','personal-test'):
        target = INPUTS/'base-views'/f'{split}.jsonl'; target.parent.mkdir(exist_ok=True)
        # Strip historical absolute feature-cache locations. Audio paths/hashes,
        # original membership, labels and speaker identities are preserved.
        rows = read_rows(BASE/'views'/f'{split}.jsonl')
        for row in rows:
            for key in ('features','feature_path','features_sha256','feature_sha256'):
                row.pop(key, None)
        target.write_text(''.join(json.dumps(r, sort_keys=True, allow_nan=False)+'\n' for r in rows))
        bindings[str(target.relative_to(ROOT))] = digest(target)
    for name in ('compact','full'):
        p = INPUTS/f'{name}-initializer.pt'
        shutil.copyfile(PARENT_RUN/name/'best.pt', p)
        bindings[str(p.relative_to(ROOT))] = digest(p)
    for name in ('catalog-parent.json','normalization.json'):
        bindings[str((INPUTS/name).relative_to(ROOT))] = digest(INPUTS/name)
    catalog = json.loads(source_catalog.read_text())
    write(HERE/'voices.json', designed_voices())
    write(HERE/'generation-plan.json', generation_plan(catalog))
    cfg = dict(experiment='iris-release-v1', baseline='vcm-mvp-hey-iris-v3',
        requirements=[f'ME-{i:02d}' for i in range(1,9)]+[f'UC-{i:02d}' for i in range(1,11)],
        hypothesis='Speaker-balanced whole commands and cross-speaker command invariance improve accepted exact commands on held-out voices within fixed development error limits',
        input_bindings=bindings, catalog_classes=len(catalog),
        generation_plan_sha256=digest(HERE/'generation-plan.json'), voices_sha256=digest(HERE/'voices.json'),
        generation=dict(batch=4, temperature=.8, top_p=.95, max_new_tokens=224, cuda_memory_fraction=.35),
        training=dict(epochs=16, steps_per_epoch=96, min_epochs=8, patience=6, batch=32,
            microbatch=8, learning_rate=.0001, weight_decay=.001, smoothing=.02, clip=5.,
            seeds=[53,67], primary_seed=53, contrastive_weight=.1, contrastive_temperature=.1,
            source_balance='16 human + 16 synthetic per optimizer update',
            unknown_sampling_probability=.2, priority_probability=.8,
            gain_db=[-6,6], masks=dict(frequency_bins=[0,4],frequency_probability=.5,
                                     time_frames_max=10,time_fraction_max=.05,time_probability=.5)),
        loss_citation='https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html',
        masking_citation='https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html',
        demo_risk=dict(human=.03, synthetic=.01),
        initialization='continuation of our own V2 encoder and direct head, no external acoustic initializer',
        selection='dev accepted macro human-speaker exact recall, human correct count, synthetic correct count, cost; thresholds fit development/calibration only',
        run_budget='compact/full x CE/SupCon seed53, then chosen width x CE/SupCon seed67',
        promotion=dict(personal_correct_min=30, personal_positive_n=36,
            human_correct_not_below_baseline=True, human_unknown_not_above_baseline=True,
            official_exact_not_below_baseline=True),
        benchmark_rows=196, benchmark_commit='ab39857cf73dd067d068fed388c321ab160dc191',
        test_used_for_selection=False, original_human_personal_positive_n=0,
        original_human_wake_positive_n=0, physical_pi_measured=False,
        no_asr=True, no_llm_in_runtime=True,
        source_snapshot_commit='600be8a561d436c70d1e02f92aa44f9b693bef3f',
        code_before_fitting={p:digest(ROOT/p) for p in ('scripts/train/iris_diversity.py',
            'scripts/models/mvp_static.py','scripts/models/speaker_demo.py','vcm/frontend.py')})
    write(path, cfg)
    return cfg


def generate():
    cfg = freeze()
    complete = HERE/'generation-complete.json'
    manifest = CACHE/'generated.jsonl'
    if complete.exists():
        from scripts.iris_release import resolve_audio
        receipt = json.loads(complete.read_text())
        if (digest(manifest) != receipt['manifest_sha256'] or
                digest(HERE/'protocol.json') != receipt['protocol_sha256']):
            raise ValueError('Changed completed synthesis manifest or protocol')
        rows = read_rows(manifest)
        plan = json.loads((HERE/'generation-plan.json').read_text())
        if len(rows) != receipt['rows'] or {r['id'] for r in rows} != {r['id'] for r in plan}:
            raise ValueError('Incomplete published synthesis snapshot')
        for row in rows:
            if digest(resolve_audio(row, root=ROOT)) != row['sha256']:
                raise ValueError('Changed completed synthesis audio: '+row['id'])
        return receipt
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel
    voices = designed_voices()
    plan = json.loads((HERE/'generation-plan.json').read_text())
    if digest(HERE/'generation-plan.json') != cfg['generation_plan_sha256']:
        raise ValueError('Changed generation plan')
    source = json.loads((ROOT/'data/sources/qwen-synthetic-completion.source.json').read_text())
    for r in source['files']:
        if digest(ROOT/r['path']) != r['sha256']:
            raise ValueError('Changed synthesis source: '+r['path'])
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.35)
    def load(variant):
        return Qwen3TTSModel.from_pretrained(str(ROOT/'.cache/tts/models'/('Qwen3-TTS-12Hz-1.7B-'+variant)),
            device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    anchor_dir = CACHE/'anchors'; anchor_dir.mkdir(exist_ok=True)
    missing = [v for v in voices if not (anchor_dir/(v['id']+'.json')).exists()]
    if missing:
        design = load('VoiceDesign')
        for voice in missing:
            torch.manual_seed(voice['design_seed'])
            text = 'The room is quiet and I am speaking clearly at a comfortable pace.'
            waves, rate = design.generate_voice_design(text=text,language='English',instruct=voice['prompt'],
                max_new_tokens=384,do_sample=True,temperature=.8)
            native = anchor_dir/(voice['id']+'.wav'); sf.write(native,waves[0],rate,subtype='PCM_16')
            write(native.with_suffix('.json'), {**voice,'native_audio':str(native.relative_to(ROOT)),
                'native_sha256':digest(native),'text':text})
            print('ANCHOR',voice['id'],flush=True)
        del design; torch.cuda.empty_cache()
    anchors = {v['id']:json.loads((anchor_dir/(v['id']+'.json')).read_text()) for v in voices}
    if len({a['native_sha256'] for a in anchors.values()}) != 24:
        raise ValueError('Duplicate designed voice anchor')
    model = load('Base'); clone = {}
    for voice in voices:
        anchor = anchors[voice['id']]; native = ROOT/anchor['native_audio']
        if digest(native) != anchor['native_sha256']: raise ValueError('Changed voice anchor')
        clone[voice['id']] = model.create_voice_clone_prompt(ref_audio=str(native),ref_text=anchor['text'])[0]
    path = CACHE/'generated.jsonl'
    prior = read_rows(path) if path.exists() else []
    for row in prior:
        if digest(ROOT/row['audio']) != row['sha256']: raise ValueError('Changed generated audio')
    done = {r['id'] for r in prior}; began = time.perf_counter()
    with path.open('a') as stream:
        for voice in voices:
            cells = [r for r in plan if r['voice_id']==voice['id']]
            for offset in range(0,len(cells),cfg['generation']['batch']):
                batch = cells[offset:offset+cfg['generation']['batch']]
                if all(r['id'] in done for r in batch): continue
                if any(r['id'] in done for r in batch):
                    raise ValueError('Partial generation batch needs explicit recovery; seeds cannot silently change')
                seed = int(hashlib.sha256((voice['id']+'/'+str(offset)).encode()).hexdigest()[:8],16)
                torch.manual_seed(seed)
                waves, rate = model.generate_voice_clone(text=[r['text'] for r in batch],language=['English']*len(batch),
                    voice_clone_prompt=[clone[voice['id']]]*len(batch),non_streaming_mode=True,
                    do_sample=True,temperature=.8,top_p=.95,max_new_tokens=224)
                if len(waves) != len(batch): raise ValueError('Synthesis count mismatch')
                for row, wave in zip(batch,waves):
                    pcm = resample_poly(np.asarray(wave,dtype='float32'),16000//math.gcd(16000,rate),rate//math.gcd(16000,rate)).astype('float32')
                    if not 1600 <= len(pcm) <= 240000 or not np.isfinite(pcm).all():
                        raise ValueError('Generated duration/nonfinite QA failed: '+row['id'])
                    if float(np.sqrt(np.mean(pcm**2))) < 1e-4 or np.mean(np.abs(pcm)>=.999) > .01:
                        raise ValueError('Generated energy/clipping QA failed: '+row['id'])
                    target = CACHE/'audio'/voice['id']/(row['id'].split('/')[-1]+'.wav')
                    target.parent.mkdir(exist_ok=True,parents=True)
                    if target.exists(): raise FileExistsError('Unreceipted generated waveform '+str(target))
                    sf.write(target,pcm,16000,subtype='PCM_16'); saved,_ = sf.read(target,dtype='float32')
                    record = {**row,'audio':str(target.relative_to(ROOT)),'sha256':digest(target),
                        'pcm_sha256':hashlib.sha256(saved.tobytes()).hexdigest(),'samples':len(saved),
                        'synthetic':True,'kind':'synthetic','source':'Qwen3-original-designed-iris-release-v1',
                        'anchor_sha256':anchors[voice['id']]['native_sha256'],'generation_seed':seed,
                        'qa_eligible':True,'metric_eligible':True,'slot_metadata_conflict':False,
                        'reference_status':'supported' if row['command'] else 'unsupported',
                        'listener_verified':False,'accent_verified':False}
                    stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); done.add(row['id'])
                print('GENERATED',len(done),len(plan),round(time.perf_counter()-began,1),flush=True)
    write(HERE/'generation-complete.json',dict(rows=len(done),manifest_sha256=digest(path),
        protocol_sha256=digest(HERE/'protocol.json'),voice_n=24,original_human_n=0,
        independent_listening=False,generation_seconds=time.perf_counter()-began))


def prepare():
    cfg=freeze(); path=CACHE/'data-audit.json'
    if path.exists():
        audit=json.loads(path.read_text())
        if audit['protocol_sha256']!=digest(HERE/'protocol.json') or audit['catalog_sha256']!=digest(HERE/'catalog.json'):
            raise ValueError('Changed prepared command protocol/catalog')
        for split,expected in audit['views'].items():
            if digest(CACHE/'views'/f'{split}.jsonl')!=expected:raise ValueError('Changed prepared command view: '+split)
        return audit
    generated=read_rows(CACHE/'generated.jsonl')
    complete=json.loads((HERE/'generation-complete.json').read_text())
    if digest(CACHE/'generated.jsonl') != complete['manifest_sha256']:
        raise ValueError('Changed generated manifest')
    planned=json.loads((HERE/'generation-plan.json').read_text())
    if {r['id'] for r in generated}!={r['id'] for r in planned}:raise ValueError('Incomplete generation')
    views={s:read_rows(INPUTS/'base-views'/f'{s}.jsonl') for s in
           ('train','dev','calibration','test','holdout','personal-test')}
    views['voice-test']=[]
    for row in generated:views[row['split']].append(row)
    # personal-test is an intentional diagnostic subset of a separate source,
    # not added to a fitting split. Check every independent cohort pair.
    overlap={}
    for i,a in enumerate(views):
        for b in list(views)[i+1:]:
            cells={k:len({r.get(k) for r in views[a] if r.get(k)} &
                         {r.get(k) for r in views[b] if r.get(k)})
                   for k in ('id','speaker','pcm_sha256','pair_id')}
            overlap[a+'/'+b]=cells
            if any(cells.values()):raise ValueError('Cross-split identity overlap '+a+'/'+b+str(cells))
    catalog=json.loads((INPUTS/'catalog-parent.json').read_text())
    counts=Counter(r['class_index'] for r in views['train'])
    for c in catalog:c['training_recordings']=counts[c['command_id']]
    write(HERE/'catalog.json',catalog)
    for split,rows in views.items():
        target=CACHE/'views'/f'{split}.jsonl';target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))
    audit=dict(protocol_sha256=digest(HERE/'protocol.json'),split_counts={s:len(v) for s,v in views.items()},
        source_counts={s:dict(Counter(r['source'] for r in v)) for s,v in views.items()},
        cross_split_overlap=overlap,views={s:digest(CACHE/'views'/f'{s}.jsonl') for s in views},
        original_human_personal_positive_n=0,independent_listening=False,
        original_benchmark_unchanged=True,catalog_sha256=digest(HERE/'catalog.json'))
    write(path,audit);write(HERE/'data-audit.json',audit);return audit


def gain_features(values,norm,rng):
    if rng is None:return values
    std=np.asarray(norm['std'],dtype='float32')[:,None]
    mean=np.asarray(norm.get('mean',[0.]*40),dtype='float32')[:,None]
    offset=float(rng.uniform(-6,6))*math.log(10)/10
    raw=values*std+mean
    # A gain offset in log-power space with the identical frontend floor.
    return ((np.maximum(raw+offset,math.log(1e-8))-mean)/std).astype('float32')


def sample_batch(rows,rng,priority,*,groups=None):
    if groups is None:
        groups=defaultdict(lambda:defaultdict(list))
        for r in rows:groups[bool(r['synthetic'])][r['class_index']].append(r)
    result=[]
    for kind in (False,True):
        labels=list(groups[kind]);preferred=[x for x in priority if x in labels and x]
        for _ in range(8):
            if rng.random()<.2 and 0 in labels:label=0
            elif preferred and rng.random()<.8:label=int(rng.choice(preferred))
            else:label=int(rng.choice(labels))
            candidates=groups[kind][label]
            speakers=sorted({r['speaker'] for r in candidates});speaker=str(rng.choice(speakers))
            selected=[r for r in candidates if r['speaker']==speaker]
            result.append(selected[int(rng.integers(len(selected)))])
            others=[s for s in speakers if s!=speaker]
            second=str(rng.choice(others)) if others else speaker
            selected=[r for r in candidates if r['speaker']==second]
            result.append(selected[int(rng.integers(len(selected)))])
    return result


def accepted_key(records):
    people=defaultdict(list);synthetic=0
    for r in records:
        if r['expected_index'] in (None,0):continue
        correct=r.get('accepted_index')==r['expected_index']
        if r['synthetic']:synthetic+=int(correct)
        else:people[r['speaker']].append(correct)
    return (float(np.mean([np.mean(v) for v in people.values()])) if people else 0.,
            sum(sum(v) for v in people.values()),synthetic)


def load_arrays(rows,norm):
    import soundfile as sf
    from vcm.frontend import features
    features(np.zeros(400,dtype='float32'),norm['mean'],norm['std'])
    directory=CACHE/'features';directory.mkdir(exist_ok=True)
    def one(row):
        target=directory/(row['pcm_sha256']+'.npy')
        if target.exists():x=np.load(target,allow_pickle=False)
        else:
            from scripts.iris_release import resolve_audio
            audio=resolve_audio(row,root=ROOT)
            if digest(audio)!=row['sha256']:raise ValueError('Changed audio '+row['id'])
            pcm,rate=sf.read(audio,dtype='float32')
            if rate!=16000 or pcm.ndim!=1:raise ValueError('Unexpected audio format')
            x=features(pcm,norm['mean'],norm['std'])
            with target.open('wb') as stream:np.save(stream,x,allow_pickle=False)
        if x.shape[0]!=40 or not x.shape[1] or not np.isfinite(x).all():raise ValueError('Invalid features')
        return row['id'],x
    with ThreadPoolExecutor(max_workers=12) as pool:return dict(pool.map(one,rows))


def _batch(rows,arrays,device,norm,rng=None):
    import torch
    lengths=[arrays[r['id']].shape[1] for r in rows]
    x=np.zeros((len(rows),40,max(lengths)),dtype='float32')
    for i,r in enumerate(rows):
        a=gain_features(arrays[r['id']],norm,rng)
        if rng is not None:
            a=a.copy()
            if rng.random()<.5:
                width=int(rng.integers(5));start=int(rng.integers(41-width));a[start:start+width]=0
            if rng.random()<.5:
                width=int(rng.integers(min(10,int(a.shape[1]*.05))+1))
                start=int(rng.integers(a.shape[1]-width+1));a[:,start:start+width]=0
        x[i,:,:lengths[i]]=a
    return torch.from_numpy(x).to(device),torch.tensor(lengths,device=device),torch.tensor([r['class_index'] for r in rows],device=device)


def _predict(model,rows,arrays,device,norm):
    import torch
    result=[];model.eval();ordered=sorted(rows,key=lambda r:arrays[r['id']].shape[1])
    with torch.no_grad():
        for offset in range(0,len(ordered),32):
            batch=ordered[offset:offset+32];x,l,y=_batch(batch,arrays,device,norm)
            logits=model(x,l).cpu().numpy()
            result.extend(dict(id=r['id'],speaker=r['speaker'],synthetic=r['synthetic'],split=r['split'],
                expected_index=r['class_index'],logits=v.tolist()) for r,v in zip(batch,logits))
    return result


def _apply(records,catalog,policy):
    from scripts.runtime.mvp_static import decide
    for r in records:
        d=decide(r['logits'],catalog,policy)
        r['raw_index']=d['class_index'];r['accepted_index']=d['class_index'] if d['command'] else None
    return records


def _policy(records,catalog,cfg):
    from scripts.runtime.mvp_static import fit_policy
    trained=[c['command_id'] for c in catalog if c['training_recordings']]
    policy=fit_policy(records,trained,catalog=catalog,risk=cfg['demo_risk'],
        threshold_strategy='observed',grouping='class',min_correct=2)
    policy.update(enabled=True,qualified=False,qualification_reason='Demo policy; human wake/ownership and physical Pi unmeasured')
    return policy


def capture_device_rng(device):
    if not str(device).startswith('cuda'):return None
    import torch
    return torch.cuda.get_rng_state(device)


def restore_device_rng(state,device):
    if state is None or not str(device).startswith('cuda'):return
    import torch
    if isinstance(state,(list,tuple)):
        if len(state)!=1:raise ValueError('Expected one selected GPU RNG state')
        state=state[0]
    torch.cuda.set_rng_state(state.cpu(),device)


def completed_selection(folder,expected):
    path=folder/'selection.json'
    if not path.exists():return None
    selected=json.loads(path.read_text());inputs=json.loads((folder/'run-inputs.json').read_text())
    if any(inputs.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed fit inputs changed; declare a new experiment/run')
    if digest(folder/'best.pt')!=selected['checkpoint_sha256']:
        raise ValueError('Completed fit checkpoint changed')
    return selected


def train(model='compact',arm='ce',seed=53,device='cuda'):
    import torch
    from scripts.models.mvp_static import StaticCommandNet
    from scripts.models.speaker_demo import cross_speaker_contrastive
    cfg=freeze();audit=prepare();hp=cfg['training']
    out=CACHE/'runs'/f'{model}-{arm}-s{seed}';out.mkdir(exist_ok=True,parents=True)
    if model not in ('compact','full') or arm not in ('ce','supcon') or seed not in (53,67):raise ValueError('Unregistered run')
    initializer=INPUTS/f'{model}-initializer.pt'
    expected=dict(protocol_sha256=digest(HERE/'protocol.json'),data_audit_sha256=digest(CACHE/'data-audit.json'),
        initializer_sha256=digest(initializer),model=model,arm=arm,seed=seed,
        trainer_sha256=digest(Path(__file__)),model_source_sha256=digest(ROOT/'scripts/models/mvp_static.py'),
        contrastive_source_sha256=digest(ROOT/'scripts/models/speaker_demo.py'),
        frontend_sha256=digest(ROOT/'vcm/frontend.py'),policy_source_sha256=digest(ROOT/'scripts/runtime/mvp_static.py'))
    cached=completed_selection(out,expected)
    if cached is not None:return cached
    torch.set_num_threads(4);torch.manual_seed(seed);random.seed(seed)
    if device=='cuda':torch.cuda.set_per_process_memory_fraction(.12)
    catalog=json.loads((HERE/'catalog.json').read_text());norm=json.loads((INPUTS/'normalization.json').read_text())
    rows=read_rows(CACHE/'views/train.jsonl');dev=[r for r in read_rows(CACHE/'views/dev.jsonl') if r['class_index'] is not None]
    arrays=load_arrays(rows+dev,norm)
    net=StaticCommandNet(len(catalog),128 if model=='compact' else 192).to(device)
    initializer=INPUTS/f'{model}-initializer.pt'
    state=torch.load(initializer,map_location='cpu',weights_only=True);net.load_state_dict(state['weights'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=hp['learning_rate'],weight_decay=hp['weight_decay'])
    rng=np.random.default_rng(seed);speakers={s:i for i,s in enumerate(sorted({r['speaker'] for r in rows}))}
    priority=sorted({r['class_index'] for r in rows if r.get('source')=='Qwen3-original-designed-iris-release-v1' and r['class_index']})
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:groups[bool(r['synthetic'])][r['class_index']].append(r)
    binding=dict(**expected,host=socket.gethostname(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=list(__import__('sys').argv))
    write(out/'run-inputs.json',binding)
    history=[];best=None;stale=0;first=1;began=time.perf_counter()
    if (out/'last.pt').exists():
        saved=torch.load(out/'last.pt',map_location=device,weights_only=True)
        if any(saved['binding'].get(k)!=v for k,v in expected.items()):raise ValueError('Resume inputs changed')
        net.load_state_dict(saved['weights']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        restore_device_rng(saved['cuda_rng'],device)
        random.setstate(saved['python_rng']);rng.bit_generator.state=saved['numpy_rng']
        history=saved['history'];best=saved['best'];stale=saved['stale'];first=saved['epoch']+1
    for epoch in range(first,hp['epochs']+1):
        net.train();losses=[];pair_rates=[]
        lr=hp['learning_rate']*(.05+.95*.5*(1+math.cos(math.pi*(epoch-1)/hp['epochs'])))
        for group in optimizer.param_groups:group['lr']=lr
        for step in range(hp['steps_per_epoch']):
            batch=sample_batch(rows,rng,priority,groups=groups);optimizer.zero_grad(set_to_none=True);total=0.
            for offset in range(0,32,8):
                micro=batch[offset:offset+8];x,l,y=_batch(micro,arrays,device,norm,rng)
                logits,embedding=net.forward_with_embedding(x,l)
                sp=torch.tensor([speakers[r['speaker']] for r in micro],device=device)
                contrast=cross_speaker_contrastive(embedding,y,sp,temperature=.1)
                loss=(torch.nn.functional.cross_entropy(logits,y,label_smoothing=.02)+(.1 if arm=='supcon' else 0.)*contrast)/4
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();total+=float(loss.detach())
                pair_rates.append(float((((y[:,None]==y[None,:])&(sp[:,None]!=sp[None,:])&(y[:,None]>0)).any(1)).float().mean()))
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(total)
        records=_predict(net,dev,arrays,device,norm)
        policy=_policy(records,catalog,cfg);key=accepted_key(_apply(records,catalog,policy))
        record=dict(epoch=epoch,loss=float(np.mean(losses)),paired_anchor_fraction=float(np.mean(pair_rates)),
                    accepted_selection=list(key),seconds=time.perf_counter()-began)
        history.append(record);print(f'{model}-{arm}-s{seed}',json.dumps(record),flush=True)
        if best is None or key>tuple(best['key']):
            torch.save(dict(weights={k:v.detach().cpu() for k,v in net.state_dict().items()},
                config=net.config,epoch=epoch),out/'best.pt')
            best=dict(epoch=epoch,key=list(key),checkpoint_sha256=digest(out/'best.pt'));stale=0
        else:stale+=1
        save=dict(binding=binding,weights=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            history=history,best=best,stale=stale,torch_rng=torch.get_rng_state(),
            cuda_rng=capture_device_rng(device),
            python_rng=random.getstate(),numpy_rng=rng.bit_generator.state)
        torch.save(save,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if epoch>=hp['min_epochs'] and stale>=hp['patience']:break
    result={**best,'model':model,'arm':arm,'seed':seed,'parameters':sum(p.numel() for p in net.parameters()),
        'training_seconds':time.perf_counter()-began,'epochs_completed':len(history),'device':device,
        'checkpoint':str((out/'best.pt').relative_to(ROOT)),'test_used_for_selection':False}
    write(out/'selection.json',result);return result


def select():
    """Choose width before seed67 repetition; final recipe only on development."""
    cfg=freeze();selections=[]
    for p in sorted((CACHE/'runs').glob('*/selection.json')):selections.append(json.loads(p.read_text()))
    primary=[s for s in selections if s['seed']==53]
    if len(primary)!=4:raise ValueError('Need exactly four primary fits')
    key=lambda s:(*s['key'],-s['parameters'])
    initial=max(primary,key=key);width=initial['model']
    write(HERE/'width-selection-before-repeat.json',dict(model=width,runs=primary,seed=67,arms=['ce','supcon'],test_used_for_selection=False))
    repeated=[s for s in selections if s['seed']==67 and s['model']==width]
    if len(repeated)!=2:return {'stage':'repeat','model':width,'arms':['ce','supcon'],'seed':67}
    chosen=max([s for s in selections if s['model']==width],key=key)
    write(HERE/'development-selection.json',dict(**chosen,all_runs=selections,
        rule=cfg['selection'],test_used_for_selection=False))
    return chosen


def export():
    import torch
    from scripts.models.mvp_static import StaticCommandNet
    from scripts.runtime.mvp_static import Runtime
    from scripts.train.iris_export import export_onnx,export_ncnn
    selected=select()
    if selected.get('stage')=='repeat':raise ValueError('Run seed67 repetition first')
    cfg=freeze();out=CACHE/'selected-export'
    if (out/'bundle.json').exists() and (HERE/'selection-before-test.json').exists():return str(out)
    out.mkdir(exist_ok=True)
    checkpoint=ROOT/selected['checkpoint']
    if digest(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Changed selected checkpoint')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net=StaticCommandNet(**saved['config']);net.load_state_dict(saved['weights']);net.eval();torch.set_num_threads(1)
    onnx=export_onnx(net,out/'model.onnx')
    catalog=json.loads((HERE/'catalog.json').read_text());write(out/'catalog.json',catalog)
    shutil.copyfile(INPUTS/'normalization.json',out/'normalization.json')
    norm=json.loads((out/'normalization.json').read_text())
    rows=[r for split in ('dev','calibration') for r in read_rows(CACHE/'views'/f'{split}.jsonl') if r['class_index'] is not None]
    records=_predict(net,rows,load_arrays(rows,norm),'cpu',norm)
    policy=_policy(records,catalog,cfg);write(out/'policy.json',policy)
    write(out/'bundle.json',dict(format='mvp-static-v1',parameters=selected['parameters'],
        files={n:digest(out/n) for n in ('model.onnx','catalog.json','normalization.json','policy.json')},
        inference_mode='complete_utterance',profile='benchmark-compatible',qualified=False,
        checkpoint_sha256=selected['checkpoint_sha256']))
    native=export_ncnn(net,out/'model.onnx',out/'ncnn')
    write(HERE/'selection-before-test.json',dict(development_selection=selected,
        protocol_sha256=digest(HERE/'protocol.json'),data_audit_sha256=digest(CACHE/'data-audit.json'),
        bundle_sha256=digest(out/'bundle.json'),policy_sha256=digest(out/'policy.json'),
        onnx=onnx,ncnn=native,test_used_for_selection=False))
    (CACHE/'development-predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in _apply(records,catalog,policy)))
    return str(out)


def promotion_guards(results,official,baseline):
    personal=results['personal-test']['overall']
    human=results['test']['strata']['human']
    prior=baseline['project']['test']['strata']['human']
    score=official['accepted']['command_level']
    old=baseline['official']['accepted']['command_level']
    return dict(personal_population_unchanged=personal['positive_n']==36,
        personal_30_of_36=personal['correct_accepts']>=30,
        human_correct_not_below_baseline=human['correct_accepts']>=prior['correct_accepts'],
        human_unknown_not_above_baseline=human['unknown_accepts']<=prior['unknown_accepts'],
        official_population_unchanged=score['n']==old['n']==196,
        official_exact_not_below_baseline=score['accuracy']>=old['accuracy'])


def evaluate():
    from scripts.runtime.mvp_static import Runtime
    from scripts.test.iris_metrics import infer_rows,summarize
    cfg=freeze();lock=json.loads((HERE/'selection-before-test.json').read_text())
    out=CACHE/'selected-export'
    if digest(out/'bundle.json')!=lock['bundle_sha256']:raise ValueError('Changed pre-test selection')
    runtime=Runtime(out,engine='ncnn');results={}
    for split in ('test','holdout','personal-test','voice-test'):
        path=CACHE/f'{split}-predictions.jsonl'
        if path.exists():records=read_rows(path)
        else:
            with path.open('x') as stream:records=infer_rows(runtime,read_rows(CACHE/'views'/f'{split}.jsonl'),stream)
        results[split]=summarize(records,runtime.catalog)
    # Preserve the official scorer unchanged. Its source and input rows remain pinned.
    from scripts.test.iris_benchmark import official_score
    official=official_score(runtime,CACHE/'official-results')
    model=lock['development_selection']['model']
    baseline=json.loads((INPUTS/'baseline-results.json').read_text())['models'][model]
    guards=promotion_guards(results,official,baseline)
    promoted=all(guards.values())
    source=out if promoted else INPUTS/'fallback-bundles'/model
    release=CACHE/'release-export'
    if release.exists():
        if digest(release/'bundle.json')!=digest(source/'bundle.json'):
            raise ValueError('Previously frozen release choice changed')
    else:shutil.copytree(source,release)
    # Test can veto the predeclared candidate, never adjust its thresholds.
    # The fallback width was fixed by development before any test was opened.
    chosen_results=results;chosen_official=official
    if not promoted:
        fallback=Runtime(release,engine='ncnn');chosen_results={}
        for split in ('test','holdout','personal-test','voice-test'):
            path=CACHE/f'fallback-{split}-predictions.jsonl'
            if path.exists():records=read_rows(path)
            else:
                with path.open('x') as stream:records=infer_rows(fallback,read_rows(CACHE/'views'/f'{split}.jsonl'),stream)
            chosen_results[split]=summarize(records,fallback.catalog)
        chosen_official=official_score(fallback,CACHE/'fallback-official-results')
    choice=dict(experiment='iris-release-v1',width=model,promoted=promoted,guards=guards,
        selected='iris-release-v1' if promoted else 'vcm-mvp-hey-iris-v3',
        source=str(source.relative_to(ROOT)),bundle=str(release.relative_to(ROOT)),
        bundle_sha256=digest(release/'bundle.json'),candidate_bundle_sha256=lock['bundle_sha256'],
        selection_before_test_sha256=digest(HERE/'selection-before-test.json'),
        threshold_retuned_after_test=False,strict_qualification_passed=False)
    write(HERE/'release-choice.json',choice)
    result=dict(experiment='iris-release-v1',selected=model,selection_before_test=lock,
        results=results,official=official,baseline=baseline,promotion=choice,
        release_results=chosen_results,release_official=chosen_official,strict_qualification_passed=False,
        original_human_personal_positive_n=0,physical_pi_measured=False,live_microphone_measured=False)
    write(HERE/'recognition-results.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','generate','prepare','train','select','export','evaluate'])
    p.add_argument('--model',choices=['compact','full'],default='compact')
    p.add_argument('--arm',choices=['ce','supcon'],default='ce');p.add_argument('--seed',type=int,default=53)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    result=train(args.model,args.arm,args.seed,args.device) if args.stage=='train' else globals()[args.stage]()
    print(json.dumps(result,default=str,allow_nan=False),flush=True)


if __name__=='__main__':main()
