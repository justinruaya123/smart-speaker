"""Pinned synthetic corpus and from-scratch Hey Iris fitting, isolated from human-only studies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from scripts.runtime.mvp_static import digest
from scripts.train.mvp_static_train import ROOT, HERE, CACHE, PREP, read_rows, write

WAKE=CACHE/'wake-v1'


def generation_seed(voice,index):
    return int(hashlib.sha256((voice+'/'+str(index)).encode()).hexdigest()[:8],16)


def save_waveform(path,pcm):
    import soundfile as sf
    pcm=np.asarray(pcm,dtype='float32')
    if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('Invalid generated PCM')
    sf.write(path,pcm,16000,subtype='PCM_16')
    saved,_=sf.read(path,dtype='float32')
    return dict(sha256=digest(path),pcm_sha256=hashlib.sha256(saved.tobytes()).hexdigest(),
                samples=len(saved),rms=float(np.sqrt(np.mean(saved**2))))


def generation_plan(voices, catalog):
    near=['Hey Siri.','Hey Irish.','Iris.','Hey.','Hey Jarvis.','Hey Alice.',
          'Hey Paris.','They hire us.','A virus.','Here it is.','Good morning.','Are you ready?']
    negative=['Play Mystery Planet.','Turn on the attic light.','Set brightness to 140 percent.',
              'Set a timer for minus five seconds.','Call a stranger.','Set a timer.']
    personal=[c for c in catalog if c['command'] and (c['command']['slots'].get('light_id')=='light.ceiling' or
        c['command']['slots'].get('song_id') in {'song.talk_that_talk','song.queen_of_hearts','song.basics','song.trouble',
             'song.brave','song.gone','song.when_we_were_kids','song.enemy'})]
    result=[]
    for voice in voices:
        cells=[('Hey Iris.',True,None,'wake') for _ in range(24 if voice['split']=='train' else 8)]
        cells += [(text,False,None,'wake_negative') for text in near]
        cells += [(text,False,None,'command') for text in negative]
        cells += [(c['wordings'][0],False,c['command'],'command') for c in personal]
        for index,(text,positive,command,role) in enumerate(cells):
            key=hashlib.sha256((voice['id']+'/'+str(index)+'/'+text).encode()).hexdigest()[:20]
            result.append(dict(id='iris1/'+key,voice_id=voice['id'],speaker=voice['id'],split=voice['split'],
                text=text,wake_positive=positive,command=command,role=role,pair_id='iris1/'+key))
    return result


def window(pcm,positive,rng):
    pcm=np.asarray(pcm,dtype='float32')
    if positive and len(pcm)>25600:raise ValueError('Wake phrase too long; refusing to truncate')
    if len(pcm)>25600:
        offset=int(rng.integers(len(pcm)-25600+1));return pcm[offset:offset+25600].copy()
    result=np.zeros(25600,dtype='float32');offset=int(rng.integers(25600-len(pcm)+1))
    result[offset:offset+len(pcm)]=pcm;return result


def freeze():
    path=HERE/'wake-protocol.json'
    if path.exists():return json.loads(path.read_text())
    voices=json.loads((ROOT/'experiments/speaker-synthetic-completion-v1/voices.json').read_text())
    new=[('train-in','train','A clear adult man with an Indian English accent and medium-low pitch, natural conversational delivery.'),
         ('train-scot','train','A clear adult woman with a mild Scottish English accent and medium pitch, natural conversational delivery.'),
         ('test-ph','test','A clear adult man speaking English with a Philippine English accent and medium pitch.'),
         ('test-us','test','A clear adult woman with a neutral American English accent and lower pitch.')]
    voices += [dict(id=i,split=s,prompt=p+' Clean dry speech, no music.',design_seed=71017+n,
                    identity_source='original_synthetic_voice_design',accent_verified=False) for n,(i,s,p) in enumerate(new)]
    catalog=json.loads((HERE/'catalog.json').read_text());plan=generation_plan(voices,catalog)
    write(HERE/'wake-voices.json',voices);write(HERE/'wake-generation-plan.json',plan)
    cfg=dict(experiment='hey-iris-v1',seed=61,initialization='all deployed wake parameters random; no pretrained embeddings',
        source_receipt='data/sources/qwen-synthetic-completion.source.json',
        source_sha256=digest(ROOT/'data/sources/qwen-synthetic-completion.source.json'),
        plan_sha256=digest(HERE/'wake-generation-plan.json'),voices_sha256=digest(HERE/'wake-voices.json'),
        normalization_sha256=digest(PREP/'normalization.json'),window_samples=25600,step_samples=1280,
        generation=dict(batch=4,temperature=.8,top_p=.95,max_new_tokens=160,cuda_memory_fraction=.35),
        training=dict(epochs=20,steps_per_epoch=96,batch=64,learning_rate=.001),
        augmentation=dict(time_position='Random full-phrase position inside 1.6 seconds; no positive truncation',
            gain_db=[-6,6],white_noise_snr_db=[20,35],probability=.5,
            citation='https://github.com/tensorflow/tensorflow/blob/v2.20.0/tensorflow/examples/speech_commands/input_data.py',
            note='Adaptation of time shift, volume and additive background noise; white noise here, no recorded room/RIR or speed perturbation'),
        real_positive_recordings=0,independent_listening=False,demographic_labels='requested synthetic conditions; not verified people',
        physical_pi_measured=False,qualified=False)
    write(path,cfg);return cfg


def generate():
    import torch
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel
    cfg=freeze();WAKE.mkdir(parents=True,exist_ok=True)
    source=json.loads((ROOT/cfg['source_receipt']).read_text())
    if digest(ROOT/cfg['source_receipt'])!=cfg['source_sha256']:raise ValueError('Changed TTS source receipt')
    for record in source['files']:
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('Changed TTS source model')
    voices=json.loads((HERE/'wake-voices.json').read_text());plan=json.loads((HERE/'wake-generation-plan.json').read_text())
    if digest(HERE/'wake-generation-plan.json')!=cfg['plan_sha256']:raise ValueError('Changed generation plan')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(cfg['generation']['cuda_memory_fraction'])
    anchors={r['id']:r for r in json.loads((ROOT/'.cache/speaker-synthetic-completion-v1/anchors.json').read_text())}
    def load(variant):
        return Qwen3TTSModel.from_pretrained(str(ROOT/'.cache/tts/models'/('Qwen3-TTS-12Hz-1.7B-'+variant)),
            device_map='cuda:0',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    missing=[v for v in voices if v['id'] not in anchors]
    if missing:
        design=load('VoiceDesign')
        for voice in missing:
            target=WAKE/'anchors'/f'{voice["id"]}.json'
            if target.exists():anchors[voice['id']]=json.loads(target.read_text());continue
            torch.manual_seed(voice['design_seed'])
            text='The room is quiet and I am speaking clearly at a comfortable pace.'
            waves,rate=design.generate_voice_design(text=text,language='English',instruct=voice['prompt'],
                                                   max_new_tokens=384,do_sample=True,temperature=.8)
            native=target.with_suffix('.wav');native.parent.mkdir(parents=True,exist_ok=True)
            sf.write(native,waves[0],rate,subtype='PCM_16')
            anchor={**voice,'native_audio':str(native.relative_to(ROOT)),'native_sha256':digest(native),'text':text}
            write(target,anchor);anchors[voice['id']]=anchor;print('ANCHOR',voice['id'],flush=True)
        del design;torch.cuda.empty_cache()
    from scripts.train.speaker_synthetic_data import ANCHOR_TEXT
    model=load('Base');clone={}
    for voice in voices:
        anchor=anchors[voice['id']];native=ROOT/anchor['native_audio']
        if digest(native)!=anchor['native_sha256']:raise ValueError('Changed voice anchor')
        clone[voice['id']]=model.create_voice_clone_prompt(ref_audio=str(native),ref_text=anchor.get('text',ANCHOR_TEXT))[0]
    manifest=WAKE/'generated.jsonl';existing=read_rows(manifest) if manifest.exists() else []
    for r in existing:
        if digest(ROOT/r['audio'])!=r['sha256']:raise ValueError('Changed generated waveform')
    done={r['id'] for r in existing};began=time.perf_counter()
    with manifest.open('a') as stream:
        for voice in voices:
            todo=[r for r in plan if r['voice_id']==voice['id'] and r['id'] not in done]
            for offset in range(0,len(todo),cfg['generation']['batch']):
                batch=todo[offset:offset+cfg['generation']['batch']]
                seed=generation_seed(voice['id'],offset);torch.manual_seed(seed)
                waves,rate=model.generate_voice_clone(text=[r['text'] for r in batch],language=['English']*len(batch),
                    voice_clone_prompt=[clone[voice['id']]]*len(batch),non_streaming_mode=True,do_sample=True,
                    temperature=.8,top_p=.95,max_new_tokens=160)
                if len(waves)!=len(batch):raise ValueError('TTS count mismatch')
                for row,wave in zip(batch,waves):
                    target=WAKE/'audio'/voice['id']/(row['id'].split('/')[-1]+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():raise FileExistsError('Unreceipted generation output: '+str(target))
                    pcm=resample_poly(np.asarray(wave,dtype='float32'),16000//math.gcd(16000,rate),rate//math.gcd(16000,rate)).astype('float32')
                    saved=save_waveform(target,pcm)
                    record={**row,'audio':str(target.relative_to(ROOT)),**saved,
                        'synthetic':True,'kind':'synthetic','source':'Qwen3-original-designed-iris-v1',
                        'anchor_sha256':anchors[voice['id']]['native_sha256'],'generation_seed':seed,
                        'listener_verified':False,'accent_verified':False}
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();done.add(row['id'])
                print('GENERATED',len(done),len(plan),round(time.perf_counter()-began,1),flush=True)
    write(HERE/'wake-generation-complete.json',dict(rows=len(done),manifest_sha256=digest(manifest),
        protocol_sha256=digest(HERE/'wake-protocol.json'),seconds=time.perf_counter()-began))


def fold_rows(split):
    generated=read_rows(WAKE/'generated.jsonl')
    rows=[r for r in generated if r['split']==split]
    original=read_rows(CACHE/'views'/f'{split}.jsonl')
    rng=np.random.default_rng(61)
    for kind in (False,True):
        candidates=[r for r in original if r['synthetic']==kind and 'iris' not in str(r.get('text','')).lower()]
        order=rng.permutation(len(candidates))[:(512 if split=='train' else 64)]
        rows += [{**candidates[i],'wake_positive':False} for i in order]
    return rows


def read_pcm(row):
    import soundfile as sf
    path=ROOT/row['audio']
    if digest(path)!=row['sha256']:raise ValueError('Wake audio identity mismatch')
    pcm,rate=sf.read(path,dtype='float32')
    if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Malformed wake PCM')
    return pcm


def training_arrays(rows,norm,*,training,hard=False):
    from vcm.frontend import features
    rng=np.random.default_rng(61);arrays=[];labels=[];excluded=[]
    for row in rows:
        pcm=read_pcm(row);positive=row['wake_positive']
        if positive:
            active=np.flatnonzero(np.abs(pcm)>max(.002,float(np.max(np.abs(pcm)))*.02))
            if not len(active):excluded.append({'id':row['id'],'reason':'silent_positive'});continue
            pcm=pcm[max(0,active[0]-1600):min(len(pcm),active[-1]+1601)]
            if len(pcm)>25600:excluded.append({'id':row['id'],'reason':'phrase_exceeds_window'});continue
        repeats=(12 if positive else 2) if training else (3 if positive else 1)
        if hard and training and row.get('role')=='wake_negative':repeats=24
        for _ in range(repeats):
            wav=window(pcm,positive,rng)
            if training:
                wav=wav*10**(rng.uniform(-6,6)/20)
                if rng.random()<.5:
                    rms=max(float(np.sqrt(np.mean(wav**2))),.0001)
                    wav=wav+rng.normal(0,rms/10**(rng.uniform(20,35)/20),len(wav)).astype('float32')
                wav=np.clip(wav,-1,1)
            arrays.append(features(wav,norm['mean'],norm['std']));labels.append(int(positive))
    return np.stack(arrays),np.asarray(labels,dtype='int64'),excluded


def train(variant='v1'):
    import torch
    from scripts.models.mvp_wake import WakeNet
    cfg=freeze();complete=json.loads((HERE/'wake-generation-complete.json').read_text())
    if digest(WAKE/'generated.jsonl')!=complete['manifest_sha256']:raise ValueError('Changed wake corpus')
    out=WAKE/('training' if variant=='v1' else 'training-balanced-v2');out.mkdir(exist_ok=False)
    if variant!='v1':
        write(HERE/'wake-balanced-v2-protocol.json',dict(parent_protocol_sha256=digest(HERE/'wake-protocol.json'),
            generated_manifest_sha256=complete['manifest_sha256'],seed=61,epochs=20,steps_per_epoch=96,
            changed_factor='Near-wake negative windows: 24 random full-phrase positions instead of 2; same architecture, original audio and loss',
            reason='Calibration streams exposed high near-wake scores; fixed-window development specificity did not transfer to sliding windows',
            no_wake_test_used=True,real_positive_recordings=0))
    norm=json.loads((PREP/'normalization.json').read_text())
    if digest(PREP/'normalization.json')!=cfg['normalization_sha256']:raise ValueError('Changed wake normalization')
    torch.set_num_threads(4);torch.manual_seed(61);torch.cuda.set_per_process_memory_fraction(.12)
    x,y,excluded=training_arrays(fold_rows('train'),norm,training=True,hard=variant!='v1')
    dx,dy,dev_excluded=training_arrays(fold_rows('dev'),norm,training=False)
    write(out/'data-audit.json',dict(training_windows=len(y),positive_windows=int(y.sum()),
        development_windows=len(dy),excluded_training=excluded,excluded_development=dev_excluded,
        normalization_sha256=cfg['normalization_sha256'],generated_manifest_sha256=complete['manifest_sha256'],
        derivative_parent_split_preserved=True,independent_human_positive_count=0))
    model=WakeNet().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(61);groups=[np.flatnonzero(y==label) for label in (0,1)]
    if any(len(group)==0 for group in groups):raise ValueError('Missing wake training class')
    history=[];best=None;began=time.perf_counter()
    for epoch in range(1,cfg['training']['epochs']+1):
        model.train();losses=[]
        for step in range(cfg['training']['steps_per_epoch']):
            ids=np.concatenate([rng.choice(group,32) for group in groups])
            batch=torch.from_numpy(x[ids]).cuda();target=torch.from_numpy(y[ids]).cuda()
            optimizer.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(batch),target)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        model.eval();scores=[]
        with torch.no_grad():
            for offset in range(0,len(dy),64):scores.extend(model(torch.from_numpy(dx[offset:offset+64]).cuda()).argmax(1).cpu().tolist())
        scores=np.asarray(scores);retention=float(np.mean(scores[dy==1]==1));specificity=float(np.mean(scores[dy==0]==0))
        key=(retention+specificity,specificity,retention)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),dev_recall=retention,dev_specificity=specificity,seconds=time.perf_counter()-began)
        history.append(row);print('WAKE',json.dumps(row),flush=True)
        if best is None or key>tuple(best['key']):
            torch.save(dict(weights=model.cpu().state_dict(),config=model.config,epoch=epoch),out/'best.pt');model.cuda()
            best=dict(epoch=epoch,key=list(key),sha256=digest(out/'best.pt'))
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    write(out/'selection.json',dict(**best,parameters=sum(p.numel() for p in model.parameters()),
        initialization='random',seed=61,training_seconds=time.perf_counter()-began,
        test_used_for_selection=False,real_wake_positive_recordings=0))


def streaming_scores(rows,model,norm):
    import torch
    from vcm.frontend import features
    results=[]
    for index,row in enumerate(rows,1):
        pcm=read_pcm(row);padded=np.concatenate((np.zeros(32000,dtype='float32'),pcm,np.zeros(48000,dtype='float32')))
        windows=[features(padded[end-25600:end],norm['mean'],norm['std']) for end in range(25600,len(padded)+1,1280)]
        values=[]
        with torch.no_grad():
            for offset in range(0,len(windows),128):
                logits=model(torch.from_numpy(np.stack(windows[offset:offset+128]))).cpu().numpy().astype('float64')
                p=np.exp(logits-logits.max(1,keepdims=True));values.extend((p[:,1]/p.sum(1)).tolist())
        results.append(dict(id=row['id'],positive=row['wake_positive'],synthetic=row['synthetic'],
            speaker=row.get('speaker',row.get('speaker_id')),seconds=len(pcm)/16000,scores=values))
        if index%50==0:print('WAKE_STREAM',index,len(rows),flush=True)
    return results


def stream_metrics(records,threshold):
    from scripts.runtime.mvp_wake import RollingWake
    positives=sum(r['positive'] for r in records);negative_seconds=sum(r['seconds'] for r in records if not r['positive'])
    correct=false=duplicates=0
    for row in records:
        values=iter(row['scores']);stream=RollingWake(lambda pcm:next(values),threshold=threshold)
        stream.push(np.zeros(24320,dtype='float32'))
        hits=sum(stream.push(np.zeros(1280,dtype='float32'))==1. for _ in row['scores'])
        if row['positive']:correct+=int(hits>0);duplicates+=max(0,hits-1)
        else:false+=hits
    return dict(positive_n=positives,detected=correct,missed=positives-correct,duplicate_detections=duplicates,
        negative_n=len(records)-positives,negative_seconds=negative_seconds,false_wakes=false,
        recall=correct/positives if positives else None,false_wakes_per_hour=false*3600/negative_seconds if negative_seconds else None,
        negative_exposure='separate clips; resets and synthetic silence padding excluded from exposure; not household continuous audio')


def export_engines(model,out):
    import torch
    import onnxruntime as ort
    import ncnn
    import pnnx
    out=Path(out);model=model.cpu().eval();torch.set_num_threads(1);torch.manual_seed(61)
    torch.onnx.export(model,(torch.randn(1,40,158),),str(out/'iris.onnx'),input_names=['features'],output_names=['wake'],opset_version=17,dynamo=False)
    pnnx.export(model,str(out/'iris.pt'),inputs=torch.randn(1,40,158),fp16=False)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(str(out/'iris.onnx'),options,providers=['CPUExecutionProvider'])
    net=ncnn.Net();net.opt.num_threads=1;net.opt.use_vulkan_compute=False
    net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    net.opt.use_bf16_packed=False;net.opt.use_bf16_storage=False;net.opt.use_packing_layout=False
    if net.load_param(str(out/'iris.ncnn.param')) or net.load_model(str(out/'iris.ncnn.bin')):raise ValueError('Wake ncnn load failed')
    errors=[]
    for _ in range(5):
        x=torch.randn(1,40,158)
        with torch.no_grad():expected=model(x).numpy().reshape(-1)
        actual=session.run(None,{'features':x.numpy()})[0].reshape(-1)
        data=np.ascontiguousarray(x[0].numpy());mat=ncnn.Mat(data).clone()
        with net.create_extractor() as ex:
            if ex.input('in0',mat):raise ValueError('Wake ncnn input failed')
            code,value=ex.extract('out0')
            if code:raise ValueError('Wake ncnn extraction failed')
            nc=np.array(value,copy=True).reshape(-1)
        errors.append(max(float(np.abs(actual-expected).max()),float(np.abs(nc-expected).max())))
    if max(errors)>2e-5:raise ValueError('Wake engine numerical parity failed: '+str(errors))
    return dict(max_absolute_error=max(errors),files={f:digest(out/f) for f in ('iris.onnx','iris.ncnn.param','iris.ncnn.bin')})


def export_and_calibrate(variant='v1'):
    import torch
    from scripts.models.mvp_wake import WakeNet
    training=WAKE/('training' if variant=='v1' else 'training-balanced-v2')
    selection=json.loads((training/'selection.json').read_text());checkpoint=training/'best.pt'
    if digest(checkpoint)!=selection['sha256']:raise ValueError('Changed selected wake weights')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True);model=WakeNet(**state['config']);model.load_state_dict(state['weights']);model.eval()
    norm=json.loads((PREP/'normalization.json').read_text());out=WAKE/('export' if variant=='v1' else 'export-balanced-v2');out.mkdir(exist_ok=False)
    engines=export_engines(model,out)
    records=streaming_scores(fold_rows('calibration'),model,norm);write(out/'calibration-stream-scores.json',records)
    best=None
    for threshold in (.5,.6,.7,.8,.85,.9,.95,.98,.995,.999,.9999,1.):
        metric=stream_metrics(records,threshold)
        if metric['false_wakes'] or metric['duplicate_detections']:continue
        key=(metric['detected'],-threshold)
        if best is None or key>best[0]:best=(key,threshold,metric)
    if best is None:raise RuntimeError('No negative-constrained wake policy; preserve failed export')
    write(out/'iris.json',dict(files=engines['files'],normalization=norm,
        threshold=best[1],release=.3,consecutive=2,cooldown_samples=16000,qualified=False,
        initialization='random',parameters=selection['parameters'],real_wake_positive_recordings=0,
        calibration=best[2],engine_parity_max_error=engines['max_absolute_error']))
    write(HERE/('wake-selection-before-test.json' if variant=='v1' else 'wake-selection-before-test-balanced-v2.json'),dict(checkpoint_sha256=digest(checkpoint),
        model_sha256=digest(out/'iris.onnx'),policy_sha256=digest(out/'iris.json'),
        manifest_sha256=digest(WAKE/'generated.jsonl'),test_used_for_selection=False))
    print('WAKE_CALIBRATED',json.dumps(best[2]),flush=True)


def evaluate(variant='v1'):
    import torch
    from scripts.models.mvp_wake import WakeNet
    lock=json.loads((HERE/('wake-selection-before-test.json' if variant=='v1' else 'wake-selection-before-test-balanced-v2.json')).read_text())
    export=WAKE/('export' if variant=='v1' else 'export-balanced-v2')
    training=WAKE/('training' if variant=='v1' else 'training-balanced-v2')
    if digest(export/'iris.json')!=lock['policy_sha256'] or digest(WAKE/'generated.jsonl')!=lock['manifest_sha256']:
        raise ValueError('Changed locked wake inputs')
    state=torch.load(training/'best.pt',map_location='cpu',weights_only=True)
    if digest(training/'best.pt')!=lock['checkpoint_sha256']:raise ValueError('Changed wake checkpoint')
    model=WakeNet(**state['config']);model.load_state_dict(state['weights']);model.eval();torch.set_num_threads(1)
    meta=json.loads((export/'iris.json').read_text());records=streaming_scores(fold_rows('test'),model,meta['normalization'])
    write(export/'test-stream-scores.json',records);metric=stream_metrics(records,meta['threshold'])
    write(HERE/('wake-results.json' if variant=='v1' else 'wake-results-balanced-v2.json'),dict(synthetic_and_recorded_negative_test=metric,qualified=False,
        real_positive_n=0,owner_backend_trained=False,physical_pi_measured=False))
    print('WAKE_TEST',json.dumps(metric),flush=True)


def prepare_augmented():
    """Add immutable functional views; never edit original corpus or test membership."""
    from collections import Counter
    import copy
    import soundfile as sf
    from scripts.runtime.pcv_schema import baseline_cases,canonical
    plan=json.loads((HERE/'augmentation-v2-plan.json').read_text())
    complete=json.loads((HERE/'wake-generation-complete.json').read_text())
    if digest(WAKE/'generated.jsonl')!=complete['manifest_sha256']:raise ValueError('Changed completion corpus')
    here=ROOT/'experiments/vcm-mvp-hey-iris-v2';cache=ROOT/'.cache/vcm-mvp-hey-iris-v2'
    here.mkdir(exist_ok=False);cache.mkdir(exist_ok=False)
    catalog=json.loads((HERE/'catalog.json').read_text());lookup={canonical(c['command']):c['command_id'] for c in catalog}
    additions=[];excluded=[]
    for row in read_rows(WAKE/'generated.jsonl'):
        pcm=read_pcm(row);peak=float(np.max(np.abs(pcm),initial=0));rms=float(np.sqrt(np.mean(pcm**2)))
        if not .025<=len(pcm)/16000<=15 or rms<.002 or np.mean(np.abs(pcm)>=.999)>.01:
            excluded.append(dict(id=row['id'],reason='signal_quality_or_duration',rms=rms,peak=peak));continue
        expected=row['command'];additions.append({**row,'class_index':lookup[canonical(expected)],
            'expected':expected,'speaker':'iris-voice/'+row['voice_id'],'qa_eligible':True,'metric_eligible':True,
            'reference_status':'supported' if expected else 'unsupported','slot_metadata_conflict':False,
            'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest()})
    views={s:read_rows(CACHE/'views'/f'{s}.jsonl') for s in ('train','dev','calibration','test','holdout')}
    for split in ('train','dev','calibration'):views[split]+= [r for r in additions if r['split']==split]
    views['personal-test']=[r for r in additions if r['split']=='test']
    bindings={}
    for split,rows in views.items():
        path=cache/'views'/f'{split}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(canonical(r)+'\n' for r in rows));bindings[str(path.relative_to(ROOT))]=digest(path)
    overlaps={}
    for i,a in enumerate(views):
        for b in list(views)[i+1:]:
            cells={f:len({r.get(f) for r in views[a] if r.get(f)} & {r.get(f) for r in views[b] if r.get(f)})
                   for f in ('id','speaker','pcm_sha256','pair_id')}
            if any(cells.values()):raise ValueError('Augmented cross-split overlap: '+a+'/'+b+str(cells))
            overlaps[a+'/'+b]=cells
    counts=Counter(r['class_index'] for r in views['train'])
    for c in catalog:c['training_recordings']=counts[c['command_id']]
    write(here/'catalog.json',catalog)
    cfg=copy.deepcopy(json.loads((HERE/'protocol.json').read_text()))
    checkpoints={}
    for name in ('compact','full'):
        parent=CACHE/'runs/longer-v2-20261017'/name
        selected=json.loads((parent/'selection.json').read_text());checkpoint=parent/'best.pt'
        if digest(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Changed parent direct checkpoint')
        checkpoints[name]=dict(path=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint))
    priority={lookup[canonical(c['command'])] for c in baseline_cases()}
    priority.update(r['class_index'] for r in additions if r['expected'])
    cfg.update(experiment='vcm-mvp-hey-iris-v2',parent_protocol_sha256=digest(HERE/'protocol.json'),
        catalog_sha256=digest(here/'catalog.json'),checkpoints=checkpoints,**plan['training'],
        priority_classes=sorted(priority),demo_risk=plan['demo_risk'],
        initialization=plan['initialization'],calibration='Unqualified practical demo policy; zero fitting UNKNOWN accepts, <=3% human and <=1% synthetic known errors per predicted family',
        inputs={**cfg['inputs'],**bindings,str((WAKE/'generated.jsonl').relative_to(ROOT)):complete['manifest_sha256'],
                str((HERE/'augmentation-v2-plan.json').relative_to(ROOT)):digest(HERE/'augmentation-v2-plan.json')})
    write(here/'protocol.json',cfg)
    write(here/'data-audit.json',dict(rows={s:len(rows) for s,rows in views.items()},
        additions={s:sum(r['split']==s for r in additions) for s in ('train','dev','calibration','test')},
        excluded_synthetic=excluded,cross_split_overlap=overlaps,human_positive_additions=0,
        original_test_and_holdout_unchanged=True,independent_listening=False,
        actual_saved_pcm_hashes_recomputed=True,augmentation_plan_sha256=digest(HERE/'augmentation-v2-plan.json')))
    print('AUGMENTED_PREPARED',json.dumps({s:len(rows) for s,rows in views.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','generate','train','export','evaluate','augment'])
    parser.add_argument('--variant',choices=['v1','balanced-v2'],default='v1')
    args=parser.parse_args()
    if args.stage in ('train','export','evaluate'):{'train':train,'export':export_and_calibrate,'evaluate':evaluate}[args.stage](args.variant)
    else:{'freeze':freeze,'generate':generate,'augment':prepare_augmented}[args.stage]()


if __name__=='__main__':main()
