"""Development selection, historical test comparison, and file/microphone entry."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import json
import time
import numpy as np
from einops import pack, rearrange
from .keyword_data import ROOT,DATA,CACHE,REPORTS,VOCAB,HEADS,FRONTEND,rows,dump,digest,feature_list
from .keyword_runtime import decode_record,finite_slots,KeywordRecognizer,KeywordPipeline,softmax
from .audio import features,load_audio

FAMILIES=HEADS['intent']
GROUPS=[c for c in FAMILIES if c not in ('UNKNOWN','BRIGHTNESS','VOLUME')]+[c+'/'+op for c in ('BRIGHTNESS','VOLUME') for op in ('SET','INCREASE','DECREASE')]


def engine(mode):
    import onnxruntime as ort
    option=ort.SessionOptions();option.intra_op_num_threads=2;option.inter_op_num_threads=1
    return ort.InferenceSession(str(CACHE/mode/'model.onnx'),option,providers=['CPUExecutionProvider'])


def infer(mode,data,tag):
    cache=CACHE/mode/(tag+'.npz');receipt=cache.with_suffix('.json')
    signature={'model_sha256':digest(CACHE/mode/'model.onnx'),'ids':[r['id'] for r in data],'audio':[r['audio_sha256'] for r in data]}
    if cache.exists() and receipt.exists() and json.loads(receipt.read_text())==signature:
        with np.load(cache) as f:return {k:f[k] for k in f.files}
    session=engine(mode);arrays=feature_list(data);intent=[];ready=[];keywords=[];lengths=[]
    for offset in range(0,len(data),16):
        chunk=arrays[offset:offset+16];lens=np.array([x.shape[1] for x in chunk],dtype=np.int64)
        x=np.zeros((len(chunk),80,int(max(lens))),dtype=np.float32)
        for i,a in enumerate(chunk):x[i,:,:a.shape[1]]=a
        a,b,c,d=session.run(None,{'features':x,'lengths':lens});intent.append(a);ready.append(b)
        for i,l in enumerate(d):keywords.append(c[i,:int(l)]);lengths.append(int(l))
    result={'intent':pack(intent,'* c')[0],'ready':pack(ready,'* c')[0],'keywords':pack(keywords,'* c')[0],'lengths':np.array(lengths,dtype=np.int64)}
    np.savez_compressed(cache,**result);dump(receipt,signature);print('CPU INFERENCE',mode,tag,len(data),flush=True)
    return result


def decoded(raw,evidence,thresholds=None):
    result=[];offset=0
    for i,l in enumerate(raw['lengths']):
        result.append(decode_record(raw['intent'][i],raw['ready'][i],raw['keywords'][offset:offset+l],int(l),evidence,thresholds));offset+=l
    return result


def intent_inference(data,tag,kind='micro'):
    graph=ROOT/'.cache/keyword-v2/mixed/model.onnx' if kind=='cnn' else CACHE/'micro-intent/model.onnx'
    path=CACHE/('cnn-intent' if kind=='cnn' else 'micro-intent')/(tag+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if kind=='cnn':
        previous=json.loads((ROOT/'data/keyword-v2/protocol.json').read_text())
        if previous['frontend']!=asdict(FRONTEND) or previous['heads']!=HEADS or previous['vocabulary']!=VOCAB:
            raise ValueError('Earlier intent candidate is incompatible with the current frontend or labels')
    signature={'graph':digest(graph),'ids':[r['id'] for r in data],'audio':[r['audio_sha256'] for r in data]}
    if path.exists() and path.with_suffix('.json').exists() and json.loads(path.with_suffix('.json').read_text())==signature:
        with np.load(path) as f:return f['intent'],f['ready']
    if kind=='cnn':
        import onnxruntime as ort
        option=ort.SessionOptions();option.intra_op_num_threads=2;option.inter_op_num_threads=1
        session=ort.InferenceSession(str(graph),option,providers=['CPUExecutionProvider'])
    else:session=engine('micro-intent')
    arrays=feature_list(data);intent=[];ready=[]
    for start in range(0,len(data),16):
        part=arrays[start:start+16];lengths=np.array([x.shape[1] for x in part],dtype=np.int64);x=np.zeros((len(part),80,int(max(lengths))),dtype=np.float32)
        for i,a in enumerate(part):x[i,:,:a.shape[1]]=a
        outputs=session.run(None,{'features':x,'lengths':lengths});intent.append(outputs[0]);ready.append(outputs[1])
    a,b=pack(intent,'* c')[0],pack(ready,'* c')[0];np.savez_compressed(path,intent=a,ready=b);dump(path.with_suffix('.json'),signature)
    return a,b


def group(label):
    if not label:return 'UNKNOWN'
    c=label['intent'];return c+'/'+label['operation'] if c in ('BRIGHTNESS','VOLUME') else c


def exact(prediction,row):
    return row['labels']['ready']=='READY' and prediction['valid'] and prediction['labels']==row['labels']


def choose_thresholds(predictions,data,stage,intent_thresholds=None):
    """Exact threshold candidates plus a global negative-error budget; no class-score mutation."""
    keys=FAMILIES[1:] if stage=='intent' else GROUPS
    negative=np.array([r['labels']['intent']=='UNKNOWN' if stage=='intent' else r['labels']['ready']!='READY' for r in data])
    budget=int(.01*int(negative.sum()));options={};denominators={}
    for key in keys:
        if stage=='intent':
            eligible=[i for i,o in enumerate(predictions) if o['family']==key]
            truth=lambda i:data[i]['labels']['intent']==key
            score=lambda i:predictions[i]['family_confidence']
            positives=sum(r['labels']['intent']==key for r in data)
        else:
            eligible=[i for i,o in enumerate(predictions) if o['valid'] and group(o['labels'])==key and o['family_confidence']>=intent_thresholds[o['family']]]
            truth=lambda i:exact(predictions[i],data[i])
            score=lambda i:predictions[i]['score']
            positives=sum(r['labels']['ready']=='READY' and group(r['labels'])==key for r in data)
        denominators[key]=positives
        order=sorted(eligible,key=score,reverse=True);candidates={0:{'threshold':1.1,'accepted':0,'correct':0,'false_accepts':0,'precision':None,'utility':0.}}
        correct=0;fp=0
        for j,i in enumerate(order):
            correct+=int(truth(i));fp+=int(negative[i]);n=j+1
            if j+1<len(order) and score(order[j+1])==score(i):continue
            if n<10 or correct/n<.95 or fp>budget:continue
            threshold=(score(i)+score(order[j+1]))/2 if j+1<len(order) else max(0.,score(i)-1e-5)
            value={'threshold':float(threshold),'accepted':n,'correct':correct,'false_accepts':fp,'precision':correct/n,'utility':correct/max(positives,1)}
            previous=candidates.get(fp)
            if previous is None or (value['correct'],value['precision'])>(previous['correct'],previous['precision'] or 0.):candidates[fp]=value
        options[key]=list(candidates.values())
    states={0:(0.,{})}
    for key in keys:
        newer={}
        for spent,(utility,selection) in states.items():
            for candidate in options[key]:
                total=spent+candidate['false_accepts']
                if total>budget:continue
                value=utility+candidate['utility']
                if total not in newer or value>newer[total][0]:newer[total]=(value,{**selection,key:candidate})
        states=newer
    spent,(utility,selected)=max(states.items(),key=lambda item:(item[1][0],-item[0]))
    thresholds={k:v['threshold'] for k,v in selected.items()};thresholds['UNKNOWN']=1.1
    return thresholds,{'groups':selected,'positive_support':denominators,'negative_n':int(negative.sum()),'false_accept_budget':budget,'selected_false_accepts':spent,'macro_utility_sum':utility}


def accepted(o,policy):
    return bool(o['valid'] and o['family']!='UNKNOWN' and o['family_confidence']>=policy['intent_thresholds'][o['family']] and o['score']>=policy['thresholds'].get(group(o['labels']),1.1))


def metrics(predictions,data,policy=None):
    take=np.array([accepted(o,policy) if policy else o['valid'] for o in predictions],dtype=bool)
    correct=np.array([exact(o,r) for o,r in zip(predictions,data)],dtype=bool)
    intent_take=np.array([o['family']!='UNKNOWN' and (not policy or o['family_confidence']>=policy['intent_thresholds'][o['family']]) for o in predictions],dtype=bool)
    positive=np.array([r['labels']['ready']=='READY' for r in data]);families=np.array([r['labels']['intent'] for r in data]);guesses=np.array([o['family'] for o in predictions])
    def divide(a,b):return float(a/b) if b else None
    overall={'n':len(data),'positive_n':int(positive.sum()),'accepted':int(take.sum()),'correct_accepted':int((take&correct).sum()),
        'precision':divide((take&correct).sum(),take.sum()),'exact_recall':divide((take&correct).sum(),positive.sum()),'negative_n':int((~positive).sum()),
        'false_accepts':int((take&~positive).sum()),'negative_false_accept_rate':divide((take&~positive).sum(),(~positive).sum()),'family_accuracy':float(np.mean(families==guesses))}
    per_intent={}
    for c in FAMILIES:
        true=families==c;emitted=take&(guesses==c);family_emit=intent_take&(guesses==c)
        per_intent[c]={'n':int(true.sum()),'positive_n':int((true&positive).sum()),'family_correct':int((true&(guesses==c)).sum()),
            'family_recall':divide((true&(guesses==c)).sum(),true.sum()),'accepted_intent_recall':divide((true&family_emit).sum(),true.sum()),
            'accepted_intent_precision':divide((true&family_emit).sum(),family_emit.sum()),'accepted_outputs':int(emitted.sum()),'correct_commands':int((emitted&correct).sum()),
            'command_precision':divide((emitted&correct).sum(),emitted.sum()),'command_recall':divide((emitted&correct).sum(),(true&positive).sum()),
            'negative_false_accepts':int((true&~positive&take).sum()),'rejection_rate':divide((true&~take).sum(),true.sum())}
    operations={}
    for key in GROUPS:
        if '/' not in key:continue
        true=np.array([r['labels']['ready']=='READY' and group(r['labels'])==key for r in data]);emitted=take&np.array([group(o['labels'])==key for o in predictions])
        operations[key]={'positive_n':int(true.sum()),'correct':int((true&correct&take).sum()),'recall':divide((true&correct&take).sum(),true.sum()),'accepted_outputs':int(emitted.sum()),'precision':divide((emitted&correct).sum(),emitted.sum())}
    return {'overall':overall,'per_intent':per_intent,'operations':operations}


def calibrate(mode):
    out=REPORTS[mode];lock=out/'selection-before-test.json'
    if lock.exists():raise ValueError('Selection already frozen')
    specification={'version':'keyword-v4-selection','data':'human development only','evidence_candidates':[0.,.75,1.5],
        'thresholds':'Exact per-intent intent thresholds, then per-family/per-numeric-operation command thresholds. Dynamic programming maximizes summed group recall subject to 95% precision, ten accepts per enabled group and a shared 1% negative false-accept budget.',
        'candidate_choice':'Highest macro command recall, then overall exact recall. Zero cue bias remains an incumbent option. Test is not used.',
        'numeric_support':'SET, INCREASE, DECREASE gated separately for each family; a group with fewer than ten supported development accepts abstains.',
        'intent_candidates':['shared keyword model','earlier 1D CNN candidate with zero cue bias','micro Conv2D with zero cue bias'] if mode=='mixed' else ['shared keyword model'],
        'parameter_decoder':'Ordered fixed-word CTC with exact forward scoring of 0-100 grammar and a missing-number competitor. No downstream family change.'}
    dump(out/'selection-specification.json',specification)
    data=[r for r in rows(DATA/'human.jsonl') if r['split']=='dev'];raw=infer(mode,data,'human-dev');candidates=[]
    variants=[('shared',raw,e) for e in specification['evidence_candidates']]
    if mode=='mixed':
        for name in ('cnn','micro'):
            a,b=intent_inference(data,'human-dev',name);variants.append((name,{**raw,'intent':a,'ready':b},0.))
    for intent_model,values,evidence in variants:
        guess=decoded(values,evidence);ith,igates=choose_thresholds(guess,data,'intent');thresholds,gates=choose_thresholds(guess,data,'command',ith)
        policy={'evidence':evidence,'intent_model':intent_model,'intent_thresholds':ith,'thresholds':thresholds}
        result=metrics(guess,data,policy)
        macro=float(np.mean([m['command_recall'] for c,m in result['per_intent'].items() if c!='UNKNOWN' and m['command_recall'] is not None]))
        candidates.append({'policy':policy,'intent_gates':igates,'command_gates':gates,'metrics':result,'macro_command_recall':macro})
    chosen=max(candidates,key=lambda x:(x['macro_command_recall'],x['metrics']['overall']['exact_recall']))
    sources=[ROOT/'scripts'/folder/name for folder,name in [('models','keyword_vcm.py'),('models','keyword_intent.py'),('models','vcm.py'),('train','keyword_data.py'),('train','keyword_training.py'),('train','synthetic_data.py'),('runtime','keyword_runtime.py'),('test','keyword_eval.py'),('runtime','audio.py'),('runtime','levels.py'),('runtime','routing.py'),('runtime','pipeline.py')]]
    frozen={'created_utc':datetime.now(timezone.utc).isoformat(),'policy':chosen['policy'],'vocabulary':VOCAB,'protocol_sha256':digest(DATA/'protocol.json'),
        'model_sha256':digest(CACHE/mode/'model.onnx'),'implementation_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'selection_specification_sha256':digest(out/'selection-specification.json'),'development':chosen['metrics'],'intent_gates':chosen['intent_gates'],'command_gates':chosen['command_gates'],
        'test_used_for_this_selection':False,'historical_test_previously_reported':True}
    if chosen['policy']['intent_model'] in ('micro','cnn'):
        ipath=ROOT/'.cache/keyword-v2/mixed/model.onnx' if chosen['policy']['intent_model']=='cnn' else CACHE/'micro-intent/model.onnx'
        frozen['intent_model_sha256']=digest(ipath);frozen['intent_model_path']=ipath.relative_to(ROOT).as_posix()
    dump(out/'development-search.json',candidates);dump(lock,frozen)
    values=next(v for name,v,e in variants if name==chosen['policy']['intent_model'] and e==chosen['policy']['evidence'])
    dump(out/'development-diagnostics.json',[{'id':r['id'],'transcript':r['transcript'],'reference':r['labels'],**o} for r,o in zip(data,decoded(values,chosen['policy']['evidence']))])
    print('SELECTED',mode,json.dumps(chosen['metrics']['overall']),flush=True)


def evaluate(mode):
    out=REPORTS[mode]
    if (out/'results.json').exists():raise ValueError('Historical tests already scored for this version')
    selection=json.loads((out/'selection-before-test.json').read_text());policy=selection['policy']
    for name,h in selection['implementation_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen implementation changed: '+name)
    if digest(CACHE/mode/'model.onnx')!=selection['model_sha256']:raise ValueError('Frozen graph changed')
    if 'intent_model_path' in selection and digest(ROOT/selection['intent_model_path'])!=selection['intent_model_sha256']:
        raise ValueError('Frozen intent graph changed')
    if digest(DATA/'protocol.json')!=selection['protocol_sha256']:raise ValueError('Frozen data protocol changed')
    if digest(out/'selection-specification.json')!=selection['selection_specification_sha256']:raise ValueError('Frozen selection specification changed')
    result={'mode':mode,'tests':{},'historical_comparison_not_fresh_holdout':True,'new_words_tested':False,'live_microphone_tested':False,'physical_pi_tested':False}
    for name in ('human','synthetic'):
        data=[r for r in rows(DATA/(name+'.jsonl')) if r['split']=='test'];raw=infer(mode,data,name+'-test')
        if policy.get('intent_model') in ('micro','cnn'):
            a,b=intent_inference(data,name+'-test',policy['intent_model']);raw={**raw,'intent':a,'ready':b}
        guess=decoded(raw,policy['evidence']);gated=decoded(raw,policy['evidence'],policy['intent_thresholds'])
        result['tests'][name]={'gated':metrics(gated,data,policy),'before_confidence_gates':metrics(guess,data)}
        dump(CACHE/mode/(name+'-test-predictions.json'),[{'id':r['id'],'reference':r['labels'],'accepted':accepted(o,policy),'correct':exact(o,r),**o} for r,o in zip(data,gated)])
    dump(out/'results.json',result);print('RESULTS',mode,json.dumps({s:v['gated']['overall'] for s,v in result['tests'].items()}),flush=True)


def check(mode):
    import torch
    from .keyword_vcm import KeywordVCM
    torch.set_num_threads(2)
    saved=torch.load(CACHE/mode/'best.pt',map_location='cpu',weights_only=True);model=KeywordVCM(saved['width']).eval();model.load_state_dict(saved['state'])
    session=engine(mode);errors=[]
    for b,t in ((1,43),(1,401),(3,237)):
        x=torch.randn(b,80,t);lens=torch.tensor([t-17*i for i in range(b)])
        with torch.inference_mode():expected=model(x,lens)
        actual=session.run(None,{'features':x.numpy(),'lengths':lens.numpy()})
        for a,e in zip(actual,expected):np.testing.assert_allclose(a,e.numpy(),atol=3e-4,rtol=3e-4);errors.append(float(np.max(np.abs(a-e.numpy()))))
        # Padding must not change valid predictions.
        padded=torch.nn.functional.pad(x,(0,37))
        extra=session.run(None,{'features':padded.numpy(),'lengths':lens.numpy()})
        for i in (0,1):np.testing.assert_allclose(extra[i],actual[i],atol=3e-4,rtol=3e-4)
    if mode=='mixed':
        from .keyword_intent import MicroIntent
        weights=torch.load(CACHE/'micro-intent/best.pt',map_location='cpu',weights_only=True);micro=MicroIntent().eval();micro.load_state_dict(weights['state']);ms=engine('micro-intent')
        for b,t in ((1,39),(3,231)):
            x=torch.randn(b,80,t);lengths=torch.tensor([t-i*17 for i in range(b)])
            with torch.inference_mode():e=micro(x,lengths)
            a=ms.run(None,{'features':x.numpy(),'lengths':lengths.numpy()})
            for av,ev in zip(a,e):np.testing.assert_allclose(av,ev.numpy(),atol=3e-4,rtol=3e-4)
        snapshot=json.loads((ROOT/'.cache/keyword-v2/source-snapshot.json').read_text())
        namespace={'__name__':'tinyvcm.keyword_v2_verification','__package__':'tinyvcm'}
        exec(compile(snapshot['scripts/models/keyword_vcm.py']['text'],'saved_keyword_v2_model','exec'),namespace)
        weights=torch.load(ROOT/'.cache/keyword-v2/mixed/best.pt',map_location='cpu',weights_only=True)
        cnn=namespace['KeywordVCM'](weights['width']).eval();cnn.load_state_dict(weights['state'])
        import onnxruntime as ort
        option=ort.SessionOptions();option.intra_op_num_threads=2
        cs=ort.InferenceSession(str(ROOT/'.cache/keyword-v2/mixed/model.onnx'),option,providers=['CPUExecutionProvider'])
        for b,t in ((1,41),(3,227)):
            x=torch.randn(b,80,t);lengths=torch.tensor([t-i*17 for i in range(b)])
            with torch.inference_mode():e=cnn(x,lengths)
            a=cs.run(None,{'features':x.numpy(),'lengths':lengths.numpy()})
            for av,ev in zip(a,e):np.testing.assert_allclose(av,ev.numpy(),atol=3e-4,rtol=3e-4)
    def cue(text):return [{'word':w,'confidence':1.,'start_ms':i*40.,'end_ms':(i+1)*40.} for i,w in enumerate(text.split())]
    from .levels import spoken_number
    for c in ('BRIGHTNESS','VOLUME'):
        for n in range(101):
            for suffix in ('',' percent'):
                labels,_,_=finite_slots(c,cue('set '+c.lower()+' to '+spoken_number(n)+suffix));assert labels['amount']==str(n) and labels['operation']=='SET'
        for direction,op in [('up','INCREASE'),('down','DECREASE')]:
            labels,_,_=finite_slots(c,cue(c.lower()+' '+direction));assert labels['operation']==op and labels['amount']=='10'
        for text in ('fifty brightness set','set '+c.lower()+' to one hundred and one percent',c.lower()+' up and down','set '+c.lower()+' to fifty or sixty','set '+c.lower()+' to fifty point five'):
            try:finite_slots(c,cue(text))
            except ValueError:pass
            else:raise AssertionError(text)
    from .keyword_runtime import ctc_cues,numeric_cues
    # A word whose per-frame probability is below blank can still have strong CTC sequence evidence.
    logits=np.full((12,len(VOCAB)),-12.,dtype=np.float32);logits[:,0]=4.
    for i,w in [(0,'set'),(2,'brightness'),(4,'to'),(11,'percent')]:logits[i,VOCAB.index(w)]=15.
    for i in range(5,11):logits[i,VOCAB.index('fifty')]=3.9
    observed=ctc_cues(logits,12);assert 'fifty' not in [x['word'] for x in observed]
    revised,trace=numeric_cues(logits,12,observed,'BRIGHTNESS');assert trace['value']==50
    parsed,_,_=finite_slots('BRIGHTNESS',revised);assert parsed['amount']=='50'
    empty=logits.copy();empty[:,VOCAB.index('fifty')]=-12
    _,trace=numeric_cues(empty,12,ctc_cues(empty,12),'BRIGHTNESS');assert trace['null_won']
    # Verify a rejected VCM never invokes semantic parameter parsing.
    from . import keyword_runtime as runtime
    original=runtime.finite_slots
    def forbidden(*args):raise AssertionError('Reader invoked after rejected intent')
    runtime.finite_slots=forbidden
    try:
        fi=np.zeros(len(FAMILIES));fi[FAMILIES.index('BRIGHTNESS')]=20.
        rejected=runtime.decode_record(fi,np.array([0.,20.]),logits,12,0.,{c:1.1 for c in FAMILIES});assert not rejected['valid']
    finally:runtime.finite_slots=original
    protocol=json.loads((DATA/'protocol.json').read_text());human=rows(DATA/'human.jsonl');synth=rows(DATA/'synthetic.jsonl');aux=rows(DATA/'words.jsonl')+rows(DATA/'context.jsonl')
    sets={s:{r['speaker_id'] for r in human if r['split']==s} for s in ('train','dev','test')}
    assert not (sets['train']&sets['test'] or sets['train']&sets['dev'] or sets['dev']&sets['test'])
    assert not ({r['speaker_id'] for r in aux}&(sets['dev']|sets['test']))
    hashes={s:{r['audio_sha256'] for r in human if r['split']==s} for s in sets};aux_hash={r['audio_sha256'] for r in aux}
    assert not (aux_hash&(hashes['dev']|hashes['test']))
    for name,h in protocol['manifests'].items():assert digest(DATA/name)==h
    for r in human+synth+aux:assert digest(ROOT/r['audio'])==r['audio_sha256'],r['id']
    for name,h in protocol['installed_before'].items():assert digest(ROOT/name)==h,name
    for name,h in protocol['prior_results_before'].items():assert digest(ROOT/name)==h,name
    result={'onnx_max_abs_error':max(errors),'variable_length_batch_and_padding_parity':True,'absolute_parser_cases':404,'parser_cases_are_not_spoken_accuracy':True,
        'finite_numeric_ctc_subthreshold_and_missing_number_cases':True,'reader_not_invoked_after_intent_rejection':True,
        'ordered_conflicting_cues_rejected':True,'human_speakers_and_hashes_disjoint':True,'auxiliary_train_excludes_human_dev_test':True,
        'audio_hashes_verified':len(human+synth+aux),'installed_and_previous_results_unchanged':True}
    if (REPORTS[mode]/'selection-before-test.json').exists():
        start=time.perf_counter();recognizer=KeywordRecognizer(mode);session_ms=(time.perf_counter()-start)*1000
        development=[r for r in human if r['split']=='dev'];selected=[]
        for c in FAMILIES:selected += [r for r in development if r['labels']['intent']==c][:6]
        pcm=[load_audio(ROOT/r['audio'])[0] for r in selected];latency=[]
        for p in pcm[:3]:recognizer.predict(p)
        for p in pcm:
            start=time.perf_counter();recognizer.predict(p);latency.append((time.perf_counter()-start)*1000)
        result['cpu_timing']={'n':len(pcm),'p50_ms':float(np.median(latency)),'p95_ms':float(np.percentile(latency,95)),'session_load_ms':session_ms,
            'threads':2,'includes':'frontend, chosen intent head, accepted-intent keyword reader, finite numeric graph and typed validation',
            'excludes':'disk, speech duration, VAD and 400 ms endpoint waiting','same_preloaded_development_pcm':True}
        pipeline=KeywordPipeline(mode);replay=[]
        for r,p in zip(selected,pcm):replay.append({'id':r['id'],'reference':r['labels'],'results':pipeline.process(p)})
        dump(REPORTS[mode]/'development-endpoint-replay.json',replay)
        result['endpoint_replay']={'n':len(replay),'partition':'development','full_historical_test_not_replayed':True,'wake_detector_tested':False}
        demonstrations=[]
        for source,all_rows in [('human',human),('synthetic',synth)]:
            for family,operation,amount in [('BRIGHTNESS','SET','50'),('BRIGHTNESS','SET','100'),('BRIGHTNESS','INCREASE','10'),('POWER_ON','NONE','NONE')]:
                match=next((r for r in all_rows if r['split']=='dev' and r['labels']['intent']==family and r['labels']['operation']==operation and r['labels']['amount']==amount),None)
                if match:
                    p=load_audio(ROOT/match['audio'])[0];demonstrations.append({'source':source,'id':match['id'],'transcript':match['transcript'],'reference':match['labels'],'result':recognizer.predict(p)})
        dump(REPORTS[mode]/'ordered-cue-examples.json',demonstrations)
    dump(REPORTS[mode]/'checks.json',result);print('CHECKS',mode,json.dumps(result),flush=True)


def report(mode):
    out=REPORTS[mode];result=json.loads((out/'results.json').read_text());selected=json.loads((out/'selection-before-test.json').read_text());training=json.loads((out/'training.json').read_text())
    def pct(x):return 'n/a' if x is None else f'{100*x:.2f}%'
    intent_model=selected['policy']['intent_model'];parameters=training['parameters'];graph_bytes=training['onnx_bytes']
    description='A small Conv2D/BiLSTM model predicts VCM intent, readiness, and a fixed sequence of command cues.'
    if intent_model in ('cnn','micro'):
        path=ROOT/'.cache/keyword-v2/mixed/export.json' if intent_model=='cnn' else REPORTS['mixed']/'micro-intent-training.json'
        primary=json.loads(path.read_text());parameters+=primary['parameters'];graph_bytes+=primary['onnx_bytes']
        description=('A temporal 1D CNN' if intent_model=='cnn' else 'A small Conv2D model')+' first selects and gates the VCM intent. A separate fixed Conv2D/BiLSTM keyword reader runs only after acceptance.'
    lines=[f'# {mode.title()} ordered fixed-keyword VCM','',
        description+' The accepted intent owns parsing; the keyword reader cannot select another family. The numeric grammar accepts only 0-100. This is an experimental pipeline with typed JSON and no action executor.','',
        'This version incorporates the requested Kaggle Fluent Speech corpus and studies Speech-to-Intent-Micro. The same historical human/synthetic test is reused for comparison; it is not an untouched confirmation after prior research iterations. Development selected all current weights and gates before this round of test inference.','',
        f"Selected intent candidate: {intent_model}. Loaded graphs: {parameters:,} total parameters and {graph_bytes:,} ONNX bytes. Keyword model selected epoch: {training['selected_epoch']}.",'',
        'The keyword-model checkpoints were selected using greedy CTC development decoding. The finite-number forward decoder and confidence policy were subsequently selected/calibrated on human development before test inference.','',
        '## Each intent','', '| Intent | Human examples | Valid human commands | Raw intent recall | Gated intent recall | Accepted command precision | Exact command recall |','|---|---:|---:|---:|---:|---:|---:|']
    m=result['tests']['human']['gated']['per_intent']
    for c,v in m.items():lines.append(f"| {c} | {v['n']} | {v['positive_n']} | {pct(v['family_recall'])} | {pct(v['accepted_intent_recall'])} | {pct(v['command_precision'])} | {pct(v['command_recall'])} |")
    for c in FAMILIES:
        lines+=['',f'## {c}','', '| Speech | N | Raw family recall | Exact command recall before confidence gates | Correct accepted / valid commands | Output precision |','|---|---:|---:|---:|---:|---:|']
        for source in ('human','synthetic'):
            v=result['tests'][source]['gated']['per_intent'][c];raw=result['tests'][source]['before_confidence_gates']['per_intent'][c]
            lines.append(f"| {source} | {v['n']} | {pct(v['family_recall'])} | {pct(raw['command_recall'])} | {v['correct_commands']}/{v['positive_n']} | {pct(v['command_precision'])} |")
        if c=='UNKNOWN':lines+=['', 'UNKNOWN has no valid commands. Its relevant measure is rejection and false acceptance; command recall is undefined.']
    lines+=['','## Numeric operations','','| Operation | Human positives | Correct before gates | Correct accepted | Accepted precision |','|---|---:|---:|---:|---:|']
    for key,v in result['tests']['human']['gated']['operations'].items():
        raw=result['tests']['human']['before_confidence_gates']['operations'][key]
        lines.append(f"| {key} | {v['positive_n']} | {raw['correct']} | {v['correct']} | {pct(v['precision'])} |")
    lines+=['','## Overall','','| Speech | Accepted | Correct | Precision | Exact recall | Negative false accepts |','|---|---:|---:|---:|---:|---:|']
    for source,test in result['tests'].items():
        v=test['gated']['overall'];lines.append(f"| {source} | {v['accepted']} | {v['correct_accepted']} | {pct(v['precision'])} | {pct(v['exact_recall'])} | {v['false_accepts']}/{v['negative_n']} |")
    lines+=['','## Reproduce and use','','```powershell','.venv/Scripts/python.exe scripts/keywords.py predict --mode '+mode+' --audio path/to/command.wav',
        '.venv/Scripts/python.exe scripts/keywords.py predict --mode '+mode+' --audio path/to/recording.wav --wake-end-ms 900 --vad',
        '.venv/Scripts/python.exe scripts/keywords.py list-mics','.venv/Scripts/python.exe scripts/keywords.py listen --mode '+mode+' --mic 2 --vad-debug','```','',
        'The wake end time is supplied by an external wake detector; no wake-word detector was trained or validated. The listen command processes endpointed speech and must be armed by the assistant integration. CTC emission intervals are approximate cues, not verified acoustic word boundaries.','',
        'Thresholds are selected exactly on development, with a shared negative-error budget and independent numeric-operation gates. No accepted intent is replaced by a parameter prediction. Unsupported groups abstain. See selection-before-test.json and development-search.json. Test accuracy below those targets is a failed generalization gate, even when development passed.','',
        'The TTS accents remain requested rather than verified. Human semantic labels are silver annotations, and generated transcript fidelity is unadjudicated. Names, artists and new-word tests are outside scope. CPU/replay checks do not establish live-microphone or Pi performance.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


def main(args):
    modes=['mixed','synthetic'] if args.mode=='both' else [args.mode]
    if args.command in ('calibrate','evaluate','check','report'):
        for mode in modes:globals()[args.command](mode)
    elif args.command=='list-mics':
        import sounddevice as sd
        print(json.dumps([{'index':i,**d} for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0],indent=2))
    elif args.command=='listen':
        from .capture import listen
        listen(KeywordPipeline(args.mode),args)
    elif args.command=='predict':
        if args.audio is None:raise ValueError('predict requires --audio')
        pcm,_=load_audio(args.audio)
        if args.vad:result=KeywordPipeline(args.mode).process(pcm,args.wake_end_ms)
        else:
            start=round(args.wake_end_ms*16)
            if start<0 or start>=len(pcm):raise ValueError('Wake end lies outside audio')
            result=KeywordRecognizer(args.mode).predict(pcm[start:],args.wake_end_ms)
        print(json.dumps(result,indent=2,ensure_ascii=False))
