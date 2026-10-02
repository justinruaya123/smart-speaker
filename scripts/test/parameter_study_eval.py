"""Catalog/enrollment ablations, untouched speaker test and matched CPU timing."""
from collections import defaultdict, Counter
import json
import platform
import random
import time
import numpy as np
import torch
from einops import rearrange, pack, reduce
from .parameter_study_data import ROOT, OUT, DATA, dump, digest, pcm, lexicon, phones
from .parameter_study import collate, decode
from .parameter_models import KeywordEncoder
from .stt import BrightnessSTT
from .stt_runtime import stt_features
from .levels import spoken_number


def infix_distance(pattern, text):
    """Myers bit-vector Levenshtein with a free leading text boundary.

    The exact minimum edit distance from pattern to ANY text substring.
    Python integers allow queries beyond the machine word length.
    """
    if not pattern:return 0
    peq={}
    for i,c in enumerate(pattern):peq[c]=peq.get(c,0)|(1<<i)
    pv=~0;mv=0;score=len(pattern);best=score;high=1<<(len(pattern)-1)
    for c in text:
        eq=peq.get(c,0);xv=eq|mv;xh=(((eq&pv)+pv)^pv)|eq
        ph=mv|~(xh|pv);mh=pv&xh
        if ph&high:score+=1
        if mh&high:score-=1
        ph <<= 1  # free prefix: D[0,j] = 0, not j
        pv=(mh<<1)|~(xv|ph);mv=ph&xv;best=min(best,score)
    return best


def check_matcher():
    rng=random.Random(443)
    for _ in range(500):
        a=[rng.randrange(4) for _ in range(rng.randrange(1,35))]
        b=[rng.randrange(4) for _ in range(rng.randrange(0,70))]
        previous=[0]*(len(b)+1)
        for i,aa in enumerate(a):
            current=[i+1]
            for j,bb in enumerate(b):current.append(min(previous[j+1]+1,current[j]+1,previous[j]+(aa!=bb)))
            previous=current
        assert infix_distance(a,b)==min(previous),(a,b)


def query_rows(rows, split):
    queries=[]
    for i,r in enumerate(rows):
        if r['split']!=split:continue
        for t in r['targets']:
            role=t['role'];group=('isolated_number' if r['kind']=='word' else 'natural_number') if role=='number' else role
            queries.append(dict(index=i,id=r['id'],role=role,reference=t['value'],group=group,novel=t['novel'] if 'novel' in t else False))
    # Same fixed original negative recordings are presented to every reader.
    negatives=[i for i,r in enumerate(rows) if r['split']==split and not r['targets']][:300]
    for role in ('number','artist','song','device','word'):
        for i in negatives:queries.append(dict(index=i,id=rows[i]['id'],role=role,reference=None,group='negative_'+role,novel=False))
    return queries


def summarize(records, threshold=None):
    positives=[r for r in records if r['reference'] is not None]
    accepted=[r for r in records if r['prediction'] is not None and (threshold is None or r['confidence']>=threshold)]
    correct=sum(r['prediction']==r['reference'] for r in accepted)
    negatives=[r for r in records if r['reference'] is None]
    return dict(n=len(records),positives=len(positives),negatives=len(negatives),correct=correct,emitted=len(accepted),
                recall=correct/len(positives) if positives else None,precision=correct/len(accepted) if accepted else None,
                false_accepts=sum(r['reference'] is None for r in accepted),
                negative_false_accept_rate=sum(r['reference'] is None for r in accepted)/len(negatives) if negatives else None)


def threshold_on_dev(records):
    # Operating point fixed without test outcomes. A failed role abstains.
    eligible=sorted((r for r in records if r['prediction'] is not None),key=lambda r:-r['confidence'])
    best=0;chosen=2.;correct=0;details=None
    for i,r in enumerate(eligible):
        correct+=r['prediction']==r['reference']
        if i+1<len(eligible) and eligible[i+1]['confidence']==r['confidence']:continue
        if i+1>=10 and correct/(i+1)>=.95 and correct>best:
            best=correct;chosen=r['confidence'];details=dict(correct=correct,emitted=i+1,precision=correct/(i+1))
    return dict(threshold=chosen,minimum_emissions=10,target_precision=.95,passed=details is not None,development=details)


class Matcher:
    def __init__(self,name,vocab,catalog,classes=None,prototypes=None):
        self.name=name;self.vocab=vocab;self.catalog=catalog;self.prototypes=prototypes or {};self.patterns={};self.oov=[]
        dictionary=lexicon()
        for role,values in catalog.items():
            patterns=[]
            for value in values:
                aliases=[spoken_number(int(value))] if role=='number' else [value]
                if role=='number' and value=='100':aliases.append('hundred')
                for alias in aliases:
                    sequence=phones(alias,dictionary) if name=='phoneme' else list(alias)
                    if sequence is None:self.oov.append((role,value));continue
                    boundary='|' if name=='phoneme' else ' '
                    patterns.append((value,[boundary]+sequence+[boundary]))
            self.patterns[role]=patterns

    def predict(self, output,role):
        first,aux=output
        if self.name=='fixed':
            shifted=first-first.max();prob=np.exp(shifted);prob/=prob.sum()
            candidates=[(i,k.split('::',1)[1]) for i,k in enumerate(self.vocab) if k.startswith(role+'::') and k.split('::',1)[1] in self.catalog.get(role,[])]
            if not candidates:return None,0.
            i,value=max(candidates,key=lambda kv:prob[kv[0]])
            if prob[0]>=prob[i]:return None,float(prob[i])
            return value,float(prob[i])
        if self.name=='prototype':
            candidates=[(value,vector) for value,vector in self.prototypes.get(role,{}).items() if value in self.catalog.get(role,[])]
            if not candidates:return None,0.
            scores=[float(aux@vector) for _,vector in candidates];order=np.argsort(scores)[::-1]
            best=int(order[0]);gap=scores[best]-(scores[int(order[1])] if len(order)>1 else 0.)
            # Margin penalizes colliding names; cosine is not a probability.
            return candidates[best][0],max(0.,min(1.,(scores[best]+gap)/2))
        tokens=decode(first,int(aux),self.vocab)
        boundary='|' if self.name=='phoneme' else ' '
        tokens=[boundary]+tokens+[boundary]
        scores={}
        for value,pattern in self.patterns.get(role,[]):
            similarity=1-infix_distance(pattern,tokens)/len(pattern)
            # Tie-breaking favors a complete compound such as twenty one over one.
            score=min(1.,similarity)*(1-.01/max(1,len(pattern)))
            scores[value]=max(scores.get(value,-1.),score)
        if not scores:return None,0.
        values=sorted(scores,key=lambda v:(-scores[v],v));best=values[0]
        if scores[best]<=0:return None,0.
        if len(values)>1 and abs(scores[best]-scores[values[1]])<1e-12:return None,scores[best]
        return best,scores[best]


def embedding_prototypes(rows, outputs, shots):
    values=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split'] not in ('train','enroll'):continue
        for t in r['targets']:values[(t['role'],t['value'])].append(i)
    result=defaultdict(dict);counts={}
    for (role,value),indices in sorted(values.items()):
        selected=[];speakers=set()
        for i in indices:
            if rows[i]['speaker'] in speakers:continue
            selected.append(i);speakers.add(rows[i]['speaker'])
            if len(selected)>=shots:break
        vectors,_=pack([rearrange(outputs[i][1],'e -> 1 e') for i in selected],'* e')
        vector=reduce(vectors,'n e -> e','mean');vector/=max(1e-12,np.linalg.norm(vector))
        result[role][value]=vector;counts[role+'::'+value]=len(selected)
    return dict(result),counts


def batch_outputs(name, saved, rows, features):
    # Score through the deployment engine. CUDA convolutions can differ enough
    # in floating-point confidence to cross a tightly calibrated threshold.
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(OUT/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    output={}
    indices=list(range(len(rows))) if name=='prototype' else [i for i,r in enumerate(rows) if r['split'] in ('dev','test')]
    for offset in range(0,len(indices),32):
        x,l,ii=collate(indices[offset:offset+32],features)
        a,b=session.run(None,{'features':x.numpy(),'lengths':l.numpy()})
        for j,i in enumerate(ii):output[i]=(a[j,:int(b[j])].copy(),int(b[j])) if name in ('phoneme','character') else (a[j].copy(),b[j].copy())
    return output


def evaluate(rows,features):
    check_matcher();torch.set_num_threads(4)
    catalog=json.loads((DATA/'catalogs.json').read_text());report={'models':{},'protocol':{
        'manifest_sha256':digest(DATA/'manifest.jsonl'),'all_weights_randomly_initialized':True,'synthetic_audio':False,
        'parameter_only':True,'slot_role_supplied_from_annotation':True,'full_VCM_pipeline_accuracy':False,
        'no_verified_word_boundaries':True,'training_audio_enrollment_only':True,'operating_point':'per-role dev precision >=95%, >=10 emissions; otherwise abstain',
        'test_used_for_selection':False,'unknown_queries':'300 original background recordings per role, repeated across roles',
        'home_assistant_proxy':'published room/place entity labels; no user Home Assistant inventory',
        'phoneme_decoder':'greedy phone CTC followed by exact infix edit-distance catalog matching; not a WFST or a paper reproduction'},
        'hardware':{'platform':platform.platform(),'CPU_threads':2,'provider':'CPUExecutionProvider','physical_pi':False,'live_microphone':False}}
    dev=query_rows(rows,'dev');test=query_rows(rows,'test');test_indices=sorted({q['index'] for q in test if q['reference'] is not None})
    # Identical, mixed-duration waveform set selected by recording index, not model outcomes.
    indices=random.Random(941).sample(test_indices,min(96,len(test_indices)));timing_pcm={i:pcm(rows[i]) for i in indices}
    timing_role={i:next(q['role'] for q in test if q['index']==i and q['reference'] is not None) for i in indices}
    from datetime import datetime,timezone
    locked={'created_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':digest(DATA/'manifest.jsonl'),'models':{}}
    pending=[]
    for name in ('fixed','prototype','phoneme','character'):
        saved=torch.load(OUT/f'{name}.pt',weights_only=False,map_location='cpu')
        if saved['manifest_sha256']!=digest(DATA/'manifest.jsonl'):
            raise ValueError('Checkpoint does not belong to the frozen study manifest')
        exported=json.loads((OUT/f'{name}-training.json').read_text())
        if exported['manifest_sha256']!=saved['manifest_sha256'] or exported['graph_sha256']!=digest(OUT/f'{name}.onnx'):
            raise ValueError('Export does not belong to this checkpoint/manifest')
        outputs=batch_outputs(name,saved,rows,features)
        variants={};protos={};enrollment_counts={}
        for shots in ([1,3,5] if name=='prototype' else [0]):
            if shots:protos,enrollment_counts=embedding_prototypes(rows,outputs,shots)
            for mode in ('fixed','expanded'):
                key=f'{mode}_{shots}shot' if shots else mode
                matcher=Matcher(name,saved['vocabulary'],catalog[mode],prototypes=protos)
                development=[]
                for q in dev:
                    predicted,confidence=matcher.predict(outputs[q['index']],q['role'])
                    development.append({**q,'prediction':predicted,'confidence':confidence})
                policies={role:threshold_on_dev([r for r in development if r['role']==role]) for role in catalog['expanded']}
                variants[key]=(matcher,policies,development)
                locked['models'][name+'/'+key]={'graph_sha256':digest(OUT/f'{name}.onnx'),'policies':policies}
        pending.append((name,saved,outputs,variants,enrollment_counts))
        print('Locked development thresholds',name,flush=True)
    # Persist every weight/threshold/catalog choice before scoring the held-out split.
    locked['catalog_sha256']=digest(DATA/'catalogs.json');dump(OUT/'selection-before-test.json',locked)
    for name,saved,outputs,variants,enrollment_counts in pending:
        model_report={'training':json.loads((OUT/f'{name}-training.json').read_text()),'variants':{},'enrollment_counts_5shot':enrollment_counts}
        for key,(matcher,policies,development) in variants.items():
            records=[]
            for q in test:
                predicted,confidence=matcher.predict(outputs[q['index']],q['role'])
                records.append({**q,'prediction':predicted,'confidence':confidence,'accepted':predicted is not None and confidence>=policies[q['role']]['threshold']})
            def gated(rr):return summarize([{**r,'prediction':r['prediction'] if r['accepted'] else None} for r in rr])
            groups={g:[r for r in records if r['group']==g] for g in sorted({r['group'] for r in records})}
            novel=[r for r in records if r['novel']]
            model_report['variants'][key]={'raw':summarize(records),'gated':gated(records),
                'groups':{g:{'raw':summarize(rr),'gated':gated(rr)} for g,rr in groups.items()},
                'novel_names':{'raw':summarize(novel),'gated':gated(novel),'by_role':{role:{'raw':summarize([r for r in novel if r['role']==role]),'gated':gated([r for r in novel if r['role']==role])} for role in ('artist','song','device')}},
                'policies':policies,'catalog_OOV':matcher.oov,
                'per_number':{str(v):{'raw':summarize([r for r in records if r['role']=='number' and r['reference']==str(v)]),'gated':gated([r for r in records if r['role']=='number' and r['reference']==str(v)])} for v in range(101)}}
            dump(OUT/f'{name}-{key}-predictions.json',records)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        begin=time.perf_counter();session=ort.InferenceSession(str(OUT/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider']);load_ms=(time.perf_counter()-begin)*1000
        matcher=variants['expanded_5shot' if name=='prototype' else 'expanded'][0]
        timings=[];graph_times=[];rtf=[];parity=[]
        for repeat in range(4):
            for i in indices:
                begin=time.perf_counter();f=stt_features(timing_pcm[i]);t=time.perf_counter()
                a,b=session.run(None,{'features':rearrange(f,'t m -> 1 t m'),'lengths':np.array([len(f)],dtype=np.int64)})
                graph_elapsed=(time.perf_counter()-t)*1000
                actual=(a[0],b[0]);prediction=matcher.predict(actual,timing_role[i]);elapsed=(time.perf_counter()-begin)*1000
                if repeat:
                    timings.append(elapsed);graph_times.append(graph_elapsed);rtf.append(elapsed/1000/(len(timing_pcm[i])/16000))
                else:
                    expected=matcher.predict(outputs[i],timing_role[i]);parity.append(prediction[0]==expected[0] and abs(prediction[1]-expected[1])<1e-4)
        if not all(parity):raise AssertionError((name,'batched/single CPU task parity',Counter(parity)))
        model_report['latency']={'unique_utterances':len(indices),'repeats':3,'n':len(timings),'p50_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),
            'graph_p50_ms':float(np.median(graph_times)),'median_rtf':float(np.median(rtf)),'session_construction_ms':load_ms,
            'includes':'mono 16k PCM frontend, CPU ONNX, decoding and expanded catalog matching; no disk I/O, capture or endpoint wait',
            'task_batch_single_CPU_parity_samples':sum(parity),'timing_recording_ids':[rows[i]['id'] for i in indices]}
        report['models'][name]=model_report;dump(OUT/'results.json',report)
        print('Held-out result',name,json.dumps(model_report['variants']['expanded_5shot' if name=='prototype' else 'expanded']['gated']),flush=True)
    return report


def scale_catalog(rows):
    """Timing only: text distractors change catalog size, never create audio."""
    import re
    import onnxruntime as ort
    dictionary=lexicon();rng=random.Random(882)
    words=sorted(w for w in dictionary if re.fullmatch('[a-z]{3,10}',w))
    rng.shuffle(words)
    base=json.loads((DATA/'catalogs.json').read_text())['expanded']['artist']
    names=list(dict.fromkeys(base+[words[i]+' '+words[i+1] for i in range(0,10000,2)]))
    test=[r for r in rows if r['split']=='test' and any(t['role']=='artist' for t in r['targets'])][:8]
    samples=[pcm(r) for r in test]
    result={'scope':'Timing only; generated text-name distractors, original human audio. No accuracy or music-catalog claim.',
            'names_seed':882,'threads':2,'utterance_ids':[r['id'] for r in test],'models':{}}
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    for name in ('phoneme','character'):
        saved=torch.load(OUT/f'{name}.pt',map_location='cpu',weights_only=False)
        session=ort.InferenceSession(str(OUT/f'{name}.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        sizes={}
        for size in (100,1000,5000):
            started=time.perf_counter();matcher=Matcher(name,saved['vocabulary'],{'artist':names[:size]})
            build_ms=(time.perf_counter()-started)*1000;times=[]
            for repeat in range(4):
                for audio in samples:
                    started=time.perf_counter();f=stt_features(audio)
                    a,b=session.run(None,{'features':rearrange(f,'t m -> 1 t m'),'lengths':np.array([len(f)],dtype=np.int64)})
                    matcher.predict((a[0],b[0]),'artist')
                    if repeat:times.append((time.perf_counter()-started)*1000)
            sizes[str(size)]={'declared_names':size,'usable_pronunciations':len(matcher.patterns['artist']),
                'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'n':len(times),'catalog_build_ms':build_ms}
        result['models'][name]=sizes
    dump(OUT/'catalog-scaling.json',result);return result
