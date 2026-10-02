"""CPU-only catalog retrieval, abstention, comparisons, and isolated adapter."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import platform
import random
import time
import numpy as np
from einops import rearrange, reduce
from .speech_embedding_data import ROOT, DATA, OUT, REPORT, prepare, token_data, preserved_check
from .parameter_study_data import dump, digest, pcm, lexicon, phones
from .parameter_study import collate
from .parameter_study_eval import query_rows
from .speech_embedding_study import ort_session, token_batch
from .stt_runtime import stt_features
from .levels import spoken_number


MODES = {'dual':['text_local','text_global'],
         'dual_kws':['text_local','text_global','kws','audio_1','audio_3','audio_5','hybrid_3']}
CATALOGS = ('fixed','expanded','playlist10','playlist25')


def fingerprint(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def metadata(name):
    saved=json.loads((OUT/f'{name}-training.json').read_text())
    if saved['protocol_sha256']!=digest(DATA/'protocol.json'):
        raise ValueError('Model protocol changed')
    for kind in ('audio','text'):
        if saved['graphs'][kind]['sha256']!=digest(OUT/f'{name}-{kind}.onnx'):
            raise ValueError('Model graph changed')
    return saved


def queries(rows,split):
    result=query_rows(rows,split)
    # Artist-vs-song confusion matters even when a caller supplies a slot role.
    for role,other in [('artist','song'),('song','artist')]:
        for i,r in enumerate(rows):
            if r['split']==split and any(t['role']==other for t in r['targets']) and not any(t['role']==role for t in r['targets']):
                result.append(dict(index=i,id=r['id'],role=role,reference=None,novel=False,group='other_role_'+role))
    return result


def encode_audio(name, rows, features, splits):
    saved=metadata(name)
    key='-'.join(splits)+'-'+saved['graphs']['audio']['sha256'][:12]
    path=OUT/f'{name}-outputs-{key}.npz'
    if path.exists():
        data=np.load(path)
        return {int(i):(a,b) for i,a,b in zip(data['indices'],data['embeddings'],data['logits'])}
    indices=[i for i,r in enumerate(rows) if r['split'] in splits]
    session=ort_session(OUT/f'{name}-audio.onnx');result={}
    for first in range(0,len(indices),32):
        x,l,ii=collate(indices[first:first+32],features)
        a,b=session.run(None,{'features':x.numpy(),'lengths':l.numpy()})
        for j,i in enumerate(ii):result[i]=(a[j].copy(),b[j].copy())
    np.savez(path,indices=indices,embeddings=rearrange([result[i][0] for i in indices],'b w e -> b w e'),
             logits=rearrange([result[i][1] for i in indices],'b c -> b c'))
    return result


def enroll(rows,outputs,shots):
    pool=defaultdict(list)
    for i,r in enumerate(rows):
        if r['split'] not in ('train','enroll'):continue
        for t in r['targets']:pool[t['role'],t['value']].append(i)
    result={};details={}
    for (role,value),indices in sorted(pool.items()):
        selected=[];speakers=set()
        for i in sorted(indices,key=lambda i:rows[i]['id']):
            if rows[i]['speaker'] in speakers:continue
            selected.append(i);speakers.add(rows[i]['speaker'])
            if len(selected)==shots:break
        vectors=rearrange([outputs[i][0][0] for i in selected],'n e -> n e')
        vector=reduce(vectors,'n e -> e','mean');vector/=max(1e-12,float(np.linalg.norm(vector)))
        result[role,value]=vector
        details[role+'::'+value]=dict(speakers=len(speakers),ids=[rows[i]['id'] for i in selected])
    return result,details


class CatalogMatcher:
    """Precompute text or enrolled name vectors; resolve only an authorized role.

    A mode is chosen explicitly. There is no automatic reader fallback.
    """
    def __init__(self,name,mode,catalog,prototypes=None):
        self.name=name;self.mode=mode;self.version=fingerprint(catalog)
        self.ids={};overrides={}
        if 'entries' in catalog:
            original=catalog;catalog=defaultdict(list);seen=set()
            from .parameter_study_data import normalize
            for entry in original['entries']:
                role=entry['role'];value=normalize(entry['name']);identity=entry['id']
                if role not in ('artist','song','device') or not value or not isinstance(identity,str) or not identity:
                    raise ValueError('Catalog entries require a supported role, name, and stable string ID')
                if (role,value) in seen:raise ValueError('Duplicate normalized name in catalog')
                if identity in self.ids.values():raise ValueError('Duplicate entity ID in catalog')
                seen.add((role,value));catalog[role].append(value);self.ids[role,value]=identity
                if 'pronunciation' in entry:overrides[role,value]=entry['pronunciation']
            catalog=dict(catalog)
        self.catalog=catalog
        self.saved=metadata(name);self.entries={};self.matrices={};self.ambiguous={};self.unavailable={}
        self.prototypes=prototypes or {};dictionary=lexicon()
        vocab={v:i for i,v in enumerate(self.saved['vocabulary'])}
        session=ort_session(OUT/f'{name}-text.onnx') if mode.startswith('text') or mode.startswith('hybrid') else None
        for role,values in catalog.items():
            by_phone=defaultdict(set);sequences={}
            for value in values:
                sequence=overrides.get((role,value),phones(spoken_number(int(value)) if role=='number' else value,dictionary))
                if sequence is not None and (not isinstance(sequence,list) or not sequence or any(p not in vocab or p=='<pad>' for p in sequence)):
                    raise ValueError('Pronunciation must use existing non-padding phone classes')
                if sequence:
                    sequences[value]=[vocab[p] for p in sequence]
                    by_phone[tuple(sequence)].add(value)
            self.ambiguous[role]={v for vs in by_phone.values() if len(vs)>1 for v in vs}
            if mode=='kws':
                entries=[v for v in values if role+'::'+v in self.saved['classes']]
                self.matrices[role]=[self.saved['classes'].index(role+'::'+v) for v in entries]
            elif mode.startswith('audio'):
                entries=[v for v in values if (role,v) in self.prototypes]
                self.matrices[role]=rearrange([self.prototypes[role,v] for v in entries],'n e -> n e') if entries else np.empty((0,96),dtype='float32')
            else:
                entries=[v for v in values if v in sequences]
                if entries:
                    t,l=token_batch([sequences[v] for v in entries])
                    self.matrices[role]=session.run(None,{'tokens':t.numpy(),'lengths':l.numpy()})[0]
                else:self.matrices[role]=np.empty((0,96),dtype='float32')
            self.entries[role]=entries
            self.unavailable[role]=[v for v in values if v not in entries]

    def rank(self,output,role):
        entries=self.entries.get(role,[])
        if not entries:return dict(prediction=None,confidence=0.,margin=0.,top5=[],reason='no_supported_candidates')
        windows,logits=output
        if not np.isfinite(windows).all() or not np.isfinite(logits).all():
            return dict(prediction=None,confidence=0.,margin=0.,top5=[],reason='nonfinite_model_output')
        if self.mode=='kws':
            prob=np.exp(logits-logits.max());prob/=prob.sum()
            scores=prob[self.matrices[role]]
        elif self.mode.startswith('audio'):
            scores=self.matrices[role]@windows[0]
        else:
            similarities=windows@rearrange(self.matrices[role],'n e -> e n')
            scores=similarities[0] if self.mode=='text_global' else reduce(similarities,'w n -> n','max')
            if self.mode.startswith('hybrid'):
                # Predeclared multimodal score; missing audio retains text enrollment.
                scores=scores.copy()
                for j,value in enumerate(entries):
                    if (role,value) in self.prototypes:
                        scores[j]=.5*scores[j]+.5*float(windows[0]@self.prototypes[role,value])
        order=np.argsort(-scores,kind='stable');best=int(order[0]);value=entries[best]
        gap=float(scores[best]-(scores[int(order[1])] if len(order)>1 else 0.))
        confidence=float(scores[best]) if self.mode=='kws' else max(0.,min(1.,(float(scores[best])+gap)/2))
        prediction=value;reason='ranked'
        if value in self.ambiguous.get(role,set()) or (len(order)>1 and gap<=1e-6):
            prediction=None;reason='ambiguous_name'
        if self.mode=='kws' and prob[0]>=scores[best]:prediction=None;reason='other_class'
        return dict(prediction=prediction,confidence=confidence,margin=gap,
                    top5=[entries[int(j)] for j in order[:5]],reason=reason)

    def resolve(self,output,role,accepted_intent,gate,calibrated_version):
        base=dict(intent=accepted_intent,role=role,status='abstain',value=None,entity_id=None,
                  catalog_version=self.version,action_executed=False)
        owns={'artist':{'MEDIA_PLAY'},'song':{'MEDIA_PLAY'},'device':set(),'number':set(),'word':set()}
        if accepted_intent not in owns.get(role,set()):return {**base,'reason':'intent_does_not_own_slot'}
        if calibrated_version!=self.version:return {**base,'reason':'catalog_requires_calibration'}
        if not gate or not gate.get('passed'):return {**base,'reason':'quality_gate_failed'}
        threshold=gate.get('threshold')
        if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
            return {**base,'reason':'invalid_calibration'}
        ranked=self.rank(output,role)
        if ranked['prediction'] is None or ranked['confidence']<threshold:return {**base,'reason':'low_confidence_or_ambiguous'}
        return {**base,'status':'accepted','value':ranked['prediction'],
                'entity_id':self.ids.get((role,ranked['prediction']),role+':'+fingerprint(ranked['prediction'])[:16]),
                'confidence':ranked['confidence'],'confidence_is_probability':False,'reason':'matched_catalog'}


def records_for(matcher,outputs,query):
    result=[]
    for q in query:
        if q['role'] not in matcher.catalog:continue
        result.append({**q,**matcher.rank(outputs[q['index']],q['role']),
                       'in_catalog':q['reference'] in matcher.catalog[q['role']],
                       'has_representation':q['reference'] in matcher.entries[q['role']]})
    return result


def gate_on_dev(records):
    eligible=sorted([r for r in records if r['prediction'] is not None],key=lambda r:-r['confidence'])
    unknown=sum(not r['in_catalog'] for r in records)
    correct=0;false=0;best=0;selected=None
    for j,r in enumerate(eligible):
        correct+=r['reference'] is not None and r['prediction']==r['reference']
        false+=not r['in_catalog']
        if j+1<len(eligible) and eligible[j+1]['confidence']==r['confidence']:continue
        if j+1>=10 and correct/(j+1)>=.95 and false/max(1,unknown)<=.01 and correct>best:
            best=correct;selected=dict(threshold=r['confidence'],emitted=j+1,correct=correct,
                                      precision=correct/(j+1),unknown_false_accepts=false)
    return dict(passed=selected is not None,threshold=selected['threshold'] if selected else 1.1,
                target_precision=.95,minimum_emissions=10,max_unknown_false_accept_rate=.01,
                development=selected,unknown_trials=unknown)


def summarize(records,gates):
    refs=[r for r in records if r['reference'] is not None]
    known=[r for r in refs if r['in_catalog']]
    accepted=[r for r in records if r['prediction'] is not None and gates[r['role']]['passed'] and r['confidence']>=gates[r['role']]['threshold']]
    correct=sum(r['reference'] is not None and r['prediction']==r['reference'] for r in refs)
    acorrect=sum(r['reference'] is not None and r['prediction']==r['reference'] for r in accepted)
    unknown=[r for r in records if not r['in_catalog']]
    false=sum(not r['in_catalog'] for r in accepted)
    return dict(trials=len(records),reference_queries=len(refs),in_catalog_queries=len(known),
                represented_queries=sum(r['has_representation'] for r in refs),raw_correct=correct,
                raw_top1=correct/max(1,len(refs)),top5_correct=sum(r['reference'] in r['top5'] for r in refs),
                in_catalog_raw_top1=correct/len(known) if known else None,accepted_correct=acorrect,emitted=len(accepted),
                precision=acorrect/len(accepted) if accepted else None,recall=acorrect/len(refs) if refs else None,
                unknown_trials=len(unknown),unknown_false_accepts=false,unknown_far=false/len(unknown) if unknown else None,
                absent_name_trials=sum(r['reference'] is not None for r in unknown),
                absent_name_false_accepts=sum(r['reference'] is not None and not r['in_catalog'] for r in accepted))


def evaluate(rows,protocol,features):
    if (OUT/'results.json').exists():raise ValueError('Test results already exist. Preserve them; rerun only in a new evidence version.')
    specification=dict(modes=MODES,catalogs=CATALOGS,hybrid_text_weight=.5,
                       confidence='clamp((top cosine + runner-up margin)/2,0,1); softmax for fixed KWS',
                       unknowns='same 300 human backgrounds per role; wrong music roles; names absent from each catalog',
                       role_supplied=True,playlist_selection='top training-frequency names only; no per-query catalog',
                       cpu_threads=2,blas_threads=1,test_used_for_selection=False)
    dump(OUT/'evaluation-specification.json',specification)
    catalogs=json.loads((DATA/'catalogs.json').read_text())
    dev=queries(rows,'dev');test=queries(rows,'test')
    locked=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=digest(DATA/'protocol.json'),
                specification_sha256=digest(OUT/'evaluation-specification.json'),catalog_sha256=digest(DATA/'catalogs.json'),
                source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [
                    ROOT/'scripts/models/speech_embedding.py',ROOT/'scripts/train/speech_embedding_study.py',
                    ROOT/'scripts/test/speech_embedding_eval.py',ROOT/'scripts/train/speech_embedding_data.py',
                    ROOT/'experiments/speechembedding/run.py']},variants={})
    pending=[]
    for name,modes in MODES.items():
        output=encode_audio(name,rows,features,['train','enroll','dev'])
        prototypes={};coverage={}
        for shots in (1,3,5):prototypes[shots],coverage[shots]=enroll(rows,output,shots)
        dump(OUT/f'{name}-enrollment.json',coverage)
        for mode in modes:
            shots=int(mode.split('_')[-1]) if mode.startswith(('audio','hybrid')) else 0
            for catalog_name in CATALOGS:
                matcher=CatalogMatcher(name,mode,catalogs[catalog_name],prototypes.get(shots))
                records=records_for(matcher,output,dev)
                gates={role:gate_on_dev([r for r in records if r['role']==role]) for role in matcher.catalog}
                key=name+'/'+mode+'/'+catalog_name
                development={role:summarize([r for r in records if r['role']==role],gates) for role in matcher.catalog}
                locked['variants'][key]=dict(graphs=metadata(name)['graphs'],catalog_version=matcher.version,gates=gates,development=development,
                                             unavailable_names=matcher.unavailable,ambiguous_names={k:sorted(v) for k,v in matcher.ambiguous.items()})
                pending.append((key,matcher,gates))
                dump(OUT/(key.replace('/','-')+'-dev.json'),records)
        print('Calibrated on CPU development:',name,flush=True)
    # Select preferred text-only mode on dev, never by the subsequent test table.
    choices=[k for k in locked['variants'] if '/text_' in k and k.endswith('/expanded')]
    def devscore(key):
        groups=locked['variants'][key]['development']
        return .5*groups['artist']['raw_top1']+.5*groups['song']['raw_top1']
    locked['preferred_text_variant']=max(choices,key=devscore)
    locked['selection_completed_utc']=datetime.now(timezone.utc).isoformat()
    dump(OUT/'selection-before-test.json',locked)
    result=dict(protocol=specification,selection_sha256=digest(OUT/'selection-before-test.json'),
                test_started_utc=datetime.now(timezone.utc).isoformat(),preferred_text_variant=locked['preferred_text_variant'],
                hardware=dict(platform=platform.platform(),cpu_threads=2,blas_threads=1,provider='CPUExecutionProvider',physical_pi=False,live_microphone=False),variants={})
    training_names={(t['role'],t['value']) for r in rows if r['split']=='train' for t in r['targets']}
    for name in MODES:
        output=encode_audio(name,rows,features,['test'])
        for key,matcher,gates in pending:
            if matcher.name!=name:continue
            records=records_for(matcher,output,test)
            for r in records:
                g=gates[r['role']];r['accepted']=bool(r['prediction'] is not None and g['passed'] and r['confidence']>=g['threshold'])
                r['name_seen_in_training']=(r['role'],r['reference']) in training_names
            summaries={role:summarize([r for r in records if r['role']==role],gates) for role in matcher.catalog}
            groups={group:summarize([r for r in records if r['group']==group],gates) for group in sorted({r['group'] for r in records})}
            novel={role:summarize([r for r in records if r['role']==role and r['novel']],gates) for role in matcher.catalog}
            coverage={role:{kind:summarize([r for r in records if r['role']==role and r['reference'] is not None and r['name_seen_in_training']==seen],gates)
                            for kind,seen in [('seen',True),('unseen',False)]} for role in matcher.catalog}
            result['variants'][key]=dict(by_role=summaries,by_group=groups,novel=novel,by_training_coverage=coverage,gates=gates)
            dump(OUT/(key.replace('/','-')+'-test.json'),records)
        print('Scored frozen test:',name,flush=True)
    result['preservation']=preserved_check(protocol)
    dump(OUT/'results.json',result)
    print('Saved held-out results',OUT/'results.json',flush=True)


def checks(rows,protocol):
    """Software/data checks are independent of spoken-name accuracy."""
    from concurrent.futures import ThreadPoolExecutor
    failures=[]
    def verify(row):
        return row['id'] if digest(ROOT/row['audio'])!=row['audio_sha256'] else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        failures=[v for v in pool.map(verify,rows) if v]
    if failures:raise AssertionError(('changed original audio',failures))
    result=dict(original_human_files_rehashed=len(rows),audio_hash_mismatches=0,
                synthetic_recordings=0,split_audit=protocol['counts'],preservation=preserved_check(protocol),models={})
    for name in MODES:
        saved=metadata(name)
        result['models'][name]={'graphs':saved['graphs'],'initialization':saved['initialization']}
    from .speech_embedding import contrastive
    import torch
    values=contrastive(torch.zeros(3,3),torch.tensor([[1,1,0],[1,1,0],[0,0,1]],dtype=torch.bool))
    assert torch.allclose(values,torch.full((3,),math.log(3)))
    # Fixtures test software behavior; they are explicitly not acoustic evidence.
    fixture={'version':'checks-v1','entries':[
        {'role':'artist','id':'artist:right','name':'right'},
        {'role':'artist','id':'artist:wright','name':'wright'},
        {'role':'artist','id':'artist:custom','name':'qzcustom','pronunciation':['K','AH','S','T','AH','M']}]}
    matcher=CatalogMatcher('dual','text_local',fixture)
    assert matcher.entries['artist']==['right','wright','qzcustom']
    assert matcher.ambiguous['artist']=={'right','wright'}
    output=(rearrange([matcher.matrices['artist'][0]]*10,'w e -> w e'),np.zeros(len(matcher.saved['classes']),dtype='float32'))
    assert matcher.rank(output,'artist')['prediction'] is None
    gate={'passed':True,'threshold':0.}
    assert matcher.resolve(output,'artist','UNKNOWN',gate,matcher.version)['reason']=='intent_does_not_own_slot'
    assert matcher.resolve(output,'artist','MEDIA_PLAY',gate,'stale')['reason']=='catalog_requires_calibration'
    assert matcher.resolve(output,'artist','MEDIA_PLAY',{'passed':False},matcher.version)['reason']=='quality_gate_failed'
    for threshold in (-1.,float('nan'),float('inf'),None,1.1):
        assert matcher.resolve(output,'artist','MEDIA_PLAY',{'passed':True,'threshold':threshold},matcher.version)['reason']=='invalid_calibration'
    changed=CatalogMatcher('dual','text_local',{'version':'checks-v2','entries':fixture['entries']})
    assert matcher.version!=changed.version
    output=(rearrange([matcher.matrices['artist'][2]]*10,'w e -> w e'),output[1])
    accepted=matcher.resolve(output,'artist','MEDIA_PLAY',gate,matcher.version)
    assert accepted['entity_id']=='artist:custom' and accepted['intent']=='MEDIA_PLAY' and not accepted['action_executed']
    for entry in ({'role':'artist','id':'a','name':'custom','pronunciation':['NEW_PHONE']},
                  {'role':'artist','id':'a','name':'custom','pronunciation':[]}):
        try:CatalogMatcher('dual','text_local',{'entries':[entry]})
        except ValueError:pass
        else:raise AssertionError('Invalid phone vocabulary accepted')
    result['adapter_checks']=dict(homophone_abstention=True,explicit_pronunciation=True,stable_entity_id=True,
                                 rejected_intent=True,catalog_invalidation=True,invalid_gate=True,
                                 closed_phone_vocabulary=True,actions_executed=False)
    result['multi_positive_loss_check']=True
    # Actual PCM preprocessing parity and single/batch output checks.
    from .parameter_study import cached_features
    features=cached_features(rows)
    candidates=[i for i,r in enumerate(rows) if r['split']=='dev' and any(t['role'] in ('artist','song') for t in r['targets'])]
    candidates=random.Random(71).sample(candidates,min(24,len(candidates)))
    for i in candidates:
        assert np.allclose(stt_features(pcm(rows[i])),features[i],atol=1e-6)
    for name in MODES:
        session=ort_session(OUT/f'{name}-audio.onnx');maximum=0.
        for start in range(0,len(candidates),8):
            x,l,ii=collate(candidates[start:start+8],features)
            batch=session.run(None,{'features':x.numpy(),'lengths':l.numpy()})
            for j,i in enumerate(ii):
                single=session.run(None,{'features':rearrange(features[i],'t f -> 1 t f'),'lengths':np.array([len(features[i])],dtype='int64')})
                maximum=max(maximum,max(float(np.max(np.abs(b[j]-s[0]))) for b,s in zip(batch,single)))
        assert maximum<1e-4,(name,maximum)
        result['models'][name]['natural_audio_batch_single_max_abs']=maximum
    result['natural_audio_parity_recordings']=len(candidates)
    dump(OUT/'checks.json',result)
    return result


def benchmark(rows,features):
    """Matched timing, separated from checkpoint/threshold selection and test scoring."""
    from threadpoolctl import threadpool_limits
    chosen=[i for i,r in enumerate(rows) if r['split']=='test' and any(t['role'] in ('artist','song') for t in r['targets'])]
    chosen=random.Random(817).sample(chosen,min(48,len(chosen)))
    audio={i:pcm(rows[i]) for i in chosen}
    roles={i:next(t['role'] for t in rows[i]['targets'] if t['role'] in ('artist','song')) for i in chosen}
    catalog=json.loads((DATA/'catalogs.json').read_text())['expanded']
    result=dict(audio_ids=[rows[i]['id'] for i in chosen],recordings=len(chosen),provider='CPUExecutionProvider',
                intra_threads=2,inter_threads=1,blas_threads=1,passes=3,warmup_passes=1,
                includes='preloaded PCM -> frontend -> acoustic graph -> catalog matching',
                excludes='disk, speech duration, endpoint wait (default 400 ms), session/catalog construction',models={})
    import gc
    with threadpool_limits(limits=1):
        for name in MODES:
            start=time.perf_counter();session=ort_session(OUT/f'{name}-audio.onnx');load_ms=(time.perf_counter()-start)*1000
            output=encode_audio(name,rows,features,['train','enroll','dev'])
            prototypes,_=enroll(rows,output,3)
            for mode in (['text_local','text_global'] if name=='dual' else ['text_local','kws','audio_3','hybrid_3']):
                start=time.perf_counter();matcher=CatalogMatcher(name,mode,catalog,prototypes);compile_ms=(time.perf_counter()-start)*1000
                elapsed=[]
                for repeat in range(4):
                    for i in chosen:
                        start=time.perf_counter();x=stt_features(audio[i])
                        a,b=session.run(None,{'features':rearrange(x,'t f -> 1 t f'),'lengths':np.array([len(x)],dtype='int64')})
                        matcher.rank((a[0],b[0]),roles[i])
                        if repeat:elapsed.append((time.perf_counter()-start)*1000)
                result['models'][name+'/'+mode]=dict(p50_ms=float(np.percentile(elapsed,50)),p95_ms=float(np.percentile(elapsed,95)),
                    calls=len(elapsed),session_load_ms=load_ms,catalog_compile_ms=compile_ms,graphs=metadata(name)['graphs'])
            del output;gc.collect()
        # Re-time historical comparators on exactly the same PCM and thread policy.
        from .parameter_study_eval import Matcher, embedding_prototypes
        for name in ('fixed','prototype'):
            training=json.loads((ROOT/'.cache/parameter-study'/f'{name}-training.json').read_text())
            start=time.perf_counter();session=ort_session(ROOT/'.cache/parameter-study'/f'{name}.onnx');load_ms=(time.perf_counter()-start)*1000
            prototypes={}
            if name=='prototype':
                indices=[i for i,r in enumerate(rows) if r['split'] in ('train','enroll') and any(t['role'] in ('artist','song') for t in r['targets'])]
                subrows=[rows[i] for i in indices];suboutputs={}
                for j,i in enumerate(indices):
                    x=features[i];a,b=session.run(None,{'features':rearrange(x,'t f -> 1 t f'),'lengths':np.array([len(x)],dtype='int64')})
                    suboutputs[j]=(a[0],b[0])
                prototypes,_=embedding_prototypes(subrows,suboutputs,5)
            start=time.perf_counter();matcher=Matcher(name,training['vocabulary'],catalog,prototypes=prototypes);compile_ms=(time.perf_counter()-start)*1000
            elapsed=[]
            for repeat in range(4):
                for i in chosen:
                    start=time.perf_counter();x=stt_features(audio[i]);a,b=session.run(None,{'features':rearrange(x,'t f -> 1 t f'),'lengths':np.array([len(x)],dtype='int64')})
                    matcher.predict((a[0],b[0]),roles[i])
                    if repeat:elapsed.append((time.perf_counter()-start)*1000)
            result['models']['historical/'+name]=dict(p50_ms=float(np.percentile(elapsed,50)),p95_ms=float(np.percentile(elapsed,95)),
                calls=len(elapsed),session_load_ms=load_ms,catalog_compile_ms=compile_ms,parameters=training['parameters'],graph_bytes=training['graph_bytes'])
        from .phonetic_reader import PhoneticCatalogReader, catalog_fixture
        path=ROOT/'.cache/phonetic-v2/wide';training=json.loads((path/'training.json').read_text())
        start=time.perf_counter();session=ort_session(path/'phoneme.onnx');load_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();matcher=PhoneticCatalogReader(training['vocabulary'],lexicon(),catalog_fixture(catalog),[]);compile_ms=(time.perf_counter()-start)*1000
        elapsed=[]
        for repeat in range(4):
            for i in chosen:
                start=time.perf_counter();x=stt_features(audio[i]);a,b=session.run(None,{'features':rearrange(x,'t f -> 1 t f'),'lengths':np.array([len(x)],dtype='int64')})
                matcher.predict(a[0,:int(b[0])],roles[i],'natural')
                if repeat:elapsed.append((time.perf_counter()-start)*1000)
        result['models']['historical/phonetic_wide_graph']=dict(p50_ms=float(np.percentile(elapsed,50)),p95_ms=float(np.percentile(elapsed,95)),
            calls=len(elapsed),session_load_ms=load_ms,catalog_compile_ms=compile_ms,parameters=training['parameters'],graph_bytes=training['graph_bytes'])
        # Scaling isolates catalog matching and construction from the acoustic pass.
        dictionary=lexicon();words=sorted(w for w in dictionary if w.isalpha() and 3<=len(w)<=9)
        rng=random.Random(818);names=[]
        while len(names)<5000:
            value=' '.join(rng.sample(words,2))
            if value not in names:names.append(value)
        lock=json.loads((OUT/'selection-before-test.json').read_text());name,mode,_=lock['preferred_text_variant'].split('/')
        outputs=encode_audio(name,rows,features,['test'])
        result['scaling']=dict(protocol='timing only: generated dictionary text; no generated audio or accuracy claim',model=name+'/'+mode,catalogs={})
        for size in (100,1000,5000):
            start=time.perf_counter();matcher=CatalogMatcher(name,mode,{'artist':names[:size]});compile_ms=(time.perf_counter()-start)*1000
            elapsed=[]
            for repeat in range(4):
                for i in chosen[:8]:
                    start=time.perf_counter();matcher.rank(outputs[i],'artist')
                    if repeat:elapsed.append((time.perf_counter()-start)*1000)
            result['scaling']['catalogs'][size]=dict(compile_ms=compile_ms,matrix_bytes=matcher.matrices['artist'].nbytes,
                match_p50_ms=float(np.percentile(elapsed,50)),match_p95_ms=float(np.percentile(elapsed,95)))
    dump(OUT/'timing.json',result);print('Matched timing complete',flush=True)
    return result


def query_file(args):
    if args.intent!='MEDIA_PLAY' or args.role not in ('artist','song'):
        return dict(status='abstain',reason='intent_does_not_own_slot',intent=args.intent,acoustic_calls=0,action_executed=False)
    if not args.audio:raise ValueError('--audio is required for an accepted music query')
    lock=json.loads((OUT/'selection-before-test.json').read_text())
    name,mode,_=lock['preferred_text_variant'].split('/')
    catalogs=json.loads((DATA/'catalogs.json').read_text())
    catalog=catalogs[args.catalog] if args.catalog in CATALOGS else json.loads((ROOT/args.catalog).read_text())
    matcher=CatalogMatcher(name,mode,catalog)
    calibration=lock['variants'].get(name+'/'+mode+'/'+args.catalog,{})
    if calibration and any(calibration['graphs'][kind]['sha256']!=matcher.saved['graphs'][kind]['sha256'] for kind in ('audio','text')):
        return dict(status='abstain',reason='model_requires_calibration',intent=args.intent,acoustic_calls=0,action_executed=False)
    audio=pcm({'audio':args.audio});x=stt_features(audio);session=ort_session(OUT/f'{name}-audio.onnx')
    a,b=session.run(None,{'features':rearrange(x,'t f -> 1 t f'),'lengths':np.array([len(x)],dtype='int64')})
    output=(a[0],b[0]);ranked=matcher.rank(output,args.role)
    decision=matcher.resolve(output,args.role,args.intent,calibration.get('gates',{}).get(args.role),calibration.get('catalog_version'))
    return dict(decision=decision,research_diagnostics=ranked,model=name,mode=mode,
                confidence_is_probability=False,parameter_only=True,timing_verified=False)


def historical_comparison(rows):
    """Import preserved predictions, asserting identical music query identities."""
    expected={role:{(r['id'],t['value']) for r in rows if r['split']=='test' for t in r['targets'] if t['role']==role}
              for role in ('artist','song')}
    result={}
    for label,path in [
        ('Earlier fixed KWS',ROOT/'.cache/parameter-study/fixed-expanded-predictions.json'),
        ('Earlier audio enrollment, 5 speakers',ROOT/'.cache/parameter-study/prototype-expanded_5shot-predictions.json'),
        ('Earlier phoneme greedy matcher',ROOT/'.cache/parameter-study/phoneme-expanded-predictions.json'),
        ('Phonetic v2 wide phrase graph',ROOT/'.cache/phonetic-v2/wide_graph_expanded-test.json')]:
        records=json.loads(path.read_text());groups={}
        for role in ('artist','song'):
            rr=[r for r in records if r['role']==role and r['reference'] is not None]
            assert {(r['id'],r['reference']) for r in rr}==expected[role],(label,role,'query mismatch')
            groups[role]=dict(n=len(rr),correct=sum(r['prediction']==r['reference'] for r in rr),
                              novel_n=sum(r['novel'] for r in rr),novel_correct=sum(r['novel'] and r['prediction']==r['reference'] for r in rr))
        result[label]=dict(by_role=groups,path=str(path.relative_to(ROOT)),sha256=digest(path))
    return result


def report(rows,protocol):
    result=json.loads((OUT/'results.json').read_text())
    lock=json.loads((OUT/'selection-before-test.json').read_text())
    timing=json.loads((OUT/'timing.json').read_text()) if (OUT/'timing.json').exists() else None
    historical=historical_comparison(rows);dump(OUT/'historical-comparison.json',historical)
    def ratio(k,n):return f'{k}/{n} ({100*k/max(1,n):.1f}%)'
    def raw(group):return ratio(group['raw_correct'],group['reference_queries'])
    lines=['# Speech embeddings for dynamic music-name catalogs','',
           '20 September 2026. Research-only, original human recordings, random initialization.','',
           '**Recommendation: use a pronunciation-aware catalog reader for text-only additions. These new compact embeddings demonstrate the mechanism, but their failed name-quality gates prevent deployment. BC-ResNet is a promising alternative audio backbone; its fixed-word paper does not establish dynamic artist recognition.**','',
           'Entry point: [experiments/speechembedding](../../experiments/speechembedding/README.md). '+
           'Paper investigation: [BC-ResNet](BCRESNET.md). Model/dataset rationale: [research notes](RESEARCH.md).','',
           '## What was built','',
           'Two candidates share a temporal depthwise residual acoustic encoder and a separate convolutional pronunciation encoder. Each returns a normalized 96-dimensional vector. '
           'Four ordered temporal bins preserve coarse sequence order; ten fixed feature windows offer weak name localization within an intact command. '
           'The model never generates a transcript. A catalog is embedded once and queried with cosine similarity. Windows are not verified word boundaries.','',
           '`dual` learns audio/pronunciation contrastive matching. `dual_kws` adds a fixed-label keyword loss. '
           'Both use the same seed, initial weights and sampling recipe, so the keyword loss is the intended ablation. '
           'Duplicate phrases are positives rather than in-batch negatives. All acoustic and pronunciation-encoder weights start randomly. '
           'CMUdict supplies only textual pronunciations. No pretrained weights, TTS, generated commands, splicing or acoustic augmentation are used.','',
           'The study additionally compares whole-command versus window matching, one/three/five-speaker audio prototypes, and a predeclared 50:50 audio/text score where an audio prototype exists. '
           'Audio enrollment embeds whole original commands because reliable name-only timestamps are unavailable. Missing audio retains text only in the explicitly selected hybrid experiment.','',
           '## Data and evaluation boundaries','',
           f"The exact earlier parameter-study corpus is frozen again: {protocol['counts']['train']:,} train, {protocol['counts']['enroll']:,} enrollment, {protocol['counts']['dev']:,} development, {protocol['counts']['test']:,} test recordings. "
           'There are 166 artist and 99 song training targets, versus 6,341 number targets. '
           'Test music targets are 77 artists and 45 songs; reserved unseen-name targets are 16 artists and only one song. '
           'These are curated public-source subsets, not full published-dataset benchmark scores.','',
           'Training/enrollment speakers are disjoint from development and test, and file hashes are deduplicated. '
           'Enrollment may use training speakers; reserved-name enrollment recordings never enter gradient updates. '
           'The prior test recordings have already been reported in earlier studies, so this is fresh evaluation on a preserved holdout, not a newly blind test. '
           'No test labels or errors select these weights, thresholds or catalog lists.','',
           'Slot roles are supplied from annotations. This measures conditional name/parameter retrieval, not automatic artist-versus-title parsing or complete-command accuracy. '
           'The data contain artist/song names, not a validated collection of custom playlist names. Small “playlist” catalogs are offline fixtures selected by training frequency. '
           'There are no Spotify requests, user-library accesses, playback actions or Home Assistant operations.','',
           f"The development-selected text reader is **{lock['preferred_text_variant']}**. "
           'Each model/catalog/role needs at least 95% development precision over at least ten emissions and at most 1% unknown false accepts. '
           'Failure disables that role. These thresholds are operating-point estimates, not statistical guarantees; cosine scores are not probabilities. '
           'Weights, graphs, catalogs and thresholds were written to `selection-before-test.json` before test inference.','',
           '## Same-query comparison: expanded catalogs','',
           'All rows below score the same 77 artist and 45 song references against 127 artist / 109 song catalog entries. '
           'Historical rows are imported from preserved predictions and their recording/reference identities are checked. '
           'Phonetic v2 had additional real-English training, so it is a practical comparator rather than a controlled encoder-only ablation.','',
           '| Reader | Artists: raw exact | Songs: raw exact | Accepted artist / song correct |',
           '| --- | ---: | ---: | ---: |']
    for label,record in historical.items():
        a=record['by_role']['artist'];s=record['by_role']['song']
        lines.append(f"| {label} (preserved) | {ratio(a['correct'],a['n'])} | {ratio(s['correct'],s['n'])} | See preserved gates; name roles disabled |")
    for name,modes in MODES.items():
        for mode in modes:
            v=result['variants'][name+'/'+mode+'/expanded']['by_role'];a=v['artist'];s=v['song']
            lines.append(f"| {name} / {mode} | {raw(a)} | {raw(s)} | {a['accepted_correct']} / {s['accepted_correct']} |")
    preferred=result['variants'][lock['preferred_text_variant']]
    lines+=['','Top-five retrieval is diagnostic only; returning a candidate list does not establish a correct accepted command.','',
            '| Preferred reader role | Top 1 | Top 5 | Representable references | Unknown false accepts / trials, gated |',
            '| --- | ---: | ---: | ---: | ---: |']
    for role in ('artist','song','device'):
        v=preferred['by_role'][role]
        lines.append(f"| {role} | {raw(v)} | {ratio(v['top5_correct'],v['reference_queries'])} | {v['represented_queries']}/{v['reference_queries']} | {v['unknown_false_accepts']}/{v['unknown_trials']} |")
    lines+=['','A zero false-accept count alongside zero recall means the role is disabled; it is not a successful recognizer. '
            'Unknown trials include ordinary speech and commands for the other music role. The same background recordings are repeated across roles and are not independent new recordings.','',
            '## Adding unseen names and optional recordings','',
            '| Reader, expanded catalog | Reserved artists | Reserved songs | Accepted correct, both roles |',
            '| --- | ---: | ---: | ---: |']
    for key in ['dual/text_local/expanded','dual_kws/text_local/expanded','dual_kws/audio_1/expanded','dual_kws/audio_3/expanded','dual_kws/audio_5/expanded','dual_kws/hybrid_3/expanded']:
        v=result['variants'][key]['novel'];a=v['artist'];s=v['song']
        lines.append(f"| {key} | {raw(a)} | {raw(s)} | {a['accepted_correct']+s['accepted_correct']} |")
    lines+=['','These names were excluded from all gradient training, including transcript matches. Adding catalog text changes the text vectors, not acoustic classes. '
            'An unavailable dictionary pronunciation remains unavailable unless explicitly supplied in the existing phone vocabulary. '
            'Audio-only matching can represent names with recordings but cannot create prototypes from text. One held-out song is far too little evidence for generalization.','']
    lines+=['The reserved-name set above is distinct from naturally unseen names that occur only outside training. The preferred text reader gives:','',
            '| Role | Names seen in training: raw exact queries | All names unseen in training: raw exact queries |',
            '| --- | ---: | ---: |']
    for role in ('artist','song'):
        v=preferred['by_training_coverage'][role]
        lines.append(f"| {role} | {raw(v['seen'])} | {raw(v['unseen'])} |")
    lines.append('')
    enrollments=json.loads((OUT/'dual_kws-enrollment.json').read_text())
    catalog=json.loads((DATA/'catalogs.json').read_text())
    lines+=['| Enrollment request | Artists with zero / fewer than requested speakers | Songs with zero / fewer than requested speakers |',
            '| --- | ---: | ---: |']
    for shots in (1,3,5):
        cells=[]
        for role in ('artist','song'):
            counts=[enrollments[str(shots)].get(role+'::'+v,{}).get('speakers',0) for v in catalog['expanded'][role]]
            cells.append(f"{sum(n==0 for n in counts)} / {sum(n<shots for n in counts)} of {len(counts)}")
        lines.append(f"| {shots} | {cells[0]} | {cells[1]} |")
    lines+=['','## Smaller playlist fixtures','',
            'These lists are fixed before test from training frequency. An omitted name is an unknown-name trial; the evaluator never inserts the answer into a per-query shortlist.','',
            '| Catalog / role, preferred text mode | Catalog entries | References inside catalog | Raw correct / inside catalog | Gated absent-name false accepts |',
            '| --- | ---: | ---: | ---: | ---: |']
    name,mode,_=lock['preferred_text_variant'].split('/')
    for size in ('playlist10','playlist25','fixed','expanded'):
        v=result['variants'][name+'/'+mode+'/'+size]['by_role']
        for role in ('artist','song'):
            s=v[role]
            lines.append(f"| {size} / {role} | {len(catalog[size][role])} | {s['in_catalog_queries']}/{s['reference_queries']} | {ratio(s['raw_correct'],s['in_catalog_queries'])} | {s['absent_name_false_accepts']}/{s['absent_name_trials']} |")
    lines+=['','## Other roles remain separate','',
            '| Preferred reader group | Raw exact | Accepted correct |', '| --- | ---: | ---: |']
    for group in ('isolated_number','natural_number','word','device'):
        s=preferred['by_group'][group]
        lines.append(f"| {group} | {raw(s)} | {s['accepted_correct']}/{s['reference_queries']} |")
    lines+=['','Number labels are included to expose corpus imbalance, not to claim a new numeric grammar. The reader does not implement connected volume or brightness validation. '
            'Room/place labels remain device-name proxies. Artist, song, number and room results must not be pooled into one headline accuracy.','',
            '## Model size and measured cost','',
            '| Model | Total parameters | Audio / text ONNX bytes | Selected epoch / epochs run | Fit seconds |',
            '| --- | ---: | ---: | ---: | ---: |']
    for model in MODES:
        v=metadata(model)
        lines.append(f"| {model} | {v['total_parameters']:,} | {v['graphs']['audio']['bytes']:,} / {v['graphs']['text']['bytes']:,} | {v['epoch']} / {v['fitting']['epochs_run']} | {v['fitting']['seconds']:.1f} |")
    if timing:
        lines+=['','Timing uses identical preloaded PCM from 48 original music commands, CPU ONNX with two intra-op threads, one inter-op thread and one BLAS thread. '
                'One warmup pass and three measured passes include frontend, acoustic model and matching; disk, speech duration, session/catalog construction and the default 400 ms endpoint wait are excluded. '
                'These are Windows workstation measurements, not Pi or live-microphone results.','',
                '| Reader | p50 / p95 ms | Session load ms | Catalog construction ms |', '| --- | ---: | ---: | ---: |']
        for key,v in timing['models'].items():
            lines.append(f"| {key} | {v['p50_ms']:.2f} / {v['p95_ms']:.2f} | {v['session_load_ms']:.2f} | {v['catalog_compile_ms']:.2f} |")
        lines+=['','Text graph inference occurs at catalog construction, not on every utterance. Reported construction includes pronunciation lookup and text-graph session setup. '
                'Separate catalog-only scaling uses generated dictionary **text** distractors, not generated speech or a large-catalog accuracy benchmark.','',
                '| Names | Matching p50 / p95 ms | Catalog construction ms | Vector matrix bytes |', '| --- | ---: | ---: | ---: |']
        for size,v in timing['scaling']['catalogs'].items():
            lines.append(f"| {size} | {v['match_p50_ms']:.3f} / {v['match_p95_ms']:.3f} | {v['compile_ms']:.1f} | {v['matrix_bytes']:,} |")
    lines+=['','## Interpretation and next experiment','',
            'This study does not identify a universally best model. It tests a compact, single-seed, from-scratch recipe with very sparse name supervision. '
            'Its acoustic and text spaces can be compared, but simple cosine retrieval remains vulnerable to surrounding command words, accent variation, phonetic neighbors and insufficient training coverage. '
            'Those are plausible failure contributors, not experimentally separated causal estimates. Dictionary/enrollment coverage and per-query outputs are saved so data gaps remain visible.','',
            'For text-only music catalogs, the next architecture to test is a compact phonetic or audio/text keyword reader with a sequence verifier. '
            'Use a cheap embedding search to retrieve candidates, then verify the candidate phone sequence against local acoustic frames. '
            'BC-ResNet can be tested as an alternative acoustic backbone inside that system. A fixed KWS head remains useful for stable command words, while VCM retains intent ownership. '
            'Optional original recordings can provide additional pronunciation evidence for unusual names. Neither pooled word accuracy nor a fixed-word paper result establishes open-name recognition.','',
            'The next version should first add original connected artist/song/playlist recordings from new speakers, including confusable names, accents, background speech and names absent from the active catalog. '
            'Do not tune this version on its test errors. Evaluate a new, preserved protocol with several seeds and enough positive/unknown trials per role.','',
            '## Verification and artifacts','',
            'ONNX parity covers variable durations, batch sizes and padding. Original file hashes, preprocessing parity, duplicate-positive contrastive loss, homophones, explicit pronunciations, stable IDs, rejected intent, invalid thresholds and catalog-version invalidation are checked separately from acoustic accuracy. '
            'The adapter returns research diagnostics and a typed decision; failed gates abstain. Its caller must supply an already accepted MEDIA_PLAY intent. No device actions are executed.','',
            'An initial frontend check failed a tight tolerance with the default 24 BLAS threads (maximum absolute difference 0.0000019223). '
            'The diagnostic reproduced exact equality with the cached training features at one BLAS thread. The entry point now pins one BLAS thread for calibration, test, timing and file queries; '
            'the original feature cache and trained weights were not changed. See `feature-parity-diagnostic.json` and the preserved initial check log.','',
            'Candidate checkpoints, ONNX graphs, fitting histories, development/test predictions, enrollment IDs, selection lock, checks and timing are in [.cache/speechembedding-v1](../../.cache/speechembedding-v1/). '
            'The frozen manifest, protocol and catalogs are in [data/speechembedding-v1](../../data/speechembedding-v1/). '
            'Implementation is registered as `runtime: false` in the shared source map. Installed VCM/STT files, configuration and previous study results remain hash-identical.','']
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    dump(OUT/'final-preservation.json',preserved_check(protocol))
    print('Report:',REPORT/'REPORT.md',flush=True)
