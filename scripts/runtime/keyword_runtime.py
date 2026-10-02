"""Research CPU command path: ordered fixed cues -> accepted VCM -> typed slots."""
import json
import re
import numpy as np
from einops import rearrange
from .keyword_data import ROOT, DATA, CACHE, REPORTS, VOCAB, HEADS, FRONTEND, digest
from .audio import features
from .levels import level_parameter, NUMBER_WORDS, spoken_number, percentage
from .routing import COLORS, validate_arguments
from .synthetic_data import catalog

LIGHT=set('brightness bright brighter brighten dim dimmer dimming darker darken light lights lighting lamp lamps'.split())
UP=set('up increase raise higher more brighten brighter louder boost'.split())
DOWN=set('down decrease reduce lower less dim dimmer dimming darker darken quieter softer'.split())
VOLUME=set('volume sound audio louder quieter softer'.split())


def softmax(x):
    p=np.exp(x-np.max(x,axis=-1,keepdims=True));return p/np.sum(p,axis=-1,keepdims=True)


def ctc_cues(logits,length):
    probabilities=softmax(logits[:length]); ids=np.argmax(probabilities,axis=-1)
    cues=[];previous=-1
    for i,index in enumerate(ids):
        index=int(index)
        if index and index!=previous:
            cues.append({'word':VOCAB[index],'confidence':float(probabilities[i,index]),'start_ms':i*40.,'end_ms':(i+1)*40.})
        elif index and cues:
            cues[-1]['end_ms']=(i+1)*40.;cues[-1]['confidence']=max(cues[-1]['confidence'],float(probabilities[i,index]))
        previous=index
    return cues


def cue_family(cues):
    words={c['word'] for c in cues};light=bool(words&LIGHT)
    if words&{'not','never','cancel','instead'}:return 'UNKNOWN'
    if {'on','off'}<=words or (words&UP and words&DOWN):return 'UNKNOWN'
    if words&{'resume','continue'}:return 'MEDIA_RESUME'
    if words&{'play','start','playing'} or ('music' in words and 'on' in words):return 'MEDIA_PLAY'
    if light and words&COLORS:return 'COLOR_SET'
    if light and words&{'on','off'}:return 'POWER_ON' if 'on' in words else 'POWER_OFF'
    if words&VOLUME:return 'VOLUME'
    if light and (words&(UP|DOWN) or 'brightness' in words):return 'BRIGHTNESS'
    if words&{'date','day','today'}:return 'DATE_QUERY'
    if 'time' in words:return 'TIME_QUERY'
    if words&(UP|DOWN):return 'VOLUME'
    return None


def numeric_cues(logits,length,cues,family):
    """Score the finite 0-100 word grammar in order using the SAME acoustic posteriors."""
    words=[c['word'] for c in cues];numbers=[i for i,w in enumerate(words) if w in NUMBER_WORDS]
    anchors=[i for i,w in enumerate(words) if w in (LIGHT if family=='BRIGHTNESS' else VOLUME)]
    if not anchors or (numbers and numbers[0]<anchors[0]):return cues,None
    if set(words)&{'minus','negative','point','decimal','half','quarter','or','not','and','thousand','million'}:return cues,None
    if numbers:
        try:percentage(' '.join(words))
        except ValueError:return cues,None
        first,last=numbers[0],numbers[-1]+1
    else:
        boundaries=[i for i,w in enumerate(words) if w in ('to','by') and i>anchors[0]]
        if not boundaries:return cues,None
        first=boundaries[-1]+1
        percent=[i for i,w in enumerate(words) if w=='percent' and i>=first]
        last=percent[0] if percent else len(words)
    prefix=words[:first];suffix=words[last:]
    if any(w not in NUMBER_WORDS and w not in ('<other>',) for w in words[first:last]):return cues,None
    sequences=[prefix+spoken_number(n).split()+suffix for n in range(101)]+[prefix+suffix]
    indices={w:i for i,w in enumerate(VOCAB)};sizes=np.array([2*len(s)+1 for s in sequences]);width=int(max(sizes))
    labels=np.zeros((len(sequences),width),dtype=np.int64)
    for j,sequence in enumerate(sequences):labels[j,1:2*len(sequence)+1:2]=[indices[w] for w in sequence]
    valid=np.arange(width)[None,:]<sizes[:,None]
    skip=(labels[:,2:]!=0)&(labels[:,2:]!=labels[:,:-2])
    alpha=np.full(labels.shape,-np.inf,dtype=np.float64);alpha[:,0]=0.
    logp=logits[:length]-np.logaddexp.reduce(logits[:length],axis=-1,keepdims=True)
    for frame in logp:
        update=alpha.copy();update[:,1:]=np.logaddexp(update[:,1:],alpha[:,:-1])
        update[:,2:]=np.logaddexp(update[:,2:],np.where(skip,alpha[:,:-2],-np.inf))
        alpha=np.where(valid,update+frame[labels],-np.inf)
    ends=sizes-1;scores=np.logaddexp(alpha[np.arange(len(sequences)),ends],alpha[np.arange(len(sequences)),np.maximum(ends-1,0)])
    best=int(np.argmax(scores));prob=softmax(scores);trace={'value':best if best<101 else None,'confidence':float(prob[best]),'null_won':best==101,
        'decoder':'exact CTC sum over ordered finite-number sequences; no new model or fallback'}
    if best==101:return cues,trace
    begin=cues[first]['start_ms'] if first<len(cues) else cues[-1]['end_ms']
    end=cues[last]['start_ms'] if last<len(cues) else length*40.
    end=max(begin+40.,end);value_words=spoken_number(best).split()
    replacement=[{'word':w,'confidence':float(prob[best]),'start_ms':begin+(end-begin)*i/len(value_words),'end_ms':begin+(end-begin)*(i+1)/len(value_words),'source':'finite_numeric_graph'} for i,w in enumerate(value_words)]
    return cues[:first]+replacement+cues[last:],trace


def finite_slots(family,cues):
    """The selected family cannot be changed here. Keep lexical order and rejects."""
    words=[c['word'] for c in cues]; wordset=set(words)
    text=' '.join('filler' if w=='<other>' else w for w in words)
    if wordset&{'not','never','cancel','instead','unless','if','when','then','or'}:raise ValueError('conflicting_or_conditional_cues')
    args={};operation='NONE';amount='NONE';color='NONE';action=None
    if family in ('POWER_ON','POWER_OFF','BRIGHTNESS','COLOR_SET'):
        if not wordset&LIGHT:raise ValueError('missing_light_context')
        args['target']={'id':'example.main_lights'}
    if family.startswith('POWER_'):
        wanted='on' if family=='POWER_ON' else 'off';opposite='off' if wanted=='on' else 'on'
        if wanted not in wordset or opposite in wordset:raise ValueError('power_cue_conflict')
        verbs=[i for i,w in enumerate(words) if w in {'turn','switch'}]
        if verbs and min(verbs)>words.index(wanted):raise ValueError('power_cues_out_of_order')
        args['on']=wanted=='on'
    elif family in ('BRIGHTNESS','VOLUME'):
        opposite=VOLUME if family=='BRIGHTNESS' else LIGHT
        if wordset&opposite:raise ValueError('opposite_property')
        anchors=[i for i,w in enumerate(words) if w in (LIGHT if family=='BRIGHTNESS' else VOLUME)]
        numbers=[i for i,w in enumerate(words) if w in NUMBER_WORDS]
        if numbers and anchors and min(numbers)<min(anchors):raise ValueError('number_precedes_property')
        if not anchors and not wordset&(UP|DOWN):raise ValueError('missing_level_context')
        args.update(level_parameter(text))
        if 'percent' in args:operation='SET';amount=str(args['percent']);action='SET_'+family
        else:
            delta=args['delta_percentage_points'];operation='INCREASE' if delta>0 else 'DECREASE';amount=str(abs(delta));action='ADJUST_'+family
    elif family=='COLOR_SET':
        colors=wordset&COLORS
        if len(colors)!=1:raise ValueError('missing_or_conflicting_color')
        color=next(iter(colors));args['color']=color
    elif family=='MEDIA_PLAY':
        if not wordset&{'play','start','playing','music'}:raise ValueError('missing_play_cue')
        if wordset&{'resume','continue','pause','stop'}:raise ValueError('conflicting_play_cue')
    elif family=='MEDIA_RESUME':
        if not wordset&{'resume','continue'}:raise ValueError('missing_resume_cue')
    elif family=='TIME_QUERY':
        if 'time' not in wordset:raise ValueError('missing_time_cue')
    elif family=='DATE_QUERY':
        if not wordset&{'date','day','today'}:raise ValueError('missing_date_cue')
    else:raise ValueError('unknown_intent')
    if wordset&{'after','before','until'} or (wordset&{'minutes','hours','seconds'} and family not in ('TIME_QUERY','DATE_QUERY')):
        raise ValueError('scheduled_command')
    result=validate_arguments(family,args,catalog(),action=action)
    if result['status']!='accepted':raise ValueError(result.get('reason','typed_validation'))
    important=[c['confidence'] for c in cues if c['word'] in (NUMBER_WORDS|COLORS|UP|DOWN|{'on','off','resume','continue','play','time','date','day'})]
    confidence=float(np.exp(np.mean(np.log(np.clip(important,.000001,1))))) if important else 0.
    return {'intent':family,'ready':'READY','operation':operation,'amount':amount,'color':color},result,confidence


def decode_record(intent,ready,ctc,length,evidence=0.,intent_thresholds=None):
    cues=ctc_cues(ctc,length);hint=cue_family(cues)
    adjusted=np.array(intent,copy=True)
    if hint is not None:adjusted[HEADS['intent'].index(hint)]+=evidence
    p=softmax(adjusted);pr=softmax(ready);index=int(np.argmax(p));family=HEADS['intent'][index]
    outcome={'family':family,'family_confidence':float(p[index]),'ready_confidence':float(pr[1]),'cues':cues,
        'cue_family':hint,'intent_margin':float(np.sort(p)[-1]-np.sort(p)[-2]),'labels':None,'command':None,'score':0.,'valid':False}
    if intent_thresholds is not None and (family=='UNKNOWN' or p[index]<intent_thresholds[family]):
        outcome['reason']='intent_confidence_gate';return outcome
    if family in ('BRIGHTNESS','VOLUME'):
        cues,trace=numeric_cues(ctc,length,cues,family)
        outcome.update(cues=cues,numeric_graph=trace)
    try:
        labels,command,slot_confidence=finite_slots(family,cues)
        outcome.update(labels=labels,command=command,slot_confidence=slot_confidence,score=float(p[index]*pr[1]*slot_confidence),valid=bool(pr[1]>=.5))
    except ValueError as exc:outcome['reason']=str(exc)
    return outcome


class KeywordRecognizer:
    def __init__(self,mode='mixed',threads=2):
        import onnxruntime as ort
        self.mode=mode; self.bundle=json.loads((REPORTS[mode]/'selection-before-test.json').read_text())
        if digest(CACHE/mode/'model.onnx')!=self.bundle['model_sha256']:raise ValueError('Acoustic graph changed')
        if digest(DATA/'protocol.json')!=self.bundle['protocol_sha256']:raise ValueError('Protocol changed')
        if VOCAB!=self.bundle['vocabulary']:raise ValueError('Keyword vocabulary changed')
        for name,value in self.bundle['implementation_sha256'].items():
            if digest(ROOT/name)!=value:raise ValueError('Frozen implementation changed: '+name)
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(CACHE/mode/'model.onnx'),options,providers=['CPUExecutionProvider'])
        self.intent_session=None
        if self.bundle['policy'].get('intent_model') in ('micro','cnn'):
            path=ROOT/self.bundle['intent_model_path']
            if digest(path)!=self.bundle['intent_model_sha256']:raise ValueError('Intent graph changed')
            self.intent_session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])

    def predict(self,pcm,offset_ms=0.):
        pcm=np.asarray(pcm,dtype=np.float32)
        if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('Expected finite nonempty mono PCM')
        x=features(pcm,FRONTEND)
        feed={'features':rearrange(x,'c t -> 1 c t'),'lengths':np.array([x.shape[1]],dtype=np.int64)}
        if self.intent_session is not None:
            outputs=self.intent_session.run(None,feed);intent,ready=outputs[:2]
            p=softmax(intent[0]);index=int(np.argmax(p));family=HEADS['intent'][index]
            if family=='UNKNOWN' or p[index]<self.bundle['policy']['intent_thresholds'][family]:
                return {'status':'rejected','reason':'intent_confidence_gate','intent':family,'intent_accepted':False,'confidence':float(p[index]),'keyword_cues':[],
                    'parameter_invoked':False,'keyword_reader_invoked':False,'experimental':True,'device_actions_executed':False}
            _,_,ctc,length=self.session.run(None,feed)
        else:intent,ready,ctc,length=self.session.run(None,feed)
        decoded=decode_record(intent[0],ready[0],ctc[0],int(length[0]),self.bundle['policy']['evidence'],self.bundle['policy']['intent_thresholds'])
        key=decoded['family']+'/'+decoded['labels']['operation'] if decoded['labels'] and decoded['family'] in ('BRIGHTNESS','VOLUME') else decoded['family']
        threshold=self.bundle['policy']['thresholds'].get(key,1.1)
        accepted=decoded['valid'] and decoded['score']>=threshold
        intent_accepted=decoded['family']!='UNKNOWN' and decoded['family_confidence']>=self.bundle['policy']['intent_thresholds'][decoded['family']]
        result=dict(decoded['command']) if accepted else {'status':'rejected','reason':decoded.get('reason','confidence_gate')}
        result.update(intent=decoded['family'],intent_accepted=bool(intent_accepted),
            confidence=decoded['score'],keyword_cues=[{**c,'start_ms':c['start_ms']+offset_ms,'end_ms':c['end_ms']+offset_ms} for c in decoded['cues']],
            cue_timing='CTC emission intervals; not verified word boundaries',experimental=True,device_actions_executed=False,parameter_invoked=bool(intent_accepted and decoded['family'] in ('BRIGHTNESS','VOLUME','COLOR_SET')))
        return result


class KeywordPipeline:
    def __init__(self,mode='mixed'):
        from .config import load_config
        from .pipeline import EnergyVAD
        self.config=load_config(ROOT/'pipeline.toml');self.config['runtime']['vad']=True
        self.recognizer=KeywordRecognizer(mode);self.vad=EnergyVAD(self.config['audio'])

    def utterance(self,pcm,timeout=False):
        if timeout:return {'status':'rejected','reason':'utterance_timeout','parameter_invoked':False}
        return self.recognizer.predict(pcm)

    def process(self,pcm,wake_end_ms=0.):
        pcm=np.asarray(pcm,dtype=np.float32)
        if pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all():raise ValueError('Expected finite nonempty mono PCM')
        if not np.isfinite(wake_end_ms) or wake_end_ms<0 or round(wake_end_ms*16)>=len(pcm):raise ValueError('Wake end must lie within the recording')
        start=round(wake_end_ms*16);remaining=pcm[start:]
        self.vad.reset();self.vad.noise=.003;step=round(16000*self.config['audio']['block_ms']/1000)
        padded=np.pad(remaining,(0,self.vad.silence_limit+step));results=[]
        for offset in range(0,len(padded),step):
            item=self.vad.push(padded[offset:offset+step])
            if item is not None:
                speech,timeout=item;first=max(0,offset+len(padded[offset:offset+step])-len(speech));absolute_ms=(start+first)/16
                result={'status':'rejected','reason':'utterance_timeout'} if timeout else self.recognizer.predict(speech,absolute_ms)
                result.update(speech_start_ms=absolute_ms,speech_end_ms=(start+offset+len(padded[offset:offset+step]))/16,wake_end_ms=wake_end_ms,
                    speech_timing='VAD segment including pre-roll and endpoint silence; not aligned word boundaries')
                results.append(result)
        return results or [{'status':'rejected','reason':'no_speech','wake_end_ms':wake_end_ms}]
