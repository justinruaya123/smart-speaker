"""Offline annotation and preparation of the pinned HF recording pool only.

Text here is supplied supervision, never a runtime transcript or decoder output.
Ambiguous metadata remains visible but is excluded from exact-command metrics.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.pcv_updated_data import HERE, CACHE, REVISION
from scripts.train.pcv_public_data import sha, save, read_rows, _words, audit_split_rows
from scripts.train.pcv_three_engine import immutable
from scripts.test.pcv_public_qualification import verify_bindings
from scripts.runtime.pcv_schema import (INTENTS, DEFAULT_CUSTOM, baseline_cases, extended_cases,
    command, canonical, normalize_text, compile_inventory)
from vcm.catalog import number_words, PHONES


SMALL = dict(zip(('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(), range(20)))
SMALL.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20,100,10))))
NUMBER_TOKENS = set(SMALL) | {'hundred','thousand','million','billion','and','minus','negative'}


def integer_words(n):
    if n < 0: return 'minus '+integer_words(-n)
    if n <= 100: return number_words(n)
    for scale, name in [(10**9,'billion'),(10**6,'million'),(1000,'thousand'),(100,'hundred')]:
        if n >= scale:
            return integer_words(n//scale)+' '+name+(' '+integer_words(n%scale) if n%scale else '')


def spoken(text):
    text=text.lower().replace('’',"'")
    text=re.sub(r'\b([ap])\.?\s*m\.?\b', lambda m:m[1]+' m', text)
    text=re.sub(r'(?<=\d)([ap])m\b',lambda m:' '+m[1]+' m',text)
    text=re.sub(r'\b(\d{1,2}):00\b',lambda m:integer_words(int(m[1])),text)
    text=re.sub(r'\b(\d{1,2}):(\d{2})\b',lambda m:integer_words(int(m[1]))+' '+('oh ' if 0<int(m[2])<10 else '')+integer_words(int(m[2])),text)
    text=text.replace('%',' percent').replace('°',' degrees ')
    text=re.sub(r'(?<!\w)-?\d+',lambda m:integer_words(int(m[0])),text)
    return normalize_text(text)


def number_value(text):
    parts=text.split(); sign=-1 if parts and parts[0] in ('minus','negative') else 1
    if sign<0: parts=parts[1:]
    total=current=0
    for part in parts:
        if part=='and': continue
        if part in SMALL: current+=SMALL[part]
        elif part=='hundred': current=max(1,current)*100
        elif part in {'thousand','million','billion'}:
            scale={'thousand':1000,'million':10**6,'billion':10**9}[part]
            total+=max(1,current)*scale;current=0
        else: raise ValueError('Not a finite number')
    if not parts: raise ValueError('Missing number')
    return sign*(total+current)


NUM = r'(?:minus |negative )?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?: (?:and |hundred |thousand )?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))*'
BASE = {spoken(r['text']):r['command'] for r in baseline_cases()+extended_cases()}
GENERIC = {
 'PLAY_MUSIC':['play','play the music','start the music','put on the music','play a song','play song','play songs','i want to hear song','i want to hear music','play me some music','play some songs'],
 'WEATHER':['check weather','current weather','current weather forecast','how is the weather','how is the weather today','what is the weather','what is the weather today','what is the weather like','weather forecast','weather report','tell me the current weather','what is the current weather','what is the weather outside','is it raining outside','is it snowing now','is weather cloudy'],
 'TIME':['current time','time now','what is the current time','what is the time','what is the time now','what time is it now','what time is it right now',"what's the time", "what's the time of day right now",'tell me what time is it','give me the time','tell me the current time','tell me the time it is','what is the current time right now'],
 'PAUSE':['pause music','pause the music','pause this song','pause the audio','pause the audiobook','pause the book'],
 'STOP':['stop music','stop playing music','stop song','stop the music','stop it','stop the audio','turn off the music'],
 'NEXT':['next','skip','skip this song','skip to the next song','play the next song','continue to next song','move to the next song in the list'],
 'LIST_REMINDERS':['what are my reminders','show me my reminders','show pending reminders','tell me all the pending reminders','check my reminders','tell me the pending reminders','give me a list of all my reminders','read me my next reminders','read the names of my lists of reminders'],
}
for intent, phrases in GENERIC.items():
    for text in phrases: BASE[text]=command(intent)
for operation,phrases in {
 'up':['louder','make it louder','make the music louder','turn it up','turn sound up','turn the sound up','turn up the volume','turn volume up','increase volume','increase the sound volume','speak louder','get louder','adjust media volume up','increase the volume a bit','turn it up a little','turn up your volume','increase the volume of the speaker'],
 'down':['quieter','make the music softer','turn down the volume','turn sound down','volume lower','decrease the volume','decrease volume','lower volume','make the volume lower','reduce the volume','speak quieter to me','make it a little quieter','lower your volume','lower music volume','reduce the speaker volume'],
}.items():
    for text in phrases: BASE[text]=command('VOLUME',operation=operation)
BASE['remind me to study for my exam']=command('CREATE_REMINDER',task_id='task.study')


def polite(text):
    text=re.sub(r'^(?:hey )?(?:olly|alexa|ok google) ', '', text)
    for _ in range(3):
        text=re.sub(r'^(?:please |can you please |could you please |could you |can you |help me |do me a favor and )','',text)
    return re.sub(r' (?:please|for me)$','',text)


def light_target(text):
    text=text.strip()
    text=re.sub(r'^(?:the |all |my )','',text)
    if text in ('light','lights','lamp','lamps','lighting','brightness','light brightness','brightness level','color','colour'):
        return 'light.all'
    for light in DEFAULT_CUSTOM['lights']:
        names=[light['name'],*light.get('aliases',[]),light['id'].split('.')[1]]
        if text in names or re.fullmatch(r'(?:lights?|lamps?) (?:in|for) (?:the )?'+light['id'].split('.')[1],text):
            return light['id']
    return None


def annotate(row):
    text=spoken(row.get('transcript','')); t=polite(text)
    expected=None;status='ambiguous';reason='unresolved_reference';conflict=False
    coarse=row.get('command','OUT_OF_SCOPE');coarse={'VOLUME_UP':'VOLUME','VOLUME_DOWN':'VOLUME'}.get(coarse,coarse)
    if row.get('published_split')=='numerals':
        return dict(text=text,expected=None,reference_status='auxiliary',reason='isolated_numeral',slot_metadata_conflict=False)
    if 'not understood' in row.get('whisper_check',''):
        reason='publisher_unintelligible_not_verified_OOV'
    elif row.get('out_of_scope') or coarse=='OUT_OF_SCOPE':
        status='unsupported';reason='publisher_out_of_scope'
    elif t in BASE:
        expected=BASE[t];status='supported' if expected else 'unsupported';reason='finite_definition'
    elif re.search(r'\b(?:and then|and tell|confirm|tomorrow|every|next week|yesterday|tonight)\b',t):
        status='unsupported';reason='extra_action_or_unrepresented_date'
    elif coarse in {'CALL','MESSAGE','PLAY_MUSIC','CREATE_REMINDER'}:
        if coarse=='CALL' and re.fullmatch(r'call (?:home|mom|mother)',t):
            expected=command('CALL',contact_id='contact.home' if t=='call home' else 'contact.mom')
        elif coarse=='MESSAGE' and t in ('send message','send a message','message','send my message'):
            expected=command('MESSAGE')
        status='supported' if expected else 'unsupported';reason='finite_name_or_content_scope'
    elif coarse in {'LIGHT_ON','LIGHT_OFF'}:
        if any(x in t.split() for x in ['brightness','color','green','red','blue','twenty','fifty','seventy']):
            reason='family_slot_conflict'
        else:
            op='on' if coarse=='LIGHT_ON' else 'off'
            m=re.fullmatch(r'(?:turn|switch|power|shut|put) '+op+r' (.+)',t) or re.fullmatch(r'(?:turn |switch |power |make |get )?(.+?)(?: go| turn| turned)? '+op,t)
            if t in ('lights out','lights out please'):m=re.fullmatch(r'(.+) out',t)
            if m:
                target=light_target(m[1]);expected=command(coarse,light_id=target) if target else None
                status='supported' if target else 'unsupported';reason='explicit_light_target'
    elif coarse=='TIMER':
        m=re.fullmatch(r'(?:timer|(?:set|start) (?:a |the )?timer(?: for)?|countdown for) (.+)',t)
        if m:
            parts=list(re.finditer(r'('+NUM+r') (seconds?|minutes?|hours?)',m[1]))
            remainder=re.sub(r'('+NUM+r') (seconds?|minutes?|hours?)','',m[1]).replace('and','').strip()
            if parts and not remainder:
                seconds=sum(number_value(p[1])*({'second':1,'minute':60,'hour':3600}[p[2].rstrip('s')]) for p in parts)
                if 1<=seconds<=86400:
                    expected=command('TIMER',duration_seconds=seconds);status='supported';reason='explicit_complete_duration'
                    slot=spoken(row.get('slot_value',''))
                    values=list(re.finditer(r'('+NUM+r') (seconds?|minutes?|hours?)',slot))
                    if values:
                        old=sum(number_value(p[1])*({'second':1,'minute':60,'hour':3600}[p[2].rstrip('s')]) for p in values)
                        conflict=seconds!=old
            if status!='supported':reason='ambiguous_or_invalid_duration'
    elif coarse=='ALARM':
        m=re.fullmatch(r'(?:alarm|alarm me at|alarm set|wake me up at|(?:set|create|put on) (?:a |an |the |my )?alarm(?: clock)?(?: for| at)?|new alarm for|i need an alarm for) ('+NUM+r') ([ap]) m',t)
        if m:
            hour=number_value(m[1])
            if 1<=hour<=12:
                expected=command('ALARM',time=f'{hour%12+(12 if m[2]=="p" else 0):02d}:00',timezone='Asia/Manila');status='supported';reason='explicit_clock_time'
    elif coarse=='TEMPERATURE':
        m=re.fullmatch(r'(?:temperature|(?:set|change|adjust|make) (?:the |room |the room )?(?:temperature|thermostat)(?: to)?|set it to) ('+NUM+r')(?: degrees?(?: celsius)?)?',t)
        if m:
            value=number_value(m[1])
            if -50<=value<=60:
                expected=command('TEMPERATURE',degrees=value,unit='C');status='supported';reason='explicit_temperature'
            else:status='unsupported';reason='temperature_outside_declared_unit_bounds'
    elif coarse in {'BRIGHTNESS','VOLUME'}:
        if ' by ' in t:
            status='unsupported';reason='unrepresented_relative_amount'
        else:
            m=re.fullmatch(r'(?:(?:set|adjust|change|dim|lower|increase|decrease|turn) )?(.+?)(?: to| at)? ('+NUM+r')(?: percent)?',t)
            if m:
                value=number_value(m[2]);subject=m[1]
                if coarse=='VOLUME' and subject in ('volume','the volume','the volume of the music'):
                    if 0<=value<=100:expected=command('VOLUME',operation='set',percent=value)
                    status='supported' if expected else 'unsupported';reason='explicit_volume_amount'
                elif coarse=='BRIGHTNESS':
                    subject=re.sub(r'^(?:the )?brightness of (?:the )?','',subject)
                    target=light_target(subject)
                    if target and 0<=value<=100:
                        expected=command('BRIGHTNESS',light_id=target,percent=value);status='supported';reason='explicit_brightness_target'
                    elif any(w in subject.split() for w in ('brightness','light','lights','lamp')) and not re.search(r'\b(green|bin|dean|deem|deep|game|beam)\b',subject):
                        status='unsupported';reason='unknown_light_or_invalid_amount'
    elif coarse=='COLOR':
        m=re.fullmatch(r'(?:(?:set|switch|change|make|adjust|turn|color) )?(.+?)(?: to| into)? (?:the color )?(red|blue|green|yellow|white|orange|purple|pink|cyan|magenta|warm white|cool white)',t)
        if m:
            target=light_target(m[1])
            if target:expected=command('COLOR',light_id=target,color=m[2]);status='supported';reason='explicit_color_target'
            elif any(w in m[1].split() for w in ('light','lights','lamp')):status='unsupported';reason='unknown_light'
    if expected and expected['intent']!=coarse and coarse not in {'VOLUME_UP','VOLUME_DOWN'}:
        # Clear typed text wins, but the discrepancy remains an audit finding.
        conflict=True
    return dict(text=text,expected=expected,reference_status=status,reason=reason,slot_metadata_conflict=conflict)


def assign_split(row):
    original=row['published_split']
    if original in ('test','holdout'):return original
    key=str(row['source'])+'/'+str(row['speaker_id'])
    if key=='real_voice/202322013':return 'train'
    if key=='real_voice/202322013_speaker2':return 'dev'
    if key=='real_voice/202521746':return 'train'
    n=int(hashlib.sha256(('pcv-updated-v1/'+key).encode()).hexdigest()[:12],16)%100
    return 'train' if n<75 else ('dev' if n<88 else 'calibration')


def pair_identity(row):
    name=Path(row['file']).stem
    if row['source']=='SNIPS':
        match=re.search(r'snips_smart-speaker_(\d+)_(?:close|far)',name)
        if match:return 'SNIPS/'+match[1]
    # Source-qualified speaker grouping is additionally mandatory; these suffixes
    # identify alternate channels/noise versions even when file hashes differ.
    name=re.sub(r'^(?:train|test|holdout|numerals)_\d+_', '',name)
    name=re.sub(r'_(?:clean|noisy|close|far)$','',name)
    return str(row['source'])+'/'+str(row['speaker_id'])+'/'+name


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from einops import reduce
    from vcm.frontend import features, SPEC
    audit=json.loads((HERE/'audio-audit.json').read_text())
    verify_bindings(ROOT,{audit['manifest']:audit['manifest_sha256']})
    original=read_rows(ROOT/audit['manifest']);words=_words()
    definitions=baseline_cases()+extended_cases();base_ids={spoken(r['text']):r['id'] for r in baseline_cases()}
    annotated=[]; excluded=[];missing_words=Counter()
    for source in original:
        item=annotate(source);split=assign_split(source)
        row={**source,**item,'split':split,'speaker':str(source['source'])+'/'+str(source['speaker_id']),
             'pair':pair_identity(source),'synthetic':bool(source['is_synthetic']),
             'phrase_id':base_ids.get(item['text']),'intent':item['expected']['intent'] if item['expected'] else 'OOV'}
        flags=source['waveform_flags']
        bad_audio=bool(flags) and not (source['transcript_source']=='silence' and set(flags)<= {'low_rms'})
        reason='waveform_quality' if bad_audio else None
        # No label repairs are inferred from a teacher or a test prediction.
        if item['reference_status']=='ambiguous':reason='ambiguous_reference'
        phones=[];missing=[]
        for i,token in enumerate(item['text'].split()):
            ps=['EY'] if token=='a' and item['text'].split()[i+1:i+2]==['m'] else words.get(token)
            if ps:phones.extend(PHONES.index(p)+1 for p in ps)
            else:missing.append(token)
        if missing:reason=reason or 'missing_pronunciation';missing_words.update(missing)
        row['phones']=phones
        row['intent_index']=INTENTS.index(row['intent'])+1 if row['expected'] else 0
        row['target_index']=int(row['expected']['slots']['light_id']!='light.all') if row['expected'] and 'light_id' in row['expected']['slots'] else -100
        if row['reference_status']=='auxiliary':row['intent_index']=-100
        row['training_eligible']=reason is None
        row['exclusion_reason']=reason
        if reason:excluded.append({'id':row['id'],'split':split,'reason':reason,'reference_status':row['reference_status']})
        # Original test/holdout rows stay present, including uncertain references.
        if split in ('test','holdout') or not reason:
            annotated.append(row)
        if split=='train' and not reason and row['reference_status']!='auxiliary':
            definitions.append({'id':'HF-'+hashlib.sha256(item['text'].encode()).hexdigest()[:14],
                                'text':item['text'],'command':item['expected'],'tier':'hf_train'})
    integrity=audit_split_rows(annotated)
    catalog=compile_inventory(definitions,words)
    prepared=CACHE/'prepared'
    if prepared.exists():raise FileExistsError('Prepared version exists; do not overwrite frozen evidence')
    prepared.mkdir(parents=True)
    selected={r['id']:r for r in annotated}
    outputs=[]
    def one(pair):
        source,payload=pair
        pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
        if rate!=16000 or pcm.ndim!=1 or hashlib.sha256(pcm.tobytes()).hexdigest()!=source['pcm_sha256']:
            raise ValueError('Source PCM identity changed: '+source['id'])
        value=features(pcm)
        if value.shape[1]<len(source['phones'])+sum(a==b for a,b in zip(source['phones'],source['phones'][1:])):
            source['training_eligible']=False;source['exclusion_reason']='insufficient_CTC_frames'
        path=prepared/'features'/(source['pcm_sha256']+'.npy')
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,value,allow_pickle=False)
        return {**source,'features':str(path),'features_sha256':sha(path),'frames':value.shape[1]}
    acquired=json.loads((HERE/'acquisition-complete.json').read_text())
    verify_bindings(ROOT,{r['path']:r['sha256'] for r in acquired['files']})
    with ThreadPoolExecutor(max_workers=6) as pool:
        for file in acquired['files']:
            path=ROOT/file['path'];part=path.name.split('-')[0];index=0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['audio']):
                pairs=[]
                for value in batch.to_pylist():
                    identity=f'{part}/{path.stem}/{index:06d}';index+=1
                    if identity in selected:pairs.append((selected[identity],value['audio']['bytes']))
                outputs.extend(pool.map(one,pairs))
            print('PREPARED_SOURCE',path.name,'total',len(outputs),flush=True)
    # Fit normalization only on human training recordings. Isolated numbers have
    # equal utterance weight; synthetic and held-out audio never affect it.
    means=[];squares=[]
    for row in outputs:
        if row['split']=='train' and not row['synthetic'] and row['training_eligible'] and row['reference_status']!='auxiliary':
            x=np.load(row['features']).astype('float64')
            means.append(reduce(x,'c t -> c','mean'));squares.append(reduce(x*x,'c t -> c','mean'))
    mean=reduce(means,'n c -> c','mean');std=np.sqrt(np.maximum(reduce(squares,'n c -> c','mean')-mean**2,1e-6))
    save(prepared/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'frontend':SPEC,'human_training_utterances':len(means)})
    manifests={}
    for split in ('train','dev','calibration','test','holdout'):
        subset=[r for r in outputs if r['split']==split]
        path=prepared/(split+'.jsonl')
        with path.open('x',encoding='utf8') as stream:
            for row in subset:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=True,allow_nan=False)+'\n')
        manifests[split]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'rows':len(subset)}
    immutable(HERE/'catalog.json',catalog)
    immutable(HERE/'custom-slots.json',DEFAULT_CUSTOM)
    immutable(HERE/'baseline.json',baseline_cases())
    immutable(HERE/'annotation-exclusions.json',excluded)
    counts=Counter((r['split'],'synthetic' if r['synthetic'] else 'human',r['reference_status']) for r in outputs)
    coverage={}
    for split in manifests:
        for origin in ('human','synthetic'):
            rows=[r for r in outputs if r['split']==split and r['synthetic']==(origin=='synthetic')]
            coverage[split+'/'+origin]={'rows':len(rows),'exact_baseline_cells':len({r['phrase_id'] for r in rows if r['phrase_id']}),
                'positive_intents':dict(Counter(r['intent'] for r in rows if r['expected'])),
                'baseline_denominators':{b['id']:sum(r['phrase_id']==b['id'] for r in rows) for b in baseline_cases()}}
    summary={'source_revision':REVISION,'source_only':True,'fallback_used':False,'distillation_used':False,
        'test_inference_accessed':False,'integrity':integrity,'counts':{'/'.join(k):v for k,v in counts.items()},
        'exclusions':dict(Counter(r['reason'] for r in excluded)),'missing_pronunciations':dict(missing_words),
        'slot_metadata_conflicts':sum(r['slot_metadata_conflict'] for r in outputs),'coverage':coverage,
        'catalog_entries':len(catalog['entries']),'catalog_built_from':'predeclared schema definitions and eligible fit-split text only',
        'test_and_holdout_membership':'all published rows retained; ambiguous references excluded only from exact metrics',
        'segmentation':'publisher utterance boundaries preserved; whole waveforms decoded and checked; no ASR, splicing or cropping; no independent listening claim'}
    immutable(HERE/'data-audit.json',summary)
    paths=[Path(__file__),ROOT/'scripts/test/test_pcv_updated.py',HERE/'catalog.json',HERE/'custom-slots.json',
           HERE/'baseline.json',HERE/'audio-audit.json',HERE/'data-audit.json',prepared/'normalization.json']
    immutable(HERE/'data-freeze.json',{'manifests':manifests,'bindings':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'acquisition_sha256':sha(HERE/'acquisition-protocol.json'),'source_revision':REVISION,'test_inference_accessed':False})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('coverage','missing_pronunciations')},indent=2),flush=True)


if __name__=='__main__':
    prepare()
