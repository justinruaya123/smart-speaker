"""Finite typed command/catalog helpers; these are not acoustic evidence."""
import re

PHONES = 'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split()


def number_words(n):
    if type(n) is not int or not 0<=n<=100: raise ValueError('Integer 0..100 required')
    units='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    if n<20: return units[n]
    if n==100: return 'one hundred'
    tens='zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
    return tens[n//10]+(' '+units[n%10] if n%10 else '')


def validate_command(c):
    if not isinstance(c,dict) or not isinstance(c.get('intent'),str): return False
    if c['intent'] in {'BRIGHTNESS','VOLUME_ABSOLUTE'}:
        return set(c)<= {'intent','value','entity_id'} and type(c.get('value')) is int and 0<=c['value']<=100
    if c['intent'] in {'BRIGHTNESS_DELTA','VOLUME_DELTA'}:
        return set(c)<= {'intent','delta','entity_id'} and type(c.get('delta')) is int and -100<=c['delta']<=100
    fields={'POWER_ON':{'intent','entity_id'},'POWER_OFF':{'intent','entity_id'},
            'MEDIA_PLAY':{'intent','track_id'},'MEDIA_RESUME':{'intent'},'MEDIA_STOP':{'intent'},'MEDIA_PAUSE':{'intent'},
            'VOLUME':{'intent','direction'},'TEMPERATURE_RELATIVE':{'intent','direction','area'},
            'COLOR_SET':{'intent','entity_id','color'}}
    if c['intent'] not in fields or not set(c)<=fields[c['intent']]: return False
    if any(type(v) is not str or not v for v in c.values()): return False
    if c['intent'] in {'POWER_ON','POWER_OFF','COLOR_SET'} and 'entity_id' not in c: return False
    if c['intent']=='COLOR_SET' and 'color' not in c: return False
    if c['intent'] in {'VOLUME','TEMPERATURE_RELATIVE'} and c.get('direction') not in {'up','down'}: return False
    return True


def resolve_entity(entities,alias,area=None,capability='power'):
    matches=[e for e in entities if e.get('exposed') and not e.get('disabled') and alias in e['aliases'] and (area is None or e['area']==area)]
    if len(matches)!=1: return {'reason':'ambiguous' if matches else 'unknown'}
    e=matches[0]
    if capability not in e['capabilities']: return {'reason':'unsupported_capability'}
    return {'id':e['id'],'reason':'available' if e['available'] else 'unavailable'}


def music_phrases(tracks):
    out={}
    def add(phrase,command):
        if phrase in out and out[phrase]!=command: out[phrase]=None
        else: out[phrase]=command
    for track in tracks:
        titles=track.get('aliases',[]) or [track['title']]
        for title in titles:
            if 'by' in title.split(): raise ValueError('Title needs an explicit alias without reserved BY')
            c={'intent':'MEDIA_PLAY','track_id':track['id']}
            add(title,c)
            for artist in track['artists']: add(title+' by '+artist,c)
    return out


def dictionary(path):
    words={}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(';;;'): continue
        parts=line.split(); word=re.sub(r'\(\d+\)$','',parts[0].lower())
        phones=[re.sub(r'\d','',p) for p in parts[1:] if p!='#']
        if all(p in PHONES for p in phones): words.setdefault(word,[PHONES.index(p)+1 for p in phones])
    return words


def pronounce(text,words):
    tokens=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
    if not tokens: raise ValueError('Empty pronunciation')
    out=[]
    for token in tokens:
        if token not in words: raise ValueError(f'Dictionary OOV: {token}')
        out.extend(words[token])
    return out


def fsc_command(row):
    action,obj,location=row['action'],row['object'],row['location']
    text=row['transcription'].lower()
    if obj=='lights' and action in {'activate','deactivate'}:
        return {'intent':'POWER_ON' if action=='activate' else 'POWER_OFF','entity_id':'light.'+('all' if location=='none' else location)}
    if obj=='music' and action in {'activate','deactivate'}:
        return {'intent':('MEDIA_RESUME' if 'resume' in text else 'MEDIA_PLAY') if action=='activate' else ('MEDIA_STOP' if 'stop' in text else 'MEDIA_PAUSE')}
    if obj=='volume' and action in {'increase','decrease'}:
        return {'intent':'VOLUME','direction':'up' if action=='increase' else 'down'}
    if obj=='heat' and action in {'increase','decrease'}:
        return {'intent':'TEMPERATURE_RELATIVE','direction':'up' if action=='increase' else 'down','area':location}
    return None


def compile_phrases(fixtures):
    """Compile exact full phrases. Unknown/extra tokens are never stripped."""
    out={}
    def add(phrase,command):
        if phrase in out and out[phrase]!=command: out[phrase]=None
        else: out[phrase]=command
    for e in fixtures['entities']:
        if not e.get('exposed') or e.get('disabled'): continue
        for alias in e['aliases']:
            # Availability is a mutable action-preview property, not a lexical exclusion.
            if 'power' in e['capabilities']:
                add('turn on '+alias,{'intent':'POWER_ON','entity_id':e['id']})
                add('turn off '+alias,{'intent':'POWER_OFF','entity_id':e['id']})
            if 'brightness' in e['capabilities']:
                for value in range(101):
                    words=number_words(value)
                    add(f'set {alias} to {words} percent',{'intent':'BRIGHTNESS','value':value,'entity_id':e['id']})
                    for verb,sign in [('increase',1),('decrease',-1)]:
                        add(f'{verb} {alias} by {words} percent',{'intent':'BRIGHTNESS_DELTA','delta':sign*value,'entity_id':e['id']})
            if 'color' in e['capabilities']:
                for color in fixtures['colors']: add(f'set {alias} to {color}',{'intent':'COLOR_SET','color':color,'entity_id':e['id']})
    for value in range(101): add(f'set volume to {number_words(value)} percent',{'intent':'VOLUME_ABSOLUTE','value':value})
    for phrase,command in music_phrases(fixtures['tracks']).items():
        add(phrase,command); add('play '+phrase,command)
    return out
