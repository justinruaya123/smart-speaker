"""Frozen, original-human-only corpus and catalog protocol. No audio synthesis."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from einops import reduce
from .levels import percentage, spoken_number

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.cache/parameter-study'
DATA = ROOT / 'data/parameter-study'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)/2**32


def normalize(text):
    text = text.lower().replace('\u2019', "'").replace('-', ' ')
    text = re.sub(r'\b\d+\b', lambda m: spoken_number(int(m[0])) if int(m[0])<=100 else m[0], text)
    text = text.replace('%', ' percent ')
    return ' '.join(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text))


def pcm(row):
    audio, rate = sf.read(ROOT/row['audio'], dtype='float32')
    if audio.ndim == 2:
        audio = reduce(audio, 't c -> t', 'mean')
    if rate != 16000:
        import math
        divisor = math.gcd(rate,16000)
        audio = resample_poly(audio,16000//divisor,rate//divisor).astype('float32')
    if not np.isfinite(audio).all() or len(audio)<1600 or len(audio)>128000:
        raise ValueError('Audio outside finite 0.1 to 8 second protocol')
    return audio


def lexicon():
    words = {}
    for line in (ROOT/'STT/alternatives/sources/cmudict.dict').read_text().splitlines():
        parts = line.split('#',1)[0].split()
        if not parts or parts[0].startswith(';;;'): continue
        word = re.sub(r'\(\d+\)$', '', parts[0])
        words.setdefault(word, [re.sub(r'\d', '', p) for p in parts[1:] if not p.startswith('#')])
    return words


def phones(text, dictionary):
    result = []
    for word in text.split():
        if word not in dictionary: return None
        if result: result.append('|')
        result.extend(dictionary[word])
    return result


def prepare():
    DATA.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.jsonl').exists():
        audit=json.loads((DATA/'audit.json').read_text())
        if digest(DATA/'manifest.jsonl')!=audit['manifest_sha256']:
            raise ValueError('Frozen study manifest changed; use a new versioned protocol')
        return read_rows(DATA/'manifest.jsonl')
    original = read_rows(ROOT/'data/asr.jsonl')
    paths = {r['audio'] for r in original}
    rows=[]; seen=set(); exclusions=Counter();annotation_quarantine=[]
    for r in original:
        if r['audio'] in seen: continue
        if r['source']=='SLURP-real' and '-headset' not in r['audio'] and r['audio'].replace('.flac','-headset.flac') in paths: continue
        if r['source']=='SnipsSmartLights' and not r['audio'].endswith('_close.wav'): continue
        if r['source']=='MSWC-English': continue  # richer original word pool below
        if r['split']=='calibration': exclusions['baseline_calibration']+=1; continue
        seen.add(r['audio'])
        split=r['split']; speaker=r['speaker_id']
        if r['source'] in ('SnipsSmartLights','GoogleSpeechCommandsV2'):
            f=fraction('parameter-study/'+speaker); split='train' if f<.75 else 'dev' if f<.87 else 'test'
        text=normalize(r['transcript']); targets=[]
        if r['source']=='GoogleSpeechCommandsV2':
            value=percentage(text)
            targets.append({'role':'number','value':str(value),'surface':text} if value is not None else {'role':'word','value':text,'surface':text})
        if r.get('command') in ('BRIGHTNESS','VOLUME'):
            try: value=percentage(text)
            except ValueError: value=None
            if value is not None: targets.append({'role':'number','value':str(value),'surface':spoken_number(value)})
        for entity in r.get('entities',[]):
            role={'artist_name':'artist','song_name':'song','house_place':'device'}.get(entity.get('type'))
            if role:
                tokens={t['id']:t['surface'] for t in r.get('tokens',[])}
                surface=normalize(' '.join(tokens[i] for i in entity['span']))
                if surface: targets.append({'role':role,'value':surface,'surface':surface})
        # Publisher token/entity annotations occasionally describe another
        # sentence. Ignore tokenization/punctuation differences, but never fit
        # a name which does not occur in the supplied transcript at all.
        compact=lambda s: re.sub(r'[^a-z0-9]','',s.lower())
        if any(t['role'] in ('artist','song','device') and compact(t['surface']) not in compact(text) for t in targets):
            exclusions['inconsistent_entity_annotation']+=1
            annotation_quarantine.append({'id':r['id'],'text':text,'targets':targets})
            continue
        rows.append(dict(id=r['id'],audio=r['audio'],text=text,speaker=speaker,split=split,source=r['source'],
                         license=r.get('license'),kind='word' if r['source']=='GoogleSpeechCommandsV2' else 'natural',targets=targets,command=r.get('command'),
                         source_split=r.get('source_split'),timing_verified=False))
    for r in read_rows(ROOT/'data/stt/brightness.jsonl'):
        if r['kind']!='word' or r['audio'] in seen: continue
        seen.add(r['audio'])
        speaker=r['speaker']; f=fraction('parameter-study/'+speaker)
        split='train' if f<.75 else 'dev' if f<.87 else 'test'
        if speaker.startswith('fsdd/'):
            name=speaker.split('/')[-1];split='dev' if name=='theo' else 'test' if name=='yweweler' else 'train'
        value=r.get('word_value'); text=normalize(r['text'])
        if value is None:
            try: value=percentage(text)
            except ValueError: pass
        target={'role':'number','value':str(value),'surface':text} if value is not None else {'role':'word','value':text,'surface':text}
        rows.append(dict(id=r['id'],audio=r['audio'],text=text,speaker=speaker,split=split,source=r['source'],
                         license='See source manifests and acquisition receipts',kind='word',targets=[target],timing_verified=False))
    # A deterministic bounded study: keep all labeled commands/words and a fixed
    # sample of ordinary English. This is not a complete public-dataset benchmark.
    backgrounds=sorted([r for r in rows if not r['targets']],key=lambda r:fraction(r['id']))
    keep_background={r['id'] for split,cap in [('train',7000),('dev',1000),('test',1000)]
                     for r in [x for x in backgrounds if x['split']==split][:cap]}
    rows=[r for r in rows if r['targets'] or r['id'] in keep_background]
    # Freeze novel-name selection using training counts alone, before fitting.
    counts=Counter((t['role'],t['value']) for r in rows if r['split']=='train' for t in r['targets'] if t['role'] in ('artist','song','device'))
    novel=sorted(k for k,n in counts.items() if n>=2 and fraction('novel/'+':'.join(k))<.25)
    novel_surfaces={value for role,value in novel}
    for r in rows:
        if r['split']=='train' and (any(' '+name+' ' in ' '+r['text']+' ' for name in novel_surfaces)
                or any((t['role'],t['value']) in novel for t in r['targets'])):
            r['split']='enroll'
        for t in r['targets']: t['novel']=(t['role'],t['value']) in novel
    # Enrollment shares TRAIN speakers only. It is never used by gradient fitting.
    # Test query speakers remain disjoint from both training and enrollment.
    sha_rows=defaultdict(list)
    for i,r in enumerate(rows):
        info=sf.info(ROOT/r['audio'])
        r['duration']=info.duration
        if not .1<=info.duration<=8: exclusions['duration']+=1; continue
        r['audio_sha256']=digest(ROOT/r['audio']);sha_rows[r['audio_sha256']].append(r)
        if i%5000==0: print('Audit original human recordings',i,'/',len(rows),flush=True)
    clean=[];quarantine=[]
    for sha,group in sha_rows.items():
        labels={(r['text'],r['split']) for r in group}
        if len(labels)>1: quarantine.extend(r['id'] for r in group); continue
        clean.append(group[0])
    clean.sort(key=lambda r:r['id'])
    speaker_splits=defaultdict(set)
    for r in clean: speaker_splits[r['speaker']].add('train' if r['split']=='enroll' else r['split'])
    leaks={s:sorted(v) for s,v in speaker_splits.items() if len(v)>1}
    if leaks: raise ValueError(f'Cross-split speakers: {leaks}')
    catalog=defaultdict(set);fixed=defaultdict(set)
    for r in clean:
        for t in r['targets']:
            catalog[t['role']].add(t['value'])
            if r['split']=='train': fixed[t['role']].add(t['value'])
    if any(value in fixed[role] for role,value in novel):
        raise ValueError('A reserved name leaked into acoustic training')
    catalog['number']=set(map(str,range(101)))
    fixed['number']=set(map(str,range(101)))  # grammar text does not supply acoustic examples
    payload={'fixed':{k:sorted(v) for k,v in fixed.items()},'expanded':{k:sorted(v) for k,v in catalog.items()},
             'novel':[dict(role=k[0],value=k[1]) for k in novel],
             'protocol':'Catalog ontology is declared before training; no per-query oracle catalog. Novel names selected by training counts and hash only. No Spotify or Home Assistant connection.'}
    dump(DATA/'catalogs.json',payload)
    (DATA/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in clean),encoding='utf-8')
    audit={'rows':len(clean),'by_split':dict(Counter(r['split'] for r in clean)),
           'source_split':{s:dict(Counter(r['split'] for r in clean if r['source']==s)) for s in sorted({r['source'] for r in clean})},
           'speakers':{s:len({r['speaker'] for r in clean if r['split']==s}) for s in ('train','enroll','dev','test')},
           'cross_split_speakers':leaks,'quarantined_duplicate_ids':quarantine,'excluded':dict(exclusions),
           'annotation_quarantine':annotation_quarantine,'reserved_name_training_overlap':0,
           'manifest_sha256':digest(DATA/'manifest.jsonl'),'synthetic_or_concatenated_recordings':0,
           'novel_names':len(novel),'text_normalization':'lowercase; integer words for 0..100; percent spoken as percent; punctuation removed',
           'input_manifests':{p:digest(ROOT/p) for p in ['data/asr.jsonl','data/stt/brightness.jsonl']}}
    dump(DATA/'audit.json',audit);print(json.dumps(audit),flush=True)
    return clean
