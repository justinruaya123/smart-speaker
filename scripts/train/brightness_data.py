"""Versioned brightness data, original human recordings and split auditing."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import re
import tarfile
import urllib.request
import numpy as np
from .audio import load_audio
from .levels import spoken_number, percentage
from .text import normalize_text
from .parameter_backend import parse_brightness

REVISION='0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218'
NUMBERS=set(' '.join(spoken_number(i) for i in range(101)).split())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def partition(speaker):
    value=int(hashlib.sha256(('brightness-v1/'+speaker).encode()).hexdigest()[:8],16)%100
    return 'train' if value<75 else 'dev' if value<87 else 'test'


def normal(text):
    text=text.replace('%',' percent ')
    text=re.sub(r'-(?=\d)', 'minus ', text)
    text=re.sub(r'\b\d+\b',lambda m:spoken_number(int(m[0])) if int(m[0])<=100 else ' '.join(spoken_number(int(x)) for x in m[0]),text)
    return normalize_text(text)


def fetch_context(root):
    """Only pinned public word archives; stop reading once selected words end."""
    root=Path(root);source=root/'STT/data/raw/mswc/splits.tar.gz'
    selected={}
    with tarfile.open(source) as archive:
        member=next(m for m in archive if m.name.endswith('train.csv'))
        for row in csv.DictReader(io.TextIOWrapper(archive.extractfile(member),encoding='utf-8')):
            if row['WORD'] in {'brightness','percent'} and row['VALID'].lower()=='true':
                selected[row['LINK'].replace('/','_',1)]=row
    out=root/'data/audio/stt_context';out.mkdir(parents=True,exist_ok=True)
    failures=[]
    def download(item):
        shard,word=item
        url=f'https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/{REVISION}/data/opus/en/train/audio/{shard}.tar.gz?download=true'
        for attempt in range(2):
            try:
                with urllib.request.urlopen(url,timeout=45) as stream,tarfile.open(fileobj=stream,mode='r|gz') as archive:
                    for member in archive:
                        if not member.isfile():continue
                        name=Path(member.name).name;current=name.split('_common_voice_')[0]
                        if current>word:break
                        if name not in selected or (out/name).exists():continue
                        payload=archive.extractfile(member).read()
                        if len(payload)!=member.size:raise IOError('Incomplete archive member')
                        (out/name).write_bytes(payload)
                return
            except Exception as exc:
                print('Word acquisition retry',shard,attempt,str(exc),flush=True)
                if attempt==1:failures.append({'shard':shard,'error':str(exc)})
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,[(11,'brightness'),(53,'percent')]))
    rows=[]
    for name,row in selected.items():
        path=out/name
        if path.exists():
            rows.append({'id':'mswc-context/'+name,'audio':path.relative_to(root).as_posix(),
                'transcript':row['WORD'],'speaker_id':'mswc/'+row['SPEAKER'],'source':'MSWC-English',
                'sha256':sha(path),'revision':REVISION,'license':'CC-BY-4.0'})
    write_json(root/'data/stt/context_receipt.json',{'rows':rows,'failures':failures,'revision':REVISION})
    print('Context word files',len(rows),Counter(r['transcript'] for r in rows),flush=True)


def load_pcm(row,root,cache=None):
    def audio(path):
        if cache is not None and path in cache:return cache[path]
        pcm,_=load_audio(Path(root)/path)
        if cache is not None:cache[path]=pcm
        return pcm
    if 'components' in row or row.get('kind')=='constructed':
        raise ValueError('Only original human recordings are permitted in the current training protocol')
    return audio(row['audio'])


def prepare(root,extra_manifest=None,refresh=False):
    root=Path(root);output=root/'data/stt-human/brightness.jsonl'
    if extra_manifest and not refresh:raise ValueError('An extra manifest requires explicit refresh=True')
    if output.exists() and not refresh:
        audit=json.loads((root/'data/stt-human/brightness_audit.json').read_text())
        if sha(output)!=audit['manifest_sha256']:raise ValueError('Frozen STT manifest changed; use an explicit audited refresh')
        if sha(root/'data/stt-human/vocabulary.json')!=audit.get('vocabulary_sha256'):
            raise ValueError('STT vocabulary changed or lacks provenance; explicitly refresh the dataset')
        rows=read_rows(output)
        for row in rows:
            if 'audio' in row and sha(root/row['audio'])!=row['audio_sha256']:
                raise ValueError('STT recording changed: '+row['audio'])
        return rows
    rng=random.Random(20260919)
    catalog=json.loads((root/'artifacts/bundle.json').read_text())['catalog']
    base=read_rows(root/'data/asr.jsonl')
    base=list({r['audio']:r for r in reversed(base)}.values())
    known_speakers={r['speaker_id'] for r in base if r['split']=='train' and r.get('speaker_id')}
    known_audio={r['audio'] for r in base if r['split']=='train'}
    names={r['audio'] for r in base}
    natural=[];negatives=defaultdict(list);words=[]
    for r in base:
        if r['split']=='calibration':continue
        path=r['audio']
        if r['source']=='SLURP-real' and not path.endswith('-headset.flac') and path.replace('.flac','-headset.flac') in names:continue
        if r['source']=='SnipsSmartLights' and r.get('distance')!='close':continue
        split=r['split']
        # SmartLights is held out for new STT only; retired recognizer saw these files.
        if r['source']=='SnipsSmartLights':split=partition(r['speaker_id'])
        text=normal(r['transcript'])
        row={'id':r['id'].split('/level-repeat-')[0], 'audio':path,'text':text,'split':split,
             'speaker':r['speaker_id'],'source':r['source'],'kind':'natural',
             'asr_training_overlap':path in known_audio}
        if r['source']=='MSWC-English':
            if text=='brightness':row['kind']='word';row['split']='train';words.append(row)
            continue
        if r.get('command')=='BRIGHTNESS' or r.get('source_intent') in {'SetLightBrightness','IncreaseBrightness','DecreaseBrightness'}:
            natural.append(row)
        else:
            row['text']='<unknown>';row['kind']='negative';negatives[split].append(row)
    for split,rr in negatives.items():
        rng.shuffle(rr);natural.extend(rr[:1200 if split=='train' else 250])
    # Reuse only isolated recordings, not old STT checkpoints or its command split.
    for r in read_rows(root/'STT/data/numeric_manifest.jsonl'):
        if r['kind']!='human_word':continue
        words.append({'id':r['id'],'audio':'STT/'+r['audio'],'text':r['text'],'split':r['split'],
                      'speaker':r['speaker'],'source':r['source'],'kind':'word','asr_training_overlap':False})
    receipt=root/'data/stt/context_receipt.json'
    if receipt.exists():
        for r in json.loads(receipt.read_text())['rows']:
            # Reserve genuinely new context speakers for dev/test compositions.
            split='train' if r['speaker_id'] in known_speakers else ('dev' if int(hashlib.sha256(r['speaker_id'].encode()).hexdigest()[:8],16)%2 else 'test')
            words.append({'id':r['id'],'audio':r['audio'],'text':r['transcript'],'split':split,
                          'speaker':r['speaker_id'],'source':r['source'],'kind':'word',
                          'asr_training_overlap':r['speaker_id'] in known_speakers})
    # Keep every Common Voice speaker on one side of the incumbent's training boundary.
    for r in words:
        if r['speaker'].startswith('mswc/'):
            r['split']='train' if r['speaker'] in known_speakers else ('dev' if int(hashlib.sha256(r['speaker'].encode()).hexdigest()[:8],16)%2 else 'test')
    rows=natural+words
    if extra_manifest:
        for r in read_rows(root/extra_manifest):
            kind=r.get('kind','natural')
            if kind not in {'natural','word','negative'}:raise ValueError('Overlay kind must be natural, word or negative')
            rows.append({'id':r['id'],'audio':r['audio'],'text':'<unknown>' if kind=='negative' else normal(r['transcript']),
                         'split':r['split'],'speaker':r['speaker_id'],'source':r['source'],
                         'kind':kind,'asr_training_overlap':False})
    grouped=defaultdict(list);digests={}
    for r in rows:
        path=(root/r['audio']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Dataset audio must be inside the workspace')
        if r['split'] not in {'train','dev','test'}:raise ValueError('Expected train, dev or test split')
        key=sha(path);digests[r['audio']]=key;r['audio_sha256']=key
        grouped[key].append(r)
    cleaned=[];quarantined=[]
    for group in grouped.values():
        if len({r['split'] for r in group})>1 or len({r['text'] for r in group})>1:
            quarantined.extend(group);continue
        cleaned.append(group[0])
    rows=cleaned
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Distinct STT recordings must have unique ids')
    speakers={s:{r['speaker'] for r in rows if r['split']==s} for s in ['train','dev','test']}
    overlap=(speakers['train']&speakers['dev'])|(speakers['train']&speakers['test'])|(speakers['dev']&speakers['test'])
    if overlap:raise ValueError(f'Conflicting speaker splits: {sorted(overlap)[:5]}')
    vocabulary=['<blank>','<unknown>']+sorted(NUMBERS|{'brightness','percent'}|{w for r in rows if r['split']=='train' and r['text']!='<unknown>' for w in r['text'].split()})
    for r in rows:
        parsed=parse_brightness(r['text'],catalog)
        r['reference']=parsed['command'] if parsed.get('status')=='accepted' else None
        r['value']=parsed['command']['arguments']['percent'] if parsed.get('status')=='accepted' else None
        if r['kind']=='word':
            try:r['word_value']=percentage(r['text'])
            except ValueError:r['word_value']=None
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(root/'data/stt-human/vocabulary.json',vocabulary)
    write_json(root/'data/stt-human/brightness_audit.json',{'manifest_sha256':sha(output),
        'vocabulary_sha256':sha(root/'data/stt-human/vocabulary.json'),
        'counts':dict(Counter(r['kind']+'/'+r['split'] for r in rows)),
        'natural_absolute':dict(Counter(r['source']+'/'+r['split'] for r in rows if r['kind']=='natural' and r['value'] is not None)),
        'speaker_overlap':len(overlap),'quarantined':len(quarantined),'synthetic_or_constructed_recordings':0,
        'vocabulary_size':len(vocabulary),'source_manifest_sha256':sha(root/'data/asr.jsonl'),
        'note':'SmartLights test is unseen by new STT but seen by retired recognizer; report separately. Only original human recordings are included.'})
    print('Prepared original-human brightness',len(rows),'vocabulary',len(vocabulary),flush=True)
    return rows
