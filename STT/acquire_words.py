"""Select number-word MSWC audio from pinned public train archives."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import csv
import io
import json
from pathlib import Path
import tarfile
import time
import urllib.request

from acquire import fetch
from common import ROOT, NUM_WORDS, split_speaker, digest, write_json, write_jsonl

REVISION='0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218'
BASE=f'https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/{REVISION}/data/opus/en/train/audio/'
WORDS=NUM_WORDS | {'volume','percent','play','music','song','set','to','by'}

def first_word(i):
    url=BASE+str(i)+'.tar.gz?download=true'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=30) as stream:
                with tarfile.open(fileobj=stream,mode='r|gz') as tar:
                    name=next(m.name for m in tar if m.isfile())
            word=Path(name).name.split('_common_voice_')[0]
            print('Shard',i,'starts',word,flush=True)
            return str(i),word
        except Exception:
            if attempt==2:raise

def select_metadata():
    archive=ROOT/'data/raw/mswc/splits.tar.gz'
    while not archive.exists():
        print('Waiting for verified MSWC split metadata',flush=True);time.sleep(20)
    selected={};counts=Counter()
    with tarfile.open(archive,'r:gz') as tar:
        member=next(m for m in tar if m.name.endswith('train.csv'))
        for r in csv.DictReader(io.TextIOWrapper(tar.extractfile(member),encoding='utf-8')):
            word=r['WORD']
            if word not in WORDS or r['VALID'].lower()!='true':continue
            speaker='mswc/'+r['SPEAKER']
            split=split_speaker(speaker)
            cap=700 if split=='train' else 150
            key=(word,split)
            if counts[key]>=cap:continue
            link=r['LINK'];filename=link.replace('/','_',1)
            selected[filename]={'id':'mswc/'+link,'source':'MSWC-English','audio':'data/audio/mswc/'+filename,
                'text':word,'speaker':speaker,'split':split,'group':'mswc/'+Path(link).stem,
                'kind':'human_word','entities':[],'intent':None,'license':'CC-BY-4.0',
                'acquisition':'fresh_download','source_partition':'official_train',
                'boundary_quality':'automatic_common_voice_word_alignment'}
            counts[key]+=1
    write_json(ROOT/'evidence/mswc_selected.json',selected)
    print('Requested MSWC words',dict(counts),flush=True)
    return selected

def download_shard(i,selected):
    result=[]
    url=BASE+str(i)+'.tar.gz?download=true'
    (ROOT/'data/audio/mswc').mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=50) as stream:
                with tarfile.open(fileobj=stream,mode='r|gz') as tar:
                    for member in tar:
                        name=Path(member.name).name
                        if not member.isfile() or name not in selected:continue
                        r=dict(selected[name]);dest=ROOT/r['audio']
                        payload=tar.extractfile(member).read()
                        import hashlib
                        r['audio_sha256']=hashlib.sha256(payload).hexdigest()
                        if not dest.exists():dest.write_bytes(payload)
                        r['url']=url;r['revision']=REVISION
                        result.append(r)
            print('MSWC shard',i,'selected',len(result),flush=True)
            return result
        except Exception as exc:
            print('MSWC retry',i,attempt,str(exc),flush=True)
            if attempt==2:return result

def main():
    index=ROOT/'evidence/mswc_archive_index.json'
    if index.exists():
        starts=json.loads(index.read_text())
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            starts=dict(pool.map(first_word,range(87)))
        write_json(index,starts)
    selected=select_metadata()
    shards=set()
    for word in WORDS:
        candidates=[int(i) for i,start in starts.items() if start<=word]
        if candidates:
            last=max(candidates)
            shards.add(last)
            if starts[str(last)]==word and last>0:shards.add(last-1)
    write_json(ROOT/'evidence/mswc_shards.json',{'requested_words':sorted(WORDS),'shards':sorted(shards)})
    print('MSWC shards to stream',sorted(shards),flush=True)
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download_shard,i,selected) for i in sorted(shards)]
        for future in as_completed(futures):
            rows+=future.result()
            rows=list({r['id']:r for r in rows}.values())
            write_jsonl(ROOT/'data/mswc.jsonl',rows)
    write_json(ROOT/'evidence/mswc_acquisition.json',{'count':len(rows),'words':dict(Counter(r['text'] for r in rows)),
        'note':'Pinned upstream archives streamed; individual saved clips have SHA256. Archive hashes not verified because archives are not retained.'})

if __name__=='__main__':main()
