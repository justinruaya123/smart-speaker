"""Download a balanced speaker-separated human numeral subset from its publisher."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import io
import json
import random
import time
import urllib.request
from common import ROOT,split_speaker,say_number,write_json,write_jsonl

def main():
    raw=ROOT/'data/raw/mlend'
    revision=json.loads((raw/'revision.json').read_text())['sha']
    rows=list(csv.DictReader((raw/'attributes.csv').open(newline='',encoding='utf-8')))
    random.Random(20260919).shuffle(rows)
    counts=Counter();selected=[]
    for r in rows:
        value=int(r['Numeral'])
        if value>100:continue
        speaker='mlend/'+r['Speaker'];split=split_speaker(speaker)
        key=(value,split)
        cap=35 if split=='train' else 12
        if counts[key]>=cap:continue
        counts[key]+=1
        selected.append((r,value,speaker,split))
    audio=ROOT/'data/audio/mlend';audio.mkdir(parents=True,exist_ok=True)
    print('Selected',len(selected),'human recordings across',len({x[2] for x in selected}),'speakers',flush=True)
    def get(item):
        r,value,speaker,split=item
        name=r['filename'];dest=audio/name
        url=f'https://raw.githubusercontent.com/MLEndDatasets/SpokenNumerals/{revision}/MLEndSND_audiofiles/{name}'
        for attempt in range(4):
            try:
                if not dest.exists():
                    with urllib.request.urlopen(url,timeout=20) as response:
                        payload=response.read()
                    if payload[:4] not in {b'RIFF',b'RF64'}:raise ValueError('Not WAV audio')
                    dest.write_bytes(payload)
                return {'id':'mlend/'+name,'source':'MLEndSpokenNumerals','audio':dest.relative_to(ROOT).as_posix(),
                    'text':'hundred' if value==100 else say_number(value),'value':value,'speaker':speaker,'split':split,
                    'group':'mlend/'+name,'kind':'human_word','entities':[],'intent':None,
                    'label_type':'numeric category, canonical English word target (not an independently verified verbatim transcript)',
                    'intonation':r['Intonation'],'license':'See publisher dataset terms; no license file in GitHub snapshot',
                    'acquisition':'fresh_download','url':url,'revision':revision,'audio_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
            except Exception as exc:
                if attempt==3:return {'error':str(exc),'filename':name}
    good=[];fail=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures=[pool.submit(get,item) for item in selected]
        for i,future in enumerate(as_completed(futures),1):
            r=future.result()
            (fail if 'error' in r else good).append(r)
            if i%100==0:
                print('MLEnd',i,'/',len(selected),'failures',len(fail),flush=True)
                write_jsonl(ROOT/'data/mlend.jsonl',good)
    write_jsonl(ROOT/'data/mlend.jsonl',good)
    write_json(ROOT/'evidence/mlend_acquisition.json',{'requested':len(selected),'downloaded':len(good),'failures':fail,
        'values':dict(Counter(r['value'] for r in good)),'revision':revision,'sampling':'Deterministic shuffled cap 35 train, 12 dev, 12 test per numeral; speaker hash determines split before sampling.'})
    print('MLEnd completed',len(good),flush=True)

if __name__=='__main__':main()
