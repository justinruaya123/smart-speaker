"""Acquire the exact user-requested public Kaggle corpus version, without credentials."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
URL='https://www.kaggle.com/api/v1/datasets/download/tommyngx/fluent-speech-corpus?datasetVersionNumber=1'
DEST=ROOT/'.cache/keyword-study-sources/kaggle-fluent-v1.zip'

if __name__=='__main__':
    receipt=DEST.with_suffix('.json')
    if DEST.exists() and receipt.exists():
        print('Existing public archive retained',DEST);raise SystemExit(0)
    DEST.parent.mkdir(parents=True,exist_ok=True);partial=DEST.with_suffix('.partial')
    started=time.perf_counter();sha=hashlib.sha256();count=0;next_log=128*1024**2
    with urllib.request.urlopen(urllib.request.Request(URL,headers={'User-Agent':'fixed-keyword-research'}),timeout=60) as source,partial.open('wb') as output:
        expected=int(source.headers.get('Content-Length','0'))
        while chunk:=source.read(8*1024**2):
            output.write(chunk);sha.update(chunk);count+=len(chunk)
            if count>=next_log:
                print('downloaded MiB',round(count/1024**2),'seconds',round(time.perf_counter()-started),flush=True);next_log+=128*1024**2
    if expected and count!=expected:raise ValueError('Incomplete download')
    import zipfile
    if not zipfile.is_zipfile(partial):raise ValueError('Public response is not a dataset ZIP')
    partial.replace(DEST)
    result={'requested_url':'https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus','version':1,'download_api':URL,
        'bytes':count,'sha256':sha.hexdigest(),'seconds':time.perf_counter()-started,'credentials_used':False}
    receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
