"""Add pinned isolated-word negatives to the prepared FSC experiment, immutably."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcm.catalog import dictionary, pronounce
from vcm.data import read_manifest, audit_rows, feature_receipt, load_verified_features
from vcm.frontend import features
from vcm.integrity import load, save, digest, identity


def prepare(base,output):
    root=Path(__file__).resolve().parents[1]; base=Path(base); output=Path(output)
    source=load(root/'data/sources/speech_commands.source.json')
    archive=root/source['path']
    if digest(archive)!=source['sha256']: raise ValueError('Speech Commands hash mismatch')
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); (output/'audio').mkdir(); (output/'features').mkdir()
    words=dictionary(root/'data/raw/cmudict/cmudict.dict')
    rows=[r for s in ('train','dev','calibration','test') for r in read_manifest(base/f'{s}.jsonl')]
    with tarfile.open(archive,'r:gz') as tar:
        names=[m.name.removeprefix('./') for m in tar.getmembers() if m.isfile()]
        valid=set(tar.extractfile('./validation_list.txt').read().decode().splitlines())
        test=set(tar.extractfile('./testing_list.txt').read().decode().splitlines())
    choices={s:[] for s in ('train','dev','calibration','test')}
    for name in names:
        if not name.endswith('.wav') or name.startswith('_'): continue
        speaker=Path(name).stem.split('_nohash_')[0]
        split='test' if name in test else ('calibration' if int(identity(speaker)[:8],16)%2 else 'dev') if name in valid else 'train'
        if Path(name).parts[0] not in words: continue
        choices[split].append(name)
    limits={'train':2000,'dev':100,'calibration':200,'test':200}
    selected={name:s for s,names in choices.items() for name in sorted(names,key=identity)[:limits[s]]}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            member_name=member.name.removeprefix('./')
            if member_name not in selected: continue
            payload=tar.extractfile(member).read(); pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
            if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all(): raise ValueError('Invalid SC PCM')
            key='sc-'+identity(member_name)[:20]; text=Path(member_name).parts[0]
            f=features(pcm); phones=pronounce(text,words)
            if len(phones)+sum(a==b for a,b in zip(phones,phones[1:]))>f.shape[1]: continue
            audio=output/'audio'/f'{key}.wav'; audio.write_bytes(payload)
            feat=output/'features'/f'{key}.npy'; np.save(feat,f)
            rows.append({'id':key,'source':'Speech Commands v0.02','source_revision':source['sha256'],'speaker':'sc:'+Path(member_name).stem.split('_nohash_')[0],
                         'session':'unknown','split':selected[member_name],'text':text,'expected':None,'valid_request':False,
                         'synthetic':False,'sample_rate':rate,'channels':1,'samples':len(pcm),'audio':audio.resolve().as_posix(),
                         'features':feat.resolve().as_posix(),**feature_receipt(feat),'audio_sha256':hashlib.sha256(payload).hexdigest(),
                         'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),'phones':phones,
                         'annotation':'published isolated-word folder label; not a complete command'})
    audit=audit_rows(rows); total=np.zeros(40); square=np.zeros(40); count=0
    for r in rows:
        if r['split']!='train': continue
        f=load_verified_features(r).astype('float64'); total+=f.sum(1); square+=(f*f).sum(1); count+=f.shape[1]
    mean=total/count; std=np.sqrt(np.maximum(square/count-mean**2,1e-4))
    save(output/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_split':'train','frames':count})
    catalog=load(base/'catalog.json'); save(output/'catalog.json',catalog)
    for split in choices:
        records=[{**r,'catalog_hash':identity(catalog)} for r in rows if r['split']==split]
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    save(output/'audit.json',{**audit,'parent_audit':digest(base/'audit.json'),'source':source,'negative_limits':limits,
                             'negative_scope':'isolated words only; no continuous, far-field or adversarial command benchmark',
                             'normalization_fit':'train only','near_duplicate_audit':'not performed'})
    print(json.dumps(audit),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--base',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); prepare(a.base,a.output)
