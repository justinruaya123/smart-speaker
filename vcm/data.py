"""Pinned FSC preparation; no old local manifests or model weights are inputs."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import shutil
from collections import Counter
import numpy as np
import soundfile as sf
from .catalog import dictionary, pronounce, fsc_command, PHONES
from .frontend import features, SPEC
from .integrity import digest, save, identity


def feature_receipt(path):
    return {'features_sha256':digest(path),'frontend_spec':identity(SPEC)}


def load_verified_features(row):
    if not row.get('features_sha256') or not row.get('frontend_spec'):
        raise ValueError('Missing feature provenance; regenerate preparation under the current protocol')
    if row['frontend_spec']!=identity(SPEC): raise ValueError('Feature frontend identity mismatch')
    if digest(row['features'])!=row['features_sha256']: raise ValueError('Feature cache hash mismatch')
    value=np.load(row['features'],allow_pickle=False)
    if value.ndim!=2 or value.shape[0]!=40 or value.dtype!=np.float32 or not np.isfinite(value).all():
        raise ValueError('Invalid feature array')
    return value


def audit_rows(rows):
    speakers={}; hashes={}; ids=set()
    for r in rows:
        if r['id'] in ids: raise ValueError('Duplicate recording ID')
        ids.add(r['id'])
        for values,key,label in [(speakers,r['speaker'],'speaker'),(hashes,r['pcm_sha256'],'hash')]:
            if key in values and values[key]!=r['split']: raise ValueError(f'Cross-split {label} overlap')
            values[key]=r['split']
    return {'rows':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
            'speakers':{s:len({r['speaker'] for r in rows if r['split']==s}) for s in sorted({r['split'] for r in rows})},
            'cross_split_speaker_overlap':0,'cross_split_pcm_overlap':0}


def prepare(root, output):
    root=Path(root); output=Path(output)
    if output.exists(): raise FileExistsError('Choose a fresh preparation output; frozen manifests are never overwritten')
    source=json.loads((root/'data/sources/fluent.source.json').read_text())
    archive=root/source['path']
    if digest(archive)!=source['sha256']: raise ValueError('FSC archive hash mismatch')
    cmu=json.loads((root/'data/sources/cmudict.source.json').read_text())
    for item in cmu['files']:
        if digest(root/item['path'])!=item['sha256']: raise ValueError('Dictionary hash mismatch')
    words=dictionary(root/'data/raw/cmudict/cmudict.dict')
    output.mkdir(parents=True)
    rows=[]; exclusions=[]; phrases={}; totals=np.zeros(40); squares=np.zeros(40); frames=0
    extracted=output/'source'
    extracted.mkdir()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            target=(extracted/member.name).resolve()
            if not target.is_relative_to(extracted.resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
            if member.isfile():
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream: shutil.copyfileobj(tar.extractfile(member),stream)
    if True:
        prefix='fluent_speech_commands_dataset/'
        tables={split:list(csv.DictReader(io.StringIO((extracted/(prefix+'data/'+name+'_data.csv')).read_text())))
                for split,name in [('train','train'),('dev','valid'),('test','test')]}
        calibration=set(sorted({r['speakerId'] for r in tables['dev']})[5:])
        for original,table in tables.items():
            for row in table:
                split='calibration' if original=='dev' and row['speakerId'] in calibration else original
                name=prefix+row['path']
                if '..' in PurePosixPath(name).parts: raise ValueError('Unsafe archive member')
                audio=extracted/name
                payload=audio.read_bytes()
                pcm,rate=sf.read(io.BytesIO(payload),dtype='float32')
                key=Path(name).stem
                if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all(): raise ValueError('Invalid source PCM')
                text=row['transcription'].lower().strip()
                try: labels=pronounce(text,words)
                except ValueError as exc:
                    exclusions.append({'id':key,'reason':str(exc)}); continue
                f=features(pcm)
                if len(labels)+sum(a==b for a,b in zip(labels,labels[1:])) > f.shape[1]:
                    exclusions.append({'id':key,'reason':'infeasible_ctc'}); continue
                feat=output/'features'/f'{key}.npy'; feat.parent.mkdir(exist_ok=True); np.save(feat,f)
                command=fsc_command(row)
                record={'id':key,'source':'FSC','source_revision':source['revision'],'speaker':'fsc:'+row['speakerId'],
                        'session':'unknown','split':split,'text':text,'expected':command,'valid_request':True,
                        'unsupported_reference':None if command else {k:row[k] for k in ('action','object','location')},
                        'synthetic':False,'sample_rate':rate,'channels':1,'samples':len(pcm),
                        'audio':audio.resolve().as_posix(),'features':feat.resolve().as_posix(),**feature_receipt(feat),
                        'audio_sha256':hashlib.sha256(payload).hexdigest(),'pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
                        'phones':labels,'clipped_samples':int(np.sum(np.abs(pcm)>=1.)), 'annotation':'published FSC CSV; not independently listened'}
                rows.append(record)
                if split=='train':
                    totals+=np.sum(f,axis=1); squares+=np.sum(f.astype('float64')**2,axis=1); frames+=f.shape[1]
                    if command is not None:
                        if text in phrases and phrases[text]['command']!=command: raise ValueError('Conflicting command label')
                        phrases[text]={'phrase':text,'phones':labels,'command':command}
                if len(rows)%5000==0: print('Prepared',len(rows),flush=True)
    audit=audit_rows(rows)
    mean=totals/frames; std=np.sqrt(np.maximum(squares/frames-mean**2,1e-4))
    catalog={'version':1,'phone_inventory':['<blank>']+PHONES,'commands':[phrases[k] for k in sorted(phrases)],
             'scope':'FSC finite command phrases seen in training; no arbitrary names, numeric/color acoustic claim'}
    save(output/'catalog.json',catalog)
    save(output/'normalization.json',{'mean':mean.tolist(),'std':std.tolist(),'fit_split':'train','frames':frames})
    for split in ('train','dev','calibration','test'):
        selected=[{**r,'catalog_hash':identity(catalog)} for r in rows if r['split']==split]
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    save(output/'audit.json',{**audit,'source':source,'dictionary':cmu,'exclusions':exclusions,
                             'unsupported_requests':sum(r['expected'] is None for r in rows),
                             'protocol':'original train/test speakers; sorted original validation speakers first 5 development, last 5 calibration',
                             'human_review':False,'near_duplicate_audit':'not performed; exact PCM hashes checked'})
    print(json.dumps(audit),flush=True)
    return output


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
