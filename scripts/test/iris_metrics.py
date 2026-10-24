"""Complete-command replay and denominator-preserving finite metrics."""
from pathlib import Path
from collections import defaultdict
import json
import hashlib
import os
import uuid
from scripts.train.iris_validation import digest
ROOT=Path(__file__).resolve().parents[2]

def count_metrics(records):
    positive=[r for r in records if r['expected_index'] not in (None,0)]
    negative=[r for r in records if r['expected_index']==0]
    eligible=positive+negative
    accepted=[r for r in eligible if r['accepted_index'] not in (None,0)]
    correct=sum(r['accepted_index']==r['expected_index'] for r in positive)
    return dict(n=len(records),positive_n=len(positive),negative_n=len(negative),
        ambiguous_n=len(records)-len(eligible),raw_correct=sum(r['raw_index']==r['expected_index'] for r in positive),
        correct_accepts=correct,accepted=len(accepted),wrong_accepted_positives=sum(
            r['accepted_index'] not in (None,0,r['expected_index']) for r in positive),
        unknown_accepts=sum(r['accepted_index'] not in (None,0) for r in negative),
        false_rejects=sum(r['accepted_index'] in (None,0) for r in positive),
        recall=correct/len(positive) if positive else None,
        precision=correct/len(accepted) if accepted else None)


def summarize(records, catalog):
    result={'overall':count_metrics(records),'strata':{},'per_intent':{},'per_value':{}}
    for kind in ('human','synthetic'):
        subset=[r for r in records if r['synthetic']==(kind=='synthetic')]
        result['strata'][kind]=count_metrics(subset)
        for intent in sorted({c['command']['intent'] for c in catalog if c['command']}):
            rows=[r for r in subset if r['expected_index'] not in (None,0) and catalog[r['expected_index']]['command']['intent']==intent]
            result['per_intent'].setdefault(intent,{})[kind]=count_metrics(rows)
    groups=defaultdict(list)
    for r in records:
        if r['expected_index'] in (None,0):continue
        cmd=catalog[r['expected_index']]['command']
        for role,value in cmd['slots'].items():groups[(cmd['intent'],role,str(value),r['synthetic'])].append(r)
    result['per_value']={'/'.join(map(str,key)):count_metrics(rows) for key,rows in sorted(groups.items())}
    roles=defaultdict(lambda:dict(n=0,raw_correct=0,accepted_correct=0,wrong_accepted=0,rejected=0))
    for r in records:
        if r['expected_index'] in (None,0):continue
        expected=catalog[r['expected_index']]['command']
        raw=catalog[r['raw_index']]['command'] if r['raw_index'] is not None else None
        accepted=catalog[r['accepted_index']]['command'] if r['accepted_index'] not in (None,0) else None
        for role,value in expected['slots'].items():
            cell=roles[f"{expected['intent']}/{role}/{'synthetic' if r['synthetic'] else 'human'}"]
            cell['n']+=1
            correct=lambda cmd:bool(cmd and cmd['intent']==expected['intent'] and cmd['slots'].get(role)==value)
            cell['raw_correct']+=int(correct(raw));cell['accepted_correct']+=int(correct(accepted))
            cell['wrong_accepted']+=int(accepted is not None and not correct(accepted))
            cell['rejected']+=int(accepted is None)
    result['per_slot_role']=dict(sorted(roles.items()))
    for field in ('source','accent_group','speaker'):
        groups=defaultdict(list)
        for r in records:groups[str(r.get(field,'unknown'))].append(r)
        result[field]={key:count_metrics(rows) for key,rows in groups.items()}
    return result


def infer_rows(runtime, rows, stream=None):
    import soundfile as sf
    lookup={json.dumps(c['command'],sort_keys=True):c['command_id'] for c in runtime.catalog}
    results=[]
    for i,row in enumerate(rows,1):
        from scripts.iris_release import resolve_audio
        audio=resolve_audio(row,root=ROOT)
        if digest(audio)!=row['sha256']:raise ValueError('Audio identity mismatch: '+row['id'])
        pcm,rate=sf.read(audio,dtype='float32')
        event=runtime.recognize(pcm,rate)
        diag=event.get('diagnostic_candidate') or {}
        record={key:row.get(key) for key in ('id','speaker','split','synthetic','source','accent_group')}
        record.update(expected_index=row['class_index'],raw_index=diag.get('class_index'),
            accepted_index=lookup.get(json.dumps(event['output']['command'],sort_keys=True)) if event['output']['accepted'] else None,
            logits=diag.get('logits'),capture_fault=event['output'].get('reason')=='capture_fault',event=event)
        results.append(record)
        if stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        if i%500==0:print('REPLAY',row['split'],i,len(rows),flush=True)
    return results


def replay_rows(runtime, rows, path):
    """Reuse only complete, hash-bound replays; preserve interrupted output.

    The immutable completion receipt is published after the complete JSONL.
    A partial/unbound existing file fails closed rather than changing a test
    denominator. New work uses a unique temporary file, leaving interrupted
    temporary output available for diagnosis.
    """
    rows=list(rows);path=Path(path);receipt=path.with_suffix(path.suffix+'.complete.json')
    encode=lambda value:json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    ids=[row['id'] for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('Replay source contains duplicate sample IDs')
    code_root=Path(__file__).resolve().parents[2]
    source_names=('scripts/test/iris_metrics.py','scripts/train/iris_validation.py',
                  'scripts/train/iris_schema.py','vcm/frontend.py')
    files={'bundle.json':digest(runtime.root/'bundle.json')}
    files.update({name:digest(runtime.root/name) for name in runtime.meta['files']})
    if runtime.engine=='ncnn':
        engine=json.loads((runtime.root/'ncnn/engine.json').read_text())
        files['ncnn/engine.json']=digest(runtime.root/'ncnn/engine.json')
        files.update({'ncnn/'+name:digest(runtime.root/'ncnn'/name) for name in engine['files']})
    binding=dict(schema_version=1,rows=len(rows),manifest_rows_sha256=hashlib.sha256(encode(rows)).hexdigest(),
        ordered_ids_sha256=hashlib.sha256(encode(ids)).hexdigest(),engine=runtime.engine,model_files=files,
        source_files={name:digest(code_root/name) for name in source_names})
    def verify_records(records):
        if [r.get('id') for r in records]!=ids:raise ValueError('Incomplete or reordered replay sample IDs')
        for record,row in zip(records,rows):
            if record.get('expected_index')!=row['class_index'] or any(record.get(k)!=row.get(k) for k in
                ('speaker','split','synthetic','source','accent_group')):
                raise ValueError('Cached replay labels/strata differ from the frozen manifest')
        return records
    if path.exists() or receipt.exists():
        if not path.is_file() or not receipt.is_file():
            raise ValueError('Incomplete/unbound replay; preserve it and use a new output: '+str(path))
        complete=json.loads(receipt.read_text())
        if complete.get('binding')!=binding or complete.get('predictions_sha256')!=digest(path):
            raise ValueError('Cached replay input/model/policy/source or output hash changed: '+str(path))
        return verify_records([json.loads(line) for line in path.read_text().splitlines() if line.strip()])
    path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        records=verify_records(infer_rows(runtime,rows,stream))
        stream.flush();os.fsync(stream.fileno())
    if any(digest(runtime.root/name)!=value for name,value in files.items()):
        raise ValueError('Model/policy changed during replay; temporary output preserved')
    if any(digest(code_root/name)!=value for name,value in binding['source_files'].items()):
        raise ValueError('Replay code changed during evaluation; temporary output preserved')
    complete=dict(binding=binding,predictions_sha256=digest(temporary),completed=True)
    # Both publications are exclusive. Never overwrite another process's run.
    os.link(temporary,path);temporary.unlink()
    with receipt.open('x') as stream:stream.write(json.dumps(complete,indent=2,allow_nan=False)+'\n')
    return records
