"""Reference immutable v1 audio; generate only v2 additions and reduced Dia."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import revision as rev
from revision import HERE,ROOT,V1,CACHE,OLD_CACHE,VOICES,read_rows,write_rows,write_json,file_hash,identity,canonical


def validate_reuse_reference(previous,current):
    if canonical(previous)==canonical(current):return
    amended=dict(current)
    digest=amended.pop('supersedes_reference_sha256',None)
    for key in ('expected_command','reference_class','critical_zero_accept'):amended[key]=previous[key]
    if (previous['expected_command'] is not None or current['expected_command'] is None
            or digest!=identity(previous) or canonical(amended)!=canonical(previous)):
        raise ValueError('Reused words/reference changed without explicit v2 amendment')


def validate_reused_record(row,planned,source):
    expected={**source,**planned,'source_record_sha256':identity(source)}
    if canonical(row)!=canonical(expected):raise ValueError('Reused recording metadata changed')


def generation_complete(folder,engine,expected):
    folder=Path(folder);path=folder/f'{engine}-generation.json'
    if not path.exists():return False
    try:r=json.loads(path.read_text())
    except json.JSONDecodeError:return False
    if r.get('complete') is not True:return False
    if r.get('generated')!=expected or r.get('planned')!=expected:
        raise ValueError('Generation receipt has unexpected counts')
    if file_hash(folder/f'{engine}-manifest.jsonl')!=r['manifest_sha256']:
        raise ValueError('Completed generation manifest changed')
    return True


def wake_assignments(commands):
    selected=rev.dia_verbatims(commands);rows=[]
    for phase in (1,2):
        for uc in range(1,11):
            choices=[r for r in commands if r['phase']==phase and r['use_case']==f'UC-{uc:02}'
                     and r['expected_command'] is not None and r['id'] in selected]
            ref=sorted(choices,key=lambda r:(not r['id'].startswith('V2-'),identity(r['id'])))[0]
            rows.append(dict(id=f'WAKE-{uc:02}-{phase}',text=f'Hey Jarvis. {ref["text"]}.',
                case_type='wake_command',expected_command=ref['expected_command'],use_case=ref['use_case'],
                command_verbatim_id=ref['id'],phase=phase,continuous_generated_utterance=True))
    reference=next(r for r in commands if r['text']=='turn on kitchen lamp')
    for key,text,kind,expected in [
        ('MARVIN','Hey Marvin. Turn on kitchen lamp.','wrong_wake',None),
        ('TRAVIS','Hey Travis. Turn on kitchen lamp.','wrong_wake',None),
        ('ONLY','Hey Jarvis.','wake_only',None),
        ('NO-PAUSE','Hey Jarvis turn on kitchen lamp.','no_pause',reference['expected_command'])]:
        rows.append(dict(id=f'WAKE-{key}',text=text,case_type=kind,expected_command=expected,use_case='UC-03',
            command_verbatim_id=reference['id'],phase=1,continuous_generated_utterance=True,
            diagnostic_only=kind=='no_pause',pause_verified=False))
    plan=[]
    for engine in ('qwen','parler','dia'):
        for voice in VOICES:
            if voice['split']=='train':continue
            for ref in rows:
                key=f'{engine}/{voice["id"]}/{ref["id"]}'
                plan.append(dict(recording_id=key,verbatim_id=ref['id'],engine=engine,
                    voice_group=voice['id'],split=voice['split'],accent_requested=voice['accent'],accent_verified=False,
                    seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16),synthetic=True,
                    audio=str(CACHE/'audio'/f'{key}.wav'),native_audio=str(CACHE/'native'/f'{key}.wav'),source_version='v2'))
    return rows,plan


def generate(engine):
    import torch
    import synthesis
    from assets import verify_assets
    rev.validate();asset_hash=verify_assets(engine)
    plan=read_rows(HERE/'generation-plan.jsonl')+read_rows(HERE/'wake-generation-plan.jsonl')
    plan=[r for r in plan if r['engine']==engine and r['source_version']=='v2']
    refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')+read_rows(HERE/'wake-verbatims.jsonl')}
    path=CACHE/f'generated-{engine}.jsonl'
    existing_rows=read_rows(path) if path.exists() else []
    existing={r['recording_id']:r for r in existing_rows}
    if len(existing)!=len(existing_rows):raise ValueError('Duplicate generation provenance')
    recipe_sha=file_hash(OLD_CACHE/f'{engine}-recipe.json')
    source_sha=file_hash(Path(__file__))
    binding=identity(dict(protocol=file_hash(HERE/'protocol.json'),source=source_sha,recipe=recipe_sha,assets=asset_hash))
    marker=CACHE/f'{engine}-v2-generation-started.json'
    if marker.exists() and json.loads(marker.read_text())['binding']!=binding:
        raise ValueError('V2 generation source or protocol changed')
    expected={r['recording_id']:r for r in plan}
    for key,row in existing.items():
        planned=expected.get(key)
        if (planned is None or any(row.get(k)!=v for k,v in planned.items())
            or row.get('generation_binding')!=binding or row.get('generation_source_sha256')!=source_sha
            or row.get('generation_recipe_sha256')!=recipe_sha or row.get('tts_assets_sha256')!=asset_hash
            or row.get('model_identity_sha256')!=file_hash(OLD_CACHE/f'{engine}-model-identity.json')
            or row.get('verbatim_sha256')!=identity(refs[row['verbatim_id']])):
            raise ValueError('Previously generated v2 provenance changed')
        if file_hash(row['audio'])!=row['audio_sha256'] or file_hash(row['native_audio'])!=row['native_sha256']:
            raise ValueError('Previously generated v2 audio changed')
    write_json(marker,dict(binding=binding,planned=len(plan),batch_size=4,source_sha256=source_sha))
    def complete():
        receipt=dict(complete=len(existing)==len(plan),generated=len(existing),planned=len(plan),
            manifest_sha256=file_hash(path),binding=binding)
        write_json(CACHE/f'{engine}-v2-generation.json',receipt)
        return receipt
    if len(existing)==len(plan):return complete()
    torch.set_num_threads(4)
    generator,model=synthesis.load_engine(engine)
    if model!=json.loads((OLD_CACHE/f'{engine}-model-identity.json').read_text()):
        raise ValueError('Generator model drift')
    for voice in VOICES:
        records=[r for r in plan if r['voice_group']==voice['id']]
        for offset in range(0,len(records),4):
            batch=records[offset:offset+4]
            if all(r['recording_id'] in existing for r in batch):continue
            began=time.monotonic();torch.cuda.empty_cache()
            for attempt in range(20 if engine=='dia' else 3):
                seed=(batch[0]['seed']+attempt)%2**32;torch.manual_seed(seed)
                try:
                    waves,rate=generator([refs[r['verbatim_id']]['text'] for r in batch],voice)
                    if len(waves)!=len(batch):raise ValueError('Wrong batch size')
                    prepared=[synthesis.prepare_audio(w,rate) for w in waves];break
                except ValueError as exc:
                    with (CACHE/f'{engine}-attempts.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(canonical(dict(ids=[r['recording_id'] for r in batch],seed=seed,attempt=attempt+1,error=str(exc)))+'\n')
                    if attempt+1==(20 if engine=='dia' else 3):raise
            metadata=getattr(generator,'last_metadata',[{} for _ in batch])
            if len(metadata)!=len(batch):raise ValueError('Wrong generation metadata batch size')
            for row,wave,ready,meta in zip(batch,waves,prepared,metadata):
                if row['recording_id'] in existing:continue
                item=synthesis.persist_audio(CACHE,row,wave,rate,ready)
                item.update(batch_seed=seed,batch_size=4,batch_ids=[r['recording_id'] for r in batch],generation_attempt=attempt+1,
                    generation_metadata=meta,generation_recipe_sha256=recipe_sha,generation_source_sha256=source_sha,
                    generation_binding=binding,tts_assets_sha256=asset_hash,model_identity_sha256=file_hash(OLD_CACHE/f'{engine}-model-identity.json'),
                    voice_anchor_sha256=None if engine=='parler' else file_hash(OLD_CACHE/'anchors'/f'{voice["id"]}.wav'),
                    voice_conditioning={'type':'named_voice_description','name':voice['parler']} if engine=='parler' else {'type':'original_synthetic_audio_anchor'},
                    verbatim_sha256=identity(refs[row['verbatim_id']]))
                with path.open('a',encoding='utf-8') as stream:
                    stream.write(canonical(item)+'\n');stream.flush();os.fsync(stream.fileno())
                existing[row['recording_id']]=item
            print('GENERATED_V2',engine,len(existing),'/',len(plan),'batch_seconds',round(time.monotonic()-began,2),flush=True)
    return complete()


def assemble():
    import synthbench
    rev.validate();plan=read_rows(HERE/'generation-plan.jsonl');wake_plan=read_rows(HERE/'wake-generation-plan.jsonl')
    old_plan=read_rows(V1/'generation-plan.jsonl');old_refs={r['id']:r for r in read_rows(V1/'verbatims.jsonl')}
    refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    audit_sources={}
    for engine in ('qwen','parler','dia'):
        fresh={r['recording_id']:r for r in read_rows(CACHE/f'generated-{engine}.jsonl')}
        original={}
        if engine!='dia':
            if not generation_complete(OLD_CACHE,engine,13860):raise ValueError('Original generator incomplete')
            original={r['recording_id']:r for r in read_rows(OLD_CACHE/f'{engine}-manifest.jsonl')}
            synthbench.audit_manifest(list(original.values()),[r for r in old_plan if r['engine']==engine])
            audit_sources[engine]=dict(path=str(OLD_CACHE/f'{engine}-manifest.jsonl'),sha256=file_hash(OLD_CACHE/f'{engine}-manifest.jsonl'))
        merged=[]
        for row in [r for r in plan if r['engine']==engine]:
            if row['source_version']=='v1':
                validate_reuse_reference(old_refs[row['verbatim_id']],refs[row['verbatim_id']])
                source=original[row['recording_id']]
                merged.append({**source,**row,'source_record_sha256':identity(source)})
            else:merged.append(fresh[row['recording_id']])
        write_rows(CACHE/f'{engine}-manifest.jsonl',merged)
        write_rows(CACHE/f'wake-{engine}-manifest.jsonl',[fresh[r['recording_id']] for r in wake_plan if r['engine']==engine])
    write_json(CACHE/'reuse-audit.json',dict(sources=audit_sources,v1_protocol_sha256=file_hash(V1/'protocol.json'),
        original_verbatims_sha256=file_hash(V1/'verbatims.jsonl'),no_test_recognition=True))
    result=audit_all_splits()
    from wake_benchmark import audit_wake
    audit_wake('dev')  # Audits all wake files/splits before any fitting.
    return {**result,'wake_recordings':len(wake_plan),'wake_identity_audit_passed':True}


def audit_all_splits():
    import soundfile as sf
    import numpy as np
    rev.validate();plan=read_rows(HERE/'generation-plan.jsonl');expected={r['recording_id']:r for r in plan}
    refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    reuse=json.loads((CACHE/'reuse-audit.json').read_text());original={}
    for e,source in reuse['sources'].items():
        if file_hash(source['path'])!=source['sha256']:raise ValueError('Original manifest changed after reuse audit')
        original.update({r['recording_id']:r for r in read_rows(source['path'])})
    rows=[]
    for e in ('qwen','parler','dia'):rows+=read_rows(CACHE/f'{e}-manifest.jsonl')
    if len(rows)!=len(expected) or {r['recording_id'] for r in rows}!=set(expected):raise ValueError('Incomplete v2 corpus')
    hashes=set();voices=defaultdict(set);anchors=defaultdict(set)
    for row in rows:
        planned=expected[row['recording_id']]
        if any(row.get(k)!=v for k,v in planned.items()):raise ValueError('V2 assignment drift')
        if row['source_version']=='v1':
            validate_reused_record(row,planned,original[row['recording_id']])
        else:
            if (row['generation_source_sha256']!=file_hash(Path(__file__)) or row['verbatim_sha256']!=identity(refs[row['verbatim_id']])
                    or row['generation_recipe_sha256']!=file_hash(OLD_CACHE/f'{row["engine"]}-recipe.json')
                    or row['model_identity_sha256']!=file_hash(OLD_CACHE/f'{row["engine"]}-model-identity.json')
                    or row['tts_assets_sha256']!=file_hash(V1/'tts-assets.json')):raise ValueError('New generation identity drift')
        for key,digest in [('audio','audio_sha256'),('native_audio','native_sha256')]:
            path=Path(row[key]).resolve()
            if not any(path.is_relative_to(p.resolve()) for p in (CACHE,OLD_CACHE)):raise ValueError('Audio escapes study roots')
            if file_hash(path)!=row[digest]:raise ValueError('Audio bytes changed')
        pcm,rate=sf.read(row['audio'],dtype='float32');digest=hashlib.sha256(pcm.tobytes()).hexdigest()
        if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all() or digest!=row['pcm_sha256']:raise ValueError('Invalid PCM')
        if digest in hashes:raise ValueError('Duplicate PCM in synthetic corpus')
        hashes.add(digest);voices[row['voice_group']].add(row['split'])
        if row['engine']!='parler':
            anchor=file_hash(OLD_CACHE/'anchors'/f'{row["voice_group"]}.wav')
            if row.get('voice_anchor_sha256')!=anchor:raise ValueError('Voice anchor drift')
            anchors[anchor].add(row['split'])
    if any(len(s)>1 for s in list(voices.values())+list(anchors.values())):raise ValueError('Cross-split voice leakage')
    return dict(recordings=len(rows),unique_pcm=len(hashes),synthetic=True,text_verified_by_listener=False,
        manifests={e:file_hash(CACHE/f'{e}-manifest.jsonl') for e in ('qwen','parler','dia')},
        reduced_dia=True,third_engine_exhaustive=False)


def load_split(split):
    audit_all_splits();refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    rows=[]
    for e in ('qwen','parler','dia'):
        rows += [{**r,**refs[r['verbatim_id']]} for r in read_rows(CACHE/f'{e}-manifest.jsonl') if r['split']==split]
    return rows
