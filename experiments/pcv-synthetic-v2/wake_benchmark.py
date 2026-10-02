"""Continuous synthetic wake replay, separately reserved from command-only test."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import revision as rev
from revision import CACHE,HERE,ROOT,OLD_CACHE,read_rows,write_json,file_hash,identity,canonical
from wake_gate import from_bundle,replay_pcm


def score_trial(reference,events):
    commands=[e for e in events if e.get('stage')=='command']
    emitted=[e for e in commands if e.get('accepted') is True]
    detected=any(e.get('stage')=='wake' and e.get('status')=='detected' for e in events)
    expected=reference['expected_command'];kind=reference['case_type']
    fault=any(e.get('reason')=='capture_fault' for e in events)
    false_wake=kind in ('no_wake','wrong_wake') and detected
    if expected is not None:
        correct=detected and len(commands)==1 and len(emitted)==1 and canonical(emitted[0]['command'])==canonical(expected)
    else:
        correct=not emitted and not false_wake and (detected if kind=='wake_only' else True)
    return {**reference,'events':events,'accepted':bool(emitted),'emissions':emitted,
        'wake_detected':detected,'false_wake':false_wake,'capture_fault':fault,'correct':bool(correct and not fault)}


def summarize(rows):
    scored=[r for r in rows if not r.get('diagnostic_only',False)]
    positives=[r for r in scored if r['expected_command'] is not None]
    negatives=[r for r in scored if r['expected_command'] is None]
    wake_only=[r for r in negatives if r['case_type']=='wake_only']
    accepted=[r for r in scored if r['accepted']]
    correct=sum(r['correct'] for r in positives)
    groups=defaultdict(list)
    for r in positives:groups[f'{r.get("phase",1)}/{r["engine"]}/{r["use_case"]}'].append(r)
    strata={k:dict(correct=sum(r['correct'] for r in v),trials=len(v),recall=sum(r['correct'] for r in v)/len(v)) for k,v in groups.items()}
    result=dict(trials=len(scored),diagnostic_trials=len(rows)-len(scored),positives=len(positives),negatives=len(negatives),
        positive_correct=correct,positive_recall=correct/len(positives) if positives else None,
        negative_correct=sum(r['correct'] for r in negatives),
        wake_only_trials=len(wake_only),wake_only_correct=sum(r['correct'] for r in wake_only),
        accepted=len(accepted),accepted_precision=correct/len(accepted) if accepted else None,
        negative_false_accepts=sum(r['accepted'] for r in negatives),false_wakes=sum(r['false_wake'] for r in scored),
        invalid_trials=sum(r['capture_fault'] for r in scored),phase_engine_use_case=strata)
    result['wake_gates_passed']=(len(strata)==60 and all(s['recall']>=.90 for s in strata.values())
        and (result['accepted_precision'] or 0)>=.99 and result['negative_false_accepts']==0
        and result['negative_correct']==len(negatives) and len(wake_only)>0
        and result['false_wakes']==0 and result['invalid_trials']==0)
    return result


def reserve_test_access(binding,output):
    # A new model must never grant a second read of the same final-test corpus.
    dataset=dict(plan=file_hash(HERE/'wake-generation-plan.jsonl'),
        references=file_hash(HERE/'wake-verbatims.jsonl'),
        wake_manifests={e:file_hash(CACHE/f'wake-{e}-manifest.jsonl') for e in ('qwen','parler','dia')},
        command_manifests={e:file_hash(CACHE/f'{e}-manifest.jsonl') for e in ('qwen','parler','dia')})
    reservation=CACHE/'wake-test-access'/f'{identity(dataset)}.json'
    reservation.parent.mkdir(exist_ok=True)
    with reservation.open('x',encoding='utf-8') as stream:
        json.dump(dict(dataset=dataset,binding=binding,output=str(output),state='RESERVED_PRESERVE_ON_FAILURE'),stream)
        stream.flush();os.fsync(stream.fileno())
    return reservation


def audit_wake(split):
    import soundfile as sf
    rev.validate();refs={r['id']:r for r in read_rows(HERE/'wake-verbatims.jsonl')}
    plan={r['recording_id']:r for r in read_rows(HERE/'wake-generation-plan.jsonl')}
    rows=[];seen=set();pcm_seen=set();voices=defaultdict(set);anchors=defaultdict(set)
    for e in ('qwen','parler','dia'):
        for r in read_rows(CACHE/f'wake-{e}-manifest.jsonl'):
            if r['recording_id'] not in plan or r['recording_id'] in seen:raise ValueError('Unexpected wake recording')
            seen.add(r['recording_id'])
            if any(r.get(k)!=v for k,v in plan[r['recording_id']].items()):raise ValueError('Wake assignment changed')
            if r['verbatim_sha256']!=identity(refs[r['verbatim_id']]):raise ValueError('Wake reference changed')
            for key,digest in [('audio','audio_sha256'),('native_audio','native_sha256')]:
                if not Path(r[key]).resolve().is_relative_to(CACHE.resolve()) or file_hash(r[key])!=r[digest]:
                    raise ValueError('Wake audio identity changed')
            if r['pcm_sha256'] in pcm_seen:raise ValueError('Duplicate wake PCM')
            pcm_seen.add(r['pcm_sha256'])
            pcm,rate=sf.read(r['audio'],dtype='float32')
            if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all() or hashlib.sha256(pcm.tobytes()).hexdigest()!=r['pcm_sha256']:
                raise ValueError('Invalid wake PCM')
            voices[r['voice_group']].add(r['split'])
            if e!='parler':
                anchor=file_hash(OLD_CACHE/'anchors'/f'{r["voice_group"]}.wav')
                if r.get('voice_anchor_sha256')!=anchor:raise ValueError('Wake voice anchor changed')
                anchors[anchor].add(r['split'])
            if (r['generation_source_sha256']!=file_hash(HERE/'corpus.py') or
                r['generation_recipe_sha256']!=file_hash(OLD_CACHE/f'{e}-recipe.json') or
                r['model_identity_sha256']!=file_hash(OLD_CACHE/f'{e}-model-identity.json') or
                r['tts_assets_sha256']!=file_hash(rev.V1/'tts-assets.json') or
                r['generation_binding']!=identity(dict(protocol=file_hash(HERE/'protocol.json'),
                    source=file_hash(HERE/'corpus.py'),recipe=file_hash(OLD_CACHE/f'{e}-recipe.json'),
                    assets=file_hash(rev.V1/'tts-assets.json')))):
                raise ValueError('Wake generation provenance changed')
            rows.append({**r,**refs[r['verbatim_id']]})
    if seen!=set(plan):raise ValueError('Incomplete wake corpus')
    if any(len(s)>1 for s in list(voices.values())+list(anchors.values())):raise ValueError('Cross-split wake voice leakage')
    return [r for r in rows if r['split']==split]


def replay(gate,row):
    import soundfile as sf
    pcm,rate=sf.read(row['audio'],dtype='float32')
    if rate!=16000 or pcm.ndim!=1 or not np.isfinite(pcm).all() or hashlib.sha256(pcm.tobytes()).hexdigest()!=row['pcm_sha256']:
        raise ValueError('Wake replay PCM changed')
    # Explicit stream context and timeout flush, never speech concatenation.
    began=time.perf_counter();events=replay_pcm(gate,pcm)
    result=score_trial(row,events)
    result.update(inference_seconds=time.perf_counter()-began,source_seconds=len(pcm)/16000,
        warmup_silence_seconds=2,timeout_flush_seconds=6,ort_provider='CPUExecutionProvider')
    return result


def evaluate(bundle,output,split='test'):
    from synthbench import audit_all_splits,load_split
    output=Path(output)
    if output.exists():raise FileExistsError('Wake evaluation is immutable')
    audit_all_splits();rows=audit_wake(split)
    commands={r['recording_id']:r for r in load_split(split)}
    # Replay the actual command-only WAV as a no-wake negative, without editing it.
    for r in list(rows):
        if r['case_type']!='wake_command':continue
        key=f'{r["engine"]}/{r["voice_group"]}/{r["command_verbatim_id"]}'
        source=commands[key]
        rows.append({**source,'recording_id':f'no-wake/{key}','case_type':'no_wake',
                     'expected_command':None,'original_command':source['expected_command']})
    gate=from_bundle(bundle,ROOT/'artifacts')
    binding=gate.command.binding
    if split=='test':
        final=json.loads((CACHE/'final-test/results.json').read_text())
        if final['binding']!=binding or not final['complete']:raise ValueError('Command test must finish before wake test')
        reserve_test_access(binding,output)
    output.mkdir(parents=True);predictions=[]
    with (output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for i,row in enumerate(rows):
            result=replay(gate,row);predictions.append(result);stream.write(canonical(result)+'\n');stream.flush()
            if i%20==0:print('WAKE_REPLAY',i,'/',len(rows),flush=True)
    result=summarize(predictions)
    result.update(binding=binding,complete=len(predictions)==len(rows),split=split,
        predictions_sha256=file_hash(output/'predictions.jsonl'),synthetic_only=True,human_qualified=False,
        wake_manifest_sha256={e:file_hash(CACHE/f'wake-{e}-manifest.jsonl') for e in ('qwen','parler','dia')})
    write_json(output/'results.json',result);return result
