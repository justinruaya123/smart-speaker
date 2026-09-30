"""Synthetic-only ONNX replay and metrics; never grants human qualification."""
from collections import defaultdict
from functools import lru_cache
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from einops import pack

from study import CACHE, HERE, PCV, Contract, canonical, file_hash, identity, read_rows, write_json, validate_frozen_study
from pcvcm.contract import pronounce
from pcvcm.runtime import StreamEngine, OnnxEncoder
from pcvcm.endpoint import EndpointConfig
from pcvcm.decision import decide, candidate_allowed

EXTRA_WORDS = {'could':['K','UH','D'], 'you':['Y','UW'], 'would':['W','UH','D'],
    'like':['L','AY','K'], 'can':['K','AE','N'], 'know':['N','OW'],
    'hey':['HH','EY'], 'assistant':['AH','S','IH','S','T','AH','N','T']}


def compile_synthetic_catalog(rows):
    c = Contract(PCV/'spec')
    words = {**json.loads((PCV/'configs/lexicon.json').read_text())['words'], **EXTRA_WORDS}
    entries, seen = [], {}
    for row in rows:
        if row['expected_command'] is not None:
            c.validate(row['expected_command'])
        phones = pronounce(row['text'], words)
        key = tuple(phones)
        if key in seen and seen[key] != canonical(row['expected_command']):
            raise ValueError('Acoustic catalog collision across different commands')
        seen[key] = canonical(row['expected_command'])
        entries.append(dict(id=row['id'], text=row['text'], use_case=row['use_case'],
                            phones=phones, command=row['expected_command']))
    return dict(version='pcv-synthetic-v1', entries=entries,
                base_contract_sha256=c.sha256, pronunciation_sha256=identity(words),
                extra_pronunciations=EXTRA_WORDS, accent_or_pronunciation_verified=False)


def audit_manifest(rows, plan, root=CACHE):
    import soundfile as sf
    ids = [r['recording_id'] for r in rows]
    expected = {r['recording_id']:r for r in plan}
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f'Incomplete or duplicate generated corpus: {len(set(ids))}/{len(expected)}')
    hashes, voices = defaultdict(set), defaultdict(set)
    verbatim_hash=file_hash(HERE/'verbatims.jsonl')
    current_refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    compatible={}
    for source,digest in json.loads((HERE/'protocol.json').read_text()).get('compatible_previous_verbatims',{}).items():
        if file_hash(HERE/source)!=digest:
            raise ValueError('Preserved prior verbatim source changed')
        compatible[digest]={r['id']:r for r in read_rows(HERE/source)}
    legacy_path=HERE/'revisions/initial-generation-provenance.json'
    legacy=json.loads(legacy_path.read_text())['records'] if legacy_path.exists() else {}
    model_hashes={e:file_hash(CACHE/f'{e}-model-identity.json') for e in ('qwen','parler','dia')
                  if (CACHE/f'{e}-model-identity.json').exists()}
    recipe_hashes={e:file_hash(CACHE/f'{e}-recipe.json') for e in ('qwen','parler','dia')
                   if (CACHE/f'{e}-recipe.json').exists()}
    amendments=[]
    for amendment_name,previous_recipe in [('dia-retry-amendment.json','dia-recipe-before-retry.json'),
            ('dia-retry-budget-amendment.json','dia-recipe-three-attempts.json')]:
        amendment_path=HERE/'revisions'/amendment_name
        if amendment_path.exists():
            item=json.loads(amendment_path.read_text())
            if file_hash(HERE/'revisions'/previous_recipe)!=item['previous_recipe_sha256']:
                raise ValueError('Preserved generation recipe changed')
            amendments.append(item)
    for row in rows:
        for key in ['verbatim_id','engine','voice_group','split','seed','audio','native_audio']:
            if row[key] != expected[row['recording_id']][key]:
                raise ValueError(f'Generation assignment changed: {key}')
        if row.get('synthetic') is not True:
            raise ValueError('Synthetic provenance missing')
        if 'generation_recipe_sha256' in row:
            preserved=any(row['engine']=='dia' and row['generation_recipe_sha256']==amendment.get('previous_recipe_sha256')
                and amendment.get('retained_records',{}).get(row['recording_id'])==row['audio_sha256'] for amendment in amendments)
            if (row['generation_recipe_sha256']!=recipe_hashes.get(row['engine']) and not preserved) or row.get('model_identity_sha256')!=model_hashes.get(row['engine']):
                raise ValueError('TTS model or generation recipe changed')
        else:
            receipt=legacy.get(row['recording_id'],{})
            if any(receipt.get(k)!=row.get(k) for k in ('audio_sha256','native_sha256','batch_size')) or receipt.get('model_identity_sha256')!=model_hashes.get(row['engine']):
                raise ValueError('Initial generated recording lacks a valid preserved provenance binding')
        old_hash=row.get('verbatim_sha256')
        if old_hash!=verbatim_hash:
            previous=compatible.get(old_hash,{}).get(row['verbatim_id'])
            if previous is None or canonical(previous)!=canonical(current_refs[row['verbatim_id']]):
                raise ValueError('Generated recording binds a changed verbatim')
        audio, native = Path(root)/row['audio'], Path(root)/row['native_audio']
        if file_hash(audio) != row['audio_sha256'] or file_hash(native) != row['native_sha256']:
            raise ValueError('Generated audio changed')
        pcm, rate = sf.read(audio, dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError('Invalid derived PCM')
        digest = hashlib.sha256(pcm.tobytes()).hexdigest()
        if digest != row['pcm_sha256']:
            raise ValueError('Decoded PCM changed')
        hashes[digest].add(row['recording_id'])
        voices[row['voice_group']].add(row['split'])
    if any(len(x)>1 for x in voices.values()):
        raise ValueError('Cross-split voice leakage')
    if any(len(x)>1 for x in hashes.values()):
        raise ValueError('Duplicate generated PCM; preserve and investigate')
    return dict(recordings=len(rows), unique_pcm=len(hashes), synthetic=True,
                original_humans=0, text_verified_by_listener=False)


def load_split(split):
    validate_frozen_study()
    plan = [r for r in read_rows(HERE/'generation-plan.jsonl') if r['split']==split]
    all_rows = []
    for engine in ('qwen','parler','dia'):
        path=CACHE/f'{engine}-manifest.jsonl'
        if path.exists():
            all_rows += [r for r in read_rows(path) if r['split']==split]
    audit_manifest(all_rows, plan)
    refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    return [{**r, **refs[r['verbatim_id']]} for r in all_rows]


def audit_all_splits():
    """Identity audit may read test hashes/PCM, never fit test features or labels."""
    validate_frozen_study()
    from assets import verify_assets
    for engine in ('qwen','parler','dia'):verify_assets(engine)
    plan=read_rows(HERE/'generation-plan.jsonl')
    rows=[]
    for engine in ('qwen','parler','dia'):
        path=CACHE/f'{engine}-manifest.jsonl'
        if path.exists():
            rows.extend(read_rows(path))
    result=audit_manifest(rows,plan)
    anchors=defaultdict(set)
    for row in rows:
        if row['engine']=='parler':
            from study import VOICES
            expected_voice=next(v['parler'] for v in VOICES if v['id']==row['voice_group'])
            condition=row.get('voice_conditioning')
            if condition is not None and condition!={'type':'named_voice_description','name':expected_voice}:
                raise ValueError('Parler named voice identity changed')
            # Initial Parler rows retain a Qwen hash as an unused split marker;
            # Parler never conditions on that audio. Its actual named voices are
            # frozen in the generation recipe and initial provenance receipt.
            continue
        digest=row.get('voice_anchor_sha256')
        if not digest:
            raise ValueError('Missing voice anchor identity')
        if file_hash(CACHE/'anchors'/f'{row["voice_group"]}.wav')!=digest:
            raise ValueError('Voice anchor identity mismatch')
        anchors[digest].add(row['split'])
    if any(len(v)>1 for v in anchors.values()):
        raise ValueError('Actual voice anchor crosses splits')
    result['manifests']={e:file_hash(CACHE/f'{e}-manifest.jsonl') for e in ('qwen','parler','dia')}
    result['tts_assets_sha256']=file_hash(HERE/'tts-assets.json')
    return result


def metrics(rows):
    positives=[r for r in rows if r['expected_command'] is not None]
    negatives=[r for r in rows if r['expected_command'] is None]
    accepted=[r for r in rows if r['output']['accepted'] is True]
    correct=[r for r in accepted if r['expected_command'] is not None and canonical(r['output']['command'])==canonical(r['expected_command'])]
    raw=sum(canonical(r['candidate'])==canonical(r['expected_command']) for r in positives)
    intent=sum((r['candidate'] or {}).get('intent')==r['expected_command']['intent'] for r in positives)
    false=sum(r['output']['accepted'] for r in negatives)
    return dict(trials=len(rows), positives=len(positives), negatives=len(negatives),
        accepted=len(accepted), correct_accepted=len(correct), raw_exact=raw,
        intent_correct=intent, parameter_errors=intent-raw,
        raw_exact_recall=raw/len(positives) if positives else None,
        accepted_recall=len(correct)/len(positives) if positives else None,
        accepted_precision=len(correct)/len(accepted) if accepted else None,
        negative_false_accepts=false, negative_far=false/len(negatives) if negatives else None,
        critical_false_accepts=sum(r['output']['accepted'] and r.get('critical_zero_accept',False) for r in negatives),
        invalid_trials=sum(bool(r.get('capture_fault')) or not r.get('endpoint_detected',True) for r in rows),
        valid_negative_rejections=sum(not r['output']['accepted'] and not r.get('capture_fault')
            and r.get('endpoint_detected',True) for r in negatives),
        endpoint_failures=sum(not r.get('endpoint_detected',True) for r in rows))


@lru_cache(maxsize=1)
def intent_use_cases():
    return Contract(PCV/'spec').intent_use_case


def use_case_metrics(rows, uc):
    result=metrics([r for r in rows if r['use_case']==uc])
    emitted=[r for r in rows if r['output']['accepted'] and
             intent_use_cases().get((r['output']['command'] or {}).get('intent'))==uc]
    correct=sum(r['expected_command'] is not None and canonical(r['output']['command'])==canonical(r['expected_command']) for r in emitted)
    result['accepted']=len(emitted)
    result['accepted_precision']=correct/len(emitted) if emitted else None
    result['emitted_as_use_case']=uc
    return result


def summarize(rows):
    report={'overall':metrics(rows), 'synthetic_only':True, 'human_qualified':False}
    for dimension in ('phase','use_case','verbatim_id','engine','accent_requested'):
        groups=defaultdict(list)
        for row in rows:
            groups[str(row[dimension])].append(row)
        report[dimension]={k:metrics(v) for k,v in sorted(groups.items())}
        if dimension=='use_case':
            report[dimension]={uc:use_case_metrics(rows,uc) for uc in sorted(groups)}
    # Intersection matters: pooled engine scores can hide one broken use case.
    groups=defaultdict(list)
    for r in rows:
        groups[f'{r["phase"]}/{r["engine"]}/{r["use_case"]}'].append(r)
    report['phase_engine_use_case']={k:metrics(v) for k,v in sorted(groups.items())}
    for key,reference_rows in groups.items():
        first=reference_rows[0]
        stratum=[r for r in rows if (r['phase'],r['engine'])==(first['phase'],first['engine'])]
        report['phase_engine_use_case'][key]=use_case_metrics(stratum,first['use_case'])
    splits={r.get('split') for r in rows}
    expected_ids=set()
    if len(splits)==1 and next(iter(splits)) in ('train','dev','calibration','test'):
        expected_ids={r['recording_id'] for r in read_rows(HERE/'generation-plan.jsonl') if r['split'] in splits}
    observed_ids=[r.get('recording_id') for r in rows]
    complete=bool(expected_ids) and len(observed_ids)==len(expected_ids) and set(observed_ids)==expected_ids
    report['coverage_complete']=complete
    report['synthetic_gates_passed']=complete and len(groups)==60 and report['overall']['invalid_trials']==0 and report['overall']['critical_false_accepts']==0 and all(
        (m['accepted_precision'] or 0)>=.99 and (m['accepted_recall'] or 0)>=.90
        and m['negative_far'] is not None and m['negative_far']<=.01
        for m in report['phase_engine_use_case'].values())
    return report


def replay(runtime, row):
    import soundfile as sf
    pcm, rate=sf.read(CACHE/row['audio'], dtype='float32')
    if rate!=16000:
        raise ValueError('Source rate changed')
    runtime.reset()
    # Explicit endpoint flush, not hidden augmentation of the retained recording.
    delivered,_=pack([pcm,np.zeros(12800,dtype=np.float32)],'*')
    begin=time.perf_counter()
    for start in range(0,len(delivered),640):
        runtime.feed(delivered[start:start+640],start)
        if runtime.finalized or runtime.endpoint.fault:
            break
    output=runtime.eof()
    elapsed=time.perf_counter()-begin
    return {k:row[k] for k in ('recording_id','verbatim_id','phase','use_case','engine',
        'accent_requested','voice_group','split','expected_command','audio_sha256','pcm_sha256','critical_zero_accept')} | dict(
        output=output, candidate=runtime.candidate['command'], diagnostic=dict(runtime.diagnostic),
        endpoint_detected=runtime.finalized and not runtime.endpoint.fault,
        capture_fault=runtime.endpoint.fault, inference_seconds=elapsed,
        source_seconds=len(pcm)/16000, endpoint_flush_samples=12800,
        binding=runtime.binding)


def load_runtime(bundle):
    state_shapes = [(1,40,2),(1,64,4),(1,64,8),(1,64,16),(1,64,32),(1,64,64)]
    bundle=Path(bundle)
    metadata=json.loads((bundle/'synthetic-bundle.json').read_text())
    for name,digest in metadata['files'].items():
        if file_hash(bundle/name)!=digest:
            raise ValueError('Synthetic bundle file changed')
    sources=json.loads((bundle/'source-identity.json').read_text())
    from study import ROOT
    from pcvcm.paths import below
    for name,digest in sources.items():
        if file_hash(below(ROOT,name))!=digest:
            raise ValueError('Executing runtime/source identity changed')
    validate_frozen_study()
    if file_hash(HERE/'protocol.json')!=file_hash(bundle/'protocol.json'):
        raise ValueError('Bundle belongs to another frozen synthetic study')
    audit=json.loads((bundle/'global-audit.json').read_text())
    for engine,digest in audit['manifests'].items():
        if file_hash(CACHE/f'{engine}-manifest.jsonl')!=digest:
            raise ValueError('Corpus manifest changed after training')
    binding=identity(metadata['files'])
    if binding!=metadata['binding']:
        raise ValueError('Synthetic bundle binding mismatch')
    catalog=json.loads((bundle/'catalog.json').read_text())
    policy=json.loads((bundle/'policy.json').read_text())
    if policy['binding']!=binding or file_hash(bundle/'policy.json')!=metadata['policy_sha256']:
        raise ValueError('Policy binding mismatch')
    if 'waveform_parity' in policy:
        receipt=policy['waveform_parity']
        if file_hash(receipt['path'])!=receipt['sha256']:
            raise ValueError('Waveform parity receipt changed')
        validate_parity_receipt(receipt['path'],binding)
    return StreamEngine(Contract(PCV/'spec'),catalog,
        OnnxEncoder(bundle/'model.onnx',state_shapes),
        json.loads((bundle/'normalization.json').read_text()),
        EndpointConfig(**json.loads((bundle/'endpoint.json').read_text())),policy,binding)


def validate_parity_receipt(path,binding):
    path=Path(path)
    report=json.loads(path.read_text())
    expected={r['recording_id'] for r in read_rows(HERE/'generation-plan.jsonl') if r['split']=='dev'}
    if (report.get('binding')!=binding or report.get('complete') is not True
        or report.get('recordings')!=len(expected) or not np.isfinite(report.get('max_error',np.nan))
        or report.get('max_error',1)>2e-5
        or file_hash(path.parent/'rows.jsonl')!=report.get('rows_sha256')):
        raise ValueError('Complete waveform parity is required before calibration/test')
    rows=read_rows(path.parent/'rows.jsonl')
    if (len(rows)!=len(expected) or {r['recording_id'] for r in rows}!=expected
        or any(r.get('outputs_equal') is not True or not np.isfinite(r.get('max_error',np.nan))
            or not 0<=r['max_error']<=2e-5 for r in rows)
        or max(r['max_error'] for r in rows)!=report['max_error']):
        raise ValueError('Waveform parity rows do not establish complete numerical/decision equivalence')
    return report


def evaluate(bundle, split, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError('Evaluation receipts are immutable')
    records=load_split(split)
    started=time.perf_counter()
    runtime=load_runtime(bundle)
    load_seconds=time.perf_counter()-started
    if split=='test':
        if 'waveform_parity' not in runtime.policy or not runtime.policy.get('calibration_predictions_sha256'):
            raise ValueError('Calibration and full waveform parity must precede final test')
        audit=audit_all_splits()
        key=identity({'plan':file_hash(HERE/'generation-plan.jsonl'),'manifests':audit['manifests']})
        reservation=CACHE/'test-access'/f'{key}.json'
        reservation.parent.mkdir(exist_ok=True)
        # Reserve before any inference. Failure/interruption still exposes this set.
        with reservation.open('x',encoding='utf-8') as stream:
            json.dump(dict(binding=runtime.binding,output=str(output.resolve()),
                           plan_sha256=file_hash(HERE/'generation-plan.jsonl'),manifests=audit['manifests']),stream)
            stream.flush();os.fsync(stream.fileno())
    output.mkdir(parents=True)
    write_json(output/'started.json',dict(binding=runtime.binding,split=split,expected=len(records),complete=False))
    predictions=[]
    with (output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for index,row in enumerate(records):
            prediction=replay(runtime,row)
            predictions.append(prediction)
            stream.write(json.dumps(prediction)+'\n');stream.flush()
            if index%100==0:
                os.fsync(stream.fileno());print('REPLAY',split,index,'/',len(records),flush=True)
    report=summarize(predictions)
    elapsed=np.array([r['inference_seconds'] for r in predictions])
    durations=np.array([r['source_seconds'] for r in predictions])
    report['timing']=dict(ort_provider='CPUExecutionProvider',ort_intra_threads=1,
        ort_inter_threads=1,validated_bundle_load_seconds=load_seconds,
        inference_seconds_p50=float(np.quantile(elapsed,.50)),
        inference_seconds_p95=float(np.quantile(elapsed,.95)),
        compute_rtf_p50=float(np.quantile(elapsed/durations,.50)),
        compute_rtf_p95=float(np.quantile(elapsed/durations,.95)),
        excludes_audio_file_read=True,excludes_speaking_duration=True,
        endpoint_silence_ms=runtime.endpoint.config.trailing_silence_ms)
    report.update(binding=runtime.binding,predictions_sha256=file_hash(output/'predictions.jsonl'),
                  complete=len(predictions)==len(records),split=split)
    write_json(output/'results.json',report)
    return report
