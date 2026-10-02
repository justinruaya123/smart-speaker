"""Versioned Parler-only corpus replacement; no training or sealed inference.

The parent corpus and failed model remain immutable. Reuse the original generator
with a scoped checkpoint-loading repair, preserving batch membership and seeds.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'.cache/pcv-synthetic-v3'
PARENT=ROOT/'.cache/pcv-synthetic-v2-repair1'
V2=ROOT/'experiments/pcv-synthetic-v2'
V1=ROOT/'experiments/pcv-synthetic-v1'
MODEL=ROOT/'.cache/pcv-synthetic-v1/models/parler-tts-large-v1'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.pcv_parler_weights import digest, repair_and_verify
from scripts.pcv_atomic_io import atomic_json

ASSIGNMENT=('recording_id','verbatim_id','engine','voice_group','split','accent_requested','seed','synthetic')
MANIFESTS=[prefix+engine+'-manifest.jsonl' for prefix in ('','wake-') for engine in ('qwen','parler','dia')]


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def identity(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read_rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
def now():return datetime.now(timezone.utc).isoformat()


def activate_study():
    # Keep pure manifest helpers and tests independent of the older TTS overlay.
    for path in (V1,V2,ROOT/'.cache/pcv-synthetic-v1/parler-site'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))


def immutable_json(path,value):
    path=Path(path);payload=json.dumps(value,sort_keys=True,indent=2)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8')!=payload:raise FileExistsError('Immutable receipt changed: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(payload);f.flush();os.fsync(f.fileno())


def original_batches(rows):
    by_id={r['recording_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('Duplicate parent assignment')
    seen=set();batches=[]
    for row in rows:
        if row['recording_id'] in seen:continue
        keys=row['batch_ids']
        if (not keys or len(keys)>4 or len(keys)!=len(set(keys)) or any(k not in by_id or k in seen for k in keys)):
            raise ValueError('Incomplete or overlapping parent batch')
        batch=[by_id[k] for k in keys]
        for member in batch:
            if (member['batch_ids']!=keys or member['engine']!='parler' or member['batch_size']!=4
                or member['voice_group']!=row['voice_group'] or member['split']!=row['split']):
                raise ValueError('Inconsistent parent batch metadata')
        seen.update(keys);batches.append(batch)
    if seen!=set(by_id):raise ValueError('Incomplete batch coverage')
    return batches


def retry_seeds(batch):return [(batch[0]['seed']+attempt)%2**32 for attempt in range(3)]


def ctc_capacity(samples,phones):
    frames=max(0,1+(samples-400)//160)
    required=len(phones)+sum(a==b for a,b in zip(phones,phones[1:]))
    return dict(frames=frames,required=required,feasible=frames>=required)


def planned_row(old,cache=None):
    if cache is None:cache=CACHE
    key=old['recording_id']
    if '..' in Path(key).parts or Path(key).is_absolute():raise ValueError('Invalid recording ID')
    return {**{k:old[k] for k in ASSIGNMENT},'parent_record_sha256':identity(old),
        'source_version':'v3','accent_verified':False,'voice_anchor_sha256':None,
        'audio':str(cache/'audio'/f'{key}.wav'),'native_audio':str(cache/'native'/f'{key}.wav')}


def merge_manifest(parents,replacements):
    expected={r['recording_id']:r for r in parents if r['engine']=='parler'}
    changed={r['recording_id']:r for r in replacements}
    if len(changed)!=len(replacements) or set(changed)!=set(expected):raise ValueError('Replacement coverage differs')
    for key,row in changed.items():
        old=expected[key]
        if row.get('parent_record_sha256')!=identity(old) or any(row.get(k)!=old[k] for k in ASSIGNMENT):
            raise ValueError('Replacement assignment changed')
    return [changed.get(r['recording_id'],r) for r in parents]


def parent_rows():return read_rows(PARENT/'parler-manifest.jsonl')+read_rows(PARENT/'wake-parler-manifest.jsonl')


def ensure_sealed_unused():
    for folder in (PARENT,ROOT/'.cache/pcv-synthetic-v2'):
        for name in ('test-access','wake-test-access','final-test','wake-final-test'):
            path=folder/name
            if path.exists() and any(path.iterdir()):raise ValueError('Parent sealed access exists: '+str(path))


def environment():
    import torch,parler_tts,transformers,dac,audiotools
    versions={name:importlib.metadata.version(name) for name in
        ('torch','transformers','parler-tts','safetensors','numpy','soundfile','scipy','tokenizers','huggingface-hub',
         'descript-audio-codec','descript-audiotools')}
    sources={}
    for module in (parler_tts,transformers,dac,audiotools):
        folder=Path(module.__file__).parent
        for path in sorted(folder.rglob('*.py')):sources[str(path.relative_to(ROOT))]=digest(path)
    return dict(versions=versions,implementation_sources=sources,python=sys.version,
        torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def source_paths():
    return [Path(__file__),ROOT/'scripts/pcv_parler_weights.py',ROOT/'scripts/pcv_atomic_io.py',
        ROOT/'experiments/pcv-delivery-v1/tests/test_parler_weights.py',
        ROOT/'experiments/pcv-delivery-v1/tests/test_parler_corpus.py']


def validate_parent_sources():
    protocol=json.loads((ROOT/'experiments/pcv-synthetic-v2-repair1/protocol.json').read_text())
    for name,expected in {**protocol['shared_sources'],**protocol['wake_assets']}.items():
        if digest(ROOT/name)!=expected:raise ValueError('Parent source/asset changed: '+name)


def validate_parent_manifests():
    audit=json.loads((PARENT/'assembly-audit.json').read_text())
    plan=json.loads((ROOT/'experiments/pcv-synthetic-v2-repair1/repair-plan.json').read_text())
    for engine in ('qwen','parler','dia'):
        if digest(PARENT/f'{engine}-manifest.jsonl')!=audit['manifests'][engine]:
            raise ValueError('Parent command manifest differs from completed audit')
        name=f'wake-{engine}-manifest.jsonl'
        if digest(PARENT/name)!=plan['parent_manifests'][name]:
            raise ValueError('Parent wake manifest differs from frozen repair lineage')


def freeze():
    # Generation has its own immutable protocol; fitting is not launched here.
    activate_study()
    import revision
    from assets import verify_assets
    revision.validate();validate_parent_sources();validate_parent_manifests();ensure_sealed_unused()
    failed=PARENT/'jobs/parallel-io-repair-run-20261001/status.json'
    if json.loads(failed.read_text())['status']!='FAILED':raise ValueError('Expected preserved failed parent')
    rows=parent_rows();batches=original_batches(rows)
    if len(rows)!=16704:raise ValueError('Unexpected Parler corpus size')
    refs=read_rows(V2/'verbatims.jsonl')+read_rows(V2/'wake-verbatims.jsonl')
    if set(r['verbatim_id'] for r in rows)!=set(r['id'] for r in refs):raise ValueError('Verbatim coverage differs')
    pilot=ROOT/'.cache/pcv-delivery-v1/export-diagnosis-20261001/verified-codec/results.json'
    evidence=json.loads(pilot.read_text())
    if (evidence['dtype']!='bfloat16' or evidence['device']!='cuda' or evidence['after']['changed_before']
        or not evidence['after']['verified_after'] or evidence['after']['tensors_checked']!=944
        or evidence['after']['source_sha256']!=digest(ROOT/'scripts/pcv_parler_weights.py')):
        raise ValueError('Missing matching real CUDA weight verification')
    paths=[failed,pilot,PARENT/'training/selection.json',PARENT/'compute-budget.json',PARENT/'assembly-audit.json',
        ROOT/'experiments/pcv-synthetic-v2-repair1/repair-plan.json',
        ROOT/'experiments/pcv-synthetic-v2-repair1/protocol.json',V2/'protocol.json',
        *[V2/name for name in ('verbatims.jsonl','wake-verbatims.jsonl','generation-plan.jsonl','wake-generation-plan.jsonl','lexicon.json')],
        *[PARENT/name for name in MANIFESTS],*source_paths()]
    budget=json.loads((PARENT/'compute-budget.json').read_text())
    protocol=dict(version='pcv-synthetic-v3',scope='Generation only; no training/calibration/sealed inference in this entry point',
        reason='Restore two embedded Parler codec tensors incorrectly loaded by from_pretrained; exact whole-model verification',
        parent_version='pcv-synthetic-v2-repair1',inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},
        tts_assets_sha256=verify_assets('parler'),environment=environment(),records=len(rows),batches=len(batches),
        batch_membership_sha256=identity([[r['recording_id'] for r in batch] for batch in batches]),
        recipe=dict(batch_size=4,temperature=.9,max_new_tokens=1400,dtype='bfloat16',device='cuda',
            seeds='Original first batch record seed plus attempt minus one, modulo 2**32; attempts 1,2,3',
            batching='Exact original membership/order, including mixed command/wake batches',
            preflight='Original waveform gates plus CTC frame capacity for every command before any batch output is persisted',
            verify='All checkpoint tensors before generation, after first generated batch, and at completion',
            text_verified_by_listener=False,accent_verified=False),
        unchanged=['All Qwen and Dia manifest rows and audio','All verbatims, labels, splits, named voices, catalog, gates and strict demo definitions'],
        previous_single_clip_repair='Preserved as parent evidence; all Parler now regenerated from original attempt-one batch seeds',
        future_training=dict(status='Must freeze separate training protocol before fitting',baseline='Original human-pretrained baseline, not failed synthetic winner',
            seeds=[53,71],epochs_per_seed=40,decoder='Existing parity-verified eight CPU workers; neural training/inference CUDA',
            carried_compute_seconds=budget['seconds'],carried_epochs=len(budget['epochs']),
            limits='Preserve original global 86400-second/200-epoch budget and charge prior work',
            sealed_integrity='Reserve v2 and repair1 reused command/wake sets before any future v3 sealed inference',
            export_gate='Both original unchanged FP32 numerical fixtures <2e-5, then complete waveform parity'))
    immutable_json(HERE/'generation-protocol.json',protocol)
    CACHE.mkdir(parents=True,exist_ok=True)
    return dict(protocol_sha256=digest(HERE/'generation-protocol.json'),records=len(rows),batches=len(batches))


def validate():
    activate_study()
    import revision
    from assets import verify_assets
    revision.validate();validate_parent_sources();validate_parent_manifests();ensure_sealed_unused()
    protocol=json.loads((HERE/'generation-protocol.json').read_text())
    for name,expected in protocol['inputs'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    if verify_assets('parler')!=protocol['tts_assets_sha256'] or environment()!=protocol['environment']:
        raise ValueError('TTS assets or runtime implementation changed')
    return protocol


@contextmanager
def verified_loader():
    """Reuse the frozen generator unchanged and restore its class descriptor."""
    activate_study()
    import synthesis
    from parler_tts import ParlerTTSForConditionalGeneration as cls
    original=cls.from_pretrained;descriptor=cls.__dict__.get('from_pretrained');captured={}
    def load(unused,*args,**kwargs):
        model=original(*args,**kwargs).to('cuda').eval()
        captured.update(model=model,verification=repair_and_verify(model,MODEL))
        return model
    try:
        cls.from_pretrained=classmethod(load)
        generator,hashes=synthesis.load_engine('parler')
        if hashes!=json.loads((ROOT/'.cache/pcv-synthetic-v1/parler-model-identity.json').read_text()):
            raise ValueError('Original TTS asset identity differs')
        yield generator,captured
    finally:
        if descriptor is None:delattr(cls,'from_pretrained')
        else:cls.from_pretrained=descriptor


def generate(job):
    activate_study()
    import torch,soundfile as sf,synthesis,revision
    protocol=validate();job=Path(job).resolve()
    if not job.is_relative_to(CACHE.resolve()/'jobs'):raise ValueError('Job must be inside v3 jobs')
    job.mkdir(parents=True,exist_ok=False)
    state=dict(status='RUNNING',stage='load-verified-parler',pid=os.getpid(),started=now(),generated=0,
        planned=protocol['records'],job=str(job),protocol_sha256=digest(HERE/'generation-protocol.json'))
    def update(**fields):state.update(fields,updated=now());atomic_json(job/'status.json',state)
    # An exclusive marker deliberately survives failure; no accidental duplicate or silent resume.
    with (CACHE/'generation-started.json').open('x',encoding='utf-8') as f:
        json.dump(state,f);f.flush();os.fsync(f.fileno())
    update();started=time.monotonic()
    try:
        rows=parent_rows();batches=original_batches(rows)
        refs={r['id']:r for r in read_rows(V2/'verbatims.jsonl')+read_rows(V2/'wake-verbatims.jsonl')}
        phones={r['id']:r['phones'] for r in revision.compile_catalog(read_rows(V2/'verbatims.jsonl'))['entries']}
        path=CACHE/'replacements.jsonl'
        if path.exists():raise FileExistsError('Preserve previous replacements')
        torch.set_num_threads(4);completed=[]
        with verified_loader() as (generator,loaded):
            first_verification=loaded['verification']
            for number,batch in enumerate(batches):
                began=time.monotonic();torch.cuda.empty_cache()
                voice=next(v for v in revision.VOICES if v['id']==batch[0]['voice_group'])
                for attempt,seed in enumerate(retry_seeds(batch),1):
                    torch.manual_seed(seed)
                    try:
                        waves,rate=generator([refs[r['verbatim_id']]['text'] for r in batch],voice)
                        if len(waves)!=len(batch):raise ValueError('Wrong generation batch count')
                        ready=[synthesis.prepare_audio(w,rate) for w in waves];capacities=[]
                        for old,audio in zip(batch,ready):
                            target=phones.get(old['verbatim_id'])
                            info=sf.info(io.BytesIO(audio[1]))
                            capacity=ctc_capacity(info.frames,target) if target is not None else None
                            if capacity is not None and not capacity['feasible']:raise ValueError('CTC-infeasible command: '+old['recording_id'])
                            capacities.append(capacity)
                        break
                    except ValueError as exc:
                        immutable_json(CACHE/'attempts'/f'{number:05}-{attempt}.json',dict(batch=number,
                            ids=[r['recording_id'] for r in batch],attempt=attempt,seed=seed,error=str(exc)))
                        if attempt==3:raise
                if number==0:
                    after=repair_and_verify(loaded['model'],MODEL)
                    if after['changed_before']:raise ValueError('Generator mutated checkpoint weights')
                    immutable_json(job/'weight-verification.json',dict(before=first_verification,after_first_generation=after))
                metadata=generator.last_metadata
                if len(metadata)!=len(batch):raise ValueError('Wrong generation metadata count')
                for old,wave,audio,capacity,meta in zip(batch,waves,ready,capacities,metadata):
                    item=synthesis.persist_audio(CACHE,planned_row(old),wave,rate,audio)
                    item.update(batch_ids=old['batch_ids'],batch_size=4,batch_seed=seed,generation_attempt=attempt,
                        generation_metadata=meta,ctc_capacity=capacity,verbatim_sha256=identity(refs[old['verbatim_id']]),
                        voice_conditioning=dict(type='named_voice_description',name=voice['parler']),
                        generation_protocol_sha256=state['protocol_sha256'],generation_source_sha256=digest(__file__),
                        weight_verification_path=str(job/'weight-verification.json'),
                        weight_verification_sha256=digest(job/'weight-verification.json'),
                        tts_assets_sha256=protocol['tts_assets_sha256'])
                    with path.open('a',encoding='utf-8',newline='\n') as f:f.write(canonical(item)+'\n');f.flush();os.fsync(f.fileno())
                    completed.append(item)
                update(stage='generate-parler',generated=len(completed),batch=number+1,batches=len(batches),elapsed_seconds=time.monotonic()-started)
                print('GENERATED_VERIFIED_PARLER',len(completed),'/',len(rows),'batch_seconds',round(time.monotonic()-began,2),flush=True)
            after=repair_and_verify(loaded['model'],MODEL)
            if after['changed_before']:raise ValueError('Generator weights changed')
            immutable_json(job/'weight-verification-final.json',after)
        validate()
        if digest(HERE/'generation-protocol.json')!=state['protocol_sha256']:
            raise ValueError('Generation protocol changed during execution')
        for name in MANIFESTS:
            parent=read_rows(PARENT/name);keys={r['recording_id'] for r in parent}
            selected=[r for r in completed if r['recording_id'] in keys]
            result=merge_manifest(parent,selected)
            destination=CACHE/name
            with destination.open('x',encoding='utf-8',newline='\n') as f:
                for row in result:f.write(canonical(row)+'\n')
                f.flush();os.fsync(f.fileno())
        receipt=dict(complete=True,generated=len(completed),planned=len(rows),manifest_sha256=digest(path),
            manifests={name:digest(CACHE/name) for name in MANIFESTS},protocol_sha256=state['protocol_sha256'],
            synthetic_text_and_accent_listener_verified=False,full_audio_audit_pending=True,training_started=False)
        immutable_json(CACHE/'generation-complete.json',receipt)
        update(status='SUCCEEDED',stage='generation-complete',finished=now(),elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        update(status='FAILED',error=repr(exc),finished=now(),elapsed_seconds=time.monotonic()-started)
        traceback.print_exc();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','generate','status']);p.add_argument('--job')
    args=p.parse_args()
    if args.stage=='freeze':result=freeze()
    elif args.stage=='generate':
        if not args.job:p.error('--job required')
        generate(args.job);return
    else:
        result=dict(version='pcv-synthetic-v3',jobs=[json.loads(p.read_text()) for p in sorted((CACHE/'jobs').glob('*/status.json'))],
            generation_complete=(CACHE/'generation-complete.json').exists())
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
