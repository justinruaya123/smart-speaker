"""Host adapter for the corrected corpus; original fitting and gates stay intact."""
from collections import Counter,defaultdict
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
from time import monotonic

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'.cache/pcv-synthetic-v3'
PARENT=ROOT/'.cache/pcv-synthetic-v2-repair1'
PARENT_HERE=ROOT/'experiments/pcv-synthetic-v2-repair1'
V2=ROOT/'experiments/pcv-synthetic-v2'
sys.path[:0]=[str(ROOT),str(V2),str(ROOT/'experiments/pcv-synthetic-v1'),str(ROOT/'experiments/phrase-contract-v1')]
spec=importlib.util.spec_from_file_location('_pcv_v3_generation',HERE/'generation.py')
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
import revision as rev
import version_context
from scripts import pcv_parallel as parallel
from scripts.pcv_atomic_io import reliable_budget_io,atomic_json,replace_with_retry
from revision import file_hash,identity,canonical,read_rows

ENGINES=('qwen','parler','dia')
immutable_json=gen.immutable_json


def pcm_hash(pcm):return hashlib.sha256(pcm.tobytes()).hexdigest()


def carry_budget(source,target):
    source,target=Path(source),Path(target);state=json.loads(source.read_text())
    if (state['limits']!={'epochs':200,'seconds':86400} or not 0<=state['seconds']<86400
        or not isinstance(state['epochs'],list) or len(state['epochs'])+80>200):
        raise ValueError('Invalid or exhausted inherited compute budget')
    if source.with_suffix('.lock').exists():raise ValueError('Parent budget is still locked')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())


def validate_budget(inherited=None,*,initial=False):
    if inherited is None:inherited=json.loads((HERE/'protocol.json').read_text())['inherited_budget']
    source=ROOT/inherited['path']
    if file_hash(source)!=inherited['sha256']:raise ValueError('Inherited budget source changed')
    original=json.loads(source.read_text());receipt=CACHE/'budget-inheritance.json';target=CACHE/'compute-budget.json'
    if not receipt.is_file() or not target.is_file():raise ValueError('Inherited child compute budget missing')
    expected=dict(source=inherited,inherited_state=original,copied_before_training=True)
    if json.loads(receipt.read_text())!=expected:raise ValueError('Budget inheritance receipt changed')
    current=json.loads(target.read_text());epochs=current.get('epochs');seconds=current.get('seconds')
    if (current.get('limits')!=original['limits'] or not isinstance(epochs,list)
        or epochs[:len(original['epochs'])]!=original['epochs'] or len(epochs)>original['limits']['epochs']
        or not isinstance(seconds,(float,int)) or not math.isfinite(seconds) or seconds<original['seconds']):
        raise ValueError('Inherited compute charges were reset or altered')
    if initial and current!=original:raise ValueError('Fresh training requires exact initial inherited charges')
    return current


def verify_inputs(mapping):
    for name,expected in mapping.items():
        if file_hash(ROOT/name)!=expected:raise ValueError('Frozen v3 input changed: '+name)


def generation_identity():
    """No TTS imports or sealed-access checks in a fitting/runtime process."""
    protocol=json.loads((HERE/'generation-protocol.json').read_text())
    verify_inputs(protocol['inputs']);verify_inputs(protocol['environment']['implementation_sources'])
    gen.validate_parent_sources();gen.validate_parent_manifests()
    return protocol


def require_generation_complete():
    protocol=generation_identity()
    receipt=json.loads((CACHE/'generation-complete.json').read_text())
    marker=json.loads((CACHE/'generation-started.json').read_text());job=Path(marker['job']).resolve()
    if not job.is_relative_to((CACHE/'jobs').resolve()):raise ValueError('Generation job escapes cache')
    state=json.loads((job/'status.json').read_text())
    if (state['status']!='SUCCEEDED' or not receipt.get('complete')
        or receipt['generated']!=protocol['records'] or receipt['planned']!=protocol['records']
        or receipt['protocol_sha256']!=file_hash(HERE/'generation-protocol.json')
        or receipt['manifest_sha256']!=file_hash(CACHE/'replacements.jsonl')
        or receipt['manifests']!={name:file_hash(CACHE/name) for name in gen.MANIFESTS}):
        raise ValueError('Generation is incomplete or its completion identity changed')
    checks=json.loads((job/'weight-verification.json').read_text())
    final=json.loads((job/'weight-verification-final.json').read_text())
    for result in (checks['before'],checks['after_first_generation'],final):
        if not result.get('verified_after') or result['tensors_checked']!=944 or result['source_sha256']!=file_hash(ROOT/'scripts/pcv_parler_weights.py'):
            raise ValueError('Missing exact real generator weight verification')
    if checks['after_first_generation']['changed_before'] or final['changed_before']:
        raise ValueError('Post-generation weights changed')
    return receipt,job


def freeze():
    rev.validate();gen.ensure_sealed_unused();complete,job=require_generation_complete()
    parent=json.loads((PARENT_HERE/'protocol.json').read_text())
    for name in set(parent['input_hashes'])|{'verbatims.jsonl','generation-plan.jsonl'}:
        payload=(PARENT_HERE/name).read_bytes();target=HERE/name
        if target.exists():
            if target.read_bytes()!=payload:raise FileExistsError('V3 definition differs: '+name)
        else:
            with target.open('xb') as f:f.write(payload)
    paths=[HERE/'generation-protocol.json',CACHE/'generation-complete.json',CACHE/'generation-started.json',
        job/'status.json',job/'weight-verification.json',job/'weight-verification-final.json',
        PARENT/'compute-budget.json',PARENT/'training/selection.json',
        ROOT/'.cache/semifinal-v1/longer/seed-71/selected.pt',
        ROOT/'.cache/semifinal-v1/longer/selection-before-test.json',ROOT/'.cache/semifinal-v1/longer/bundle/recipe.json',
        ROOT/'experiments/pcv-delivery-v1/demo.json',ROOT/'experiments/pcv-delivery-v1/protocol.json']
    # All version adapters are frozen together only when the corpus has completed.
    sources={**parent['shared_sources'],**{str(p.relative_to(ROOT)):file_hash(p) for p in
        [*HERE.glob('*.py'),*HERE.glob('tests/*.py'),ROOT/'scripts/pcv_parallel.py',ROOT/'scripts/pcv_atomic_io.py']}}
    protocol={**parent,'version':'pcv-synthetic-v3','parent_protocol_sha256':file_hash(PARENT_HERE/'protocol.json'),
        'revision_reason':'Exact checkpoint-loaded Parler codec; all Parler audio regenerated, Qwen/Dia unchanged',
        'audio_assignment_override':'Parler payloads/provenance only; exact parent assignment and unchanged other rows',
        'generation_protocol_sha256':file_hash(HERE/'generation-protocol.json'),
        'completion_inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        'completed_manifests':complete['manifests'],'shared_sources':sources,
        'inherited_budget':dict(path=str((PARENT/'compute-budget.json').relative_to(ROOT)),sha256=file_hash(PARENT/'compute-budget.json')),
        'execution':dict(neural_device='cuda',exact_decoder_workers=8,numerical_export_tolerance=2e-5,
            sealed_ancestors=['pcv-synthetic-v2','pcv-synthetic-v2-repair1']),
        'test_tuning':False,'initialization':'original_human_pretrained_semifinal'}
    # Repair1's single-record recipe is retained in its parent receipt, not asserted for v3.
    protocol.pop('repair_plan_sha256',None)
    immutable_json(HERE/'protocol.json',protocol)
    carry_budget(PARENT/'compute-budget.json',CACHE/'compute-budget.json')
    immutable_json(CACHE/'budget-inheritance.json',dict(source=protocol['inherited_budget'],
        inherited_state=json.loads((PARENT/'compute-budget.json').read_text()),copied_before_training=True))
    return validate()


def validate():
    rev.validate();generation_identity()
    p=json.loads((HERE/'protocol.json').read_text())
    if p['parent_protocol_sha256']!=file_hash(PARENT_HERE/'protocol.json'):raise ValueError('Parent protocol changed')
    verify_inputs(p['completion_inputs']);verify_inputs(p['shared_sources']);verify_inputs(p['wake_assets'])
    for name,expected in {**p['input_hashes'],'verbatims.jsonl':p['verbatims_sha256'],'generation-plan.jsonl':p['generation_plan_sha256']}.items():
        if file_hash(HERE/name)!=expected:raise ValueError('V3 definitions changed: '+name)
    if p['completed_manifests']!={name:file_hash(CACHE/name) for name in gen.MANIFESTS}:raise ValueError('V3 manifest changed')
    validate_budget(p['inherited_budget'])
    return p


def audit_audio(row,allowed_roots,phones=None):
    import soundfile as sf
    import numpy as np
    from synthesis import audit_waveform
    for key,expected in (('audio','audio_sha256'),('native_audio','native_sha256')):
        path=Path(row[key]).resolve()
        if not any(path.is_relative_to(Path(root).resolve()) for root in allowed_roots):raise ValueError('Audio path escape')
        if file_hash(path)!=row[expected]:raise ValueError('Audio identity changed')
    native,native_rate=sf.read(row['native_audio'],dtype='float32')
    pcm,rate=sf.read(row['audio'],dtype='float32')
    if (rate!=16000 or row['sample_rate']!=rate or row['channels']!=1 or native_rate!=row['native_sample_rate']
        or pcm.ndim!=1 or native.ndim!=1 or not np.isfinite(pcm).all() or pcm_hash(pcm)!=row['pcm_sha256']):
        raise ValueError('Invalid waveform or decoded PCM identity')
    audit_waveform(native,native_rate);audit_waveform(pcm,rate)
    capacity=None if phones is None else gen.ctc_capacity(len(pcm),phones)
    if capacity is not None and not capacity['feasible']:raise ValueError('CTC-infeasible command')
    return dict(frames=len(pcm),ctc_capacity=capacity)


def audit_disjoint(rows):
    hashes=set();ids=set();voices=defaultdict(set);anchors=defaultdict(set)
    for row in rows:
        if row['pcm_sha256'] in hashes or row['recording_id'] in ids:raise ValueError('Duplicate command/wake audio or ID')
        hashes.add(row['pcm_sha256']);ids.add(row['recording_id']);voices[row['voice_group']].add(row['split'])
        if row['engine']!='parler':anchors[row['voice_anchor_sha256']].add(row['split'])
    if any(len(v)>1 for v in list(voices.values())+list(anchors.values())):raise ValueError('Cross-split voice or anchor overlap')
    return dict(recordings=len(rows),unique_pcm=len(hashes),voice_splits={k:sorted(v) for k,v in voices.items()})


def audit_all_splits():
    protocol=validate();complete,job=require_generation_complete()
    replacements=read_rows(CACHE/'replacements.jsonl')
    by_id={r['recording_id']:r for r in replacements}
    if len(by_id)!=len(replacements) or len(replacements)!=complete['generated']:raise ValueError('Replacement coverage differs')
    refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')+read_rows(HERE/'wake-verbatims.jsonl')}
    phones={r['id']:r['phones'] for r in rev.compile_catalog(read_rows(HERE/'verbatims.jsonl'))['entries']}
    parents={r['recording_id']:r for r in gen.parent_rows()}
    batches=gen.original_batches(list(parents.values()));batch_map={r['recording_id']:batch for batch in batches for r in batch}
    voice_map={v['id']:v for v in rev.VOICES};all_rows=[];counts=Counter()
    generation_protocol=json.loads((HERE/'generation-protocol.json').read_text())
    generation_sha=file_hash(HERE/'generation-protocol.json');generation_source=file_hash(HERE/'generation.py')
    weight_sha=file_hash(job/'weight-verification.json')
    for name in gen.MANIFESTS:
        prior=read_rows(PARENT/name);keys={r['recording_id'] for r in prior}
        expected=gen.merge_manifest(prior,[r for key,r in by_id.items() if key in keys])
        rows=read_rows(CACHE/name)
        if rows!=expected:raise ValueError('Merged manifest differs from declared replacements')
        for row in rows:
            key=row['recording_id'];ref=refs[row['verbatim_id']]
            if row['engine']=='parler':
                plan=gen.planned_row(parents[key])
                if any(row.get(k)!=v for k,v in plan.items()):raise ValueError('Replacement assignment drift')
                attempt=row['generation_attempt'];batch=batch_map[key]
                if (type(attempt) is not int or not 1<=attempt<=3 or row['batch_ids']!=[r['recording_id'] for r in batch]
                    or row['batch_size']!=4 or row['batch_seed']!=gen.retry_seeds(batch)[attempt-1]
                    or row['generation_protocol_sha256']!=generation_sha
                    or row['generation_source_sha256']!=generation_source
                    or row['verbatim_sha256']!=gen.identity(ref)
                    or row['weight_verification_path']!=str(job/'weight-verification.json')
                    or row['weight_verification_sha256']!=weight_sha
                    or row['tts_assets_sha256']!=generation_protocol['tts_assets_sha256']
                    or row['voice_conditioning']!={'type':'named_voice_description','name':voice_map[row['voice_group']]['parler']}):
                    raise ValueError('Replacement generation provenance drift')
                roots=[CACHE]
            else:
                roots=[rev.OLD_CACHE,rev.CACHE]
                if file_hash(rev.OLD_CACHE/'anchors'/f'{row["voice_group"]}.wav')!=row['voice_anchor_sha256']:
                    raise ValueError('Reused voice anchor changed')
            capacity=audit_audio(row,roots,phones=phones.get(row['verbatim_id']))
            if row['engine']=='parler' and capacity['ctc_capacity']!=row['ctc_capacity']:raise ValueError('Saved CTC preflight differs')
            counts[f'{row["engine"]}/{row["split"]}/{"wake" if name.startswith("wake-") else "command"}']+=1
        all_rows.extend(rows)
    disjoint=audit_disjoint(all_rows)
    commands=sum(n for key,n in counts.items() if key.endswith('/command'));wakes=sum(n for key,n in counts.items() if key.endswith('/wake'))
    if (commands,wakes)!=(protocol['expected_clips'],len(read_rows(HERE/'wake-generation-plan.jsonl'))):raise ValueError('Incomplete combined corpus')
    return dict(recordings=commands,wake_recordings=wakes,combined_recordings=len(all_rows),
        unique_pcm=commands,combined_unique_pcm=disjoint['unique_pcm'],synthetic=True,original_humans=0,
        text_verified_by_listener=False,accent_verified=False,counts=dict(counts),
        manifests={e:file_hash(CACHE/f'{e}-manifest.jsonl') for e in ENGINES},
        wake_manifests={e:file_hash(CACHE/f'wake-{e}-manifest.jsonl') for e in ENGINES},
        generation_complete_sha256=file_hash(CACHE/'generation-complete.json'),
        reduced_dia=True,third_engine_exhaustive=False,parent_non_parler_rows_unchanged=True,
        replaced_parler_recordings=len(replacements),ctc_checked_all_commands=True)


def load_split(split):
    if split not in ('train','dev','calibration','test'):raise ValueError('Unknown split')
    audit_all_splits();refs={r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    return [{**r,**refs[r['verbatim_id']]} for e in ENGINES for r in read_rows(CACHE/f'{e}-manifest.jsonl') if r['split']==split]


def audit_wake(split):
    audit_all_splits();refs={r['id']:r for r in read_rows(HERE/'wake-verbatims.jsonl')}
    return [{**r,**refs[r['verbatim_id']]} for e in ENGINES for r in read_rows(CACHE/f'wake-{e}-manifest.jsonl') if r['split']==split]


def reserve_ancestors(kind,bundle,output,*,parents=None,protocol_sha256=None):
    if kind not in ('command','wake'):raise ValueError('Unknown sealed set')
    parents=parents or [(rev.CACHE,V2),(PARENT,PARENT_HERE)]
    protocol_sha256=protocol_sha256 or file_hash(HERE/'protocol.json');paths=[]
    # Partial reservations survive any failure and intentionally block retry.
    for cache,here in parents:
        commands={e:file_hash(cache/f'{e}-manifest.jsonl') for e in ENGINES}
        if kind=='command':
            dataset=dict(plan=file_hash(here/'generation-plan.jsonl'),manifests=commands);folder='test-access'
        else:
            dataset=dict(plan=file_hash(here/'wake-generation-plan.jsonl'),references=file_hash(here/'wake-verbatims.jsonl'),
                wake_manifests={e:file_hash(cache/f'wake-{e}-manifest.jsonl') for e in ENGINES},command_manifests=commands)
            folder='wake-test-access'
        path=cache/folder/f'{identity(dataset)}.json';path.parent.mkdir(exist_ok=True)
        with path.open('x',encoding='utf-8') as f:
            json.dump(dict(dataset=dataset,child_protocol_sha256=protocol_sha256,bundle=str(bundle),output=str(output),
                state='RESERVED_PRESERVE_ON_FAILURE'),f);f.flush();os.fsync(f.fileno())
        paths.append(path)
    return paths


@contextmanager
def bind():
    import study,synthbench,fitting,finishing,reporting,wake_benchmark,final_report
    from pcvcm import compute
    changes=[]
    def set_(module,key,value):changes.append((module,key,getattr(module,key)));setattr(module,key,value)
    with version_context.bind(),reliable_budget_io(types.SimpleNamespace(atomic_json=atomic_json)):
        try:
            class MonotoneComputeBudget(compute.ComputeBudget):
                # Group the elapsed subtraction before adding prior charges.
                # (base + uptime) - start can lose an ULP when elapsed is zero.
                def __enter__(self):
                    if not self.path.is_file():raise ValueError('Inherited child compute budget missing')
                    if self.path.resolve()==(CACHE/'compute-budget.json').resolve():validate_budget(initial=True)
                    return super().__enter__()
                def _save(self):
                    self.state['seconds']=max(self.state['seconds'],self.base_seconds+(monotonic()-self.started))
                    temporary=self.path.with_suffix('.pending')
                    compute.write_json(temporary,self.state);replace_with_retry(temporary,self.path)
                def check(self):
                    self._save()
                    charged=max(self.state['seconds'],self.base_seconds+(monotonic()-self.started))
                    if charged>=self.limits['seconds']:raise RuntimeError('Global compute budget exhausted (time)')
            for module in (study,synthbench,fitting,finishing,reporting,wake_benchmark,final_report):
                for key,value in (('HERE',HERE),('CACHE',CACHE)):
                    if hasattr(module,key):set_(module,key,value)
            for module in (study,synthbench):set_(module,'validate_frozen_study',validate)
            for module in (synthbench,fitting):
                set_(module,'load_split',load_split);set_(module,'audit_all_splits',audit_all_splits)
            set_(fitting,'development',parallel.development);set_(fitting,'ComputeBudget',MonotoneComputeBudget)
            set_(wake_benchmark,'audit_wake',audit_wake)
            synthbench.intent_use_cases.cache_clear();yield
        finally:
            for module,key,value in reversed(changes):setattr(module,key,value)
            synthbench.intent_use_cases.cache_clear()
