"""Immutable one-record training-data repair; the parent study is preserved.

Only the failed training recording is regenerated. Shared fitting, export and
recognition functions execute in a process-local child context, then restore.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT / 'experiments/pcv-synthetic-v2'
CACHE = ROOT / '.cache/pcv-synthetic-v2-repair1'
sys.path.insert(0, str(PARENT))
import revision as rev
import corpus
import version_context
from revision import canonical, file_hash, identity, read_rows, write_rows

ENGINES = ('qwen', 'parler', 'dia')
FAILED_JOB = rev.CACHE / 'jobs/revised-run-20261001/status.json'
DIAGNOSIS = rev.CACHE / 'diagnosis-ctc-20261001/audit.json'
TARGET = 'parler/train-gb-f/UC04-BRI-K-073-A'
MUTABLE = {'audio', 'native_audio', 'audio_sha256', 'native_sha256', 'pcm_sha256',
    'duration_seconds', 'rms', 'clipped_fraction', 'native_metrics',
    'batch_seed', 'generation_attempt', 'generation_metadata',
    'generation_source_sha256', 'source_version', 'repair_parent_record_sha256',
    'repair_plan_sha256', 'ctc_capacity'}


def immutable_json(path, value):
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2) + '\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != payload:
            raise FileExistsError(f'Immutable receipt changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def ctc_capacity(samples, phones):
    frames = max(0, 1 + (samples - 400) // 160)
    required = len(phones) + sum(a == b for a, b in zip(phones, phones[1:]))
    return dict(frames=frames, required=required, feasible=frames >= required)


def require_feasible(prepared, phones):
    import soundfile as sf
    info = sf.info(io.BytesIO(prepared[1]))
    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError('CTC preflight requires mono 16 kHz PCM')
    result = ctc_capacity(info.frames, phones)
    if not result['feasible']:
        raise ValueError(f'CTC-infeasible replacement: {result}')
    return result


def merge_records(parents, replacements, plan):
    parents_by_id = {r['recording_id']: r for r in parents}
    changed = {r['recording_id']: r for r in replacements}
    allowed = {r['recording_id']: r for r in plan['records']}
    if (len(parents_by_id) != len(parents) or len(changed) != len(replacements)
            or len(allowed) != len(plan['records']) or set(changed) != set(allowed)):
        raise ValueError('Missing or duplicate repair assignment')
    for key, new in changed.items():
        old = parents_by_id.get(key)
        if old is None or old['split'] != 'train' or new.get('split') != 'train':
            raise ValueError('Only named training recordings may be repaired')
        digest = identity(old)
        if allowed[key]['parent_record_sha256'] != digest or new.get('repair_parent_record_sha256') != digest:
            raise ValueError('Parent recording identity changed')
        if new.get('source_version') != 'v2-repair1':
            raise ValueError('Missing versioned repair lineage')
        if {k:v for k,v in old.items() if k not in MUTABLE} != {k:v for k,v in new.items() if k not in MUTABLE}:
            raise ValueError('Repair changed frozen assignment or conditioning')
    return [changed.get(r['recording_id'], r) for r in parents]


def parent_manifests():
    return {name: file_hash(rev.CACHE / name) for e in ENGINES
            for name in (f'{e}-manifest.jsonl', f'wake-{e}-manifest.jsonl')}


def freeze():
    import psutil
    parent = rev.validate()
    state = json.loads(FAILED_JOB.read_text())
    if state['status'] != 'FAILED' or state['steps'][-1]['name'] != 'prepare':
        raise ValueError('Expected preserved preparation failure')
    if psutil.pid_exists(state['pid']):
        raise RuntimeError('Parent controller PID still exists; inspect identity before proceeding')
    for folder in ('training', 'test-access', 'wake-test-access', 'final-test', 'wake-final-test'):
        if (rev.CACHE / folder).exists():
            raise ValueError('Parent fitting or sealed evaluation has already started')
    diagnosis = json.loads(DIAGNOSIS.read_text())
    failure = diagnosis['failures']
    if len(failure) != 1 or failure[0]['recording_id'] != TARGET or failure[0]['split'] != 'train':
        raise ValueError('Repair is limited to the one diagnosed training clip')
    old = next(r for r in read_rows(rev.CACHE/'parler-manifest.jsonl') if r['recording_id'] == TARGET)
    if identity(old) != failure[0]['record_sha256']:
        raise ValueError('Diagnosis no longer binds the failed record')
    plan = dict(version='pcv-synthetic-v2-repair1', parent_protocol_sha256=file_hash(PARENT/'protocol.json'),
        parent_manifests=parent_manifests(), failed_job=dict(path=str(FAILED_JOB), sha256=file_hash(FAILED_JOB)),
        diagnosis=dict(path=str(DIAGNOSIS), sha256=file_hash(DIAGNOSIS)),
        records=[dict(recording_id=TARGET, parent_record_sha256=identity(old), parent=old,
            attempts=[2, 3], batch_ids=old['batch_ids'])],
        reason='One 0.2554375-second training clip has 24 frames for a 30-frame CTC target',
        held_out_audio_unchanged=True, test_audio_used_for_repair=False,
        recipe='Same engine, voice, batch texts and generation recipe; next original retry seeds; preflight waveform and CTC')
    immutable_json(HERE/'repair-plan.json', plan)
    names = set(parent['input_hashes']) | {'verbatims.jsonl', 'generation-plan.jsonl'}
    for name in names:
        payload = (PARENT/name).read_bytes(); destination = HERE/name
        if destination.exists():
            if destination.read_bytes() != payload: raise FileExistsError(destination)
        else:
            with destination.open('xb') as stream: stream.write(payload)
    protocol = {**parent, 'version':'pcv-synthetic-v2-repair1',
        'parent_protocol_sha256':file_hash(PARENT/'protocol.json'),
        'revision_reason':plan['reason'], 'repair_plan_sha256':file_hash(HERE/'repair-plan.json'),
        'audio_assignment_override':'Only repair-plan training IDs; every other parent row unchanged',
        'shared_sources':{**parent['shared_sources'], **{str(p.relative_to(ROOT)):file_hash(p)
            for p in [*PARENT.glob('*.py'), *HERE.glob('*.py')]}}}
    immutable_json(HERE/'protocol.json', protocol)
    CACHE.mkdir(parents=True, exist_ok=True)
    return validate()


def validate():
    rev.validate()
    p = json.loads((HERE/'protocol.json').read_text())
    plan = json.loads((HERE/'repair-plan.json').read_text())
    if (file_hash(PARENT/'protocol.json') != p['parent_protocol_sha256']
            or file_hash(HERE/'repair-plan.json') != p['repair_plan_sha256']
            or parent_manifests() != plan['parent_manifests']):
        raise ValueError('Parent corpus or repair contract changed')
    for receipt in (plan['failed_job'], plan['diagnosis']):
        if file_hash(receipt['path']) != receipt['sha256']: raise ValueError('Preserved failure changed')
    for name, digest in {**p['input_hashes'], 'verbatims.jsonl':p['verbatims_sha256'],
            'generation-plan.jsonl':p['generation_plan_sha256']}.items():
        if file_hash(HERE/name) != digest: raise ValueError('Child input changed')
    for name, digest in {**p['shared_sources'], **p['wake_assets']}.items():
        if file_hash(ROOT/name) != digest: raise ValueError(f'Frozen child source/asset changed: {name}')
    return p


def generate():
    import torch
    import synthesis
    from assets import verify_assets
    validate()
    path = CACHE/'replacements.jsonl'
    if path.exists(): raise FileExistsError('Repair generation already recorded')
    plan = json.loads((HERE/'repair-plan.json').read_text()); record = plan['records'][0]; old = record['parent']
    if verify_assets('parler') != old['tts_assets_sha256']: raise ValueError('TTS assets changed')
    torch.set_num_threads(4)
    generator, model = synthesis.load_engine('parler')
    if model != json.loads((rev.OLD_CACHE/'parler-model-identity.json').read_text()):
        raise ValueError('TTS weights changed')
    if file_hash(rev.OLD_CACHE/'parler-recipe.json') != old['generation_recipe_sha256']:
        raise ValueError('Generation recipe changed')
    refs = {r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    originals = {r['recording_id']:r for r in read_rows(rev.CACHE/'parler-manifest.jsonl')}
    batch = [originals[key] for key in record['batch_ids']]
    voice = next(v for v in rev.VOICES if v['id'] == old['voice_group'])
    phones = {r['id']:r['phones'] for r in rev.compile_catalog(list(refs.values()))['entries']}[old['verbatim_id']]
    index = record['batch_ids'].index(TARGET)
    for attempt in record['attempts']:
        seed = (batch[0]['seed'] + attempt - 1) % 2**32; torch.manual_seed(seed)
        try:
            waves, rate = generator([refs[r['verbatim_id']]['text'] for r in batch], voice)
            if len(waves) != len(batch): raise ValueError('Wrong generation batch size')
            ready = synthesis.prepare_audio(waves[index], rate)
            capacity = require_feasible(ready, phones)
            metadata = generator.last_metadata[index]
        except ValueError as exc:
            immutable_json(CACHE/f'attempt-{attempt}-failed.json', dict(attempt=attempt, seed=seed, error=str(exc)))
            if attempt == record['attempts'][-1]: raise
            continue
        row = {**old, 'audio':str(CACHE/'audio'/f'{TARGET}.wav'),
            'native_audio':str(CACHE/'native'/f'{TARGET}.wav')}
        item = synthesis.persist_audio(CACHE, row, waves[index], rate, ready)
        item.update(source_version='v2-repair1', repair_parent_record_sha256=identity(old),
            repair_plan_sha256=file_hash(HERE/'repair-plan.json'), batch_seed=seed,
            generation_attempt=attempt, generation_metadata=metadata,
            generation_source_sha256=file_hash(Path(__file__)), ctc_capacity=capacity)
        merge_records([old], [item], plan)
        write_rows(path, [item])
        immutable_json(CACHE/'repair-generation.json', dict(complete=True, recordings=1,
            replacement_manifest_sha256=file_hash(path), parent_preserved=True,
            duration_seconds=item['duration_seconds'], ctc_capacity=capacity, attempt=attempt, seed=seed))
        return json.loads((CACHE/'repair-generation.json').read_text())


def assemble():
    validate(); plan = json.loads((HERE/'repair-plan.json').read_text())
    replacements = read_rows(CACHE/'replacements.jsonl')
    parents = [r for e in ENGINES for r in read_rows(rev.CACHE/f'{e}-manifest.jsonl')]
    rows = merge_records(parents, replacements, plan)
    for e in ENGINES:
        write_rows(CACHE/f'{e}-manifest.jsonl', [r for r in rows if r['engine'] == e])
        # Preserve the exact wake manifest bytes, not merely equivalent objects.
        target = CACHE/f'wake-{e}-manifest.jsonl'; payload = (rev.CACHE/target.name).read_bytes()
        if target.exists():
            if target.read_bytes() != payload: raise FileExistsError(target)
        else:
            with target.open('xb') as stream: stream.write(payload)
    result = audit_all_splits()
    import wake_benchmark
    wake_benchmark.audit_wake('dev')
    immutable_json(CACHE/'assembly-audit.json', {**result, 'wake_recordings':288, 'wake_identity_audit_passed':True})
    return result


def audit_all_splits():
    import soundfile as sf
    import numpy as np
    validate(); parent_audit = corpus.audit_all_splits()
    plan = json.loads((HERE/'repair-plan.json').read_text())
    parents = [r for e in ENGINES for r in read_rows(rev.CACHE/f'{e}-manifest.jsonl')]
    replacements = read_rows(CACHE/'replacements.jsonl')
    expected = merge_records(parents, replacements, plan)
    rows = [r for e in ENGINES for r in read_rows(CACHE/f'{e}-manifest.jsonl')]
    if rows != expected: raise ValueError('Child manifest differs from the single declared replacement')
    hashes = [r['pcm_sha256'] for r in rows]
    if len(set(hashes)) != len(hashes): raise ValueError('Repair introduces duplicate PCM')
    phones = {r['id']:r['phones'] for r in rev.compile_catalog(read_rows(HERE/'verbatims.jsonl'))['entries']}
    for r in replacements:
        old = next(p for p in parents if p['recording_id'] == r['recording_id'])
        if (r['generation_attempt'] not in (2,3) or r['batch_seed'] != (old['seed']+r['generation_attempt']-1)%2**32
                or r['repair_plan_sha256'] != file_hash(HERE/'repair-plan.json')
                or r['generation_source_sha256'] != file_hash(Path(__file__))):
            raise ValueError('Replacement generation lineage changed')
        for key,digest in (('audio','audio_sha256'), ('native_audio','native_sha256')):
            path = Path(r[key]).resolve()
            if not path.is_relative_to(CACHE.resolve()) or file_hash(path) != r[digest]:
                raise ValueError('Replacement payload changed or escapes child cache')
        pcm, rate = sf.read(r['audio'], dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all() or hashlib.sha256(pcm.tobytes()).hexdigest() != r['pcm_sha256']:
            raise ValueError('Invalid replacement PCM')
        capacity = ctc_capacity(len(pcm), phones[r['verbatim_id']])
        if not capacity['feasible'] or capacity != r['ctc_capacity']: raise ValueError('Replacement CTC drift')
    for e in ENGINES:
        if file_hash(CACHE/f'wake-{e}-manifest.jsonl') != plan['parent_manifests'][f'wake-{e}-manifest.jsonl']:
            raise ValueError('Wake corpus changed')
    return {**parent_audit, 'manifests':{e:file_hash(CACHE/f'{e}-manifest.jsonl') for e in ENGINES},
        'repair_plan_sha256':file_hash(HERE/'repair-plan.json'), 'repaired_training_recordings':len(replacements),
        'parent_manifests':parent_audit['manifests'], 'held_out_audio_unchanged':True}


def load_split(split):
    audit_all_splits(); refs = {r['id']:r for r in read_rows(HERE/'verbatims.jsonl')}
    return [{**r, **refs[r['verbatim_id']]} for e in ENGINES
            for r in read_rows(CACHE/f'{e}-manifest.jsonl') if r['split'] == split]


@contextmanager
def bind():
    import study, synthbench, fitting, finishing, reporting, wake_benchmark, final_report
    changes = []
    def set_(module, key, value):
        changes.append((module, key, getattr(module,key))); setattr(module,key,value)
    original_wake_audit = wake_benchmark.audit_wake
    def unchanged_wake(split):
        saved = wake_benchmark.HERE, wake_benchmark.CACHE
        wake_benchmark.HERE, wake_benchmark.CACHE = rev.HERE, rev.CACHE
        try: return original_wake_audit(split)
        finally: wake_benchmark.HERE, wake_benchmark.CACHE = saved
    with version_context.bind():
        try:
            for module in (study,synthbench,fitting,finishing,reporting,wake_benchmark,final_report):
                for key,value in (('HERE',HERE),('CACHE',CACHE)):
                    if hasattr(module,key): set_(module,key,value)
            for module in (study,synthbench): set_(module,'validate_frozen_study',validate)
            for module in (synthbench,fitting):
                set_(module,'load_split',load_split); set_(module,'audit_all_splits',audit_all_splits)
            set_(wake_benchmark,'audit_wake',unchanged_wake)
            synthbench.intent_use_cases.cache_clear()
            yield
        finally:
            for module,key,value in reversed(changes): setattr(module,key,value)
            synthbench.intent_use_cases.cache_clear()
