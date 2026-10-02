"""Independent calibration coverage extension; parent training/test stay frozen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scripts.train.pcv_public_data import ROOT, STUDY, CACHE, sha, save, read_rows, _words
from scripts.runtime.pcv_registry import reconciled_registry, compile_registry
from scripts.runtime.pcv_schema import normalize_text, canonical, INTENTS
from vcm.frontend import features, SPEC

CHILD = ROOT/'experiments/pcv-human-slots-v1-qualification1'
WORK = ROOT/'.cache/pcv-human-slots-v1-qualification1'


def freeze_child():
    if (CHILD/'protocol.json').exists():
        data = json.loads((CHILD/'protocol.json').read_text())
        for p, digest in data['bindings'].items():
            if sha(ROOT/p) != digest:
                raise ValueError('Qualification input identity changed: ' + p)
        return data
    CHILD.mkdir(parents=True, exist_ok=True)
    base = json.loads((STUDY/'catalog.json').read_text())
    configured = json.loads((STUDY/'custom-slots.json').read_text())
    save(CHILD/'custom-slots.json', reconciled_registry(base, configured))
    _, catalog = compile_registry(base, CHILD/'custom-slots.json', _words())
    save(CHILD/'catalog.json', catalog)
    prompts = json.loads((STUDY/'synthesis-prompts.json').read_text())
    texts = {normalize_text(r.get('spoken_text', r['text'])) for r in prompts}
    for entry in catalog['entries']:
        if entry['id'].startswith('CUSTOM-') and entry['text'] not in texts:
            prompts.append(entry)
            texts.add(entry['text'])
    save(CHILD/'calibration-prompts.json', prompts)
    paths = [STUDY/'protocol.json', STUDY/'data-freeze.json', STUDY/'training-freeze.json',
             STUDY/'baseline.json', STUDY/'output.schema.json', CHILD/'catalog.json', CHILD/'custom-slots.json', CHILD/'calibration-prompts.json',
             ROOT/'scripts/runtime/pcv_registry.py', ROOT/'scripts/train/pcv_calibration.py',
             ROOT/'scripts/test/pcv_public_qualification.py', ROOT/'scripts/test/test_pcv_qualification.py', CHILD/'run.py',
             ROOT/'experiments/pcv-synthetic-v1/synthesis.py', ROOT/'.cache/pcv-synthetic-v1/anchors/cal-us-f.wav']
    result = {'experiment': 'pcv-human-slots-v1-qualification1', 'parent': 'pcv-human-slots-v1',
        'purpose': 'Pre-test review repair: add independently held-out calibration coverage, reconcile registry, verify frozen identities, correct per-intent precision',
        'training_changed': False, 'development_changed': False, 'test_changed': False,
        'weight_selection': 'unchanged parent development-only selection; no test-selected models',
        'catalog_change': 'retain every parent command/ID and add the configured Mom alias mother; re-calibrate both models on this identical catalog',
        'calibration_voice': 'qwen/cal-us-f', 'calibration_prompts': len(prompts),
        'generation': {'engine': 'Qwen3-TTS-12Hz-1.7B-Base', 'local_files_only': True, 'dtype': 'bfloat16',
                       'batch_size': 2, 'temperature': .75, 'top_p': .95, 'max_new_tokens': 256, 'max_attempts': 3,
                       'seed_rule': 'first 8 hex of SHA256(qualification1/first_prompt_id), plus retry index'},
        'calibration_gates': {'precision': .99, 'recall': .90, 'negative_false_accept_rate': .01,
            'strata': ['all original human', 'all synthetic'], 'missing_positive_intents': 'disable policy',
            'light_target_gate': 'named probability >=0.9 for named, <=0.1 for all; uncertain presence rejects invalid_argument',
            'score_grid': '11 calibration score quantiles plus -100', 'margin_grid': [0,.002,.005,.01,.02,.05,.1],
            'intent_probability_grid': [0,.5,.7,.9]},
        'test_inference_accessed': False, 'installation_authorized': False,
        'limitations': ['Generated prompt labels not independently listened', 'Calibration shares its reserved voice with existing calibration only; never with train/dev/test',
                       'Parent public corpus has 21/93 exact human phrase coverage; SAPI test is one voice, not full qualification'],
        'bindings': {str(p.relative_to(ROOT)): sha(p) for p in paths}}
    save(CHILD/'protocol.json', result)
    return result


def generate():
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    protocol = freeze_child()
    if not (STUDY/'selection-before-test.json').exists():
        raise RuntimeError('Finish the parent fitting job first to avoid competing GPU memory allocations')
    if (WORK/'calibration.jsonl').exists():
        raise FileExistsError('Calibration already prepared; preserve immutable evidence')
    sys.path.insert(0, str(ROOT/'experiments/pcv-synthetic-v1'))
    from synthesis import prepare_audio, ANCHOR
    torch.set_num_threads(2)
    folder = ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-Base'
    assets = {str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file() and p.suffix in {'.safetensors','.json'} and '.cache' not in p.relative_to(folder).parts}
    WORK.mkdir(parents=True, exist_ok=True)
    save(WORK/'model-identity.json', assets)
    model = Qwen3TTSModel.from_pretrained(str(folder), device_map='cuda:0', dtype=torch.bfloat16,
                                         attn_implementation='sdpa', local_files_only=True)
    prompt = model.create_voice_clone_prompt(ref_audio=str(ROOT/'.cache/pcv-synthetic-v1/anchors/cal-us-f.wav'), ref_text=ANCHOR)[0]
    prompts = json.loads((CHILD/'calibration-prompts.json').read_text())
    catalog = json.loads((CHILD/'catalog.json').read_text())
    by_text = {r['text']: r for r in catalog['entries']}
    existing = {r['id']: r for r in read_rows(WORK/'generated.jsonl')} if (WORK/'generated.jsonl').exists() else {}
    for r in existing.values():
        if sha(r['audio']) != r['audio_sha256'] or sha(r['features']) != r['features_sha256']:
            raise ValueError('Previously generated calibration changed')
    for start in range(0, len(prompts), 2):
        batch = prompts[start:start+2]
        ids = ['qwen-schema-cal/'+r['id'] for r in batch]
        if all(i in existing for i in ids):
            continue
        seed = int(hashlib.sha256(('qualification1/'+batch[0]['id']).encode()).hexdigest()[:8], 16)
        began = time.perf_counter()
        for attempt in range(3):
            torch.manual_seed((seed+attempt) % 2**32)
            try:
                waves, rate = model.generate_voice_clone(text=[r.get('spoken_text', r['text']) for r in batch],
                    language=['English']*len(batch), voice_clone_prompt=[prompt]*len(batch),
                    non_streaming_mode=True, max_new_tokens=256, do_sample=True, temperature=.75, top_p=.95)
                if len(waves) != len(batch):
                    raise ValueError('Wrong generated batch size')
                prepared = [prepare_audio(wave, rate) for wave in waves]
                import io
                for r, value in zip(batch, prepared, strict=True):
                    pcm_check, _ = sf.read(io.BytesIO(value[1]), dtype='float32')
                    phones_check = by_text[normalize_text(r.get('spoken_text', r['text']))]['phones']
                    capacity = max(0, 1+(len(pcm_check)-400)//160)
                    if capacity < len(phones_check)+sum(a==b for a,b in zip(phones_check,phones_check[1:])) or value[3]['duration_seconds'] > 12:
                        raise ValueError('Calibration utterance violates finite CTC/endpoint constraints')
                break
            except ValueError as exc:
                with (WORK/'failed-attempts.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(canonical({'ids':ids, 'attempt':attempt, 'seed':seed+attempt, 'error':str(exc)})+'\n')
                if attempt == 2:
                    raise
        for r, rid, value in zip(batch, ids, prepared, strict=True):
            if rid in existing:
                continue
            text = normalize_text(r.get('spoken_text', r['text']))
            entry = by_text[text]
            if canonical(entry['command']) != canonical(r['command']):
                raise ValueError('Calibration prompt/catalog mismatch')
            native, derived, native_metrics, metrics, pcm_hash = value
            import io
            pcm, _ = sf.read(io.BytesIO(derived), dtype='float32')
            f = features(pcm)
            phones = entry['phones']
            if f.shape[1] < len(phones)+sum(a==b for a,b in zip(phones,phones[1:])) or metrics['duration_seconds'] > 12:
                raise ValueError('Calibration utterance violates finite CTC/endpoint constraints')
            paths = {k: WORK/k/(r['id']+ext) for k,ext in [('audio','.wav'),('native','.wav'),('features','.npy')]}
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.exists():
                    raise FileExistsError('Orphan calibration output; preserve and inspect')
            paths['native'].write_bytes(native)
            paths['audio'].write_bytes(derived)
            with paths['features'].open('xb') as stream:
                np.save(stream, f)
            c = r['command']
            row = {'id': rid, 'phrase_id': r['id'], 'text': text, 'expected': c, 'source': 'Qwen-schema-calibration',
                'split': 'calibration', 'speaker': 'qwen/cal-us-f', 'pair': rid, 'synthetic': True,
                'audio': str(paths['audio']), 'audio_sha256': sha(paths['audio']), 'native_audio':str(paths['native']),
                'native_sha256': sha(paths['native']), 'pcm_sha256': pcm_hash, 'native_metrics':native_metrics,
                'features':str(paths['features']), 'features_sha256':sha(paths['features']), 'phones':phones,
                'frames':f.shape[1], 'intent_index':INTENTS.index(c['intent'])+1 if c else 0,
                'target_index':int(c['slots']['light_id']!='light.all') if c and 'light_id' in c['slots'] else -100,
                'generation_seed':seed+attempt, 'prompt_listener_verified':False, **metrics}
            with (WORK/'generated.jsonl').open('a',encoding='utf8') as stream:
                stream.write(canonical(row)+'\n')
                stream.flush()
            existing[rid] = row
        print('CALIBRATION_GENERATED',len(existing),'/',len(prompts),'seconds',round(time.perf_counter()-began,1),flush=True)
    # Bind unchanged parent splits and quarantine any duplicate generated PCM.
    parent = {s:read_rows(CACHE/'prepared'/f'{s}.jsonl') for s in ('train','dev','calibration','test')}
    noncal_speakers = {r['speaker'] for s in ('train','dev','test') for r in parent[s]}
    all_hashes = {r['pcm_sha256'] for rows in parent.values() for r in rows}
    kept, excluded = [], []
    for row in existing.values():
        if row['speaker'] in noncal_speakers:
            raise ValueError('Calibration speaker leakage')
        if row['pcm_sha256'] in all_hashes:
            excluded.append({'id':row['id'], 'reason':'duplicate_pcm'})
        else:
            kept.append(row)
            all_hashes.add(row['pcm_sha256'])
    path = WORK/'calibration.jsonl'
    with path.open('x',encoding='utf8') as stream:
        for row in parent['calibration']+kept:
            stream.write(canonical(row)+'\n')
    final_assets = {p:sha(folder/p) for p in assets}
    if final_assets != assets:
        raise ValueError('TTS assets changed during generation')
    save(CHILD/'calibration-freeze.json', {'rows':len(parent['calibration'])+len(kept),'added':len(kept),'excluded':excluded,
         'manifest':str(path.relative_to(ROOT)), 'sha256':sha(path), 'generation_protocol_sha256':sha(CHILD/'protocol.json'),
         'tts_model_identity_sha256':sha(WORK/'model-identity.json'), 'parent_test_sha256':sha(CACHE/'prepared/test.jsonl'),
         'no_test_inference':True})
