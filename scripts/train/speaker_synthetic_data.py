"""Versioned GPU synthesis and explicit synthetic research preparation.

Old human-only protocols remain immutable. Generated labels describe intended
text, not independent listening, human speakers or qualified recognition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

from scripts.runtime.pcv_schema import (
    DEFAULT_CUSTOM, INTENTS, canonical, compile_inventory, normalize_text, validate_command,
)
from scripts.test.speaker_demo_suite import (
    file_digest, pcm_digest, read_audio, read_json, read_jsonl, write_new,
)
from scripts.train.speaker_dataset_install import write_audio

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'experiments/speaker-synthetic-completion-v1'
CACHE = ROOT / '.cache/speaker-synthetic-completion-v1'
COLORS = ('red', 'blue', 'green', 'yellow', 'white', 'orange', 'purple', 'pink',
          'cyan', 'magenta', 'warm white', 'cool white')
FOCUS = {'CALL', 'MESSAGE', 'LIST_REMINDERS', 'TEMPERATURE', 'CREATE_REMINDER'}
ANCHOR_TEXT = 'Hello, I am ready to help. Please speak clearly and tell me what you would like to do today.'


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _verify(bindings):
    for name, digest in bindings.items():
        if file_digest(_path(name)) != digest:
            raise ValueError('Frozen input identity mismatch: ' + name)


def build_prompts(definitions, custom):
    """Sparse families, original rejections and the entire declared color matrix."""
    cases = [dict(r) for r in definitions if r['command'] is None or r['command']['intent'] in FOCUS]
    for light in [{'id': 'light.all', 'name': 'the lights'}, *custom['lights']]:
        for color in COLORS:
            cases.append({'id': 'matrix-' + light['id'] + '-' + color.replace(' ', '_'),
                          'text': f'Set {light["name"]} color to {color}',
                          'command': {'intent': 'COLOR', 'slots': {'light_id': light['id'], 'color': color}}})
    grouped = {}
    for row in cases:
        value = row['command']
        if value is not None:
            validate_command(value)
        text = row.get('spoken_text', row['text'])
        key = normalize_text(text)
        if not key:
            raise ValueError('Empty synthesis prompt')
        if key in grouped:
            if canonical(grouped[key]['command']) != canonical(value):
                raise ValueError('conflicting synthesis prompt supervision')
            grouped[key]['case_ids'].append(row['id'])
        else:
            grouped[key] = {'id': 'prompt-' + _hash([key, value])[:16],
                            'text': text, 'spoken_text': text, 'command': value,
                            'case_ids': [row['id']]}
    return list(grouped.values())


def build_plan(prompts, voices):
    ids, identities, splits = set(), set(), set()
    rows = []
    for voice in voices:
        if voice['split'] not in {'train', 'dev', 'calibration'}:
            raise ValueError('Only synthetic train/dev/calibration splits are allowed')
        identity = _hash([voice['prompt'], voice['design_seed']])
        if voice['id'] in ids or identity in identities:
            raise ValueError('Duplicate designed voice identity or alias')
        ids.add(voice['id']); identities.add(identity); splits.add(voice['split'])
        for prompt in prompts:
            if prompt['command'] is not None:
                validate_command(prompt['command'])
            key = voice['id'] + '/' + prompt['id']
            rows.append({**prompt, 'id': 'qsc1/' + key, 'prompt_id': prompt['id'],
                         'kind': 'synthetic', 'voice_id': voice['id'], 'split': voice['split'],
                         'speaker_id': 'qwen-completion/' + voice['id'], 'pair_id': 'qsc1/' + key})
    if splits != {'train', 'dev', 'calibration'}:
        raise ValueError('All three synthetic voice splits are required')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prompt/voice cell')
    return rows


def audit_anchors(anchors, voices):
    expected = {v['id']: v['split'] for v in voices}
    seen, hashes = set(), set()
    for anchor in anchors:
        if (anchor['id'] not in expected or anchor['id'] in seen
                or anchor['split'] != expected[anchor['id']]
                or anchor['pcm_sha256'] in hashes):
            raise ValueError('Duplicate or mismatched voice anchor')
        seen.add(anchor['id']); hashes.add(anchor['pcm_sha256'])
    if seen != expected.keys():
        raise ValueError('Missing voice anchor')
    return {'voices': len(seen), 'duplicate_anchor_pcm': 0}


def audit_generated(rows, plan, existing):
    expected = {r['id']: r for r in plan}
    seen = set()
    files = {r['sha256'] for r in existing if r.get('sha256')}
    pcm = {r['pcm_sha256'] for r in existing if r.get('pcm_sha256')}
    speaker_splits = {r['speaker_id']: r['split'] for r in existing if r.get('speaker_id')}
    for row in rows:
        if row['id'] in seen or row['id'] not in expected:
            raise ValueError('Unexpected or duplicate generation cell')
        target = expected[row['id']]
        if row['kind'] != 'synthetic':
            raise ValueError('Generated audio must stay explicitly synthetic')
        if (any(row[k] != target[k] for k in ('split', 'voice_id', 'spoken_text', 'speaker_id', 'pair_id'))
                or canonical(row['command']) != canonical(target['command'])):
            raise ValueError('Generated record differs from its frozen plan')
        if row['qa_eligible'] is not True or row['audio_flags']:
            raise ValueError('Failed generation QA cannot enter fitting')
        if row['command'] is not None:
            validate_command(row['command'])
        if row['sha256'] in files or row['pcm_sha256'] in pcm:
            raise ValueError('Generated file/PCM duplicate or existing-source overlap')
        if row['speaker_id'] in speaker_splits and speaker_splits[row['speaker_id']] != row['split']:
            raise ValueError('Cross-split generated speaker overlap')
        speaker_splits[row['speaker_id']] = row['split']
        seen.add(row['id']); files.add(row['sha256']); pcm.add(row['pcm_sha256'])
    missing = sorted(expected.keys() - seen)
    return {'complete': not missing, 'planned': len(expected), 'present': len(rows),
            'missing': missing, 'splits': dict(Counter(r['split'] for r in rows)),
            'kind': 'synthetic', 'independent_listening': False,
            'human_qualification_coverage_added': 0, 'exact_pcm_overlap': 0}


def convert_waveform(waveform, rate, *, phone_count):
    from scipy.signal import resample_poly
    values = np.asarray(waveform, dtype='float32')
    if (values.ndim != 1 or not values.size or not np.isfinite(values).all()
            or rate < 8000 or np.max(np.abs(values)) > 1):
        raise ValueError('Invalid native generated waveform')
    divisor = math.gcd(int(rate), 16000)
    values = resample_poly(values, 16000 // divisor, int(rate) // divisor).astype('float32')
    frames = max(0, 1 + (len(values) - 400) // 160)
    rms = float(np.sqrt(np.mean(values ** 2)))
    clipping = float(np.mean(np.abs(values) >= .999))
    if not 1600 <= len(values) <= 192000 or rms < 1e-4 or clipping > .01 or (frames + 1) // 2 < phone_count:
        raise ValueError('Generated waveform fails duration/energy/clipping/CTC-frame QA')
    return values, {'sample_rate': 16000, 'samples': len(values), 'rms': rms,
                    'clipped_fraction': clipping, 'ctc_frames': (frames + 1) // 2}


def extend_catalog(original, prompts):
    """Preserve every existing acoustic sequence; compile only new texts."""
    from vcm.catalog import dictionary, PHONES
    integers = dictionary(ROOT / 'STT/alternatives/sources/cmudict.dict')
    words = {name: [PHONES[i - 1] for i in phones] for name, phones in integers.items()}
    words.update(read_json(ROOT / 'experiments/pcv-synthetic-v2/lexicon.json')['words'])
    known = {normalize_text(r.get('spoken_text', r['text'])): r for r in original['entries']}
    extra = []
    for row in prompts:
        key = normalize_text(row['spoken_text'])
        if key in known:
            if canonical(known[key]['command']) != canonical(row['command']):
                raise ValueError('Existing catalog conflicts with new declared prompt')
        else:
            extra.append(row)
    compiled = compile_inventory(extra, words)['entries'] if extra else []
    signatures = {tuple(r['phones']): canonical(r['command']) for r in original['entries']}
    for row in compiled:
        signature = canonical(row['command'])
        if tuple(row['phones']) in signatures and signatures[tuple(row['phones'])] != signature:
            raise ValueError('New catalog pronunciation conflicts with existing command')
    return {**original, 'entries': [*original['entries'], *compiled], 'acoustic_qualification': False}


def freeze():
    """Seal this generation version before any model call or new audio."""
    protocol_path = HERE / 'protocol.json'
    if protocol_path.exists():
        protocol = read_json(protocol_path)
        _verify(protocol['bindings'])
        return protocol
    definitions_path = ROOT / 'experiments/pcv-b1-three-engine-v1/prompts.json'
    original_path = ROOT / 'experiments/pcv-release-train-v1/catalog.json'
    prompts = build_prompts(read_json(definitions_path), DEFAULT_CUSTOM)
    descriptions = [
        ('train-en', 'train', 'A clear adult male voice speaking English with a neutral American accent, medium pitch and natural conversational delivery. Clean dry speech, no background sound.'),
        ('train-ph', 'train', 'An adult woman speaking only the supplied English text with a requested Filipino English accent, calm medium pitch and conversational pace. Clean dry speech.'),
        ('dev-en', 'dev', 'An older woman with a clear British English accent and a soft slightly textured voice, natural conversational delivery. Clean dry speech, no music.'),
        ('dev-ph', 'dev', 'A young adult man speaking only the supplied English text with a requested Philippine English accent, bright voice and relaxed conversational pace. Clean dry speech.'),
        ('cal-en', 'calibration', 'An adult man with an Australian English accent, a warm resonant voice and measured conversational delivery. Clean dry speech, no background sound.'),
        ('cal-ph', 'calibration', 'A mature woman speaking only the supplied English text with a requested Filipino English accent, a low clear voice and measured conversational pace. Clean dry speech.'),
    ]
    voices = [dict(id=name, split=split, prompt=description, design_seed=61016 + i,
                   identity_source='original_synthetic_voice_design', accent_verified=False)
              for i, (name, split, description) in enumerate(descriptions)]
    plan = build_plan(prompts, voices)
    catalog = extend_catalog(read_json(original_path), prompts)
    write_new(HERE / 'prompts.json', prompts)
    write_new(HERE / 'voices.json', voices)
    write_new(HERE / 'catalog.json', catalog)
    write_new(HERE / 'generation-plan.json', plan)
    sources = [Path(__file__), definitions_path, original_path,
               HERE / 'prompts.json', HERE / 'voices.json', HERE / 'catalog.json',
               HERE / 'generation-plan.json', HERE / 'requirements-synthesis.lock',
               ROOT / 'data/sources/qwen-synthetic-completion.source.json',
               ROOT / 'scripts/runtime/pcv_schema.py', ROOT / 'vcm/catalog.py',
               ROOT / 'vcm/frontend.py', ROOT / 'scripts/train/speaker_dataset_install.py',
               ROOT / 'scripts/test/speaker_demo_suite.py',
               ROOT / 'STT/alternatives/sources/cmudict.dict',
               ROOT / 'experiments/pcv-synthetic-v2/lexicon.json',
               ROOT / 'experiments/speaker-demo-v2/protocol.json',
               ROOT / '.cache/speaker-dataset-v1/verified/audit.json']
    protocol = {
        'experiment_id': 'speaker-synthetic-completion-v1', 'date': '2026-10-16',
        'authorization': 'User has no new human recordings and explicitly chose synthetic generation; SSH NVIDIA GPU execution confirmed.',
        'requirements': ['ME-01', 'ME-02', 'ME-03', 'ME-04', 'ME-07', 'ME-08'],
        'hypothesis': 'Additional designed synthetic voice conditions for sparse complete commands improve development recognition at fixed VCM architecture; no acoustic gain is presumed.',
        'parent_protocol': 'experiments/speaker-demo-v2/protocol.json',
        'generation_plan': 'experiments/speaker-synthetic-completion-v1/generation-plan.json',
        'voices_file': 'experiments/speaker-synthetic-completion-v1/voices.json',
        'catalog_file': 'experiments/speaker-synthetic-completion-v1/catalog.json',
        'planned': len(plan), 'prompts_per_voice': len(prompts), 'voices': len(voices),
        'generation': {'batch_size': 4, 'attempts': 3, 'dtype': 'bfloat16', 'attention': 'sdpa',
                       'max_new_tokens': 192, 'anchor_max_new_tokens': 384,
                       'temperature': .75, 'top_p': .95, 'anchor_text': ANCHOR_TEXT,
                       'seed_rule': 'SHA256([experiment,voice_id,batch_offset,attempt]) first 8 hex modulo 2**32',
                       'cuda_memory_fraction': .45},
        'scope': 'Separate synthetic train/dev/calibration; existing human and published evaluation membership preserved. No TTS model ships.',
        'quality': 'Native and derived waveform, finite energy/duration and sufficient CTC frames only; not independent listening or accent verification.',
        'catalog_rule': 'Existing 1031 entries preserved plus all declared light/color pairs; no individual test-derived addition.',
        'human_qualification': False, 'installation_allowed': False,
        'application': {'intent': .95, 'each_required_slot': .90},
        'bindings': {_relative(p): file_digest(p) for p in sources},
    }
    write_new(protocol_path, protocol)
    return protocol


def _wav(values, rate):
    buffer = io.BytesIO()
    sf.write(buffer, values, rate, format='WAV', subtype='PCM_16')
    return buffer.getvalue()


def _store_waveform(values, rate, *, native, audio, phone_count):
    # Validate floating output before PCM encoding; derive from saved native PCM.
    convert_waveform(values, rate, phone_count=phone_count)
    native.parent.mkdir(parents=True, exist_ok=True)
    write_audio(native, _wav(values, rate))
    original, saved_rate = sf.read(native, dtype='float32')
    derived, quality = convert_waveform(original, saved_rate, phone_count=phone_count)
    audio.parent.mkdir(parents=True, exist_ok=True)
    write_audio(audio, _wav(derived, 16000))
    actual, _ = sf.read(audio, dtype='float32')
    return {'audio': _relative(audio), 'sha256': file_digest(audio),
            'pcm_sha256': pcm_digest(actual), 'native_audio': _relative(native),
            'native_sha256': file_digest(native), 'native_sample_rate': int(rate), **quality}


def generate():
    """Run on one explicitly visible CUDA GPU; resume only verified saved cells."""
    import importlib.metadata
    import torch
    from qwen_tts import Qwen3TTSModel
    protocol = freeze()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is hidden by the command sandbox; run this stage through approved host execution')
    source = read_json(ROOT / 'data/sources/qwen-synthetic-completion.source.json')
    _verify({r['path']: r['sha256'] for r in source['files']})
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(protocol['generation']['cuda_memory_fraction'])
    torch.backends.cuda.matmul.allow_tf32 = False
    CACHE.mkdir(parents=True, exist_ok=True)
    voices, plan = read_json(HERE / 'voices.json'), read_json(HERE / 'generation-plan.json')
    cfg = protocol['generation']
    model_root = ROOT / '.cache/tts/models'

    def load(variant):
        model = Qwen3TTSModel.from_pretrained(
            str(model_root / ('Qwen3-TTS-12Hz-1.7B-' + variant)), device_map='cuda:0',
            dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        print('MODEL_READY', variant, torch.cuda.get_device_name(0), flush=True)
        return model

    began = time.perf_counter()
    anchor_receipt = CACHE / 'anchors.json'
    anchors = read_json(anchor_receipt) if anchor_receipt.exists() else []
    if not anchors:
        model = load('VoiceDesign')
        for voice in voices:
            saved_anchor = CACHE / 'anchors' / (voice['id'] + '.json')
            if saved_anchor.exists():
                anchor = read_json(saved_anchor)
                if any(anchor[k] != voice[k] for k in ('id', 'split', 'prompt', 'design_seed')):
                    raise ValueError('Preserved anchor does not match frozen voice condition')
                read_audio(anchor, ROOT)
                _verify({anchor['native_audio']: anchor['native_sha256']})
                anchors.append(anchor)
                continue
            torch.manual_seed(voice['design_seed'])
            waves, rate = model.generate_voice_design(
                text=ANCHOR_TEXT, instruct=voice['prompt'], language='English',
                max_new_tokens=cfg['anchor_max_new_tokens'], do_sample=True, temperature=.8)
            value = _store_waveform(waves[0], rate,
                native=CACHE / 'anchors/native' / (voice['id'] + '.wav'),
                audio=CACHE / 'anchors/pcm16' / (voice['id'] + '.wav'), phone_count=1)
            anchor = {**voice, **value, 'listener_verified': False}
            write_new(saved_anchor, anchor)
            anchors.append(anchor)
            print('ANCHOR_READY', voice['id'], value['samples'] / 16000, flush=True)
        audit_anchors(anchors, voices)
        write_new(anchor_receipt, anchors)
        del model; gc.collect(); torch.cuda.empty_cache()
    audit_anchors(anchors, voices)
    for anchor in anchors:
        read_audio(anchor, ROOT)
        _verify({anchor['native_audio']: anchor['native_sha256']})
    model = load('Base')
    clone = {r['id']: model.create_voice_clone_prompt(ref_audio=str(_path(r['native_audio'])),
              ref_text=ANCHOR_TEXT)[0] for r in anchors}
    anchor_by_id = {r['id']: r for r in anchors}
    manifest = CACHE / 'generated.jsonl'
    existing = read_jsonl(manifest) if manifest.exists() else []
    for row in existing:
        read_audio(row, ROOT)
    audit_generated(existing, plan, [])
    done = {r['id']: r for r in existing}
    entries = read_json(HERE / 'catalog.json')['entries']
    by_text = {normalize_text(r.get('spoken_text', r['text'])): r for r in entries}
    for voice in voices:
        voice_rows = [r for r in plan if r['voice_id'] == voice['id']]
        for offset in range(0, len(voice_rows), cfg['batch_size']):
            batch = voice_rows[offset:offset + cfg['batch_size']]
            if all(r['id'] in done for r in batch):
                continue
            for attempt in range(cfg['attempts']):
                if all(r['id'] in done for r in batch):
                    break
                attempt_path = CACHE / 'attempts' / f'{voice["id"]}-{offset:04d}-{attempt}.json'
                if attempt_path.exists():
                    continue
                seed = int(_hash([protocol['experiment_id'], voice['id'], offset, attempt])[:8], 16)
                torch.manual_seed(seed); torch.cuda.empty_cache()
                start = time.perf_counter()
                waves, rate = model.generate_voice_clone(
                    text=[r['spoken_text'] for r in batch], language=['English'] * len(batch),
                    voice_clone_prompt=[clone[voice['id']]] * len(batch), non_streaming_mode=True,
                    max_new_tokens=cfg['max_new_tokens'], do_sample=True,
                    temperature=cfg['temperature'], top_p=cfg['top_p'])
                if len(waves) != len(batch):
                    raise ValueError('TTS batch output count differs from planned cells')
                failures = []
                for row, waveform in zip(batch, waves):
                    if row['id'] in done:
                        continue
                    labels = by_text[normalize_text(row['spoken_text'])]['phones']
                    required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
                    try:
                        info = _store_waveform(waveform, rate,
                            native=CACHE / 'native' / voice['id'] / f'{row["prompt_id"]}-a{attempt}.wav',
                            audio=ROOT / 'data/audio/speaker-synthetic-completion-v1' / voice['id'] / f'{row["prompt_id"]}-a{attempt}.wav',
                            phone_count=required)
                        record = {**row, **info, 'role': 'command', 'source': 'Qwen3-designed-completion-v1',
                                  'license': 'Generated with Apache-2.0 Qwen weights; intended-text metadata, no source recording clone',
                                  'qa_eligible': True, 'audio_flags': [], 'metric_eligible': True,
                                  'reference_status': 'supported' if row['command'] else 'unsupported',
                                  'listener_verified': False, 'accent_verified': False,
                                  'anchor_pcm_sha256': anchor_by_id[voice['id']]['pcm_sha256'],
                                  'generation_seed': seed, 'attempt': attempt, 'generation_batch_ids': [r['id'] for r in batch]}
                        audit_generated([*done.values(), record], plan, [])
                    except ValueError as exc:
                        failures.append({'id': row['id'], 'reason': str(exc)})
                        path = CACHE / 'failed' / f'{voice["id"]}-{offset:04d}-{attempt}-{row["prompt_id"]}.npy'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            np.save(stream, waveform, allow_pickle=False)
                        continue
                    with manifest.open('a', encoding='utf8') as stream:
                        stream.write(canonical(record) + '\n')
                    done[row['id']] = record
                write_new(attempt_path, {'seed': seed, 'batch_ids': [r['id'] for r in batch],
                          'attempt': attempt, 'seconds': time.perf_counter() - start, 'failures': failures})
                status = {'stage': 'generation', 'completed': len(done), 'planned': len(plan),
                          'voice': voice['id'], 'batch_offset': offset, 'attempt': attempt,
                          'elapsed_seconds': time.perf_counter() - began}
                (CACHE / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                print(json.dumps(status), flush=True)
            if not all(r['id'] in done for r in batch):
                raise ValueError('Generation exhausted frozen attempts; retain evidence and resolve in a child version')
    result = audit_generated(list(done.values()), plan, [])
    result.update(manifest=_relative(manifest), manifest_sha256=file_digest(manifest),
                  anchors=_relative(anchor_receipt), anchors_sha256=file_digest(anchor_receipt),
                  protocol_sha256=file_digest(HERE / 'protocol.json'),
                  seconds=time.perf_counter() - began, gpu=torch.cuda.get_device_name(0),
                  torch=torch.__version__, qwen_tts=importlib.metadata.version('qwen-tts'))
    _verify(protocol['bindings'])
    write_new(CACHE / 'generation-complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'generate'))
    args = parser.parse_args()
    result = freeze() if args.stage == 'freeze' else generate()
    print(json.dumps(result if args.stage == 'generate' else {
        'experiment': result['experiment_id'], 'planned': result['planned'],
        'protocol': _relative(HERE / 'protocol.json')}, indent=2))


if __name__ == '__main__':
    main()
