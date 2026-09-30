"""Local three-family TTS generation with native audio and immutable receipts."""
import argparse
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from einops import rearrange
from study import CACHE, HERE, ROOT, VOICES, file_hash, read_rows, write_json, validate_frozen_study

ANCHOR = 'Hello, I am ready to help. Please speak clearly and tell me what you would like to do today.'
SOURCE_AT_IMPORT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def audit_waveform(waveform, rate):
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 1 or not np.isfinite(x).all() or not .2 <= len(x) / rate <= 20:
        raise ValueError('Invalid waveform shape, values or duration')
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    clipping = float(np.mean(np.abs(x) >= .999))
    if rms < 1e-3 or clipping > .01:
        raise ValueError('Silent or clipped generation')
    return dict(duration_seconds=len(x)/rate, rms=rms, clipped_fraction=clipping)


def prepare_audio(waveform,rate):
    """Audit both retained PCM payloads before creating any permanent file."""
    import soundfile as sf
    from scipy.signal import resample_poly
    audit_waveform(waveform,rate)
    native_buffer=io.BytesIO()
    sf.write(native_buffer,waveform,rate,subtype='PCM_16',format='WAV')
    native_bytes=native_buffer.getvalue()
    # Derive from the retained quantized original, not an unrecorded floating tensor.
    original, _ = sf.read(io.BytesIO(native_bytes), dtype='float32')
    native_metrics=audit_waveform(original,rate)
    divisor = math.gcd(rate, 16000)
    pcm = resample_poly(original, 16000//divisor, rate//divisor).astype(np.float32)
    derived_buffer=io.BytesIO()
    sf.write(derived_buffer,pcm,16000,subtype='PCM_16',format='WAV')
    derived_bytes=derived_buffer.getvalue()
    decoded,_=sf.read(io.BytesIO(derived_bytes),dtype='float32')
    metrics=audit_waveform(decoded,16000)
    return native_bytes,derived_bytes,native_metrics,metrics,hashlib.sha256(decoded.tobytes()).hexdigest()


def persist_audio(root, row, waveform, rate, prepared=None):
    native, path = Path(root)/row['native_audio'], Path(root)/row['audio']
    if native.exists() or path.exists():
        raise FileExistsError('Audio is immutable; preserve and investigate orphan files')
    native_bytes,derived_bytes,native_metrics,metrics,pcm_hash=prepared or prepare_audio(waveform,rate)
    for target,payload in ((native,native_bytes),(path,derived_bytes)):
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
    return {**row, 'synthetic': True, 'text_verified_by_listener': False,
            'accent_verified': False, 'native_sample_rate': rate, 'sample_rate': 16000,
            'channels': 1, 'native_sha256': file_hash(native), 'audio_sha256': file_hash(path),
            'pcm_sha256': pcm_hash,
            'native_metrics': native_metrics, **metrics}


def voice_description(voice):
    return (f'A distinct adult {voice["gender"]} speaking English with a clear '
            f'{voice["accent"]} accent, conversational pace and natural expression. '
            'Clean dry close microphone recording. Speak only the supplied text.')


def dia_text(text):
    """Close the sentence for Dia without adding or changing spoken words."""
    return text if text.rstrip().endswith(('.', '?', '!')) else text+'.'


def ensure_anchors():
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    output = CACHE/'anchors'
    output.mkdir(parents=True, exist_ok=True)
    missing = [v for v in VOICES if not (output/f'{v["id"]}.json').exists()]
    if not missing:
        for v in VOICES:
            receipt = json.loads((output/f'{v["id"]}.json').read_text())
            if file_hash(output/f'{v["id"]}.wav') != receipt['sha256']:
                raise ValueError('Voice anchor changed')
        return
    model_path = ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign'
    model = Qwen3TTSModel.from_pretrained(str(model_path), device_map='cuda:0',
                dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    for voice in missing:
        torch.manual_seed(voice['seed'])
        begin = time.monotonic()
        waves, rate = model.generate_voice_design(text=ANCHOR, language='English',
            instruct=voice_description(voice), max_new_tokens=256, do_sample=True, temperature=.8)
        audit_waveform(waves[0], rate)
        target = output/f'{voice["id"]}.wav'
        if target.exists():
            raise FileExistsError('Orphan anchor needs review')
        sf.write(target, waves[0], rate, subtype='PCM_16')
        write_json(output/f'{voice["id"]}.json', {**voice, 'text': ANCHOR,
            'sha256': file_hash(target), 'model_sha256': file_hash(model_path/'model.safetensors'),
            'seconds': time.monotonic()-begin, 'synthetic': True, 'accent_verified': False})
        print('ANCHOR', voice['id'], round(time.monotonic()-begin, 2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_engine(engine, device='cuda'):
    import torch
    if engine == 'qwen':
        from qwen_tts import Qwen3TTSModel
        folder = ROOT/'.cache/tts/models/Qwen3-TTS-12Hz-1.7B-Base'
        model = Qwen3TTSModel.from_pretrained(str(folder), device_map='cuda:0',
            dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
        prompts = {v['id']: model.create_voice_clone_prompt(ref_audio=str(CACHE/'anchors'/f'{v["id"]}.wav'), ref_text=ANCHOR)[0] for v in VOICES}
        def generate(texts, voice):
            return model.generate_voice_clone(text=texts, language=['English']*len(texts),
                voice_clone_prompt=[prompts[voice['id']]]*len(texts), non_streaming_mode=True,
                max_new_tokens=256, do_sample=True, temperature=.75, top_p=.95)
    elif engine == 'parler':
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        folder = CACHE/'models/parler-tts-large-v1'
        model = ParlerTTSForConditionalGeneration.from_pretrained(str(folder),
            torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
        tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
        def generate(texts, voice):
            description = f'{voice["parler"]} speaks with a {voice["accent"]} accent. The voice is clear, expressive and close, with a moderate speaking speed and very clear audio.'
            a = tokenizer([description]*len(texts), padding=True, return_tensors='pt').to('cuda')
            b = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output = model.generate(input_ids=a.input_ids, attention_mask=a.attention_mask,
                    prompt_input_ids=b.input_ids, prompt_attention_mask=b.attention_mask,
                    max_new_tokens=1400, do_sample=True, temperature=.9,
                    return_dict_in_generate=True)
            generate.last_metadata=[dict(returned_samples=int(n),
                padded_samples=int(x.shape[-1]),trim_basis='model_returned_EOS_length')
                for x,n in zip(output.sequences,output.audios_length)]
            return [x[:int(n)].float().cpu().numpy() for x,n in
                    zip(output.sequences,output.audios_length)], model.config.sampling_rate
    else:
        import soundfile as sf
        from scipy.signal import resample_poly
        from transformers import DiaProcessor, DiaFeatureExtractor, DiaTokenizer, DacModel, DiaForConditionalGeneration
        folder = CACHE/'models/Dia-1.6B-0626'
        codec_path=CACHE/'models/dac_44khz'
        codec=DacModel.from_pretrained(str(codec_path),local_files_only=True).to(device).eval()
        processor=DiaProcessor(DiaFeatureExtractor.from_pretrained(str(folder),local_files_only=True),
            DiaTokenizer.from_pretrained(str(folder),local_files_only=True),codec)
        model = DiaForConditionalGeneration.from_pretrained(str(folder),
            dtype=torch.float32,
            local_files_only=True).to(device).eval()
        anchors = {}
        for v in VOICES:
            audio, rate = sf.read(CACHE/'anchors'/f'{v["id"]}.wav', dtype='float32')
            div = math.gcd(rate, 44100)
            anchors[v['id']] = resample_poly(audio, 44100//div, rate//div).astype('float32')
        def generate(texts, voice):
            begin=time.monotonic()
            inputs = processor(text=[f'[S1] {ANCHOR} [S1] {dia_text(t)}' for t in texts],
                audio=[anchors[voice['id']]]*len(texts), padding=True, return_tensors='pt').to(device)
            length = processor.get_audio_prompt_len(inputs['decoder_attention_mask'])
            encoded=time.monotonic()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=1600, do_sample=True, temperature=1.8)
            generated=time.monotonic()
            ends=[]
            for tokens in outputs:
                eos=torch.where(tokens[:,0]==model.config.eos_token_id)[0]
                if not len(eos) or int(eos[0])>=int(length)+1600-16:
                    raise ValueError('Dia hit the token cap; preserve failure instead of accepting truncated audio')
                ends.append(int(eos[0]))
            waves = processor.batch_decode(outputs, audio_prompt_len=length)
            generate.last_metadata=[dict(prompt_tokens=int(length),eos_position=end,
                dtype='float32',temperature=1.8,codec_device=device,
                encoding_seconds=encoded-begin,generation_seconds=generated-encoded,
                decoding_seconds=time.monotonic()-generated) for end in ends]
            return [np.asarray(x, dtype=np.float32) for x in waves], 44100
    hashes = {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob('*')
              if p.is_file() and p.suffix in ('.safetensors', '.json') and '.cache' not in p.relative_to(folder).parts}
    if engine=='dia':
        hashes.update({'codec/'+p.relative_to(codec_path).as_posix():file_hash(p)
                       for p in codec_path.rglob('*') if p.is_file() and
                       p.suffix in ('.safetensors','.json') and '.cache' not in p.relative_to(codec_path).parts})
    return generate, hashes


def generate_engine(engine, pilot=False, batch_size=4):
    import torch
    from assets import verify_assets
    validate_frozen_study()
    assets_sha256=verify_assets(engine)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the declared generation profile')
    if engine == 'qwen':
        ensure_anchors()
    elif not all((CACHE/'anchors'/f'{v["id"]}.json').exists() for v in VOICES):
        raise RuntimeError('Create hash-bound original synthetic voice anchors first')
    verbs = {r['id']: r for r in read_rows(HERE/'verbatims.jsonl')}
    plan = [r for r in read_rows(HERE/'generation-plan.jsonl') if r['engine'] == engine]
    path = CACHE/f'{engine}-manifest.jsonl'
    existing = {r['recording_id']: r for r in read_rows(path)} if path.exists() else {}
    for row in existing.values():
        if file_hash(CACHE/row['audio']) != row['audio_sha256'] or file_hash(CACHE/row['native_audio'])!=row['native_sha256']:
            raise ValueError('Previously generated audio changed')
    generator, model_hashes = load_engine(engine)
    identity_path=CACHE/f'{engine}-model-identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=model_hashes:
        raise ValueError('TTS model identity changed during generation')
    write_json(identity_path, model_hashes)
    recipe=dict(engine=engine,model_identity_sha256=file_hash(identity_path),batch_size=batch_size,
        voice_groups=VOICES,seed_rule='sha256(engine/voice/verbatim), first record in immutable batch',
        generation={'qwen':dict(temperature=.75,top_p=.95,max_new_tokens=256,mode='voice_clone',max_attempts=3),
                    'parler':dict(temperature=.9,max_new_tokens=1400,trim='returned_EOS_length',max_attempts=3),
                    'dia':dict(temperature=1.8,max_new_tokens=1600,mode='synthetic_anchor',
                               dtype='float32',codec_device='cuda',reject_token_cap=True,max_attempts=20,
                               text_adapter='append_terminal_period_when_missing')}[engine])
    recipe_path=CACHE/f'{engine}-recipe.json'
    normalized=json.loads(json.dumps(recipe))
    if recipe_path.exists() and json.loads(recipe_path.read_text())!=normalized:
        raise ValueError('Generation recipe changed; version the corpus')
    if any(r.get('batch_size')!=batch_size for r in existing.values()):
        raise ValueError('Cannot resume with a different batch size')
    write_json(recipe_path,recipe)
    start = time.monotonic()
    count = 0
    for voice in VOICES:
        records = [r for r in plan if r['voice_group'] == voice['id']]
        for offset in range(0, len(records), batch_size):
            if pilot and offset > 0:
                break
            batch = records[offset:offset+batch_size]
            if all(r['recording_id'] in existing for r in batch):
                continue
            torch.cuda.empty_cache()
            began = time.monotonic()
            try:
                attempts=recipe['generation']['max_attempts']
                for attempt in range(attempts):
                    actual_seed=(batch[0]['seed']+attempt) % 2**32
                    torch.manual_seed(actual_seed)
                    try:
                        waveforms, rate = generator([verbs[r['verbatim_id']]['text'] for r in batch], voice)
                        if len(waveforms) != len(batch):
                            raise ValueError('TTS returned wrong batch count')
                        prepared_audio=[prepare_audio(wave,rate) for wave in waveforms]
                        break
                    except ValueError as exc:
                        with (CACHE/f'{engine}-attempts.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(ids=[r['recording_id'] for r in batch],
                                seed=actual_seed,attempt=attempt+1,error=str(exc),
                                recipe_sha256=file_hash(recipe_path)))+'\n')
                        if attempt+1==attempts:raise
                        print('RETRY',engine,voice['id'],offset,attempt+1,str(exc),flush=True)
                metadata=getattr(generator,'last_metadata',[{} for _ in batch])
                for row, wave, prepared, generation_metadata in zip(batch, waveforms,prepared_audio,metadata):
                    if row['recording_id'] in existing:
                        continue
                    item = persist_audio(CACHE, row, wave, rate,prepared)
                    item.update(batch_seed=actual_seed, generation_attempt=attempt+1,
                        tts_assets_sha256=assets_sha256,
                        batch_ids=[r['recording_id'] for r in batch],
                        batch_size=batch_size,
                        voice_anchor_sha256=None if engine=='parler' else file_hash(CACHE/'anchors'/f'{voice["id"]}.wav'),
                        voice_conditioning={'type':'named_voice_description','name':voice['parler']}
                            if engine=='parler' else {'type':'original_synthetic_audio_anchor'},
                        verbatim_sha256=file_hash(HERE/'verbatims.jsonl'),
                        model_identity_sha256=file_hash(identity_path),generation_metadata=generation_metadata,
                        generation_recipe_sha256=file_hash(recipe_path),generation_source_sha256=SOURCE_AT_IMPORT_SHA256)
                    with path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(item)+'\n'); stream.flush(); os.fsync(stream.fileno())
                    existing[row['recording_id']] = item
                    count += 1
            except Exception as exc:
                with (CACHE/f'{engine}-failures.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'ids':[r['recording_id'] for r in batch],
                        'error':repr(exc),'elapsed_seconds':time.monotonic()-began})+'\n')
                raise
            print('GENERATED', engine, len(existing), '/', len(plan), 'batch_seconds',round(time.monotonic()-began,2), flush=True)
    receipt = dict(engine=engine, generated=len(existing), planned=len(plan), complete=len(existing)==len(plan),
        run_clips=count, run_seconds=time.monotonic()-start, pilot=pilot,
        audio_text_verified=False, accent_verified=False, model_hashes=model_hashes,
        manifest_sha256=file_hash(path) if path.exists() else None)
    write_json(CACHE/f'{engine}-generation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('engine', choices=['qwen','parler','dia'])
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    generate_engine(args.engine, args.pilot, args.batch_size)
