"""Generate the separately frozen Qwen number stress set; never train on it."""
import json
import time
import numpy as np
import soundfile as sf
import torch
from .complete_data import ROOT, EXP, CACHE, read_rows, digest, save
from .audio import resample_audio
from .synthetic_tts import ANCHOR


def generate():
    from qwen_tts import Qwen3TTSModel
    plan_path = EXP / 'number-stress-plan-v2.json'
    plan = json.loads(plan_path.read_text())
    output = EXP / 'number-stress-manifest-v2.jsonl'
    if output.exists():
        raise ValueError('Preserve existing generation; use an explicit resumable run')
    torch.set_num_threads(4)
    model_root = ROOT / '.cache/tts/models/Qwen3-TTS-12Hz-1.7B-Base'
    model = Qwen3TTSModel.from_pretrained(str(model_root), device_map='cuda:0', dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True)
    prompts = {voice: model.create_voice_clone_prompt(ref_audio=str(ROOT / 'data/tts/voices' / (voice + '.wav')), ref_text=ANCHOR)[0]
        for voice in ['designed-10', 'designed-11']}
    started = time.monotonic()
    print('QWEN_STRESS_MODEL_LOADED', flush=True)
    with output.open('w', encoding='utf8') as stream:
        for offset in range(0, len(plan['rows']), 16):
            batch = plan['rows'][offset:offset + 16]
            torch.cuda.empty_cache(); torch.manual_seed(4701 + offset)
            tick = time.monotonic()
            waveforms, rate = model.generate_voice_clone(text=[r['transcript'] for r in batch], language=['English'] * len(batch),
                voice_clone_prompt=[prompts[r['voice']] for r in batch], non_streaming_mode=True,
                max_new_tokens=160, do_sample=True, temperature=.75, top_p=.95)
            for row, waveform in zip(batch, waveforms):
                pcm = resample_audio(waveform, rate)
                if not np.isfinite(pcm).all() or not 1600 <= len(pcm) <= 160000:
                    raise ValueError('Unusable generated recording: ' + row['id'])
                path = ROOT / row['audio']; path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raise ValueError('Refusing to replace stress audio')
                sf.write(path, pcm, 16000, subtype='PCM_16')
                record = {**row, 'audio_sha256': digest(path), 'duration': len(pcm) / 16000,
                    'sample_rate': 16000, 'channels': 1, 'seed': 4701 + offset,
                    'anchor_sha256': digest(ROOT / 'data/tts/voices' / (row['voice'] + '.wav'))}
                stream.write(json.dumps(record) + '\n'); stream.flush()
            print(f'QWEN_STRESS {offset + len(batch)}/{len(plan["rows"])} batch_seconds={time.monotonic()-tick:.1f}', flush=True)
    save(EXP / 'number-stress-generation-v2.json', {'complete': True, 'rows': len(plan['rows']),
        'manifest_sha256': digest(output), 'plan_sha256': digest(plan_path), 'seconds': time.monotonic() - started,
        'tts_model': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base', 'local_only': True, 'human_adjudicated': False})
