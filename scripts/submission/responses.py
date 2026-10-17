"""Offline release preparation for fixed synthetic responses; excluded from inference ZIP."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

from scripts.submission.common import ROOT, read, save, sha
from scripts.submission.response_runtime import RESPONSE_TEXTS, ensure_responses, response_id, validate_manifest

def render(*, gpu=2):
    """Offline release preparation only; imports research TTS packages on this path."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import importlib.metadata
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    if not torch.cuda.is_available():
        raise RuntimeError('Offline rendering requires the explicitly selected CUDA GPU')
    cache = ROOT/'.cache/submission/responses'
    cache.mkdir(parents=True, exist_ok=True)
    voice = next(v for v in read(ROOT/'experiments/speaker-synthetic-completion-v1/voices.json')
                 if v['id'] == 'train-ph')
    model_root = ROOT/'.cache/tts/models'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    began = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        str(model_root/'Qwen3-TTS-12Hz-1.7B-VoiceDesign'), device_map='cuda:0',
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    # Every clip derives from the same wholly synthetic anchor, not a real voice.
    anchor_text = 'Hello. I am the local smart speaker demonstration voice. I acknowledge recognized requests.'
    anchor = cache/'synthetic-voice-anchor.wav'
    if not anchor.exists():
        torch.manual_seed(voice['design_seed'])
        waves, rate = model.generate_voice_design(text=anchor_text, instruct=voice['prompt'],
            language='English', max_new_tokens=512, do_sample=True, temperature=.8)
        sf.write(anchor, waves[0], rate, subtype='PCM_16')
    del model; gc.collect(); torch.cuda.empty_cache()
    model = Qwen3TTSModel.from_pretrained(
        str(model_root/'Qwen3-TTS-12Hz-1.7B-Base'), device_map='cuda:0',
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True)
    prompt = model.create_voice_clone_prompt(ref_audio=str(anchor), ref_text=anchor_text)
    rows = []
    for index, (key, text) in enumerate(RESPONSE_TEXTS.items()):
        path = cache/'audio'/(key+'.wav'); path.parent.mkdir(parents=True, exist_ok=True)
        seed = 6101700 + index
        if not path.exists():
            torch.manual_seed(seed)
            waves, rate = model.generate_voice_clone(text=text, language='English', voice_clone_prompt=prompt,
                max_new_tokens=512, do_sample=True, temperature=.7)
            value = np.asarray(waves[0], dtype='float32')
            if value.ndim != 1 or not np.isfinite(value).all() or len(value) == 0:
                raise ValueError('TTS produced invalid waveform')
            sf.write(path, np.clip(value,-1,1), rate, subtype='PCM_16')
        with sf.SoundFile(path) as wave:
            if wave.channels != 1 or not 0 < wave.frames/wave.samplerate < 20:
                raise ValueError('Invalid recorded response duration/channels')
            row = {'id':key, 'text':text, 'path':'audio/'+path.name, 'sha256':sha(path),
                   'bytes':path.stat().st_size, 'sample_rate':wave.samplerate, 'samples':wave.frames,
                   'duration_seconds':wave.frames/wave.samplerate, 'seed':seed}
        rows.append(row)
        print('RESPONSE_READY', key, row['duration_seconds'], row['sha256'], flush=True)
    manifest = {'format':'smart-speaker-fixed-responses-v1',
        'claim':'Recognition acknowledgments only; no device action or runtime synthesis',
        'identity_source':'original_synthetic_voice_design', 'human_voice_clone':False,
        'listener_verified':False, 'voice':voice, 'anchor_sha256':sha(anchor),
        'generation':{'qwen_tts':importlib.metadata.version('qwen-tts'), 'torch':torch.__version__,
                      'models':['Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign','Qwen/Qwen3-TTS-12Hz-1.7B-Base'],
                      'device':torch.cuda.get_device_name(0), 'physical_gpu':gpu, 'seconds':time.perf_counter()-began,
                      'sampling':{'do_sample':True,'temperature':.7,'max_new_tokens':512}},
        'hub':{'repo_id':'justinruaya123/ai231-smart-speaker','repo_type':'dataset','branch':'inference',
               'revision':None,'subdirectory':'responses'}, 'responses':rows}
    receipt = cache/'rendered-manifest.json'
    if receipt.exists():
        if read(receipt)['responses'] != rows:
            raise ValueError('Preserve existing response render receipt')
    else:
        save(receipt, manifest)
    print('RESPONSE_RENDER_COMPLETE', str(receipt), flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('render','verify','download'))
    p.add_argument('--gpu',type=int,default=2)
    args=p.parse_args()
    if args.operation=='render':
        render(gpu=args.gpu)
    else:
        assets=ensure_responses()
        print(json.dumps({'responses':len(assets),'verified':True}))


if __name__=='__main__':
    main()
