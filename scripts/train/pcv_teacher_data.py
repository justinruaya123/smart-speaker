"""Pinned offline WavLM features from the supplied HF recording pool only."""
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from einops import rearrange, reduce
from scripts.train.pcv_public_data import sha, save, read_rows
from scripts.train.pcv_three_engine import immutable
from scripts.train.pcv_updated_data import HERE as PARENT, CACHE as PARENT_CACHE, REVISION
from scripts.train.pcv_updated_training import development_subset, validate_training_rows
from scripts.test.pcv_public_qualification import verify_bindings

HERE = ROOT/'experiments/pcv-updated-distill-v1'
CACHE = ROOT/'.cache/pcv-updated-distill-v1'
MODEL = 'microsoft/wavlm-base-plus'
MODEL_REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
WEIGHT_SHA = '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'


def teacher_frames(samples):
    return max(0, (samples-400)//320 + 1)


def student_frames(samples):
    return max(0, (samples-400)//160 + 1)


def check_rows(rows, fitting=False, human_only=False):
    if fitting:
        validate_training_rows(rows, human_only=human_only)
    for row in rows:
        path = (ROOT/row['parquet']).resolve()
        if not path.is_relative_to((PARENT_CACHE/'source/data').resolve()):
            raise ValueError('Teacher may use only the supplied dataset recordings')
        if row['split'] not in ('train', 'dev'):
            raise ValueError('This teacher probe cannot access calibration or test audio')


def verified_pcm(row, payload):
    import soundfile as sf
    if hashlib.sha256(payload).hexdigest() != row['audio_sha256']:
        raise ValueError('WAV identity changed: '+row['id'])
    pcm, rate = sf.read(io.BytesIO(payload), dtype='float32')
    if (rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples'] or
            hashlib.sha256(pcm.tobytes()).hexdigest() != row['pcm_sha256']):
        raise ValueError('PCM identity changed: '+row['id'])
    return pcm


def rows_for_probe():
    train = [r for r in read_rows(PARENT_CACHE/'prepared/train.jsonl')
             if r['training_eligible'] and r['reference_status'] != 'auxiliary']
    dev = development_subset(read_rows(PARENT_CACHE/'prepared/dev.jsonl'))
    check_rows(train, fitting=True)
    check_rows(dev)
    return train, dev


def freeze_acquisition():
    data = json.loads((PARENT/'data-freeze.json').read_text())
    verify_bindings(ROOT, data['bindings'])
    verify_bindings(ROOT, {v['path']: v['sha256'] for v in data['manifests'].values()})
    path = HERE/'teacher-source-protocol.json'
    if path.exists():
        frozen = json.loads(path.read_text()); verify_bindings(ROOT, frozen['bindings'])
        return frozen
    files = ['scripts/train/pcv_teacher_data.py', 'scripts/models/pcv_teacher.py',
             'scripts/test/test_pcv_teacher.py', 'scripts/train/pcv_updated_training.py',
             'experiments/pcv-updated-voice-v1/data-freeze.json']
    train, dev = rows_for_probe()
    frozen = {'experiment': 'pcv-updated-distill-v1', 'requirements': ['ME-02','ME-03','ME-04','ME-06','ME-08'],
        'owner': 'current PCV finalization task',
        'hypothesis': 'Offline pretrained audio features improve finite complete-command development accuracy before student distillation.',
        'authorization': 'User explicitly allowed a pretrained offline audio teacher on 2026-10-02; all downstream fitting/distillation audio stays in the supplied HF dataset.',
        'dataset': 'airimonda/ai231-me2-voice-commands', 'dataset_revision': REVISION,
        'model': MODEL, 'model_revision': MODEL_REVISION, 'weight_sha256': WEIGHT_SHA,
        'pretraining_provenance': 'External pretraining is the explicit exception. Pretraining speaker or recording overlap with the supplied corpus is not audited.',
        'source_only': True, 'external_recording_fallback_used': False,
        'feature_protocol': {'waveform': 'original mono 16kHz; publisher sample values, no normalization as pinned preprocessor specifies',
            'encoder': 'frozen WavLMModel in eval mode; no tokenizer or transcript decoder',
            'layers': [6,9,12], 'combination': 'arithmetic mean', 'stored_dtype': 'float16',
            'computation': 'float32 CUDA, no TF32, one complete utterance per forward, no batch padding',
            'frames': '400-sample receptive field, 320-sample stride', 'augmentation': 'none'},
        'train_ids': [r['id'] for r in train], 'development_ids': [r['id'] for r in dev],
        'human_first': True, 'test_access': 'none in teacher probe; prior v1/v3 aggregate benchmark scores are already known',
        'runtime': False, 'installation_authorized': False,
        'bindings': {p: sha(ROOT/p) for p in files}}
    immutable(path, frozen)
    return frozen


def acquire():
    import requests
    protocol = freeze_acquisition()
    destination = CACHE/'teacher'; destination.mkdir(parents=True, exist_ok=True)
    api = f'https://huggingface.co/api/models/{MODEL}/revision/{MODEL_REVISION}?blobs=true'
    response = requests.get(api, timeout=40); response.raise_for_status(); metadata = response.json()
    if metadata['sha'] != MODEL_REVISION: raise ValueError('Teacher revision changed')
    immutable(HERE/'teacher-publisher-metadata.json', metadata)
    wanted = ['README.md','config.json','preprocessor_config.json','pytorch_model.bin']
    receipts = []
    for name in wanted:
        info = next(s for s in metadata['siblings'] if s['rfilename'] == name)
        path = destination/name
        expected = info.get('lfs', {}).get('sha256')
        if name == 'pytorch_model.bin' and expected != WEIGHT_SHA:
            raise ValueError('Teacher weight identity differs from pinned protocol')
        if not path.exists():
            url = f'https://huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/{name}'
            part = path.with_suffix(path.suffix+'.part')
            with requests.get(url, stream=True, timeout=(30, 60)) as download:
                download.raise_for_status()
                with part.open('wb') as stream:
                    for chunk in download.iter_content(1024*1024): stream.write(chunk)
            if part.stat().st_size != info['size'] or (expected and sha(part) != expected):
                raise ValueError('Teacher download integrity failed: '+name)
            part.rename(path)
        if path.stat().st_size != info['size'] or (expected and sha(path) != expected):
            raise ValueError('Existing teacher asset does not match: '+name)
        receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size})
        print('TEACHER_ASSET', name, path.stat().st_size, flush=True)
    preprocessor = json.loads((destination/'preprocessor_config.json').read_text())
    if preprocessor['do_normalize'] or preprocessor['sampling_rate'] != 16000:
        raise ValueError('Pinned teacher frontend assumption failed')
    immutable(HERE/'teacher-acquisition.json', {'protocol_sha256': sha(HERE/'teacher-source-protocol.json'),
        'model_revision': MODEL_REVISION, 'files': receipts,
        'license': 'Model card points to Microsoft UNILM license; no release license inferred from dataset. Teacher stays offline and unshipped.'})
    verify_bindings(ROOT, protocol['bindings'])


def extract():
    import pyarrow.parquet as pq
    import torch
    from transformers import WavLMModel
    protocol = freeze_acquisition()
    receipt = json.loads((HERE/'teacher-acquisition.json').read_text())
    verify_bindings(ROOT, {r['path']: r['sha256'] for r in receipt['files']})
    acquisition = json.loads((PARENT/'acquisition-complete.json').read_text())
    train, dev = rows_for_probe(); rows = train+dev
    if [r['id'] for r in train] != protocol['train_ids'] or [r['id'] for r in dev] != protocol['development_ids']:
        raise ValueError('Teacher feature membership changed')
    wanted = defaultdict(dict)
    for row in rows: wanted[row['parquet']][row['parquet_row']] = row
    for item in acquisition['files']:
        if item['path'] in wanted and sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('Source parquet changed')
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = WavLMModel.from_pretrained(str(CACHE/'teacher'), local_files_only=True).cuda().eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    output = CACHE/'features'; output.mkdir(parents=True, exist_ok=True)
    index_path = CACHE/'feature-index.jsonl'
    existing = {r['id']: r for r in read_rows(index_path)} if index_path.exists() else {}
    completed = []; started = time.perf_counter()
    with index_path.open('a', encoding='utf-8') as index:
        for parquet, selected in wanted.items():
            position = 0
            for batch in pq.ParquetFile(ROOT/parquet).iter_batches(batch_size=64, columns=['audio']):
                for value in batch.to_pylist():
                    row = selected.get(position); position += 1
                    if row is None: continue
                    old = existing.get(row['id'])
                    if old:
                        if sha(ROOT/old['path']) != old['sha256']: raise ValueError('Teacher features changed')
                        completed.append(old); continue
                    pcm = verified_pcm(row, value['audio']['bytes'])
                    with torch.inference_mode():
                        result = model(rearrange(torch.from_numpy(pcm).cuda(), 't -> 1 t'), output_hidden_states=True)
                        combined = reduce([result.hidden_states[i] for i in (6,9,12)], 'layer b t c -> b t c', 'mean')
                        features = rearrange(combined, '1 t c -> c t').cpu().numpy().astype('float16')
                    if features.shape != (768, teacher_frames(len(pcm))) or not np.isfinite(features).all():
                        raise ValueError('Invalid teacher feature output: '+row['id'])
                    path = output/(row['pcm_sha256']+'.npy')
                    np.save(path, features, allow_pickle=False)
                    item = {'id': row['id'], 'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                            'pcm_sha256': row['pcm_sha256'], 'frames': features.shape[1], 'split': row['split']}
                    index.write(json.dumps(item)+'\n'); index.flush(); completed.append(item)
                    if len(completed)%100 == 0:
                        status = {'completed': len(completed), 'total': len(rows), 'seconds': time.perf_counter()-started}
                        save(CACHE/'feature-progress.json', status); print('TEACHER_FEATURES', json.dumps(status), flush=True)
    if len(completed) != len(rows): raise ValueError('Missing requested teacher features')
    immutable(HERE/'teacher-feature-freeze.json', {'index': str(index_path.relative_to(ROOT)), 'sha256': sha(index_path),
        'rows': len(completed), 'encoder_parameters': sum(p.numel() for p in model.parameters()),
        'teacher_acquisition_sha256': sha(HERE/'teacher-acquisition.json'), 'seconds': time.perf_counter()-started,
        'test_inference_accessed': False, 'source_protocol_sha256': sha(HERE/'teacher-source-protocol.json')})
    print('TEACHER_FEATURES_COMPLETE', len(completed), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'acquire': acquire()
    elif sys.argv[1] == 'extract': extract()
    else: raise SystemExit('Use acquire or extract')
