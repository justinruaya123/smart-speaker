"""Audit isolated artifacts and evaluate the frozen models without fitting."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import einops
import numpy as np
import scipy
import soundfile as sf
import torch
from common import ROOT, read_jsonl, write_json, write_jsonl, extract_number
from model import AcousticCTC
from numeric_experiment import predict, metrics, VOCAB
from catalog import accept_fields


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    before = json.loads((ROOT/'evidence/pipeline_before.json').read_text())
    current = {}
    for base, dirs, files in os.walk(ROOT.parent):
        dirs[:] = [d for d in dirs if d not in {'STT', '.git', '.venv', '.cache', '__pycache__'}]
        for name in files:
            path = Path(base)/name
            rel = path.relative_to(ROOT.parent).as_posix()
            st = path.stat()
            record = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
            if not rel.startswith('data/audio/'):
                record['sha256'] = sha(path)
            current[rel] = record
    changed = [k for k in before.keys() & current.keys() if before[k] != current[k]]
    isolation = {'before_files': len(before), 'after_files': len(current),
                 'hashed_files': sum('sha256' in r for r in before.values()),
                 'changed': changed, 'added_outside_STT': sorted(current.keys()-before.keys()),
                 'removed_outside_STT': sorted(before.keys()-current.keys()),
                 'excluded_directories': ['.git', '.venv', '.cache', '__pycache__', 'STT'],
                 'audio_check': 'Existing data/audio files compared by size and modification time; all other included files also hashed.'}
    write_json(ROOT/'evidence/pipeline_isolation.json', isolation)

    source_dir = ROOT/'data/source_receipts_existing'
    source_dir.mkdir(parents=True, exist_ok=True)
    for path in (ROOT.parent/'data/sources').glob('*.json'):
        shutil.copyfile(path, source_dir/path.name)

    selection = json.loads((ROOT/'evidence/mswc_selected.json').read_text())
    downloaded = []
    invalid = []
    for row in selection.values():
        path = ROOT/row['audio']
        if not path.exists():
            continue
        try:
            info = sf.info(path)
            if info.frames <= 0:
                raise ValueError('empty audio')
            waveform, rate = sf.read(path, dtype='float32')
            if len(waveform) != info.frames or not np.isfinite(waveform).all():
                raise ValueError('invalid or incomplete decoded audio')
            downloaded.append(dict(row, audio_sha256=sha(path), bytes=path.stat().st_size,
                                   duration_seconds=info.duration))
        except Exception as exc:
            invalid.append({'path': row['audio'], 'error': str(exc)})
    write_jsonl(ROOT/'data/mswc_downloaded_inventory.jsonl', downloaded)
    receipts = []
    for path in (ROOT/'data/raw').rglob('*.source.json'):
        receipt = json.loads(path.read_text())
        target = ROOT/receipt['path']
        receipt['exists'] = target.exists()
        receipt['verified_sha256_now'] = target.exists() and sha(target) == receipt['sha256']
        receipts.append(receipt)
    partials = [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size}
                for p in (ROOT/'data/raw').rglob('*.part')]
    acquisition = {'audited_utc': datetime.now(timezone.utc).isoformat(),
                   'download_processes': 'Stopped after repeated timeout/partial-transfer failures; completed files retained.',
                   'fresh_mlend_files': len(read_jsonl(ROOT/'data/mlend.jsonl')),
                   'fresh_fsdd_files': len(list((ROOT/'data/audio/fsdd').rglob('*.wav'))),
                   'mswc_completed_decodable_files': len(downloaded),
                   'mswc_completed_word_counts': dict(Counter(r['text'] for r in downloaded)),
                   'mswc_invalid_files': invalid,
                   'completed_download_receipts': receipts, 'partial_downloads_not_used': partials,
                   'partial_bytes': sum(r['bytes'] for r in partials)}
    write_json(ROOT/'evidence/acquisition_final.json', acquisition)

    torch.set_num_threads(4)
    parity = {}
    for name, checkpoint_name, size in [('scratch_ctc', 'commands_best.pt', None),
                                        ('numeric_ctc', 'compositions_best.pt', len(VOCAB))]:
        checkpoint = torch.load(ROOT/'runs'/name/checkpoint_name, map_location='cpu', weights_only=False)
        manifest_name = 'manifest.jsonl' if size is None else 'numeric_manifest.jsonl'
        assert checkpoint['config']['manifest_sha256'] == sha(ROOT/'data'/manifest_name)
        model = AcousticCTC() if size is None else AcousticCTC(size)
        model.load_state_dict(checkpoint['state'])
        model.eval()
        torch.manual_seed(919)
        single = torch.randn(1, 81, 80)
        batch = torch.zeros(2, 134, 80)
        batch[0, :81] = single[0]
        batch[1] = torch.randn(134, 80)
        with torch.inference_mode():
            alone, lengths = model(single, torch.tensor([81]))
            together, _ = model(batch, torch.tensor([81, 134]))
        diff = float((alone[0]-together[0, :int(lengths[0])]).abs().max())
        assert diff < 1e-4, (name, diff)
        parity[name] = diff

    rows = read_jsonl(ROOT/'data/manifest.jsonl')
    volume = [r for r in rows if r['split']=='test' and 'volume' in (r.get('intent') or '').lower()
              and extract_number(r['text']) is not None]
    items = [{'id': r['id'], 'x': torch.from_numpy(np.load(ROOT/r['feature']).astype(np.float32)),
              'target': '<unknown>', 'reference': r['text'], 'value': extract_number(r['text']),
              'source': r['source'], 'kind': 'full_natural_volume_command'} for r in volume]
    volume_predictions = predict(model, items, 'cpu')
    write_jsonl(ROOT/'runs/numeric_ctc/natural_volume_predictions.jsonl', volume_predictions)

    musical = [r for r in read_jsonl(ROOT/'runs/scratch_ctc/catalog_predictions.jsonl') if r['split']=='test']
    fixed = set(map(tuple, json.loads((ROOT/'catalogs/fixed.json').read_text())))
    dynamic = set(map(tuple, json.loads((ROOT/'catalogs/dynamic_fixture.json').read_text())))
    added = dynamic-fixed
    ev = json.loads((ROOT/'runs/scratch_ctc/evaluation.json').read_text())
    assert ev['checkpoint_sha256'] == sha(ROOT/'runs/scratch_ctc/commands_best.pt')
    new_results = {}
    for mode in ['off', 'fixed', 'dynamic']:
        total = correct = 0
        for row in musical:
            refs = set(row['references'].items()) & added
            pred = row['raw_fields'] if mode == 'off' else accept_fields(row[mode], ev['catalog_threshold'], ev['catalog_margin'])
            total += len(refs)
            correct += len(refs & set(pred.items()))
        new_results[mode] = {'new_entity_occurrences': total, 'correct': correct}
    nmanifest = read_jsonl(ROOT/'data/numeric_manifest.jsonl')
    validation = {'trained_checkpoint_batch_parity_max_abs_logit_error': parity,
                  'finite_word_model_on_full_natural_volume_commands': metrics(volume_predictions),
                  'new_music_entity_occurrences_all_request_categories': new_results,
                  'numeric_mswc_words_used': dict(Counter(r['text'] for r in nmanifest if r['source']=='MSWC-English')),
                  'frozen_unique_human_audio_across_both_experiments': len({r['audio'] for r in rows+nmanifest}),
                  'frozen_manifest_hashes': {n:sha(ROOT/'data'/n) for n in ['manifest.jsonl','numeric_manifest.jsonl']},
                  'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                                  'numpy': np.__version__, 'scipy': scipy.__version__,
                                  'soundfile': sf.__version__, 'einops': einops.__version__}}
    write_json(ROOT/'evidence/final_validation.json', validation)
    print(json.dumps({'isolation': isolation,
                      'acquisition': {k:v for k,v in acquisition.items() if k not in {'completed_download_receipts','partial_downloads_not_used'}},
                      'validation': validation}, indent=2))


if __name__ == '__main__':
    main()
