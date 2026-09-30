"""Fixed three-fold, training-speaker-only viability experiment."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .contract import file_hash, identity, read_jsonl, write_json


def speaker_folds(rows):
    if not rows or any(r.get('split') != 'train' for r in rows):
        raise ValueError('CV accepts training speakers only')
    maps = {key: {} for key in ('recording_id', 'session_id', 'utterance_parent_id',
                                'audio_sha256', 'pcm_sha256')}
    for row in rows:
        speaker = row.get('speaker_id')
        if not speaker or speaker in ('unknown', 'pending'):
            raise ValueError('Missing speaker identity')
        for key, seen in maps.items():
            value = row.get(key)
            if not value or value in ('unknown', 'pending'):
                raise ValueError(f'Missing identity: {key}')
            if value in seen and (key != 'session_id' or seen[value] != speaker):
                raise ValueError(f'Repeated identity or leakage: {key}')
            seen[value] = speaker
    speakers = sorted({r['speaker_id'] for r in rows})
    if len(speakers) < 12:
        raise ValueError('CV needs the declared 12 training speakers')
    np.random.default_rng(53).shuffle(speakers)
    return [{'fold': i, 'fit_speakers': sorted(s for j, s in enumerate(speakers) if j % 3 != i),
             'held_speakers': sorted(speakers[i::3])} for i in range(3)]


def fold_normalization(fit_rows, arrays):
    from vcm.frontend import SPEC
    total, squares, frames = np.zeros(40), np.zeros(40), 0
    for row in fit_rows:
        x = arrays[row['recording_id']].astype(np.float64)
        total += x.sum(axis=1); squares += (x * x).sum(axis=1)
        frames += x.shape[1]
    if not frames:
        raise ValueError('No fit-fold frames')
    mean = total / frames
    std = np.sqrt(np.maximum(squares / frames - mean ** 2, 1e-4))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'fit_split': 'train',
            'fit_speakers': sorted({r['speaker_id'] for r in fit_rows}),
            'fit_recording_ids': sorted(r['recording_id'] for r in fit_rows),
            'frames': frames, 'frontend_spec': SPEC}


def cv_passed(metrics):
    return len(metrics) == 3 and all(
        m['macro_exact'] >= .8 and m['worst_use_case'] >= .7
        and m['observed_phrases'] == m['required_phrases'] == 570
        and m['observed_use_cases'] == 10 for m in metrics)


def run_cv(config_path, contract, lexicon, output):
    from .training import fit
    config = json.loads(Path(config_path).read_text())
    if config.get('hard_example_multiplier') != 1. or config.get('seeds') != [53, 71]:
        raise ValueError('CV uses only the primary recipe')
    prepared = Path(config['prepared'])
    rows = read_jsonl(prepared / 'train.jsonl')
    folds = speaker_folds(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'folds.json', {'seed': 53, 'folds': folds,
        'train_manifest_sha256': file_hash(prepared / 'train.jsonl'),
        'freeze_sha256': file_hash(prepared / 'freeze.json'),
        'contract_sha256': contract.sha256, 'config_sha256': file_hash(config_path)})
    results = []
    for fold in folds:
        result = fit(config_path, contract, lexicon, output / f'fold-{fold["fold"]}', fold=fold)
        if result.get('status') != 'CV_FOLD_COMPLETE':
            write_json(output / 'stopped.json', {'fold': fold, 'result': result})
            return {'status': 'BLOCKED_CV_INCOMPLETE', 'qualified': False}
        results.append(result['final_epoch'])
    result = {'status': 'CV_PASSED' if cv_passed(results) else 'BLOCKED_CV',
              'folds': results, 'qualified': False,
              'folds_sha256': file_hash(output / 'folds.json'),
              'freeze_sha256': file_hash(prepared / 'freeze.json'),
              'contract_sha256': contract.sha256,
              'selection': 'final scheduled epoch; no held-fold tuning'}
    write_json(output / 'cv.json', result)
    return result
