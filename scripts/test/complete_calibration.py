"""E15: frozen acoustic weights; larger independent policy calibration."""
from pathlib import Path
import copy
import json
import shutil
import sys
import time
import numpy as np
import torch
from einops import pack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources
register_sources()
from tinyvcm.complete_data import DATA, EXP, CACHE, read_rows, save, digest, verify_frozen
from tinyvcm.complete_training import score_development
from tinyvcm.inference import Recognizer
from tinyvcm.audio import load_audio, features

EXPERIMENT = ROOT / 'experiments/vcm-complete-v3'


def fit():
    verify_frozen()
    protocol = json.loads((EXPERIMENT / 'protocol.json').read_text())
    source = CACHE / 'human_keywords'
    assert digest(source / 'vcm.fp32.onnx') == protocol['fixed_graph_sha256']
    if (EXPERIMENT / 'selection-before-test.json').exists():
        raise ValueError('This calibration experiment is already locked')
    all_rows = read_rows(DATA / 'manifest.jsonl')
    by_id = {r['id']: r for r in all_rows}
    catalog = json.loads((EXP / 'custom-catalog-v3.json').read_text())['catalog']
    enrollment_pairs = {by_id[id]['paired_group'] for e in catalog for id in e['enrollment_ids']}
    dev = [r for r in all_rows if r['kind'] == 'human' and r['split'] == 'dev']
    calibration = [{**r, 'fold': 'policy_fit'} for r in all_rows if r['kind'] == 'human' and r['split'] == 'calibration'
                   and r['paired_group'] not in enrollment_pairs]
    recognizer = Recognizer(source, threads=2)
    vocabulary = recognizer.metadata['acoustic_slots']['vocabulary']
    outputs = [torch.empty(len(calibration), n) for n in [len(recognizer.labels), *map(len, vocabulary.values())]]
    names = ['logits', *['slot_' + name for name in vocabulary]]
    last = time.monotonic()
    for i, row in enumerate(calibration):
        pcm, _ = load_audio(ROOT / row['audio'])
        x = recognizer.normalize(features(pcm))
        values = recognizer.vcm.run(names, {'features': x, 'lengths': np.array([x.shape[-1]], dtype=np.int64)})
        for target, value in zip(outputs, values):
            target[i] = torch.from_numpy(value[0])
        if time.monotonic() - last > 20:
            print(f'V3_CALIBRATION {i+1}/{len(calibration)}', flush=True); last = time.monotonic()
    dev_outputs = torch.load(source / 'development-logits.pt', weights_only=True)
    reports = {}
    for arm in ['v3_original', 'v3_calibration', 'v3_pooled']:
        folder = CACHE / arm
        folder.mkdir(exist_ok=False)
        shutil.copy2(source / 'vcm.fp32.onnx', folder / 'vcm.fp32.onnx')
        shutil.copy2(source / 'vcm.pt', folder / 'vcm.pt')
        metadata = copy.deepcopy(recognizer.metadata)
        if arm == 'v3_original':
            result, intent, slots = score_development(dev, dev_outputs, metadata, fit_policy=False)
            fit_ids = [r['id'] for r in dev if r['fold'] == 'policy_fit']
        else:
            indices = [i for i, r in enumerate(dev) if r['fold'] == 'selection' or
                       arm == 'v3_pooled' and r['paired_group'] not in enrollment_pairs]
            rows = [dev[i] for i in indices] + calibration
            combined = tuple(pack([d[indices], c], '* classes')[0] for d, c in zip(dev_outputs, outputs))
            result, intent, slots = score_development(rows, combined, metadata)
            fit_ids = [r['id'] for r in rows if r['fold'] == 'policy_fit']
        metadata['routing']['variants']['fp32'] = intent
        metadata['acoustic_slots']['variants']['fp32'] = slots
        metadata['installation'] = {'status': 'research_candidate_not_installed', 'experiment': 'vcm-complete-v3', 'arm': arm}
        metadata['calibration'] = {'partition': arm, 'fit_recordings': len(fit_ids), 'acoustic_weights_unchanged': True}
        save(folder / 'bundle.json', metadata)
        result.update(arm=arm, calibration_recordings=len(fit_ids), graph_sha256=digest(folder / 'vcm.fp32.onnx'),
            bundle_sha256=digest(folder / 'bundle.json'), test_used=False)
        save(EXPERIMENT / (arm + '-development.json'), result)
        save(folder / 'calibration-ids.json', fit_ids)
        reports[arm] = result
    selected = max(reports, key=lambda a: tuple(reports[a]['rank']))
    files = [EXPERIMENT / 'protocol.json', Path(__file__), DATA / 'frozen.json', ROOT / 'scripts/test/complete_commands.py']
    for arm in reports:
        files.extend(CACHE / arm / name for name in ['bundle.json', 'vcm.fp32.onnx'])
    lock = {'selected_arm': selected, 'development_ranks': {a: r['rank'] for a, r in reports.items()},
        'files': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files},
        'created_unix': time.time(), 'test_used_to_select': False, 'historical_test_already_examined': True,
        'encoder_fitting': False, 'new_numeric_head_evidence': False}
    save(EXPERIMENT / 'selection-before-test.json', lock)
    print(json.dumps(lock['development_ranks']), flush=True)


def evaluate():
    from tinyvcm.complete_eval import fixed_test, number_test
    selection = json.loads((EXPERIMENT / 'selection-before-test.json').read_text())
    for path, sha in selection['files'].items():
        if digest(ROOT / path) != sha:
            raise ValueError('Calibration lock changed: ' + path)
    report = fixed_test(selection['selected_arm'])
    numeric = number_test(selection['selected_arm'])
    save(EXPERIMENT / 'fixed-test.json', report)
    save(EXPERIMENT / 'numbers.json', numeric)
    save(EXPERIMENT / 'results.json', {'selected_arm': selection['selected_arm'],
        'selection_sha256': digest(EXPERIMENT / 'selection-before-test.json'), 'fixed': report,
        'numbers': numeric['groups'], 'installed': False, 'test_is_historical_regression': True})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'evaluate'])
    args = parser.parse_args()
    fit() if args.stage == 'fit' else evaluate()
