"""E16 fixed weight averages: unchanged architecture, development-only selection."""
from pathlib import Path
import copy
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources
register_sources()
from tinyvcm.complete_data import EXP, DATA, CACHE, digest, save, verify_frozen
from tinyvcm.complete_training import finalize

EXPERIMENT = ROOT / 'experiments/vcm-complete-v4'


def prepare():
    verify_frozen()
    protocol = json.loads((EXPERIMENT / 'protocol.json').read_text())
    checkpoints = {}
    for arm, sha in protocol['input_checkpoints'].items():
        path = ROOT / 'artifacts/vcm.pt' if arm == 'installed_baseline' else CACHE / arm / 'vcm.pt'
        assert digest(path) == sha
        checkpoints[arm] = torch.load(path, weights_only=False, map_location='cpu')
    reference = checkpoints['human_keywords']
    for ck in checkpoints.values():
        assert ck['config'] == reference['config'] and ck['normalization'] == reference['normalization']
        assert ck['model'].keys() == reference['model'].keys()
    reports = {}
    for arm, coefficients in protocol['arms'].items():
        assert abs(sum(coefficients.values()) - 1) < 1e-8
        folder = CACHE / arm
        folder.mkdir(exist_ok=False)
        ck = copy.deepcopy(reference)
        for name, tensor in reference['model'].items():
            assert all(checkpoints[a]['model'][name].shape == tensor.shape for a in coefficients)
            if tensor.is_floating_point():
                ck['model'][name] = sum(checkpoints[a]['model'][name].float() * weight for a, weight in coefficients.items())
            else:
                assert all(torch.equal(checkpoints[a]['model'][name], tensor) for a in coefficients)
                ck['model'][name] = tensor.clone()
        ck.update(training_mode='fixed_weight_average', epoch=0, coefficients=coefficients)
        torch.save(ck, folder / 'vcm.pt')
        save(folder / 'run.json', {'arm': arm, 'coefficients': coefficients,
            'protocol_sha256': digest(EXPERIMENT / 'protocol.json'), 'source_sha256': digest(__file__),
            'checkpoint_sha256': digest(folder / 'vcm.pt'), 'optimizer_used': False, 'test_used': False})
        finalize(arm)
        reports[arm] = json.loads((EXPERIMENT / (arm + '-development.json')).read_text())
    selected = max(reports, key=lambda a: tuple(reports[a]['rank']))
    files = [EXPERIMENT / 'protocol.json', Path(__file__), DATA / 'frozen.json', ROOT / 'scripts/test/complete_commands.py']
    for arm in reports:
        files.extend(CACHE / arm / name for name in ['bundle.json', 'vcm.fp32.onnx'])
    save(EXPERIMENT / 'selection-before-test.json', {'selected_arm': selected,
        'development_ranks': {a: r['rank'] for a, r in reports.items()}, 'created_unix': time.time(),
        'files': {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files},
        'test_used_to_select': False, 'historical_test_already_examined': True})
    print(json.dumps({'selected_arm': selected, 'ranks': {a: r['rank'] for a, r in reports.items()}}), flush=True)


def evaluate():
    from tinyvcm.complete_eval import fixed_test, number_test
    selection = json.loads((EXPERIMENT / 'selection-before-test.json').read_text())
    for path, sha in selection['files'].items():
        assert digest(ROOT / path) == sha, path
    if (EXPERIMENT / 'results.json').exists():
        raise ValueError('Preserve existing test receipt')
    report = fixed_test(selection['selected_arm'])
    numbers = number_test(selection['selected_arm'])
    save(EXPERIMENT / 'results.json', {'selected_arm': selection['selected_arm'], 'fixed': report,
        'numbers': numbers['groups'], 'selection_sha256': digest(EXPERIMENT / 'selection-before-test.json'),
        'installed': False, 'test_is_historical_regression': True})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else evaluate()
