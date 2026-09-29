"""Own-initialized baseline CNN, full-development selection, retained epochs."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np

from .contract import Contract, canonical, compile_catalog, file_hash, identity, read_jsonl, write_json
from .data import batch_arrays, verified_features
from .decoder import PrefixCTC, rank_commands


def select_checkpoint(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError('No completed development candidate')
    return max(candidates, key=lambda r: (r['phrases_passing'], r['worst_use_case'],
                                         r['macro_exact'], r['correct'], -r['seed'], -r['epoch']))


@contextmanager
def native_cpu_convolution():
    """Use one CPU convolution path for whole/chunk numerical comparisons."""
    import torch
    previous = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = previous


def training_smoke(steps: int = 12) -> dict:
    with native_cpu_convolution():
        result = _training_smoke(steps)
    result['cpu_convolution_backend'] = 'native; oneDNN disabled explicitly'
    result['fp32_parity_tolerance'] = 2e-5
    return result


def _training_smoke(steps: int = 12) -> dict:
    """Check gradients and state carry on artificial feature tensors only."""
    import torch
    from vcm.model import CausalVCM, STATE_SHAPES
    torch.set_num_threads(2)
    torch.manual_seed(53)
    model = CausalVCM()
    x = torch.randn(8, 40, 64)
    targets = torch.tensor([1, 2, 3, 4] * 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    losses = []
    for _ in range(steps):
        logits, _ = model(x)
        loss = torch.nn.functional.ctc_loss(
            logits.log_softmax(1).permute(2, 0, 1), targets,
            torch.full((8,), 64), torch.full((8,), 4), zero_infinity=False)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    error = 0.
    with torch.no_grad():
        full, expected_states = model(x[:1])
        for chunk in (1, 3, 4, 8, 17):
            states, outputs = None, []
            for start in range(0, 64, chunk):
                value, states = model(x[:1, :, start:start + chunk], states)
                outputs.append(value)
            error = max(error, float((torch.cat(outputs, dim=2) - full).abs().max()),
                        *(float((a - b).abs().max()) for a, b in zip(states, expected_states)))
    return {'parameters': sum(p.numel() for p in model.parameters()),
            'cache_bytes': int(sum(np.prod(s) for s in STATE_SHAPES) * 4),
            'initial_loss': losses[0], 'final_loss': losses[-1], 'losses': losses,
            'maximum_chunk_error': error, 'frame_chunks': [1, 3, 4, 8, 17],
            'human_recordings': 0, 'qualified': False,
            'evidence_type': 'generated_feature_training_smoke_NOT_SPEECH',
            'checkpoint_saved': False}


def development(model, rows: list[dict], arrays: dict, entries: list[dict], device) -> tuple[dict, list[dict]]:
    import torch
    decoder = PrefixCTC([e['phones'] for e in entries])
    records = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            x = torch.from_numpy(arrays[row['recording_id']][None]).to(device)
            logits, _ = model(x)
            prediction = rank_commands(decoder.score(logits[0].T.cpu().numpy()), entries)
            exact = canonical(prediction['command']) == canonical(row.get('expected_command'))
            records.append({'recording_id': row['recording_id'], 'phrase_id': row.get('phrase_id'),
                            'use_case': row['use_case'], 'reference_class': row['reference_class'],
                            'expected': row.get('expected_command'), 'candidate': prediction['command'],
                            'raw_exact': exact})
    positive = [r for r in records if r['reference_class'] == 'supported_positive']
    phrases, families = defaultdict(list), defaultdict(list)
    for row in positive:
        phrases[row['phrase_id']].append(row['raw_exact'])
        families[row['use_case']].append(row['raw_exact'])
    metrics = {'phrases_passing': int(sum(np.mean(v) >= .9 for v in phrases.values())),
               'worst_use_case': min(float(np.mean(v)) for v in families.values()),
               'macro_exact': float(np.mean([np.mean(v) for v in phrases.values()])),
               'correct': sum(r['raw_exact'] for r in positive),
               'positive_recordings': len(positive),
               'critical_negative_errors': sum(not r['raw_exact'] for r in records
                                                if r['reference_class'] != 'supported_positive')}
    return metrics, records


def sampling_weights(rows: list[dict], positive_fraction: float,
                     hard_multiplier: float) -> np.ndarray:
    """Balance use case, phrase, then speaker within positive/negative roles."""
    if not 0 < positive_fraction < 1 or hard_multiplier not in (1., 2.):
        raise ValueError('Frozen sampling settings changed')
    roles = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(rows):
        role = row['reference_class'] == 'supported_positive'
        phrase = row.get('phrase_id') or 'natural_unknown'
        roles[role][row['use_case']][phrase].append(i)
    if set(roles) != {False, True}:
        raise ValueError('Both positive and real negative recordings required')
    result = np.zeros(len(rows))
    confusing = {13, 30, 14, 40, 15, 50, 16, 60, 17, 70,
                 18, 80, 19, 90, 37, 73}
    for role, use_cases in roles.items():
        role_mass = positive_fraction if role else 1 - positive_fraction
        for phrases in use_cases.values():
            for indices in phrases.values():
                speakers = defaultdict(list)
                for i in indices:
                    speakers[rows[i]['speaker_id']].append(i)
                for recordings in speakers.values():
                    mass = role_mass / len(use_cases) / len(phrases) / len(speakers)
                    for i in recordings:
                        result[i] = mass / len(recordings)
        if hard_multiplier != 1.:
            indices = [i for phrases in use_cases.values()
                       for values in phrases.values() for i in values]
            before = result[indices].sum()
            for i in indices:
                command = rows[i].get('expected_command') or {}
                if (command.get('value') in confusing
                        or rows[i]['use_case'] == 'UC-10'
                        or rows[i].get('critical_zero_accept') is True):
                    result[i] *= hard_multiplier
            result[indices] *= before / result[indices].sum()
    return result / result.sum()


def fit(config_path: Path, contract: Contract, lexicon_path: Path, output: Path) -> dict:
    import torch
    from vcm.model import CausalVCM
    config = json.loads(Path(config_path).read_text())
    prepared = Path(config['prepared'])
    freeze = json.loads((prepared / 'freeze.json').read_text())
    if freeze['contract_sha256'] != contract.sha256:
        raise ValueError('Training contract differs from the frozen dataset')
    for name, digest in freeze['artifacts'].items():
        if file_hash(prepared / name) != digest:
            raise ValueError(f'Frozen metadata changed: {name}')
    if not config.get('rights_and_near_duplicate_audit_receipt'):
        raise ValueError('Supply reviewed rights and near-duplicate audit receipt before fitting')
    audit_path = Path(config['rights_and_near_duplicate_audit_receipt'])
    audit = json.loads(audit_path.read_text())
    if audit.get('frozen_data_sha256') != file_hash(prepared / 'freeze.json') or audit.get('passed') is not True:
        raise ValueError('Audit receipt is missing or bound to other data')
    if Path(output).exists():
        raise FileExistsError('Run outputs are append-only; choose a new run ID')
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU or pretrained fallback')
    torch.set_num_threads(2)
    train = read_jsonl(prepared / 'train.jsonl')
    dev = read_jsonl(prepared / 'dev.jsonl')
    norm = json.loads((prepared / 'normalization.json').read_text())
    mean = np.asarray(norm['mean'], dtype=np.float32)[:, None]
    std = np.asarray(norm['std'], dtype=np.float32)[:, None]
    arrays = {r['recording_id']: (verified_features(r, prepared) - mean) / std for r in train + dev}
    catalog = compile_catalog(contract, lexicon_path)
    Path(output).mkdir(parents=True)
    write_json(Path(output) / 'catalog.json', catalog)
    write_json(Path(output) / 'normalization.json', norm)
    protocol = {'config': config, 'config_sha256': file_hash(config_path),
                'freeze_sha256': file_hash(prepared / 'freeze.json'),
                'initialization': 'random; no external neural weights',
                'code': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')},
                'baseline_model_sha256': file_hash(Path(__import__('vcm.model', fromlist=['x']).__file__)),
                'test_accessed': False, 'calibration_accessed': False}
    write_json(Path(output) / 'protocol.json', protocol)
    weights = sampling_weights(train, config.get('positive_sampling_fraction', .8),
                               config.get('hard_example_multiplier', 1.))
    history, candidates = [], []
    all_started = time.perf_counter()
    for seed in config['seeds']:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = CausalVCM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        seed_dir = Path(output) / f'seed-{seed}'
        seed_dir.mkdir()
        for epoch in range(1, min(int(config['epochs']), 40) + 1):
            if time.perf_counter() - all_started >= config['wall_clock_seconds']:
                write_json(Path(output) / 'stopped.json', {'reason': 'compute_budget', 'history': history})
                return {'status': 'STOPPED_BUDGET', 'final_model': False}
            model.train()
            order = rng.choice(len(train), size=len(train), replace=True, p=weights)
            losses = []
            for start in range(0, len(order), config['batch_size']):
                if time.perf_counter() - all_started >= config['wall_clock_seconds']:
                    write_json(Path(output) / 'stopped.json',
                               {'reason': 'compute_budget', 'history': history,
                                'interrupted_seed': seed, 'interrupted_epoch': epoch})
                    return {'status': 'STOPPED_BUDGET', 'final_model': False}
                batch = [train[int(i)] for i in order[start:start + config['batch_size']]]
                x, lengths = batch_arrays([arrays[r['recording_id']] for r in batch])
                logits, _ = model(torch.from_numpy(x).to(device))
                targets = torch.tensor([p for r in batch for p in r['phones']], dtype=torch.long, device=device)
                loss = torch.nn.functional.ctc_loss(
                    logits.log_softmax(1).permute(2, 0, 1), targets,
                    torch.tensor(lengths), torch.tensor([len(r['phones']) for r in batch]),
                    zero_infinity=False)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite CTC loss; preserve the failed run')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
            checkpoint = seed_dir / f'epoch-{epoch:03}.pt'
            torch.save({'weights': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'seed': seed, 'torch_rng': torch.get_rng_state(),
                        'protocol_sha256': identity(protocol)}, checkpoint)
            record = {'seed': seed, 'epoch': epoch, 'ctc_loss': float(np.mean(losses)),
                      'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint)}
            if epoch in config['selection_epochs']:
                metrics, predictions = development(model, dev, arrays, catalog['entries'], device)
                record.update(metrics)
                candidates.append(dict(record))
                write_json(seed_dir / f'development-{epoch:03}.json', predictions)
            history.append(record)
            write_json(Path(output) / 'history.json', history)
            print(json.dumps(record), flush=True)
    winner = select_checkpoint(candidates)
    status = 'DEVELOPMENT_SELECTED' if winner['phrases_passing'] == 570 else 'BLOCKED_DEVELOPMENT'
    lock = {'status': status, 'winner': winner, 'candidates': candidates,
            'calibration_accessed': False, 'confirmation_accessed': False,
            'qualified': False, 'protocol_sha256': identity(protocol)}
    write_json(Path(output) / 'selection.json', lock)
    return lock
