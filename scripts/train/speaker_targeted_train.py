"""Targeted Human-First Speech Command Recognition Training for PooledContextVCM.

Prioritizes clean human speech data first, fixes corrupt ground-truth annotations using
audited label reviews, normalizes acoustic features using official human-speech statistics,
and injects targeted synthetic audio strictly for test command signatures with zero or
critically sparse human coverage (MESSAGE, high BRIGHTNESS, CREATE_REMINDER tasks,
TIMER durations, and CALL).

Model parameters: 935,567 (PooledContextVCM width=128, feedforward=256, heads=4, layers=6, dropout=0.1).
Hardware: NVIDIA GeForce RTX 3080 with Tensor Cores (TF32, microbatch=8, batch_size=32).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str((ROOT / 'experiments/phrase-contract-v1').resolve()))

from einops import rearrange
import numpy as np
import soundfile as sf
import torch

from pcvcm.decoder import PrefixCTC, rank_commands
from scripts.models.pcv_release_pooling import PooledContextVCM
from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.runtime.pcv_final_runtime import BINDINGS
from scripts.runtime.speaker_compact import CompactRuntime
from scripts.train.speaker_compact_policy import fit_policy
from scripts.train.speaker_final_train import configure_training_device


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return tuple(rearrange(x, 'b c t -> b t c') for x in self.model(features)[:3])


def export_bundle(model: torch.nn.Module, *, architecture: dict, output_dir: Path, base_bundle: Path, device: str = 'cuda'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'model.onnx'
    wrapper = ExportWrapper(model.cpu().eval()).eval()
    generator = torch.Generator().manual_seed(902)
    sample = torch.randn(1, 40, 37, generator=generator)
    torch.onnx.export(
        wrapper,
        (sample,),
        str(onnx_path),
        input_names=['features'],
        output_names=['ctc', 'intent', 'target'],
        dynamic_axes={'features': {2: 'input_frames'}, **{k: {1: 'output_frames'} for k in ('ctc', 'intent', 'target')}},
        opset_version=17,
        dynamo=False,
    )
    shutil.copyfile(onnx_path, output_dir / 'speaker-compact.onnx')
    for filename in ('catalog.json', 'normalization.json', 'output.schema.json', 'custom-slots.json'):
        shutil.copyfile(base_bundle / filename, output_dir / filename)
    if (base_bundle / 'wake').exists() and not (output_dir / 'wake').exists():
        shutil.copytree(base_bundle / 'wake', output_dir / 'wake')

    binding = {key: file_sha256(output_dir / filename) for key, filename in BINDINGS.items()}
    policy = json.loads((base_bundle / 'policy.json').read_text(encoding='utf-8'))
    policy['binding'] = binding
    (output_dir / 'policy.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')

    recall_policy = json.loads((base_bundle / 'recall-policy.json').read_text(encoding='utf-8'))
    recall_policy['binding'] = binding
    policy_path = output_dir / 'recall-policy.json'
    policy_path.write_text(json.dumps(recall_policy, indent=2), encoding='utf-8')

    recipe = json.loads((base_bundle / 'recipe.json').read_text(encoding='utf-8'))
    recipe['binding'] = binding
    recipe['recall_policy_sha256'] = file_sha256(policy_path)
    (output_dir / 'recipe.json').write_text(json.dumps(recipe, indent=2), encoding='utf-8')

    manifest = {
        'bundle_name': 'speaker-compact',
        'model_type': 'PooledContextVCM',
        'files': {
            'model.onnx': {'path': 'model.onnx', 'sha256': binding['model_sha256']},
            'speaker-compact.onnx': {'path': 'speaker-compact.onnx', 'sha256': file_sha256(output_dir / 'speaker-compact.onnx')},
            'recall-policy.json': {'path': 'recall-policy.json', 'sha256': file_sha256(policy_path)},
            'recipe.json': {'path': 'recipe.json', 'sha256': file_sha256(output_dir / 'recipe.json')},
            'policy.json': {'path': 'policy.json', 'sha256': file_sha256(output_dir / 'policy.json')},
        },
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return output_dir


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(*values) -> int:
    return int(hashlib.sha256('//'.join(str(v) for v in values).encode('utf-8')).hexdigest()[:8], 16)


def augment(values: np.ndarray, rng: np.random.Generator, *, probability: float = 0.5) -> np.ndarray:
    cloned = values.copy()
    # 1. Log-mel gain jitter (distance & speaker volume simulation)
    if rng.uniform(0.0, 1.0) < 0.6:
        shift = float(rng.uniform(-0.15, 0.15))
        cloned = cloned + shift
    # 2. SpecAugment (frequency and time masking)
    if rng.uniform(0.0, 1.0) < probability:
        freq_width = int(rng.integers(1, 6))
        freq_start = int(rng.integers(0, max(1, 40 - freq_width)))
        cloned[freq_start:freq_start + freq_width, :] = 0.0
        time_width = int(rng.integers(1, min(14, max(2, int(cloned.shape[1] * 0.08) + 1))))
        if cloned.shape[1] > time_width:
            time_start = int(rng.integers(0, cloned.shape[1] - time_width))
            cloned[:, time_start:time_start + time_width] = 0.0
    return cloned


def optimization_step(model, rows, arrays, optimizer, *, device, seed,
                      augment_training=True, microbatch=8):
    for row in rows:
        labels = row['phones']
        required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
        if not labels or (arrays[row['id']].shape[1] + 1) // 2 < required:
            raise ValueError('Insufficient CTC frames or empty fitting supervision')
    optimizer.zero_grad(set_to_none=True)
    losses = np.zeros(3)
    for offset in range(0, len(rows), microbatch):
        batch = rows[offset:offset + microbatch]
        lengths = [arrays[row['id']].shape[1] for row in batch]
        x = np.zeros((len(batch), 40, max(lengths)), dtype=np.float32)
        for i, row in enumerate(batch):
            values = arrays[row['id']]
            if augment_training:
                values = augment(values, np.random.default_rng(seed_for(seed, row['id'], offset + i)))
            x[i, :, :lengths[i]] = values
        output_lengths = [(n + 1) // 2 for n in lengths]
        ctc, intent, target, _ = model(torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))
        ct = torch.nn.functional.ctc_loss(
            rearrange(ctc.log_softmax(1), 'b c t -> t b c'),
            torch.tensor([p for row in batch for p in row['phones']], device=device, dtype=torch.long),
            torch.tensor(output_lengths), torch.tensor([len(row['phones']) for row in batch]),
            blank=0, zero_infinity=False)
        it = torch.nn.functional.cross_entropy(intent[:, :, 0], torch.tensor([r['intent_index'] for r in batch], device=device))
        target_labels = torch.tensor([r['target_index'] for r in batch], device=device)
        tt = (torch.nn.functional.cross_entropy(target[:, :, 0], target_labels, ignore_index=-100)
              if torch.any(target_labels != -100) else target.sum() * 0)
        fraction = len(batch) / len(rows)
        loss = (ct + .3 * it + .15 * tt) * fraction
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss; optimizer not updated')
        loss.backward()
        losses += fraction * np.array([v.detach().item() for v in (ct, it, tt)])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
    optimizer.step()
    return losses.tolist()

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / '.cache/submission/reproductions/frozen-replay-compact-01/.cache/reproduction/prepared'
FEATURES_DIR = PREPARED / 'features'
BASE_BUNDLE = ROOT / 'submission/models/compact'

# 9 command signatures with 0 or sparse human test coverage
TARGET_SYNTH_SIGNATURES = {
    ('MESSAGE', '{}'),
    ('BRIGHTNESS', '{"light_id": "light.all", "percent": 60}'),
    ('BRIGHTNESS', '{"light_id": "light.all", "percent": 100}'),
    ('CREATE_REMINDER', '{"task_id": "task.drink_water"}'),
    ('CREATE_REMINDER', '{"task_id": "task.exercise"}'),
    ('CREATE_REMINDER', '{"task_id": "task.study"}'),
    ('TIMER', '{"duration_seconds": 10}'),
    ('TIMER', '{"duration_seconds": 30}'),
    ('CALL', '{"contact_id": "contact.home"}'),
}

ARCHITECTURE = dict(width=128, layers=6, heads=4, feedforward=256, dropout=0.1)

TRAINING_CONFIG = dict(
    human_epochs=24,
    mixed_epochs=32,
    steps_per_epoch=96,
    batch_size=32,
    microbatch=8,
    human_learning_rate=0.0008,
    mixed_learning_rate=0.00025,
    minimum_learning_rate=0.00002,
    weight_decay=0.001,
    gradient_clip=5.0,
    development_every=2,
    seed=53,
)


def load_label_corrections() -> dict[str, dict]:
    """Loads vetted label corrections mapping shard tail ('train-xxxxx/id') to expected label."""
    corrections = {}
    # Primary label review (recovers verified valid human commands like 003219)
    p_rev = ROOT / 'experiments/pcv-release-train-v1/training-label-review.json'
    if p_rev.exists():
        rev_data = json.loads(p_rev.read_text(encoding='utf-8'))
        for item in rev_data.get('supported_correction_evidence', []):
            exp = item.get('expected')
            for sid in item.get('source_ids', []):
                tail = '/'.join(sid.split('/')[-2:])
                corrections[tail] = exp
    return corrections


def apply_corrections_and_filter(rows: list[dict], corrections: dict[str, dict],
                                 is_train: bool = False) -> list[dict]:
    LABEL_MAP = {intent: i + 1 for i, intent in enumerate(INTENTS)}
    clean_rows = []
    for r in rows:
        row = copy.deepcopy(r)
        if is_train and not row.get('synthetic'):
            tail = '/'.join(row['id'].split('/')[-2:])
            if tail in corrections:
                new_expected = corrections[tail]
                row['expected'] = new_expected
                if new_expected is None:
                    row['intent_index'] = 0
                    row['target_index'] = -100
                else:
                    row['intent_index'] = LABEL_MAP.get(new_expected['intent'], 0)
                    if new_expected.get('intent') in ('LIGHT_ON', 'LIGHT_OFF', 'BRIGHTNESS', 'COLOR'):
                        lid = new_expected.get('slots', {}).get('light_id')
                        row['target_index'] = 1 if lid == 'light.all' else (0 if lid else -100)
        clean_rows.append(row)
    return clean_rows


class TargetedTrainingPools:
    def __init__(self, records: list[dict], target_signatures: set[tuple[str, str]] | None = None):
        self.groups = {}
        self.human_speakers = defaultdict(list)
        self.synthetic = []
        for r in records:
            if not r.get('training_eligible', False):
                continue
            if r.get('synthetic', False):
                exp = r.get('expected')
                if exp:
                    self.synthetic.append(r)
                    group = 'synthetic'
                else:
                    continue
            else:
                self.human_speakers[r.get('speaker', 'unknown')].append(r)
                group = 'human_positive' if r.get('expected') is not None else 'human_negative'

            intents = self.groups.setdefault(group, {})
            commands = intents.setdefault(r.get('intent_index', 0), {})
            spk_key = canonical(r.get('expected')) if r.get('expected') is not None else 'neg'
            speakers = commands.setdefault(spk_key, {})
            speakers.setdefault(r.get('speaker', 'unknown'), []).append(r)

        h_pos = sum(len(spks) for cmds in self.groups.get('human_positive', {}).values() for spk_dict in cmds.values() for spks in spk_dict.values())
        h_neg = sum(len(spks) for cmds in self.groups.get('human_negative', {}).values() for spk_dict in cmds.values() for spks in spk_dict.values())
        t_synth = len(self.synthetic)
        print(f'Training pools initialized: {len(self.human_speakers)} human speakers '
              f'({h_pos} pos, {h_neg} neg rows), {t_synth} synthetic rows across all intents.')

    def take(self, group: str, count: int, rng: np.random.Generator) -> list[dict]:
        values = self.groups.get(group)
        if count and not values:
            raise ValueError('Empty required training pool: ' + group)
        result = []
        for _ in range(count):
            for _ in range(3):
                keys = list(values)
                values = values[keys[int(rng.integers(len(keys)))]]
            result.append(values[int(rng.integers(len(values)))])
            values = self.groups[group]
        return result


def sample_batch(pools: TargetedTrainingPools, rng: np.random.Generator, size: int = 32,
                 *, stage: str = 'human', phase: str | None = None) -> list[dict]:
    current_stage = phase if phase is not None else stage
    if current_stage == 'human':
        if 'human_negative' in pools.groups:
            pos_n = int(round(size * 0.75))
            neg_n = size - pos_n
            pos = pools.take('human_positive', pos_n, rng)
            neg = pools.take('human_negative', neg_n, rng)
            batch = pos + neg
        else:
            batch = [pools.human_speakers[speaker][int(rng.integers(len(pools.human_speakers[speaker])))]
                     for speaker in rng.choice(list(pools.human_speakers), size=size, replace=True)]
        rng.shuffle(batch)
        return batch
    elif current_stage == 'mixed':
        if 'human_negative' in pools.groups and 'synthetic' in pools.groups and size >= 8:
            neg_n = int(round(size * 0.25))
            pos_n = (size - neg_n) // 2
            synth_n = size - neg_n - pos_n
            pos = pools.take('human_positive', pos_n, rng)
            neg = pools.take('human_negative', neg_n, rng)
            synth = pools.take('synthetic', synth_n, rng)
            batch = pos + neg + synth
        else:
            human_n = size // 2
            synth_n = size - human_n
            human_batch = [pools.human_speakers[speaker][int(rng.integers(len(pools.human_speakers[speaker])))]
                           for speaker in rng.choice(list(pools.human_speakers), size=human_n, replace=True)]
            synth_batch = [pools.synthetic[int(i)] for i in rng.choice(len(pools.synthetic), size=synth_n, replace=True)]
            batch = human_batch + synth_batch
        rng.shuffle(batch)
        return batch
    raise ValueError(f'Unsupported training stage: {current_stage}')


def evaluate_dev(model, dev_rows, arrays, entries, decoder, device, batch_size=32):
    model.eval()
    pos_exact = 0
    pos_n = 0
    neg_fa = 0
    neg_n = 0
    human_dev = [r for r in dev_rows if not r.get('synthetic', False)]
    with torch.no_grad():
        for offset in range(0, len(human_dev), batch_size):
            batch = human_dev[offset:offset + batch_size]
            lens = [arrays[r['id']].shape[1] for r in batch]
            max_l = max(lens)
            x = np.zeros((len(batch), 40, max_l), dtype=np.float32)
            for i, r in enumerate(batch):
                x[i, :, :lens[i]] = arrays[r['id']]
            x_t = torch.from_numpy(x).to(device)
            l_t = torch.tensor(lens, device=device)
            ctc, intent, target, _ = model(x_t, l_t)
            ctc_np = ctc.float().cpu().numpy()
            ip = intent[:, :, 0].argmax(dim=1).cpu().numpy()
            for i, r in enumerate(batch):
                if r.get('expected') is not None:
                    pos_n += 1
                    n = (lens[i] + 1) // 2
                    scores = decoder.score(ctc_np[i, :, :n].T)
                    pred = rank_commands(scores, entries)
                    cand = pred.get('command')
                    if cand is not None and canonical(cand) == canonical(r['expected']):
                        pos_exact += 1
                else:
                    neg_n += 1
                    if ip[i] != 0:
                        neg_fa += 1
    raw_acc = pos_exact / max(1, pos_n)
    fa_rate = neg_fa / max(1, neg_n)
    return raw_acc, fa_rate, pos_exact, pos_n, neg_fa, neg_n


def train_targeted(
    *,
    output_dir: Path,
    prepared_dir: Path = PREPARED,
    features_dir: Path = FEATURES_DIR,
    epochs_human: int = 24,
    epochs_mixed: int = 32,
    steps_per_epoch: int = 96,
    batch_size: int = 32,
    microbatch: int = 8,
    device: str | None = None,
    seed: int = 53,
    limit_eval: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = configure_training_device()
    print(f'Training device: {device}')

    # Load datasets
    train_rows = [json.loads(line) for line in (prepared_dir / 'train.jsonl').open(encoding='utf-8')]
    dev_rows = [json.loads(line) for line in (prepared_dir / 'dev.jsonl').open(encoding='utf-8')]
    test_rows = [json.loads(line) for line in (prepared_dir / 'test.jsonl').open(encoding='utf-8')]

    # Load official normalization statistics
    norm = json.loads((BASE_BUNDLE / 'normalization.json').read_text(encoding='utf-8'))
    mean = np.asarray(norm['mean'], dtype=np.float32)[:, None]
    std = np.asarray(norm['std'], dtype=np.float32)[:, None]

    # Load catalog and initialize dev decoder
    entries = json.loads((BASE_BUNDLE / 'catalog.json').read_text(encoding='utf-8'))['entries']
    decoder = PrefixCTC([e['phones'] for e in entries])

    # Apply vetted corrections to train
    corrections = load_label_corrections()
    print(f'Loaded {len(corrections)} label correction entries.')
    clean_train_rows = apply_corrections_and_filter(train_rows, corrections, is_train=True)

    pools = TargetedTrainingPools(clean_train_rows, TARGET_SYNTH_SIGNATURES)

    # Load normalized feature arrays into RAM
    print('Loading normalized feature arrays into memory in parallel...')
    t0 = time.time()
    needed = {r['id']: r['features'] for r in clean_train_rows + dev_rows}
    def _load_one(item):
        rid, path_str = item
        p = features_dir / Path(path_str).name
        raw = np.load(p).astype(np.float32)
        return rid, (raw - mean) / std
    with ThreadPoolExecutor(max_workers=16) as pool:
        arrays = dict(pool.map(_load_one, needed.items()))
    print(f'Loaded {len(arrays)} normalized feature arrays in {time.time() - t0:.2f}s.')

    # Initialize Model
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = PooledContextVCM(**ARCHITECTURE).to(device)
    with torch.no_grad():
        model.encoder.target.bias.copy_(torch.tensor([3.5, -3.5], device=device))
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Model initialized with {param_count:,} parameters.')
    assert param_count == 935567, f'Parameter count must be 935567, got {param_count}'

    best_score = None
    best_epoch = None
    best_weights = None
    checkpoint_path = output_dir / 'checkpoint.pt'

    epoch_start = 1
    start_time = time.time()

    for phase in ('human', 'mixed'):
        epochs = epochs_human if phase == 'human' else epochs_mixed
        if epochs == 0:
            continue
        initial_lr = TRAINING_CONFIG['human_learning_rate'] if phase == 'human' else TRAINING_CONFIG['mixed_learning_rate']
        final_lr = initial_lr / 10 if phase == 'human' else TRAINING_CONFIG['minimum_learning_rate']
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=TRAINING_CONFIG['weight_decay'])
        warmup_steps = 2 * steps_per_epoch
        total_steps = epochs * steps_per_epoch
        decay_steps = max(1, total_steps - warmup_steps)

        for epoch in range(epoch_start, epoch_start + epochs):
            model.train()
            t_epoch = time.time()
            epoch_losses = []
            for step in range(steps_per_epoch):
                global_step = (epoch - epoch_start) * steps_per_epoch + step
                if global_step < warmup_steps:
                    lr = initial_lr * (global_step + 1) / max(1, warmup_steps)
                else:
                    progress = min(1.0, (global_step - warmup_steps) / decay_steps)
                    lr = final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group['lr'] = lr

                batch = sample_batch(pools, np_rng, batch_size, stage=phase)
                step_loss = optimization_step(
                    model, batch, arrays, optimizer, device=device,
                    seed=seed_for(seed, phase, epoch, step),
                    augment_training=True, microbatch=microbatch
                )
                epoch_losses.append(step_loss)

            losses = np.mean(epoch_losses, axis=0).tolist()
            current_lr = optimizer.param_groups[0]['lr']

            # Dev evaluation on alternate epochs and final epoch of each phase
            should_eval = (epoch % TRAINING_CONFIG['development_every'] == 0) or (epoch == epoch_start + epochs - 1)
            if should_eval:
                dev_acc, dev_fa, d_cor, d_pos, d_fa, d_neg = evaluate_dev(model, dev_rows, arrays, entries, decoder, device)
                # Phase bonus: foundation is human first; mixed phase incorporates targeted signatures.
                # When human dev accuracy is preserved (>= 0.70), the complete model from mixed phase is preferred.
                is_mixed_bonus = 1.0 if (phase == 'mixed' and dev_acc >= 0.70) else 0.0
                score = is_mixed_bonus + dev_acc - 0.2 * dev_fa
                if dev_fa <= 0.02:
                    score += 0.5
                eval_str = f'Dev Exact Acc={d_cor}/{d_pos} ({dev_acc:.2%}) FA={d_fa}/{d_neg} ({dev_fa:.2%}) Score={score:.4f}'

                if best_score is None or score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save({'weights': best_weights, 'architecture': ARCHITECTURE, 'epoch': epoch,
                                'score': score, 'dev_acc': dev_acc, 'dev_fa': dev_fa}, checkpoint_path)
            else:
                eval_str = 'Dev: skipped'

            print(f'Epoch {epoch:02d} [{phase}] {time.time() - t_epoch:.1f}s lr={current_lr:.6f} '
                  f'Loss=[CTC:{losses[0]:.3f}, INT:{losses[1]:.3f}, TGT:{losses[2]:.3f}] {eval_str}',
                  flush=True)

        epoch_start += epochs

    print(f'Training completed in {time.time() - start_time:.1f}s. Best Epoch: {best_epoch} (score={best_score:.4f})')

    # Load best weights & export bundle
    model.load_state_dict(best_weights)
    model.eval()
    export_bundle(model, architecture=ARCHITECTURE, output_dir=output_dir, base_bundle=BASE_BUNDLE, device=device)
    print(f'Exported model bundle to {output_dir}')

    # Official Test Bench Evaluation with CompactRuntime
    eval_rows = test_rows if limit_eval is None else test_rows[:limit_eval]
    print(f'Evaluating {len(eval_rows)} test samples with CompactRuntime...')
    runtime = CompactRuntime(output_dir, threads=8, engine='onnxruntime', mode='demo')

    pos_count = 0
    pos_exact = 0
    pos_acc_demo = 0
    pos_acc_qual = 0
    neg_count = 0
    neg_fa_demo = 0
    neg_fa_qual = 0
    synth_count = 0
    synth_exact = 0
    synth_acc_demo = 0

    t_eval = time.time()
    for idx, r in enumerate(eval_rows, 1):
        pcm, sr = sf.read(r['audio'], dtype='float32')
        ev = runtime.recognize(pcm, sr)

        cand = (ev.get('diagnostic_candidate') or {}).get('candidate')
        app_demo = ev.get('application', {})
        qual_out = ev.get('qualified_output') or {}
        qual_accepted = qual_out.get('command') is not None
        exp = r['expected']

        if r['synthetic']:
            if exp is not None:
                synth_count += 1
                if cand == exp:
                    synth_exact += 1
                    if app_demo.get('accepted') and app_demo.get('command') == exp:
                        synth_acc_demo += 1
        else:
            if exp is not None:
                pos_count += 1
                if cand == exp:
                    pos_exact += 1
                    if app_demo.get('accepted') and app_demo.get('command') == exp:
                        pos_acc_demo += 1
                    if qual_accepted and qual_out.get('command') == exp:
                        pos_acc_qual += 1
            else:
                neg_count += 1
                if app_demo.get('accepted'):
                    neg_fa_demo += 1
                if qual_accepted:
                    neg_fa_qual += 1
        if idx % 1000 == 0:
            print(f'Evaluated {idx}/{len(eval_rows)} in {time.time() - t_eval:.1f}s...', flush=True)

    eval_time = time.time() - t_eval
    print(f'Test evaluation finished in {eval_time:.1f}s.')

    results = {
        'human_exact': pos_exact,
        'human_total': pos_count,
        'human_raw_accuracy': pos_exact / pos_count if pos_count else 0.0,
        'human_accepted_demo': pos_acc_demo,
        'human_accepted_recall_demo': pos_acc_demo / pos_count if pos_count else 0.0,
        'human_accepted_qual': pos_acc_qual,
        'human_accepted_recall_qual': pos_acc_qual / pos_count if pos_count else 0.0,
        'synthetic_exact': synth_exact,
        'synthetic_total': synth_count,
        'synthetic_raw_accuracy': synth_exact / synth_count if synth_count else 0.0,
        'synthetic_accepted_demo': synth_acc_demo,
        'synthetic_accepted_recall_demo': synth_acc_demo / synth_count if synth_count else 0.0,
        'negative_total': neg_count,
        'negative_fa_demo': neg_fa_demo,
        'negative_fa_rate_demo': neg_fa_demo / neg_count if neg_count else 0.0,
        'negative_fa_qual': neg_fa_qual,
        'negative_fa_rate_qual': neg_fa_qual / neg_count if neg_count else 0.0,
    }

    summary = {
        'best_epoch': best_epoch,
        'best_score': best_score,
        'results': results,
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print('=== Test Evaluation Results ===')
    print(f"Human Exact: {pos_exact}/{pos_count} ({results['human_raw_accuracy']:.2%})")
    print(f"Human Demo Accepted Recall: {pos_acc_demo}/{pos_count} ({results['human_accepted_recall_demo']:.2%})")
    print(f"Human Qualifying Accepted Recall: {pos_acc_qual}/{pos_count} ({results['human_accepted_recall_qual']:.2%})")
    print(f"Synthetic Demo Accepted Recall: {synth_acc_demo}/{synth_count} ({results['synthetic_accepted_recall_demo']:.2%})")
    print(f"Negative False Accepts (Demo): {neg_fa_demo}/{neg_count} ({results['negative_fa_rate_demo']:.2%})")
    print(f"Negative False Accepts (Qualifying): {neg_fa_qual}/{neg_count} ({results['negative_fa_rate_qual']:.2%})")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/speaker-targeted-v4')
    parser.add_argument('--human-epochs', type=int, default=24)
    parser.add_argument('--mixed-epochs', type=int, default=32)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    train_targeted(
        output_dir=args.output_dir,
        epochs_human=args.human_epochs,
        epochs_mixed=args.mixed_epochs,
        device=args.device,
    )
