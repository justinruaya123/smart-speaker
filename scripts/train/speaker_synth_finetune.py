"""Targeted synthetic fine-tuning for smart-speaker compact model.

Warm-starts from speaker-targeted-v7 checkpoint and refines boundary discrimination
on hard synthetic command pairs while anchoring on human speech and negative rejection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import json
import math
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

from pcvcm.contract import canonical
from pcvcm.decoder import PrefixCTC, rank_commands
from scripts.models.pcv_release_pooling import PooledContextVCM
from scripts.runtime.pcv_schema import INTENTS

LABEL_MAP = {intent: i + 1 for i, intent in enumerate(INTENTS)}
from scripts.runtime.speaker_compact import CompactRuntime, THRESHOLDS, validate_policy
from scripts.runtime.pcv_final_runtime import BINDINGS
from scripts.train.speaker_final_train import configure_training_device
from scripts.train.speaker_targeted_train import (
    ARCHITECTURE,
    BASE_BUNDLE,
    PREPARED,
    FEATURES_DIR,
    load_label_corrections,
    apply_corrections_and_filter,
    TargetedTrainingPools,
    optimization_step,
    evaluate_dev,
    ExportWrapper,
    file_sha256,
    seed_for,
)

ROOT = Path(__file__).resolve().parents[2]

HARD_INTENTS = {
    'BRIGHTNESS', 'CREATE_REMINDER', 'LIST_REMINDERS',
    'CALL', 'VOLUME', 'LIGHT_ON', 'LIGHT_OFF', 'TEMPERATURE',
    'PLAY_MUSIC', 'STOP', 'NEXT', 'PAUSE'
}


class SynthFinetunePools:
    def __init__(self, records: list[dict]):
        self.human_pos = []
        self.human_neg = []
        self.synth_hard = []
        self.synth_standard = []
        
        for r in records:
            if not r.get('training_eligible', False):
                continue
            if r.get('synthetic', False):
                exp = r.get('expected')
                if exp:
                    intent = exp.get('intent')
                    if intent in HARD_INTENTS:
                        self.synth_hard.append(r)
                    else:
                        self.synth_standard.append(r)
            else:
                if r.get('expected') is not None:
                    self.human_pos.append(r)
                else:
                    self.human_neg.append(r)
                    
        print(f'SynthFinetunePools: {len(self.synth_hard)} hard synth, {len(self.synth_standard)} standard synth, '
              f'{len(self.human_pos)} human pos, {len(self.human_neg)} human neg.')

    def sample_batch(self, rng: np.random.Generator, size: int = 32) -> list[dict]:
        # 22 synth (68.75%), 5 human pos (15.6%), 5 human neg (15.6%)
        n_synth = 22
        n_pos = 5
        n_neg = size - n_synth - n_pos
        
        # Synth split: 14 hard synth, 8 standard synth
        hard_idx = rng.choice(len(self.synth_hard), size=14, replace=True)
        std_idx = rng.choice(len(self.synth_standard), size=8, replace=True)
        synth_samples = [self.synth_hard[i] for i in hard_idx] + [self.synth_standard[i] for i in std_idx]
        
        pos_idx = rng.choice(len(self.human_pos), size=n_pos, replace=True)
        neg_idx = rng.choice(len(self.human_neg), size=n_neg, replace=True)
        pos_samples = [self.human_pos[i] for i in pos_idx]
        neg_samples = [self.human_neg[i] for i in neg_idx]
        
        batch = synth_samples + pos_samples + neg_samples
        rng.shuffle(batch)
        return batch


def export_finetuned_bundle(model: torch.nn.Module, *, output_dir: Path, base_bundle: Path, device: str = 'cuda'):
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

    # Write calibrated conforming policy
    calib_thresholds = {
        'ALARM': 0.8,
        'BRIGHTNESS': 0.5,
        'CALL': 0.5,
        'COLOR': 0.5,
        'CREATE_REMINDER': 0.5,
        'LIGHT_OFF': 0.6,
        'LIGHT_ON': 0.6,
        'LIST_REMINDERS': 0.5,
        'MESSAGE': 0.5,
        'NEXT': 0.5,
        'PAUSE': 0.8,
        'PLAY_MUSIC': 0.6,
        'STOP': 0.7,
        'TEMPERATURE': 0.5,
        'TIME': 0.5,
        'TIMER': 0.85,
        'VOLUME': 0.5,
        'WEATHER': 0.8,
    }
    
    recall_policy = json.loads((base_bundle / 'recall-policy.json').read_text(encoding='utf-8'))
    recall_policy['binding'] = binding
    recall_policy['intent_thresholds'] = calib_thresholds
    recall_policy['slot_threshold'] = 0.9
    validate_policy(recall_policy, binding)
    
    policy_path = output_dir / 'recall-policy.json'
    policy_bytes = (json.dumps(recall_policy, indent=2) + '\n').encode('utf-8')
    policy_path.write_bytes(policy_bytes)

    recipe = json.loads((base_bundle / 'recipe.json').read_text(encoding='utf-8'))
    recipe['binding'] = binding
    recipe['recall_policy_sha256'] = hashlib.sha256(policy_bytes).hexdigest()
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


def finetune_synthetic(
    *,
    output_dir: Path,
    init_checkpoint: Path,
    prepared_dir: Path = PREPARED,
    features_dir: Path = FEATURES_DIR,
    epochs: int = 14,
    steps_per_epoch: int = 64,
    batch_size: int = 32,
    microbatch: int = 8,
    initial_lr: float = 3.5e-5,
    final_lr: float = 1.0e-5,
    device: str | None = None,
    seed: int = 77,
    limit_eval: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = configure_training_device()
    print(f'Fine-tuning device: {device}')

    train_rows = [json.loads(line) for line in (prepared_dir / 'train.jsonl').open(encoding='utf-8')]
    dev_rows = [json.loads(line) for line in (prepared_dir / 'dev.jsonl').open(encoding='utf-8')]
    test_rows = [json.loads(line) for line in (prepared_dir / 'test.jsonl').open(encoding='utf-8')]

    norm = json.loads((BASE_BUNDLE / 'normalization.json').read_text(encoding='utf-8'))
    mean = np.asarray(norm['mean'], dtype=np.float32)[:, None]
    std = np.asarray(norm['std'], dtype=np.float32)[:, None]

    entries = json.loads((BASE_BUNDLE / 'catalog.json').read_text(encoding='utf-8'))['entries']
    decoder = PrefixCTC([e['phones'] for e in entries])

    corrections = load_label_corrections()
    clean_train_rows = apply_corrections_and_filter(train_rows, corrections, is_train=True)

    pools = SynthFinetunePools(clean_train_rows)

    print('Loading normalized feature arrays into memory...')
    t0 = time.time()
    needed = {r['id']: r['features'] for r in clean_train_rows + dev_rows}
    def _load_one(item):
        rid, path_str = item
        p = features_dir / Path(path_str).name
        raw = np.load(p).astype(np.float32)
        return rid, (raw - mean) / std
    with ThreadPoolExecutor(max_workers=16) as pool:
        arrays = dict(pool.map(_load_one, needed.items()))
    print(f'Loaded {len(arrays)} feature arrays in {time.time() - t0:.2f}s.')

    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = PooledContextVCM(**ARCHITECTURE).to(device)

    print(f'Loading warm-start checkpoint from {init_checkpoint}...')
    cp = torch.load(init_checkpoint, map_location=device)
    state_dict = cp.get('weights') or cp.get('model_state_dict') or cp
    model.load_state_dict(state_dict)
    print('Checkpoint weights successfully loaded.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    warmup_steps = steps_per_epoch
    total_steps = epochs * steps_per_epoch
    decay_steps = max(1, total_steps - warmup_steps)

    best_score = None
    best_epoch = None
    best_weights = None
    checkpoint_path = output_dir / 'checkpoint.pt'

    # Initial dev evaluation
    print('Evaluating initial dev performance before fine-tuning...')
    dev_acc, dev_fa, d_cor, d_pos, d_fa, d_neg = evaluate_dev(model, dev_rows, arrays, entries, decoder, device)
    best_score = 1.0 + dev_acc - 0.2 * dev_fa
    best_epoch = 0
    best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f'Initial Checkpoint: Dev Exact Acc={d_cor}/{d_pos} ({dev_acc:.2%}) FA={d_fa}/{d_neg} ({dev_fa:.2%}) Score={best_score:.4f}')

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t_ep = time.time()
        epoch_losses = []
        for step in range(steps_per_epoch):
            global_step = (epoch - 1) * steps_per_epoch + step
            if global_step < warmup_steps:
                lr = initial_lr * (global_step + 1) / max(1, warmup_steps)
            else:
                progress = min(1.0, (global_step - warmup_steps) / decay_steps)
                lr = final_lr + 0.5 * (initial_lr - final_lr) * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group['lr'] = lr

            batch = pools.sample_batch(np_rng, batch_size)
            step_loss = optimization_step(
                model, batch, arrays, optimizer, device=device,
                seed=seed_for(seed, epoch, step),
                augment_training=False, microbatch=microbatch
            )
            epoch_losses.append(step_loss)

        losses = np.mean(epoch_losses, axis=0).tolist()
        current_lr = optimizer.param_groups[0]['lr']

        dev_acc, dev_fa, d_cor, d_pos, d_fa, d_neg = evaluate_dev(model, dev_rows, arrays, entries, decoder, device)
        score = 1.0 + dev_acc - 0.2 * dev_fa
        if dev_fa <= 0.02:
            score += 0.5
            
        eval_str = f'Dev Exact Acc={d_cor}/{d_pos} ({dev_acc:.2%}) FA={d_fa}/{d_neg} ({dev_fa:.2%}) Score={score:.4f}'
        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'weights': best_weights, 'architecture': ARCHITECTURE, 'epoch': epoch,
                        'score': score, 'dev_acc': dev_acc, 'dev_fa': dev_fa}, checkpoint_path)
            eval_str += ' [BEST]'

        print(f'Epoch {epoch:02d} {time.time() - t_ep:.1f}s lr={current_lr:.6f} '
              f'Loss=[CTC:{losses[0]:.3f}, INT:{losses[1]:.3f}, TGT:{losses[2]:.3f}] {eval_str}',
              flush=True)

    print(f'Fine-tuning finished in {time.time() - t_start:.1f}s. Best Epoch: {best_epoch} (score={best_score:.4f})')

    # Export bundle
    model.load_state_dict(best_weights)
    model.eval()
    export_finetuned_bundle(model, output_dir=output_dir, base_bundle=BASE_BUNDLE, device=device)
    print(f'Exported bundle to {output_dir}')

    # Evaluate full test bench
    eval_rows = test_rows if limit_eval is None else test_rows[:limit_eval]
    print(f'Evaluating {len(eval_rows)} test samples with CompactRuntime...')
    runtime = CompactRuntime(output_dir, threads=8, engine='onnxruntime', mode='demo')

    pos_count = 0
    pos_exact = 0
    pos_acc_demo = 0
    synth_count = 0
    synth_exact = 0
    synth_acc_demo = 0
    neg_count = 0
    neg_fa_demo = 0

    t_eval = time.time()
    for idx, row in enumerate(eval_rows, 1):
        pcm, sr = sf.read(row['audio'], dtype='float32')
        ev = runtime.recognize(pcm, sr)
        cand = (ev.get('diagnostic_candidate') or {}).get('candidate')
        app = ev.get('application', {})
        exp = row.get('expected')
        
        if exp is not None:
            is_synth = bool(row.get('synthetic'))
            match = (cand is not None and canonical(cand) == canonical(exp))
            demo_accepted = bool(app.get('accepted') and app.get('command') is not None and canonical(app.get('command')) == canonical(exp))
            if is_synth:
                synth_count += 1
                if match:
                    synth_exact += 1
                if demo_accepted:
                    synth_acc_demo += 1
            else:
                pos_count += 1
                if match:
                    pos_exact += 1
                if demo_accepted:
                    pos_acc_demo += 1
        else:
            neg_count += 1
            if app.get('accepted'):
                neg_fa_demo += 1
                
        if idx % 1000 == 0:
            print(f'Evaluated {idx}/{len(eval_rows)} in {time.time() - t_eval:.1f}s...', flush=True)

    results = {
        'human_exact': pos_exact,
        'human_total': pos_count,
        'human_raw_accuracy': pos_exact / pos_count if pos_count else 0.0,
        'human_accepted_demo': pos_acc_demo,
        'human_accepted_recall_demo': pos_acc_demo / pos_count if pos_count else 0.0,
        'synthetic_exact': synth_exact,
        'synthetic_total': synth_count,
        'synthetic_raw_accuracy': synth_exact / synth_count if synth_count else 0.0,
        'synthetic_accepted_demo': synth_acc_demo,
        'synthetic_accepted_recall_demo': synth_acc_demo / synth_count if synth_count else 0.0,
        'negative_total': neg_count,
        'negative_fa_demo': neg_fa_demo,
        'negative_fa_rate_demo': neg_fa_demo / neg_count if neg_count else 0.0,
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
    print(f"Synthetic Demo Accepted Recall: {synth_acc_demo}/{synth_count} ({results['synthetic_accepted_recall_demo']:.2%})")
    print(f"Negative False Accepts (Demo): {neg_fa_demo}/{neg_count} ({results['negative_fa_rate_demo']:.2%})")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/speaker-targeted-v10')
    parser.add_argument('--init-checkpoint', type=Path, default=ROOT / 'experiments/speaker-targeted-v7/checkpoint.pt')
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--steps-per-epoch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3.5e-5)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    finetune_synthetic(
        output_dir=args.output_dir,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        initial_lr=args.lr,
        device=args.device,
    )
