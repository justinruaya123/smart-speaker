"""Additive, device-explicit OptionB recipe; the frozen CPU trainer is untouched.

This port does not promise CPU/GPU bit identity. The source fitter retains its
exact checkpoint/export comparisons and never replaces the frozen snapshot.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from scripts.models.speaker_optionb_tiny import OptionBTinyVCM
from scripts.train.speaker_optionb_tiny import (
    augmented, complete_class_metrics, normalized, tensor_batch, train_normalization, training_view,
)
from scripts.test.speaker_demo_suite import write_new

PORT_VERSION = 'optionb-device-explicit-v1'


class _DeviceAdamW(torch.optim.AdamW):
    """Unchanged AdamW update; CPU steps must not query CUDA graph capture."""
    def _cuda_graph_capture_health_check(self):
        # Torch 2.10's base hook queries CUDA even for all-CPU parameters in a
        # CUDA-built wheel. Guard only that hook; never patch Torch globals.
        if any(p.device.type == 'cuda' for group in self.param_groups for p in group['params']):
            super()._cuda_graph_capture_health_check()


def _probe_device(device):
    """Exercise the real architecture, backward pass and AdamW, not a device flag."""
    device = torch.device(device)
    torch.set_num_threads(4)
    with torch.enable_grad(), torch.autocast(device.type, enabled=False):
        model = OptionBTinyVCM().to(device).train()
        initial = next(model.parameters()).detach().clone()
        features = torch.linspace(-1., 1., 2 * 40 * 17, device=device).reshape(2, 40, 17)
        lengths = torch.tensor([17, 13], dtype=torch.long, device=device)
        targets = torch.tensor([1, 0], dtype=torch.long, device=device)
        optimizer = _DeviceAdamW(model.parameters(), lr=.002, weight_decay=.0001)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, lengths)
        loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=.02)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite device probe loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        changed = not torch.equal(initial, next(model.parameters()).detach())
        if not changed:
            raise ValueError('Device probe did not update model weights')
        return dict(passed=True, weights_changed=changed, optimizer_steps=1, finite_loss=True,
            model_devices=sorted({str(p.device) for p in model.parameters()}),
            input_devices=sorted({str(features.device), str(lengths.device)}),
            target_device=str(targets.device),
            gradient_devices=sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None}),
            optimizer_moment_devices=sorted({str(v.device) for state in optimizer.state.values()
                                            for k, v in state.items() if k != 'step' and torch.is_tensor(v)}),
            optimizer_step_devices=sorted({str(state['step'].device) for state in optimizer.state.values()}),
            synchronized=device.type == 'cuda')


def _configure_cuda():
    import os

    # Set before the first CUDA tensor allocation. Use only Torch 2.10's new
    # precision-control family, not the incompatible legacy allow_tf32 flags.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def select_device(requested='auto'):
    """Fallback is permitted only during this bounded preflight, never fitting."""
    if requested not in ('auto', 'cuda', 'cpu'):
        raise ValueError('Device must be auto, cuda or cpu')
    if requested == 'cpu':
        return dict(requested=requested, device='cpu', reason='explicit_cpu', probe=_probe_device('cpu'))
    reason, error = 'cuda_unavailable', None
    try:
        if torch.cuda.is_available():
            _configure_cuda()
            probe = _probe_device('cuda:0')
            return dict(requested=requested, device='cuda:0', reason='cuda_probe_passed', probe=probe,
                        cuda_device_name=torch.cuda.get_device_name(0))
    except Exception as exc:
        reason, error = 'cuda_preflight_failed', f'{type(exc).__name__}: {exc}'
    if requested == 'cuda':
        raise RuntimeError('Required CUDA is unavailable or failed preflight: ' + (error or reason))
    return dict(requested=requested, device='cpu', reason=reason, cuda_preflight_error=error,
                probe=_probe_device('cpu'))


def predict_logits(model, rows, arrays, norm, device, placement):
    """Canonical DEV sorting/batches; infer on the optimizer device, return CPU arrays."""
    result = np.zeros((len(rows), 32), dtype=np.float32)
    order = sorted(range(len(rows)), key=lambda i: arrays[rows[i]['id']].shape[1])
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, enabled=False):
        for start in range(0, len(order), 32):
            indices = order[start:start + 32]
            values = [normalized(arrays[rows[i]['id']], norm) for i in indices]
            features, lengths = (value.to(device) for value in tensor_batch(values))
            logits = model(features, lengths)
            placement.update(dev_model_devices=sorted({str(p.device) for p in model.parameters()}),
                dev_input_devices=sorted({str(features.device), str(lengths.device)}),
                dev_output_device=str(logits.device))
            result[indices] = logits.detach().cpu().numpy()
    return result


def fit_model(rows, dev, arrays, output, config, *, device='auto'):
    """v1 port of canonical fit_model: only device/seed guards, handoff and provenance.

    The quotas, RNG draw order, augmentations, loss, AdamW defaults, LR schedule,
    clipping, DEV tie-breaking and phase/checkpoint rules are the original recipe.
    CPU uses the same default CPU generator without torch.manual_seed's GPU calls.
    No training exception triggers a retry or switches optimizer device.
    """
    rows = training_view(rows, dev)
    if any(r['split'] != 'dev' for r in dev):
        raise ValueError('Checkpoint selection must use dev only')
    if torch.get_default_device().type != 'cpu' or torch.get_default_dtype() != torch.float32:
        raise ValueError('Fitting requires untouched CPU/FP32 Torch defaults')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() or (output / 'training-curve.jsonl').exists():
        raise FileExistsError('Preserve the earlier fit; choose a new run ID')
    selection = select_device(device)
    write_new(output / 'training-device.json', selection)
    device = torch.device(selection['device'])
    torch.set_num_threads(config['threads'])
    torch.random.default_generator.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(config['seed'])
    human = [r for r in rows if r['kind'] == 'human']
    norm = train_normalization(human, arrays)
    write_new(output / 'normalization.json', norm)
    model = OptionBTinyVCM().to(device)
    initial = next(model.parameters()).detach().clone()
    optimizer = _DeviceAdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    pools = {key: defaultdict(lambda: defaultdict(list)) for key in ('human_positive', 'human_negative', 'synthetic_positive')}
    for index, row in enumerate(rows):
        if row['kind'] == 'synthetic' and not row['class_index']:
            raise ValueError('This protocol has no synthetic-negative sampling arm')
        group = 'synthetic_positive' if row['kind'] == 'synthetic' else ('human_positive' if row['class_index'] else 'human_negative')
        key = row['class_index'] if row['class_index'] else row['scope']
        pools[group][key][str(row.get('speaker', 'tts_unknown'))].append(index)
    if any(not value for value in pools.values()):
        raise ValueError('Missing required human-positive, human-negative or synthetic-positive pool')

    def sample(group, count):
        choices = list(pools[group])
        result = []
        for _ in range(count):
            identities = pools[group][choices[int(rng.integers(len(choices)))]]
            speakers = list(identities)
            indices = identities[speakers[int(rng.integers(len(speakers)))]]
            result.append(indices[int(rng.integers(len(indices)))])
        return result

    best_key, selected_epoch, steps, draws = None, None, 0, Counter()
    last_loss = 0.
    placement = {}
    started = time.perf_counter()
    human_count, synthetic_count = config['human_epochs'], config['mixed_epochs']
    if config['batch_size'] != 32 or min(human_count, synthetic_count, config['steps_per_epoch']) < 1:
        raise ValueError('Expected positive phase lengths and the declared 32-example sampler')
    with torch.autocast(device.type, enabled=False), (output / 'training-curve.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1, human_count + synthetic_count + 1):
            mixed = epoch > human_count
            position = (epoch - human_count - 1) / max(1, synthetic_count - 1)
            lr = (.00005 + .5 * (config['learning_rate'] - .00005) * (1 + math.cos(math.pi * position))) if mixed else config['learning_rate']
            for group in optimizer.param_groups:
                group['lr'] = lr
            model.train()
            losses = []
            for _ in range(config['steps_per_epoch']):
                indices = sample('human_positive', 12 if mixed else 24) + sample('human_negative', 8)
                if mixed:
                    indices += sample('synthetic_positive', 12)
                rng.shuffle(indices)
                selected = [rows[i] for i in indices]
                values = [augmented(arrays[r['id']], norm, rng, r['kind'], config['context_max_frames']) for r in selected]
                targets = torch.tensor([r['class_index'] for r in selected], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                features, lengths = (value.to(device) for value in tensor_batch(values))
                logits = model(features, lengths)
                loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=config['label_smoothing'])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if not steps:
                    placement.update(model_devices=sorted({str(p.device) for p in model.parameters()}),
                        input_devices=sorted({str(features.device), str(lengths.device)}),
                        target_device=str(targets.device),
                        gradient_devices=sorted({str(p.grad.device) for p in model.parameters() if p.grad is not None}),
                        optimizer_moment_devices=sorted({str(v.device) for state in optimizer.state.values()
                            for k, v in state.items() if k != 'step' and torch.is_tensor(v)}),
                        optimizer_step_devices=sorted({str(state['step'].device) for state in optimizer.state.values()}))
                losses.append(float(loss.detach()))
                last_loss = losses[-1]
                draws.update(r['id'] for r in selected)
                steps += 1
            dev_logits = predict_logits(model, dev, arrays, norm, device, placement)
            metrics = {}
            for kind in ('human', 'synthetic'):
                keep = [i for i, r in enumerate(dev) if r['kind'] == kind and r['metric_eligible']]
                metrics[kind] = complete_class_metrics([dev[i] for i in keep], np.argmax(dev_logits[keep], axis=1))
            key = (metrics['human']['f1'] or 0., metrics['human']['precision'] or 0., metrics['synthetic']['recall'] or 0.)
            checkpoint = dict(state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                              epoch=epoch, config=config, sample_counts=dict(draws))
            if mixed and (best_key is None or key > best_key):
                best_key, selected_epoch = key, epoch
                torch.save(checkpoint, output / 'best.pt')
            if epoch == human_count:
                torch.save(checkpoint, output / 'human-foundation.pt')
            event = dict(epoch=epoch, phase='mixed' if mixed else 'human', loss=float(np.mean(losses)),
                         learning_rate=lr, dev=metrics, selected_epoch=selected_epoch,
                         elapsed_seconds=time.perf_counter() - started)
            log.write(json.dumps(event, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps(event, allow_nan=False), flush=True)
    result = dict(parameters=sum(p.numel() for p in model.parameters()), optimizer_steps=steps,
                  selected_epoch=selected_epoch, selected_on='dev', fitting_splits=['train'],
                  normalization_ids=norm['ids'], last_loss=last_loss,
                  weights_changed=not torch.equal(initial, next(model.parameters()).detach()),
                  elapsed_seconds=time.perf_counter() - started,
                  unique_sampled_by_kind={kind: sum(r['id'] in draws for r in rows if r['kind'] == kind) for kind in ('human', 'synthetic')},
                  sample_counts=dict(draws), training_config=config,
                  execution_device=str(device), checkpoint_tensor_device='cpu',
                  device_selection=selection, device_placement=placement, port_version=PORT_VERSION)
    write_new(output / 'training-result.json', result)
    return result
