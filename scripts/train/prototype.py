"""E18 training only. Fixed-capacity continuation; no export, test, or installation."""
import json
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .prototype_data import ROOT, EXP, DATA, CACHE, dataset, verify, sample_epoch
from .complete_data import CUES, cue_labels, digest, save
from .complete_training import load_features, make_batch
from .prototype_validation import score_development
from .slot_model import SlotVCM
from .train_helpers import seed_everything, batches
from .optimization import head_weights, balanced_losses


def train(arm):
    verify()
    protocol = json.loads((EXP / 'protocol.json').read_text(encoding='utf-8'))
    if arm not in protocol['arms']: raise ValueError('Undeclared arm')
    cfg = protocol['training']
    folder = CACHE / arm
    if folder.exists(): raise ValueError('Preserve the existing run; declare a new retry')
    checkpoint = ROOT / protocol['initial_checkpoint']
    if digest(checkpoint) != protocol['inputs'][protocol['initial_checkpoint']]:
        raise ValueError('Initial checkpoint changed')
    if not torch.cuda.is_available(): raise RuntimeError('Declared CUDA unavailable')
    folder.mkdir(parents=True)
    training, dev = dataset({'train'}), dataset({'dev', 'calibration'})
    seed_everything(cfg['seed'])
    metadata = json.loads((ROOT / protocol['initial_bundle']).read_text(encoding='utf-8'))
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SlotVCM(**ck['config']).to('cuda')
    model.load_state_dict(ck['model'], strict=True)
    for p in model.backbone.parameters(): p.requires_grad_(True)
    keywords = nn.Linear(model.heads.project[0].normalized_shape[0], len(CUES)).to('cuda')
    keywords.load_state_dict(torch.load(checkpoint.parent / 'keyword-training-only.pt', map_location='cpu', weights_only=True))
    cache = load_features(training + dev)
    vocab = ck['config']['vocabulary']
    annotations = [r['supervision'] for r in training]
    targets = {name: torch.tensor([values.index(a['labels'][name]) if a['labels'].get(name) is not None else 0 for a in annotations], device='cuda') for name, values in vocab.items()}
    masks = {name: torch.tensor([a['labels'].get(name) is not None for a in annotations], device='cuda') for name in vocab}
    families = torch.tensor([ck['labels'].index(r['command']) for r in training], device='cuda')
    cues = torch.tensor([cue_labels(r) for r in training], dtype=torch.float32, device='cuda')
    weights = head_weights(annotations, vocab, 'cuda')
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg['encoder_lr']},
        {'params': [p for p in model.heads.parameters() if p.requires_grad], 'lr': cfg['heads_lr']},
        {'params': keywords.parameters(), 'lr': cfg['heads_lr']}], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-5)

    def infer():
        outputs = [torch.empty(len(dev), n) for n in [len(ck['labels']), *map(len, vocab.values())]]
        model.eval()
        with torch.inference_mode():
            for ids in batches(dev, cfg['batch_size']):
                x, lengths = make_batch(dev, ids, cache, ck['normalization'])
                for out, value in zip(outputs, model(x.to('cuda'), lengths.to('cuda'))):
                    out[ids] = value.cpu()
        return tuple(outputs)

    run = {'arm': arm, 'training_rows': len(training), 'validation_rows': len(dev),
        'protocol_sha256': digest(EXP / 'protocol.json'), 'data_lock_sha256': digest(DATA / 'frozen.json'),
        'training_source_sha256': digest(__file__), 'initial_checkpoint_sha256': digest(checkpoint),
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'seed': cfg['seed'],
        'model_parameters': sum(p.numel() for p in model.parameters()), 'test_used': False,
        'annotation_quality': 'programmatic semantic overlay; not human adjudicated'}
    save(folder / 'run.json', run)
    baseline, _, _ = score_development(dev, infer(), metadata)
    save(folder / 'initial-development.json', baseline)
    history, best, stale = [], None, 0
    started = time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train(); keywords.train()
        ids_epoch = sample_epoch(training, cfg, random.Random(cfg['seed'] * 1000 + epoch), protocol['arms'][arm]['source_balanced'])
        losses = []
        for ids in batches(training, cfg['batch_size'], rng=random.Random(17000 + epoch), indices=ids_epoch, frame_budget=22000):
            x, lengths = make_batch(training, ids, cache, ck['normalization'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                encoded = model.encode(x.to('cuda'), lengths.to('cuda'))
                values = model.heads(*encoded)
                heads = balanced_losses(dict(zip(vocab, values[1:])), {k:v[ids] for k,v in targets.items()}, {k:v[ids] for k,v in masks.items()}, weights)
                loss = 2 * F.cross_entropy(values[0], families[ids]) + sum(.5 * v for v in heads.values())
                loss += cfg['keyword_loss_weight'] * F.binary_cross_entropy_with_logits(keywords(encoded[0]), cues[ids])
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        result, intent_policy, slot_policy = score_development(dev, infer(), metadata)
        entry = {'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.monotonic() - started, **result}
        history.append(entry)
        rank = tuple(result['rank'])
        if best is None or rank > best:
            best, stale = rank, 0
            selected = {**ck, 'config': model.config, 'model': {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'epoch': epoch, 'seed': cfg['seed'], 'intent_policy': intent_policy, 'slot_policy': slot_policy,
                'metrics': entry, 'training_mode': 'E18_' + arm}
            torch.save(selected, folder / 'vcm.pt')
        else:
            stale += 1
        save(folder / 'history.json', history)
        print(json.dumps({'arm': arm, 'epoch': epoch, 'dev_recall': result['commands']['exact_command_success'],
            'precision': result['commands']['accepted_command_precision'], 'unknown': result['intent']['unknown_false_accept_rate'],
            'best': best, 'seconds': round(entry['seconds'], 1)}), flush=True)
        if epoch >= cfg['minimum_epochs'] and stale >= cfg['patience']: break
    run.update(complete=True, epochs=len(history), selected_epoch=selected['epoch'], checkpoint_sha256=digest(folder / 'vcm.pt'))
    save(folder / 'run.json', run); save(EXP / (arm + '-training.json'), run)

