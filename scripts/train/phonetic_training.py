"""Train phonetic CTC from random weights; leave installed runtime untouched."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from einops import rearrange, pack
from .phonetic_data import ROOT, DATA, OUT, prepare, waveform
from .parameter_study_data import dump, digest, lexicon, phones
from .parameter_study import collate, decode, export
from .stt import BrightnessSTT
from .stt_runtime import stt_features


def phone_edit(a, b):
    """Exact global Levenshtein, using Myers bit vectors for long dev speech."""
    if not a:
        return len(b)
    masks = {}
    for i, token in enumerate(a):
        masks[token] = masks.get(token, 0) | (1 << i)
    positive, negative, score = ~0, 0, len(a)
    high = 1 << (len(a)-1)
    for token in b:
        equal = masks.get(token, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(horizontal | positive)
        mh = positive & horizontal
        if ph & high:
            score += 1
        if mh & high:
            score -= 1
        ph = (ph << 1) | 1
        positive = (mh << 1) | ~(vertical | ph)
        negative = ph & vertical
    return score


def features(rows):
    from . import stt_runtime
    key = digest(DATA/'manifest.jsonl')[:16] + '-' + digest(__import__('pathlib').Path(stt_runtime.__file__))[:12]
    index = OUT / ('features-' + key + '.json')
    path = index.with_suffix('.npy')
    if not index.exists():
        sizes = [math.ceil(r['duration']*16000 - 1e-6) // 160 + 1 for r in rows]
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        array = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=(offsets[-1], 80))
        def compute(i):
            return i, stt_features(waveform(rows[i]))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for first in range(0, len(rows), 512):
                for i, value in pool.map(compute, range(first, min(first+512, len(rows)))):
                    if len(value) != sizes[i]:
                        raise ValueError('Feature frame accounting changed')
                    array[offsets[i]:offsets[i+1]] = value
                if first % 2048 == 0:
                    print('v2 original waveform features', first, '/', len(rows), flush=True)
        array.flush()
        dump(index, dict(offsets=offsets, manifest_sha256=digest(DATA/'manifest.jsonl'), dtype='float32'))
    offsets = json.loads(index.read_text())['offsets']
    array = np.load(path, mmap_mode='c')
    return [array[offsets[i]:offsets[i+1]] for i in range(len(rows))]


def train_one(name, rows, cache, epochs=36, no_cudnn=False):
    width = {'small': 64, 'wide': 96}[name]
    seed = {'small': 41, 'wide': 42}[name]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('The recorded protocol uses CUDA fitting')
    torch.backends.cudnn.enabled = not no_cudnn
    directory = OUT / name
    directory.mkdir(exist_ok=True)
    dictionary = lexicon()
    vocabulary = ['<blank>'] + sorted({p for value in dictionary.values() for p in value} | {'|'})
    lookup = {p: i for i, p in enumerate(vocabulary)}
    targets = {}
    for i, r in enumerate(rows):
        target = phones(r['text'], dictionary)
        if target is not None and len(target) + sum(a == b for a, b in zip(target, target[1:])) <= (len(cache[i])+1)//2:
            targets[i] = [lookup[p] for p in target]
    training = [i for i, r in enumerate(rows) if r['split'] == 'train' and i in targets]
    development = [i for i, r in enumerate(rows) if r['split'] == 'dev' and i in targets]
    words = [i for i in training if rows[i]['kind'] == 'word']
    commands = [i for i in training if rows[i]['kind'] != 'word' and rows[i]['targets']]
    model = BrightnessSTT(len(vocabulary), width=width, layers=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00015)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=False)
    history = []
    best = float('inf')
    bad = 0
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        begin = time.perf_counter()
        order = list(words if epoch <= 2 else training + commands * 3)
        random.shuffle(order)
        # Random buckets retain stochasticity while limiting padded work.
        order = [i for offset in range(0, len(order), 512) for i in sorted(order[offset:offset+512], key=lambda i: len(cache[i]), reverse=True)]
        total = 0
        steps = 0
        for offset in range(0, len(order), 64):
            x, lengths, indices = collate(order[offset:offset+64], cache)
            y, _ = pack([torch.tensor(targets[i], dtype=torch.long) for i in indices], '*')
            yl = torch.tensor([len(targets[i]) for i in indices], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits, output_lengths = model(x.cuda(), lengths.cuda())
            loss = loss_fn(rearrange(logits.log_softmax(-1), 'b t v -> t b v'), y.cuda(), output_lengths.cpu(), yl)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CTC loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        model.eval()
        totals = defaultdict(lambda: [0, 0])
        with torch.inference_mode():
            for offset in range(0, len(development), 64):
                x, lengths, indices = collate(development[offset:offset+64], cache)
                logits, output_lengths = model(x.cuda(), lengths.cuda())
                logits = logits.cpu().numpy()
                for j, i in enumerate(indices):
                    pred = decode(logits[j], int(output_lengths[j]), vocabulary)
                    ref = [vocabulary[k] for k in targets[i]]
                    group = 'word' if rows[i]['kind'] == 'word' else 'command' if rows[i]['targets'] else 'general'
                    totals[group][0] += phone_edit(ref, pred)
                    totals[group][1] += len(ref)
        rates = {k: a/max(1, n) for k, (a, n) in totals.items()}
        score = sum(rates.values())/len(rates)
        entry = dict(epoch=epoch, loss=total/max(1, steps), dev_phone_error=rates, selection_error=score, seconds=time.perf_counter()-begin)
        history.append(entry)
        dump(directory/'history.json', history)
        print(name, json.dumps(entry), flush=True)
        if epoch > 2 and score < best:
            best = score
            bad = 0
            torch.save(dict(state_dict=model.state_dict(), config=model.config, vocabulary=vocabulary, epoch=epoch,
                            seed=seed, initialization='random_from_scratch', manifest_sha256=digest(DATA/'manifest.jsonl'),
                            development_error=rates, selection_error=score, training_examples=len(training),
                            dev_examples=len(development)), directory/'phoneme.pt')
        else:
            bad += 1
        scheduler.step()
        if bad >= 10 and epoch >= 15:
            break
    dump(directory/'fit.json', dict(epochs_run=len(history), fit_seconds=time.perf_counter()-started,
                                   command_sampling_weight=4, word_warmup_epochs=2, cudnn_enabled=not no_cudnn))
    # Keep CUDA fitting and ONNX Runtime's CPU thread-pool lifetime separate.
    # On this Windows runtime the mixed process completed its artifacts but
    # aborted natively at shutdown. A fresh CPU worker also makes export retryable.
    subprocess.run([sys.executable, str(ROOT/'scripts/phonetic.py'), 'export', '--models', name], check=True)
    return json.loads((directory/'training.json').read_text())


def export_saved(name):
    directory = OUT/name
    saved = torch.load(directory/'phoneme.pt', weights_only=False, map_location='cpu')
    if saved['manifest_sha256'] != digest(DATA/'manifest.jsonl'):
        raise ValueError('Saved model does not belong to this manifest')
    if (directory/'fit.json').exists():
        fit = json.loads((directory/'fit.json').read_text())
    else:
        history = json.loads((directory/'history.json').read_text())
        if history[-1]['epoch'] < saved['epoch']:
            raise ValueError('Incomplete fitting history')
        fit = dict(epochs_run=len(history), fit_seconds=sum(h['seconds'] for h in history),
                   command_sampling_weight=4, word_warmup_epochs=2, cudnn_enabled=True, recovered_from_completed_history=True)
    torch.set_num_threads(2)
    model = BrightnessSTT(**saved['config']).eval()
    model.load_state_dict(saved['state_dict'])
    started = time.perf_counter()
    parity = export(model, 'phoneme', output_dir=directory, parity_lengths=([11], [95], [419], [801], [2001], [2001, 801, 95]))
    result = {k: v for k, v in saved.items() if k != 'state_dict'}
    result.update(fit)
    result.update(export_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
                  graph_bytes=(directory/'phoneme.onnx').stat().st_size, graph_sha256=digest(directory/'phoneme.onnx'),
                  onnx_parity=parity, export_process='fresh_cpu_worker')
    result['seconds'] = result['fit_seconds'] + result['export_seconds']
    dump(directory/'training.json', result)
    print('exported', name, 'from selected epoch', saved['epoch'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'export', 'evaluate', 'scale', 'check'])
    parser.add_argument('--models', choices=['small', 'wide'], nargs='+', default=['small', 'wide'])
    parser.add_argument('--epochs', type=int, default=36)
    parser.add_argument('--no-cudnn', action='store_true', help='slower CUDA kernels; avoids the observed Windows cuDNN GRU shutdown fault')
    args = parser.parse_args()
    if args.action == 'check':
        from .phonetic_eval import checks
        print(json.dumps(checks()), flush=True)
        return
    if args.action == 'export':
        for name in args.models:
            export_saved(name)
        return
    rows = prepare()
    if args.action == 'prepare':
        return
    if args.action == 'scale':
        from .phonetic_eval import scale
        scale(rows)
        return
    cache = features(rows)
    if args.action == 'train':
        if args.epochs < 3:
            raise ValueError('Need at least three epochs including word warmup')
        for name in args.models:
            train_one(name, rows, cache, args.epochs, args.no_cudnn)
    else:
        from .phonetic_eval import evaluate
        evaluate(rows, cache)
