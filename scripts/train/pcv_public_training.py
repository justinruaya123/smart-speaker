"""Matched public-speech fitting with immutable development selections."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from einops import rearrange, reduce

from scripts.models.pcv_residual import ResidualVCM, export_onnx
from scripts.train.pcv_public_data import ROOT, STUDY, CACHE, sha, save, read_rows
from scripts.test.pcv_public_eval import normalized, score, metric_summary, calibrate, accepted_candidate
from scripts.runtime.pcv_schema import canonical


def selection_rows(rows, config):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['source'], row['intent'], row['expected'] is not None)].append(row)
    selected = []
    for bucket in sorted(buckets):
        values = sorted(buckets[bucket], key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        n = config['synthetic_selection_per_source_intent'] if values[0]['synthetic'] else config['human_selection_per_source_intent']
        selected.extend(values[:n])
    return selected


def pooled_logits(logits, lengths):
    mask = torch.arange(logits.shape[2], device=logits.device)[None, :] < lengths[:, None]
    return reduce(logits * rearrange(mask, 'b t -> b 1 t'), 'b c t -> b c', 'sum') / rearrange(lengths, 'b -> b 1')


class CyclingRows:
    def __init__(self, rows, generator):
        self.rows, self.generator = rows, generator
        self.order, self.cursor = [], 0

    def take(self, n):
        result = []
        while len(result) < n:
            if self.cursor == len(self.order):
                self.order = self.generator.permutation(len(self.rows)).tolist()
                self.cursor = 0
            index = self.order[self.cursor]
            self.cursor += 1
            result.append(self.rows[index])
        return result


def freeze_training():
    path = STUDY / 'training-freeze.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for file, expected in existing['source_hashes'].items():
            if sha(ROOT/file) != expected:
                raise ValueError('Frozen training source changed: ' + file)
        return existing
    data = json.loads((CACHE/'prepared/freeze.json').read_text())
    files = ['scripts/runtime/pcv_schema.py', 'scripts/models/pcv_residual.py', 'scripts/train/pcv_public_data.py',
        'scripts/train/pcv_public_training.py', 'scripts/test/pcv_public_eval.py', 'scripts/pcv_public.py',
        'scripts/pcv_parallel.py', 'experiments/phrase-contract-v1/pcvcm/decoder.py', 'vcm/frontend.py']
    result = {'protocol_sha256': sha(STUDY/'protocol.json'), 'data_freeze_sha256': sha(STUDY/'data-freeze.json'),
        'source_hashes': {p: sha(ROOT/p) for p in files},
        'installed_artifact_hashes': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()},
        'test_inference_accessed': False, 'initialization': 'random', 'data': data,
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}
    save(path, result)
    return result


def load_arrays(rows, norm):
    arrays = {}
    for index, row in enumerate(rows, 1):
        if sha(row['features']) != row['features_sha256']:
            raise ValueError('Feature identity mismatch: ' + row['id'])
        arrays[row['id']] = normalized(row, norm)
        if index % 5000 == 0:
            print('LOAD_FEATURES', index, '/', len(rows), flush=True)
    return arrays


def train_stage(name, initial, train_rows, development, arrays, norm, entries, protocol, epochs):
    cfg = protocol['training']
    directory = CACHE/'training'/name
    if directory.exists():
        raise FileExistsError('Training stage exists; preserve it and use explicit resume')
    directory.mkdir(parents=True)
    seed = protocol['seed'] + (0 if name == 'foundation' else 100)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    device = torch.device(cfg['device'])
    model = ResidualVCM(**{k: protocol['model'][k] for k in ('width','dilations')}).to(device)
    if initial:
        checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['weights'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'] if name == 'foundation' else cfg['continuation_learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00002)
    human = [r for r in train_rows if not r['synthetic']]
    sources = {'human': CyclingRows(human, generator),
               'qwen': CyclingRows([r for r in train_rows if r['source']=='Qwen'], generator),
               'sapi': CyclingRows([r for r in train_rows if r['source']=='SAPI'], generator)}
    steps = math.ceil(len(human)/cfg['batch_size'])
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    history, best, best_key = [], None, None
    for epoch in range(1, epochs+1):
        started = time.perf_counter()
        model.train()
        sums = np.zeros(3)
        for step in range(steps):
            if name == 'mixed':
                batch = sources['human'].take(36) + sources['qwen'].take(6) + sources['sapi'].take(6)
            else:
                batch = sources['human'].take(cfg['batch_size'])
            lengths = [arrays[r['id']].shape[1] for r in batch]
            x = np.zeros((len(batch), 40, max(lengths)), dtype='float32')
            for i, row in enumerate(batch):
                x[i, :, :lengths[i]] = arrays[row['id']]
            ctc, intent, target, _ = model(torch.from_numpy(x).to(device))
            length_tensor = torch.tensor(lengths, device=device)
            acoustic = ctc_loss(rearrange(ctc.log_softmax(1), 'b c t -> t b c'),
                torch.tensor([p for r in batch for p in r['phones']], device=device, dtype=torch.long),
                torch.tensor(lengths, dtype=torch.long), torch.tensor([len(r['phones']) for r in batch], dtype=torch.long))
            semantic = torch.nn.functional.cross_entropy(pooled_logits(intent, length_tensor), torch.tensor([r['intent_index'] for r in batch], device=device))
            targets = torch.tensor([r['target_index'] for r in batch], device=device)
            target_loss = torch.nn.functional.cross_entropy(pooled_logits(target, length_tensor), targets, ignore_index=-100) if torch.any(targets != -100) else target.sum()*0
            loss = acoustic + cfg['intent_loss_weight']*semantic + cfg['target_loss_weight']*target_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            optimizer.step()
            sums += [float(acoustic.detach()), float(semantic.detach()), float(target_loss.detach())]
            if (step+1) % 200 == 0:
                print(json.dumps({'stage': name, 'epoch': epoch, 'step': step+1, 'steps': steps, 'seconds': round(time.perf_counter()-started,1)}), flush=True)
        scheduler.step()
        row = {'stage': name, 'epoch': epoch, 'steps': steps, 'seconds': time.perf_counter()-started,
               'ctc_loss': sums[0]/steps, 'intent_loss': sums[1]/steps, 'target_loss': sums[2]/steps,
               'gpu_peak_bytes': torch.cuda.max_memory_allocated()}
        checkpoint = {'weights': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                      'model_config': model.config, 'epoch': epoch, 'stage': name, 'seed': seed,
                      'generator_state': generator.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all(),
                      'sampling': {k: {'order': v.order, 'cursor': v.cursor} for k,v in sources.items()},
                      'training_freeze_sha256': sha(STUDY/'training-freeze.json')}
        torch.save(checkpoint, directory/'last.pt')
        if epoch % cfg['development_every_epochs'] == 0:
            subset = [r for r in development if not r['synthetic']] if name == 'foundation' else development
            records = score(model, subset, norm, entries, device, arrays=arrays)
            human_metrics = metric_summary([r for r in records if not r['synthetic']])['all']
            synthetic_metrics = metric_summary([r for r in records if r['synthetic']])['all']
            if name == 'foundation':
                key = (human_metrics['exact'], -human_metrics['negative_errors'], -epoch)
            else:
                key = (((human_metrics['recall'] or 0) + (synthetic_metrics['recall'] or 0))/2,
                       -human_metrics['negative_errors']-synthetic_metrics['negative_errors'], -epoch)
            row['development'] = {'human': human_metrics, 'synthetic': synthetic_metrics}
            row['selected'] = best_key is None or key > best_key
            save(directory/f'dev-{epoch:03d}.json', records)
            save(directory/f'metrics-{epoch:03d}.json', metric_summary(records))
            checkpoint_path = directory/f'epoch-{epoch:03d}.pt'
            torch.save(checkpoint, checkpoint_path)
            if row['selected']:
                best_key = key
                best = {'stage': name, 'epoch': epoch, 'checkpoint': str(checkpoint_path.relative_to(ROOT)),
                        'checkpoint_sha256': sha(checkpoint_path), 'development_key': list(key), 'development': row['development'],
                        'test_inference_accessed': False, 'initial_checkpoint': str(initial) if initial else None}
                save(directory/'selection.json', best)
        history.append(row)
        save(directory/'history.json', history)
        print(json.dumps(row), flush=True)
    return best


def train():
    freeze = freeze_training()
    protocol = json.loads((STUDY/'protocol.json').read_text())
    if sha(STUDY/'protocol.json') != freeze['protocol_sha256']:
        raise ValueError('Training protocol changed')
    for entry in freeze['data']['manifests'].values():
        if sha(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen manifest changed')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35)
    train_rows = read_rows(CACHE/'prepared/train.jsonl')
    development = selection_rows(read_rows(CACHE/'prepared/dev.jsonl'), protocol['training'])
    save(STUDY/'development-selection.json', {'ids': [r['id'] for r in development], 'method': 'predeclared deterministic source/intent/positive strata'})
    norm = json.loads((CACHE/'prepared/normalization.json').read_text())
    entries = json.loads((STUDY/'catalog.json').read_text())['entries']
    arrays = load_arrays(train_rows+development, norm)
    base = train_stage('foundation', None, train_rows, development, arrays, norm, entries, protocol, protocol['training']['foundation_epochs'])
    selections = {}
    for arm in ('human','mixed'):
        selections[arm] = train_stage(arm, ROOT/base['checkpoint'], train_rows, development, arrays, norm, entries, protocol, protocol['training']['continuation_epochs'])
    # Fixed dev-only choice is recorded before any calibration/test acoustic inference.
    winner = max(selections, key=lambda arm: selections[arm]['development_key'])
    save(STUDY/'selection-before-test.json', {'foundation': base, 'arms': selections, 'winner': winner,
        'training_freeze_sha256': sha(STUDY/'training-freeze.json'), 'test_inference_accessed': False, 'calibration_inference_accessed': False})
    print('TRAINING_COMPLETE', winner, flush=True)


def evaluate():
    import onnxruntime as ort
    import shutil
    freeze_training()
    selection = json.loads((STUDY/'selection-before-test.json').read_text())
    protocol = json.loads((STUDY/'protocol.json').read_text())
    norm = json.loads((CACHE/'prepared/normalization.json').read_text())
    catalog = json.loads((STUDY/'catalog.json').read_text())
    cal = read_rows(CACHE/'prepared/calibration.jsonl')
    test = read_rows(CACHE/'prepared/test.jsonl')
    results = {}
    for arm, selected in selection['arms'].items():
        directory = CACHE/'evaluation'/arm
        if directory.exists():
            raise FileExistsError('Evaluation output already exists; preserve sealed evidence')
        directory.mkdir(parents=True)
        checkpoint = ROOT/selected['checkpoint']
        if sha(checkpoint) != selected['checkpoint_sha256']:
            raise ValueError('Selected checkpoint identity mismatch')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model = ResidualVCM(**payload['model_config'])
        model.load_state_dict(payload['weights'])
        bundle = directory/'bundle'
        bundle.mkdir()
        export = export_onnx(model, bundle/'model.onnx')
        save(directory/'export.json', export)
        if not export['pass']:
            raise ValueError('Candidate ONNX export gate failed; retain unqualified graph and receipt')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(bundle/'model.onnx'), options, providers=['CPUExecutionProvider'])
        calibration = score(session, cal, norm, catalog['entries'])
        save(directory/'calibration-predictions.json', calibration)
        policy = calibrate(calibration)
        present = {r['expected']['intent'] for r in calibration if r['expected']}
        from scripts.runtime.pcv_schema import INTENTS
        policy['uncalibrated_intents'] = sorted(set(INTENTS)-present)
        binding = {'model_sha256': sha(bundle/'model.onnx'), 'catalog_sha256': sha(STUDY/'catalog.json'),
                   'normalization_sha256': sha(CACHE/'prepared/normalization.json')}
        policy['binding'] = binding
        save(bundle/'policy.json', policy)
        shutil.copyfile(STUDY/'catalog.json', bundle/'catalog.json')
        shutil.copyfile(CACHE/'prepared/normalization.json', bundle/'normalization.json')
        save(directory/'selection-before-test.json', {'checkpoint': selected, 'binding': binding, 'policy_sha256': sha(bundle/'policy.json'),
            'test_manifest_sha256': sha(CACHE/'prepared/test.jsonl'), 'calibration_predictions_sha256': sha(directory/'calibration-predictions.json'),
            'test_inference_accessed': False})
        predicted = score(session, test, norm, catalog['entries'])
        for r in predicted:
            candidate = accepted_candidate(r, policy)
            if candidate and candidate['intent'] in policy['uncalibrated_intents']:
                candidate = None
            r['policy_candidate'] = candidate
            r['accepted_command'] = candidate if policy['enabled'] else None
        save(directory/'test-predictions.json', predicted)
        baseline = [r for r in predicted if (r.get('phrase_id') or '').startswith('B1-')]
        slots = defaultdict(lambda: {'n': 0, 'correct': 0})
        for r in predicted:
            if r['expected']:
                for role, value in r['expected']['slots'].items():
                    s = slots[r['source']+'/'+r['expected']['intent']+'/'+role]
                    s['n'] += 1
                    s['correct'] += int(r['candidate'] is not None and r['candidate']['intent'] == r['expected']['intent'] and
                                        canonical(r['candidate']['slots'].get(role)) == canonical(value))
        report = {'arm': arm, 'selection': selected, 'export': export, 'policy': policy,
            'raw': metric_summary(predicted), 'policy_operating_point': metric_summary(predicted, 'policy_candidate'),
            'accepted': metric_summary(predicted, 'accepted_command'), 'baseline_sapi_raw': metric_summary(baseline),
            'baseline_by_phrase': [{'id': r['phrase_id'], 'source': r['source'], 'speaker': r['speaker'],
                 'expected': r['expected'], 'candidate': r['candidate'], 'raw_exact': canonical(r['expected']) == canonical(r['candidate'])} for r in baseline],
            'raw_slot_accuracy': dict(slots), 'held_out_original_human': True, 'physical_pi_measured': False,
            'limitations': ['Generated spoken text not independently listened', 'New human speaker split over previously studied public data',
                           'SAPI baseline has one test voice of one engine; not full cross-engine qualification',
                           'Raw rankings are not calibrated accepted commands', 'No installed artifact or device action']}
        save(STUDY/f'results-{arm}.json', report)
        save(bundle/'recipe.json', {'schema_version': 3, 'status': 'research_unqualified', 'state_shapes': model.state_shapes,
             'heads': ['ctc','intent','target'], 'binding': binding, 'selection_sha256': sha(directory/'selection-before-test.json'),
             'installation_authorized': False, 'policy_enabled': policy['enabled']})
        results[arm] = {'raw': report['raw']['all'], 'baseline_sapi_raw': report['baseline_sapi_raw']['all'], 'policy_enabled': policy['enabled']}
        print('TEST_RESULT', arm, json.dumps(results[arm]), flush=True)
    save(STUDY/'comparison.json', {'development_selected_arm': selection['winner'], 'arms': results,
         'test_selected_weights_or_policy': False})
