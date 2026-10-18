"""Last bounded refinement of the unchanged full finite-command VCM.

One shared human foundation and two matched mixed-data arms. Calibration and
published evaluation never enter fitting or checkpoint selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from einops import rearrange

from scripts.models.pcv_release_pooling import PooledContextVCM
from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.runtime.speaker_confidence import application_decision, slot_confidences
from scripts.test.speaker_demo_suite import file_digest, read_json, read_jsonl, write_new
from scripts.train import speaker_synthetic_data as generation

ROOT = generation.ROOT
HERE = ROOT / 'experiments/speaker-final-refinement-v1'
CACHE = ROOT / '.cache/speaker-final-refinement-v1'
PREPARED = ROOT / '.cache/speaker-synthetic-completion-v1/prepared-v1'
ARCHITECTURE = dict(width=192, layers=6, heads=4, feedforward=384, dropout=.1)
APPLICATION = dict(mode='demo', intent_threshold=.95, slot_threshold=.90)
TRAINING = dict(seed=53, human_epochs=24, mixed_epochs=32, steps_per_epoch=96,
                batch_size=32, microbatch=8, human_learning_rate=.0008,
                mixed_learning_rate=.00025, minimum_learning_rate=.00002,
                weight_decay=.001, gradient_clip=5., intent_weight=.3, target_weight=.15,
                development_every=4, decoder_workers=6)


def configure_training_device():
    """Prefer CUDA; CPU-only hosts must not call CUDA configuration APIs."""
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.45)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def seed_for(*values):
    return int(generation._hash(list(values))[:8], 16)


def save_progress(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def augment(values, rng, *, probability=.5):
    """Conservative SpecAugment adaptation; zero means human-training mean."""
    result = np.array(values, dtype=np.float32, copy=True)
    if result.ndim != 2 or result.shape[0] != 40 or not 0 <= probability <= 1:
        raise ValueError('Expected 40-bin features and a valid augmentation probability')
    for axis, maximum in ((0, 4), (1, min(10, int(.05 * result.shape[1])))):
        if rng.random() < probability and maximum:
            width = int(rng.integers(maximum + 1))
            start = int(rng.integers(result.shape[axis] - width + 1))
            if axis == 0:
                result[start:start + width] = 0
            else:
                result[:, start:start + width] = 0
    return result


class TrainingPools:
    def __init__(self, rows):
        self.groups = {}
        for row in rows:
            if row['split'] != 'train' or row.get('training_eligible') is not True:
                raise ValueError('Only eligible train rows may enter training pools')
            if row['kind'] == 'human':
                group = 'human_positive' if row['expected'] is not None else 'human_negative'
            elif row['kind'] == 'synthetic':
                group = 'new_synthetic' if row['id'].startswith('qsc1/') else 'old_synthetic'
            else:
                raise ValueError('Unrecognized fitting provenance')
            intents = self.groups.setdefault(group, {})
            commands = intents.setdefault(row['intent_index'], {})
            speakers = commands.setdefault(canonical(row['expected']), {})
            speakers.setdefault(row['speaker'], []).append(row)

    def take(self, group, count, rng):
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


def sample_batch(pools, *, seed, phase, epoch, step, arm):
    if phase == 'human':
        plan = dict(human_positive=24, human_negative=8)
    elif phase == 'mixed' and arm in {'control', 'completion'}:
        plan = dict(human_positive=12, human_negative=4,
                    old_synthetic=16 if arm == 'control' else 12,
                    new_synthetic=0 if arm == 'control' else 4)
    else:
        raise ValueError('Unknown fixed training phase or arm')
    rows = []
    for group, count in plan.items():
        rows.extend(pools.take(group, count, np.random.default_rng(seed_for(seed, phase, epoch, step, group))))
    np.random.default_rng(seed_for(seed, phase, epoch, step, 'order')).shuffle(rows)
    return rows


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
        # PooledContextVCM repeats each learned utterance summary on the time axis.
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


def summarize(records):
    result = {}
    for kind in ('human', 'synthetic'):
        rows = [r for r in records if r['synthetic'] == (kind == 'synthetic')]
        positives = [r for r in rows if r['expected'] is not None]
        negatives = [r for r in rows if r['expected'] is None]
        correct = lambda row, key: row[key] is not None and canonical(row[key]) == canonical(row['expected'])
        exact = sum(correct(r, 'application_command') for r in positives)
        raw = sum(correct(r, 'candidate') for r in positives)
        accepted = sum(r['application_command'] is not None for r in rows)
        false_accepts = sum(r['application_command'] is not None for r in negatives)
        speakers = sorted({r['speaker'] for r in positives})
        macro = lambda key: float(np.mean([np.mean([correct(r, key) for r in positives if r['speaker'] == speaker])
                                           for speaker in speakers])) if speakers else 0.
        result[kind] = dict(rows=len(rows), positive_n=len(positives), negative_n=len(negatives),
            exact=exact, accepted=accepted, precision=exact / accepted if accepted else None,
            recall=exact / len(positives) if positives else None,
            negative_errors=false_accepts, false_accept_rate=false_accepts / len(negatives) if negatives else None,
            raw_exact=raw, raw_recall=raw / len(positives) if positives else None,
            raw_negative_errors=sum(r['candidate'] is not None for r in negatives),
            macro_speaker_recall=macro('application_command'), raw_macro_speaker_recall=macro('candidate'),
            positive_speakers=len(speakers))
    return result


def selection_key(metrics, epoch):
    human, synthetic = metrics['human'], metrics['synthetic']
    feasible = (human['exact'] > 0 and (human['precision'] or 0) >= .99
                and human['false_accept_rate'] is not None and human['false_accept_rate'] <= .01)
    return (int(feasible), human['macro_speaker_recall'], human['precision'] or 0,
            -human.get('negative_errors', 0), human['raw_macro_speaker_recall'],
            synthetic.get('recall') or 0, -epoch)


def score_model(model, rows, arrays, entries, scorer, *, device='cuda', batch_size=8):
    heads = []; model.eval()

    def outputs():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            lengths = [arrays[r['id']].shape[1] for r in batch]
            x = np.zeros((len(batch), 40, max(lengths)), dtype=np.float32)
            for i, row in enumerate(batch):
                x[i, :, :lengths[i]] = arrays[row['id']]
            with torch.inference_mode():
                ctc, intent, target, _ = model(torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))
            ctc, intent, target = [v.float().cpu().numpy() for v in (ctc, intent, target)]
            for i, length in enumerate(lengths):
                def probabilities(v):
                    values = np.exp(v.astype(np.float64) - np.max(v))
                    return values / values.sum()
                n = (length + 1) // 2
                heads.append((probabilities(intent[i, :, 0]), probabilities(target[i, :, 0]), n))
                yield ctc[i, :, :n].T
    records = []
    for i, (row, item) in enumerate(zip(rows, scorer.map(outputs()), strict=True)):
        prediction = item['prediction']; candidate = prediction['command']
        ip, tp, n = heads[i]; winner = int(ip.argmax())
        ci = INTENTS.index(candidate['intent']) + 1 if candidate is not None else 0
        record = dict(id=row['id'], expected=row['expected'], candidate=candidate,
            synthetic=row['synthetic'], speaker=row['speaker'], source=row['source'],
            score=prediction['score'], margin=prediction['margin'],
            intent_index=winner, intent_probability=float(ip[ci]), top_intent_probability=float(ip[winner]),
            target_named_probability=float(tp[1]), candidate_phrase_id=prediction['phrase_id'],
            slot_confidence=slot_confidences(item['scores'], entries, candidate, n))
        application = application_decision({'output': {'accepted': False, 'reason': 'unqualified_model'},
                                           'diagnostic_candidate': record}, APPLICATION)
        record.update(application_command=application['command'], application=application)
        records.append(record)
    return records


def load_inputs():
    receipt = read_json(PREPARED / 'preparation.json')
    if not receipt['research_ready_for_fitting'] or receipt['human_qualified']:
        raise ValueError('Expected separately labelled, unqualified synthetic research inputs')
    generation._verify(receipt['source_bindings']); generation._verify(receipt['implementation_bindings'])
    for name in ('train', 'dev', 'normalization', 'catalog'):
        info = receipt[name]; generation._verify({info['path']: info['sha256']})
    train, dev = read_jsonl(PREPARED / 'train.jsonl'), read_jsonl(PREPARED / 'dev.jsonl')
    if any(r['split'] != 'dev' for r in dev):
        raise ValueError('Development cannot contain calibration/test/holdout')
    TrainingPools(train)
    if {r['id'] for r in train} & {r['id'] for r in dev}:
        raise ValueError('Fitting/development identity overlap')
    return train, dev, read_json(PREPARED / 'normalization.json'), read_json(PREPARED / 'catalog.json')['entries']


def freeze():
    path = HERE / 'protocol.json'
    if path.exists():
        value = read_json(path); generation._verify(value['bindings']); return value
    load_inputs()
    files = [PREPARED / name for name in ('preparation.json', 'train.jsonl', 'dev.jsonl', 'normalization.json', 'catalog.json')]
    files += [ROOT / name for name in ('scripts/train/speaker_final_train.py', 'scripts/models/pcv_release_pooling.py',
        'scripts/models/pcv_contextual.py', 'scripts/runtime/pcv_schema.py', 'scripts/runtime/speaker_confidence.py',
        'scripts/pcv_parallel.py', 'experiments/phrase-contract-v1/pcvcm/decoder.py', 'vcm/frontend.py',
        'experiments/speaker-final-refinement-v1/references.json')]
    cfg = dict(experiment_id='speaker-final-refinement-v1', date='2026-10-16',
        authorization='User approved last fixed-architecture refinement, documented studies/augmentations and Pi4 handoff',
        requirements=['ME-01','ME-02','ME-03','ME-04','ME-05','ME-06','ME-07','ME-08'],
        hypothesis='At fixed architecture/training budget, explicitly synthetic sparse-command completion improves human development accepted complete commands without worsening rejection',
        architecture_class='PooledContextVCM', architecture=ARCHITECTURE, parameters=2066831,
        training=TRAINING, arms=['control','completion'],
        augmentation=dict(method='conservative SpecAugment adaptation', frequency_masks=1, frequency_width_max=4,
            time_masks=1, time_width_max=10, time_fraction_max=.05, each_probability=.5,
            fill='zero in human-training-normalized feature space', train_only=True, both_arms=True,
            time_warp=False, speed=False, noise=False, reverberation=False, splicing=False),
        foundation_batch=dict(human_positive=24,human_negative=8),
        control_batch=dict(human_positive=12,human_negative=4,old_synthetic=16,new_synthetic=0),
        completion_batch=dict(human_positive=12,human_negative=4,old_synthetic=12,new_synthetic=4),
        sampling='intent then complete command then declared speaker then recording; per-group common random streams',
        selection='feasible human precision>=.99/FAR<=.01 with nonzero correct accepts; macro speaker accepted recall; precision; fewer false accepts; raw macro recall; synthetic accepted recall; earlier epoch; control on arm tie',
        application=APPLICATION, normalization='unchanged eligible human training only',
        initialization='one random-init human foundation, its selected weights shared by both mixed arms',
        quality_gates=dict(precision=.99,recall=.90,human_far=.01,all_18_human_calibration_intents=True),
        maximum_int8_graph_bytes=4*1024*1024,
        quantization_gate=dict(max_human_exact_loss=1,max_synthetic_exact_loss=1,max_extra_human_negative_errors=1),
        published_test_used_for_selection=False, human_qualified=False, physical_pi_measured=False,
        torch_tf32=True, deterministic_algorithms='warn_only; seeded draws, no bitwise CUDA guarantee',
        limitation='Single-seed bounded comparison; augmentation is common to both arms so its isolated benefit is not estimated',
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in files},
        installed_metadata={str(p.relative_to(ROOT)):file_digest(p) for p in (ROOT/'artifacts').iterdir() if p.is_file()})
    write_new(path,cfg)
    return cfg


def load_arrays(rows, norm):
    arrays = {}; mean = np.asarray(norm['mean'],dtype=np.float32)[:,None]; std = np.asarray(norm['std'],dtype=np.float32)[:,None]
    for i,row in enumerate(rows,1):
        path=Path(row['features'])
        if file_digest(path)!=row['features_sha256']:
            raise ValueError('Prepared feature identity mismatch: '+row['id'])
        values=(np.load(path,allow_pickle=False).astype(np.float32)-mean)/std
        if values.shape[0]!=40 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError('Invalid normalized features')
        arrays[row['id']]=values
        if i%2500==0: print('LOADED_FEATURES',i,'/',len(rows),flush=True)
    return arrays


def fit_stage(phase, arm, initial, pools, dev, arrays, entries, scorer, cfg):
    settings=cfg['training']; output=CACHE/'training'/(phase+'-'+arm)
    output.mkdir(parents=True,exist_ok=False)
    seed=settings['seed']+(100 if phase=='mixed' else 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    model=PooledContextVCM(**cfg['architecture']).to(device)
    if sum(p.numel() for p in model.parameters())!=cfg['parameters']:
        raise ValueError('Architecture parameter count changed')
    if initial is not None:
        model.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True)['weights'])
    epochs=settings['human_epochs'] if phase=='human' else settings['mixed_epochs']
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings[phase+'_learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=settings['minimum_learning_rate'])
    best=None; best_key=None; history=[]; stage_started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); began=time.perf_counter(); losses=np.zeros(3)
        for step in range(settings['steps_per_epoch']):
            batch=sample_batch(pools,seed=settings['seed'],phase=phase,epoch=epoch,step=step,arm=arm)
            losses+=optimization_step(model,batch,arrays,optimizer,device=device,
                seed=seed_for(seed,phase,epoch,step,'augmentation'),microbatch=settings['microbatch'])
        scheduler.step()
        metric=dict(phase=phase,arm=arm,epoch=epoch,training_seconds=time.perf_counter()-began,
                    losses=(losses/settings['steps_per_epoch']).tolist(),learning_rate=scheduler.get_last_lr()[0])
        checkpoint=dict(weights=model.state_dict(),model_config=cfg['architecture'],architecture_class='PooledContextVCM',
                        epoch=epoch,phase=phase,arm=arm,seed=seed,device=device,protocol_sha256=file_digest(HERE/'protocol.json'))
        if epoch%settings['development_every']==0:
            selected_dev=[r for r in dev if not r['synthetic']] if phase=='human' else dev
            predictions=score_model(model,selected_dev,arrays,entries,scorer,device=device)
            metrics=summarize(predictions); key=selection_key(metrics,epoch)
            write_new(output/f'dev-{epoch:03d}.json',predictions)
            metric.update(development=metrics,selection_key=list(key))
            if best_key is None or key>best_key:
                path=output/f'epoch-{epoch:03d}.pt'; torch.save(checkpoint,path)
                best_key=key; best=dict(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=file_digest(path),
                    epoch=epoch,arm=arm,phase=phase,device=device,development=metrics,key=list(key),test_used_for_selection=False)
                save_progress(output/'selection.json',best)
        temporary=output/'last.tmp.pt'; torch.save(checkpoint,temporary); os.replace(temporary,output/'last.pt')
        history.append(metric); save_progress(output/'history.json',history)
        save_progress(CACHE/'status.json',dict(phase=phase,arm=arm,epoch=epoch,epochs=epochs,
                                              stage_seconds=time.perf_counter()-stage_started))
        print('FINAL_TRAIN',json.dumps(metric),flush=True)
    write_new(output/'complete.json',best)
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return best


def train():
    cfg=freeze(); device=configure_training_device()
    cfg={**cfg, 'device':device}
    if (HERE/'selection-before-calibration.json').exists():
        receipt=read_json(HERE/'selection-before-calibration.json'); generation._verify(receipt['bindings']); return receipt
    if (CACHE/'training').exists():
        raise FileExistsError('Preserve partial final training evidence; do not overwrite a run')
    rows,dev,norm,entries=load_inputs(); pools=TrainingPools(rows)
    arrays=load_arrays(rows+dev,norm)
    from scripts.pcv_parallel import ParallelScorer
    began=time.perf_counter()
    with ParallelScorer(entries,workers=cfg['training']['decoder_workers'],return_scores=True) as scorer:
        foundation=fit_stage('human','shared',None,pools,dev,arrays,entries,scorer,cfg)
        arms={arm:fit_stage('mixed',arm,ROOT/foundation['checkpoint'],pools,dev,arrays,entries,scorer,cfg)
              for arm in cfg['arms']}
    winner=max(cfg['arms'],key=lambda arm:arms[arm]['key'])
    generation._verify(cfg['bindings']); generation._verify(cfg['installed_metadata'])
    receipt=dict(experiment_id=cfg['experiment_id'],winner=winner,selected=arms[winner],arms=arms,foundation=foundation,
        seconds=time.perf_counter()-began,seed=cfg['training']['seed'],human_qualified=False,test_used_for_selection=False,
        device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
        torch=torch.__version__,cuda=torch.version.cuda,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
        bindings={str(p.relative_to(ROOT)):file_digest(p) for p in (HERE/'protocol.json',ROOT/arms[winner]['checkpoint'])})
    write_new(HERE/'selection-before-calibration.json',receipt)
    print('FINAL_TRAINING_COMPLETE',json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=('freeze','train'))
    args=parser.parse_args()
    result=freeze() if args.stage=='freeze' else train()
    print(json.dumps({'experiment_id':result['experiment_id'],'status':args.stage+'_complete'},indent=2))


if __name__=='__main__': main()
