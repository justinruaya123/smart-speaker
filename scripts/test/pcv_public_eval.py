"""Finite command evaluation; reference labels never enter acoustic inference."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
from einops import rearrange, reduce

from scripts.runtime.pcv_schema import INTENTS, canonical
from scripts.train.pcv_public_data import sha, ROOT
from scripts.pcv_parallel import ParallelScorer


def normalized(row, norm):
    x = np.load(row['features'], allow_pickle=False)
    return (x - rearrange(np.asarray(norm['mean'], dtype='float32'), 'c -> c 1')) / rearrange(np.asarray(norm['std'], dtype='float32'), 'c -> c 1')


def probabilities(x):
    x = np.asarray(x, dtype='float64')
    x = np.exp(x - x.max())
    return x/x.sum()


def metric_summary(records, accepted_key=None):
    groups = defaultdict(list)
    groups['all'] = records
    for record in records:
        groups['source/' + record['source']].append(record)
        groups['intent/' + (record['expected']['intent'] if record['expected'] else 'OOV')].append(record)
    result = {}
    for name, rows in sorted(groups.items()):
        positive = [r for r in rows if r['expected'] is not None]
        negative = [r for r in rows if r['expected'] is None]
        def actual(r):
            return r[accepted_key] if accepted_key else r['candidate']
        exact = sum(actual(r) is not None and canonical(actual(r)) == canonical(r['expected']) for r in positive)
        accepted = sum(actual(r) is not None for r in rows)
        errors = sum(actual(r) is not None for r in negative)
        result[name] = {'positive_n': len(positive), 'negative_n': len(negative), 'exact': exact,
                        'recall': exact/len(positive) if positive else None,
                        'accepted': accepted, 'precision': exact/accepted if accepted else None,
                        'negative_errors': errors, 'false_accept_rate': errors/len(negative) if negative else None}
    return result


def score(model, rows, norm, entries, device='cpu', workers=6, arrays=None):
    is_onnx = hasattr(model, 'get_inputs')
    if not is_onnx:
        import torch
    rows = list(rows)
    records, heads = [], []
    if not is_onnx:
        model.eval()
    started = time.perf_counter()
    def acoustic_outputs():
        batch_size = 1 if is_onnx else 24
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset+batch_size]
            values = [arrays[r['id']] if arrays is not None else normalized(r, norm) for r in batch]
            lengths = [a.shape[1] for a in values]
            x = np.zeros((len(batch), 40, max(lengths)), dtype='float32')
            for i, a in enumerate(values):
                x[i, :, :a.shape[1]] = a
            if is_onnx:
                feed = {'features': x}
                for port in model.get_inputs()[1:]:
                    feed[port.name] = np.zeros(port.shape, dtype='float32')
                outputs = model.run(None, feed)
                ctc, intent, target = [rearrange(o, 'b t c -> b c t') for o in outputs[:3]]
            else:
                with torch.no_grad():
                    ctc, intent, target, _ = model(torch.from_numpy(x).to(device))
                    ctc, intent, target = ctc.cpu().numpy(), intent.cpu().numpy(), target.cpu().numpy()
            for i, n in enumerate(lengths):
                heads.append((probabilities(reduce(intent[i, :, :n], 'c t -> c', 'mean')),
                              probabilities(reduce(target[i, :, :n], 'c t -> c', 'mean'))))
                yield rearrange(ctc[i, :, :n], 'c t -> t c')
    with ParallelScorer(entries, workers=workers) as scorer:
        for index, (row, item) in enumerate(zip(rows, scorer.map(acoustic_outputs()), strict=True)):
            pred = item['prediction']
            ip, tp = heads[index]
            winner = int(ip.argmax())
            candidate = pred['command']
            ci = INTENTS.index(candidate['intent'])+1 if candidate else 0
            records.append({'id': row['id'], 'source': row['source'], 'speaker': row['speaker'],
                'synthetic': row['synthetic'], 'phrase_id': row.get('phrase_id'), 'text': row['text'],
                'expected': row['expected'], 'candidate': candidate, 'score': pred['score'], 'margin': pred['margin'],
                'intent_index': winner, 'intent_probability': float(ip[ci]), 'top_intent_probability': float(ip[winner]),
                'intent_correct': winner == row['intent_index'], 'target_named_probability': float(tp[1]),
                'target_expected': row['target_index'], 'candidate_phrase_id': pred['phrase_id']})
            if len(records) % 200 == 0 or len(records) == len(rows):
                print('FINITE_EVALUATION', len(records), '/', len(rows), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    return records


def accepted_candidate(record, policy):
    c = record['candidate']
    if c is None or record['score'] is None:
        return None
    if record['intent_index'] != INTENTS.index(c['intent'])+1:
        return None
    if record['score'] < policy['min_score'] or record['margin'] < policy['min_margin'] or record['intent_probability'] < policy['min_intent_probability']:
        return None
    if c['intent'] in {'LIGHT_ON','LIGHT_OFF','BRIGHTNESS','COLOR'}:
        named = c['slots']['light_id'] != 'light.all'
        if (not named and record['target_named_probability'] > 0.9) or (named and record['target_named_probability'] < 0.1):
            return None
    return c


def calibrate(records):
    values = [r['score'] for r in records if r['score'] is not None and r['candidate'] is not None]
    thresholds = sorted(set([-100.0] + [float(x) for x in np.quantile(values, np.linspace(0, 1, 11))])) if values else [0.0]
    best = None
    best_key = (-1, -1, -1)
    feasible = []
    for score_min in thresholds:
        for margin in (0., .002, .005, .01, .02, .05, .1):
            for ip in (0., .5, .7, .9):
                policy = {'min_score': score_min, 'min_margin': margin, 'min_intent_probability': ip}
                selected = [{**r, 'accepted_command': accepted_candidate(r, policy)} for r in records]
                metrics = metric_summary(selected, 'accepted_command')['all']
                precision = metrics['precision'] or 0
                recall = metrics['recall'] or 0
                far = metrics['false_accept_rate'] or 0
                key = (int(precision >= .99 and far <= .01), recall, precision)
                if key > best_key:
                    best, best_key = {'policy': policy, 'metrics': metrics}, key
                if metrics['positive_n'] and metrics['negative_n'] and precision >= .99 and recall >= .9 and far <= .01:
                    feasible.append({'policy': policy, 'metrics': metrics})
    chosen = max(feasible, key=lambda r: r['metrics']['recall']) if feasible else best
    return {**chosen['policy'], 'enabled': bool(feasible), 'calibration_metrics': chosen['metrics'],
            'reason': 'calibration_gates_passed' if feasible else 'calibration_quality_gates_failed',
            'calibration_rows': len(records), 'grid_size': len(thresholds)*7*4,
            'minimum_recall': .9, 'minimum_precision': .99, 'maximum_false_accept_rate': .01}
