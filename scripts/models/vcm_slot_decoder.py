"""E19 research-only slot indexing and typed decisions; never selects an intent.

The embedding path emits no speech tokens. Pronunciations are catalog inputs.
Raw scores are diagnostics, not probabilities or accepted commands.
"""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from einops import rearrange, reduce
from tinyvcm.levels import spoken_number
from tinyvcm.parameter_study_data import lexicon, phones, pcm
from tinyvcm.stt_runtime import stt_features

ROOT = Path(__file__).resolve().parents[2]


def catalog_fingerprint(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def number_entries():
    result = []
    for value in range(101):
        words = [spoken_number(value)] + (['a hundred'] if value == 100 else [])
        aliases = words + [w + ' percent' for w in words]
        result.append(dict(id=f'number:{value}', role='number', value=value, aliases=aliases))
    return result


def time_entries():
    """Finite compositional pilot: hours 1..12, minutes 00/15/30/45, 3 meridiem states."""
    entries = []
    for hour in range(1, 13):
        for minute in (0, 15, 30, 45):
            for meridiem in ('unspecified', 'am', 'pm'):
                phrase = spoken_number(hour) + (' ' + spoken_number(minute) if minute else '')
                aliases = [phrase]
                if minute == 0:
                    aliases += [spoken_number(hour) + " o'clock"]
                if meridiem != 'unspecified':
                    suffix = 'a m' if meridiem == 'am' else 'p m'
                    aliases = [s + ' ' + suffix for s in aliases] + [s + (' in the morning' if meridiem == 'am' else ' in the evening') for s in aliases]
                hour24 = hour % 12 + (12 if meridiem == 'pm' else 0)
                entries.append(dict(id=f'time:{hour:02}:{minute:02}:{meridiem}', role='time',
                                    hour=hour, minute=minute, meridiem=meridiem,
                                    value=None if meridiem == 'unspecified' else f'{hour24:02}:{minute:02}', aliases=aliases))
    return entries


def ranked(scores):
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def calibrate(records, version):
    """Dev only. Score AND margin, >=10 emissions, >=95% precision, <=1% unknown FA."""
    unknown = sum(r['reference'] is None for r in records)
    best = None
    if unknown >= 20:
        thresholds = sorted({r['scores'][0][1] for r in records if r['scores']})
        for margin in (1e-6, .01, .025, .05, .1, .2, .3):
            for threshold in thresholds:
                emitted = [r for r in records if eligible(r['scores'], threshold, margin)]
                correct = sum(r['reference'] is not None and r['scores'][0][0] == r['reference'] for r in emitted)
                false = sum(r['reference'] is None for r in emitted)
                if len(emitted) >= 10 and correct / len(emitted) >= .95 and false / unknown <= .01:
                    candidate = (correct, -false, threshold, margin, len(emitted))
                    if best is None or candidate > best:
                        best = candidate
    return dict(passed=best is not None, threshold=best[2] if best else 2., margin=best[3] if best else 1.,
                development_correct=best[0] if best else 0, development_emitted=best[4] if best else 0,
                development_unknown=unknown, minimum_emissions=10, minimum_unknown=20,
                precision_target=.95, unknown_fa_target=.01, catalog_version=version)


def eligible(scores, threshold, margin):
    if not scores or not all(math.isfinite(float(s)) for _, s in scores):
        return False
    second = scores[1][1] if len(scores) > 1 else -1.
    return scores[0][1] >= threshold and scores[0][1] - second > margin


def decision(entries, scores, intent, role, gate):
    base = dict(status='abstain', intent=intent, role=role, action_executed=False,
                confidence_is_probability=False)
    owns = {'number': {'BRIGHTNESS', 'VOLUME'}, 'time': {'SET_ALARM'},
            'artist': {'MEDIA_PLAY'}, 'song': {'MEDIA_PLAY'}, 'device': {'POWER_ON', 'POWER_OFF'}}
    if intent not in owns.get(role, set()):
        return dict(base, reason='intent_does_not_own_slot')
    if gate.get('catalog_version') != catalog_fingerprint(entries):
        return dict(base, reason='catalog_requires_calibration')
    if not gate.get('passed'):
        return dict(base, reason='quality_gate_failed')
    if not all(isinstance(gate.get(k), (int, float)) and math.isfinite(gate[k]) and 0 <= gate[k] <= 1 for k in ('threshold','margin')):
        return dict(base, reason='invalid_calibration')
    if not eligible(scores, gate['threshold'], gate['margin']):
        return dict(base, reason='low_score_or_margin')
    entry = next((e for e in entries if e['id'] == scores[0][0] and e['role'] == role), None)
    if entry is None:
        return dict(base, reason='invalid_candidate')
    if role == 'time' and entry['meridiem'] == 'unspecified':
        return dict(base, status='clarify', reason='missing_meridiem')
    return dict(base, status='accepted', reason='matched_catalog', score=scores[0][1],
                **({'value': entry['value']} if role in ('number', 'time') else {'entity_id': entry['id']}))


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


class EmbeddingModel:
    def __init__(self):
        start = time.perf_counter()
        folder = ROOT / '.cache/speechembedding-v1'
        self.paths = [folder / f'dual_kws-{kind}.onnx' for kind in ('audio', 'text')]
        meta = json.loads((folder / 'dual_kws-training.json').read_text())
        for kind, path in zip(('audio','text'), self.paths):
            if hashlib.sha256(path.read_bytes()).hexdigest() != meta['graphs'][kind]['sha256']:
                raise ValueError('Frozen embedding graph hash changed')
        self.audio_session, self.text_session = [session(p) for p in self.paths]
        self.vocabulary = {p:i for i,p in enumerate(meta['vocabulary'])}
        self.dictionary = lexicon()
        self.startup_ms = (time.perf_counter() - start) * 1000

    def audio(self, path):
        wave = pcm({'audio': str(path)})
        start = time.perf_counter()
        features = stt_features(wave)
        frontend_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        windows = self.audio_session.run(None, {'features': rearrange(features, 't f -> 1 t f'),
                                                 'lengths': np.array([len(features)], dtype=np.int64)})[0][0]
        return windows, features, dict(frontend_ms=frontend_ms, inference_ms=(time.perf_counter()-start)*1000,
                                      duration_s=len(wave)/16000)

    def index(self, entries):
        ids = [e['id'] for e in entries]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate entity IDs')
        sequences, owners, missing = [], [], []
        for entry in entries:
            usable = 0
            explicit = entry.get('pronunciations', [])
            candidates = explicit + [phones(a, self.dictionary) for a in entry['aliases']]
            for sequence in candidates:
                if sequence and all(p in self.vocabulary and p != '<pad>' for p in sequence):
                    sequences.append([self.vocabulary[p] for p in sequence])
                    owners.append(entry['id']); usable += 1
            if not usable:
                missing.append(entry['id'])
        matrices = []
        for first in range(0, len(sequences), 128):
            batch = sequences[first:first+128]
            tokens = np.zeros((len(batch), max(map(len,batch))), dtype=np.int64)
            for i, sequence in enumerate(batch):
                tokens[i, :len(sequence)] = sequence
            matrices.extend(self.text_session.run(None, {'tokens': tokens, 'lengths': np.array(list(map(len,batch)),dtype=np.int64)})[0])
        matrix = rearrange(matrices, 'n e -> n e') if matrices else np.empty((0,96),dtype=np.float32)
        return dict(matrix=matrix, owners=owners, unavailable=missing, version=catalog_fingerprint(entries))

    @staticmethod
    def rank(windows, index):
        if not np.isfinite(windows).all():
            return []
        scores = reduce(windows @ rearrange(index['matrix'], 'n e -> e n'), 'w n -> n', 'max')
        by_id = {}
        for owner, score in zip(index['owners'], scores):
            by_id[owner] = max(by_id.get(owner, -1.), float(score))
        return ranked(by_id)
