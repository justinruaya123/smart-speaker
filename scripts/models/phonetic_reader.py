"""Research-only CTC phrase graphs, versioned catalogs and typed slot adapter.

No acoustic training, intent selection, actions or service integration here.
"""
from collections import defaultdict
import hashlib
import json
import re
import numpy as np
from einops import pack
from .parameter_study_data import phones, normalize
from .levels import spoken_number, level_parameter


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def catalog_fixture(values, pronunciations=None):
    entries = []
    for role, labels in sorted(values.items()):
        for label in labels:
            entries.append(dict(id=role + ':' + fingerprint(label)[:16], role=role, label=label,
                                aliases=[label], pronunciations=(pronunciations or {}).get(role+'::'+label, [])))
    payload = dict(schema=1, entries=entries)
    return {**payload, 'version': fingerprint(payload)}


def numeric_templates(training_rows):
    templates = {'{number}', '{number} percent', 'brightness {number}', 'brightness {number} percent'}
    for prefix in ('set brightness to', 'set the brightness to', 'set lights to', 'set the lights to',
                   'turn brightness to', 'turn the brightness to', 'change brightness to', 'change the brightness to'):
        for suffix in ('', ' percent'):
            templates.add(prefix+' {number}'+suffix)
    for r in training_rows:
        if r['split'] != 'train' or r.get('command') != 'BRIGHTNESS' or r['kind'] == 'word':
            continue
        try:
            parsed = level_parameter(r['text'])
        except ValueError:
            continue
        if 'percent' not in parsed:
            continue
        phrase = spoken_number(parsed['percent'])
        if re.search(r'\b'+re.escape(phrase)+r'\b', r['text']):
            templates.add(re.sub(r'\b'+re.escape(phrase)+r'\b', '{number}', r['text']))
    return sorted(templates)


class CTCGraph:
    """Shared-prefix CTC Viterbi on frame posteriors, with an optional filler.

    Scores are relative to the unrestricted frame-best path, not probabilities
    of correctness. Development calibration is required. No word times claimed.
    """
    def __init__(self, patterns, vocabulary, substring=False):
        self.vocabulary = vocabulary
        lookup = {p: i for i, p in enumerate(vocabulary)}
        self.substring = substring
        self.boundary = lookup['|']
        children = {}
        labels = [0]
        parents = [0]
        terminals = defaultdict(set)
        for value, sequence in patterns:
            node = 0
            for p in sequence:
                token = lookup[p]
                if token == 0:
                    raise ValueError('Blank is not a lexical phone')
                edge = node, token
                if edge not in children:
                    children[edge] = len(labels)
                    labels.append(token)
                    parents.append(node)
                node = children[edge]
            terminals[node].add(value)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.parents = np.asarray(parents, dtype=np.int64)
        self.skip = self.labels != self.labels[self.parents]
        self.skip[0] = False
        self.nodes = np.asarray(sorted(terminals), dtype=np.int64)
        depths = [0]*len(labels)
        for i in range(1, len(labels)):
            depths[i] = depths[parents[i]]+1
        self.depth = np.maximum(1, np.asarray([depths[n] for n in self.nodes]))
        self.values = [sorted(terminals[n]) for n in self.nodes]

    def score(self, logits):
        if not len(self.nodes):
            return []
        x = np.asarray(logits, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != len(self.vocabulary) or not np.isfinite(x).all():
            raise ValueError('Invalid CTC logits')
        # Subtracting the maximum preserves path ordering and gives every
        # departure from the unrestricted frame-best path a nonpositive cost.
        x = x - x.max(axis=-1, keepdims=True)
        if self.substring:
            edge = np.full((1, x.shape[1]), -1e6, dtype=np.float32)
            edge[0, self.boundary] = 0.
            x, _ = pack([edge, x, edge], '* phone')
        blank = np.full(len(self.labels), -1e6, dtype=np.float32)
        emit = blank.copy()
        blank[0] = 0.
        best = np.full(len(self.nodes), -1e6, dtype=np.float32)
        for frame in x:
            next_blank = np.maximum(blank, emit) + frame[0]
            advance = np.maximum(blank[self.parents], np.where(self.skip, emit[self.parents], -1e6))
            next_emit = np.maximum(emit, advance) + frame[self.labels]
            next_emit[0] = -1e6
            if self.substring:
                next_blank[0] = 0.
            blank, emit = next_blank, next_emit
            if self.substring:
                best = np.maximum(best, np.maximum(blank[self.nodes], emit[self.nodes]))
        if not self.substring:
            best = np.maximum(blank[self.nodes], emit[self.nodes])
        scores = np.exp(np.maximum(-80., best/self.depth))
        by_value = {}
        for values, score in zip(self.values, scores):
            for value in values:
                by_value[value] = max(by_value.get(value, 0.), float(score))
        return sorted(by_value.items(), key=lambda x: (-x[1], x[0]))

    def predict(self, logits):
        scores = self.score(logits)
        if not scores:
            return None, 0.
        value, score = scores[0]
        second = scores[1][1] if len(scores) > 1 else 0.
        if score <= 1e-20 or abs(score-second) < 1e-8:
            return None, 0.
        # Margin is deliberately part of the operating score; it must be
        # recalibrated when competition changes after a catalog update.
        return value, max(0., score*(1-second))


class PhoneticCatalogReader:
    def __init__(self, vocabulary, dictionary, catalog, templates):
        expected = fingerprint({k: catalog[k] for k in ('schema', 'entries')})
        if catalog.get('schema') != 1 or catalog.get('version') != expected:
            raise ValueError('Catalog version/content mismatch')
        self.version = catalog['version']
        self.entries = {}
        seen_ids = set()
        patterns = defaultdict(list)
        self.oov = []
        for entry in catalog['entries']:
            role, value = entry['role'], entry['label']
            key = role, value
            if key in self.entries or entry['id'] in seen_ids:
                raise ValueError('Duplicate catalog value or ID')
            self.entries[key] = entry
            seen_ids.add(entry['id'])
            if role == 'number':
                if not value.isdecimal() or not 0 <= int(value) <= 100 or str(int(value)) != value:
                    raise ValueError('Numeric catalog must stay within 0..100')
                aliases = [spoken_number(int(value))]
                if value == '100':
                    aliases += ['a hundred', 'hundred']
                for alias in aliases:
                    patterns['isolated_number'].append((value, phones(alias, dictionary)))
                    for template in templates:
                        sequence = phones(template.replace('{number}', alias), dictionary)
                        if sequence is not None:
                            patterns['natural_number'].append((value, sequence))
                continue
            sequences = list(entry.get('pronunciations', []))
            for alias in entry.get('aliases', [value]):
                sequence = phones(normalize(alias), dictionary)
                if sequence is not None:
                    sequences.append(sequence)
            if not sequences:
                self.oov.append(key)
            for sequence in sequences:
                if not sequence or any(p not in vocabulary or p == '<blank>' for p in sequence):
                    raise ValueError('Invalid explicit pronunciation')
                patterns[role].append((value, ['|'] + sequence + ['|']))
        self.graphs = {role: CTCGraph(items, vocabulary, substring=role not in ('isolated_number', 'natural_number')) for role, items in patterns.items()}

    def predict(self, logits, role, kind='natural'):
        group = ('isolated_number' if kind == 'word' else 'natural_number') if role == 'number' else role
        graph = self.graphs.get(group)
        return graph.predict(logits) if graph else (None, 0.)

    def resolve(self, logits, *, accepted_intent, role, gate, calibrated_catalog_version):
        allowed = {'BRIGHTNESS': {'number', 'device'}, 'VOLUME': {'number'}, 'MEDIA_PLAY': {'artist', 'song'},
                   'POWER_ON': {'device'}, 'POWER_OFF': {'device'}, 'COLOR_SET': {'device'}}
        if role not in allowed.get(accepted_intent, set()):
            return dict(status='needs_parameters', reason='intent_does_not_own_slot', intent=accepted_intent)
        if calibrated_catalog_version != self.version:
            return dict(status='needs_parameters', reason='catalog_requires_calibration', intent=accepted_intent)
        threshold = gate.get('threshold')
        if gate.get('passed') and (not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            return dict(status='needs_parameters', reason='invalid_calibration', intent=accepted_intent)
        if role == 'number' and accepted_intent != 'BRIGHTNESS':
            return dict(status='needs_parameters', reason='only_absolute_brightness_grammar_evaluated', intent=accepted_intent)
        value, score = self.predict(logits, role)
        if not gate.get('passed') or value is None or score < threshold:
            return dict(status='needs_parameters', reason='parameter_abstention', intent=accepted_intent, confidence=score)
        result = dict(status='accepted', intent=accepted_intent, role=role, confidence=score, catalog_version=self.version,
                      timing_verified=False, action_executed=False)
        if role == 'number':
            result['parameter'] = {'percent': int(value)}
        else:
            result['parameter'] = {'entity_id': self.entries[role, value]['id'], 'label': value}
        return result
