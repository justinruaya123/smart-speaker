"""Numerical decoder checks and a development-locked real-audio comparison."""
from collections import defaultdict
from datetime import datetime, timezone
import itertools
import json
import platform
import random
import time
import numpy as np
from einops import rearrange
from .phonetic_data import ROOT, DATA, OUT, waveform
from .parameter_study_data import dump, digest, lexicon
from .parameter_study import collate
from .parameter_study_eval import Matcher, summarize, threshold_on_dev
from .phonetic_reader import CTCGraph, PhoneticCatalogReader, catalog_fixture, numeric_templates, fingerprint
from .stt_runtime import stt_features
from .levels import level_parameter


def checks():
    from .phonetic_training import phone_edit
    from .parameter_study import edit
    edit_rng = random.Random(627)
    for _ in range(500):
        a = [edit_rng.randrange(12) for _ in range(edit_rng.randrange(80))]
        b = [edit_rng.randrange(12) for _ in range(edit_rng.randrange(100))]
        assert phone_edit(a, b) == edit(a, b)
    vocabulary = ['<blank>', 'A', 'B', '|']
    rng = np.random.default_rng(48)
    def collapsed(path):
        return [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]
    tests = 0
    for length in range(1, 6):
        logits = rng.normal(size=(length, 4)).astype('float32')
        relative = logits - logits.max(axis=-1, keepdims=True)
        for pattern in ([1], [2], [1, 1], [1, 2], [2, 1], [1, 2, 1]):
            expected = -1e6
            for path in itertools.product(range(4), repeat=length):
                if collapsed(path) == list(pattern):
                    expected = max(expected, sum(relative[i, p] for i, p in enumerate(path)))
            expected = np.exp(max(-80., expected/len(pattern)))
            graph = CTCGraph([('target', [vocabulary[p] for p in pattern])], vocabulary)
            actual = graph.score(logits)[0][1]
            if not np.isclose(actual, expected, atol=1e-6):
                raise AssertionError((pattern, length, actual, expected))
            tests += 1
    substring_tests = 0
    for length in range(1, 4):
        logits = rng.normal(size=(length, 4)).astype('float32')
        relative = logits - logits.max(axis=-1, keepdims=True)
        edge = np.full((1, 4), -1e6, dtype='float32')
        edge[0, 3] = 0.
        from einops import pack
        padded, _ = pack([edge, relative, edge], '* phone')
        for pattern in ([3, 1, 3], [3, 2, 3], [3, 1, 1, 3]):
            expected = -1e6
            for start in range(len(padded)):
                for end in range(start+1, len(padded)+1):
                    for path in itertools.product(range(4), repeat=end-start):
                        if collapsed(path) == list(pattern):
                            expected = max(expected, sum(padded[start+i, p] for i, p in enumerate(path)))
            graph = CTCGraph([('target', [vocabulary[p] for p in pattern])], vocabulary, substring=True)
            assert np.isclose(graph.score(logits)[0][1], np.exp(max(-80., expected/len(pattern))), atol=1e-6)
            substring_tests += 1
    from .levels import level_parameter, spoken_number
    numeric = 0
    for n in range(101):
        for text in (f'brightness {n}', f'brightness {n}%', 'brightness '+spoken_number(n), 'brightness '+spoken_number(n)+' percent'):
            assert level_parameter(text) == {'percent': n}
            numeric += 1
    rejected = 0
    for text in ('brightness 101', 'brightness minus one', 'brightness twenty thirty', 'brightness 4.5',
                 'brightness fifty or sixty', 'brightness', 'brightness percent', 'brightness twenty and set lights to fifty'):
        try:
            level_parameter(text)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(text)
    dictionary = {'a': ['A'], 'b': ['B']}
    fixture = catalog_fixture({'artist': ['a', 'b']})
    reader = PhoneticCatalogReader(vocabulary, dictionary, fixture, [])
    logits = np.array([[0., 12., 0., 0.]], dtype='float32')
    gate = {'passed': True, 'threshold': 0.}
    a = reader.resolve(logits, accepted_intent='MEDIA_PLAY', role='artist', gate=gate, calibrated_catalog_version=fixture['version'])
    assert a['status'] == 'accepted' and a['intent'] == 'MEDIA_PLAY' and not a['action_executed']
    assert reader.resolve(logits, accepted_intent='UNKNOWN', role='artist', gate=gate, calibrated_catalog_version=fixture['version'])['reason'] == 'intent_does_not_own_slot'
    assert reader.resolve(logits, accepted_intent='MEDIA_PLAY', role='artist', gate=gate, calibrated_catalog_version='stale')['reason'] == 'catalog_requires_calibration'
    assert reader.resolve(logits, accepted_intent='MEDIA_PLAY', role='artist', gate={'passed': False}, calibrated_catalog_version=fixture['version'])['status'] != 'accepted'
    for threshold in (-1, float('nan'), float('inf'), None):
        assert reader.resolve(logits, accepted_intent='MEDIA_PLAY', role='artist', gate={'passed': True, 'threshold': threshold}, calibrated_catalog_version=fixture['version'])['reason'] == 'invalid_calibration'
    tied = catalog_fixture({'artist': ['a', 'a2']}, {'artist::a2': [['A']]})
    ambiguity = PhoneticCatalogReader(vocabulary, dictionary, tied, [])
    assert ambiguity.predict(logits, 'artist')[0] is None
    tampered = dict(fixture, version='incorrect')
    try:
        PhoneticCatalogReader(vocabulary, dictionary, tampered, [])
    except ValueError:
        pass
    else:
        raise AssertionError('Tampered catalog accepted')
    result = dict(global_edit_reference_cases=500, exhaustive_ctc_cases=tests, exhaustive_keyword_filler_cases=substring_tests, numeric_text_forms=numeric, malformed_text_rejected=rejected,
                  intent_ownership=True, disabled_gate=True, invalid_threshold_rejection=True, catalog_invalidation=True, homophone_abstention=True,
                  generated_audio=False, acoustic_accuracy_claim=False)
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/'checks.json', result)
    return result


def queries(rows, split):
    result = []
    for i, row in enumerate(rows):
        if row['split'] != split:
            continue
        for target in row['targets']:
            role = target['role']
            group = ('isolated_number' if row['kind'] == 'word' else 'natural_number') if role == 'number' else role
            try:
                absolute_brightness = group == 'natural_number' and row.get('command') == 'BRIGHTNESS' and 'percent' in level_parameter(row['text'])
            except ValueError:
                absolute_brightness = False
            result.append(dict(index=i, id=row['id'], role=role, group=group, gate=group, reference=target['value'],
                               novel=target.get('novel', False), legacy=row.get('legacy_split') == split,
                               command=row.get('command'), kind=row['kind'], absolute_brightness=absolute_brightness))
    # Put natural lighting commands with no numeric slot first, then general
    # backgrounds. Selection is deterministic and shared across candidates.
    negatives = [i for i, r in enumerate(rows) if r['split'] == split and not r['targets']]
    negatives.sort(key=lambda i: (rows[i]['source'] != 'SnipsSmartLights', rows[i]['id']))
    for role in ('number', 'artist', 'song', 'device', 'word'):
        for i in negatives[:300]:
            group = 'natural_number' if role == 'number' else role
            row = rows[i]
            result.append(dict(index=i, id=row['id'], role=role, group='negative_'+group, gate=group, reference=None,
                               novel=False, legacy=row.get('legacy_split') == split, command=row.get('command'), kind='natural'))
    # An isolated-word acceptance gate also sees the same negative recordings;
    # this is a separate conditional trial, not another independent recording.
    for i in negatives[:300]:
        row = rows[i]
        result.append(dict(index=i, id=row['id'], role='number', group='negative_isolated_number', gate='isolated_number',
                           reference=None, novel=False, legacy=row.get('legacy_split') == split,
                           command=row.get('command'), kind='word'))
    return result


def session_for(name):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    start = time.perf_counter()
    session = ort.InferenceSession(str(OUT/name/'phoneme.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    return session, (time.perf_counter()-start)*1000


def infer(session, query, cache):
    output = {}
    indices = sorted({q['index'] for q in query})
    for offset in range(0, len(indices), 32):
        x, lengths, ii = collate(indices[offset:offset+32], cache)
        logits, n = session.run(None, {'features': x.numpy(), 'lengths': lengths.numpy()})
        for j, i in enumerate(ii):
            output[i] = logits[j, :int(n[j])].copy()
    return output


def predict(query, outputs, matcher, decoder):
    records = []
    for count, q in enumerate(query):
        logits = outputs[q['index']]
        value, confidence = matcher.predict(logits, q['role'], q['kind']) if decoder == 'graph' else matcher.predict((logits, len(logits)), q['role'])
        records.append({**q, 'prediction': value, 'confidence': confidence})
        if count and count % 2000 == 0:
            print('decode', decoder, count, '/', len(query), flush=True)
    return records


def groups(records, gates=None):
    def metric(items):
        if gates is None:
            return summarize(items)
        return summarize([{**r, 'prediction': r['prediction'] if gates[r['gate']]['passed'] and r['confidence'] >= gates[r['gate']]['threshold'] else None} for r in items])
    result = {g: metric([r for r in records if r['group'] == g]) for g in sorted({r['group'] for r in records})}
    result['reserved_names'] = metric([r for r in records if r['novel']])
    result['known_catalog_names'] = metric([r for r in records if r.get('in_fixed_catalog')])
    result['absolute_brightness'] = metric([r for r in records if r.get('absolute_brightness')])
    result['legacy_positive'] = {g: metric([r for r in records if r['legacy'] and r['reference'] is not None and r['group'] == g]) for g in sorted({r['group'] for r in records if r['reference'] is not None})}
    result['natural_number_by_value'] = {str(n): metric([r for r in records if r['group'] == 'natural_number' and r['reference'] == str(n)]) for n in range(101)}
    result['natural_number_by_intent'] = {str(c): metric([r for r in records if r['group'] == 'natural_number' and r['command'] == c]) for c in sorted({r['command'] for r in records if r['group'] == 'natural_number'}, key=str)}
    return result


def calibrate(records):
    gate = threshold_on_dev(records)
    if gate['passed']:
        below = [r['confidence'] for r in records if r['prediction'] is not None and r['confidence'] < gate['threshold']]
        if below:
            # Preserve exactly the selected development decisions while placing
            # the cutoff between observations, not on a floating-point score.
            original = gate['threshold']
            gate['threshold'] = (original + max(below))/2
            assert all((r['confidence'] >= original) == (r['confidence'] >= gate['threshold']) for r in records if r['prediction'] is not None)
    gate['boundary_policy'] = 'midpoint to next rejected observed score when available; development decisions unchanged'
    return gate


def evaluate(rows, cache):
    checks()
    catalog = json.loads((DATA/'catalogs.json').read_text())
    templates = numeric_templates(rows)
    dump(DATA/'grammar.json', {'templates': templates, 'training_only': True})
    fixtures = {variant: catalog_fixture(catalog[variant]) for variant in ('fixed', 'expanded')}
    for variant, fixture in fixtures.items():
        dump(DATA/('catalog-'+variant+'.json'), fixture)
    dictionary = lexicon()
    dev, test = queries(rows, 'dev'), queries(rows, 'test')
    for q in dev + test:
        q['in_fixed_catalog'] = q['role'] in ('artist', 'song', 'device') and q['reference'] in catalog['fixed'].get(q['role'], [])
    lock = dict(created_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=digest(DATA/'manifest.jsonl'),
                grammar_sha256=digest(DATA/'grammar.json'), decoder_sha256=digest(ROOT/'scripts/models/phonetic_reader.py'),
                evaluation_sha256=digest(ROOT/'scripts/test/phonetic_eval.py'),
                dictionary_sha256=digest(ROOT/'STT/alternatives/sources/cmudict.dict'),
                legacy_matcher_sha256=digest(ROOT/'scripts/test/parameter_study_eval.py'),
                protocol_sha256=digest(ROOT/'STT/phonetic-v2/PROTOCOL.md'), catalogs={v: f['version'] for v, f in fixtures.items()}, candidates={})
    pending = []
    for name in ('small', 'wide'):
        training = json.loads((OUT/name/'training.json').read_text())
        if training['manifest_sha256'] != lock['manifest_sha256'] or training['graph_sha256'] != digest(OUT/name/'phoneme.onnx'):
            raise ValueError('Model/manifest/graph provenance mismatch')
        session, load_ms = session_for(name)
        outputs = infer(session, dev, cache)
        parity = []
        representatives = {next(q['index'] for q in dev if q['gate'] == g and q['reference'] is not None) for g in {q['gate'] for q in dev}}
        parity_indices = sorted(representatives | set(random.Random(75).sample(sorted(outputs), 32-len(representatives))))
        singles = {}
        for i in parity_indices:
            x = cache[i]
            logits, length = session.run(None, {'features': rearrange(x, 't m -> 1 t m'), 'lengths': np.array([len(x)], dtype=np.int64)})
            singles[i] = logits[0, :int(length[0])]
            error = float(np.max(np.abs(logits[0, :int(length[0])] - outputs[i])))
            if not np.allclose(logits[0, :int(length[0])], outputs[i], atol=1e-4, rtol=1e-5):
                raise ValueError('CPU single/batch inference mismatch')
            parity.append(error)
        dump(OUT/name/'cpu-task-parity.json', dict(cases=len(parity), max_abs_error=max(parity), atol=1e-4, rtol=1e-5,
                                                decoded_and_gated_decisions_checked_separately=True))
        for decoder in ('greedy', 'graph'):
            for variant in ('fixed', 'expanded'):
                start = time.perf_counter()
                matcher = (PhoneticCatalogReader(training['vocabulary'], dictionary, fixtures[variant], templates) if decoder == 'graph'
                           else Matcher('phoneme', training['vocabulary'], catalog[variant]))
                compile_ms = (time.perf_counter()-start)*1000
                records = predict(dev, outputs, matcher, decoder)
                gates = {g: calibrate([r for r in records if r['gate'] == g]) for g in sorted({r['gate'] for r in records})}
                key = name+'_'+decoder+'_'+variant
                single_records = predict([q for q in dev if q['index'] in singles], singles, matcher, decoder)
                batched_records = [r for r in records if r['index'] in singles]
                decision_parity = []
                for single, batch in zip(single_records, batched_records):
                    if single['prediction'] != batch['prediction'] or abs(single['confidence']-batch['confidence']) > 1e-4:
                        raise ValueError('CPU single/batch decoded parameter mismatch')
                    gate = gates[batch['gate']]
                    equal = (single['prediction'] is not None and single['confidence'] >= gate['threshold']) == (batch['prediction'] is not None and batch['confidence'] >= gate['threshold'])
                    if not equal:
                        raise ValueError('CPU single/batch gated decision mismatch')
                    decision_parity.append(equal)
                dump(OUT/(key+'-dev.json'), records)
                lock['candidates'][key] = dict(gates=gates, graph_sha256=training['graph_sha256'], session_load_ms=load_ms,
                                               cpu_single_batch_decisions=len(decision_parity),
                                               catalog_compile_ms=compile_ms, development=groups(records, gates),
                                               graph_states={k: len(g.labels) for k, g in matcher.graphs.items()} if decoder == 'graph' else None,
                                               dictionary_oov=matcher.oov)
                pending.append((key, name, decoder, variant, matcher, gates))
                print('development complete', key, {g: p['passed'] for g, p in gates.items()}, flush=True)
    # No test inference or scoring has happened before this lock is written.
    dump(OUT/'selection-before-test.json', lock)
    result = dict(protocol=lock, hardware=dict(platform=platform.platform(), provider='CPUExecutionProvider', intra_threads=2, inter_threads=1, blas_threads=1,
                  live_microphone=False, physical_pi=False), candidates={})
    outputs_by_model = {}
    for name in ('small', 'wide'):
        session, _ = session_for(name)
        outputs_by_model[name] = infer(session, test, cache)
    for key, name, decoder, variant, matcher, gates in pending:
        records = predict(test, outputs_by_model[name], matcher, decoder)
        dump(OUT/(key+'-test.json'), records)
        result['candidates'][key] = dict(raw=groups(records), gated=groups(records, gates))
        if variant == 'expanded':
            fixed_gates = lock['candidates'][name+'_'+decoder+'_fixed']['gates']
            result['candidates'][key]['fixed_thresholds_after_update'] = groups(records, fixed_gates)
        print('test complete', key, flush=True)
    # Identical natural command PCM, selected without inspecting predictions.
    by_index = {}
    for q in test:
        if q['reference'] is not None and q['kind'] != 'word' and (q['index'] not in by_index or q.get('absolute_brightness')):
            by_index[q['index']] = q
    brightness_indices = {q['index'] for q in test if q.get('absolute_brightness')}
    chosen = sorted(set(random.Random(59).sample(sorted(by_index), min(48, len(by_index)))) | brightness_indices)
    audio = {i: waveform(rows[i]) for i in chosen}
    result['timing_audio_ids'] = [rows[i]['id'] for i in chosen]
    for key, name, decoder, variant, matcher, gates in pending:
        if variant != 'expanded':
            continue
        session, _ = session_for(name)
        elapsed = []
        brightness_elapsed = []
        for repeat in range(4):
            for i in chosen:
                start = time.perf_counter()
                x = stt_features(audio[i])
                logits, length = session.run(None, {'features': rearrange(x, 't m -> 1 t m'), 'lengths': np.array([len(x)], dtype=np.int64)})
                logits = logits[0, :int(length[0])]
                q = by_index[i]
                if decoder == 'graph':
                    matcher.predict(logits, q['role'], q['kind'])
                else:
                    matcher.predict((logits, len(logits)), q['role'])
                if repeat:
                    duration = (time.perf_counter()-start)*1000
                    elapsed.append(duration)
                    if i in brightness_indices:
                        brightness_elapsed.append(duration)
        result['candidates'][key]['timing'] = dict(p50_ms=float(np.percentile(elapsed, 50)), p95_ms=float(np.percentile(elapsed, 95)),
                                                  calls=len(elapsed), includes='frontend, CPU ONNX, decoder/catalog',
                                                  excludes='file reads, speech duration, VAD endpoint wait, cold construction')
        result['candidates'][key]['brightness_timing'] = dict(p50_ms=float(np.percentile(brightness_elapsed, 50)),
                                                            p95_ms=float(np.percentile(brightness_elapsed, 95)),
                                                            calls=len(brightness_elapsed), recordings=len(brightness_indices))
        print('timing', key, result['candidates'][key]['timing'], flush=True)
    result['promotion'] = {'installed': False, 'reason': 'Isolated study; current runtime unchanged. See per-group gates and coverage.'}
    dump(OUT/'results.json', result)
    print('wrote', str(OUT/'results.json'), flush=True)


def scale(rows):
    """Timing only: generated catalog TEXT, never generated speech or accuracy."""
    catalog = json.loads((DATA/'catalogs.json').read_text())
    dictionary = lexicon()
    words = sorted(w for w in dictionary if w.isalpha() and 3 <= len(w) <= 9)
    rng = random.Random(617)
    names = list(catalog['expanded']['artist'])
    while len(names) < 5000:
        name = ' '.join(rng.sample(words, 2))
        if name not in names:
            names.append(name)
    selected = [r for r in rows if r['split'] == 'test' and any(t['role'] == 'artist' for t in r['targets'])][:8]
    audio = [waveform(r) for r in selected]
    result = dict(protocol='Timing only, eight unchanged original artist-query recordings; random dictionary-word text distractors; no accuracy claim or synthetic audio',
                  audio_ids=[r['id'] for r in selected], models={})
    for name in ('small', 'wide'):
        training = json.loads((OUT/name/'training.json').read_text())
        if training['graph_sha256'] != digest(OUT/name/'phoneme.onnx'):
            raise ValueError('Model graph changed')
        session, _ = session_for(name)
        for decoder in ('greedy', 'graph'):
            records = []
            for size in (100, 1000, 5000):
                values = {'artist': names[:size]}
                start = time.perf_counter()
                matcher = (PhoneticCatalogReader(training['vocabulary'], dictionary, catalog_fixture(values), []) if decoder == 'graph'
                           else Matcher('phoneme', training['vocabulary'], values))
                build = (time.perf_counter()-start)*1000
                elapsed = []
                for repeat in range(4):
                    for pcm in audio:
                        start = time.perf_counter()
                        x = stt_features(pcm)
                        logits, lengths = session.run(None, {'features': rearrange(x, 't m -> 1 t m'), 'lengths': np.array([len(x)], dtype=np.int64)})
                        logits = logits[0, :int(lengths[0])]
                        if decoder == 'graph':
                            matcher.predict(logits, 'artist')
                        else:
                            matcher.predict((logits, len(logits)), 'artist')
                        if repeat:
                            elapsed.append((time.perf_counter()-start)*1000)
                records.append(dict(names=size, graph_states=len(matcher.graphs['artist'].labels) if decoder == 'graph' else None,
                                    oov=len(matcher.oov), build_ms=build, p50_ms=float(np.percentile(elapsed, 50)), p95_ms=float(np.percentile(elapsed, 95))))
                print('catalog timing', name, decoder, records[-1], flush=True)
            result['models'][name+'_'+decoder] = records
    dump(OUT/'catalog-scaling.json', result)
