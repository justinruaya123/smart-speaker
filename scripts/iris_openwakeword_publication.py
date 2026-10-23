"""One-candidate CPU-only openWakeWord study; never installs/promotes a model.

Usage (the corpus is the existing native-wake study, read-only):
  python -m scripts.iris_openwakeword_publication acquire
  python -m scripts.iris_openwakeword_publication lock --corpus-root PATH
  python -m scripts.iris_openwakeword_publication evaluate --corpus-root PATH
  python -m scripts.iris_openwakeword_publication verify --corpus-root PATH

Run in the unchanged runtime lock for evaluation. Tests may use the original
root environment. No TTS, fitting, microphone, playback, actions or imports
of external repository code occur. All outputs are write-once.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

from scripts.iris_wake_publication import object_digest, read_case_pcm, panel_metrics, write_once

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/iris-openwakeword-publication-v1'
ASSETS = ROOT / '.cache/iris-openwakeword-publication-v1/models'
PROTOCOL_SHA = '716f6c4c5661426d4024c6a68036c67eb4d785466b02d74ed5b417f94b7a9279'
COHORTS = (
    ('baseline', 'baseline-protocol.json', 'baseline',
     '7de025d6159634659a591857ade090c88971703b619ced221a1ba2b48ac236e1'),
    ('companion-test', 'companion-protocol.json', 'test',
     'e4ae7a783d9168a514e09aa0709ae6f7f0cfb6c413c2be0e1263a0c05a77cdbf'),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_membership(cases, records):
    expected = [c['id'] for c in cases]
    observed = [c['id'] for c in records]
    if not expected or len(set(expected)) != len(expected) or observed != expected:
        raise ValueError('Changed, empty, duplicated, missing or reordered cohort')


def cohort_inputs(corpus_root):
    if digest(EXPERIMENT / 'protocol.json') != PROTOCOL_SHA:
        raise ValueError('Changed frozen candidate protocol')
    groups = []
    for name, filename, split, expected in COHORTS:
        path = ROOT / 'experiments/iris-wake-publication-v1' / filename
        if digest(path) != expected:
            raise ValueError('Changed canonical cohort protocol')
        cases = [r for r in json.loads(path.read_text())['cases'] if r['split'] == split]
        if (len(cases), sum(c['wake_positive'] for c in cases)) != (100, 32):
            raise ValueError('Changed canonical cohort counts')
        cache = Path(corpus_root) / ('baseline' if name == 'baseline' else 'companion')
        records = []
        for case in cases:
            receipt_path = cache / 'receipts' / (case['id'] + '.json')
            receipt = json.loads(receipt_path.read_text())
            wav = cache / 'audio' / (case['id'] + '.wav')
            if (receipt['id'] != case['id'] or receipt['protocol_sha256'] != expected
                    or receipt['case_sha256'] != object_digest(case)
                    or digest(wav) != receipt['wav_sha256']):
                raise ValueError('Changed original audio/receipt identity')
            records.append(dict(id=case['id'], case=case, receipt=receipt,
                                receipt_sha256=digest(receipt_path)))
        validate_membership(cases, records)
        groups.append(dict(name=name, cases=cases, records=records, cache=cache))
    return groups


def acquire(assets=ASSETS):
    # Only public immutable URLs from the pinned provenance register. No login,
    # account creation, credential lookup, guessed direct-download or installer.
    import urllib.request
    from scripts.runtime.iris_openwakeword import FILES
    assets = Path(assets)
    assets.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((EXPERIMENT / 'provenance.json').read_text())
    for name, expected in FILES.items():
        source = next(r for r in provenance if r['path'] == 'public/wakeword/' + name)
        path = assets / name
        if path.exists() and digest(path) == expected:
            continue
        with urllib.request.urlopen(source['url'], timeout=30) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) != source['bytes'] or hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Public model download identity mismatch')
        if path.exists():
            raise ValueError('Refusing to overwrite changed model')
        with path.open('xb') as stream:
            stream.write(payload)
    return dict(status='public_assets_hash_verified', files=FILES)


def lock(corpus_root, assets=ASSETS):
    from scripts.runtime.iris_openwakeword import FILES
    groups = cohort_inputs(corpus_root)  # hashes/metadata only, no decoded PCM
    for name, expected in FILES.items():
        if digest(Path(assets) / name) != expected:
            raise ValueError('Changed external graph')
    value = dict(schema_version=1, frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 protocol_sha256=PROTOCOL_SHA, graph_sha256=FILES,
                 source_sha256={p: digest(ROOT / p) for p in (
                     'scripts/runtime/iris_openwakeword.py',
                     'scripts/iris_openwakeword_publication.py',
                     'scripts/iris_wake_publication.py',
                     'scripts/test/iris_openwakeword_reference.py')},
                 cohort_sha256={g['name']: object_digest(g['records']) for g in groups},
                 candidate_pcm_decoded_or_inferred_before_lock=False,
                 exposed_regression_not_blind=True, training_performed=False,
                 qualified=False, policy=dict(threshold=.5, release=.3, consecutive=2, cooldown_samples=16000))
    write_once(EXPERIMENT / 'selection-before-regression.json', value)
    return value


def checked_lock(corpus_root, assets=ASSETS):
    from scripts.runtime.iris_openwakeword import FILES
    saved = json.loads((EXPERIMENT / 'selection-before-regression.json').read_text())
    groups = cohort_inputs(corpus_root)
    if (saved['protocol_sha256'] != PROTOCOL_SHA or saved['graph_sha256'] != FILES
            or saved['cohort_sha256'] != {g['name']: object_digest(g['records']) for g in groups}
            or any(digest(ROOT / p) != sha for p, sha in saved['source_sha256'].items())
            or any(digest(Path(assets) / p) != sha for p, sha in FILES.items())):
        raise ValueError('Changed locked candidate/source/cohort')
    return groups


def evaluate(corpus_root, assets=ASSETS):
    import numpy as np
    import onnxruntime as ort
    from scripts.runtime.iris_openwakeword import HeyIrisOpenWakeWord
    groups = checked_lock(corpus_root, assets)
    if (EXPERIMENT / 'results.json').exists():
        raise ValueError('Regression already completed; never retune/rerun into old evidence')
    write_once(EXPERIMENT / 'regression-started.json', dict(
        selection_sha256=digest(EXPERIMENT / 'selection-before-regression.json'),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    detector = HeyIrisOpenWakeWord(assets)
    begin = time.perf_counter()
    panels = []
    for group in groups:
        records, scores = [], []
        for case, binding in zip(group['cases'], group['records']):
            pcm = read_case_pcm(case, binding['receipt'], group['cache'])
            detector.reset()
            padded = np.concatenate((np.zeros(32000, np.float32), pcm, np.zeros(48000, np.float32)))
            hits, trace = [], []
            for offset in range(0, len(padded), 640):
                triggered = detector.push(padded[offset:offset + 640])
                trace.extend(detector.last_scores)
                if triggered == 1.0:
                    hits.append((detector.samples - 32000) / 16000)
            record = dict(id=case['id'], voice=case['voice'], text=case['text'],
                          positive=case['wake_positive'], synthetic=True,
                          wav_sha256=binding['receipt']['wav_sha256'],
                          decoded_pcm_s16le_sha256=binding['receipt']['decoded_pcm_s16le_sha256'],
                          hits=len(hits), hit_seconds=hits, seconds=len(pcm) / 16000,
                          max_score=max((v for _, v in trace), default=0), score_count=len(trace))
            records.append(record)
            scores.append(dict(id=case['id'], samples_scores=trace))
        validate_membership(group['cases'], records)
        metrics = panel_metrics(group['cases'], records)
        write_once(EXPERIMENT / (group['name'] + '-cases.json'), records)
        cache_path = ASSETS.parent / (group['name'] + '-scores.json')
        write_once(cache_path, scores)
        panels.append(dict(name=group['name'], metrics=metrics,
                           cases_sha256=digest(EXPERIMENT / (group['name'] + '-cases.json')),
                           score_trace_sha256=digest(cache_path)))
    passed = all(p['metrics']['synthetic_smoke_pass'] for p in panels)
    result = dict(schema_version=1, status='acoustic_gate_passed' if passed else 'blocked_acoustic_gate',
                  gate_passed=passed, promoted=False, installed=False, qualified=False,
                  lock_sha256=digest(EXPERIMENT / 'selection-before-regression.json'),
                  panels=panels, python=sys.version, numpy=np.__version__,
                  onnxruntime=ort.__version__, platform=platform.platform(),
                  cpu_threads=1, wall_seconds=time.perf_counter() - begin,
                  scope='Previously exposed synthetic regression; no independent humans/Pi/live microphone',
                  no_training=True, no_environment_mutation=True, no_test_retuning=True)
    write_once(EXPERIMENT / 'results.json', result)
    return result


def verify(corpus_root, assets=ASSETS):
    groups = checked_lock(corpus_root, assets)
    result = json.loads((EXPERIMENT / 'results.json').read_text())
    if result['lock_sha256'] != digest(EXPERIMENT / 'selection-before-regression.json'):
        raise ValueError('Changed result selection binding')
    for group, panel in zip(groups, result['panels']):
        path = EXPERIMENT / (group['name'] + '-cases.json')
        rows = json.loads(path.read_text())
        validate_membership(group['cases'], rows)
        if (panel['name'] != group['name'] or panel['cases_sha256'] != digest(path)
                or panel['metrics'] != panel_metrics(group['cases'], rows)):
            raise ValueError('Changed case-level results/metrics')
    if len(result['panels']) != len(groups) or result['gate_passed'] != all(
            p['metrics']['synthetic_smoke_pass'] for p in result['panels']):
        raise ValueError('Changed gate result')
    return dict(status='regression_receipts_verified', gate_passed=result['gate_passed'],
                results_sha256=digest(EXPERIMENT / 'results.json'), promoted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('acquire', 'lock', 'evaluate', 'verify'))
    parser.add_argument('--corpus-root', type=Path)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    args = parser.parse_args()
    if args.command == 'acquire':
        result = acquire(args.assets)
    else:
        if args.corpus_root is None:
            parser.error('--corpus-root is required; existing corpus is read-only')
        result = globals()[args.command](args.corpus_root, args.assets)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
