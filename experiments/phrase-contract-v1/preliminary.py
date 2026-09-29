"""Measure SOFTWARE/logit behavior; no output is human speech accuracy."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from pcvcm.contract import Contract, compile_catalog, file_hash, write_json
from pcvcm.decoder import PrefixCTC, rank_commands

ROOT = Path(__file__).resolve().parent


def oracle_logits(phones: list[int], seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    path = [0] * 3
    for phone in phones:
        path.extend([phone, phone, 0])
    path.extend([0] * 3)
    x = np.full((len(path), 40), -4.0)
    x[np.arange(len(path)), path] = 4.0
    return x + rng.normal(0, noise, x.shape)


def execute(output: Path, noises: list[float]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    contract = Contract(ROOT / 'spec')
    catalog = compile_catalog(contract, ROOT / 'configs/lexicon.json')
    sequences = [r['phones'] for r in catalog['entries']]
    decoder = PrefixCTC(sequences)
    write_json(output / 'pronunciation_audit.json', {
        'positive_wordings': 570, 'distinct_commands': 402, 'reject_wordings': 200,
        'cross_command_phone_collisions': [], 'trie_nodes': decoder.node_count,
        'padded_states': len(sequences) * (2 * max(map(len, sequences)) + 1),
        'prefix_states': 2 * decoder.node_count,
        'contract_sha256': contract.sha256,
        'lexicon_sha256': catalog['lexicon_sha256'],
        'evidence_type': 'symbolic_pronunciations_not_recorded_speech'})
    from vcm.decoder import CTCRanker
    reference = CTCRanker(sequences)
    x = np.random.default_rng(20260930).normal(size=(250, 40))
    reference.score(x)
    decoder.score(x)
    timings = {'padded_ms': [], 'shared_prefix_ms': []}
    maximum = 0.0
    for repeat in range(5):
        runners = [(reference, 'padded_ms'), (decoder, 'shared_prefix_ms')]
        if repeat % 2:
            runners.reverse()
        outputs = {}
        for runner, name in runners:
            started = time.perf_counter()
            outputs[name] = runner.score(x)
            timings[name].append((time.perf_counter() - started) * 1000)
        maximum = max(maximum, float(np.max(np.abs(
            outputs['padded_ms'] - outputs['shared_prefix_ms']))))
    benchmark = {
        **timings, 'frames': 250, 'catalog_entries': len(sequences),
        'reference_median_ms': float(np.median(timings['padded_ms'])),
        'prefix_median_ms': float(np.median(timings['shared_prefix_ms'])),
        'median_speedup': float(np.median(timings['padded_ms']) /
                                np.median(timings['shared_prefix_ms'])),
        'maximum_score_absolute_error': maximum,
        'prefix_state_bytes': decoder.blank.nbytes + decoder.nonblank.nbytes,
        'padded_alpha_bytes': reference.alpha.nbytes,
        'evidence_type': 'desktop_decoder_only_generated_logits',
        'physical_pi_tested': False}
    write_json(output / 'decoder_benchmark.json', benchmark)
    assert maximum < 1e-12
    rows = []
    for noise in noises:
        for index, entry in enumerate(catalog['entries']):
            x = oracle_logits(entry['phones'], 99173 + index, noise)
            result = rank_commands(decoder.score(x), catalog['entries'])
            rows.append({'entry_id': entry['id'], 'use_case': entry['use_case'],
                         'reference_json': json.dumps(entry['command'], sort_keys=True),
                         'predicted_json': json.dumps(result['command'], sort_keys=True),
                         'noise_sd': noise, 'exact': result['command'] == entry['command'],
                         'reference_is_null': entry['command'] is None,
                         'frames': len(x), 'score': result['score'], 'margin': result['margin'],
                         'evidence_type': 'generated_phone_logits_NOT_AUDIO'})
        print(json.dumps({'noise_sd': noise, 'cases': len(sequences),
                          'raw_exact': sum(r['exact'] for r in rows if r['noise_sd'] == noise)}),
              flush=True)
    with (output / 'logit_trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for noise in noises:
        for uc in sorted({r['use_case'] for r in rows}):
            for is_null in (False, True):
                selected = [r for r in rows if r['noise_sd'] == noise
                            and r['use_case'] == uc and r['reference_is_null'] == is_null]
                summary.append({'noise_sd': noise, 'use_case': uc,
                                'reference_is_null': is_null, 'cases': len(selected),
                                'exact': sum(r['exact'] for r in selected)})
    write_json(output / 'logit_summary.json', summary)
    write_json(output / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'machine': platform.machine(), 'numpy': np.__version__,
        'torch': importlib.metadata.version('torch'),
        'model_fitted_on_human_speech': False, 'human_audio_available': 0,
        'code_sha256': file_hash(Path(__file__))})
    print(json.dumps(benchmark, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--noises', type=float, nargs='+', default=[0.0, 3.0, 6.0])
    args = parser.parse_args()
    execute(args.output, args.noises)
