"""Exhaustive generated-logit decoder test, explicitly NOT acoustic evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

from pcvcm.contract import Contract, canonical, compile_catalog, write_json
from pcvcm.decoder import PrefixCTC, rank_commands
from preliminary import oracle_logits

ROOT = Path(__file__).resolve().parent


def run(start: int, stop: int, noise: float, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    contract = Contract(ROOT / 'spec')
    catalog = compile_catalog(contract, ROOT / 'configs/lexicon.json')
    entries = catalog['entries']
    decoder = PrefixCTC([r['phones'] for r in entries])
    rows = []
    started = time.perf_counter()
    fields = ['entry_id', 'use_case', 'expected', 'predicted', 'exact',
              'noise_sd', 'frames', 'score', 'margin', 'evidence_type']
    with (output / 'per_case.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(start, min(stop, len(entries))):
            entry = entries[index]
            logits = oracle_logits(entry['phones'], 99173 + index, noise)
            prediction = rank_commands(decoder.score(logits), entries)
            row = dict(entry_id=entry['id'], use_case=entry['use_case'],
                       expected=canonical(entry['command']),
                       predicted=canonical(prediction['command']),
                       exact=canonical(entry['command']) == canonical(prediction['command']),
                       noise_sd=noise, frames=len(logits),
                       score=prediction['score'], margin=prediction['margin'],
                       evidence_type='generated_logits_NOT_AUDIO')
            rows.append(row)
            writer.writerow(row)
            stream.flush()
    summary = {'cases': len(rows), 'raw_exact': sum(r['exact'] for r in rows),
               'range': [start, stop], 'noise_sd': noise,
               'elapsed_seconds': time.perf_counter() - started,
               'human_recordings': 0, 'accepted_precision': None,
               'final_model': False, 'evidence_type': 'generated_logits_NOT_AUDIO'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=770)
    parser.add_argument('--noise', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.start, args.stop, args.noise, args.output)
