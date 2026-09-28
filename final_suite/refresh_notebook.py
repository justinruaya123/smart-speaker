"""Refresh data and execute the source-controlled review notebook without fitting."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def refresh(output: Path) -> None:
    evidence = json.loads(gzip.decompress((ROOT / 'final_suite/data/evidence.json.gz').read_bytes()))
    inputs = {
        'audit': 'final_suite/reports/dataset_audit.json',
        'scope': 'final_suite/reports/scope_probe.json',
        'recipe': 'visualizer/assets/recipe.json',
        'training': 'experiments/vcm_services_v6/reports/training.json',
        'selection': 'experiments/vcm_services_v6/reports/selection.json',
    }
    for name, relative in inputs.items():
        evidence[name] = json.loads((ROOT / relative).read_text())
    manifest = ROOT / 'experiments/vcm_services_v6/data/probe_manifest.jsonl'
    evidence['sample_manifest'] = json.loads(manifest.read_text().splitlines()[0])
    packed = gzip.compress(json.dumps(evidence, ensure_ascii=False,
                                      separators=(',', ':')).encode(), mtime=0)
    (ROOT / 'final_suite/data/notebook_evidence.json.gz').write_bytes(packed)
    notebook = nbformat.read(ROOT / 'VCM_Final_Review.ipynb', as_version=4)
    nbformat.validate(notebook)
    NotebookClient(notebook, timeout=240, kernel_name='python3',
                   resources={'metadata': {'path': str(ROOT)}}).execute()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'final_suite/reports/VCM_Final_Review.executed.ipynb')
    refresh(parser.parse_args().output)
