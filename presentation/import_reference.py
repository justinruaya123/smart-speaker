"""Import a supplied actual reference-PC receipt, separate from adapter timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iris_publication_data import MODEL_HASHES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    payload = args.receipt.read_bytes()
    result = json.loads(payload)
    if result.get('format') != 'optionb-pc-benchmark-results-v1' or result.get('target') != 'pc':
        raise ValueError('not an actual original reference PC receipt')
    if any(MODEL_HASHES.get(k) != v for k, v in result['model']['binding'].items()):
        raise ValueError('reference graph mismatch')
    if not result.get('publication_provenance', {}).get('measurement_not_quiesced'):
        raise ValueError('this named import is specifically the measured under-load reference run')
    folder = ROOT / 'presentation/evidence'
    target = folder / 'pc-reference-under-load.json'
    target.write_bytes(payload)
    binding = {'source_path': 'experiments/iris-publication-v1/reference-pc-under-load-20261024.json',
               'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
               'note': 'Actual release-owner reference-driver measurement, not the new adapter or an idle-host/end-to-end result.'}
    (folder / 'pc-reference-source.json').write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(binding, indent=2))


if __name__ == '__main__':
    main()
