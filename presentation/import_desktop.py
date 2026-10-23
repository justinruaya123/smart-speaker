"""Attach an actual audited product desktop receipt, not an estimated metric."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.iris_publication_data import MODEL_HASHES, validate_measurement


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    source = args.receipt.resolve()
    raw = json.loads(source.read_bytes())
    measurement, detail, parity = raw['measurement'], raw['detail'], raw['independent_parity']
    core = source.parents[2]
    hashes = {name: digest(core / 'submission/models/optionb-tiny' / name) for name in ('model.onnx', 'policy.json')}
    if hashes != {name: MODEL_HASHES[name] for name in hashes} or measurement['model_sha256'] != hashes['model.onnx']:
        raise ValueError('actual product graph/policy identity mismatch')
    if digest(core / 'scripts/runtime/iris_epoch44.py') != detail['runtime_source_sha256']:
        raise ValueError('actual product runtime source mismatch')
    if not all(parity.get(key) is True for key in ('full_typed_outputs_match_frozen_gold', 'raw_and_accepted_decisions_match_frozen_gold')):
        raise ValueError('missing independent frozen-gold decision parity')
    if (measurement['source_n'], measurement['observation_n'], parity['wav_bytes_verified'], parity['decoded_pcm_verified'], parity['all_recording_repeat_pairs_verified']) != (1146, 3438, 1146, 1146, 3438):
        raise ValueError('original recording/PCM/repetition identity mismatch')
    if measurement['observation_n'] % measurement['source_n']:
        raise ValueError('nonintegral timing repeats')
    folder = ROOT / 'presentation/evidence'
    target = folder / 'desktop-product.json'
    target.write_bytes(source.read_bytes())
    result = {
        'schema_version': 1, 'format': 'iris-publication-measurement-v1', 'target': 'desktop',
        'label': '2026-10-24 product-lock adapter / not quiesced',
        'hashes': hashes, 'hardware': {**measurement['hardware'], 'cpu': measurement['hardware']['processor']},
        'timing_scope': 'recognize_pcm', 'timing_excludes': detail['timing_excludes'],
        'model_scope': measurement['model_scope'],
        'latency': {'p50_ms': detail['latency_ms']['end_to_end']['p50'], 'p95_ms': detail['latency_ms']['end_to_end']['p95']},
        'original_recordings': measurement['source_n'], 'timed_calls': measurement['observation_n'],
        'repeats': measurement['observation_n'] // measurement['source_n'],
        'receipt_path': 'presentation/evidence/desktop-product.json', 'receipt_sha256': digest(source),
        'measured_at': measurement['utc'], 'measurement_not_quiesced': raw['measurement_not_quiesced'],
        'environment': raw['environment'], 'accepted': measurement['accepted'],
        'ambiguous_unscored': measurement['excluded_reference_n'], 'independent_parity': parity,
        'peak_process_rss_mib': max(item['peak_rss_mib'] for item in detail['resource_samples']),
        'resource_scope': detail['resource_scope'], 'candidate_revision': raw['candidate_revision'],
        'policy_identity_evidence': 'Exact core policy bytes verified at import; the hash-bound actual adapter source pins this policy. The incoming wrapper does not itself contain a policy hash.',
    }
    validate_measurement(result, 'desktop')
    (folder / 'desktop-measurement.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'receipt_sha256': result['receipt_sha256'], 'latency': result['latency'], 'peak_process_rss_mib': result['peak_process_rss_mib']}, indent=2))


if __name__ == '__main__':
    main()
