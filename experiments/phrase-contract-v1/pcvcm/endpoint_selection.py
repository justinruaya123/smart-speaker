"""Four-value development-only endpoint selection with reviewed end times."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from einops import rearrange
from .contract import compile_catalog, file_hash, read_jsonl, write_json
from .data import below
from .endpoint import Endpoint, EndpointConfig
from .evidence import audit_partitions, validate_reference

THRESHOLDS = (-50, -45, -40, -35)


def validate_pause_coverage(rows):
    observed = {r.get('pause_category') for r in rows if r.get('reviewed_internal_pauses')}
    if not {'number', 'name', 'negation'} <= observed:
        raise ValueError('Curated number/name/negation internal-pause evidence required')


def choose_threshold(records):
    if len(records) != 4 or {r['dbfs'] for r in records} != set(THRESHOLDS):
        raise ValueError('Evaluate exactly the four declared endpoint thresholds')
    return min(records, key=lambda r: (r['misses'],
        r['median_absolute_delay_ms'] if r['median_absolute_delay_ms'] is not None else float('inf'),
        abs(r['dbfs'] + 45), r['dbfs']))


def evaluate_endpoint(pcm, row, dbfs, engine=None):
    end = row.get('speech_end_sample')
    if row.get('split') != 'dev' or type(end) is not int or not 0 < end <= len(pcm):
        raise ValueError('Development-only reviewed speech end sample required')
    if dbfs not in THRESHOLDS:
        raise ValueError('Undeclared threshold')
    endpoint = Endpoint(EndpointConfig(rms_threshold=10 ** (dbfs / 20)))
    if engine is not None:
        engine.reset(); engine.endpoint = endpoint
    decision = None
    for start in range(0, len(pcm), 640):
        if engine is None:
            endpoint.feed(pcm[start:start + 640], start)
        else:
            decision = engine.feed(pcm[start:start + 640], start)
        if endpoint.end_sample is not None or endpoint.fault:
            break
    detected = endpoint.end_sample
    premature = False
    for start, stop in row.get('reviewed_internal_pauses', []):
        if not 0 <= start < stop < end or stop - start > 9600:
            raise ValueError('Reviewed internal pause must be inside speech and at most 600 ms')
        premature |= detected is not None and detected <= stop
    return {'recording_id': row.get('recording_id'), 'dbfs': dbfs,
            'endpoint_sample': detected, 'reference_end_sample': end,
            'miss': detected is None or detected < end or endpoint.fault is not None,
            'delay_ms': (detected - end) / 16 if detected is not None else None,
            'premature_pause': premature, 'fault': endpoint.fault,
            'decision': decision, 'diagnostic': dict(engine.diagnostic) if engine else None}


class TorchEncoder:
    """Reference encoder for endpoint selection/parity only; never shipped."""
    def __init__(self, model):
        self.model = model.cpu().eval(); self.reset()

    def reset(self):
        self.states = None

    def __call__(self, features):
        import torch
        with torch.no_grad():
            logits, self.states = self.model(torch.from_numpy(rearrange(features, 'c t -> 1 c t')), self.states)
        return rearrange(logits, '1 c t -> t c').numpy()


def select_endpoint(run, manifest, audio_root, output, contract, lexicon):
    import soundfile as sf
    import torch
    from vcm.model import CausalVCM
    from .runtime import StreamEngine
    from .training import native_cpu_convolution
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads((run / 'selection.json').read_text())
    if selection['status'] != 'DEVELOPMENT_SELECTED':
        raise ValueError('Endpoint selection requires the development-selected checkpoint')
    checkpoint = Path(selection['winner']['checkpoint'])
    if file_hash(checkpoint) != selection['winner']['checkpoint_sha256']:
        raise ValueError('Selected weights changed')
    rows = read_jsonl(manifest)
    audit_partitions(rows)
    for row in rows:
        validate_reference(row, contract)
        if row.get('split') != 'dev' or row.get('human_reviewed') is not True or row.get('synthetic') is not False:
            raise ValueError('Only reviewed human development audio is permitted')
    annotated = [r for r in rows if r.get('nonspeech') is not True]
    validate_pause_coverage(annotated)
    if {r.get('phrase_id') for r in rows if r['reference_class'] == 'supported_positive'} != set(contract.by_id):
        raise ValueError('Endpoint selection requires the full development suite')
    norm = json.loads((run / 'normalization.json').read_text())
    model = CausalVCM()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['weights'])
    engine = StreamEngine(contract, compile_catalog(contract, lexicon), TorchEncoder(model), norm,
        EndpointConfig(), {'binding': 'endpoint-development', 'enabled': False}, 'endpoint-development')
    output.mkdir(parents=True)
    results = []
    with native_cpu_convolution():
        for dbfs in THRESHOLDS:
            trials = []
            for row in annotated:
                path = below(audio_root, row['audio'])
                if file_hash(path) != row['audio_sha256']:
                    raise ValueError('Original audio identity changed')
                pcm, rate = sf.read(path, dtype='float32')
                if rate != 16000 or pcm.ndim != 1 or not np.isfinite(pcm).all() or hashlib.sha256(pcm.tobytes()).hexdigest() != row['pcm_sha256']:
                    raise ValueError('PCM identity/format mismatch')
                trials.append(evaluate_endpoint(pcm, row, dbfs, engine))
            write_json(output / f'trials-{abs(dbfs)}.json', trials)
            delays = [abs(t['delay_ms']) for t in trials if t['delay_ms'] is not None]
            results.append({'dbfs': dbfs, 'misses': sum(t['miss'] for t in trials),
                'premature_pauses': sum(t['premature_pause'] for t in trials),
                'median_absolute_delay_ms': statistics.median(delays) if delays else None,
                'trials_sha256': file_hash(output / f'trials-{abs(dbfs)}.json')})
    winner = choose_threshold(results)
    config = asdict(EndpointConfig(rms_threshold=10 ** (winner['dbfs'] / 20)))
    write_json(output / 'endpoint.json', config)
    receipt = {'status': 'ENDPOINT_SELECTED' if winner['misses'] == 0 and winner['premature_pauses'] == 0 else 'BLOCKED_ENDPOINT',
               'split': 'dev', 'candidates': results, 'winner': winner,
               'manifest_sha256': file_hash(manifest), 'checkpoint_sha256': file_hash(checkpoint),
               'selection_sha256': file_hash(run / 'selection.json'),
               'endpoint_sha256': file_hash(output / 'endpoint.json'), 'qualified': False}
    write_json(output / 'selection.json', receipt)
    return receipt
