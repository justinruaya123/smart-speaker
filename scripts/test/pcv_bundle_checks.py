"""Read-only CPU waveform/stream checks for the new research bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/phrase-contract-v1'))

import numpy as np
from einops import rearrange, pack
import onnxruntime as ort
import soundfile as sf

from pcvcm.decoder import PrefixCTC, rank_commands
from vcm.frontend import StreamingFrontend, features
from scripts.train.pcv_public_data import CACHE, sha, save, read_rows
from scripts.train.pcv_calibration import CHILD, WORK, freeze_child
from scripts.test.pcv_public_qualification import verify_parent, verify_bindings
from scripts.runtime.pcv_schema import canonical


def check_bundle(arm, rows):
    bundle = WORK/'evaluation'/arm/'bundle'
    recipe = json.loads((bundle/'recipe.json').read_text())
    policy = json.loads((bundle/'policy.json').read_text())
    lock = json.loads((bundle.parent/'selection-before-test.json').read_text())
    verify_bindings(bundle, {name:recipe['binding'][key] for name,key in
        [('model.onnx','model_sha256'),('catalog.json','catalog_sha256'),
         ('custom-slots.json','custom_slots_sha256'),('normalization.json','normalization_sha256')]})
    assert recipe['binding'] == policy['binding'] == lock['binding']
    assert sha(bundle/'policy.json') == lock['policy_sha256']
    assert sha(bundle.parent/'selection-before-test.json') == recipe['selection_lock_sha256']
    assert sha(bundle/'output.schema.json') == sha(ROOT/'experiments/pcv-human-slots-v1/output.schema.json')
    started = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(bundle/'model.onnx'), options, providers=['CPUExecutionProvider'])
    catalog = json.loads((bundle/'catalog.json').read_text())['entries']
    decoder = PrefixCTC([r['phones'] for r in catalog])
    load_seconds = time.perf_counter()-started
    norm = json.loads((bundle/'normalization.json').read_text())
    states = recipe['state_shapes']
    assert [p.name for p in session.get_inputs()] == ['features', *[f'state_{i}' for i in range(len(states))]]
    assert [p.shape for p in session.get_inputs()[1:]] == states
    assert [p.name for p in session.get_outputs()][:3] == ['ctc','intent','target']
    results, calls_ms = [], []
    # One warm-up is outside timing; every measured utterance starts fresh.
    warm = {'features':np.zeros((1,40,4),dtype='float32')}
    warm.update({f'state_{i}':np.zeros(s,dtype='float32') for i,s in enumerate(states)})
    session.run(None,warm)
    for row in rows:
        verify_bindings(ROOT, {row['audio']:row['audio_sha256'],row['features']:row['features_sha256']})
        pcm, rate = sf.read(row['audio'],dtype='float32')
        assert rate == 16000 and pcm.ndim == 1
        reference_features = features(pcm,norm['mean'],norm['std'])
        feed = {'features':rearrange(reference_features,'c t -> 1 c t')}
        feed.update({f'state_{i}':np.zeros(s,dtype='float32') for i,s in enumerate(states)})
        full = session.run(None,feed)
        full_scores = decoder.score(rearrange(full[0],'1 t c -> t c'))
        full_prediction = rank_commands(full_scores,catalog)
        decoder.reset()
        front = StreamingFrontend(norm['mean'],norm['std'])
        stream_states = [np.zeros(s,dtype='float32') for s in states]
        segments, head_parts = [], [[],[],[]]
        elapsed = {'frontend':0.,'onnx':0.,'decoder':0.}
        for start in range(0,len(pcm),1000):
            began = time.perf_counter()
            segment = front.feed(pcm[start:start+1000])
            elapsed['frontend'] += time.perf_counter()-began
            if not segment.shape[1]:
                continue
            segments.append(segment)
            feed = {'features':rearrange(segment,'c t -> 1 c t')}
            feed.update({f'state_{i}':s for i,s in enumerate(stream_states)})
            began = time.perf_counter()
            actual = session.run(None,feed)
            span = time.perf_counter()-began
            elapsed['onnx'] += span
            calls_ms.append(span*1000)
            stream_states = actual[3:]
            for pieces,head in zip(head_parts,actual[:3],strict=True):
                pieces.append(head)
            began = time.perf_counter()
            decoder.feed(rearrange(actual[0],'1 t c -> t c'))
            elapsed['decoder'] += time.perf_counter()-began
        combined,_ = pack(segments,'c *')
        heads = [pack(parts,'b * c')[0] for parts in head_parts]
        f_error = float(np.max(np.abs(combined-reference_features)))
        head_error = max(float(np.max(np.abs(a-b))) for a,b in zip(heads,full[:3],strict=True))
        state_error = max(float(np.max(np.abs(a-b))) for a,b in zip(stream_states,full[3:],strict=True))
        finite = np.isfinite(full_scores)
        stream_scores = decoder.scores()
        score_error = float(np.max(np.abs(full_scores[finite]-stream_scores[finite])))
        prediction = rank_commands(stream_scores,catalog)
        same = canonical(prediction['command']) == canonical(full_prediction['command'])
        passed = f_error < 2e-5 and head_error < 2e-5 and state_error < 2e-5 and score_error < 2e-5 and same
        results.append({'id':row['id'],'source':row['source'],'audio_seconds':len(pcm)/rate,
            'feature_error':f_error,'head_error':head_error,'state_error':state_error,'score_error':score_error,
            'same_raw_command':same,'pass':passed,'cpu_seconds':elapsed,
            'processing_to_audio_ratio':sum(elapsed.values())/(len(pcm)/rate)})
    return {'arm':arm,'pass':all(r['pass'] for r in results),'rows':results,
        'bundle_binding':recipe['binding'],'session_and_catalog_load_seconds':load_seconds,
        'catalog_entries':len(catalog),'prefix_nodes':decoder.node_count,'onnx_threads':1,
        'pcm_chunk_samples':1000,'onnx_call_count':len(calls_ms),
        'onnx_call_p50_ms':float(np.quantile(calls_ms,.5)),'onnx_call_p95_ms':float(np.quantile(calls_ms,.95)),
        'limitations':['Desktop CPU only; concurrent unrelated GPU work may affect timing',
                      'Disk read, utterance capture, wake/VAD and endpoint wait are excluded',
                      'Development waveform parity is not additional accuracy or physical Pi evidence']}


def main():
    freeze_child()
    verify_parent()
    protocol_path = CHILD/'bundle-check-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert sha(Path(__file__)) == protocol['source_sha256']
    assert sha(CACHE/'prepared/dev.jsonl') == protocol['development_manifest_sha256']
    target = CHILD/'bundle-check-results.json'
    if target.exists():
        raise FileExistsError('Preserve previous check receipt')
    all_rows = {r['id']:r for r in read_rows(CACHE/'prepared/dev.jsonl')}
    rows = [all_rows[i] for i in protocol['ids']]
    results = [check_bundle(arm,rows) for arm in ('human','mixed')]
    verify_parent()
    save(target,{'protocol_sha256':sha(protocol_path),'pass':all(r['pass'] for r in results),'arms':results})
    print(json.dumps({'pass':all(r['pass'] for r in results),'receipt':str(target)}),flush=True)
    return 0 if all(r['pass'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
