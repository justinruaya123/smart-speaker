"""Versioned whole-utterance synthesis; evaluation voices never supply fitting data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.runtime.pcv_schema import canonical, normalize_text
from scripts.runtime.vcm_intent_slots_entities import catalog_ids
from scripts.train.vcm_intent_slots_data import rows, sha, write_new
from scripts.test.speaker_demo_suite import pcm_digest

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'vcm-mixed-final-v2'
VOICE_SPLITS = {'train': ['af_sarah', 'am_adam'], 'dev': ['af_sky'],
                'calibration': ['am_echo'], 'enrollment': ['bf_alice', 'bf_lily', 'bm_fable']}


def make_plan(protocol, contract):
    from final.tts_sanity import merge_cases
    from scripts.runtime.vcm_intent_slots_contract import verify_contract
    from scripts.train.vcm_intent_slots_mixed_data import annotate_reference
    verify_contract(contract)
    if (protocol['experiment_id'] != STUDY or protocol['voices'] != VOICE_SPLITS
            or protocol['contract_hash'] != contract['contract_hash']
            or protocol['catalog_hash'] != contract['catalog_hash']
            or protocol['thresholds'] != {'intent': .95, 'slot': .90}
            or protocol['final_tts_fit_allowed'] is not False):
        raise ValueError('Changed mixed-study or reserved final-test contract')
    cases = merge_cases(contract)
    unique = {r['spoken_text']: r for r in cases}
    reserved = set(protocol['final_tts_voices'])
    if reserved & {v for vs in protocol['voices'].values() for v in vs}:
        raise ValueError('Final voices cannot supply fitting recordings')
    selected = catalog_ids(contract)
    selected['light'].discard('light.all')
    enrollment = []
    for role, entities in selected.items():
        for entity in sorted(entities):
            examples = [r for r in unique.values()
                        if (r['command'] or {}).get('slots', {}).get(role+'_id') == entity
                        and annotate_reference({'transcript': r['spoken_text'], 'command': r['command']},
                                               contract)['role_labels'].get(role) == 'present']
            if not examples:
                raise ValueError('No preserved enrollment reference: '+entity)
            case = min(examples, key=lambda r: (len(r['spoken_text']), r['spoken_text']))
            enrollment.append({**case, 'entity_role': role, 'entity_id': entity})
    plan = []
    for split, voices in protocol['voices'].items():
        forms = enrollment if split == 'enrollment' else unique.values()
        for voice in voices:
            for case in forms:
                identity = hashlib.sha256(canonical([STUDY, split, voice, case['spoken_text'],
                                                    case.get('entity_role')]).encode()).hexdigest()
                record = {'id': STUDY+'/'+identity, 'split': split, 'voice_id': voice,
                          'speaker_id': 'Kokoro-82M/'+voice, 'recording_group': identity,
                          'pair_id': identity, 'transcript': case['spoken_text'],
                          'command': case['command'], 'reference_case': case['case_id'],
                          'reference_status': 'supported' if case['command'] is not None else 'unsupported',
                          'kind': 'synthetic', 'role': 'command',
                          'source': protocol['source']['repository'],
                          'source_revision': protocol['source']['revision'],
                          'license': protocol['source']['license'],
                          'supervision_status': 'reference_derived_silver',
                          'acoustic_listening_verified': False, 'qa_eligible': False,
                          'metric_eligible': True, 'speed': protocol['generator']['speed']}
                for key in ('entity_role', 'entity_id'):
                    if key in case:
                        record[key] = case[key]
                plan.append(record)
    return plan


def correction_plan(original_path, protocol, contract):
    correction = json.loads((ROOT/'experiments/vcm-mixed-final-v2/preparation-correction-v1.json').read_text())
    if sha(original_path) != correction['original_generation_plan_sha256']:
        raise ValueError('Changed original generation plan')
    old = json.loads(Path(original_path).read_text())
    old_ids = {r['id'] for r in old}
    complete = make_plan(protocol, contract)
    new = [r for r in complete if r['id'] not in old_ids]
    new_ids = {r['id'] for r in complete}
    removed = [r for r in old if r['id'] not in new_ids]
    if (len(new) != 3 or len(removed) != 3 or any(
            r.get('entity_id') != 'contact.home' or r['split'] != 'enrollment' for r in new+removed)):
        raise ValueError('Unexpected enrollment correction scope')
    return new


def generate(plan_path, protocol_path, assets, output, provider='cuda', correction_of=None):
    import importlib.metadata
    import soundfile as sf
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    from final.tts_sanity import to_pcm16
    protocol = json.loads(Path(protocol_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    contract = json.loads((ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json').read_text())
    expected = (correction_plan(correction_of, protocol, contract) if correction_of
                else make_plan(protocol, contract))
    if canonical(plan) != canonical(expected):
        raise ValueError('Changed or incomplete fitting generation plan')
    if importlib.metadata.version('kokoro-onnx') != protocol['generator']['version']:
        raise ValueError('Unpinned generator package')
    files = {n: Path(assets)/n for n in protocol['generator']['files']}
    if any(sha(path) != protocol['generator']['files'][n] for n, path in files.items()):
        raise ValueError('Generator asset binding mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    ort.set_seed(protocol['generator']['seed'])
    if provider == 'cuda':
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible CUDA GPU required')
        ort.preload_dlls()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if providers[0] not in ort.get_available_providers():
            raise ValueError('CUDA ONNX Runtime unavailable')
    elif provider == 'cpu':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError('Explicit generator provider required')
    session = ort.InferenceSession(str(files['kokoro-v1.0.onnx']), options, providers=providers)
    if provider == 'cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise ValueError('Generator silently lost its requested CUDA provider')
    tts = Kokoro.from_session(session, str(files['voices-v1.0.bin']))
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT/'.cache'):
        raise ValueError('Generated payload outside ignored study cache')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('native', 'audio'):
        (output/name).mkdir()
    started = time.perf_counter()
    write_new(output/'run-identity.json', {'plan': {'path': str(plan_path), 'sha256': sha(plan_path)},
              'protocol': {'path': str(protocol_path), 'sha256': sha(protocol_path)},
              'provider': provider, 'session_providers': session.get_providers(),
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'final_tts_data_used': False, 'generator': protocol['generator'],
              'source_sha256': sha(__file__),
              'pcm_conversion_source_sha256': sha(ROOT/'final/tts_sanity.py'),
              'correction_of': str(correction_of) if correction_of else None})
    completed = []
    with (output/'generation.jsonl').open('x', encoding='utf8') as log:
        for n, item in enumerate(plan):
            identifier = item['id'].split('/')[-1]
            row = {**item, 'status': 'generation_fault'}
            try:
                audio, rate = tts.create(item['transcript'], voice=item['voice_id'],
                                         speed=item['speed'], lang='en-us', trim=False)
                native = output/'native'/f'{identifier}.npy'
                np.save(native, np.asarray(audio))
                row.update(native_audio=str(native), native_sha256=sha(native), native_rate=rate)
                native_wav = native.with_suffix('.wav')
                sf.write(native_wav, audio, rate, subtype='FLOAT')
                row.update(native_wav=str(native_wav), native_wav_sha256=sha(native_wav))
                pcm = to_pcm16(audio, rate)
                path = output/'audio'/f'{identifier}.wav'
                sf.write(path, pcm, 16000, subtype='PCM_16')
                decoded, decoded_rate = sf.read(path, dtype='float32')
                row.update(audio=str(path), sha256=sha(path), pcm_sha256=pcm_digest(decoded, decoded_rate),
                           samples=len(decoded), sample_rate=decoded_rate, channels=1,
                           technical_qa='finite_nonclipped_nonsilent_mono16k_capture_length', status='generated')
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
            completed.append(row)
            log.write(canonical(row)+'\n'); log.flush()
            if (n+1) % 50 == 0:
                print('GENERATED', n+1, '/', len(plan), 'seconds', round(time.perf_counter()-started, 1), flush=True)
    write_new(output/'manifest.json', completed)
    report = {'planned': len(plan), 'status_counts': dict(Counter(r['status'] for r in completed)),
              'manifest_sha256': sha(output/'manifest.json'), 'seconds': time.perf_counter()-started,
              'independent_listening_performed': False, 'final_tts_data_used': False}
    write_new(output/'generation-receipt.json', report)
    print(json.dumps(report), flush=True)
    return report


def verify_generation(manifest_path, protocol_path):
    manifest_path = Path(manifest_path)
    identity = json.loads((manifest_path.parent/'run-identity.json').read_text())
    receipt = json.loads((manifest_path.parent/'generation-receipt.json').read_text())
    plan_binding = identity['plan']
    if (sha(plan_binding['path']) != plan_binding['sha256']
            or sha(protocol_path) != identity['protocol']['sha256']
            or sha(manifest_path) != receipt['manifest_sha256']):
        raise ValueError('Changed generation plan, protocol or manifest binding')
    plan = json.loads(Path(plan_binding['path']).read_text())
    manifest = json.loads(manifest_path.read_text())
    index = {r['id']: r for r in manifest}
    if (len(index) != len(manifest) or len(plan) != receipt['planned']
            or len(plan) != len(manifest) or set(index) != {r['id'] for r in plan}
            or dict(Counter(r['status'] for r in manifest)) != receipt['status_counts']):
        raise ValueError('Incomplete or duplicate generated corpus')
    for row in plan:
        actual = index[row['id']]
        if any(canonical(actual.get(k)) != canonical(v) for k,v in row.items()):
            raise ValueError('Generated record differs from frozen plan')
        if actual['status'] not in ('generated', 'generation_fault'):
            raise ValueError('Unknown generation status')
        for field, hash_field in [('native_audio','native_sha256'), ('native_wav','native_wav_sha256')]:
            if actual.get(field) and sha(actual[field]) != actual[hash_field]:
                raise ValueError('Changed preserved native generation')
    return manifest


def prepare(protocol_path, generated, output, correction_generated=None):
    from scripts.train.vcm_intent_slots_mixed_data import annotate_reference, audit_views
    protocol = json.loads(Path(protocol_path).read_text())
    contract = json.loads((ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json').read_text())
    base = json.loads((ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json').read_text())
    for binding in base['views'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Changed original frozen human/regression view')
    generated_paths = [Path(generated)] + ([Path(correction_generated)] if correction_generated else [])
    generated = [r for path in generated_paths for r in verify_generation(path, protocol_path)]
    if len({r['id'] for r in generated}) != len(generated):
        raise ValueError('Repeated original/correction generation identity')
    annotated = [annotate_reference(r, contract) for r in generated if r['status'] == 'generated']
    default_enrollment = [r for r in annotated if r['split']=='enrollment'
                          and r['role_labels'].get(r['entity_role']) != 'present']
    bad_ids = {r['id'] for r in default_enrollment}
    good = [r for r in annotated if r['id'] not in bad_ids]
    wanted = catalog_ids(contract); wanted['light'].discard('light.all')
    enrolled = Counter((r['entity_role'],r['entity_id']) for r in good if r['split']=='enrollment')
    if any(enrolled[(role,entity)] != 3 for role,names in wanted.items() for entity in names):
        raise ValueError('Missing three named-present enrollment recordings per entity')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    views = {}
    for split in ('train', 'dev', 'calibration', 'enrollment'):
        human = rows(base['views'][split]['path']) if split in base['views'] else []
        subset = human + [r for r in good if r['split']==split]
        path = output/(split+'.jsonl')
        with path.open('x', encoding='utf8') as stream:
            for row in subset:
                stream.write(canonical(row)+'\n')
        views[split] = {'path': str(path.resolve()), 'sha256': sha(path)}
    for split in ('test', 'holdout'):
        views[split] = base['views'][split]
    fresh = ROOT/'.cache/vcm-mixed-final-v2/fresh-human-01/fresh-human.jsonl'
    if not fresh.is_file():
        raise ValueError('Missing independently frozen fresh-person evaluation manifest')
    fresh_audit = fresh.parent/'audit.json'
    fresh_receipt = json.loads(fresh_audit.read_text())
    if (fresh_receipt['manifest']['sha256'] != sha(fresh)
            or fresh_receipt['contract_hash'] != contract['contract_hash']
            or fresh_receipt['VCM_results_accessed'] is not False
            or fresh_receipt['source']['sha256'] != protocol['evaluation']['fresh_human_source']['tar_sha256']
            or fresh_receipt['source']['revision'] != protocol['evaluation']['fresh_human_source']['revision']):
        raise ValueError('Changed frozen fresh-person source/reference/manifest binding')
    views['fresh_human'] = {'path': str(fresh), 'sha256': fresh_receipt['manifest']['sha256']}
    inputs = {'audio_root': base['audio_root'], 'views': views,
              'original_inputs': {'path': str(ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json'),
                                  'sha256': sha(ROOT/'experiments/vcm-intent-slots-v1/local-inputs.json')},
              'protocol': {'path': str(Path(protocol_path).resolve()), 'sha256': sha(protocol_path)},
              'contract': {'path': str(ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json'),
                           'sha256': sha(ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json')},
              'generation_faults_retained': [r['id'] for r in generated if r['status'] != 'generated'],
              'default_only_enrollment_excluded_retained': sorted(bad_ids),
              'generation_bindings': [{'path': str(path.resolve()), 'sha256': sha(path),
                                       'receipt_sha256': sha(path.parent/'generation-receipt.json'),
                                       'identity_sha256': sha(path.parent/'run-identity.json')} for path in generated_paths],
              'preparation_correction': {'path': str(ROOT/'experiments/vcm-mixed-final-v2/preparation-correction-v1.json'),
                                        'sha256': sha(ROOT/'experiments/vcm-mixed-final-v2/preparation-correction-v1.json')},
              'exclusion_rule': 'technical waveform failures only, before fitting; no VCM results',
              'acoustic_listening_verified': False}
    inputs['fresh_audit'] = {'path': str(fresh_audit), 'sha256': sha(fresh_audit)}
    write_new(output/'inputs.json', inputs)
    audit_views({s:b['path'] for s,b in views.items()}, {'views': views}, output/'audit',
                protocol=protocol, contract=contract, audio_root=base['audio_root'])
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'generate', 'prepare'])
    parser.add_argument('--protocol', type=Path, default=ROOT/'experiments/vcm-mixed-final-v2/protocol.json')
    parser.add_argument('--plan', type=Path, default=ROOT/'.cache/vcm-mixed-final-v2/generation-plan.json')
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--generated', type=Path)
    parser.add_argument('--correction-of', type=Path)
    parser.add_argument('--correction-generated', type=Path)
    parser.add_argument('--provider', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if args.command == 'plan':
        protocol = json.loads(args.protocol.read_text())
        contract = json.loads((ROOT/'.cache/vcm-mixed-final-v2/slot-contract.json').read_text())
        value = (correction_plan(args.correction_of, protocol, contract) if args.correction_of
                 else make_plan(protocol, contract))
        write_new(args.output, value)
        print(json.dumps({'plan_sha256': sha(args.output), 'counts': dict(Counter(r['split'] for r in value))}))
    elif args.command == 'generate':
        generate(args.plan, args.protocol, args.assets, args.output, args.provider, args.correction_of)
    else:
        print(json.dumps(prepare(args.protocol, args.generated, args.output, args.correction_generated)))


if __name__ == '__main__':
    main()
