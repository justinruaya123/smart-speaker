"""E18 development/export only. Selection is frozen here before held-out testing."""
import copy
import json
import time
import numpy as np
import torch

from .prototype_data import ROOT, EXP, DATA, CACHE, dataset, verify
from .complete_data import digest, save
from .complete_training import score_development, ExportModel
from .slot_model import SlotVCM


def validate(arm):
    import onnx
    from .audio import load_audio, features
    from .inference import Recognizer
    verify()
    if (EXP / (arm + '-validation.json')).exists(): raise ValueError('Validation already frozen')
    protocol = json.loads((EXP / 'protocol.json').read_text(encoding='utf-8'))
    folder = CACHE / arm
    ck = torch.load(folder / 'vcm.pt', map_location='cpu', weights_only=False)
    torch.set_num_threads(4)
    model = SlotVCM(**ck['config']).eval(); model.load_state_dict(ck['model'], strict=True)
    wrapper = ExportModel(model).eval()
    names = ['logits', *['slot_' + n for n in ck['config']['vocabulary']], 'embedding']
    graph = folder / 'vcm.fp32.onnx'
    torch.onnx.export(wrapper, (torch.randn(1, 40, 160), torch.tensor([160])), str(graph),
        input_names=['features', 'lengths'], output_names=names, opset_version=17, dynamo=False,
        dynamic_axes={'features': {0:'batch', 2:'frames'}, 'lengths': {0:'batch'}, **{n:{0:'batch'} for n in names}})
    exported = onnx.load(graph)
    for key, value in [('vcm_labels', ck['labels']), ('slot_vocabulary', ck['config']['vocabulary'])]:
        prop = exported.metadata_props.add(); prop.key = key; prop.value = json.dumps(value)
    onnx.checker.check_model(exported); onnx.save(exported, graph)
    bundle = json.loads((ROOT / protocol['initial_bundle']).read_text(encoding='utf-8'))
    bundle['files'] = {'vcm.fp32.onnx': {'bytes': graph.stat().st_size, 'sha256': digest(graph)}}
    bundle['vcm']['checkpoint_sha256'] = digest(folder / 'vcm.pt')
    bundle['installation'] = {'status': 'research_candidate_not_installed', 'experiment': EXP.name, 'arm': arm}
    bundle['experiment'] = {'status': 'development_selected', 'seed': ck['seed'], 'epoch': ck['epoch'], 'protocol_sha256': digest(EXP / 'protocol.json')}
    save(folder / 'bundle.json', bundle)
    recognizer = Recognizer(folder, threads=2)
    rows = dataset({'dev', 'calibration'})
    outputs = [torch.empty(len(rows), n) for n in [len(ck['labels']), *map(len, ck['config']['vocabulary'].values())]]
    parity, last = 0., time.monotonic()
    for i, row in enumerate(rows):
        pcm, _ = load_audio(ROOT / row['audio'])
        x = recognizer.normalize(features(pcm)); lengths = np.array([x.shape[-1]], dtype=np.int64)
        values = recognizer.vcm.run(names, {'features': x, 'lengths': lengths})
        for out, value in zip(outputs, values): out[i] = torch.from_numpy(value[0])
        if i < 20:
            with torch.inference_mode(): expected = wrapper(torch.from_numpy(x), torch.from_numpy(lengths))
            parity = max(parity, max(float(np.max(np.abs(a.numpy() - b))) for a,b in zip(expected, values)))
        if time.monotonic() - last > 20:
            print(f'CPU_VALIDATION {arm} {i+1}/{len(rows)}', flush=True); last = time.monotonic()
    if parity >= 1e-4: raise ValueError('Export parity failed')
    result, intent, slots = score_development(rows, tuple(outputs), bundle)
    bundle['routing']['variants']['fp32'] = intent
    bundle['acoustic_slots']['variants']['fp32'] = slots
    save(folder / 'bundle.json', bundle)
    torch.save(tuple(outputs), folder / 'validation-logits.pt')
    result.update(arm=arm, graph_sha256=digest(graph), bundle_sha256=digest(folder / 'bundle.json'),
        parity_max_abs_error=parity, graph_bytes=graph.stat().st_size, single_recording_cpu_onnx=True,
        test_used=False, validation_rows=len(rows), validation_source_sha256=digest(__file__))
    save(EXP / (arm + '-validation.json'), result)
    print(json.dumps({'arm': arm, 'rank': result['rank'], 'parity': parity}), flush=True)


def lock():
    from .prototype_eval import calibrate_custom
    verify()
    path = EXP / 'selection-before-test.json'
    if path.exists(): raise ValueError('Selection already frozen')
    protocol = json.loads((EXP / 'protocol.json').read_text(encoding='utf-8'))
    reports = {arm: json.loads((EXP / (arm + '-validation.json')).read_text(encoding='utf-8')) for arm in protocol['arms']}
    arm = max(reports, key=lambda a: tuple(reports[a]['rank']))
    custom = calibrate_custom(CACHE / arm)
    files = [EXP / 'protocol.json', DATA / 'frozen.json', DATA / 'overlay.jsonl', CACHE / arm / 'vcm.fp32.onnx',
        CACHE / arm / 'bundle.json', ROOT / 'scripts/test/prototype.py', ROOT / 'scripts/test/prototype_validation.py']
    files.extend(ROOT / name for name in [
        'experiments/vcm-complete-v1/custom-catalog-v3.json',
        'experiments/vcm-complete-v1/number-stress-generation-v2.json',
        'experiments/vcm-complete-v1/number-stress-manifest-v2.jsonl',
        'data/vcm-complete-v1/number-test-plan.json',
        '.cache/vcm-complete-v1/human_keywords-direct-test.jsonl',
        '.cache/vcm-complete-v1/human_keywords-silero-test.jsonl',
        '.cache/vcm-complete-v1/human_keywords/vcm.fp32.onnx',
        'artifacts/silero-vad.onnx', 'pipeline.toml'])
    receipt = {'selected_arm': arm, 'ranks': {a:r['rank'] for a,r in reports.items()},
        'files': {p.relative_to(ROOT).as_posix():digest(p) for p in files},
        'custom_policy': custom, 'historical_test_previously_examined': True,
        'rule': 'Development quality gate, complete recall, worst-family recall, precision; never test selection'}
    save(path, receipt)
    print(json.dumps(receipt), flush=True)
