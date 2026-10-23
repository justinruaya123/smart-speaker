"""Explicit opt-in synthetic Hey Iris study; no runtime/actions or VCM fitting.

Raw provider output, decoded audio and feature/checkpoint intermediates belong
under the ignored .cache/iris-wake-publication-v1 directory, never in Git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/iris-wake-publication-v1'
CACHE = ROOT / '.cache/iris-wake-publication-v1'
BASELINE_VOICES = (
    'en-US-AndrewNeural', 'en-US-JennyNeural',
    'en-GB-RyanNeural', 'en-GB-LibbyNeural',
    'en-PH-JamesNeural', 'en-PH-RosaNeural',
    'fil-PH-AngeloNeural', 'fil-PH-BlessicaNeural',
)
NEGATIVES = ('Hey Siri', 'Hey Jarvis', 'Hi Iris', 'Hey Irish', 'Hey Alice',
             'Iris flower', 'turn on the lights', 'ordinary conversation')


def baseline_cases(available):
    """Declare the whole cohort before touching any audio or model scores."""
    inventory = {v['ShortName']: v for v in available}
    missing = set(BASELINE_VOICES) - inventory.keys()
    if missing:
        raise ValueError('Missing required voices: ' + ', '.join(sorted(missing)))
    rows = []
    for voice in BASELINE_VOICES:
        for kind, texts, rates in (
            ('positive', ('Hey Iris', 'Hey, Iris'), ('-10%', '+10%')),
            ('negative', NEGATIVES, ('+0%',)),
        ):
            for text in texts:
                for rate in rates:
                    rows.append(dict(id=f'baseline-{len(rows):03d}', split='baseline',
                                     kind=kind, wake_positive=kind == 'positive',
                                     voice=voice, text=text, rate=rate, synthetic=True,
                                     voice_metadata={k: inventory[voice][k] for k in ('Gender', 'Locale')}))
    for name, amplitude in (('silence', 0), ('white_quiet', .001),
                            ('white_loud', .04), ('pink', .02)):
        rows.append(dict(id=f'baseline-control-{name}', split='baseline', kind='control',
                         wake_positive=False, synthetic=True, voice=None, text=None,
                         control=name, amplitude=amplitude, seconds=10, seed=731))
    return rows


ADAPTATION_VOICES = {
    'train': ('en-US-GuyNeural', 'en-US-AriaNeural', 'en-US-ChristopherNeural',
              'en-US-MichelleNeural', 'en-US-BrianNeural', 'en-US-EmmaNeural',
              'en-GB-ThomasNeural', 'en-GB-SoniaNeural', 'en-AU-WilliamMultilingualNeural',
              'en-AU-NatashaNeural', 'en-CA-ClaraNeural', 'en-CA-LiamNeural'),
    'dev': ('en-IE-ConnorNeural', 'en-IE-EmilyNeural', 'en-IN-PrabhatNeural', 'en-IN-NeerjaNeural'),
    'calibration': ('en-HK-SamNeural', 'en-HK-YanNeural', 'en-SG-WayneNeural', 'en-SG-LunaNeural'),
    'test': ('en-US-EricNeural', 'en-US-AvaNeural', 'en-US-SteffanNeural', 'en-GB-MaisieNeural',
             'en-NZ-MitchellNeural', 'en-NZ-MollyNeural', 'en-ZA-LukeNeural', 'en-ZA-LeahNeural'),
}
TRAIN_EXTRA_NEGATIVES = (
    'Hey Cyrus', 'Hey Ibis', 'Hey Eros', 'Hey Paris', 'Hey Irish!', 'Hi, Iris',
    'Hello Iris', 'Iris', 'Hey', 'I rate this', 'The iris flower is purple',
    'Could you play some music', 'Set a timer for ten seconds', 'The weather is warm',
    'Please stop the music', 'We are having a normal conversation.',
)


def adaptation_cases(available):
    inventory = {v['ShortName']: v for v in available}
    voices = [v for group in ADAPTATION_VOICES.values() for v in group]
    if len(set(voices)) != len(voices) or set(voices).intersection(BASELINE_VOICES):
        raise ValueError('Voice split overlap')
    if set(voices) - inventory.keys():
        raise ValueError('Required adaptation voices unavailable')
    rows = []
    for split, group in ADAPTATION_VOICES.items():
        texts = ('Hey Iris', 'Hey, Iris', 'Hey Iris!', 'Hey, Iris.') if split == 'train' else ('Hey Iris', 'Hey, Iris')
        rates = {'train': ('-15%', '+0%', '+15%'), 'dev': ('-5%', '+5%'),
                 'calibration': ('-12%', '+12%'), 'test': ('-7%', '+7%')}[split]
        if split == 'test':
            texts = ('Hey Iris.', 'Hey, Iris!')
        negatives = NEGATIVES + TRAIN_EXTRA_NEGATIVES if split == 'train' else NEGATIVES
        index = 0
        for voice in group:
            for kind, phrases, speeds in (('positive', texts, rates), ('negative', negatives, ('+0%',))):
                for text in phrases:
                    for rate in speeds:
                        rows.append(dict(id=f'{split}-{index:04d}', split=split, kind=kind,
                                         role='wake_positive' if kind == 'positive' else 'wake_negative',
                                         voice=voice, speaker=voice, text=text, rate=rate,
                                         wake_positive=kind == 'positive', synthetic=True,
                                         voice_metadata={k: inventory[voice][k] for k in ('Gender', 'Locale')}))
                        index += 1
        for seed in range(3 if split == 'train' else 1):
            for name, amplitude in (('silence', 0), ('white_quiet', .001), ('white_loud', .04), ('pink', .02)):
                rows.append(dict(id=f'{split}-control-{seed}-{name}', split=split, kind='control',
                                 role='control', voice=None, speaker=None, text=None, wake_positive=False,
                                 synthetic=True, control=name, amplitude=amplitude, seconds=10, seed=873 + seed))
    return rows


def assert_fit_membership(rows, split):
    if not rows or any(r['split'] != split or not r['synthetic'] for r in rows):
        raise ValueError('Invalid fit membership: only newly generated declared split may be decoded')


def freeze_adaptation(available, *, out=EXPERIMENT):
    baseline_path = Path(out) / 'baseline-results.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['metrics']['synthetic_smoke_pass']:
        raise ValueError('No adaptation needed: baseline passed')
    rows = adaptation_cases(available)
    result = dict(schema_version=1, experiment='iris-wake-publication-v1-companion',
                  hypothesis='A separately trained same-architecture wake-only companion can pass unseen-voice synthetic smoke',
                  failed_baseline_sha256=digest(baseline_path),
                  architecture='scripts.models.mvp_wake.WakeNet(width=64)', parameters=82114,
                  initialization='random seed 731; no pretrained checkpoint; no VCM data/model/normalizer changes',
                  seed=731, cpu_threads=4, epochs=30, steps_per_epoch=64, batch_size=64,
                  optimizer='AdamW lr=.001 weight_decay=.001; CE; balanced labels; clip_grad_norm=5',
                  training_arrays='Reuse iris_wake_helpers.training_arrays training=True hard=True, seed61, unchanged augmentation/trim; retain exclusions',
                  normalization='TRAIN only: helper training=False windows, unnormalized logmel; binwise mean/std over all frames, std floor .1',
                  dev_selection='Maximize (dev balanced accuracy, -dev cross entropy); strict improvement only, earliest exact tie; no calibration/test PCM decoded',
                  calibration_selection='Selected DEV checkpoint only; maximize detected wakes with zero false/duplicates, lower threshold tie; fixed grid; TEST exactly once after hash lock',
                  threshold_grid=[.5, .6, .7, .8, .85, .9, .95, .98, .995, .999, .9999, 1.],
                  policy=dict(release=.3, consecutive=2, cooldown_samples=16000),
                  window_samples=25600, step_samples=1280,
                  source_bindings={p: digest(ROOT / p) for p in (
                      'scripts/runtime/mvp_wake.py', 'scripts/models/mvp_wake.py',
                      'scripts/train/iris_wake_helpers.py', 'vcm/frontend.py')},
                  positive_recall_minimum=.9, false_wakes_maximum=0, duplicate_wakes_maximum=0,
                  splits={s: dict(voices=list(v), positive=sum(r['split'] == s and r['wake_positive'] for r in rows),
                                   negative=sum(r['split'] == s and not r['wake_positive'] for r in rows))
                          for s, v in ADAPTATION_VOICES.items()},
                  first_panel_usage='Regression only, completely excluded from companion data and selection',
                  corpus='Entirely new opt-in Edge TTS generation; no old VCM heldout or other corpus audio in any split',
                  test_access='Generation converts provider MP3 to PCM16 and hashes bytes/headers; no test numerical PCM loading, feature extraction or model inference before selection lock',
                  rights='Raw synthesized corpus is not redistributed; synthesis voice IDs are not people/demographics.',
                  original_human_positive_n=0, qualified=False, cases=rows)
    write_once(Path(out) / 'companion-protocol.json', result)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    """Idempotence is allowed, changed protocol/receipts are never overwritten."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Refusing immutable output change: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def verify_generated(case, receipt, wav):
    if receipt['case_sha256'] != object_digest(case):
        raise ValueError('Changed synthesis case binding')
    if digest(wav) != receipt['wav_sha256']:
        raise ValueError('Changed generated audio')


def freeze_baseline(available, *, out=EXPERIMENT):
    """The tracked protocol is written before any provider generation."""
    folder = ROOT / 'artifacts/iris/wake'
    policy = json.loads((folder / 'iris.json').read_text())
    result = dict(schema_version=1, experiment='iris-wake-publication-v1-baseline',
                  hypothesis='The unchanged native graph/gate passes a diverse synthetic smoke panel',
                  base_revision='c0628940400642f204ff98e26908549b1c07e41c',
                  positive_recall_minimum=.9, false_wakes_maximum=0,
                  duplicate_wakes_maximum=0, original_human_positive_n=0, qualified=False,
                  model_sha256=digest(folder / 'iris.onnx'), policy_sha256=digest(folder / 'iris.json'),
                  baseline_policy={k: policy[k] for k in ('threshold', 'release', 'consecutive', 'cooldown_samples')},
                  source_bindings={p: digest(ROOT / p) for p in (
                      'scripts/runtime/mvp_wake.py', 'scripts/models/mvp_wake.py',
                      'scripts/train/iris_wake_helpers.py', 'vcm/frontend.py')},
                  window_samples=25600, step_samples=1280,
                  stream='Reset each clip; 32000 leading and 48000 trailing zero samples; feed 640 samples; actual HeyIris CPU ONNX',
                  negative_exposure='Only original clip duration; padding excluded; not continuous household audio',
                  generation=dict(provider='Microsoft Edge TTS via edge-tts', version='7.3.1',
                                  concurrency=3, maximum_attempts=4, codec='ffmpeg PCM16 mono 16000Hz'),
                  rights='No raw TTS corpus redistribution. Voice metadata is not human demographic evidence.',
                  cases=baseline_cases(available))
    write_once(Path(out) / 'baseline-protocol.json', result)
    return result


async def generate_cases(cases, cache, protocol_sha256, *, synthesizer=None, decoder=None):
    """Only this explicitly invoked asynchronous stage contacts the TTS service."""
    import asyncio
    import array
    import random
    import subprocess
    import wave
    import sys

    cache = Path(cache)
    (cache / 'audio').mkdir(parents=True, exist_ok=True)
    (cache / 'receipts').mkdir(exist_ok=True)
    semaphore = asyncio.Semaphore(3)

    async def edge_synth(case, path):
        import edge_tts
        await edge_tts.Communicate(case['text'], case['voice'], rate=case['rate']).save(str(path))

    def ffmpeg_decode(mp3, wav):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(mp3), '-ac', '1',
                        '-ar', '16000', '-c:a', 'pcm_s16le', str(wav)], check=True,
                       capture_output=True, timeout=45)

    async def one(case):
        if not case['synthetic'] or not case['id'].replace('-', '').replace('_', '').isalnum():
            raise ValueError('Expected closed synthetic case ID')
        wav = cache / 'audio' / (case['id'] + '.wav')
        mp3 = wav.with_suffix('.mp3')
        receipt_path = cache / 'receipts' / (case['id'] + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['protocol_sha256'] != protocol_sha256:
                raise ValueError('Changed generation protocol')
            verify_generated(case, receipt, wav)
            if case.get('voice') and digest(mp3) != receipt['mp3_sha256']:
                raise ValueError('Changed provider audio')
            return receipt
        async with semaphore:
            if case.get('voice'):
                for attempt in range(4):
                    try:
                        await (synthesizer or edge_synth)(case, mp3)
                        (decoder or ffmpeg_decode)(mp3, wav)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2 ** attempt)
            else:
                rng = random.Random(case['seed'])
                values = array.array('h')
                smooth = 0.
                for _ in range(round(case['seconds'] * 16000)):
                    value = rng.gauss(0, case['amplitude'])
                    if case['control'] == 'pink':
                        smooth = .98 * smooth + .02 * value
                        value = smooth * 6
                    values.append(round(max(-.99, min(.99, value)) * 32767))
                if sys.byteorder != 'little':
                    values.byteswap()
                with wave.open(str(wav), 'wb') as f:
                    f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    f.writeframes(values.tobytes())
            with wave.open(str(wav), 'rb') as f:
                if f.getnchannels() != 1 or f.getframerate() != 16000 or f.getsampwidth() != 2:
                    raise ValueError('Invalid synthesized WAV format')
                pcm = f.readframes(f.getnframes())
                samples = f.getnframes()
            receipt = dict(id=case['id'], case_sha256=object_digest(case),
                           protocol_sha256=protocol_sha256, wav_sha256=digest(wav),
                           decoded_pcm_s16le_sha256=hashlib.sha256(pcm).hexdigest(),
                           mp3_sha256=digest(mp3) if case.get('voice') else None,
                           samples=samples, seconds=samples / 16000, sample_rate=16000,
                           synthetic=True, generator='edge-tts==7.3.1' if case.get('voice') else 'seeded software control')
            write_once(receipt_path, receipt)
            print('GENERATED', case['id'], samples, flush=True)
            return receipt

    records = await asyncio.gather(*(one(case) for case in cases))
    write_once(cache / 'generation-complete.json', records)
    return records


def fit_normalization(rows, reader):
    import numpy as np
    from vcm.frontend import SPEC
    from scripts.train.iris_wake_helpers import training_arrays
    assert_fit_membership(rows, 'train')
    raw, _, excluded = training_arrays(rows, {'mean': [0.] * 40, 'std': [1.] * 40},
                                      training=False, reader=reader)
    values = raw.astype('float64')
    return dict(frontend=SPEC, mean=values.mean(axis=(0, 2)).tolist(),
                std=np.maximum(values.std(axis=(0, 2)), .1).tolist(),
                source_ids=[r['id'] for r in rows], fit_split='train',
                source_n=len(rows), statistic_windows=len(raw), excluded=excluded,
                original_human_positive_n=0)


def validate_recipe(cfg):
    if (not 1 <= cfg['epochs'] <= 30 or not 1 <= cfg['cpu_threads'] <= 4 or
            not 1 <= cfg['steps_per_epoch'] <= 64 or not 2 <= cfg['batch_size'] <= 64 or
            cfg['batch_size'] % 2 or cfg['parameters'] != 82114):
        raise ValueError('Refusing unbounded or different-architecture companion training')


def development_key(logits, labels):
    import numpy as np
    values = np.asarray(logits, dtype='float64')
    labels = np.asarray(labels)
    pred = values.argmax(1)
    recall = float(np.mean(pred[labels == 1] == 1))
    specificity = float(np.mean(pred[labels == 0] == 0))
    shifted = values - values.max(1, keepdims=True)
    loss = float(np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]))
    return ((recall + specificity) / 2, -loss), dict(recall=recall, specificity=specificity, cross_entropy=loss)


def fit_companion(protocol_path=None, cache=None, out=EXPERIMENT):
    """Bounded random-initialized CPU wake fit. Never call the legacy trainer."""
    import time
    import sys
    import platform
    import numpy as np
    import torch
    from scripts.models.mvp_wake import WakeNet
    from scripts.train.iris_wake_helpers import training_arrays
    protocol_path = Path(protocol_path or EXPERIMENT / 'companion-protocol.json')
    cache = Path(cache or CACHE / 'companion')
    out = Path(out)
    cfg = json.loads(protocol_path.read_text())
    validate_recipe(cfg)
    for path, expected in cfg['source_bindings'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Changed frozen source ' + path)
    if (out / 'fit-results.json').exists():
        raise ValueError('Completed companion fit exists: do not refit/reselect')
    rows = cfg['cases']
    receipts = json.loads((cache / 'generation-complete.json').read_text())
    if [r['id'] for r in receipts] != [r['id'] for r in rows]:
        raise ValueError('Incomplete generated cohort')
    by_id = {r['id']: r for r in receipts}
    split_rows = {s: [r for r in rows if r['split'] == s] for s in ADAPTATION_VOICES}
    # Reading headers/hashes is not feature extraction or heldout model scoring.
    speech_hash_groups = {}
    for row in rows:
        receipt = by_id[row['id']]
        if receipt['protocol_sha256'] != digest(protocol_path) or receipt['case_sha256'] != object_digest(row):
            raise ValueError('Changed frozen input bindings')
        if row['voice']:
            speech_hash_groups.setdefault(receipt['decoded_pcm_s16le_sha256'], set()).add(row['split'])
    if any(len(groups) > 1 for groups in speech_hash_groups.values()):
        raise ValueError('Speech PCM crosses declared voice-disjoint partitions')
    inputs = dict(protocol_sha256=digest(protocol_path), trainer_source_sha256=digest(__file__),
                  generation_sha256=digest(cache / 'generation-complete.json'),
                  partition_sha256={s: object_digest([dict(case=r, receipt=by_id[r['id']]) for r in rs])
                                    for s, rs in split_rows.items()},
                  source_bindings=cfg['source_bindings'], python=sys.version,
                  torch=torch.__version__, numpy=np.__version__, platform=platform.platform(),
                  test_numeric_pcm_loaded_before_fit=False, test_used_for_selection=False,
                  note='Deterministic software controls may share PCM; no speech PCM cross-split overlap')
    write_once(out / 'fit-inputs.json', inputs)
    work = cache / 'training'
    work.mkdir(exist_ok=True)
    loaded = {}

    def reader(row):
        if row['split'] not in ('train', 'dev'):
            raise ValueError('Attempted heldout PCM access in fit')
        if row['id'] not in loaded:
            loaded[row['id']] = read_case_pcm(row, by_id[row['id']], cache)
        return loaded[row['id']]

    train, dev = split_rows['train'], split_rows['dev']
    assert_fit_membership(train, 'train')
    assert_fit_membership(dev, 'dev')
    norm = fit_normalization(train, reader)
    write_once(work / 'normalization.json', norm)
    x, y, excluded = training_arrays(train, norm, training=True, hard=True, reader=reader)
    dx, dy, dev_excluded = training_arrays(dev, norm, training=False, reader=reader)
    audit = dict(training_windows=len(y), positive_windows=int(y.sum()), development_windows=len(dy),
                 excluded_training=excluded, excluded_development=dev_excluded,
                 normalization_sha256=digest(work / 'normalization.json'),
                 feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                 dev_feature_sha256=hashlib.sha256(dx.tobytes()).hexdigest(),
                 decoded_numeric_splits=sorted({r['split'] for r in rows if r['id'] in loaded}),
                 data_counts=cfg['splits'], human_positive_n=0)
    write_once(out / 'fit-data-audit.json', audit)
    np.savez(work / 'features.npz', x=x, y=y, dx=dx, dy=dy)
    torch.set_num_threads(cfg['cpu_threads'])
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    model = WakeNet()
    if sum(p.numel() for p in model.parameters()) != cfg['parameters']:
        raise ValueError('Changed wake architecture')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    rng = np.random.default_rng(cfg['seed'])
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    history, best, first = [], None, 1
    if (work / 'last.pt').exists():
        saved = torch.load(work / 'last.pt', map_location='cpu', weights_only=True)
        if saved['input_sha256'] != digest(out / 'fit-inputs.json'):
            raise ValueError('Changed resume inputs')
        model.load_state_dict(saved['weights'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state = saved['numpy_rng']
        history, best, first = saved['history'], saved['best'], saved['epoch'] + 1
    began = time.perf_counter()
    for epoch in range(first, cfg['epochs'] + 1):
        model.train()
        losses = []
        for _ in range(cfg['steps_per_epoch']):
            indices = np.concatenate([rng.choice(group, cfg['batch_size'] // 2) for group in groups])
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x[indices]))
            loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y[indices]))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite companion training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            values = np.concatenate([model(torch.from_numpy(dx[k:k + 64])).numpy()
                                     for k in range(0, len(dx), 64)])
        key, metric = development_key(values, dy)
        record = dict(epoch=epoch, loss=float(np.mean(losses)), development=metric,
                      seconds=time.perf_counter() - began)
        history.append(record)
        if best is None or key > tuple(best['key']):
            torch.save(dict(weights={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            config=model.config, epoch=epoch), work / 'best.pt')
            best = dict(epoch=epoch, key=list(key), checkpoint_sha256=digest(work / 'best.pt'))
        torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), numpy_rng=rng.bit_generator.state,
                        input_sha256=digest(out / 'fit-inputs.json'), epoch=epoch,
                        history=history, best=best), work / 'last.tmp')
        (work / 'last.tmp').replace(work / 'last.pt')
        print('COMPANION_TRAIN', json.dumps(record), flush=True)
    result = dict(**best, history=history, parameters=cfg['parameters'], seed=cfg['seed'],
                  optimizer_updates=cfg['epochs'] * cfg['steps_per_epoch'], cpu_threads=cfg['cpu_threads'],
                  normalization_sha256=digest(work / 'normalization.json'),
                  inputs_sha256=digest(out / 'fit-inputs.json'), protocol_sha256=digest(protocol_path),
                  test_numeric_pcm_loaded=False, calibration_numeric_pcm_loaded=False,
                  test_used_for_selection=False, qualified=False, real_human_positive_n=0)
    write_once(out / 'fit-results.json', result)
    return result


def select_calibration(records, thresholds):
    from scripts.train.iris_wake_helpers import stream_metrics
    if not records or any(r.get('split') != 'calibration' for r in records):
        raise ValueError('Threshold selection requires calibration records only')
    grid = []
    for threshold in thresholds:
        metric = stream_metrics(records, threshold)
        grid.append(dict(threshold=threshold, metrics=metric))
    feasible = [r for r in grid if not r['metrics']['false_wakes'] and not r['metrics']['duplicate_detections']]
    if not feasible:
        raise ValueError('No feasible calibration threshold')
    selected = max(feasible, key=lambda r: (r['metrics']['detected'], -r['threshold']))
    return dict(**selected, grid=grid, test_used_for_selection=False)


def calibrate_export(protocol_path=None, cache=None, out=EXPERIMENT):
    """Select on calibration using actual exported ONNX scores, lock before TEST."""
    import shutil
    import warnings
    import numpy as np
    import torch
    import onnxruntime as ort
    from scripts.models.mvp_wake import WakeNet
    from scripts.runtime.mvp_wake import HeyIris
    protocol_path = Path(protocol_path or EXPERIMENT / 'companion-protocol.json')
    cache = Path(cache or CACHE / 'companion')
    out = Path(out)
    if (out / 'selection-before-test.json').exists():
        raise ValueError('Selection already locked; no reselection')
    cfg = json.loads(protocol_path.read_text())
    fit = json.loads((out / 'fit-results.json').read_text())
    if fit['protocol_sha256'] != digest(protocol_path):
        raise ValueError('Changed fit protocol')
    checkpoint = cache / 'training/best.pt'
    if digest(checkpoint) != fit['checkpoint_sha256']:
        raise ValueError('Changed DEV-selected checkpoint')
    norm_path = cache / 'training/normalization.json'
    if digest(norm_path) != fit['normalization_sha256']:
        raise ValueError('Changed train-only normalization')
    norm = json.loads(norm_path.read_text())
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = WakeNet(**saved['config']).eval()
    model.load_state_dict(saved['weights'])
    if sum(p.numel() for p in model.parameters()) != 82114:
        raise ValueError('Different wake architecture')
    export = cache / 'export-calibration'
    export.mkdir(exist_ok=True)
    torch.manual_seed(cfg['seed'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.onnx.export(model, (torch.randn(1, 40, 158),), str(export / 'iris.onnx'),
                          input_names=['features'], output_names=['wake'], opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(export / 'iris.onnx'), options, providers=['CPUExecutionProvider'])
    errors = []
    for _ in range(5):
        x = torch.randn(1, 40, 158)
        with torch.no_grad():
            expected = model(x).numpy()
        actual = session.run(None, {'features': x.numpy()})[0]
        errors.append(float(np.abs(expected - actual).max()))
    if max(errors) > 2e-5:
        raise ValueError('Wake CPU ONNX parity failed')
    probe_meta = dict(files={'iris.onnx': digest(export / 'iris.onnx')}, normalization=norm,
                      threshold=1., **cfg['policy'], qualified=False, parameters=82114,
                      real_wake_positive_recordings=0, supported_engines=['onnxruntime'])
    write_once(export / 'iris.json', probe_meta)
    detector = HeyIris(export)
    cases = [r for r in cfg['cases'] if r['split'] == 'calibration']
    assert_fit_membership(cases, 'calibration')
    records, receipts = [], []
    for case in cases:
        receipt = json.loads((cache / 'receipts' / (case['id'] + '.json')).read_text())
        if receipt['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Changed calibration protocol')
        result = stream_pcm(detector, read_case_pcm(case, receipt, cache))
        records.append(dict(id=case['id'], split='calibration', positive=case['wake_positive'],
                            synthetic=True, seconds=result['seconds'], scores=result['scores']))
        receipts.append(receipt)
    selected = select_calibration(records, cfg['threshold_grid'])
    write_once(cache / 'calibration-stream-scores.json', records)
    selected_dir = cache / 'selected'
    selected_dir.mkdir(exist_ok=True)
    shutil.copyfile(export / 'iris.onnx', selected_dir / 'iris.onnx')
    meta = {**probe_meta, 'threshold': selected['threshold'], 'epoch': saved['epoch'],
            'initialization': 'random seed731; independent wake-only study',
            'experiment': cfg['experiment'], 'protocol_sha256': digest(protocol_path),
            'calibration': selected['metrics'], 'engine_parity_max_error': max(errors)}
    write_once(selected_dir / 'iris.json', meta)
    calibration = dict(**selected, inputs=receipts,
                       checkpoint_sha256=digest(checkpoint), protocol_sha256=digest(protocol_path),
                       onnx_parity=dict(max_absolute_error=max(errors), trials=len(errors),
                                        exporter='torch.onnx.export dynamo=False opset17',
                                        warnings=[str(w.message) for w in caught]))
    write_once(out / 'calibration-results.json', calibration)
    test_cases = [r for r in cfg['cases'] if r['split'] == 'test']
    lock = dict(protocol_sha256=digest(protocol_path), model_sha256=digest(selected_dir / 'iris.onnx'),
                policy_sha256=digest(selected_dir / 'iris.json'), checkpoint_sha256=digest(checkpoint),
                normalization_sha256=digest(norm_path), source_sha256=digest(__file__),
                fit_result_sha256=digest(out / 'fit-results.json'),
                calibration_sha256=digest(out / 'calibration-results.json'),
                test_cases_sha256=object_digest(test_cases),
                test_inputs_sha256=object_digest([json.loads((cache / 'receipts' / (r['id'] + '.json')).read_text())
                                                 for r in test_cases]),
                test_features_or_inference_before_lock=False, qualified=False,
                selected_folder=str(selected_dir.relative_to(ROOT)).replace('\\', '/'))
    write_once(out / 'selection-before-test.json', lock)
    return lock


def validate_test_lock(protocol_path, cases, receipts, lock):
    canonical = [r for r in json.loads(Path(protocol_path).read_text())['cases'] if r['split'] == 'test']
    if (digest(protocol_path) != lock['protocol_sha256'] or cases != canonical or
            object_digest(cases) != lock['test_cases_sha256'] or
            [r['id'] for r in receipts] != [r['id'] for r in cases] or
            object_digest(receipts) != lock['test_inputs_sha256']):
        raise ValueError('Changed locked test cohort/inputs')


def read_case_pcm(case, receipt, cache):
    import numpy as np
    import wave
    wav = Path(cache) / 'audio' / (case['id'] + '.wav')
    verify_generated(case, receipt, wav)
    with wave.open(str(wav), 'rb') as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
            raise ValueError('Invalid generated PCM format')
        data = f.readframes(f.getnframes())
    if hashlib.sha256(data).hexdigest() != receipt['decoded_pcm_s16le_sha256']:
        raise ValueError('Changed decoded PCM identity')
    return np.frombuffer(data, dtype='<i2').astype('float32') / 32768


def stream_pcm(detector, pcm):
    """Run the real RollingWake instance, not a threshold-only approximation."""
    import numpy as np
    scores = []
    score = detector.score
    detector.reset()

    def observed(window):
        value = score(window)
        scores.append(value)
        return value

    detector.score = observed
    padded = np.concatenate((np.zeros(32000, dtype='float32'), pcm,
                             np.zeros(48000, dtype='float32')))
    hits = []
    try:
        for offset in range(0, len(padded), 640):
            if detector.push(padded[offset:offset + 640]) == 1.:
                hits.append((offset + min(640, len(padded) - offset) - 32000) / 16000)
    finally:
        detector.score = score
    return dict(hits=len(hits), hit_seconds=hits, max_score=max(scores, default=0.),
                score_count=len(scores), scores=scores, seconds=len(pcm) / 16000)


def panel_metrics(cases, records):
    expected = [r['id'] for r in cases]
    observed = [r['id'] for r in records]
    if not expected or len(set(expected)) != len(expected) or observed != expected:
        raise ValueError('Incomplete/repeated/reordered cohort')
    pairs = list(zip(cases, records))
    positives = [(c, r) for c, r in pairs if c['wake_positive']]
    negatives = [(c, r) for c, r in pairs if not c['wake_positive']]
    detected = sum(r['hits'] > 0 for _, r in positives)
    duplicates = sum(max(0, r['hits'] - 1) for _, r in positives)
    false_wakes = sum(r['hits'] for _, r in negatives)
    negative_seconds = sum(r['seconds'] for _, r in negatives)
    recall = detected / len(positives) if positives else None
    return dict(positive_n=len(positives), detected=detected, missed=len(positives) - detected,
                recall=recall, duplicate_detections=duplicates, negative_n=len(negatives),
                negative_seconds=negative_seconds, false_wakes=false_wakes,
                confusable_n=sum(c['kind'] == 'negative' for c, _ in negatives),
                control_n=sum(c['kind'] == 'control' for c, _ in negatives),
                synthetic_smoke_pass=bool(recall is not None and recall >= .9 and not false_wakes and not duplicates))


def evaluate_panel(protocol_path, folder, cache, result_path, *, cases=None, lock=None):
    """A fresh single-thread CPU ONNX pass; existing results cannot be replaced."""
    import time
    import platform
    import sys
    import onnxruntime as ort
    from scripts.runtime.mvp_wake import HeyIris
    protocol_path, folder, cache, result_path = map(Path, (protocol_path, folder, cache, result_path))
    if result_path.exists():
        raise ValueError('Evaluation already exists: refusing test reselection/rerun')
    protocol = json.loads(protocol_path.read_text())
    cases = protocol['cases'] if cases is None else cases
    for source, expected in protocol['source_bindings'].items():
        if digest(ROOT / source) != expected:
            raise ValueError('Changed frozen source: ' + source)
    expected = lock or protocol
    if digest(folder / 'iris.onnx') != expected['model_sha256'] or digest(folder / 'iris.json') != expected['policy_sha256']:
        raise ValueError('Changed locked model/policy')
    detector = HeyIris(folder)
    receipts = [json.loads((cache / 'receipts' / (c['id'] + '.json')).read_text()) for c in cases]
    if lock is not None:
        validate_test_lock(protocol_path, cases, receipts, lock)
        marker = cache / 'test-evaluation-started.json'
        if marker.exists():
            raise ValueError('Locked test was already started: no repeat or reselection')
        write_once(marker, dict(selection_sha256=object_digest(lock), evaluator_sha256=digest(__file__)))
    elif cases != protocol['cases']:
        raise ValueError('Baseline evaluation requires complete canonical cohort')
    predictions, raw = [], []
    began = time.perf_counter()
    for case, receipt in zip(cases, receipts):
        if receipt['protocol_sha256'] != digest(protocol_path):
            raise ValueError('Generation did not use frozen protocol')
        pcm = read_case_pcm(case, receipt, cache)
        row = dict(id=case['id'], **stream_pcm(detector, pcm))
        raw.append(row)
        predictions.append({k: v for k, v in row.items() if k != 'scores'})
        print('NATIVE_STREAM', row['id'], row['hits'], row['max_score'], flush=True)
    result = dict(schema_version=1, protocol_sha256=digest(protocol_path),
                  source_sha256=digest(__file__), model_sha256=digest(folder / 'iris.onnx'),
                  policy_sha256=digest(folder / 'iris.json'),
                  actual_engine='onnxruntime CPUExecutionProvider, threads=1',
                  ort_version=ort.__version__, python=sys.version, platform=platform.platform(),
                  metrics=panel_metrics(cases, predictions), predictions=predictions,
                  inputs=receipts, evaluation_seconds=time.perf_counter() - began,
                  qualified=False, real_human_positive_n=0, live_microphone_measured=False,
                  negative_exposure='Original clip durations only; no padding; not household continuous audio')
    write_once(cache / (result_path.stem + '-scores.json'), raw)
    write_once(result_path, result)
    print('RESULT', json.dumps(result['metrics'], sort_keys=True), flush=True)
    return result


def publication_verdict(baseline, companion):
    def passes(metric):
        return (metric['positive_n'] > 0 and metric['detected'] / metric['positive_n'] >= .9
                and metric['false_wakes'] == 0 and metric['duplicate_detections'] == 0)
    baseline_pass, companion_pass = passes(baseline), passes(companion)
    return dict(baseline_pass=baseline_pass, companion_pass=companion_pass,
                synthetic_acceptance_pass=baseline_pass or companion_pass,
                publish_companion=not baseline_pass and companion_pass,
                release_blocker=not (baseline_pass or companion_pass), qualified=False,
                real_human_positive_n=0, live_microphone_measured=False,
                test_reselection_permitted=False)


def study_summary(out=EXPERIMENT):
    """Reconstruct decisions from complete published cohorts, not stored green flags."""
    out = Path(out)
    names = ('baseline-protocol.json', 'baseline-results.json', 'companion-protocol.json',
             'companion-inputs.json', 'fit-inputs.json', 'fit-data-audit.json', 'fit-results.json',
             'calibration-results.json', 'selection-before-test.json', 'companion-test-results.json')
    data = {name: json.loads((out / name).read_text()) for name in names}
    bp, br = data['baseline-protocol.json'], data['baseline-results.json']
    cp, cr = data['companion-protocol.json'], data['companion-test-results.json']
    lock = data['selection-before-test.json']
    all_inputs = data['companion-inputs.json']
    if [r['id'] for r in all_inputs] != [r['id'] for r in cp['cases']]:
        raise ValueError('Incomplete input receipt cohort')
    input_by_id = {r['id']: r for r in all_inputs}
    for case in cp['cases']:
        row = input_by_id[case['id']]
        if row['case_sha256'] != object_digest(case) or row['protocol_sha256'] != digest(out / 'companion-protocol.json'):
            raise ValueError('Changed input case/partition')
    fit_inputs = data['fit-inputs.json']
    if fit_inputs['generation_sha256'] != digest(out / 'companion-inputs.json'):
        raise ValueError('Changed frozen generation manifest')
    for split, expected in fit_inputs['partition_sha256'].items():
        actual = object_digest([dict(case=r, receipt=input_by_id[r['id']]) for r in cp['cases'] if r['split'] == split])
        if actual != expected:
            raise ValueError('Changed immutable partition binding')
    test = [r for r in cp['cases'] if r['split'] == 'test']
    baseline = panel_metrics(bp['cases'], br['predictions'])
    companion = panel_metrics(test, cr['predictions'])
    if baseline != br['metrics'] or companion != cr['metrics']:
        raise ValueError('Stored metrics differ from exact cohort reconstruction')
    validate_test_lock(out / 'companion-protocol.json', test, cr['inputs'], lock)
    for field in ('model_sha256', 'policy_sha256'):
        if br[field] != bp[field] or cr[field] != lock[field]:
            raise ValueError('Result model/policy identity mismatch')
    if (lock['fit_result_sha256'] != digest(out / 'fit-results.json') or
            lock['calibration_sha256'] != digest(out / 'calibration-results.json') or
            cp['failed_baseline_sha256'] != digest(out / 'baseline-results.json')):
        raise ValueError('Changed fit/calibration/baseline receipt chain')
    per_voice = {}
    false_wakes = []
    for case, record in zip(test, cr['predictions']):
        if case['voice']:
            cell = per_voice.setdefault(case['voice'], dict(positive_n=0, detected=0, false_wakes=0))
            if case['wake_positive']:
                cell['positive_n'] += 1
                cell['detected'] += int(record['hits'] > 0)
            else:
                cell['false_wakes'] += record['hits']
        if not case['wake_positive'] and record['hits']:
            false_wakes.append(dict(id=case['id'], voice=case['voice'], text=case['text'], hits=record['hits']))
    fit = data['fit-results.json']
    calibration = data['calibration-results.json']
    audit = data['fit-data-audit.json']
    verdict = publication_verdict(baseline, companion)
    return dict(schema_version=1, experiment='iris-wake-publication-v1',
                verdict=verdict, baseline=baseline, companion_test=companion,
                candidate_model_sha256=lock['model_sha256'], candidate_policy_sha256=lock['policy_sha256'],
                candidate_checkpoint_sha256=lock['checkpoint_sha256'], candidate_parameters=fit['parameters'],
                dev_selected_epoch=fit['epoch'], actual_fit_epochs=len(fit['history']),
                optimizer_updates=fit['optimizer_updates'], cpu_threads=fit['cpu_threads'],
                calibration=dict(threshold=calibration['threshold'], metrics=calibration['metrics']),
                candidate_test_per_voice=per_voice, candidate_test_false_wakes=false_wakes,
                candidate_artifact_publication='NOT PROMOTED: the predeclared test smoke gate failed' if not verdict['companion_pass'] else 'Synthetic gate passed; any distribution still needs an explicit artifact manifest',
                release_disposition='Do not auto-merge a release requiring the synthetic wake gate. Do not lower frozen or candidate thresholds using these tests.' if verdict['release_blocker'] else 'Synthetic smoke only; qualification remains false',
                raw_and_intermediate_location='.cache/iris-wake-publication-v1 on D:; excluded from Git',
                training_source_commit='ce1a0d5', baseline_evaluation_source_commit='952d784',
                execution_source_sha256=dict(baseline=br['source_sha256'],
                    fitting=data['fit-inputs.json']['trainer_source_sha256'],
                    selection=lock['source_sha256'], test=cr['source_sha256']),
                preserved='Existing Iris wake, frozen VCM/frontend/architecture/trainers, normalization/policy and dependency locks unchanged',
                limitations=[
                    'Entire cohort is synthesized/provider-labelled speech or seeded software controls, not humans or live microphone recordings.',
                    'Synthesis decodes MP3 to PCM16 during generation; no numeric heldout PCM/features/model calls are made for fit/selection before the test lock.',
                    f"Training helper excluded {len(audit['excluded_training'])} overlong/silent train and {len(audit['excluded_development'])} DEV positives; all TEST positives remain in the denominator.",
                    'Normalization lists admitted train source IDs and separately records exclusions; excluded positives do not contribute statistic windows.',
                    'Deterministic silence/noise controls can share PCM across partitions; speech PCM hashes and voice IDs are cross-partition disjoint.',
                    'Provider output is not byte-repeatable by guarantee; raw corpus is not redistributed. Frozen local receipts are auditable; regenerations are new runs.',
                    'Native HeyIris RollingWake CPU ONNX was executed; this is not full owner/live-microphone/provider or physical-device qualification.',
                    'No new runtime default or public companion artifact is selected after the failed test.',
                ], receipt_bindings={name: digest(out / name) for name in names})


def main(argv=None):
    import argparse
    import asyncio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=(
        'baseline-freeze', 'baseline-generate', 'baseline-evaluate', 'companion-freeze',
        'companion-generate', 'companion-inputs', 'companion-fit', 'companion-lock', 'companion-test',
        'summary', 'verify'))
    parser.add_argument('--output', type=Path, default=EXPERIMENT, help='New immutable receipt directory')
    parser.add_argument('--cache', type=Path, default=CACHE, help='Ignored raw audio/intermediate root, preferably D:')
    args = parser.parse_args(argv)
    if args.stage in ('summary', 'verify'):
        result = study_summary(args.output)
        if args.stage == 'summary':
            write_once(args.output / 'study-results.json', result)
            print(json.dumps(result, indent=2, allow_nan=False))
            if result['verdict']['release_blocker']:
                raise SystemExit(2)
        else:
            print(json.dumps(dict(receipt_chain_verified=True, acoustic_acceptance_pass=result['verdict']['synthetic_acceptance_pass'],
                                  qualified=False, baseline=result['baseline'], companion=result['companion_test']), sort_keys=True))
        return
    prefix = 'baseline' if args.stage.startswith('baseline') else 'companion'
    protocol = args.output / (prefix + '-protocol.json')
    if args.stage.endswith('-freeze'):
        available = json.loads((args.cache / 'available-voices.json').read_text())
        result = freeze_baseline(available, out=args.output) if prefix == 'baseline' else freeze_adaptation(available, out=args.output)
        print('FROZEN', digest(protocol), len(result['cases']))
    elif args.stage.endswith('-generate'):
        cfg = json.loads(protocol.read_text())
        asyncio.run(generate_cases(cfg['cases'], args.cache / prefix, digest(protocol)))
    elif args.stage == 'baseline-evaluate':
        evaluate_panel(protocol, ROOT / 'artifacts/iris/wake', args.cache / prefix,
                       args.output / 'baseline-results.json')
    elif args.stage == 'companion-inputs':
        records = json.loads((args.cache / prefix / 'generation-complete.json').read_text())
        write_once(args.output / 'companion-inputs.json', records)
        print('INPUT_MANIFEST', digest(args.output / 'companion-inputs.json'), len(records))
    elif args.stage == 'companion-fit':
        fit_companion(protocol, args.cache / prefix, args.output)
    elif args.stage == 'companion-lock':
        print(json.dumps(calibrate_export(protocol, args.cache / prefix, args.output), indent=2))
    else:
        lock = json.loads((args.output / 'selection-before-test.json').read_text())
        cfg = json.loads(protocol.read_text())
        evaluate_panel(protocol, args.cache / prefix / 'selected', args.cache / prefix,
                       args.output / 'companion-test-results.json',
                       cases=[r for r in cfg['cases'] if r['split'] == 'test'], lock=lock)


if __name__ == '__main__':
    main()
