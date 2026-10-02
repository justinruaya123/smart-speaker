"""Behavioral checks and bounded-memory replay for wake-pipeline-v1.

Run: .venv/Scripts/python.exe scripts/test/wake_pipeline.py [--replay | --parity]
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import register_sources, load_config, save_json, digest
register_sources()

import argparse
import copy
import contextlib
import io
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from einops import pack, repeat
from tinyvcm.activation import WakeCommandStream
from tinyvcm.audio import CaptureResampler, resample_audio
from tinyvcm.wake_models import SileroVAD, HeyJarvis, SileroProbability


class ProbabilityFixture:
    def reset(self):
        pass
    def __call__(self, pcm):
        return .9 if np.max(np.abs(pcm)) > .05 else .01


class WakeFixture:
    score = 0.
    def reset(self):
        self.score = 0.
    def push(self, pcm):
        self.score = .9 if np.max(pcm) > .65 else 0.
        return self.score


class WakeBehavior(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'pipeline.toml')
        self.calls = []
        def utterance(pcm, timeout=False):
            if timeout:
                return {'status':'rejected','reason':'utterance_timeout'}
            self.calls.append(pcm)
            return {'status':'accepted','command':{'fixture':True}}
        self.pipeline = SimpleNamespace(config=self.config, bundle=ROOT/'artifacts', utterance=utterance)
        vad = SileroVAD(None, self.config['audio'], probability=ProbabilityFixture())
        self.stream = WakeCommandStream(self.pipeline, WakeFixture(), vad)

    def feed(self, amplitude, frames):
        result = []
        for _ in range(frames):
            result.extend(self.stream.push(np.full(512, amplitude, dtype=np.float32)))
        return result

    def arm(self):
        result = self.feed(.7, 3) + self.feed(0., 6)
        self.assertTrue(any(x.get('status') == 'command_ready' for x in result))

    def test_speech_without_wake_never_reaches_vcm(self):
        self.feed(.3, 50); self.feed(0., 30)
        self.assertEqual(self.calls, [])

    def test_one_command_per_wake_and_wake_pcm_excluded(self):
        self.arm(); self.feed(.3, 20)
        events = self.feed(0., 14)
        self.assertEqual(len(self.calls), 1)
        self.assertLess(float(np.max(self.calls[0])), .65)
        self.assertTrue(any(x.get('status') == 'accepted' for x in events))
        self.feed(.3, 20); self.feed(0., 14)
        self.assertEqual(len(self.calls), 1)

    def test_repeated_activation(self):
        for _ in range(2):
            self.feed(0., 40); self.arm(); self.feed(.3, 20); self.feed(0., 14)
        self.assertEqual(len(self.calls), 2)

    def test_missing_command_times_out(self):
        self.arm()
        events = self.feed(0., 160)
        self.assertTrue(any(x.get('reason') == 'command_start_timeout' for x in events))
        self.assertEqual(self.calls, [])

    def test_missing_wake_pause_is_bounded(self):
        events = self.feed(.7, 70)
        self.assertTrue(any(x.get('reason') == 'wake_word_not_separated' for x in events))
        self.assertEqual(self.calls, [])

    def test_long_command_is_rejected_before_inference(self):
        self.arm()
        events = self.feed(.3, 260)
        self.assertTrue(any(x.get('reason') == 'utterance_timeout' for x in events))
        self.assertEqual(self.calls, [])

    def test_reset_discards_partial_command(self):
        self.arm(); self.feed(.3, 10); self.stream.reset(); self.feed(.3, 20); self.feed(0., 14)
        self.assertEqual(self.calls, [])

    def test_invalid_pcm(self):
        with self.assertRaises(ValueError):
            self.stream.push(np.array([np.nan], dtype=np.float32))

    def test_resampler_matches_whole_recording(self):
        for rate in (16000, 32000, 44100, 48000, 88200, 96000):
            rng = np.random.default_rng(17)
            pcm = rng.normal(0, .1, round(rate * .8)).astype(np.float32)
            resampler = CaptureResampler(rate)
            blocks = [resampler.push(pcm[i:i+resampler.size]) for i in range(0,len(pcm),resampler.size)]
            if rate != 16000:
                blocks.append(resampler.push(np.zeros(resampler.size,dtype=np.float32)))
            streamed, _ = pack(blocks, '*')
            np.testing.assert_allclose(streamed, resample_audio(pcm, rate), atol=1e-6, rtol=1e-5)

    def test_real_models_reject_silence(self):
        wake = HeyJarvis(ROOT/'artifacts')
        scores = [wake.push(np.zeros(1280, dtype=np.float32)) for _ in range(40)]
        self.assertLess(max(scores), self.config['wake']['threshold'])
        vad = SileroVAD(ROOT/'artifacts', self.config['audio'])
        self.assertEqual(vad.segment(np.zeros(32000, dtype=np.float32)), [])

    def capture_fixture(self, overflow_at=None):
        """Exercise the actual callback/queue/native-rate loop with a fake device."""
        from tinyvcm.capture import listen_neural
        command = [(0., 40), (.7, 3), (0., 6), (.3, 20), (0., 14)]
        blocks = [np.full(512 * frames, level, dtype=np.float32) for level, frames in command * 2]
        pcm, _ = pack(blocks, '*')
        pcm = repeat(pcm, 'samples -> (samples native)', native=3)
        pcm = np.pad(pcm, (0, -len(pcm) % 1920 + 1920))
        audio = repeat(pcm, 'samples -> samples channels', channels=2).copy()
        audio[:, 0] = 0  # Only the explicitly selected second channel has audio.
        settings = []
        class Stream:
            def __init__(self, **kwargs):
                settings.append(kwargs)
                self.callback, self.offset = kwargs['callback'], 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            @property
            def active(self):
                if self.offset >= len(audio): return False
                i = self.offset // 1920
                chunk = audio[self.offset:self.offset + 1920].copy()
                self.offset += 1920
                self.callback(chunk, len(chunk), None, i == overflow_at)
                # Callback buffers are borrowed: later mutation must not leak.
                chunk.fill(np.nan)
                return True
        fake = SimpleNamespace(query_devices=lambda:[{'name':'fixture','max_input_channels':2,'default_samplerate':48000}],
            query_hostapis=lambda:[], default=SimpleNamespace(device=[0,0]),
            check_input_settings=lambda **kwargs:None, InputStream=Stream, PortAudioError=RuntimeError)
        self.pipeline.live_stream = lambda:self.stream
        args = SimpleNamespace(mic=0, channel=1, capture_rate=None, seconds=0, vad_debug=True)
        output = io.StringIO()
        with patch.dict(sys.modules, {'sounddevice':fake}), contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            listen_neural(self.pipeline, args)
        self.assertEqual(settings[0]['samplerate'], 48000)
        self.assertEqual(settings[0]['channels'], 2)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_native_microphone_callback_reaches_command_model(self):
        events = self.capture_fixture()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(events[-1]['overflows'], 0)
        self.assertFalse(events[-1]['audio_saved'])

    def test_capture_overflow_discards_incomplete_command(self):
        events = self.capture_fixture(overflow_at=46)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(events[-1]['overflows'], 1)
        self.assertTrue(any(e.get('reason') == 'audio_overflow' for e in events))

    def test_standalone_bundle_rejects_transcription_backend(self):
        from tinyvcm.inference import Recognizer
        with self.assertRaises(ValueError):
            Recognizer(ROOT/'artifacts', parameter_backend='stt')

    def test_file_and_live_modes_share_one_vad_session(self):
        from tinyvcm.pipeline import Pipeline
        pipeline = Pipeline(ROOT/'artifacts', self.config)
        self.assertIs(pipeline.vad, pipeline.live_stream().vad)


def parity():
    """Compare with exact upstream adapters cached during the declared setup.

    These research-only imports never enter the runtime package. Refuse altered
    upstream source before executing either reference class.
    """
    import ast
    import os
    import pathlib
    import warnings
    from collections import deque
    from typing import Union, List, Callable, Deque
    import torch
    sources = {
        'oww_utils.py': ('AudioFeatures', '977e6e77f91ab0340352add859c5ada3803394f376daf4e27df07380e1def2ab'),
        'silero_utils.py': ('OnnxWrapper', '4d645d4f78d35fdb9f7d79f89bdb24e3e1b56a18fd896b13975a386ecfd55404')}
    classes = {}
    for name, (class_name, expected) in sources.items():
        path = ROOT/'.cache/wake-pipeline-v1/upstream'/name
        if digest(path) != expected:
            raise ValueError(f'Upstream reference hash mismatch: {name}')
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        namespace = dict(np=np, os=os, pathlib=pathlib, deque=deque, Union=Union, List=List,
                         Callable=Callable, Deque=Deque, torch=torch, warnings=warnings, __file__=str(path))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        classes[class_name] = namespace[class_name]
    folder = ROOT/'artifacts'
    reference = classes['AudioFeatures'](melspec_model_path=str(folder/'wake-melspectrogram.onnx'),
        embedding_model_path=str(folder/'wake-embedding.onnx'), inference_framework='onnx')
    reference_vad = classes['OnnxWrapper'](str(folder/'silero-vad.onnx'), force_onnx_cpu=True)
    wake, vad = HeyJarvis(folder), SileroProbability(folder)
    rng = np.random.default_rng(17)
    score_error = embedding_error = vad_error = 0.
    for i in range(60):
        pcm = rng.uniform(-.25, .25, 1280).astype(np.float32)
        actual = wake.push(pcm)
        reference((np.clip(pcm, -1, 1) * 32767).astype(np.int16))
        if i >= 15:
            features = reference.get_features(16)
            expected = wake.head.run(None, {wake.head.get_inputs()[0].name:features})[0].item()
            score_error = max(score_error, abs(actual-expected))
            embedding_error = max(embedding_error, float(np.max(np.abs(wake.embeddings-features[0]))))
        vad_error = max(vad_error, abs(vad(pcm[:512])-reference_vad(torch.from_numpy(pcm[:512]),16000).item()))
    result = dict(openwakeword_score_max_abs_error=score_error, openwakeword_embedding_max_abs_error=embedding_error,
        silero_score_max_abs_error=vad_error, frames=60, seed=17,
        reference_sources={name:sha for name,(_,sha) in sources.items()},
        scope='Numerical parity with pinned upstream adapters after wake warmup on deterministic random PCM; not acoustic wake accuracy.')
    save_json(ROOT/'experiments/wake-pipeline-v1/pretrained-parity.json', result)
    print(json.dumps(result, indent=2))
    assert max(score_error, embedding_error, vad_error) < 1e-6


def replay():
    """Read manifests linewise and decode just one test recording at a time."""
    from tinyvcm.validation import evaluate
    config = load_config(ROOT/'pipeline.toml')
    rows = []
    with (ROOT/'data/vcm.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['split']=='test':
                rows.append(row)
    ids = {r['id'] for r in rows}
    refs = {}
    with (ROOT/'data/references.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['id'] in ids:
                refs[row['id']] = row
    result, predictions = evaluate(config,ROOT/'artifacts',rows,ROOT,refs)
    per_family = {}
    for mode in ('vcm','pipeline'):
        totals = {}
        for row,prediction in zip(rows,predictions):
            reference = refs.get(row['id'],{}).get('reference')
            finite = reference and reference.get('status')=='accepted' and not (set(reference['command']['arguments']) & {'title','artist','selection_text'})
            bucket = totals.setdefault(row['command'], {'recordings':0,'valid_commands':0,'correct_commands':0,'accepted_outputs':0,'false_accepts_on_unknown':0})
            bucket['recordings'] += 1; bucket['valid_commands'] += bool(finite)
            outputs = prediction[mode]
            bucket['correct_commands'] += bool(finite and len(outputs)==1 and outputs[0].get('status')=='accepted' and outputs[0].get('command')==reference['command'] and outputs[0].get('vcm_command')==row['command'])
            for output in outputs:
                if output.get('status')=='accepted' and 'command' in output:
                    predicted = totals.setdefault(output.get('vcm_command','UNKNOWN'), {'recordings':0,'valid_commands':0,'correct_commands':0,'accepted_outputs':0,'false_accepts_on_unknown':0})
                    predicted['accepted_outputs'] += 1
                    bucket['false_accepts_on_unknown'] += row['command']=='UNKNOWN'
        per_family[mode] = totals
    result.update(experiment='wake-pipeline-v1',per_family_commands=per_family,
        scope='Historical command-only recordings, no wake phrase. Tests Silero endpointing and unchanged VCM; not live wake accuracy.',
        vcm_sha256=digest(ROOT/'artifacts/vcm.fp32.onnx'),config_sha256=digest(ROOT/'pipeline.toml'),
        fresh_holdout=False,live_microphone_tested=False,physical_pi_tested=False)
    save_json(ROOT/'experiments/wake-pipeline-v1/replay-results.json',result)
    path=ROOT/'.cache/wake-pipeline-v1/replay-predictions.jsonl'
    with path.open('w',encoding='utf-8') as f:
        for row in predictions:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:{x:result[k][x] for x in ['commands','exact_finite_command_recall','accepted_command_precision']} for k in ['vcm','pipeline']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--replay',action='store_true')
    mode.add_argument('--parity',action='store_true')
    args=parser.parse_args()
    if args.replay:
        replay()
    elif args.parity:
        parity()
    else:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WakeBehavior))
        save_json(ROOT/'experiments/wake-pipeline-v1/software-checks.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'scope':'State-machine fixtures, resampling continuity and real pretrained silence execution; not spoken wake accuracy.'})
        raise SystemExit(not result.wasSuccessful())
