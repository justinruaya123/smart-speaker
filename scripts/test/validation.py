"""Current component checks and held-out evaluation; no historical test tree."""
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .audio import load_audio
from .pipeline import EnergyVAD, Pipeline


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def check_vad_recovery(audio):
    """Preserved regression fixtures for the historical energy VAD."""
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    # Keep the regression signal/timing contract independent of user tuning.
    settings = {**audio, "block_ms": 40, "preroll_ms": 200, "silence_ms": 400,
                "max_utterance_seconds": 8., "calibration_ms": 1000,
                "energy_floor": .008, "noise_multiplier": 3.}
    for rate in (16000, 48000):
        block = round(rate * .04)
        wave = (np.sqrt(2) * np.sin(2 * np.pi * 100 * np.arange(block) / rate)).astype(np.float32)
        quiet, background, speech = .003 * wave, .012 * wave, .06 * wave
        def feed(vad, chunk, count):
            results = []
            for _ in range(count):
                result = vad.push(chunk)
                if result is not None:
                    results.append(result)
            return results

        live = EnergyVAD(settings, rate, calibrate=True)
        require(not feed(live, background, 25) and not live.calibrating and not live.active,
                "Ambient calibration emitted speech")
        require(abs(live.noise - .012) < 1e-6, "Calibration missed elevated background")
        require(not feed(live, background, 500) and not live.active,
                "Steady background repeated the timeout loop")
        for _ in range(2):
            require(not feed(live, speech, 5) and live.active, "Speech above noise did not start")
            require(not feed(live, background, 9), "Endpoint fired before 400 ms")
            result = live.push(background)
            require(result is not None and not result[1] and not live.active,
                    "Exactly 400 ms of silence failed to end speech")

        # Calibration is robust to a brief transient; a later noise-floor step
        # is allowed one bounded timeout, then must recover and hear new speech.
        stepped = EnergyVAD(settings, rate, calibrate=True)
        stepped.push(speech)
        feed(stepped, quiet, 24)
        require(abs(stepped.noise - .003) < 1e-6, "Startup transient poisoned calibration")
        raised = .03 * wave
        results = feed(stepped, raised, 500)
        require(len(results) == 1 and results[0][1] and len(results[0][0]) == 8 * rate,
                "Noise-floor change did not recover after one exact eight-second timeout")
        require(not stepped.calibrating and not stepped.active and abs(stepped.noise - .03) < 1e-6,
                "Timeout did not refresh the noise estimate")
        feed(stepped, .15 * wave, 5)
        result = feed(stepped, raised, 10)
        require(len(result) == 1 and not result[0][1], "No speech recognition after timeout recovery")

        offset = np.full(block, .2, dtype=np.float32)
        dc = EnergyVAD(settings, rate, calibrate=True)
        require(not feed(dc, offset, 525) and not dc.active, "DC offset triggered VAD")
        require(dc.raw_rms > .19 and dc.rms < 1e-6, "Raw/DC-free diagnostics disagree")
        original = offset + speech
        feed(dc, original, 5)
        result = feed(dc, offset, 10)
        require(len(result) == 1 and not result[0][1], "Speech with DC offset did not endpoint")
        require(np.array_equal(result[0][0][4 * block:5 * block], original),
                "VAD DC filtering changed the audio sent to the model")

        # File replay must retain speech beginning at the first sample.
        from einops import repeat
        recording = repeat(speech, "t -> (r t)", r=25)
        offline = EnergyVAD(settings, rate)
        result = offline.segment(recording)
        require(len(result) == 1 and np.array_equal(result[0][0][:len(recording)], recording),
                "Calibration consumed the start of an audio file")

    return {"noisy_startup": True, "background_only_seconds": 20,
            "noise_step_recovery": True, "timeouts_before_recovery": 1,
            "speech_after_recovery": True, "dc_offset_ignored": True,
            "model_pcm_preserved": True, "exact_silence_ms": 400,
            "exact_timeout_seconds": 8, "rates_tested": [16000, 48000],
            "replay_onset_preserved": True}


def run_checks(config, bundle):
    """Real graph smoke checks plus explicitly labelled routing fixtures."""
    from .config import resolve_switches
    checks = {}
    def require(condition, message):
        if not condition:
            raise AssertionError(message)
    require(resolve_switches({"vad": True, "stt": False, "pipeline": "vad and not stt"})["pipeline"],
            "Boolean expression resolution")
    for bad in ({"a": "__import__('os')"}, {"a": "b", "b": "a"}, {"a": "missing"}, {"a": 1}):
        try:
            resolve_switches(bad)
        except (ValueError, SyntaxError):
            pass
        else:
            raise AssertionError("Invalid boolean expression accepted")
    checks["config"] = {"status": "passed", "safe_boolean_expressions": True}
    pcm = (.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    if config["tests"]["vad"]:
        recovery = check_vad_recovery(config["audio"])
        vad = EnergyVAD(config["audio"])
        require(not vad.segment(np.zeros(16000, dtype=np.float32)), "Silence triggered VAD")
        sequence = np.pad(pcm, (8000, 8000))
        from einops import pack
        twice, _ = pack([sequence, sequence], "*")
        found = vad.segment(twice)
        require(len(found) == 2 and not any(timeout for _, timeout in found), "Repeated VAD endpointing")
        from einops import repeat
        found = vad.segment(repeat(pcm, "t -> (r t)", r=10))
        require(found and found[0][1], "Overlong speech did not time out")
        from .capture import resolve_microphone
        devices = [{"max_input_channels": 0}, {"max_input_channels": 2}]
        require(resolve_microphone(devices, [], 1)[0] == 1, "Numeric microphone selection")
        try:
            resolve_microphone(devices, [], 0)
        except ValueError:
            pass
        else:
            raise AssertionError("Output-only device accepted")
        # Simulated PortAudio callbacks exercise the actual live-capture loop at
        # 48 kHz, including channel selection, fallback, overflow and two endpoints.
        import contextlib
        import io
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from .capture import listen
        rate = 48000
        tone = (.1 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)).astype(np.float32)
        lead = round(rate * (config["audio"].get("calibration_ms", 1000) / 1000 + .2))
        native, _ = pack([np.pad(tone, (lead, rate//2)), np.pad(tone, (rate//2, rate//2))], "*")
        native += (.05 + np.sqrt(2) * .012 * np.sin(2 * np.pi * 100 * np.arange(len(native)) / rate)).astype(np.float32)
        overflow_first = True
        block = round(rate * config["audio"]["block_ms"] / 1000)
        class FakePortAudioError(Exception):
            pass
        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs, self.offset = kwargs, 0
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            @property
            def active(self):
                if self.offset >= len(native):
                    return False
                audio = np.zeros((min(block, len(native)-self.offset), 2), dtype=np.float32)
                audio[:, 1] = native[self.offset:self.offset+block]
                self.kwargs["callback"](audio, len(audio), None, overflow_first and self.offset == 0)
                self.offset += len(audio)
                return True
        def input_settings(**kwargs):
            if kwargs["samplerate"] == 16000:
                raise FakePortAudioError("native rate only")
        fake = SimpleNamespace(InputStream=FakeStream, PortAudioError=FakePortAudioError,
            check_input_settings=input_settings, default=SimpleNamespace(device=(1, 1)),
            query_hostapis=lambda: [{"name": "Simulated"}],
            query_devices=lambda: [{"max_input_channels": 0}, {"max_input_channels": 2,
                "name": "Simulated input", "hostapi": 0, "default_samplerate": rate}])
        lengths = []
        cfg = copy.deepcopy(config); cfg["runtime"]["vad"] = True
        cfg['wake']['enabled'] = False
        cfg['audio']['vad_backend'] = 'energy'
        stub = SimpleNamespace(config=cfg, utterance=lambda audio, timeout:
            (lengths.append(len(audio)) or {"status": "fixture", "timeout": timeout}))
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"sounddevice": fake}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            listen(stub, SimpleNamespace(mic=1, channel=1, capture_rate=None, vad_debug=True))
        output = [json.loads(line) for line in stdout.getvalue().splitlines()]
        require(len(lengths) == 2 and all(16000 <= n < 32000 for n in lengths), "Native microphone resampling/endpoints")
        require(any(item.get("reason") == "audio_overflow" for item in output), "Overflow must be rejected")
        require(json.loads(stderr.getvalue())["capture_rate"] == 48000, "Native-rate fallback")
        require(any(item.get("status") == "ready" for item in output), "Calibration ready event missing")
        levels = [item for item in output if item.get("status") == "levels"]
        require(levels and all({"rms", "ac_rms", "noise_rms", "threshold"} <= item.keys() for item in levels),
                "Microphone level diagnostics missing")
        require(all(not item.get("timeout") for item in output), "Noisy speech capture timed out")
        clean_endpoints = len(lengths)
        # A new ambient level after calibration must not alternate speech
        # starts and timeouts forever in the real capture orchestration.
        initial = (.003 * np.sqrt(2) * np.sin(2 * np.pi * 100 * np.arange(lead) / rate)).astype(np.float32)
        changed = (.03 * np.sqrt(2) * np.sin(2 * np.pi * 100 * np.arange(rate * 20) / rate)).astype(np.float32)
        loud = (.15 * np.sqrt(2) * np.sin(2 * np.pi * 100 * np.arange(rate // 5) / rate)).astype(np.float32)
        native, _ = pack([initial, changed, loud, changed[:rate]], "*")
        overflow_first = False
        lengths.clear()
        def recovered_utterance(audio, timeout):
            if timeout:
                return {"status": "rejected", "reason": "utterance_timeout"}
            lengths.append(len(audio))
            return {"status": "fixture"}
        stub.utterance = recovered_utterance
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {"sounddevice": fake}), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            listen(stub, SimpleNamespace(mic=1, channel=1, capture_rate=rate, vad_debug=True))
        recovered = [json.loads(line) for line in stdout.getvalue().splitlines()]
        timeouts = [item for item in recovered if item.get("status") == "rejected" and item.get("reason") == "utterance_timeout"]
        require(len(timeouts) == 1 and "vad" in timeouts[0], "Capture timeout recovery/diagnostics")
        require(sum(item.get("status") == "ready" for item in recovered) == 2 and len(lengths) == 1,
                "Capture did not resume recognition after recalibration")
        checks["vad"] = {"status": "passed", "backend": "energy_legacy_regression",
                         "silence": True, "repeated_utterances": 2,
                         "timeout": True, "numeric_microphone": True, "simulated_native_capture_rate": 48000,
                         "simulated_capture_endpoints": clean_endpoints, "overflow_rejected": True, "live_microphone": False,
                         "noise_recovery": recovery, "capture_recalibration": True, "level_diagnostics": True}
    if config["tests"]["vcm"]:
        from .inference import Recognizer
        model = Recognizer(bundle, threads=config["validation"]["threads"])
        scores = model.classify(pcm)
        require(np.isfinite(scores).all() and abs(float(scores.sum()) - 1) < 1e-6, "VCM probabilities")
        silence = model.predict(np.zeros(16000, dtype=np.float32), stream=False)
        require(silence["status"] == "rejected" and not silence.get("parameter_invoked"), "Silence must abstain")
        require(model.parameter_reader is None and model.stream is None, "VCM-only path loaded another model")
        checks["vcm"] = {"status": "passed", "labels": model.labels, "silence": silence["status"],
                         "parameter_model_loaded": False, "providers": model.vcm.get_providers()}
    if config['tests'].get('stt'):
        from .stt_runtime import AcousticSTT
        stt=AcousticSTT(bundle,threads=config['validation']['threads'])
        recognized=stt.recognize(pcm)
        require(isinstance(recognized['text'],str) and np.isfinite(recognized['confidence']), 'STT decode')
        checks['stt']={'status':'passed','real_graph_executed':True,'vcm_required':False,
                       'providers':stt.session.get_providers(),'vocabulary_size':len(stt.vocabulary)}
    if config['parameters']['enabled']:
        checks['brightness_contract']=check_brightness_contract(config,bundle,pcm)
    if config["tests"]["pipeline"]:
        cfg = copy.deepcopy(config)
        cfg["runtime"] = {"vad": True, "vcm": True}
        pipeline = Pipeline(bundle, cfg)
        require(pipeline.process(np.zeros(16000, dtype=np.float32))[0]["reason"] == "no_speech", "VAD bypass")
        results = pipeline.process(pcm)
        require(results and all("status" in r for r in results), "Full pipeline result")
        # Inject probabilities into an independent in-memory copy to check handoff.
        # Production artifacts and their disabled custom-media policy remain untouched.
        model = pipeline.recognizer
        original = copy.deepcopy(model.metadata)
        vocab = model.metadata["acoustic_slots"]["vocabulary"]
        def onehot(values, value):
            p = np.zeros(len(values)); p[values.index(value)] = 1.; return p
        slot = {name: onehot(values, values[0]) for name, values in vocab.items()}
        slot["guard"] = onehot(vocab["guard"], "READY")
        slot["media_mode"] = onehot(vocab["media_mode"], "CUSTOM")
        slot["media"] = onehot(vocab["media"], "OTHER")
        family = ["MEDIA_PLAY"]
        model.classify_with_slots = lambda audio: (onehot(model.labels, family[0]), slot, None)
        calls = []
        if model.parameter_reader is not None:
            model.parameter_reader.resolve_brightness = lambda audio: calls.append(len(audio))
        blocked = model.predict(pcm, stream=False)
        require(not blocked["parameter_invoked"] and not calls, "Custom media reached a brightness reader")
        policy = model.metadata["acoustic_slots"]["variants"]["fp32"]
        policy.update(custom_media_threshold=.9, custom_guard_threshold=.9)
        unresolved = model.predict(pcm, stream=False)
        require(unresolved["status"] != "accepted" and not calls, "Retired custom recognizer path reactivated")
        family[0] = "POWER_ON"
        finite = model.predict(pcm, stream=False)
        require(not finite["parameter_invoked"] and not calls, "Finite uncertainty invoked a parameter reader")
        family[0] = "UNKNOWN"
        unknown = model.predict(pcm, stream=False)
        require(unknown["status"] == "rejected" and not calls, "Rejected intent invoked a parameter reader")
        model.metadata = original
        checks["pipeline"] = {"status": "passed", "real_vad_vcm_replay": True,
            "routing_fixtures": ["custom media has no unapproved reader", "retired custom path cannot reactivate",
                                 "finite uncertainty abstains", "UNKNOWN never invokes parameter reader"],
            "production_thresholds_modified": False}
    from .config import select_backend
    for attempt in (lambda: select_backend(config,'asr'),):
        try: attempt()
        except ValueError: pass
        else: raise AssertionError('Retired backend was accepted')
    require('asr' not in json.loads((Path(bundle)/'bundle.json').read_text()).get('runtime_components',[]),
            'Retired model remains in runtime components')
    checks['retired_model']={'status':'passed','legacy_backend_rejected':True,'no_automatic_fallback':True}
    if config['audio'].get('vad_backend') == 'silero':
        from .wake_models import HeyJarvis, SileroVAD
        wake = HeyJarvis(bundle)
        neural = SileroVAD(bundle, config['audio'])
        quiet = np.zeros(640, dtype=np.float32)
        scores = [wake.push(quiet) for _ in range(75)]
        require(max(x for x in scores if x is not None) < config['wake']['threshold'], 'Silent wake activation')
        require(not neural.segment(np.zeros(16000, dtype=np.float32)), 'Neural VAD accepted silence')
        checks['pretrained_activation'] = {'status':'passed', 'real_onnx_graphs':4,
            'silence_seconds':3, 'wake_phrase':'hey jarvis', 'live_wake_accuracy_measured':False}
    return checks


def check_brightness_contract(config,bundle,pcm):
    """Parser coverage and intent ownership, distinct from acoustic accuracy."""
    from .parameter_backend import parse_brightness
    from .levels import spoken_number
    from .inference import Recognizer
    backend=config['parameters']['backend']
    model=Recognizer(bundle,parameter_backend=backend,threads=config['validation']['threads'])
    for value in range(101):
        for expression in (str(value),str(value)+'%',spoken_number(value),spoken_number(value)+' percent'):
            result=parse_brightness('brightness '+expression,model.catalog)
            assert result.get('status')=='accepted' and result['command']['arguments']['percent']==value,expression
    for expression in ('brightness -1','brightness 101','brightness 2.5','brightness fifty sixty',
                       'volume fifty','brightness','brightness fifty volume twenty','brightness fifty <unknown>'):
        assert parse_brightness(expression,model.catalog).get('status')!='accepted',expression
    scores=np.zeros(len(model.labels));scores[model.labels.index('UNKNOWN')]=1
    model.metadata['acoustic_slots']['enabled']=False
    model.classify=lambda audio:scores
    assert model.predict(pcm,stream=False)['status']=='rejected'
    assert model.parameter_reader.engine is None
    scores[:]=0;scores[model.labels.index('BRIGHTNESS')]=1
    model.parameter_reader.recognize=lambda audio:{'text':'brightness fifty percent','confidence':1.}
    model.parameter_reader.metadata['brightness_parameters'][backend]['threshold']=.5
    result=model.predict(pcm,stream=False)
    assert result['vcm_command']=='BRIGHTNESS' and result['command']['arguments']['percent']==50
    assert result[backend+'_invoked'] and result['parameter_source']==backend
    model.parameter_reader.recognize=lambda audio:{'text':'volume fifty','confidence':1.}
    rejected=model.predict(pcm,stream=False)
    assert rejected['vcm_command']=='BRIGHTNESS' and rejected['status']!='accepted'
    model.parameter_reader.recognize=lambda audio:{'text':'brightness fifty','confidence':.1}
    assert model.predict(pcm,stream=False)['status']=='needs_clarification'
    assert not hasattr(model, "asr")
    return {'status':'passed','backend':backend,'parser_forms':404,'malformed_forms':8,
            'fixtures':'UNKNOWN stays model-free; VCM owns intent; conflicts and low confidence abstain; no alternate fallback',
            'acoustic_accuracy_test':False}


def audit_data(root):
    root = Path(root)
    descriptor = json.loads((root / "data/dataset.json").read_text(encoding="utf-8"))
    index = json.loads((root / "data/audio-index.json").read_text(encoding="utf-8"))
    splits = {}
    for task in ("vcm", "asr"):
        path = root / f"data/{task}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["current_manifests"][task]:
            raise ValueError(f"Changed {task} manifest; update the reviewed dataset descriptor first")
        rows = read_rows(path)
        splits[task] = dict(Counter(r["split"] for r in rows))
        for field in ("id", "audio", "speaker_id", "paired_group"):
            groups = defaultdict(set)
            for row in rows:
                if row.get(field):
                    groups[row[field]].add(row["split"])
            if any(len(s) > 1 for s in groups.values()):
                raise ValueError(f"{task}: {field} leaks across dataset splits")
        if any(row["audio"] not in index for row in rows):
            raise ValueError("Manifest audio missing from the integrity index")
    for relative, expected in index.items():
        path = (root / relative).resolve()
        if not path.is_relative_to((root / "data/audio").resolve()) or not path.is_file():
            raise ValueError(f"Missing or invalid current audio path: {relative}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Changed audio: {relative}")
    if hashlib.sha256((root / "data/references.jsonl").read_bytes()).hexdigest() != descriptor["references_sha256"]:
        raise ValueError("Reference labels changed")
    return {"audio_files_verified": len(index), "split_counts": splits,
            "recording_speaker_pair_split_leaks": 0, "reference_quality": "programmatic silver"}


def evaluate(config, bundle, rows, root, references):
    from .metrics import metrics
    from .text import normalize_text
    started = time.perf_counter()
    selected = config["tests"]
    vcm = pipeline = None
    if selected["vcm"]:
        from .inference import Recognizer
        vcm = Recognizer(bundle, threads=config["validation"]["threads"])
    if selected["pipeline"]:
        cfg = copy.deepcopy(config); cfg["runtime"].update(vad=True,vcm=True)
        pipeline = Pipeline(bundle, cfg)
    vad = EnergyVAD(config["audio"]) if selected["vad"] else None
    records, endpoint_records = [], []
    speech_count = segments_count = timeouts = 0
    word_errors = words = char_errors = chars = exact_text = 0
    command_counts = {name: Counter() for name in ("vcm", "pipeline")}
    predictions = []
    for i, row in enumerate(rows, 1):
        pcm, rate = load_audio(Path(root) / row["audio"])
        entry = {"id": row["id"], "gold": row["command"]}
        if vad:
            segments = vad.segment(pcm)
            speech_count += bool(segments); segments_count += len(segments)
            timeouts += sum(int(timeout) for _, timeout in segments)
        reference = references.get(row["id"], {}).get("reference")
        finite = reference and reference.get("status") == "accepted" and not (
            set(reference["command"]["arguments"]) & {"title", "artist", "selection_text"})
        for name, engine, output in (("vcm", vcm, records), ("pipeline", pipeline, endpoint_records)):
            if engine is None:
                continue
            results = [engine.predict(pcm, stream=False)] if name == "vcm" else engine.process(pcm)
            # All segments count as emissions; multiple endpoints cannot earn a complete-command success.
            result = results[0] if len(results) == 1 else {"status": "rejected", "reason": "multiple_endpoints"}
            output.append({"gold": row["command"], "predicted": result.get("vcm_command", "UNKNOWN"),
                           "accepted": result.get("intent_accepted", False)})
            counts = command_counts[name]
            counts["finite_references"] += bool(finite)
            counts["emitted"] += sum(r.get("status") == "accepted" and "command" in r for r in results)
            counts["correct_complete"] += bool(finite and result.get("status") == "accepted"
                and result.get("vcm_command") == row["command"] and result.get("command") == reference["command"])
            counts["parameter_calls"] += sum(bool(r.get("parameter_invoked")) for r in results)
            entry[name] = results
        predictions.append(entry)
        if i % 200 == 0:
            print(f"VALIDATE {i}/{len(rows)} elapsed={time.perf_counter()-started:.0f}s", flush=True)
    report = {"examples": len(rows), "elapsed_seconds": time.perf_counter()-started,
              "quality_gate_passed": False, "target_hardware_measured": False}
    for name, engine, output in (("vcm", vcm, records), ("pipeline", pipeline, endpoint_records)):
        if engine is not None:
            labels = (engine if name == "vcm" else engine.recognizer).labels
            counts = command_counts[name]
            report[name] = {"intent": metrics(output, labels), "commands": dict(counts),
                "exact_finite_command_recall": counts["correct_complete"] / max(1, counts["finite_references"]),
                "accepted_command_precision": counts["correct_complete"] / max(1, counts["emitted"])}
    if vad:
        report["vad"] = {"recordings_with_speech": speech_count, "endpoints": segments_count, "timeouts": timeouts,
                         "boundary_accuracy_assessed": False}
    command_paths = [report[name] for name in ("vcm", "pipeline") if name in report]
    report["quality_gate_passed"] = bool(command_paths) and all(
        part["intent"]["gates"]["inherited_pass"]
        and not part["intent"]["gates"]["additional_low_per_intent_precision"]
        and part["accepted_command_precision"] >= .95
        and part["exact_finite_command_recall"] >= .60 for part in command_paths)
    report["quality_gate_criteria"] = {
        "per_family_raw_recall": .80, "per_family_accepted_recall": .60,
        "global_and_per_family_accepted_precision": .95, "maximum_unknown_false_accept_rate": .01,
        "finite_command_recall": .60, "finite_command_precision": .95,
        "scope": "selected command paths only; VAD-only does not establish command quality"}
    report["quality_gate_note"] = "Software quality gates are separate from production readiness. Target-device, live microphone, reviewed-reference and custom-parameter acceptance remain unestablished."
    return report, predictions


def validate_dataset(config, root):
    root = Path(root)
    audit = audit_data(root)
    split, maximum = config["validation"]["split"], config["validation"]["max_examples"]
    rows = [r for r in read_rows(root / "data/vcm.jsonl") if r["split"] == split]
    if maximum:
        rows = rows[:maximum]
    refs = {r["id"]: r for r in read_rows(root / "data/references.jsonl")}
    report, predictions = evaluate(config, root / "artifacts", rows, root, refs)
    report.update(split=split, limited=bool(maximum), data_audit=audit)
    out = root / ".cache/validation/predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
    report["prediction_log"] = ".cache/validation/predictions.jsonl"
    return report


def validate_samples(config, bundle, sample_root):
    sample_root = Path(sample_root)
    rows = read_rows(sample_root / "samples.jsonl")
    refs = {r["id"]: r["annotation"] for r in rows}
    report, _ = evaluate(config, bundle, rows, sample_root, refs)
    report["scope"] = "packaged held-out smoke samples; not the full validation set"
    return report
